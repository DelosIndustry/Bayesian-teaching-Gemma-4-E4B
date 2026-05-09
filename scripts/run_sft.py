#!/usr/bin/env python3
"""E4B Full SFT training entry point.

Uses TRL's SFTTrainer with hyperparameters from an alignment-handbook style
YAML config. Loads local JSONL dataset and uses the tokenizer's chat template
to format the 'messages' column.

Usage:
    accelerate launch --config_file configs/accelerate_zero3.yaml \
        scripts/run_sft.py configs/e4b_bayesian_sft.yaml

    # Resume from checkpoint:
    accelerate launch --config_file configs/accelerate_zero3.yaml \
        scripts/run_sft.py configs/e4b_bayesian_sft.yaml --resume_from_checkpoint
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

# Add src/ to path for data_utils import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data_utils import load_jsonl_dataset, validate_messages

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="E4B Full SFT Training")
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML config file (alignment-handbook style)",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        nargs="?",
        const="true",
        default=None,
        help="Resume from checkpoint. Pass a path or 'true' for latest.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def main():
    args = parse_args()
    config = load_config(args.config)

    # Set seed for reproducibility
    seed = config.get("seed", 42)
    set_seed(seed)

    # Model and tokenizer
    model_name = config["model_name_or_path"]
    torch_dtype = getattr(torch, config.get("torch_dtype", "bfloat16"))

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model: {model_name} (dtype={torch_dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    )

    # Load dataset from local JSONL path specified in config
    dataset_path = config["dataset_path"]
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = load_jsonl_dataset(dataset_path)

    # Validate messages format
    stats = validate_messages(dataset)
    logger.info(
        f"Dataset: {stats['total']} records, "
        f"{stats['valid']} valid, {stats['invalid']} invalid"
    )
    if stats["invalid"] > 0:
        logger.warning(f"{stats['invalid']} records have invalid message format")

    logger.info(f"Dataset ready: {len(dataset)} examples with 'messages' column")

    # Build SFTConfig from YAML config
    # Only pass parameters compatible with TRL >= 1.0 API
    sft_config_kwargs = {
        "output_dir": config["output_dir"],
        "max_length": config.get("max_seq_length", 2048),
        # Training hyperparameters
        "per_device_train_batch_size": config.get("per_device_train_batch_size", 4),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 5),
        "learning_rate": config.get("learning_rate", 2e-6),
        "num_train_epochs": config.get("num_train_epochs", 1),
        "warmup_ratio": config.get("warmup_ratio", 0.1),
        "lr_scheduler_type": config.get("lr_scheduler_type", "cosine"),
        # Precision
        "bf16": config.get("bf16", True),
        # Logging
        "logging_steps": config.get("logging_steps", 10),
        "logging_strategy": config.get("logging_strategy", "steps"),
        # Saving
        "save_strategy": config.get("save_strategy", "steps"),
        "save_steps": config.get("save_steps", 200),
        "save_total_limit": config.get("save_total_limit", 3),
        # Other
        "seed": seed,
        "report_to": config.get("report_to", "none"),
        "dataloader_num_workers": config.get("preprocessing_num_workers", 4),
        # Gradient checkpointing
        "gradient_checkpointing": config.get("gradient_checkpointing", True),
        "gradient_checkpointing_kwargs": config.get(
            "gradient_checkpointing_kwargs", {"use_reentrant": False}
        ),
    }

    # Filter out parameters not supported by this version of SFTConfig
    import inspect
    supported_params = set(inspect.signature(SFTConfig.__init__).parameters.keys())
    filtered_kwargs = {k: v for k, v in sft_config_kwargs.items() if k in supported_params}
    removed = set(sft_config_kwargs.keys()) - set(filtered_kwargs.keys())
    if removed:
        logger.warning(f"SFTConfig: dropped unsupported params: {removed}")

    sft_config = SFTConfig(**filtered_kwargs)

    # Create trainer - SFTTrainer will apply chat template to 'messages' column
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Determine checkpoint for resuming
    resume_checkpoint = None
    if args.resume_from_checkpoint is not None:
        if args.resume_from_checkpoint == "true":
            # Find latest checkpoint in output_dir
            output_dir = Path(config["output_dir"])
            checkpoints = sorted(output_dir.glob("checkpoint-*"))
            if checkpoints:
                resume_checkpoint = str(checkpoints[-1])
                logger.info(f"Resuming from latest checkpoint: {resume_checkpoint}")
            else:
                logger.warning("No checkpoints found, starting from scratch")
        else:
            resume_checkpoint = args.resume_from_checkpoint
            logger.info(f"Resuming from specified checkpoint: {resume_checkpoint}")

    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save final model
    logger.info(f"Saving final model to {config['output_dir']}")
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
