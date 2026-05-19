#!/usr/bin/env python3
"""Run interactive evaluation on a trained Gemma 4 E4B SFT model.

Example:
    python scripts/evaluate.py \
        --model-path models/e4b-full-bayesian \
        --teaching bayesian \
        --domain flight \
        --output-dir results/

    # Quick smoke test (first 10 users only):
    python scripts/evaluate.py \
        --model-path models/e4b-full-bayesian \
        --teaching bayesian --domain flight \
        --max-users 10
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure repo root is importable as a package root (so `src.xxx` works).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from transformers import set_seed  # noqa: E402

from src.inference import GemmaInference  # noqa: E402
from src.evaluation import InteractiveEvaluator, save_results  # noqa: E402

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("evaluate")


DOMAIN_FILES = {
    # name -> (interaction_jsonl, heldout_json)
    "flight": ("flight.jsonl", "flight.json"),
    "flight_2features": ("flight_2features.jsonl", "flight_2features.json"),
    "flight_3features": ("flight_3features.jsonl", "flight_3features.json"),
    "flight_5features": ("flight_5features.jsonl", "flight_5features.json"),
    "flight_6features": ("flight_6features.jsonl", "flight_6features.json"),
    "flight_7features": ("flight_7features.jsonl", "flight_7features.json"),
    "hotel": ("hotel.jsonl", "hotel.json"),
}


def parse_args():
    p = argparse.ArgumentParser(description="Interactive eval for Gemma 4 E4B SFT")
    p.add_argument("--model-path", required=True)
    p.add_argument(
        "--teaching",
        default="unknown",
        help="Label for the teaching method (base/bayesian/oracle), used in filename and metadata.",
    )
    p.add_argument(
        "--domain",
        default="flight",
        choices=sorted(DOMAIN_FILES.keys()),
    )
    p.add_argument("--data-root", default=str(REPO_ROOT / "data" / "eval"))
    p.add_argument("--output-dir", default=str(REPO_ROOT / "results"))
    p.add_argument(
        "--quantization",
        default="bf16",
        choices=["bf16", "int8", "int4"],
    )
    p.add_argument("--device-map", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--num-rounds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--evaluation-mode",
        default="paper",
        choices=["paper", "interaction"],
        help="paper evaluates 100 heldout option sets after each feedback round; interaction scores only the 5 feedback rounds.",
    )
    p.add_argument("--heldout-batch-size", type=int, default=8)
    p.add_argument(
        "--output",
        default=None,
        help="Optional exact output JSON path. Defaults to the conventional results filename.",
    )
    p.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Limit number of users (useful for smoke tests).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    inter_name, heldout_name = DOMAIN_FILES[args.domain]
    interaction_path = str(Path(args.data_root) / "interaction" / inter_name)
    heldout_path = str(Path(args.data_root) / "heldout" / heldout_name)

    if not Path(interaction_path).exists():
        raise FileNotFoundError(interaction_path)
    if not Path(heldout_path).exists():
        raise FileNotFoundError(heldout_path)

    inference = GemmaInference(
        model_path=args.model_path,
        quantization=args.quantization,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
    )
    mem = inference.get_memory_usage()
    logger.info(
        "Model memory: allocated=%.0f MiB, reserved=%.0f MiB",
        mem["allocated_mib"], mem["reserved_mib"],
    )

    evaluator = InteractiveEvaluator(
        inference=inference,
        interaction_path=interaction_path,
        heldout_path=heldout_path,
        num_rounds=args.num_rounds,
        teaching=args.teaching,
        domain=args.domain,
        evaluation_mode=args.evaluation_mode,
        heldout_batch_size=args.heldout_batch_size,
    )

    t0 = time.time()
    results = evaluator.evaluate_all(max_users=args.max_users)
    elapsed = time.time() - t0

    quant_label = args.quantization
    # Build a conventional filename: {model}_{teaching}_{quant}_{domain}.json
    model_slug = Path(args.model_path).name
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = (
            Path(args.output_dir)
            / f"{model_slug}_{args.teaching}_{quant_label}_{args.domain}.json"
        )

    meta = {
        "model_path": args.model_path,
        "model_slug": model_slug,
        "teaching": args.teaching,
        "quantization": quant_label,
        "domain": args.domain,
        "num_rounds": args.num_rounds,
        "max_users": args.max_users,
        "evaluation_mode": args.evaluation_mode,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "memory_mib": mem,
        "elapsed_seconds_total": elapsed,
    }
    save_results(results, meta, str(out_path))

    s = results["summary"]
    logger.info("Done in %.1fs (%.1f min)", elapsed, elapsed / 60)
    logger.info(
        "Round accuracies: %s",
        ", ".join(f"R{i+1}={a:.3f}" for i, a in enumerate(s["round_accuracies"])),
    )
    logger.info("Learning effect (R_last - R1): %.3f", s["learning_effect"])
    logger.info("Parse failure rate: %.2f%%", s["parse_failure_rate"] * 100)


if __name__ == "__main__":
    main()
