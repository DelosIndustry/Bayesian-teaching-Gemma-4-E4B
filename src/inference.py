"""Unified inference interface for Gemma 4 E4B models.

Loads the model with AutoModelForCausalLM (bf16 by default) and exposes
generate()/generate_batch() that apply the chat template for us. Designed
for use from evaluation and quantization modules.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

VALID_QUANTIZATION_LABELS = {"bf16", "int8", "int4"}


class GemmaInference:
    """Thin wrapper over HF Transformers for chat-style inference."""

    def __init__(
        self,
        model_path: str,
        quantization: Optional[str] = "bf16",
        device_map: str = "cuda:0",
        max_new_tokens: int = 32,
        attn_implementation: str = "eager",
    ):
        """
        Args:
            model_path: local directory or HF hub id.
            quantization: 'bf16', 'int8', or 'int4'.
            device_map: 'cuda:N' for single GPU, 'auto' for multi-GPU dispatch.
            max_new_tokens: generation budget per call.
            attn_implementation: 'eager' works on every torch/transformers
                combination; 'sdpa' is faster if supported.
        """
        if quantization is None:
            quantization = "bf16"
        if quantization not in VALID_QUANTIZATION_LABELS:
            raise ValueError(f"unknown quantization label: {quantization}")

        self.model_path = model_path
        self.quantization = quantization
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens

        logger.info("Loading tokenizer from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only models use left-padding for batched generation.
        self.tokenizer.padding_side = "left"

        load_kwargs = {
            "device_map": device_map,
            "attn_implementation": attn_implementation,
        }
        if quantization == "int8":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == "int4":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "bf16":
            load_kwargs["dtype"] = torch.bfloat16

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        logger.info("Loading model from %s (%s)", model_path, quantization)
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self.model.eval()
        logger.info("Model loaded in %.1fs", time.time() - t0)

    def _device(self) -> torch.device:
        """Device to place input tensors on."""
        if isinstance(self.device_map, str) and self.device_map.startswith("cuda"):
            return torch.device(self.device_map)
        return next(self.model.parameters()).device

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate a response for a single conversation.

        Args:
            messages: list of {role, content} dicts.
            temperature: 0.0 means greedy decoding.
            max_new_tokens: override the instance default.

        Returns:
            Decoded assistant response (no special tokens).
        """
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        ).to(self._device())

        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": temperature})
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = out[0][prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_batch(
        self,
        batch_messages: list[list[dict]],
        temperature: float = 0.0,
        max_new_tokens: Optional[int] = None,
    ) -> list[str]:
        """Batched generation with left padding.

        All conversations are tokenized and padded together. This is notably
        faster than calling generate() in a Python loop for batches >= 4.
        """
        # Render each conversation to text first, then tokenize as a batch.
        # This lets the tokenizer apply left-padding uniformly.
        prompts = [
            self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in batch_messages
        ]
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=False
        ).to(self._device())

        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": temperature})
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)

        prompt_len = enc["input_ids"].shape[1]
        new_tokens = out[:, prompt_len:]
        return [
            self.tokenizer.decode(row, skip_special_tokens=True).strip()
            for row in new_tokens
        ]

    def get_memory_usage(self) -> dict:
        """Report current GPU memory in MiB."""
        if not torch.cuda.is_available():
            return {"allocated_mib": 0.0, "reserved_mib": 0.0, "peak_mib": 0.0}
        dev = self._device()
        if dev.type != "cuda":
            dev = torch.device("cuda:0")
        return {
            "allocated_mib": torch.cuda.memory_allocated(dev) / (1024**2),
            "reserved_mib": torch.cuda.memory_reserved(dev) / (1024**2),
            "peak_mib": torch.cuda.max_memory_allocated(dev) / (1024**2),
        }
