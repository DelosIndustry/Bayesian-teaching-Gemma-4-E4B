#!/usr/bin/env python3
"""Quick sanity check for the saved SFT model.

Loads the final checkpoint and runs a few single-turn generations using the
exact prompt format seen in training data. Verifies that:
  1. Model + tokenizer load without errors from the saved directory
  2. Chat template applies correctly
  3. Model produces a well-formed "The best option is Flight N." response

Usage:
    python scripts/sanity_check.py
    python scripts/sanity_check.py --model-path <path>
"""

import argparse
import logging
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = (
    "/home/sdh/Bayesian-teaching-Gemma-4-E4B/models/e4b-full-bayesian"
)

# Three prompts matching the training distribution (4-feature flight selection).
# Each is a "round 1" single-turn query. The model should output a terse
# "The best option is Flight N." line.
TEST_PROMPTS = [
    """Help me select the best flights for my trips. I have specific preferences for what I like and dislike in a flight, and these preferences remain the same. You need to figure out my preferences and select the best flights for me. Use your best judgment if you are unsure. Do not say you need more information.
Which flight is the best option?

Flight 1:
departure time: 10:48 AM, duration: 14 hr 9 min, number of stops: 0, price: $190
Flight 2:
departure time: 06:00 AM, duration: 18 hr 3 min, number of stops: 1, price: $820
Flight 3:
departure time: 08:24 PM, duration: 8 hr 18 min, number of stops: 0, price: $1000""",
    """Help me select the best flights for my trips. I have specific preferences for what I like and dislike in a flight, and these preferences remain the same. You need to figure out my preferences and select the best flights for me. Use your best judgment if you are unsure. Do not say you need more information.
Which flight is the best option?

Flight 1:
departure time: 07:30 AM, duration: 5 hr 45 min, number of stops: 0, price: $420
Flight 2:
departure time: 11:15 PM, duration: 9 hr 10 min, number of stops: 2, price: $180
Flight 3:
departure time: 03:00 PM, duration: 6 hr 20 min, number of stops: 1, price: $310""",
    """Help me select the best flights for my trips. I have specific preferences for what I like and dislike in a flight, and these preferences remain the same. You need to figure out my preferences and select the best flights for me. Use your best judgment if you are unsure. Do not say you need more information.
Which flight is the best option?

Flight 1:
departure time: 09:00 AM, duration: 12 hr 0 min, number of stops: 1, price: $650
Flight 2:
departure time: 02:30 PM, duration: 7 hr 15 min, number of stops: 0, price: $890
Flight 3:
departure time: 06:45 AM, duration: 11 hr 30 min, number of stops: 0, price: $540""",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity check trained SFT model")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    logger.info(f"Loading model from {args.model_path} (bf16) onto {args.device}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="eager",
    )
    model.eval()
    load_time = time.time() - t0
    logger.info(f"Model loaded in {load_time:.1f}s")

    mem_gib = torch.cuda.memory_allocated(args.device) / (1024**3)
    logger.info(f"GPU memory allocated: {mem_gib:.2f} GiB")

    parse_ok = 0
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        logger.info(f"--- Test {i}/{len(TEST_PROMPTS)} ---")
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        ).to(args.device)

        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen_time = time.time() - t0

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        n_new = len(new_tokens)
        tok_per_s = n_new / gen_time if gen_time > 0 else float("inf")

        logger.info(f"  Response: {response!r}")
        logger.info(
            f"  Generated {n_new} tokens in {gen_time:.2f}s "
            f"({tok_per_s:.1f} tok/s)"
        )

        # Simple regex-free parse: look for "Flight N" where N in 1..9
        ok = any(f"Flight {n}" in response for n in range(1, 10))
        if ok:
            parse_ok += 1
            logger.info("  ✓ Response contains a Flight selection")
        else:
            logger.warning("  ✗ Response does not contain a Flight selection")

    logger.info(
        f"\nSanity check summary: {parse_ok}/{len(TEST_PROMPTS)} responses "
        f"contain a parseable Flight selection."
    )

    if parse_ok == len(TEST_PROMPTS):
        logger.info("All checks passed. Model is ready for evaluation.")
        return 0
    logger.warning("Some responses failed to parse. Inspect responses above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
