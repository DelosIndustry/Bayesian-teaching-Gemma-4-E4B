"""Data loading utilities for E4B Full SFT training.

Loads local JSONL training data as a HuggingFace Dataset with a 'messages'
column. SFTTrainer handles chat template application automatically when
the dataset has a 'messages' column.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset

logger = logging.getLogger(__name__)

VALID_ROLES = {"user", "assistant"}


def load_jsonl_dataset(path: str) -> Dataset:
    """Load a local JSONL file as a HuggingFace Dataset.

    Each line should be a JSON object with at least a 'messages' field
    containing a list of {role, content} dicts.

    Args:
        path: Path to the JSONL file.

    Returns:
        HuggingFace Dataset with 'messages' column (and 'idx' if present).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    logger.info(f"Loading dataset from {path}")

    records = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "messages" not in record:
                    logger.warning(f"Line {line_num}: missing 'messages' field, skipping")
                    continue
                if not record["messages"]:
                    logger.warning(f"Line {line_num}: empty messages list, skipping")
                    continue
                records.append(record)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON parse error: {e}, skipping")

    logger.info(f"Loaded {len(records)} records from {path.name}")

    # Build dataset with messages column
    dataset = Dataset.from_list(records)
    return dataset


def validate_messages(dataset: Dataset) -> dict:
    """Validate that messages have correct roles (user/assistant).

    Checks each record's messages for valid alternating user/assistant roles.

    Args:
        dataset: HuggingFace Dataset with 'messages' column.

    Returns:
        Dict with validation stats: total, valid, invalid, role_issues.
    """
    total = len(dataset)
    valid = 0
    invalid = 0
    role_issues = []

    for i, example in enumerate(dataset):
        messages = example["messages"]
        is_valid = True

        for j, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role not in VALID_ROLES:
                is_valid = False
                role_issues.append(
                    f"Record {i}, message {j}: invalid role '{role}'"
                )
                break

            if not content:
                is_valid = False
                role_issues.append(
                    f"Record {i}, message {j}: empty content"
                )
                break

        if is_valid:
            valid += 1
        else:
            invalid += 1

    if role_issues and len(role_issues) <= 10:
        for issue in role_issues:
            logger.warning(issue)
    elif role_issues:
        logger.warning(f"Found {len(role_issues)} role issues (showing first 5):")
        for issue in role_issues[:5]:
            logger.warning(f"  {issue}")

    return {
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "role_issues": role_issues,
    }


def get_token_length_stats(dataset: Dataset, tokenizer) -> dict:
    """Compute token length distribution for the dataset.

    Applies the chat template to each conversation and measures token counts.

    Args:
        dataset: Dataset with 'messages' column.
        tokenizer: The model tokenizer with chat template support.

    Returns:
        Dict with min, max, mean, median, p95, p99, and num_truncated stats.
    """
    lengths = []
    for example in dataset:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        tokens = tokenizer(text, truncation=False)
        lengths.append(len(tokens["input_ids"]))

    lengths.sort()
    n = len(lengths)

    stats = {
        "min": lengths[0],
        "max": lengths[-1],
        "mean": sum(lengths) / n,
        "median": lengths[n // 2],
        "p95": lengths[int(n * 0.95)],
        "p99": lengths[int(n * 0.99)],
        "num_truncated": sum(1 for l in lengths if l > 2048),
        "total": n,
    }

    logger.info(
        f"Token length stats: min={stats['min']}, max={stats['max']}, "
        f"mean={stats['mean']:.0f}, median={stats['median']}, "
        f"p95={stats['p95']}, p99={stats['p99']}, "
        f"truncated (>2048): {stats['num_truncated']}/{n}"
    )

    return stats
