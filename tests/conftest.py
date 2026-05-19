"""Shared test helpers for quantization/sample-efficiency tests."""

from pathlib import Path
from typing import Optional


def make_fake_result_file(
    round_accuracies: list[float],
    num_users: int,
    path: Optional[Path] = None,
) -> Path:
    """Placeholder helper; implemented by later quant-analysis tasks."""
    raise NotImplementedError("make_fake_result_file is implemented in a later task")
