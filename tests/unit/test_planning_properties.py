"""Planning property tests for the quantization master script."""

import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from scripts.run_quant_efficiency import (
    ExperimentCondition,
    format_result_filename,
    parse_result_filename,
    validate_quantization,
)
from src.inference import GemmaInference


@given(label=st.text())
@settings(max_examples=200)
def test_quantization_label_white_list(label):
    """Feature: quantization-and-sample-efficiency, Property 1: Quantization label white-list"""
    if label in {"bf16", "int8", "int4"}:
        validate_quantization(label)
        ExperimentCondition("bayesian", label, "flight", "models/x", "x")
        assert format_result_filename("x", "bayesian", label, "flight").endswith(
            f"_{label}_flight.json"
        )
    else:
        with pytest.raises(ValueError):
            validate_quantization(label)
        with pytest.raises(ValueError):
            GemmaInference(model_path="unused", quantization=label)


@given(
    slug=st.from_regex(r"[a-zA-Z0-9._-]{1,30}", fullmatch=True),
    teaching=st.sampled_from(["bayesian", "oracle"]),
    quant=st.sampled_from(["bf16", "int8", "int4"]),
    domain=st.sampled_from(["flight", "hotel"]),
)
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_result_file_path_round_trip(slug, teaching, quant, domain):
    """Feature: quantization-and-sample-efficiency, Property 2: Path round-trip"""
    name = format_result_filename(slug, teaching, quant, domain)
    assert parse_result_filename(name) == (slug, teaching, quant, domain)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_text(
            json.dumps(
                {
                    "summary": {"round_accuracies": [0, 0, 0, 0, 0]},
                    "per_user": [],
                }
            )
        )
        assert parse_result_filename(path.name)[1:] == (teaching, quant, domain)
