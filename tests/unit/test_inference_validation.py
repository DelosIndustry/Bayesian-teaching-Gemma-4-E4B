"""Feature: quantization-and-sample-efficiency, inference validation tests."""

import pytest

from src.inference import GemmaInference


def test_gemma_inference_rejects_unknown_quantization_label():
    with pytest.raises(ValueError, match="unknown quantization label: fp8"):
        GemmaInference(model_path="unused-model", quantization="fp8")
