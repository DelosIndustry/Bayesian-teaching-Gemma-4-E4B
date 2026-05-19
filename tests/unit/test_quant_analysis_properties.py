"""Property tests for quantization/sample-efficiency analysis."""

import numpy as np
from hypothesis import given, settings, strategies as st

from src.quant_analysis import (
    _convergence_absolute,
    _convergence_relative,
    compute_AUC,
    compute_robustness,
)


@given(rs=st.lists(st.floats(0, 1, allow_nan=False), min_size=5, max_size=5))
@settings(max_examples=200)
def test_auc_equals_trapezoidal_property(rs):
    """Feature: quantization-and-sample-efficiency, Property 4: AUC = trapezoidal"""
    expected = ((rs[0] + rs[4]) / 2 + rs[1] + rs[2] + rs[3]) / 4
    assert np.isclose(compute_AUC(rs), expected)
    assert np.isclose(compute_AUC(rs), np.trapezoid(rs, x=[1, 2, 3, 4, 5]) / 4)
    if len(set(rs)) == 1:
        assert compute_AUC(rs) == rs[0]


@given(
    correct=st.lists(
        st.lists(st.booleans(), min_size=5, max_size=5),
        min_size=1,
        max_size=100,
    ),
    threshold=st.integers(1, 5),
)
@settings(max_examples=200)
def test_convergence_round_monotone_reachability(correct, threshold):
    """Feature: quantization-and-sample-efficiency, Property 6: Convergence_Round monotone reachability"""
    matrix = np.asarray(correct, dtype=bool)
    conv_abs = _convergence_absolute(matrix, threshold=threshold)
    cumulative = np.cumsum(matrix.astype(int), axis=1)

    assert np.all((1 <= conv_abs) & (conv_abs <= 6))
    reached = cumulative[:, -1] >= threshold
    assert np.array_equal(conv_abs <= 5, reached)
    if threshold < 5:
        conv_next = _convergence_absolute(matrix, threshold=threshold + 1)
        assert np.all(conv_abs <= conv_next)

    all_true = np.ones((1, 5), dtype=bool)
    assert _convergence_absolute(all_true, threshold=threshold).tolist() == [threshold]
    all_false = np.zeros((1, 5), dtype=bool)
    assert _convergence_absolute(all_false, threshold=threshold).tolist() == [6]

    low = _convergence_relative(matrix, r5_threshold=0.2)
    high = _convergence_relative(matrix, r5_threshold=0.8)
    assert np.all(low <= high)


@given(
    r5_bf16=st.floats(0, 1, allow_nan=False),
    r5_int8=st.floats(0, 1, allow_nan=False),
    le_bf16=st.floats(0, 1, allow_nan=False),
    le_int8=st.floats(0, 1, allow_nan=False),
)
@settings(max_examples=200)
def test_robustness_arithmetic_consistency(r5_bf16, r5_int8, le_bf16, le_int8):
    """Feature: quantization-and-sample-efficiency, Property 3: Robustness arithmetic"""
    per_condition = {
        ("bayesian", "bf16", "flight"): {
            "summary": {
                "round_accuracies": [r5_bf16 - le_bf16, 0, 0, 0, r5_bf16],
                "learning_effect": le_bf16,
            },
            "per_user": [],
        },
        ("bayesian", "int8", "flight"): {
            "summary": {
                "round_accuracies": [r5_int8 - le_int8, 0, 0, 0, r5_int8],
                "learning_effect": le_int8,
            },
            "per_user": [],
        },
        ("oracle", "bf16", "flight"): {
            "summary": {
                "round_accuracies": [r5_bf16 - le_bf16, 0, 0, 0, r5_bf16],
                "learning_effect": le_bf16,
            },
            "per_user": [],
        },
        ("oracle", "int8", "flight"): {
            "summary": {
                "round_accuracies": [r5_int8 - le_int8, 0, 0, 0, r5_int8],
                "learning_effect": le_int8,
            },
            "per_user": [],
        },
    }

    robustness = compute_robustness(per_condition)
    bayes = robustness["by_condition"]["bayesian_int8_flight"]
    gap = robustness["gaps"]["int8_flight"]

    assert np.isclose(bayes["delta_R5"], r5_bf16 - r5_int8)
    assert np.isclose(bayes["delta_LE"], le_bf16 - le_int8)
    assert np.isclose(gap["gap_R5"], 0.0)
    assert np.isclose(gap["gap_LE"], 0.0)
