"""Basic hand-crafted tests for quantization analysis helpers."""

import numpy as np

from src.quant_analysis import (
    _convergence_absolute,
    compute_AUC,
    compute_robustness,
    compute_round_accuracies,
)


def test_compute_round_accuracies__manual_example():
    per_user = [
        {"rounds": [{"round_idx": i, "correct": v} for i, v in enumerate(row)]}
        for row in (
            [True, True, False, False, True],
            [False, True, False, True, True],
            [True, False, False, True, False],
        )
    ]

    assert compute_round_accuracies(per_user) == [
        2 / 3,
        2 / 3,
        0.0,
        2 / 3,
        2 / 3,
    ]


def test_compute_auc__all_equal_returns_input():
    for value in (0.0, 0.5, 1.0):
        assert compute_AUC([value] * 5) == value


def test_robustness_drop__simple_subtraction():
    result = compute_robustness(
        {
            ("bayesian", "bf16", "flight"): {
                "summary": {"round_accuracies": [0.3, 0.4, 0.5, 0.6, 0.7]},
                "per_user": [],
            },
            ("bayesian", "int8", "flight"): {
                "summary": {"round_accuracies": [0.3, 0.4, 0.5, 0.6, 0.65]},
                "per_user": [],
            },
        }
    )

    row = result["by_condition"]["bayesian_int8_flight"]
    assert round(row["delta_R5"], 10) == 0.05


def test_convergence_absolute__all_correct_returns_threshold():
    matrix = np.asarray([[True, True, True, True, True]])
    assert _convergence_absolute(matrix, threshold=3).tolist() == [3]


def test_convergence_absolute__all_wrong_returns_R_plus_1():
    matrix = np.asarray([[False, False, False, False, False]])
    assert _convergence_absolute(matrix, threshold=3).tolist() == [6]
