"""Analysis utilities for quantization robustness and sample efficiency."""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

ConditionKey = tuple[str, str, str]  # teaching, quantization, domain

TEACHINGS = {"bayesian", "oracle"}
QUANTS = {"bf16", "int8", "int4"}
RESULT_RE = re.compile(
    r"^(?P<slug>.+)_(?P<teaching>bayesian|oracle)_"
    r"(?P<quant>bf16|int8|int4)_(?P<domain>.+)\.json$"
)


class BrokenResultFileError(ValueError):
    """Raised when a result file is present but cannot be analyzed."""


def parse_result_filename(name: str) -> tuple[str, str, str, str]:
    """Parse `{slug}_{teaching}_{quant}_{domain}.json` result filenames."""
    match = RESULT_RE.match(Path(name).name)
    if not match:
        raise ValueError(f"not a result filename: {name}")
    return (
        match.group("slug"),
        match.group("teaching"),
        match.group("quant"),
        match.group("domain"),
    )


def _validate_result_payload(path: Path, payload: dict[str, Any]) -> None:
    if not isinstance(payload.get("summary"), dict):
        raise BrokenResultFileError(f"{path}: missing summary object")
    if "round_accuracies" not in payload["summary"]:
        raise BrokenResultFileError(f"{path}: missing summary.round_accuracies")
    if "per_user" not in payload:
        raise BrokenResultFileError(f"{path}: missing per_user")


def _load_result_file(path: Path) -> tuple[ConditionKey, dict[str, Any]]:
    try:
        slug, teaching, quant, domain = parse_result_filename(path.name)
    except ValueError as exc:
        raise BrokenResultFileError(str(exc)) from exc
    try:
        with open(path) as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise BrokenResultFileError(f"{path}: invalid JSON: {exc}") from exc
    _validate_result_payload(path, payload)
    payload.setdefault("model_slug", slug)
    payload.setdefault("teaching", teaching)
    payload.setdefault("quantization", quant)
    payload.setdefault("domain", domain)
    return (teaching, quant, domain), payload


def load_results_from_dir(
    base_results_dir: Path, quant_results_dir: Path
) -> dict[ConditionKey, dict[str, Any]]:
    """Load baseline bf16 results plus quantized results from disk."""
    loaded: dict[ConditionKey, dict[str, Any]] = {}
    for directory, allowed_quants in (
        (Path(base_results_dir), {"bf16"}),
        (Path(quant_results_dir), {"bf16", "int8", "int4"}),
    ):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                _, _, quant, _ = parse_result_filename(path.name)
            except ValueError:
                continue
            if quant not in allowed_quants:
                continue
            key, payload = _load_result_file(path)
            loaded[key] = payload
    return loaded


def compute_round_accuracies(
    per_user: list[dict[str, Any]], num_rounds: int = 5
) -> list[float]:
    """Compute per-round accuracy from `per_user` traces."""
    correct = np.zeros(num_rounds, dtype=float)
    total = np.zeros(num_rounds, dtype=float)
    for user in per_user:
        for idx, round_result in enumerate(user.get("rounds", [])[:num_rounds]):
            round_idx = int(round_result.get("round_idx", idx))
            if 0 <= round_idx < num_rounds:
                if (
                    round_result.get("heldout_total") is not None
                    and round_result.get("heldout_correct") is not None
                ):
                    total[round_idx] += float(round_result.get("heldout_total") or 0)
                    correct[round_idx] += float(round_result.get("heldout_correct") or 0)
                else:
                    total[round_idx] += 1
                    correct[round_idx] += float(bool(round_result.get("correct", False)))
    return [
        float(c / t) if t else 0.0 for c, t in zip(correct.tolist(), total.tolist())
    ]


def compute_per_user_round_correctness(
    per_user: list[dict[str, Any]], num_rounds: int = 5
) -> np.ndarray:
    """Return a `(users, rounds)` matrix of bool correctness or heldout accuracy."""
    matrix = np.zeros((len(per_user), num_rounds), dtype=float)
    for user_idx, user in enumerate(per_user):
        for idx, round_result in enumerate(user.get("rounds", [])[:num_rounds]):
            round_idx = int(round_result.get("round_idx", idx))
            if 0 <= round_idx < num_rounds:
                if round_result.get("accuracy") is not None:
                    matrix[user_idx, round_idx] = float(round_result["accuracy"])
                elif round_result.get("heldout_total"):
                    matrix[user_idx, round_idx] = (
                        float(round_result.get("heldout_correct", 0))
                        / float(round_result["heldout_total"])
                    )
                else:
                    matrix[user_idx, round_idx] = float(
                        bool(round_result.get("correct", False))
                    )
    return matrix


def compute_AUC(round_accuracies: list[float] | np.ndarray) -> float:
    """Compute normalized trapezoidal AUC over rounds."""
    values = np.asarray(round_accuracies, dtype=float)
    if values.size == 0:
        return 0.0
    if values.size == 1:
        return float(values[0])
    x = np.arange(1, values.size + 1)
    return float(np.trapezoid(values, x=x) / (values.size - 1))


def _convergence_absolute(
    correct_matrix: np.ndarray, threshold: int = 3
) -> np.ndarray:
    """First round where cumulative correct count reaches `threshold`."""
    matrix = np.asarray(correct_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("correct_matrix must be 2D")
    users, rounds = matrix.shape
    convergence = np.full(users, rounds + 1, dtype=int)
    cumulative = np.cumsum(matrix, axis=1)
    for user_idx in range(users):
        hits = np.flatnonzero(cumulative[user_idx] >= threshold)
        if hits.size:
            convergence[user_idx] = int(hits[0] + 1)
    return convergence


def _convergence_relative(
    correct_matrix: np.ndarray, r5_threshold: float, factor: float = 0.8
) -> np.ndarray:
    """First round where cumulative accuracy reaches `r5_threshold * factor`."""
    matrix = np.asarray(correct_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("correct_matrix must be 2D")
    users, rounds = matrix.shape
    target = float(r5_threshold) * factor
    convergence = np.full(users, rounds + 1, dtype=int)
    cumulative_mean = np.cumsum(matrix.astype(float), axis=1) / np.arange(
        1, rounds + 1
    )
    for user_idx in range(users):
        hits = np.flatnonzero(cumulative_mean[user_idx] >= target)
        if hits.size:
            convergence[user_idx] = int(hits[0] + 1)
    return convergence


def compute_sample_efficiency(correct_matrix: np.ndarray) -> dict[str, Any]:
    """Compute sample-efficiency metrics for one condition."""
    matrix = np.asarray(correct_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("correct_matrix must be 2D")
    users, rounds = matrix.shape
    round_acc = (
        matrix.astype(float).mean(axis=0) if users else np.zeros(rounds, dtype=float)
    )
    metrics: dict[str, Any] = {
        f"R{i + 1}": float(round_acc[i]) for i in range(min(rounds, 5))
    }
    for i in range(rounds, 5):
        metrics[f"R{i + 1}"] = 0.0

    le = float(round_acc[-1] - round_acc[0]) if rounds else 0.0
    sample_efficiency_score = (
        float(round_acc[2] - round_acc[0]) if rounds >= 3 else le
    )
    conv_rel = _convergence_relative(matrix, float(round_acc[-1]) if rounds else 0.0)
    conv_abs = _convergence_absolute(matrix, threshold=3)
    cumulative_mean = (
        np.cumsum(matrix.astype(float), axis=1) / np.arange(1, rounds + 1)
        if rounds
        else np.zeros((users, 0), dtype=float)
    )
    learning_curve_std = (
        float(np.std(cumulative_mean, axis=0).mean()) if users and rounds else 0.0
    )

    metrics.update(
        {
            "LE": le,
            "sample_efficiency_score": sample_efficiency_score,
            "AUC": compute_AUC(round_acc),
            "convergence_round_relative": conv_rel.tolist(),
            "convergence_round_absolute": conv_abs.tolist(),
            "convergence_round_mean": float(np.mean(conv_rel)) if users else 0.0,
            "convergence_round_median": float(np.median(conv_rel)) if users else 0.0,
            "learning_curve_std": learning_curve_std,
        }
    )
    return metrics


def _condition_metrics(result: dict[str, Any]) -> dict[str, float]:
    summary = result.get("summary", {})
    round_acc = summary.get("round_accuracies")
    if not round_acc:
        round_acc = [summary.get(f"R{i}", 0.0) for i in range(1, 6)]
    le = summary.get("learning_effect", summary.get("LE"))
    if le is None:
        le = float(round_acc[-1] - round_acc[0]) if round_acc else 0.0
    return {
        "R1": float(round_acc[0]) if round_acc else 0.0,
        "R5": float(round_acc[-1]) if round_acc else 0.0,
        "LE": float(le),
    }


def compute_per_user_delta_LE(
    correct_bf16: np.ndarray, correct_quant: np.ndarray
) -> np.ndarray:
    """Per-user LE drop: `(bf16_R5-bf16_R1) - (quant_R5-quant_R1)`."""
    bf16 = np.asarray(correct_bf16, dtype=float)
    quant = np.asarray(correct_quant, dtype=float)
    if bf16.shape != quant.shape:
        raise ValueError(f"shape mismatch: {bf16.shape} != {quant.shape}")
    if bf16.ndim != 2 or bf16.shape[1] == 0:
        return np.asarray([], dtype=float)
    le_bf16 = bf16[:, -1] - bf16[:, 0]
    le_quant = quant[:, -1] - quant[:, 0]
    return le_bf16 - le_quant


def _summarize_delta(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"values": [], "mean": 0.0, "median": 0.0, "iqr": 0.0}
    q75, q25 = np.percentile(arr, [75, 25])
    return {
        "values": arr.tolist(),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "iqr": float(q75 - q25),
    }


def compute_robustness(
    per_condition: dict[ConditionKey, dict[str, Any]]
) -> dict[str, Any]:
    """Compute R5/LE drops and Bayesian-vs-Oracle robustness gaps."""
    by_condition: dict[str, dict[str, Any]] = {}
    gaps: dict[str, dict[str, float]] = {}
    deltas: dict[tuple[str, str, str], dict[str, float]] = {}

    domains = sorted({domain for _, _, domain in per_condition})
    for domain in domains:
        for teaching in sorted({t for t, _, d in per_condition if d == domain}):
            baseline = per_condition.get((teaching, "bf16", domain))
            if baseline is None:
                continue
            baseline_metrics = _condition_metrics(baseline)
            baseline_matrix = compute_per_user_round_correctness(
                baseline.get("per_user", [])
            )
            for quant in ("bf16", "int8", "int4"):
                result = per_condition.get((teaching, quant, domain))
                if result is None:
                    continue
                metrics = _condition_metrics(result)
                delta_r5 = baseline_metrics["R5"] - metrics["R5"]
                delta_le = baseline_metrics["LE"] - metrics["LE"]
                key = f"{teaching}_{quant}_{domain}"
                row = {
                    "teaching": teaching,
                    "quant": quant,
                    "domain": domain,
                    **metrics,
                    "delta_R5": delta_r5,
                    "delta_LE": delta_le,
                }
                if quant != "bf16":
                    quant_matrix = compute_per_user_round_correctness(
                        result.get("per_user", [])
                    )
                    try:
                        per_user_delta = compute_per_user_delta_LE(
                            baseline_matrix, quant_matrix
                        )
                        row["per_user_delta_LE"] = _summarize_delta(per_user_delta)
                    except ValueError:
                        row["per_user_delta_LE"] = {
                            "values": [],
                            "mean": 0.0,
                            "median": 0.0,
                            "iqr": 0.0,
                        }
                by_condition[key] = row
                deltas[(teaching, quant, domain)] = {
                    "delta_R5": delta_r5,
                    "delta_LE": delta_le,
                }

        for quant in ("int8", "int4"):
            bayes = deltas.get(("bayesian", quant, domain))
            oracle = deltas.get(("oracle", quant, domain))
            if bayes and oracle:
                gaps[f"{quant}_{domain}"] = {
                    "quant": quant,
                    "domain": domain,
                    "gap_R5": oracle["delta_R5"] - bayes["delta_R5"],
                    "gap_LE": oracle["delta_LE"] - bayes["delta_LE"],
                }

    return {"by_condition": by_condition, "gaps": gaps}


def align_users_across_teachings(
    per_user_bayesian: list[dict[str, Any]], per_user_oracle: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    """Align Bayesian and Oracle correctness matrices by shared user_idx."""
    bayes_by_user = {u["user_idx"]: u for u in per_user_bayesian}
    oracle_by_user = {u["user_idx"]: u for u in per_user_oracle}
    common = sorted(set(bayes_by_user) & set(oracle_by_user))
    bayes = compute_per_user_round_correctness([bayes_by_user[u] for u in common])
    oracle = compute_per_user_round_correctness([oracle_by_user[u] for u in common])
    return bayes, oracle


def _rank_biserial(diff: np.ndarray) -> float:
    nonzero = diff[diff != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    pos = float(ranks[nonzero > 0].sum())
    neg = float(ranks[nonzero < 0].sum())
    denom = pos + neg
    return (pos - neg) / denom if denom else 0.0


def run_significance_tests(
    per_user_delta_le_paired_diff: dict[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    """Run paired tests on Oracle-minus-Bayesian robustness deltas."""
    outputs: dict[str, dict[str, Any]] = {}
    for quant, values in per_user_delta_le_paired_diff.items():
        diff = np.asarray(values, dtype=float)
        nonzero = diff[diff != 0]
        test_name = f"wilcoxon_delta_LE_{quant}"
        if nonzero.size == 0:
            outputs[quant] = {
                "test_name": test_name,
                "statistic": None,
                "p_value": None,
                "n_pairs": 0,
                "effect_size": 0.0,
                "interpretation": "degenerate",
            }
            continue

        wilcoxon = stats.wilcoxon(
            diff, zero_method="wilcox", alternative="two-sided"
        )
        std = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
        cohen_d = float(np.mean(diff) / std) if std > 0 else 0.0
        result: dict[str, Any] = {
            "test_name": test_name,
            "statistic": float(wilcoxon.statistic),
            "p_value": float(wilcoxon.pvalue),
            "n_pairs": int(nonzero.size),
            "effect_size": _rank_biserial(diff),
            "cohen_d": cohen_d,
            "interpretation": "significant" if wilcoxon.pvalue < 0.05 else "n.s.",
        }
        if diff.size >= 3:
            shapiro = stats.shapiro(diff)
            result["shapiro_p_value"] = float(shapiro.pvalue)
            if shapiro.pvalue > 0.05:
                paired_t = stats.ttest_1samp(diff, popmean=0.0)
                result["paired_t"] = {
                    "statistic": float(paired_t.statistic),
                    "p_value": float(paired_t.pvalue),
                }
            else:
                result["normality_warning"] = "Shapiro-Wilk p <= 0.05"
        if wilcoxon.pvalue < 0.05:
            print(f"\033[1m{test_name}: p={wilcoxon.pvalue:.4g}, significant\033[0m")
        outputs[quant] = result
    return outputs


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def _resolve_output(out: Path, filename: str) -> Path:
    out = Path(out)
    return out if out.suffix else out / filename


def _condition_key_to_str(key: ConditionKey) -> str:
    teaching, quant, domain = key
    return f"{teaching}_{quant}_{domain}"


def write_sample_efficiency_summary(
    out: Path, by_cond: dict[ConditionKey, dict[str, Any]]
) -> Path:
    payload: dict[str, Any] = {}
    for key, result in by_cond.items():
        if "per_user" in result:
            matrix = compute_per_user_round_correctness(result["per_user"])
            payload[_condition_key_to_str(key)] = compute_sample_efficiency(matrix)
        else:
            payload[_condition_key_to_str(key)] = result
    path = _resolve_output(out, "sample_efficiency_summary.json")
    atomic_write_json(path, payload)
    return path


def write_robustness_summary(out: Path, robustness: dict[str, Any]) -> Path:
    path = _resolve_output(out, "robustness_summary.json")
    atomic_write_json(path, robustness)
    return path


def write_significance_tests(out: Path, tests: dict[str, Any]) -> Path:
    path = _resolve_output(out, "significance_tests.json")
    atomic_write_json(path, tests)
    return path


def write_efficiency_table_csv(
    out: Path, results: dict[ConditionKey, dict[str, Any]]
) -> Path:
    path = _resolve_output(out, "efficiency_table.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fields = [
        "teaching",
        "quant",
        "domain",
        "allocated_mib",
        "reserved_mib",
        "peak_mib",
        "elapsed_seconds_total",
        "elapsed_seconds_per_user",
        "mean_seconds_per_round",
        "mean_tokens_per_round",
    ]
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (teaching, quant, domain), result in sorted(results.items()):
            memory = result.get("memory_mib", {})
            summary = result.get("summary", {})
            latency = summary.get("latency", {})
            num_users = summary.get("num_users") or result.get("max_users") or 0
            elapsed = result.get(
                "elapsed_seconds_total", summary.get("elapsed_seconds_total", 0.0)
            )
            writer.writerow(
                {
                    "teaching": teaching,
                    "quant": quant,
                    "domain": domain,
                    "allocated_mib": memory.get("allocated_mib", 0.0),
                    "reserved_mib": memory.get("reserved_mib", 0.0),
                    "peak_mib": memory.get("peak_mib", 0.0),
                    "elapsed_seconds_total": elapsed,
                    "elapsed_seconds_per_user": (
                        elapsed / num_users if num_users else 0.0
                    ),
                    "mean_seconds_per_round": latency.get(
                        "mean_seconds_per_round", 0.0
                    ),
                    "mean_tokens_per_round": latency.get(
                        "mean_tokens_per_round", 0.0
                    ),
                }
            )
    os.replace(tmp, path)
    return path


def write_robustness_table_csv(out: Path, robustness: dict[str, Any]) -> Path:
    path = _resolve_output(out, "robustness_table.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fields = ["teaching", "quant", "domain", "R1", "R5", "LE", "delta_R5", "delta_LE"]
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(robustness.get("by_condition", {}).values(), key=str):
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(tmp, path)
    return path
