"""Visualization helpers for quantization robustness experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.quant_analysis import ConditionKey, atomic_write_json

TEACHING_COLORS = {"bayesian": "#4C78A8", "oracle": "#F58518"}
QUANT_STYLES = {"bf16": "-", "int8": "--", "int4": ":"}
QUANT_ORDER = ["bf16", "int8", "int4"]
TEACHING_ORDER = ["bayesian", "oracle"]


def _rounds(record: dict[str, Any]) -> list[float]:
    summary = record.get("summary", record)
    if "round_accuracies" in summary:
        return [float(v) for v in summary["round_accuracies"]]
    return [float(summary.get(f"R{i}", 0.0)) for i in range(1, 6)]


def _r5_le(record: dict[str, Any]) -> tuple[float, float]:
    rounds = _rounds(record)
    summary = record.get("summary", record)
    le = summary.get("learning_effect", summary.get("LE"))
    if le is None:
        le = rounds[-1] - rounds[0] if rounds else 0.0
    return (rounds[-1] if rounds else 0.0, float(le))


def _filter_domain(
    by_cond: dict[ConditionKey, dict[str, Any]], domain: str
) -> list[tuple[ConditionKey, dict[str, Any]]]:
    return [
        (key, value)
        for key, value in by_cond.items()
        if len(key) == 3 and key[2] == domain
    ]


def make_learning_curves_figure(
    by_cond: dict[ConditionKey, dict[str, Any]], domain: str = "flight"
):
    """Create a 6-line learning-curve figure for one domain."""
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    for teaching in TEACHING_ORDER:
        for quant in QUANT_ORDER:
            record = by_cond.get((teaching, quant, domain))
            if record is None:
                continue
            ys = _rounds(record)
            xs = list(range(1, len(ys) + 1))
            ax.plot(
                xs,
                ys,
                color=TEACHING_COLORS[teaching],
                linestyle=QUANT_STYLES[quant],
                marker="o",
                linewidth=2.0,
                label=f"{teaching.title()} {quant}",
            )
    ax.set_title(f"Learning Curves ({domain.title()})")
    ax.set_xlabel("Round")
    ax.set_ylabel("Round Accuracy")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return fig


def make_r5_le_bar_figure(by_cond: dict[ConditionKey, dict[str, Any]]):
    """Create a two-panel R5 and Learning Effect bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=False)
    width = 0.36
    x = np.arange(len(QUANT_ORDER))
    domain = "flight"
    if by_cond:
        domain = sorted({key[2] for key in by_cond})[0]

    for offset, teaching in zip((-width / 2, width / 2), TEACHING_ORDER):
        r5_values = []
        le_values = []
        for quant in QUANT_ORDER:
            record = by_cond.get((teaching, quant, domain), {})
            r5, le = _r5_le(record)
            r5_values.append(r5)
            le_values.append(le)
        axes[0].bar(
            x + offset,
            r5_values,
            width,
            label=teaching.title(),
            color=TEACHING_COLORS[teaching],
        )
        axes[1].bar(
            x + offset,
            le_values,
            width,
            label=teaching.title(),
            color=TEACHING_COLORS[teaching],
        )

    for ax, title, ylabel in (
        (axes[0], "R5 Accuracy", "Accuracy"),
        (axes[1], "Learning Effect", "R5 - R1"),
    ):
        ax.set_title(title)
        ax.set_xlabel("Quantization")
        ax.set_xticks(x, QUANT_ORDER)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylim(0.0, 1.0)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def make_scatter_per_user_efficiency(bayesian_eff, oracle_eff):
    """Create Bayesian-vs-Oracle per-user efficiency scatter."""
    bayes = np.asarray(bayesian_eff, dtype=float)
    oracle = np.asarray(oracle_eff, dtype=float)
    n = min(bayes.size, oracle.size)
    bayes = bayes[:n]
    oracle = oracle[:n]

    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.scatter(bayes, oracle, s=18, alpha=0.7, color="#54A24B")
    if n:
        lo = float(min(bayes.min(), oracle.min()))
        hi = float(max(bayes.max(), oracle.max()))
    else:
        lo, hi = -1.0, 1.0
    if lo == hi:
        lo -= 0.1
        hi += 0.1
    ax.plot([lo, hi], [lo, hi], color="#666666", linewidth=1.2, linestyle="--")
    ax.set_title("Per-User Sample Efficiency")
    ax.set_xlabel("Bayesian Efficiency")
    ax.set_ylabel("Oracle Efficiency")
    ax.grid(True, alpha=0.25)
    return fig


def make_cross_domain_robustness_figure(robustness_flight, robustness_hotel):
    """Create an optional Flight-vs-Hotel robustness comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=True)
    width = 0.36
    x = np.arange(len(QUANT_ORDER[1:]))
    for ax, teaching in zip(axes, TEACHING_ORDER):
        for offset, (domain, robustness) in zip(
            (-width / 2, width / 2),
            (("flight", robustness_flight), ("hotel", robustness_hotel)),
        ):
            values = []
            for quant in QUANT_ORDER[1:]:
                row = robustness.get("by_condition", {}).get(
                    f"{teaching}_{quant}_{domain}", {}
                )
                values.append(float(row.get("delta_LE", 0.0)))
            ax.bar(x + offset, values, width, label=domain.title())
        ax.set_title(teaching.title())
        ax.set_xlabel("Quantization")
        ax.set_xticks(x, QUANT_ORDER[1:])
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Delta LE")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Cross-Domain Robustness")
    fig.tight_layout()
    return fig


def save_figures(figures: list[tuple[str, Any]], out_dir: Path) -> None:
    """Atomically save all figures as PDF and PNG, or none if any is None."""
    if any(fig is None for _, fig in figures):
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_paths: list[tuple[Path, Path]] = []
    try:
        for name, fig in figures:
            for ext in ("pdf", "png"):
                final = out_dir / f"{name}.{ext}"
                tmp = out_dir / f".{name}.{ext}.tmp.{ext}"
                fig.savefig(tmp, dpi=300, bbox_inches="tight")
                tmp_paths.append((tmp, final))
        for tmp, final in tmp_paths:
            os.replace(tmp, final)
    except BaseException:
        for tmp, _ in tmp_paths:
            if tmp.exists():
                tmp.unlink()
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def write_robustness_latex_table(robustness, out: Path) -> None:
    rows = sorted(
        robustness.get("by_condition", {}).values(),
        key=lambda r: (r.get("teaching", ""), r.get("quant", ""), r.get("domain", "")),
    )
    lines = [
        "\\begin{tabular}{lllrrrrr}",
        "\\toprule",
        "Teaching & Quant & Domain & R1 & R5 & LE & $\\Delta$R5 & $\\Delta$LE \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row.get('teaching')} & {row.get('quant')} & {row.get('domain')} & "
            f"{row.get('R1', 0):.3f} & {row.get('R5', 0):.3f} & "
            f"{row.get('LE', 0):.3f} & {row.get('delta_R5', 0):.3f} & "
            f"{row.get('delta_LE', 0):.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _atomic_write_text(Path(out), "\n".join(lines) + "\n")


def write_sample_efficiency_latex_table(by_cond, out: Path) -> None:
    lines = [
        "\\begin{tabular}{lllrrr}",
        "\\toprule",
        "Teaching & Quant & Domain & R3-R1 & AUC & Convergence \\\\",
        "\\midrule",
    ]
    for (teaching, quant, domain), record in sorted(by_cond.items()):
        summary = record.get("summary", record)
        lines.append(
            f"{teaching} & {quant} & {domain} & "
            f"{summary.get('sample_efficiency_score', 0):.3f} & "
            f"{summary.get('AUC', 0):.3f} & "
            f"{summary.get('convergence_round_mean', 0):.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _atomic_write_text(Path(out), "\n".join(lines) + "\n")


def write_captions_md(out: Path, captions: dict[str, str]) -> None:
    chunks = []
    for name, caption in captions.items():
        chunks.append(f"## {name}\n\n{caption}\n")
    _atomic_write_text(Path(out), "\n".join(chunks))
