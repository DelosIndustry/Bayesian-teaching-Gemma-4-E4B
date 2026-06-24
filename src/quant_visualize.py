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

# Okabe-Ito inspired, print-friendly colors.
TEACHING_COLORS = {"bayesian": "#0072B2", "oracle": "#D55E00"}
QUANT_STYLES = {"bf16": "-", "int8": "--", "int4": "-."}
QUANT_MARKERS = {"bf16": "o", "int8": "s", "int4": "^"}
QUANT_ORDER = ["bf16", "int8", "int4"]
TEACHING_ORDER = ["bayesian", "oracle"]
TEACHING_LABELS = {"bayesian": "Bayesian", "oracle": "Oracle"}
QUANT_LABELS = {"bf16": "BF16", "int8": "INT8", "int4": "INT4"}


def _set_paper_style() -> None:
    """Use compact, publication-oriented matplotlib defaults."""
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 450,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "axes.titleweight": "semibold",
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _style_axes(ax, *, grid_axis: str = "y") -> None:
    """Apply clean paper-style axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(axis="both", colors="#333333", width=0.8, length=3)
    ax.grid(True, axis=grid_axis)
    ax.set_axisbelow(True)


def _annotate_bars(ax, bars, *, dy: float = 0.006) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#333333",
            rotation=0,
        )


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
    _set_paper_style()
    fig, ax = plt.subplots(figsize=(6.1, 3.55))
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
                marker=QUANT_MARKERS[quant],
                markersize=5.2,
                markerfacecolor="white",
                markeredgewidth=1.2,
                linewidth=2.1,
                label=f"{TEACHING_LABELS[teaching]}-{QUANT_LABELS[quant]}",
            )
    ax.set_title(f"Interaction Learning Curves ({domain.title()})", pad=8)
    ax.set_xlabel("Interaction round")
    ax.set_ylabel("Accuracy")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(0.32, 0.75)
    ax.set_yticks(np.arange(0.35, 0.76, 0.10))
    _style_axes(ax, grid_axis="y")
    ax.legend(
        frameon=True,
        fancybox=False,
        edgecolor="#D0D0D0",
        facecolor="white",
        framealpha=0.95,
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        borderaxespad=0.0,
        handlelength=2.2,
        columnspacing=1.0,
    )
    fig.tight_layout()
    return fig


def make_r5_le_bar_figure(by_cond: dict[ConditionKey, dict[str, Any]]):
    """Create a two-panel R5 and Learning Effect bar chart."""
    _set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.45), sharey=False)
    width = 0.34
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
        bars_r5 = axes[0].bar(
            x + offset,
            r5_values,
            width,
            label=TEACHING_LABELS[teaching],
            color=TEACHING_COLORS[teaching],
            edgecolor="#222222",
            linewidth=0.55,
            alpha=0.92,
        )
        bars_le = axes[1].bar(
            x + offset,
            le_values,
            width,
            label=TEACHING_LABELS[teaching],
            color=TEACHING_COLORS[teaching],
            edgecolor="#222222",
            linewidth=0.55,
            alpha=0.92,
        )
        _annotate_bars(axes[0], bars_r5, dy=0.008)
        _annotate_bars(axes[1], bars_le, dy=0.006)

    for ax, title, ylabel in (
        (axes[0], "Final Accuracy (R5)", "Accuracy"),
        (axes[1], "Learning Effect (R5 - R1)", "Accuracy gain"),
    ):
        ax.set_title(title, pad=8)
        ax.set_xlabel("Quantization")
        ax.set_xticks(x, [QUANT_LABELS[q] for q in QUANT_ORDER])
        ax.set_ylabel(ylabel)
        _style_axes(ax, grid_axis="y")
    axes[0].set_ylim(0.0, 0.80)
    axes[1].set_ylim(0.0, 0.38)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=True,
        fancybox=False,
        edgecolor="#D0D0D0",
        facecolor="white",
        framealpha=0.95,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        columnspacing=1.4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def make_scatter_per_user_efficiency(bayesian_eff, oracle_eff):
    """Create Bayesian-vs-Oracle per-user efficiency scatter."""
    _set_paper_style()
    bayes = np.asarray(bayesian_eff, dtype=float)
    oracle = np.asarray(oracle_eff, dtype=float)
    n = min(bayes.size, oracle.size)
    bayes = bayes[:n]
    oracle = oracle[:n]

    fig, ax = plt.subplots(figsize=(4.25, 4.05))
    ax.scatter(
        bayes,
        oracle,
        s=22,
        alpha=0.62,
        color="#009E73",
        edgecolor="white",
        linewidth=0.35,
    )
    if n:
        lo = float(min(bayes.min(), oracle.min()))
        hi = float(max(bayes.max(), oracle.max()))
    else:
        lo, hi = -1.0, 1.0
    if lo == hi:
        lo -= 0.1
        hi += 0.1
    pad = max((hi - lo) * 0.04, 0.03)
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], color="#4D4D4D", linewidth=1.2, linestyle="--")
    ax.fill_between([lo, hi], [lo, hi], [lo, lo], color="#0072B2", alpha=0.06)
    ax.text(
        lo + (hi - lo) * 0.05,
        lo + (hi - lo) * 0.90,
        "Oracle > Bayesian",
        fontsize=7.5,
        color="#666666",
    )
    ax.text(
        lo + (hi - lo) * 0.52,
        lo + (hi - lo) * 0.08,
        "Bayesian > Oracle",
        fontsize=7.5,
        color="#666666",
    )
    ax.set_title("Per-User Sample Efficiency", pad=8)
    ax.set_xlabel("Bayesian efficiency")
    ax.set_ylabel("Oracle efficiency")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    _style_axes(ax, grid_axis="both")
    fig.tight_layout()
    return fig


def make_cross_domain_robustness_figure(robustness_flight, robustness_hotel):
    """Create an optional Flight-vs-Hotel robustness comparison."""
    _set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.45), sharey=True)
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
            ax.bar(
                x + offset,
                values,
                width,
                label=domain.title(),
                edgecolor="#222222",
                linewidth=0.55,
            )
        ax.set_title(TEACHING_LABELS[teaching], pad=8)
        ax.set_xlabel("Quantization")
        ax.set_xticks(x, [QUANT_LABELS[q] for q in QUANT_ORDER[1:]])
        _style_axes(ax, grid_axis="y")
    axes[0].set_ylabel("$\\Delta$LE")
    axes[1].legend(frameon=True, fancybox=False, edgecolor="#D0D0D0")
    fig.suptitle("Cross-Domain Robustness", fontsize=10, fontweight="semibold")
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
                fig.savefig(tmp, dpi=450, bbox_inches="tight", facecolor="white")
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
