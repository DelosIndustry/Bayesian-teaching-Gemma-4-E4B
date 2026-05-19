#!/usr/bin/env python3
"""Master pipeline for quantization robustness and sample efficiency."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import transformers

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.quant_analysis import (  # noqa: E402
    ConditionKey,
    atomic_write_json,
    compute_per_user_delta_LE,
    compute_per_user_round_correctness,
    compute_robustness,
    load_results_from_dir,
    parse_result_filename,
    run_significance_tests,
    write_efficiency_table_csv,
    write_robustness_summary,
    write_robustness_table_csv,
    write_sample_efficiency_summary,
    write_significance_tests,
)
from src.quant_visualize import (  # noqa: E402
    make_cross_domain_robustness_figure,
    make_learning_curves_figure,
    make_r5_le_bar_figure,
    make_scatter_per_user_efficiency,
    save_figures,
    write_captions_md,
    write_robustness_latex_table,
    write_sample_efficiency_latex_table,
)


VALID_TEACHINGS = ("bayesian", "oracle")
VALID_QUANTS = ("bf16", "int8", "int4")
VALID_DOMAINS = ("flight", "hotel")
_LAST_EVALUATE_ENTRIES: list[dict] = []


@dataclass(frozen=True)
class ExperimentCondition:
    teaching: str
    quantization: str
    domain: str
    model_path: str
    model_slug: str

    def __post_init__(self):
        validate_quantization(self.quantization)


def validate_quantization(label: str) -> str:
    if label not in VALID_QUANTS:
        raise ValueError(f"unknown quantization label: {label}")
    return label


def format_result_filename(slug: str, teaching: str, quant: str, domain: str) -> str:
    validate_quantization(quant)
    if teaching not in VALID_TEACHINGS:
        raise ValueError(f"unknown teaching label: {teaching}")
    if domain not in VALID_DOMAINS:
        raise ValueError(f"unknown domain label: {domain}")
    return f"{slug}_{teaching}_{quant}_{domain}.json"


def plan_conditions(
    teachings: list[str],
    quants: list[str],
    domains: list[str],
    model_paths: dict[str, str],
) -> list[ExperimentCondition]:
    """Build deterministic domain -> teaching -> quant cartesian product."""
    plan = []
    for domain in domains:
        for teaching in teachings:
            model_path = model_paths[teaching]
            slug = Path(model_path).name
            for quant in quants:
                plan.append(
                    ExperimentCondition(
                        teaching=teaching,
                        quantization=quant,
                        domain=domain,
                        model_path=model_path,
                        model_slug=slug,
                    )
                )
    return plan


def resolve_result_path(
    cond: ExperimentCondition,
    output_dir: Path,
    legacy_results_dir: Path,
    reuse_legacy_bf16: bool = True,
) -> Path:
    legacy = (
        Path(legacy_results_dir)
        / format_result_filename(
            cond.model_slug, cond.teaching, cond.quantization, cond.domain
        )
    )
    if (
        reuse_legacy_bf16
        and cond.quantization == "bf16"
        and cond.domain == "flight"
        and legacy.exists()
    ):
        return legacy
    return Path(output_dir) / format_result_filename(
        cond.model_slug, cond.teaching, cond.quantization, cond.domain
    )


def should_skip(path: Path, force: bool) -> bool:
    return Path(path).exists() and not force


def _is_legacy_path(path: Path, legacy_results_dir: Path) -> bool:
    try:
        return path.parent.resolve() == Path(legacy_results_dir).resolve()
    except FileNotFoundError:
        return False


def _remove_partial(path: Path) -> None:
    for candidate in (path, path.with_suffix(path.suffix + ".tmp")):
        if candidate.exists():
            candidate.unlink()


def run_evaluate_stage(
    plan: list[ExperimentCondition],
    output_dir: Path,
    seed: int,
    force: bool,
    max_users: Optional[int],
    legacy_results_dir: Path = REPO_ROOT / "results",
    evaluation_mode: str = "paper",
    heldout_batch_size: int = 8,
) -> list[Path]:
    """Run missing conditions via subprocess and reuse existing baselines."""
    global _LAST_EVALUATE_ENTRIES
    _LAST_EVALUATE_ENTRIES = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_paths: list[Path] = []
    for cond in plan:
        target = resolve_result_path(
            cond,
            output_dir,
            legacy_results_dir,
            reuse_legacy_bf16=evaluation_mode == "interaction",
        )
        legacy_reuse = (
            cond.quantization == "bf16"
            and cond.domain == "flight"
            and _is_legacy_path(target, legacy_results_dir)
        )

        if legacy_reuse and target.exists():
            result_paths.append(target)
            _LAST_EVALUATE_ENTRIES.append(
                {**asdict(cond), "status": "reused", "path": str(target)}
            )
            continue

        if should_skip(target, force):
            result_paths.append(target)
            continue

        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate.py"),
            "--model-path",
            cond.model_path,
            "--teaching",
            cond.teaching,
            "--quantization",
            cond.quantization,
            "--domain",
            cond.domain,
            "--num-rounds",
            "5",
            "--seed",
            str(seed),
            "--evaluation-mode",
            evaluation_mode,
            "--heldout-batch-size",
            str(heldout_batch_size),
            "--output",
            str(target),
        ]
        if max_users is not None:
            cmd.extend(["--max-users", str(max_users)])

        completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            _remove_partial(target)
            _LAST_EVALUATE_ENTRIES.append(
                {
                    **asdict(cond),
                    "status": "failed",
                    "path": str(target),
                    "error": f"returncode={completed.returncode}",
                }
            )
            continue

        if target.exists():
            result_paths.append(target)
            entry = {**asdict(cond), "status": "produced", "path": str(target)}
            try:
                with open(target) as f:
                    payload = json.load(f)
                parse_failure = payload.get("summary", {}).get(
                    "parse_failure_rate", 0.0
                )
                if parse_failure > 0.05:
                    print(
                        f"WARNING: parse_failure_rate={parse_failure:.3f} "
                        f"for {target}"
                    )
                    entry["flagged"] = True
            except json.JSONDecodeError:
                entry["flagged"] = True
            _LAST_EVALUATE_ENTRIES.append(entry)
    return result_paths


def _sample_efficiency_by_condition(
    results: dict[ConditionKey, dict]
) -> dict[ConditionKey, dict]:
    from src.quant_analysis import compute_sample_efficiency

    output = {}
    for key, result in results.items():
        matrix = compute_per_user_round_correctness(result.get("per_user", []))
        output[key] = compute_sample_efficiency(matrix)
    return output


def _significance_inputs(results: dict[ConditionKey, dict]) -> dict[str, torch.Tensor]:
    diffs = {}
    for quant in ("int8", "int4"):
        required = [
            ("bayesian", "bf16", "flight"),
            ("bayesian", quant, "flight"),
            ("oracle", "bf16", "flight"),
            ("oracle", quant, "flight"),
        ]
        if not all(key in results for key in required):
            continue
        bayes_bf16 = compute_per_user_round_correctness(
            results[("bayesian", "bf16", "flight")]["per_user"]
        )
        bayes_quant = compute_per_user_round_correctness(
            results[("bayesian", quant, "flight")]["per_user"]
        )
        oracle_bf16 = compute_per_user_round_correctness(
            results[("oracle", "bf16", "flight")]["per_user"]
        )
        oracle_quant = compute_per_user_round_correctness(
            results[("oracle", quant, "flight")]["per_user"]
        )
        bayes_delta = compute_per_user_delta_LE(bayes_bf16, bayes_quant)
        oracle_delta = compute_per_user_delta_LE(oracle_bf16, oracle_quant)
        n = min(bayes_delta.size, oracle_delta.size)
        diffs[quant] = oracle_delta[:n] - bayes_delta[:n]
    return diffs


def run_analyze_stage(result_paths: list[Path], output_dir: Path) -> dict:
    """Load result files, compute metrics, and write analysis artifacts."""
    del result_paths  # The disk layout is the source of truth for analysis.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results_from_dir(REPO_ROOT / "results", output_dir)
    sample_efficiency = _sample_efficiency_by_condition(results)
    robustness = compute_robustness(results)
    significance = run_significance_tests(_significance_inputs(results))

    write_sample_efficiency_summary(output_dir, sample_efficiency)
    write_robustness_summary(output_dir, robustness)
    write_significance_tests(output_dir, significance)
    write_efficiency_table_csv(output_dir, results)
    write_robustness_table_csv(output_dir, robustness)
    return {
        "sample_efficiency": sample_efficiency,
        "robustness": robustness,
        "significance": significance,
    }


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _condition_from_key_string(key: str) -> ConditionKey:
    teaching, quant, *domain_parts = key.split("_")
    return (teaching, quant, "_".join(domain_parts))


def _per_user_efficiency(result: dict) -> list[float]:
    matrix = compute_per_user_round_correctness(result.get("per_user", []))
    if matrix.shape[1] < 3:
        return []
    return (matrix[:, 2].astype(float) - matrix[:, 0].astype(float)).tolist()


def run_visualize_stage(output_dir: Path, include_cross_domain: bool) -> None:
    """Generate figures, tables, and Korean captions."""
    output_dir = Path(output_dir)
    sample_path = output_dir / "sample_efficiency_summary.json"
    robustness_path = output_dir / "robustness_summary.json"
    if not sample_path.exists() or not robustness_path.exists():
        run_analyze_stage([], output_dir)

    sample_raw = _load_json(sample_path)
    sample_by_cond = {
        _condition_from_key_string(key): value for key, value in sample_raw.items()
    }
    robustness = _load_json(robustness_path)
    raw_results = load_results_from_dir(REPO_ROOT / "results", output_dir)

    figures = [
        ("learning_curves", make_learning_curves_figure(sample_by_cond, domain="flight")),
        ("bar_R5_LE", make_r5_le_bar_figure(sample_by_cond)),
    ]
    bayes_eff = []
    oracle_eff = []
    if ("bayesian", "bf16", "flight") in raw_results:
        bayes_eff = _per_user_efficiency(raw_results[("bayesian", "bf16", "flight")])
    if ("oracle", "bf16", "flight") in raw_results:
        oracle_eff = _per_user_efficiency(raw_results[("oracle", "bf16", "flight")])
    figures.append(
        ("scatter_per_user_eff", make_scatter_per_user_efficiency(bayes_eff, oracle_eff))
    )

    if include_cross_domain:
        hotel_rows = [
            row
            for row in robustness.get("by_condition", {}).values()
            if row.get("domain") == "hotel"
        ]
        if hotel_rows:
            figures.append(
                (
                    "cross_domain_robustness",
                    make_cross_domain_robustness_figure(robustness, robustness),
                )
            )

    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    save_figures(figures, figures_dir)
    write_captions_md(
        figures_dir / "captions.md",
        {
            "learning_curves": "라운드별 정확도 변화와 양자화 조건별 학습 곡선을 비교한 그림.",
            "bar_R5_LE": "최종 정확도와 학습 효과를 Bayesian 및 Oracle 조건별로 요약한 그림.",
            "scatter_per_user_eff": "동일 사용자 기준의 샘플 효율성을 Bayesian과 Oracle 사이에서 비교한 그림.",
            **(
                {
                    "cross_domain_robustness": "Flight와 Hotel 도메인에서 양자화 강건성 차이를 비교한 그림."
                }
                if include_cross_domain
                else {}
            ),
        },
    )
    write_robustness_latex_table(robustness, tables_dir / "robustness_table.tex")
    write_sample_efficiency_latex_table(
        sample_by_cond, tables_dir / "sample_efficiency_table.tex"
    )


def _git_sha() -> Optional[str]:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return completed.stdout.strip()
    return None


def _version_or_none(module_name: str) -> Optional[str]:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def write_run_manifest(
    output_dir: Path,
    plan: list[ExperimentCondition],
    seed: int,
    started_at: datetime,
    stage: str = "unknown",
    include_cross_domain: bool = False,
    force: bool = False,
    evaluation_mode: str = "paper",
    conditions: Optional[list[dict]] = None,
) -> Path:
    """Write run metadata and condition entries."""
    output_dir = Path(output_dir)
    gpu_name = None
    gpu_count = 0
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
    if conditions is None:
        conditions = _LAST_EVALUATE_ENTRIES or [asdict(cond) for cond in plan]

    manifest = {
        "schema_version": 1,
        "timestamp": started_at.isoformat(timespec="seconds"),
        "git_commit_sha": _git_sha(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "bnb_version": _version_or_none("bitsandbytes"),
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "seed": seed,
        "stage": stage,
        "include_cross_domain": include_cross_domain,
        "force": force,
        "evaluation_mode": evaluation_mode,
        "planned_conditions": [asdict(cond) for cond in plan],
        "conditions": conditions,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_manifest.json"
    atomic_write_json(path, manifest)
    return path


def _set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    transformers.set_seed(seed)
    torch.manual_seed(seed)


def _bitsandbytes_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401

        return True
    except Exception:
        return False


def _validate_cross_domain_inputs() -> None:
    missing = [
        path
        for path in (
            REPO_ROOT / "data" / "eval" / "heldout" / "hotel.json",
            REPO_ROOT / "data" / "eval" / "interaction" / "hotel.jsonl",
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing cross-domain data: " + ", ".join(str(p) for p in missing)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quantization efficiency pipeline")
    parser.add_argument(
        "--stage", choices=["evaluate", "analyze", "visualize", "all"], default="all"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include-cross-domain", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bayesian-model", default="models/gemma2-9b-bayesian")
    parser.add_argument("--oracle-model", default="models/gemma2-9b-oracle")
    parser.add_argument("--output-dir", default="results/quant_efficiency")
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument(
        "--evaluation-mode",
        choices=["paper", "interaction"],
        default="paper",
    )
    parser.add_argument("--heldout-batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _set_seed(args.seed)
    if args.include_cross_domain:
        _validate_cross_domain_inputs()

    output_dir = Path(args.output_dir)
    domains = ["flight"] + (["hotel"] if args.include_cross_domain else [])
    model_paths = {
        "bayesian": args.bayesian_model,
        "oracle": args.oracle_model,
    }
    plan = plan_conditions(
        teachings=list(VALID_TEACHINGS),
        quants=list(VALID_QUANTS),
        domains=domains,
        model_paths=model_paths,
    )
    if not _bitsandbytes_available():
        print("WARNING: bitsandbytes unavailable; skipping int8/int4 conditions")
        plan = [cond for cond in plan if cond.quantization == "bf16"]

    started_at = datetime.now()
    result_paths: list[Path] = []
    if args.stage in {"evaluate", "all"}:
        result_paths = run_evaluate_stage(
            plan=plan,
            output_dir=output_dir,
            seed=args.seed,
            force=args.force,
            max_users=args.max_users,
            evaluation_mode=args.evaluation_mode,
            heldout_batch_size=args.heldout_batch_size,
        )
        write_run_manifest(
            output_dir,
            plan,
            args.seed,
            started_at,
            stage="evaluate",
            include_cross_domain=args.include_cross_domain,
            force=args.force,
            evaluation_mode=args.evaluation_mode,
        )

    if args.stage in {"analyze", "all"}:
        run_analyze_stage(result_paths, output_dir)

    if args.stage in {"visualize", "all"}:
        run_visualize_stage(output_dir, include_cross_domain=args.include_cross_domain)

    if args.stage != "evaluate":
        write_run_manifest(
            output_dir,
            plan,
            args.seed,
            started_at,
            stage=args.stage,
            include_cross_domain=args.include_cross_domain,
            force=args.force,
            evaluation_mode=args.evaluation_mode,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
