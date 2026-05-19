"""Skip policy idempotency tests."""

import json
from types import SimpleNamespace

from scripts import run_quant_efficiency as rq


def _write_result(path, teaching, quant, marker):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model_path": "models/x",
                "model_slug": path.name.split("_")[0],
                "teaching": teaching,
                "quantization": quant,
                "domain": "flight",
                "num_rounds": 5,
                "max_users": None,
                "timestamp": "now",
                "memory_mib": {},
                "elapsed_seconds_total": 1.0,
                "summary": {
                    "round_accuracies": [0.1, 0.2, 0.3, 0.4, 0.5],
                    "parse_failure_rate": 0.0,
                    "marker": marker,
                },
                "per_user": [],
            }
        )
    )


def test_skip_policy_idempotency(tmp_path, monkeypatch):
    """Feature: quantization-and-sample-efficiency, Property 5: Skip policy idempotency"""
    legacy = tmp_path / "legacy"
    output = tmp_path / "out"
    model_paths = {
        "bayesian": "models/gemma2-9b-bayesian",
        "oracle": "models/gemma2-9b-oracle",
    }
    plan = rq.plan_conditions(
        ["bayesian", "oracle"], ["bf16", "int8", "int4"], ["flight"], model_paths
    )
    for teaching, model_path in model_paths.items():
        slug = model_path.split("/")[-1]
        _write_result(
            legacy / rq.format_result_filename(slug, teaching, "bf16", "flight"),
            teaching,
            "bf16",
            marker=0,
        )

    counter = {"value": 0}
    calls = []

    def fake_run(cmd, cwd=None, check=False):
        calls.append(cmd)
        counter["value"] += 1
        target = cmd[cmd.index("--output") + 1]
        teaching = cmd[cmd.index("--teaching") + 1]
        quant = cmd[cmd.index("--quantization") + 1]
        _write_result(output / target.split("/")[-1], teaching, quant, counter["value"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(rq.subprocess, "run", fake_run)

    rq.run_evaluate_stage(
        plan,
        output,
        seed=42,
        force=False,
        max_users=None,
        legacy_results_dir=legacy,
        evaluation_mode="interaction",
    )
    first_files = {
        path.name: path.read_text() for path in sorted(output.glob("*.json"))
    }
    assert len(calls) == 4

    rq.run_evaluate_stage(
        plan,
        output,
        seed=42,
        force=False,
        max_users=None,
        legacy_results_dir=legacy,
        evaluation_mode="interaction",
    )
    second_files = {
        path.name: path.read_text() for path in sorted(output.glob("*.json"))
    }
    assert len(calls) == 4
    assert first_files == second_files
    assert sum(1 for e in rq._LAST_EVALUATE_ENTRIES if e["status"] == "produced") == 0

    rq.run_evaluate_stage(
        plan,
        output,
        seed=42,
        force=True,
        max_users=None,
        legacy_results_dir=legacy,
        evaluation_mode="interaction",
    )
    assert len(calls) == 8
