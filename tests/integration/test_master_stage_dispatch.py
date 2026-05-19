"""Integration tests for master evaluate-stage dispatch."""

import json
from types import SimpleNamespace

from scripts import run_quant_efficiency as rq


def _fake_result(path, teaching, quant):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model_path": "models/x",
                "model_slug": "x",
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
                },
                "per_user": [],
            }
        )
    )


def test_evaluate_stage_reuses_two_bf16_baselines_and_runs_four_conditions(
    tmp_path, monkeypatch
):
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
        _fake_result(
            legacy / rq.format_result_filename(slug, teaching, "bf16", "flight"),
            teaching,
            "bf16",
        )

    calls = []

    def fake_run(cmd, cwd=None, check=False):
        calls.append(cmd)
        target = cmd[cmd.index("--output") + 1]
        teaching = cmd[cmd.index("--teaching") + 1]
        quant = cmd[cmd.index("--quantization") + 1]
        _fake_result(output / target.split("/")[-1], teaching, quant)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(rq.subprocess, "run", fake_run)

    result_paths = rq.run_evaluate_stage(
        plan,
        output,
        seed=42,
        force=False,
        max_users=None,
        legacy_results_dir=legacy,
        evaluation_mode="interaction",
    )

    assert len(calls) == 4
    assert len(result_paths) == 6
    statuses = [entry["status"] for entry in rq._LAST_EVALUATE_ENTRIES]
    assert statuses.count("reused") == 2
    assert statuses.count("produced") == 4
