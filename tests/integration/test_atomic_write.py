"""Feature: quantization-and-sample-efficiency, atomic result write tests."""

import json

import pytest

from src import evaluation


def test_atomic_write_removes_tmp_and_leaves_final_absent_on_interrupt(
    tmp_path, monkeypatch
):
    target = tmp_path / "result.json"

    def interrupting_dump(payload, file_obj, indent=None):
        file_obj.write('{"partial":')
        raise KeyboardInterrupt

    monkeypatch.setattr(evaluation.json, "dump", interrupting_dump)

    with pytest.raises(KeyboardInterrupt):
        evaluation.atomic_write_json(target, {"large": list(range(1000))})

    assert not target.exists()
    assert not target.with_suffix(target.suffix + ".tmp").exists()


def test_atomic_write_success_path_writes_valid_json(tmp_path):
    target = tmp_path / "result.json"
    payload = {"summary": {"round_accuracies": [0.1, 0.2]}}

    evaluation.atomic_write_json(target, payload)

    assert json.loads(target.read_text()) == payload
