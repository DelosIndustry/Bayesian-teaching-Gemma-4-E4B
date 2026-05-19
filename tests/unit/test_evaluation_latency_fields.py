"""Feature: quantization-and-sample-efficiency, latency field tests."""

import json

from src.evaluation import InteractiveEvaluator, RoundResult


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class _FakeInference:
    tokenizer = _FakeTokenizer()

    def generate(self, messages):
        return "The best option is Flight 1."


class _FakeInferenceFlight2:
    tokenizer = _FakeTokenizer()

    def generate(self, messages):
        return "The best option is Flight 2."


def _option(price: int) -> str:
    return (
        "Flight 1: departure time: 10:00 AM, duration: 1 hr 0 min, "
        f"number of stops: 0, price: ${price}"
    )


def test_round_result_latency_fields_are_json_serializable():
    result = RoundResult(
        round_idx=0,
        response="The best option is Flight 1.",
        prediction_0idx=0,
        ground_truth_0idx=0,
        correct=True,
        parse_ok=True,
        generation_seconds=0.01,
        response_tokens=6,
    )

    payload = json.loads(json.dumps(result.__dict__))
    assert payload["generation_seconds"] == 0.01
    assert payload["response_tokens"] == 6


def test_evaluate_all_summary_includes_latency(tmp_path):
    interaction_path = tmp_path / "interaction.jsonl"
    heldout_path = tmp_path / "heldout.json"

    rounds = [
        {
            "options": [
                _option(100),
                _option(200).replace("Flight 1", "Flight 2"),
                _option(300).replace("Flight 1", "Flight 3"),
            ],
            "user_idx": 0,
        },
        {
            "options": [
                _option(100),
                _option(200).replace("Flight 1", "Flight 2"),
                _option(300).replace("Flight 1", "Flight 3"),
            ],
            "user_idx": 0,
        },
    ]
    interaction_path.write_text(
        json.dumps({"idx": 0, "reward_fn": [0.0, 0.0, 0.0, -1.0], "rounds": rounds})
        + "\n"
    )
    heldout_path.write_text(json.dumps({"user_idxs": [{"idxs": [0]}]}))

    evaluator = InteractiveEvaluator(
        inference=_FakeInference(),
        interaction_path=str(interaction_path),
        heldout_path=str(heldout_path),
        num_rounds=2,
        evaluation_mode="interaction",
    )
    results = evaluator.evaluate_all()
    summary = results["summary"]

    assert "latency" in summary
    assert summary["latency"]["mean_seconds_per_round"] >= 0
    assert summary["latency"]["mean_seconds_per_user"] >= 0
    assert summary["latency"]["mean_tokens_per_round"] > 0
    assert len(summary["latency"]["per_round_seconds"]) == 2
    json.dumps(results)


def test_interaction_ground_truth_uses_round_user_idx(tmp_path):
    interaction_path = tmp_path / "interaction.jsonl"
    heldout_path = tmp_path / "heldout.json"

    options = [
        _option(100),
        _option(200).replace("Flight 1", "Flight 2"),
        _option(300).replace("Flight 1", "Flight 3"),
    ]
    interaction_path.write_text(
        json.dumps(
            {
                "idx": 0,
                # Raw feature scoring would prefer Flight 1, so this catches
                # accidental recomputation instead of using the stored label.
                "reward_fn": [0.0, 0.0, 0.0, -1.0],
                "rounds": [{"options": options, "user_idx": 1}],
            }
        )
        + "\n"
    )
    heldout_path.write_text(json.dumps({"user_idxs": [{"idxs": [0]}]}))

    evaluator = InteractiveEvaluator(
        inference=_FakeInferenceFlight2(),
        interaction_path=str(interaction_path),
        heldout_path=str(heldout_path),
        num_rounds=1,
        evaluation_mode="interaction",
    )
    results = evaluator.evaluate_all()
    round_result = results["per_user"][0]["rounds"][0]

    assert round_result["ground_truth_0idx"] == 1
    assert round_result["prediction_0idx"] == 1
    assert round_result["correct"] is True
    assert results["summary"]["round_accuracies"] == [1.0]


def test_paper_mode_scores_heldout_sets_after_each_round(tmp_path):
    interaction_path = tmp_path / "interaction.jsonl"
    heldout_path = tmp_path / "heldout.json"

    options = [
        _option(100),
        _option(200).replace("Flight 1", "Flight 2"),
        _option(300).replace("Flight 1", "Flight 3"),
    ]
    interaction_path.write_text(
        json.dumps(
            {
                "idx": 0,
                "reward_fn": [0.0, 0.0, 0.0, -1.0],
                "rounds": [{"options": options, "user_idx": 0}],
            }
        )
        + "\n"
    )
    heldout_path.write_text(
        json.dumps(
            {
                "all_options": [options, options],
                "user_idxs": [{"idxs": [0, 1]}],
            }
        )
    )

    evaluator = InteractiveEvaluator(
        inference=_FakeInference(),
        interaction_path=str(interaction_path),
        heldout_path=str(heldout_path),
        num_rounds=1,
        evaluation_mode="paper",
        heldout_batch_size=2,
    )
    results = evaluator.evaluate_all()
    summary = results["summary"]
    round_result = results["per_user"][0]["rounds"][0]

    assert summary["evaluation_mode"] == "paper"
    assert summary["heldout_sets_per_round"] == 2
    assert summary["round_accuracies"] == [0.5]
    assert round_result["heldout_correct"] == 1
    assert round_result["heldout_total"] == 2
    assert round_result["accuracy"] == 0.5
