"""Interactive evaluation for Bayesian teaching on the flight-selection task.

Paper-style evaluation: each user has N rounds (typically 5). In each round,
the teacher (model under test) sees the full conversation history and the
current round's 3 flight options, and must select the best flight for the
(fixed but unknown to the model) user. After each round, the environment
provides feedback noting which option was incorrect or correct.

Data layout (flight domain, same for all feature counts):
    eval/interaction/flight{_Nfeatures}.jsonl
        one line per user, with fields:
            idx: int, user id (0..U-1)
            reward_fn: list[float], per-feature user preference weights (for info)
            rounds: list of {options: [str, str, str], user_idx: int}
                user_idx is the 0-indexed option chosen by the simulated user
    eval/heldout/flight{_Nfeatures}.json
        {
          features: [...],
          all_options: list[list[str]]  # option sets, indexed by user_idx
          user_idxs: list[{reward_fn, idxs: list[int]}]
              idxs[k] is the ground-truth answer (0/1/2) for option set k,
              for this user.
        }

Key fact: interaction and heldout data encode labels differently. For
interaction rounds, rounds[r].user_idx is already the ground-truth answer.
For heldout option set k, heldout.user_idxs[u].idxs[k] is the answer.

The feedback scheme follows the training data (bayesian.jsonl):
    - First user turn: the selection prompt.
    - Assistant turn: model response (parsed to extract Flight N).
    - Subsequent user turn: "Your option Flight X is incorrect. ..." or
      positive acknowledgement if correct.
    - Repeat for the next round's options.

We mimic this exactly so the model's learned chat pattern is reproduced.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from src.inference import GemmaInference

logger = logging.getLogger(__name__)


# Matches a Flight selection in the assistant's response. We prefer a tight
# pattern near "option is Flight N" but fall back to any "Flight N" mention.
_CHOICE_STRICT = re.compile(
    r"option\s+is\s+(?:Flight|Hotel)\s+(\d+)", re.IGNORECASE
)
_CHOICE_LOOSE = re.compile(r"(?:Flight|Hotel)\s+(\d+)", re.IGNORECASE)


def parse_prediction(response: str, num_options: int = 3) -> Optional[int]:
    """Extract a 0-indexed flight selection from the model response.

    Returns None if no valid Flight number is found or the number is
    outside [1, num_options].
    """
    m = _CHOICE_STRICT.search(response) or _CHOICE_LOOSE.search(response)
    if not m:
        return None
    n = int(m.group(1))
    if 1 <= n <= num_options:
        return n - 1
    return None


# Initial selection prompt used in training data. We reuse the literal string
# from bayesian.jsonl so the model sees the exact distribution it was fit on.
def _initial_prompt_prefix(item_label: str = "Flight") -> str:
    item_lower = item_label.lower()
    item_plural = f"{item_lower}s"
    return (
        f"Help me select the best {item_plural} for my trips. I have specific"
        f" preferences for what I like and dislike in a {item_lower}, and these"
        " preferences remain the same. You need to figure out my preferences"
        f" and select the best {item_plural} for me. Use your best judgment if"
        " you are unsure. Do not say you need more information.\n\n"
        f"Which {item_lower} is the best option?\n"
    )


def _format_options(options: list[str]) -> str:
    """Format 3 flight options as the training data does.

    Each option in the eval data is a single string like
        "Flight 1: departure time: ..., duration: ..., ..."
    Training data renders these without the leading "Flight N:" because
    "Flight N" is used as a section header, then attributes on the next line.
    We split at the first ":" and reformat to match training exactly.
    """
    lines = []
    for opt in options:
        if ":" in opt:
            header, attrs = opt.split(":", 1)
            lines.append(f"{header.strip()}:\n{attrs.strip()}")
        else:
            lines.append(opt)
    return "\n".join(lines)


def build_initial_user_message(options: list[str], item_label: str = "Flight") -> str:
    return _initial_prompt_prefix(item_label) + "\n" + _format_options(options)


def build_next_user_message(
    prev_pred_1idx: int,
    correct: bool,
    correct_1idx: int,
    next_options: list[str],
    teaching: str = "bayesian",
    item_label: str = "Flight",
) -> str:
    """Build the user feedback + next round prompt.

    Mirrors the phrasing of training-data conversations exactly. The two
    teaching modes have different feedback distributions in their training
    data, so we must reproduce the matching one at eval time:

      - bayesian.jsonl: incorrect rounds include "I prefer Flight Y" (the
        ground-truth answer is revealed). Correct rounds just say "correct".
      - oracle.jsonl: all rounds show the correct selection; feedback is
        always "Your option Flight X is correct." (the assistant was trained
        to always pick the right answer, so "incorrect" feedback is OOD).

    Using the wrong feedback form at eval cripples the model's learning-
    effect signal because it's OOD relative to training.
    """
    if correct:
        feedback = f"Your option {item_label} {prev_pred_1idx} is correct."
    else:
        if teaching == "bayesian":
            feedback = (
                f"Your option {item_label} {prev_pred_1idx} is incorrect. "
                f"I prefer {item_label} {correct_1idx}."
            )
        elif teaching == "oracle":
            # Oracle was never trained on "incorrect" feedback at all. The
            # closest in-distribution signal is to treat the correct answer
            # as the reference ("I prefer Flight Y.") without the "incorrect"
            # preamble. This keeps eval faithful to training distribution.
            feedback = f"I prefer {item_label} {correct_1idx}."
        else:
            # Base model or unknown: use the neutral Bayesian-style hint.
            feedback = (
                f"Your option {item_label} {prev_pred_1idx} is incorrect. "
                f"I prefer {item_label} {correct_1idx}."
            )
    body = (
        f"{feedback}\n\n"
        f"Which {item_label.lower()} is the best option?\n\n"
        + _format_options(next_options)
    )
    return body


@dataclass
class RoundResult:
    round_idx: int
    response: str
    prediction_0idx: Optional[int]
    ground_truth_0idx: int
    correct: bool
    parse_ok: bool
    generation_seconds: float
    response_tokens: int
    accuracy: Optional[float] = None
    heldout_correct: Optional[int] = None
    heldout_total: Optional[int] = None
    heldout_parse_failures: Optional[int] = None


@dataclass
class UserResult:
    user_idx: int
    rounds: list[RoundResult] = field(default_factory=list)

    @property
    def parse_failures(self) -> int:
        return sum(1 for r in self.rounds if not r.parse_ok)


class InteractiveEvaluator:
    """Run paper-style 5-round evaluation over all users of a domain."""

    def __init__(
        self,
        inference: GemmaInference,
        interaction_path: str,
        heldout_path: str,
        num_rounds: int = 5,
        teaching: str = "bayesian",
        domain: str = "flight",
        evaluation_mode: str = "paper",
        heldout_batch_size: int = 8,
    ):
        self.inf = inference
        self.num_rounds = num_rounds
        self.teaching = teaching
        self.domain = domain
        self.item_label = "Hotel" if domain == "hotel" else "Flight"
        if evaluation_mode not in {"paper", "interaction"}:
            raise ValueError(f"unknown evaluation_mode: {evaluation_mode}")
        self.evaluation_mode = evaluation_mode
        self.heldout_batch_size = heldout_batch_size

        logger.info("Loading interaction data: %s", interaction_path)
        self.interactions: list[dict] = []
        with open(interaction_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.interactions.append(json.loads(line))

        logger.info("Loading heldout data: %s", heldout_path)
        with open(heldout_path) as f:
            self.heldout = json.load(f)
        # Sanity: heldout.user_idxs must be at least as long as interactions.
        if len(self.heldout["user_idxs"]) < len(self.interactions):
            raise ValueError(
                "heldout.user_idxs shorter than interactions: "
                f"{len(self.heldout['user_idxs'])} < {len(self.interactions)}"
            )

    def _ground_truth(self, user_id: int, option_set_idx: int) -> int:
        """0-indexed heldout answer for user u in option set k."""
        return int(self.heldout["user_idxs"][user_id]["idxs"][option_set_idx])

    @staticmethod
    def _parse_flight_features(opt_str: str) -> list[float]:
        """Parse numeric features from a flight option string.
        
        Returns [departure_minutes, duration_minutes, num_stops, price].
        """
        # Departure time -> minutes from midnight
        dt_match = re.search(r'departure time: (\d+):(\d+) (AM|PM)', opt_str)
        if not dt_match:
            raise ValueError(f"Cannot parse departure time from: {opt_str}")
        h, m, ampm = int(dt_match.group(1)), int(dt_match.group(2)), dt_match.group(3)
        if ampm == 'PM' and h != 12:
            h += 12
        if ampm == 'AM' and h == 12:
            h = 0
        dep_minutes = h * 60 + m

        # Duration -> minutes (handles "X hr Y min", "X hr", "Y min")
        dur_match = re.search(r'duration: (?:(\d+) hr)?\s*(?:(\d+) min)?', opt_str)
        if not dur_match:
            raise ValueError(f"Cannot parse duration from: {opt_str}")
        dur_h = int(dur_match.group(1)) if dur_match.group(1) else 0
        dur_m = int(dur_match.group(2)) if dur_match.group(2) else 0
        dur_minutes = dur_h * 60 + dur_m

        # Number of stops
        stops_match = re.search(r'number of stops: (\d+)', opt_str)
        if not stops_match:
            raise ValueError(f"Cannot parse stops from: {opt_str}")
        stops = int(stops_match.group(1))

        # Price
        price_match = re.search(r'price: \$(\d+)', opt_str)
        if not price_match:
            raise ValueError(f"Cannot parse price from: {opt_str}")
        price = int(price_match.group(1))

        return [float(dep_minutes), float(dur_minutes), float(stops), float(price)]

    def _ground_truth_from_options(
        self, reward_fn: list[float], options: list[str]
    ) -> int:
        """Compute ground-truth answer by scoring each option with reward_fn.
        
        The user's reward function is a weight vector over features. The
        preferred flight is the one with the highest dot product:
            score(flight) = sum(w_i * feature_i)
        """
        scores = []
        for opt in options:
            feats = InteractiveEvaluator._parse_flight_features(opt)
            score = sum(w * f for w, f in zip(reward_fn, feats))
            scores.append(score)
        return int(scores.index(max(scores)))

    @staticmethod
    def _ground_truth_from_feature_vectors(
        reward_fn: list[float], option_features: list[list[float]]
    ) -> int:
        """Compute ground truth from normalized feature vectors in eval data."""
        scores = [
            sum(w * f for w, f in zip(reward_fn, feats))
            for feats in option_features
        ]
        return int(scores.index(max(scores)))

    def _count_response_tokens(self, response: str) -> int:
        """Count response tokens with the model tokenizer when available."""
        tokenizer = getattr(self.inf, "tokenizer", None)
        if tokenizer is None:
            return 0
        return len(tokenizer.encode(response, add_special_tokens=False))

    def _heldout_gt(self, user_idx: int, option_set_idx: int) -> int:
        """Ground truth for one heldout option set from the dataset."""
        return int(self.heldout["user_idxs"][user_idx]["idxs"][option_set_idx])

    def _interaction_gt(self, user_rec: dict, round_idx: int, round_rec: dict) -> int:
        """Ground truth for one feedback round.

        In the interaction JSONL files, ``rounds[*].user_idx`` is the selected
        option index, not an index into the heldout option-set pool. Use it
        directly so the feedback shown to the model matches the simulated user.
        """
        if "user_idx" in round_rec:
            return int(round_rec["user_idx"])
        if "score" in round_rec:
            scores = round_rec["score"]
            return int(max(range(len(scores)), key=lambda i: scores[i]))
        reward_fn = user_rec["reward_fn"]
        if "rounds_numpy" in user_rec:
            return self._ground_truth_from_feature_vectors(
                reward_fn, user_rec["rounds_numpy"][round_idx]
            )
        return self._ground_truth_from_options(reward_fn, round_rec["options"])

    def _generate_batch(self, batch_messages: list[list[dict]]) -> list[str]:
        """Generate a batch when the inference wrapper supports it."""
        if hasattr(self.inf, "generate_batch"):
            return self.inf.generate_batch(batch_messages)
        return [self.inf.generate(messages) for messages in batch_messages]

    def _evaluate_heldout_after_round(
        self,
        history: list[dict],
        interaction_pred_1idx: int,
        interaction_correct: bool,
        interaction_gt_1idx: int,
        user_idx: int,
    ) -> tuple[int, int, int, float, int]:
        """Evaluate 100 heldout option sets after one interaction round."""
        all_options = self.heldout["all_options"]
        correct = 0
        parse_failures = 0
        total_tokens = 0
        total_seconds = 0.0

        for start in range(0, len(all_options), self.heldout_batch_size):
            option_batch = all_options[start : start + self.heldout_batch_size]
            batch_messages = []
            for options in option_batch:
                batch_messages.append(
                    history
                    + [
                        {
                            "role": "user",
                            "content": build_next_user_message(
                                interaction_pred_1idx,
                                interaction_correct,
                                interaction_gt_1idx,
                                options,
                                teaching=self.teaching,
                                item_label=self.item_label,
                            ),
                        }
                    ]
                )

            gen_start = time.perf_counter()
            responses = self._generate_batch(batch_messages)
            total_seconds += time.perf_counter() - gen_start

            for offset, response in enumerate(responses):
                option_set_idx = start + offset
                pred_0idx = parse_prediction(
                    response, num_options=len(all_options[option_set_idx])
                )
                total_tokens += self._count_response_tokens(response)
                if pred_0idx is None:
                    parse_failures += 1
                    continue
                if pred_0idx == self._heldout_gt(user_idx, option_set_idx):
                    correct += 1

        return correct, len(all_options), parse_failures, total_seconds, total_tokens

    def evaluate_user(self, user_rec: dict) -> UserResult:
        """Run 5-round evaluation for a single user."""
        u = user_rec["idx"]
        rounds = user_rec["rounds"][: self.num_rounds]
        reward_fn = user_rec["reward_fn"]
        messages: list[dict] = []
        result = UserResult(user_idx=u)

        for r, rd in enumerate(rounds):
            options = rd["options"]
            gt_0idx = self._interaction_gt(user_rec, r, rd)

            if r == 0:
                messages.append(
                    {
                        "role": "user",
                        "content": build_initial_user_message(
                            options, item_label=self.item_label
                        ),
                    }
                )
            else:
                prev = result.rounds[-1]
                prev_pred_1idx = (
                    (prev.prediction_0idx + 1) if prev.parse_ok else 0
                )
                # Ground truth for the PREVIOUS round, so feedback can reveal
                # which Flight was actually preferred. This matches the
                # training-data feedback distribution exactly.
                prev_gt_1idx = prev.ground_truth_0idx + 1
                messages.append(
                    {
                        "role": "user",
                        "content": build_next_user_message(
                            prev_pred_1idx,
                            prev.correct,
                            prev_gt_1idx,
                            options,
                            teaching=self.teaching,
                            item_label=self.item_label,
                        ),
                    }
                )

            gen_start = time.perf_counter()
            response = self.inf.generate(messages)
            generation_seconds = time.perf_counter() - gen_start
            response_tokens = self._count_response_tokens(response)
            pred_0idx = parse_prediction(response, num_options=len(options))
            parse_ok = pred_0idx is not None
            correct = parse_ok and pred_0idx == gt_0idx

            heldout_correct = None
            heldout_total = None
            heldout_parse_failures = None
            heldout_accuracy = None
            if self.evaluation_mode == "paper":
                history_after_interaction = messages + [
                    {"role": "assistant", "content": response}
                ]
                interaction_pred_1idx = (pred_0idx + 1) if parse_ok else 0
                (
                    heldout_correct,
                    heldout_total,
                    heldout_parse_failures,
                    heldout_seconds,
                    heldout_tokens,
                ) = self._evaluate_heldout_after_round(
                    history_after_interaction,
                    interaction_pred_1idx,
                    correct,
                    gt_0idx + 1,
                    u,
                )
                heldout_accuracy = heldout_correct / heldout_total if heldout_total else 0.0
                generation_seconds += heldout_seconds
                response_tokens += heldout_tokens

            result.rounds.append(
                RoundResult(
                    round_idx=r,
                    response=response,
                    prediction_0idx=pred_0idx,
                    ground_truth_0idx=gt_0idx,
                    correct=correct,
                    parse_ok=parse_ok,
                    generation_seconds=generation_seconds,
                    response_tokens=response_tokens,
                    accuracy=heldout_accuracy,
                    heldout_correct=heldout_correct,
                    heldout_total=heldout_total,
                    heldout_parse_failures=heldout_parse_failures,
                )
            )
            # Keep the assistant turn in history so the model sees its own prior
            # answer on the next turn (mirrors training data exactly).
            messages.append({"role": "assistant", "content": response})

        return result

    def evaluate_all(
        self,
        max_users: Optional[int] = None,
        progress_every: int = 25,
    ) -> dict:
        users = self.interactions if max_users is None else self.interactions[:max_users]
        n = len(users)
        logger.info("Evaluating %d users, %d rounds each", n, self.num_rounds)

        per_user: list[UserResult] = []
        t0 = time.time()
        for i, user_rec in enumerate(users, 1):
            per_user.append(self.evaluate_user(user_rec))
            if i % progress_every == 0 or i == n:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (n - i) / rate if rate > 0 else 0.0
                logger.info(
                    "  %d/%d users (%.1f users/min, ETA %.1f min)",
                    i, n, rate * 60, eta / 60,
                )

        # Aggregate per-round accuracy across all users.
        round_correct = [0.0] * self.num_rounds
        round_total = [0.0] * self.num_rounds
        round_parse_fail = [0] * self.num_rounds
        round_seconds = [0.0] * self.num_rounds
        round_tokens = [0] * self.num_rounds
        for u in per_user:
            for r in u.rounds:
                if r.heldout_total is not None and r.heldout_correct is not None:
                    round_total[r.round_idx] += r.heldout_total
                    round_correct[r.round_idx] += r.heldout_correct
                    round_parse_fail[r.round_idx] += r.heldout_parse_failures or 0
                else:
                    round_total[r.round_idx] += 1
                    if r.correct:
                        round_correct[r.round_idx] += 1
                    if not r.parse_ok:
                        round_parse_fail[r.round_idx] += 1
                round_seconds[r.round_idx] += r.generation_seconds
                round_tokens[r.round_idx] += r.response_tokens

        round_acc = [
            (c / t) if t else 0.0 for c, t in zip(round_correct, round_total)
        ]
        total_preds = sum(round_total)
        total_fail = sum(round_parse_fail)
        total_seconds = sum(round_seconds)
        total_tokens = sum(round_tokens)
        parse_fail_rate = total_fail / total_preds if total_preds else 0.0
        elapsed_seconds_total = time.time() - t0

        latency = {
            "mean_seconds_per_round": (
                total_seconds / total_preds if total_preds else 0.0
            ),
            "mean_seconds_per_user": total_seconds / n if n else 0.0,
            "mean_tokens_per_round": total_tokens / total_preds if total_preds else 0.0,
            "per_round_seconds": [
                (s / t) if t else 0.0 for s, t in zip(round_seconds, round_total)
            ],
        }

        summary = {
            "num_users": n,
            "num_rounds": self.num_rounds,
            "evaluation_mode": self.evaluation_mode,
            "heldout_sets_per_round": len(self.heldout.get("all_options", []))
            if self.evaluation_mode == "paper"
            else None,
            "round_accuracies": round_acc,
            "learning_effect": (round_acc[-1] - round_acc[0]) if round_acc else 0.0,
            "parse_failure_rate": parse_fail_rate,
            "per_round_parse_failures": round_parse_fail,
            "elapsed_seconds": elapsed_seconds_total,
            "elapsed_seconds_total": elapsed_seconds_total,
            "elapsed_seconds_per_user": elapsed_seconds_total / n if n else 0.0,
            "latency": latency,
        }
        if parse_fail_rate > 0.05:
            logger.warning(
                "Parse failure rate %.2f%% exceeds 5%% threshold — inspect responses.",
                parse_fail_rate * 100,
            )
        logger.info(
            "Round accuracies: %s | learning_effect=%.3f | parse_fail=%.2f%%",
            ["%.3f" % a for a in round_acc],
            summary["learning_effect"],
            parse_fail_rate * 100,
        )
        return {
            "summary": summary,
            "per_user": [_user_to_dict(u) for u in per_user],
        }


def _user_to_dict(u: UserResult) -> dict:
    return {
        "user_idx": u.user_idx,
        "rounds": [asdict(r) for r in u.rounds],
    }


REQUIRED_RESULT_KEYS = {
    "model_path",
    "model_slug",
    "teaching",
    "quantization",
    "domain",
    "num_rounds",
    "max_users",
    "timestamp",
    "memory_mib",
    "elapsed_seconds_total",
    "summary",
    "per_user",
}


def atomic_write_json(path: str | Path, payload: dict) -> None:
    """Write JSON via a temporary file and atomic replacement."""
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


def save_results(results: dict, meta: dict, output_path: str) -> None:
    """Save results as {meta..., ...results} to output_path (JSON)."""
    out = {**meta, **results}
    missing = REQUIRED_RESULT_KEYS - set(out)
    if missing:
        raise ValueError(f"Result payload missing required keys: {sorted(missing)}")
    atomic_write_json(output_path, out)
    logger.info("Saved results to %s", output_path)
