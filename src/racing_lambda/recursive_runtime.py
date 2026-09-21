"""Permanent, result-blind runtime loop for racing recursive self-improvement.

This speeds up ``RacingRecursiveImprovementGate`` search without weakening any
of its safety invariants:

* Several ``PARALLEL_CANDIDATES`` mutations of the same mode (``full`` or
  ``simple``) are explored concurrently against the same baseline and the
  same races, so one doomed candidate no longer has to occupy the only
  search slot for its full evaluation window before a different mutation
  gets a turn.
* A candidate that is already statistically conclusively worse than baseline
  (a Wald SPRT ``REJECT``) is abandoned as soon as that is known -- from
  ``min_early_rejection_races`` races -- instead of always burning the full
  ``min_future_races`` window on a doomed candidate.

Unlike the sibling market-prediction runtime, this module never writes into
``active_model`` on its own: this repository's autonomous scope excludes
parameter promotion, so ``PROMOTION_PROPOSED`` candidates only ever produce a
frozen proposal for human review. The single supported path from a proposal
into ``active_model`` is :func:`apply_human_approved_promotion`, which
requires a recorded :class:`RacingHumanApproval`.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, Sequence

from .controlled_rsi_validation import ControlledPrediction, ControlledRsiValidationLoop, TrialMetrics
from .recursive_self_improvement import (
    Mode,
    RacingFrozenTrial,
    RacingFutureEvaluation,
    RacingHumanApproval,
    RacingPromotionReport,
    RacingRecursiveImprovementGate,
    RacingRsiCandidate,
    approved_parameters,
    candidate_manifest_digest,
    freeze_promotion_report,
    parameter_manifest_digest,
    sequential_loss_improvement_test,
    trial_manifest_digest,
)
from .simple_leading_signal_v02 import SimpleHorseFeatures, SimpleLeadingSignalLambdaV02, SimpleRaceContext
from .simple_realtime_rsi import SimpleRealtimeRsiSignal


RUNTIME_SCHEMA_VERSION = "racing-recursive-runtime-v1"
PARALLEL_CANDIDATES = 3
DEFAULT_MIN_FUTURE_RACES = 8
DEFAULT_MIN_EARLY_REJECTION_RACES = 5

# Only ``rsi_weight`` is actually implemented by either prediction path today
# (``rsi_candidate_parameters()`` on both models attests only this key), so
# it is the only lever the search mutates even though the allow-list in
# ``recursive_self_improvement.ALLOWED_PARAMETERS`` reserves more names for
# future candidate implementations.
MUTATION_SCHEDULES: Mapping[Mode, tuple[tuple[str, tuple[object, ...]], ...]] = {
    "simple": (("rsi_weight", (0.20, 0.0, 0.10, 0.30, 0.40, 0.50)),),
    "full": (("rsi_weight", (0.25, 0.0, 0.10, 0.35, 0.50, 0.75, 1.00)),),
}


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("recursive RSI timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _source_commit(value: str | None) -> str:
    commit = (value or "0" * 40).lower()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("source commit must be a full Git SHA")
    return commit


def _candidate_from_payload(payload: Mapping[str, object]) -> RacingRsiCandidate:
    return RacingRsiCandidate(
        mode=str(payload["mode"]),
        candidate_id=str(payload["candidate_id"]),
        parent_version=str(payload["parent_version"]),
        generation=int(payload["generation"]),
        created_at=_parse_time(str(payload["created_at"])),
        source_commit=str(payload["source_commit"]),
        parameter_manifest_sha256=str(payload["parameter_manifest_sha256"]),
        parameters=dict(payload["parameters"]),
        parent_report_sha256=(
            str(payload["parent_report_sha256"])
            if payload.get("parent_report_sha256") is not None
            else None
        ),
    )


def _trial_from_payload(payload: Mapping[str, object]) -> RacingFrozenTrial:
    return RacingFrozenTrial(
        mode=str(payload["mode"]),
        candidate_id=str(payload["candidate_id"]),
        race_id=str(payload["race_id"]),
        registered_at=_parse_time(str(payload["registered_at"])),
        prediction_frozen_at=_parse_time(str(payload["prediction_frozen_at"])),
        scheduled_start=_parse_time(str(payload["scheduled_start"])),
        baseline_prediction_sha256=str(payload["baseline_prediction_sha256"]),
        candidate_prediction_sha256=str(payload["candidate_prediction_sha256"]),
        candidate_manifest_sha256=str(payload["candidate_manifest_sha256"]),
        input_sha256=str(payload["input_sha256"]),
    )


def _evaluation_from_payload(payload: Mapping[str, object]) -> RacingFutureEvaluation:
    return RacingFutureEvaluation(
        mode=str(payload["mode"]),
        candidate_id=str(payload["candidate_id"]),
        race_id=str(payload["race_id"]),
        prediction_frozen_at=_parse_time(str(payload["prediction_frozen_at"])),
        scheduled_start=_parse_time(str(payload["scheduled_start"])),
        result_known_at=_parse_time(str(payload["result_known_at"])),
        baseline_prediction_sha256=str(payload["baseline_prediction_sha256"]),
        candidate_prediction_sha256=str(payload["candidate_prediction_sha256"]),
        candidate_manifest_sha256=str(payload["candidate_manifest_sha256"]),
        trial_manifest_sha256=str(payload["trial_manifest_sha256"]),
        baseline_brier_loss=float(payload["baseline_brier_loss"]),
        candidate_brier_loss=float(payload["candidate_brier_loss"]),
        baseline_top3_hits=int(payload["baseline_top3_hits"]),
        candidate_top3_hits=int(payload["candidate_top3_hits"]),
        baseline_recovery_rate=float(payload["baseline_recovery_rate"]),
        candidate_recovery_rate=float(payload["candidate_recovery_rate"]),
        baseline_max_bug_hit=bool(payload["baseline_max_bug_hit"]),
        candidate_max_bug_hit=bool(payload["candidate_max_bug_hit"]),
    )


def _next_parameters(mode: Mode, active: Mapping[str, object], attempt: int) -> dict[str, object]:
    schedule = MUTATION_SCHEDULES[mode]
    search_size = 1
    for _, choices in schedule:
        search_size *= len(choices)
    for offset in range(search_size):
        cursor = (attempt + offset) % search_size
        parameters = deepcopy(dict(active))
        for name, choices in schedule:
            parameters[name] = deepcopy(choices[cursor % len(choices)])
            cursor //= len(choices)
        if parameters != dict(active):
            return parameters
    raise RuntimeError(f"recursive RSI search space contains no alternative configuration for mode={mode}")


def _new_candidate(
    state: Mapping[str, object], source_commit: str, created_at: datetime
) -> RacingRsiCandidate:
    mode = str(state["mode"])
    active = dict(state["active_model"])
    attempt = int(state.get("attempt", 0)) + 1
    generation = int(active["generation"]) + 1
    parameters = _next_parameters(mode, dict(active["parameters"]), attempt)
    return RacingRsiCandidate(
        mode=mode,
        candidate_id=f"{mode}-rsi-g{generation}-a{attempt}",
        parent_version=str(active["model_id"]),
        generation=generation,
        created_at=created_at,
        source_commit=_source_commit(source_commit),
        parameter_manifest_sha256=parameter_manifest_digest(parameters),
        parameters=parameters,
        parent_report_sha256=(str(active["promotion_report_sha256"]) if generation > 1 else None),
    )


def _new_slot(candidate: RacingRsiCandidate, attempt: int) -> dict[str, object]:
    return {
        "attempt": attempt,
        "candidate": candidate.sealed_payload(),
        "candidate_manifest_sha256": candidate_manifest_digest(candidate),
        "trials": [],
        "evaluations": [],
        "pending_trial": None,
    }


def _seed_slot(state: dict[str, object], source_commit: str, created_at: datetime) -> dict[str, object]:
    candidate = _new_candidate(state, source_commit, created_at)
    state["attempt"] = int(state.get("attempt", 0)) + 1
    return _new_slot(candidate, state["attempt"])


def ensure_candidate_slots(
    state: dict[str, object],
    source_commit: str,
    now: datetime | None = None,
    *,
    parallel_candidates: int = PARALLEL_CANDIDATES,
) -> None:
    """Top up the candidate pool to ``parallel_candidates`` concurrent slots."""
    slots = state.setdefault("candidate_slots", [])
    created = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    while len(slots) < parallel_candidates:
        slots.append(_seed_slot(state, source_commit, created - timedelta(microseconds=1)))


def bootstrap_state(
    mode: Mode,
    default_parameters: Mapping[str, object],
    source_commit: str,
    created_at: datetime | None = None,
    *,
    parallel_candidates: int = PARALLEL_CANDIDATES,
) -> dict[str, object]:
    if mode not in ("full", "simple"):
        raise ValueError("mode must be full or simple")
    now = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    state: dict[str, object] = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "mode": mode,
        "active_model": {
            "model_id": f"{mode}-current",
            "generation": 0,
            "parameters": deepcopy(dict(default_parameters)),
            "promotion_report_sha256": None,
        },
        "attempt": 0,
        "candidate_slots": [],
        "completed_candidates": [],
        "previous_state_sha256": None,
        "updated_at": now.isoformat(),
    }
    ensure_candidate_slots(state, source_commit, now, parallel_candidates=parallel_candidates)
    return state


def _file_digest(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def load_state(path: str | Path) -> dict[str, object]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != RUNTIME_SCHEMA_VERSION:
        raise ValueError("unsupported racing recursive RSI runtime state")
    state = dict(raw)
    expected = state.pop("state_sha256", None)
    actual = sha256(_canonical(state)).hexdigest()
    state["state_sha256"] = expected
    if expected != actual:
        raise ValueError("recursive RSI state hash mismatch")
    for slot in state["candidate_slots"]:
        _candidate_from_payload(slot["candidate"])
    return state


def write_state(
    state: Mapping[str, object], path: str | Path, previous_path: str | Path | None = None
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite recursive RSI state: {destination}")
    payload = deepcopy(dict(state))
    payload.pop("state_sha256", None)
    payload["previous_state_sha256"] = (
        _file_digest(previous_path) if previous_path and Path(previous_path).exists() else None
    )
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    payload["state_sha256"] = sha256(_canonical(payload)).hexdigest()
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def register_simple_race_trials(
    state: dict[str, object],
    root: str | Path,
    *,
    context: SimpleRaceContext,
    horses: Sequence[SimpleHorseFeatures],
    baseline_rsi_signals: Sequence[SimpleRealtimeRsiSignal] | None,
    candidate_rsi_signals: Sequence[SimpleRealtimeRsiSignal] | None,
    captured_at: datetime,
    prediction_frozen_at: datetime,
    registered_at: datetime | None = None,
) -> list[ControlledPrediction]:
    """Freeze one race's PRE_RACE trial for every parallel simple-mode slot."""
    if state["mode"] != "simple":
        raise ValueError("register_simple_race_trials requires simple-mode runtime state")
    slots = state.get("candidate_slots", [])
    if not slots:
        raise ValueError("runtime state has no candidate slots")
    for slot in slots:
        if slot.get("pending_trial") is not None:
            raise ValueError("a recursive RSI trial is already awaiting its outcome")

    baseline_model = SimpleLeadingSignalLambdaV02(**dict(state["active_model"]["parameters"]))
    predictions: list[ControlledPrediction] = []
    for slot in slots:
        candidate = _candidate_from_payload(slot["candidate"])
        candidate_model = SimpleLeadingSignalLambdaV02(**dict(candidate.parameters))
        loop = ControlledRsiValidationLoop(root, candidate)
        prediction = loop.run_simple_prediction(
            baseline_model=baseline_model,
            candidate_model=candidate_model,
            context=context,
            horses=horses,
            baseline_rsi_signals=baseline_rsi_signals,
            candidate_rsi_signals=candidate_rsi_signals,
            captured_at=captured_at,
            prediction_frozen_at=prediction_frozen_at,
            registered_at=registered_at,
        )
        trial_payload = {
            **prediction.trial.sealed_payload(),
            "trial_manifest_sha256": trial_manifest_digest(prediction.trial),
        }
        slot.setdefault("trials", []).append(trial_payload)
        slot["pending_trial"] = trial_payload
        predictions.append(prediction)
    return predictions


def settle_simple_race_result(
    state: dict[str, object],
    root: str | Path,
    *,
    result_known_at: datetime,
    metrics_by_slot: Sequence[TrialMetrics],
) -> list[RacingFutureEvaluation]:
    """Settle whichever slots currently have a pending trial for this race."""
    if state["mode"] != "simple":
        raise ValueError("settle_simple_race_result requires simple-mode runtime state")
    slots = state.get("candidate_slots", [])
    if len(metrics_by_slot) != len(slots):
        raise ValueError("metrics_by_slot must have one entry per parallel slot")
    pending_indexes = [index for index, slot in enumerate(slots) if slot.get("pending_trial") is not None]
    if not pending_indexes:
        return []

    evaluations: list[RacingFutureEvaluation] = []
    for index in pending_indexes:
        slot = slots[index]
        trial = _trial_from_payload(slot["pending_trial"])
        candidate = _candidate_from_payload(slot["candidate"])
        loop = ControlledRsiValidationLoop(root, candidate)
        evaluation = loop.settle_result(
            trial, result_known_at=result_known_at, metrics=metrics_by_slot[index]
        )
        slot.setdefault("evaluations", []).append(
            {
                **evaluation.__dict__,
                "prediction_frozen_at": evaluation.prediction_frozen_at.isoformat(),
                "scheduled_start": evaluation.scheduled_start.isoformat(),
                "result_known_at": evaluation.result_known_at.isoformat(),
            }
        )
        slot["pending_trial"] = None
        evaluations.append(evaluation)
    return evaluations


def evaluate_and_rotate_candidates(
    state: dict[str, object],
    root: str | Path,
    source_commit: str,
    now: datetime | None = None,
    *,
    min_future_races: int = DEFAULT_MIN_FUTURE_RACES,
    min_early_rejection_races: int = DEFAULT_MIN_EARLY_REJECTION_RACES,
    false_promotion_rate: float = 0.05,
    false_rejection_rate: float = 0.10,
) -> list[dict[str, object]]:
    """Evaluate every concluded parallel slot and reseed its replacement.

    ``active_model`` is never written here, promoted or otherwise: this
    repository's recursive RSI has no autonomous parameter-promotion
    authority. A ``PROMOTION_PROPOSED`` outcome only freezes a proposal for
    human review; see :func:`apply_human_approved_promotion` for the sole
    supported path into production parameters.
    """
    if not 2 <= min_early_rejection_races < min_future_races:
        raise ValueError("min_early_rejection_races must be at least 2 and below min_future_races")
    mode = str(state["mode"])
    slots = state.get("candidate_slots", [])
    conclusions: list[dict[str, object]] = []
    for index, slot in enumerate(slots):
        evaluations_payload = slot.get("evaluations", [])
        races_so_far = len(evaluations_payload)
        if races_so_far < min_early_rejection_races:
            continue
        evaluations = [_evaluation_from_payload(value) for value in evaluations_payload]

        if races_so_far < min_future_races:
            # A Wald SPRT verdict is valid evidence at any sample size, so a
            # candidate that is *already* statistically conclusively worse
            # than baseline can be abandoned now instead of burning the rest
            # of the min_future_races window on a doomed candidate. Never
            # promotes on a partial window: only an early REJECT
            # short-circuits here.
            early_evidence = sequential_loss_improvement_test(
                [row.baseline_brier_loss for row in evaluations],
                [row.candidate_brier_loss for row in evaluations],
                alpha=false_promotion_rate,
                beta=false_rejection_rate,
            )
            if early_evidence.decision != "REJECT":
                continue
            candidate = _candidate_from_payload(slot["candidate"])
            conclusions.append(
                {
                    "index": index,
                    "entry": {
                        "status": "EARLY_REJECTED",
                        "mode": mode,
                        "candidate_id": candidate.candidate_id,
                        "generation": candidate.generation,
                        "evaluated_races": races_so_far,
                        "mean_loss_improvement": early_evidence.mean_improvement,
                        "log_likelihood_ratio": early_evidence.log_likelihood_ratio,
                    },
                }
            )
            continue

        candidate = _candidate_from_payload(slot["candidate"])
        trials = [_trial_from_payload(value) for value in slot.get("trials", [])]
        report = RacingRecursiveImprovementGate(
            mode,
            min_future_races=min_future_races,
            false_promotion_rate=false_promotion_rate,
            false_rejection_rate=false_rejection_rate,
        ).evaluate(candidate, evaluations, trials)
        destination = Path(root) / mode / candidate.candidate_id / "PROMOTION" / "proposal.json"
        if not destination.exists():
            freeze_promotion_report(report, destination)
        conclusions.append({"index": index, "entry": report.to_dict()})

    if not conclusions:
        return []

    created = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    results: list[dict[str, object]] = []
    for conclusion in conclusions:
        entry = {**conclusion["entry"], "completed_at": created.isoformat()}
        state.setdefault("completed_candidates", []).append(entry)
        next_candidate = _new_candidate(state, source_commit, created - timedelta(microseconds=1))
        state["attempt"] = int(state["attempt"]) + 1
        state["candidate_slots"][conclusion["index"]] = _new_slot(next_candidate, state["attempt"])
        results.append(entry)
    return results


def apply_human_approved_promotion(
    state: dict[str, object],
    candidate: RacingRsiCandidate,
    report: RacingPromotionReport,
    approval: RacingHumanApproval,
) -> None:
    """Update ``active_model`` -- the only path in or out of this function.

    Requires a recorded :class:`RacingHumanApproval` chained to ``report``
    and ``candidate`` (see :func:`approve_promotion`); nothing in
    :func:`evaluate_and_rotate_candidates` can reach this on its own.
    """
    if candidate.mode != state["mode"]:
        raise ValueError("cannot promote a candidate from another racing mode")
    parameters = approved_parameters(candidate, report, approval)
    state["active_model"] = {
        "model_id": candidate.candidate_id,
        "generation": candidate.generation,
        "parameters": parameters,
        "promotion_report_sha256": report.report_sha256,
    }
