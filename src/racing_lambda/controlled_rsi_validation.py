"""Operational PRE_RACE/RESULT boundary for controlled racing RSI trials."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Generic, Mapping, Sequence, TypeVar

from .recursive_self_improvement import (
    RacingFutureEvaluation,
    RacingFrozenTrial,
    RacingPromotionReport,
    RacingRecursiveImprovementGate,
    RacingRsiCandidate,
    candidate_manifest_digest,
    canonicalize_parameter_keys,
    approve_promotion,
    approved_parameters,
    freeze_candidate,
    freeze_human_approval,
    freeze_promotion_report,
    freeze_trial,
    trial_manifest_digest,
)


def _plain(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def payload_digest(value: object) -> str:
    body = json.dumps(_plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(body.encode("utf-8")).hexdigest()


PredictionT = TypeVar("PredictionT")


def _aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _verify_candidate_parameters(candidate: RacingRsiCandidate, model: object) -> None:
    describe = getattr(model, "rsi_candidate_parameters", None)
    if not callable(describe):
        raise ValueError("candidate model cannot attest its RSI parameters")
    actual = describe()
    if not isinstance(actual, Mapping):
        raise ValueError("candidate model RSI parameters must be a mapping")
    # A pre-rename candidate may still carry old wsi_* parameter keys spelled
    # rsi_*; canonicalize before matching against the model's current (new
    # spelling only) attestation. The candidate's own stored parameters dict
    # is never rewritten - only this local comparison view.
    for name, expected in canonicalize_parameter_keys(candidate.parameters).items():
        if name not in actual:
            raise ValueError(f"candidate parameter is not implemented by prediction path: {name}")
        if _plain(actual[name]) != _plain(expected):
            raise ValueError(f"candidate model does not match frozen parameter: {name}")


@dataclass(frozen=True)
class ControlledPrediction(Generic[PredictionT]):
    """An official prediction returned only after PRE_RACE attestation."""

    mode: str
    race_id: str
    official_prediction: PredictionT
    candidate_prediction: PredictionT
    trial: RacingFrozenTrial

    def __post_init__(self) -> None:
        if self.mode != self.trial.mode or self.race_id != self.trial.race_id:
            raise ValueError("controlled prediction does not match its frozen trial")
        if payload_digest(self.official_prediction) != self.trial.baseline_prediction_sha256:
            raise ValueError("official prediction does not match frozen baseline")
        if payload_digest(self.candidate_prediction) != self.trial.candidate_prediction_sha256:
            raise ValueError("candidate prediction does not match frozen candidate")


@dataclass(frozen=True)
class TrialMetrics:
    baseline_brier_loss: float
    candidate_brier_loss: float
    baseline_top3_hits: int
    candidate_top3_hits: int
    baseline_recovery_rate: float
    candidate_recovery_rate: float
    baseline_max_bug_hit: bool
    candidate_max_bug_hit: bool


class ControlledRsiValidationLoop:
    """Persist every candidate, prediction trial, result metric, and proposal.

    Both prediction modes use this same write-once boundary, but their storage
    trees and candidates remain separated by ``candidate.mode``.
    """

    def __init__(self, root: str | Path, candidate: RacingRsiCandidate) -> None:
        self.root = Path(root) / candidate.mode / candidate.candidate_id
        self.candidate = candidate
        self.candidate_hash = candidate_manifest_digest(candidate)
        manifest = self.root / "CANDIDATE" / "candidate.json"
        if manifest.exists():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if data.get("candidate_manifest_sha256") != self.candidate_hash:
                raise ValueError("stored candidate manifest does not match requested candidate")
        else:
            freeze_candidate(candidate, manifest)

    def register_pre_race(
        self,
        *,
        race_id: str,
        registered_at: datetime,
        prediction_frozen_at: datetime,
        scheduled_start: datetime,
        input_payload: object,
        baseline_prediction: object,
        candidate_prediction: object,
    ) -> RacingFrozenTrial:
        trial = RacingFrozenTrial(
            mode=self.candidate.mode,
            candidate_id=self.candidate.candidate_id,
            race_id=race_id,
            registered_at=registered_at,
            prediction_frozen_at=prediction_frozen_at,
            scheduled_start=scheduled_start,
            baseline_prediction_sha256=payload_digest(baseline_prediction),
            candidate_prediction_sha256=payload_digest(candidate_prediction),
            candidate_manifest_sha256=self.candidate_hash,
            input_sha256=payload_digest(input_payload),
        )
        freeze_trial(trial, self.root / race_id / "PRE_RACE" / "trial.json")
        return trial

    def run_full_prediction(
        self,
        *,
        baseline_model: object,
        candidate_model: object,
        snapshots: Sequence[object],
        scheduled_start: datetime,
        prediction_frozen_at: datetime,
        registered_at: datetime | None = None,
    ) -> ControlledPrediction[object]:
        """Run both 本格先行予測λ versions and freeze before returning."""
        if self.candidate.mode != "full":
            raise ValueError("full prediction requires a full-mode RSI candidate")
        _verify_candidate_parameters(self.candidate, candidate_model)
        if "wsi_weight" in canonicalize_parameter_keys(self.candidate.parameters) and getattr(
            candidate_model, "wsi_learning_summary_", None
        ) is None:
            raise ValueError("full RSI candidate configuring wsi_weight requires trained historical WSI")
        if not snapshots:
            raise ValueError("full prediction requires PRE_RACE snapshots")
        race_id = str(getattr(snapshots[0], "race_id", "")).strip()
        if not race_id:
            raise ValueError("full prediction snapshot race_id is required")
        frozen = _aware(prediction_frozen_at, "prediction_frozen_at")
        start = _aware(scheduled_start, "scheduled_start")
        observed_at = []
        for snapshot in snapshots:
            try:
                observed_at.append(
                    _aware(datetime.fromisoformat(str(getattr(snapshot, "observed_at"))), "snapshot observed_at")
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("full prediction requires valid snapshot observed_at") from exc
        if any(observed > frozen for observed in observed_at):
            raise ValueError("full prediction cannot freeze before its latest snapshot")
        if frozen >= start:
            raise ValueError("full prediction must be frozen before scheduled_start")
        baseline = baseline_model._score_jra_race_core(
            snapshots, scheduled_start=scheduled_start
        )
        candidate = candidate_model._score_jra_race_core(
            snapshots, scheduled_start=scheduled_start
        )
        trial = self.register_pre_race(
            race_id=race_id,
            registered_at=registered_at or prediction_frozen_at,
            prediction_frozen_at=prediction_frozen_at,
            scheduled_start=scheduled_start,
            input_payload=snapshots,
            baseline_prediction=baseline,
            candidate_prediction=candidate,
        )
        return ControlledPrediction("full", race_id, baseline, candidate, trial)

    def run_simple_prediction(
        self,
        *,
        baseline_model: object,
        candidate_model: object,
        context: object,
        horses: Sequence[object],
        baseline_wsi_signals: Sequence[object] | None,
        candidate_wsi_signals: Sequence[object] | None,
        captured_at: datetime,
        prediction_frozen_at: datetime,
        registered_at: datetime | None = None,
    ) -> ControlledPrediction[object]:
        """Run both 簡易式先行予測λ versions and freeze before returning."""
        if self.candidate.mode != "simple":
            raise ValueError("simple prediction requires a simple-mode RSI candidate")
        _verify_candidate_parameters(self.candidate, candidate_model)
        race_id = str(getattr(context, "race_id", "")).strip()
        scheduled_start = getattr(context, "scheduled_start", None)
        if not race_id or not isinstance(scheduled_start, datetime):
            raise ValueError("simple prediction requires race_id and scheduled_start")
        captured = _aware(captured_at, "captured_at")
        frozen = _aware(prediction_frozen_at, "prediction_frozen_at")
        start = _aware(scheduled_start, "scheduled_start")
        if not captured <= frozen < start:
            raise ValueError(
                "simple prediction capture and freeze must precede scheduled_start"
            )
        candidate_wsi_weight = canonicalize_parameter_keys(self.candidate.parameters).get("wsi_weight", 0.0)
        if float(candidate_wsi_weight) > 0 and not candidate_wsi_signals:
            raise ValueError("simple RSI candidate configuring wsi_weight requires current PRE_RACE WSI signals")
        baseline = baseline_model._rank_core(
            context, horses, baseline_wsi_signals, captured_at=captured_at
        )
        candidate = candidate_model._rank_core(
            context, horses, candidate_wsi_signals, captured_at=captured_at
        )
        trial = self.register_pre_race(
            race_id=race_id,
            registered_at=registered_at or prediction_frozen_at,
            prediction_frozen_at=prediction_frozen_at,
            scheduled_start=scheduled_start,
            input_payload={
                "context": context,
                "horses": horses,
                "baseline_wsi_signals": baseline_wsi_signals or (),
                "candidate_wsi_signals": candidate_wsi_signals or (),
            },
            baseline_prediction=baseline,
            candidate_prediction=candidate,
        )
        return ControlledPrediction("simple", race_id, baseline, candidate, trial)

    def settle_result(
        self,
        trial: RacingFrozenTrial,
        *,
        result_known_at: datetime,
        metrics: TrialMetrics,
    ) -> RacingFutureEvaluation:
        if trial.mode != self.candidate.mode or trial.candidate_id != self.candidate.candidate_id:
            raise ValueError("trial belongs to another RSI candidate")
        evaluation = RacingFutureEvaluation(
            mode=trial.mode,
            candidate_id=trial.candidate_id,
            race_id=trial.race_id,
            prediction_frozen_at=trial.prediction_frozen_at,
            scheduled_start=trial.scheduled_start,
            result_known_at=result_known_at,
            baseline_prediction_sha256=trial.baseline_prediction_sha256,
            candidate_prediction_sha256=trial.candidate_prediction_sha256,
            candidate_manifest_sha256=trial.candidate_manifest_sha256,
            trial_manifest_sha256=trial_manifest_digest(trial),
            **asdict(metrics),
        )
        destination = self.root / trial.race_id / "RESULT" / "evaluation.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(_plain(evaluation), handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        return evaluation

    def propose(
        self,
        evaluations: Sequence[RacingFutureEvaluation],
        trials: Sequence[RacingFrozenTrial],
    ) -> RacingPromotionReport:
        report = RacingRecursiveImprovementGate(self.candidate.mode).evaluate(
            self.candidate, evaluations, trials
        )
        freeze_promotion_report(report, self.root / "PROMOTION" / "proposal.json")
        return report

    def record_human_approval(
        self,
        report: RacingPromotionReport,
        *,
        approver: str,
        approved_at: datetime,
    ) -> dict[str, object]:
        """Record approval and only then release candidate parameters."""
        approval = approve_promotion(
            report, self.candidate, approver=approver, approved_at=approved_at
        )
        freeze_human_approval(
            approval, self.root / "PROMOTION" / "human_approval.json"
        )
        return approved_parameters(self.candidate, report, approval)
