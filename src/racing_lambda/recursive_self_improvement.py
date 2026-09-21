"""Controlled Recursive Self-Improvement for both racing prediction modes.

This RSI is not Relative Strength Index.  Each full/simple candidate is sealed
before a future race prediction, evaluated only after the official result, and
can emit only a human-review promotion proposal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isfinite, log
from pathlib import Path
from statistics import mean, stdev
from typing import Literal, Mapping, Sequence


Mode = Literal["full", "simple"]
RECURSIVE_SELF_IMPROVEMENT_VERSION = "racing-recursive-self-improvement-v2"
FIXED_RECENT_CORRELATION_WEIGHT = 0.10
FIXED_PRIOR_CORRELATION_WEIGHT = 0.90
ALLOWED_PARAMETERS: Mapping[str, frozenset[str]] = {
    "full": frozenset({"rsi_periods", "rsi_feature_set", "rsi_weight", "market_feature_set"}),
    "simple": frozenset({"short_rsi_period", "rsi_feature_set", "rsi_weight", "odds_snapshot_minutes"}),
}


def _utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _digest(value: str, name: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{name} must be a 64-character SHA-256")
    return normalized


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def parameter_manifest_digest(parameters: Mapping[str, object]) -> str:
    return sha256(_canonical(dict(parameters))).hexdigest()


@dataclass(frozen=True)
class RacingRsiCandidate:
    mode: Mode
    candidate_id: str
    parent_version: str
    generation: int
    created_at: datetime
    source_commit: str
    parameter_manifest_sha256: str
    parameters: Mapping[str, object]
    parent_report_sha256: str | None = None
    recent_correlation_weight: float = FIXED_RECENT_CORRELATION_WEIGHT
    prior_correlation_weight: float = FIXED_PRIOR_CORRELATION_WEIGHT

    def __post_init__(self) -> None:
        if self.mode not in ("full", "simple"):
            raise ValueError("mode must be full or simple")
        if not self.candidate_id.strip() or not self.parent_version.strip():
            raise ValueError("candidate_id and parent_version are required")
        if self.generation < 1:
            raise ValueError("generation must be positive")
        _utc(self.created_at, "created_at")
        _digest(self.parameter_manifest_sha256, "parameter_manifest_sha256")
        if self.parent_report_sha256 is not None:
            _digest(self.parent_report_sha256, "parent_report_sha256")
        if (self.generation == 1) != (self.parent_report_sha256 is None):
            raise ValueError("generation 1 has no parent report; later generations require one")
        if len(self.source_commit) != 40 or any(char not in "0123456789abcdef" for char in self.source_commit.lower()):
            raise ValueError("source_commit must be a full Git commit SHA")
        unknown = set(self.parameters) - ALLOWED_PARAMETERS[self.mode]
        if unknown:
            raise ValueError(f"candidate attempts non-allow-listed changes: {sorted(unknown)}")
        if not self.parameters:
            raise ValueError("candidate must change at least one allow-listed parameter")
        if self.parameter_manifest_sha256 != parameter_manifest_digest(self.parameters):
            raise ValueError("parameter_manifest_sha256 does not match candidate parameters")
        if (
            self.recent_correlation_weight != FIXED_RECENT_CORRELATION_WEIGHT
            or self.prior_correlation_weight != FIXED_PRIOR_CORRELATION_WEIGHT
        ):
            raise ValueError("fixed racing PCA correlation weights cannot be changed")

    def sealed_payload(self) -> dict[str, object]:
        return {
            **asdict(self),
            "created_at": _utc(self.created_at, "created_at").isoformat(),
            "parameters": dict(self.parameters),
        }


def candidate_manifest_digest(candidate: RacingRsiCandidate) -> str:
    return sha256(_canonical(candidate.sealed_payload())).hexdigest()


@dataclass(frozen=True)
class RacingFrozenTrial:
    """Pre-race attestation for one mode-specific candidate comparison."""

    mode: Mode
    candidate_id: str
    race_id: str
    registered_at: datetime
    prediction_frozen_at: datetime
    scheduled_start: datetime
    baseline_prediction_sha256: str
    candidate_prediction_sha256: str
    candidate_manifest_sha256: str
    input_sha256: str

    def __post_init__(self) -> None:
        if self.mode not in ("full", "simple") or not self.race_id.strip():
            raise ValueError("valid mode and race_id are required")
        registered = _utc(self.registered_at, "registered_at")
        frozen = _utc(self.prediction_frozen_at, "prediction_frozen_at")
        start = _utc(self.scheduled_start, "scheduled_start")
        if not registered <= frozen < start:
            raise ValueError("trial registration and prediction freeze must precede race start")
        _digest(self.baseline_prediction_sha256, "baseline_prediction_sha256")
        _digest(self.candidate_prediction_sha256, "candidate_prediction_sha256")
        _digest(self.candidate_manifest_sha256, "candidate_manifest_sha256")
        _digest(self.input_sha256, "input_sha256")

    def sealed_payload(self) -> dict[str, object]:
        return {
            **asdict(self),
            "registered_at": _utc(self.registered_at, "registered_at").isoformat(),
            "prediction_frozen_at": _utc(self.prediction_frozen_at, "prediction_frozen_at").isoformat(),
            "scheduled_start": _utc(self.scheduled_start, "scheduled_start").isoformat(),
        }


def trial_manifest_digest(trial: RacingFrozenTrial) -> str:
    return sha256(_canonical(trial.sealed_payload())).hexdigest()


@dataclass(frozen=True)
class RacingFutureEvaluation:
    mode: Mode
    candidate_id: str
    race_id: str
    prediction_frozen_at: datetime
    scheduled_start: datetime
    result_known_at: datetime
    baseline_prediction_sha256: str
    candidate_prediction_sha256: str
    candidate_manifest_sha256: str
    trial_manifest_sha256: str
    baseline_brier_loss: float
    candidate_brier_loss: float
    baseline_top3_hits: int
    candidate_top3_hits: int
    baseline_recovery_rate: float
    candidate_recovery_rate: float
    baseline_max_bug_hit: bool
    candidate_max_bug_hit: bool

    def __post_init__(self) -> None:
        if self.mode not in ("full", "simple") or not self.race_id.strip():
            raise ValueError("valid mode and race_id are required")
        frozen = _utc(self.prediction_frozen_at, "prediction_frozen_at")
        start = _utc(self.scheduled_start, "scheduled_start")
        result = _utc(self.result_known_at, "result_known_at")
        if not frozen < start <= result:
            raise ValueError("prediction must be frozen before start and official result")
        _digest(self.baseline_prediction_sha256, "baseline_prediction_sha256")
        _digest(self.candidate_prediction_sha256, "candidate_prediction_sha256")
        _digest(self.candidate_manifest_sha256, "candidate_manifest_sha256")
        _digest(self.trial_manifest_sha256, "trial_manifest_sha256")
        numeric = (
            self.baseline_brier_loss,
            self.candidate_brier_loss,
            self.baseline_recovery_rate,
            self.candidate_recovery_rate,
        )
        if any(not isinstance(value, (int, float)) or not isfinite(float(value)) for value in numeric):
            raise ValueError("evaluation metrics must be finite numbers")
        if self.baseline_brier_loss < 0 or self.candidate_brier_loss < 0:
            raise ValueError("Brier loss cannot be negative")
        if self.baseline_recovery_rate < 0 or self.candidate_recovery_rate < 0:
            raise ValueError("recovery rate cannot be negative")
        if not 0 <= self.baseline_top3_hits <= 3 or not 0 <= self.candidate_top3_hits <= 3:
            raise ValueError("top3_hits must be between zero and three")


@dataclass(frozen=True)
class RacingPromotionReport:
    status: str
    mode: Mode
    candidate_id: str
    generation: int
    evaluated_races: int
    baseline_mean_brier_loss: float
    candidate_mean_brier_loss: float
    loss_improvement: float
    baseline_top3_hits: int
    candidate_top3_hits: int
    baseline_mean_recovery_rate: float
    candidate_mean_recovery_rate: float
    baseline_max_bug_hits: int
    candidate_max_bug_hits: int
    gates: Mapping[str, bool]
    candidate_manifest_sha256: str
    report_sha256: str

    def __post_init__(self) -> None:
        if self.status not in ("PROMOTION_PROPOSED", "REJECTED"):
            raise ValueError("invalid promotion report status")
        if self.mode not in ("full", "simple"):
            raise ValueError("invalid promotion report mode")
        if self.generation < 1 or self.evaluated_races < 1:
            raise ValueError("invalid promotion report generation or race count")
        _digest(self.candidate_manifest_sha256, "candidate_manifest_sha256")
        _digest(self.report_sha256, "report_sha256")

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "gates": dict(self.gates),
            "autonomous_source_edits": False,
            "autonomous_main_merge": False,
            "betting_authority": False,
            "investment_system_access": False,
            "human_approval_required": True,
        }


def _promotion_report_unsigned(report: RacingPromotionReport) -> dict[str, object]:
    return {
        "version": RECURSIVE_SELF_IMPROVEMENT_VERSION,
        "status": report.status,
        "mode": report.mode,
        "candidate_id": report.candidate_id,
        "generation": report.generation,
        "evaluated_races": report.evaluated_races,
        "baseline_mean_brier_loss": report.baseline_mean_brier_loss,
        "candidate_mean_brier_loss": report.candidate_mean_brier_loss,
        "loss_improvement": report.loss_improvement,
        "baseline_top3_hits": report.baseline_top3_hits,
        "candidate_top3_hits": report.candidate_top3_hits,
        "baseline_mean_recovery_rate": report.baseline_mean_recovery_rate,
        "candidate_mean_recovery_rate": report.candidate_mean_recovery_rate,
        "baseline_max_bug_hits": report.baseline_max_bug_hits,
        "candidate_max_bug_hits": report.candidate_max_bug_hits,
        "gates": dict(report.gates),
        "candidate_manifest_sha256": report.candidate_manifest_sha256,
    }


def verify_promotion_report(report: RacingPromotionReport) -> None:
    """Fail closed if a proposal was forged or changed after evaluation."""
    expected = sha256(_canonical(_promotion_report_unsigned(report))).hexdigest()
    if report.report_sha256 != expected:
        raise ValueError("promotion report integrity check failed")
    gates_pass = bool(report.gates) and all(value is True for value in report.gates.values())
    expected_status = "PROMOTION_PROPOSED" if gates_pass else "REJECTED"
    if report.status != expected_status:
        raise ValueError("promotion report status does not match its gates")


@dataclass(frozen=True)
class RacingHumanApproval:
    """Write-once human decision required before candidate parameters can load."""

    status: Literal["HUMAN_APPROVED"]
    mode: Mode
    candidate_id: str
    generation: int
    approver: str
    approved_at: datetime
    report_sha256: str
    candidate_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.status != "HUMAN_APPROVED":
            raise ValueError("approval status must be HUMAN_APPROVED")
        if self.mode not in ("full", "simple") or not self.approver.strip():
            raise ValueError("valid mode and named human approver are required")
        if self.generation < 1:
            raise ValueError("approval generation must be positive")
        _utc(self.approved_at, "approved_at")
        _digest(self.report_sha256, "report_sha256")
        _digest(self.candidate_manifest_sha256, "candidate_manifest_sha256")

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "approved_at": _utc(self.approved_at, "approved_at").isoformat(),
            "autonomous_approval": False,
        }


def approve_promotion(
    report: RacingPromotionReport,
    candidate: RacingRsiCandidate,
    *,
    approver: str,
    approved_at: datetime,
) -> RacingHumanApproval:
    verify_promotion_report(report)
    if report.status != "PROMOTION_PROPOSED":
        raise ValueError("only PROMOTION_PROPOSED can be human approved")
    candidate_hash = candidate_manifest_digest(candidate)
    if (
        report.mode != candidate.mode
        or report.candidate_id != candidate.candidate_id
        or report.generation != candidate.generation
        or report.candidate_manifest_sha256 != candidate_hash
    ):
        raise ValueError("approval candidate does not match promotion report")
    return RacingHumanApproval(
        status="HUMAN_APPROVED",
        mode=candidate.mode,
        candidate_id=candidate.candidate_id,
        generation=candidate.generation,
        approver=approver,
        approved_at=approved_at,
        report_sha256=report.report_sha256,
        candidate_manifest_sha256=candidate_hash,
    )


def approved_parameters(
    candidate: RacingRsiCandidate,
    report: RacingPromotionReport,
    approval: RacingHumanApproval,
) -> dict[str, object]:
    """The only supported boundary for loading an RSI candidate into service."""
    verify_promotion_report(report)
    candidate_hash = candidate_manifest_digest(candidate)
    if report.status != "PROMOTION_PROPOSED" or approval.status != "HUMAN_APPROVED":
        raise ValueError("candidate has not completed controlled promotion")
    expected = (
        candidate.mode,
        candidate.candidate_id,
        candidate.generation,
        report.report_sha256,
        candidate_hash,
    )
    actual = (
        approval.mode,
        approval.candidate_id,
        approval.generation,
        approval.report_sha256,
        approval.candidate_manifest_sha256,
    )
    report_identity = (report.mode, report.candidate_id, report.generation)
    candidate_identity = (candidate.mode, candidate.candidate_id, candidate.generation)
    if (
        actual != expected
        or report_identity != candidate_identity
        or report.candidate_manifest_sha256 != candidate_hash
    ):
        raise ValueError("approval chain does not match candidate and report")
    return dict(candidate.parameters)


@dataclass(frozen=True)
class SequentialEvidence:
    """Wald SPRT verdict on whether candidate Brier loss beats baseline.

    Tests H0: true mean improvement <= 0 against H1: true mean improvement
    >= ``min_effect``, controlling the false-promotion rate at ``alpha`` and
    the false-rejection rate at ``beta``.  Unlike a fixed-N threshold peeked
    at repeatedly, a Wald SPRT boundary crossing is valid evidence at
    whatever sample size it first occurs, so this is safe to evaluate before
    a full evaluation window has accumulated.
    """

    decision: str  # "CONTINUE" | "PROMOTE" | "REJECT"
    races: int
    mean_improvement: float
    log_likelihood_ratio: float
    upper_boundary: float
    lower_boundary: float


def sequential_loss_improvement_test(
    baseline_losses: Sequence[float],
    candidate_losses: Sequence[float],
    *,
    min_effect: float = 0.001,
    alpha: float = 0.05,
    beta: float = 0.10,
) -> SequentialEvidence:
    if len(baseline_losses) != len(candidate_losses):
        raise ValueError("baseline and candidate loss series must be paired")
    if not baseline_losses:
        raise ValueError("at least one paired observation is required")
    if min_effect <= 0:
        raise ValueError("min_effect must be positive")
    if not 0.0 < alpha < 0.5 or not 0.0 < beta < 0.5:
        raise ValueError("alpha and beta must be in (0, 0.5)")

    races = len(baseline_losses)
    diffs = [float(b) - float(c) for b, c in zip(baseline_losses, candidate_losses)]
    upper = log((1.0 - beta) / alpha)
    lower = log(beta / (1.0 - alpha))
    mean_diff = mean(diffs)
    if races < 2:
        return SequentialEvidence("CONTINUE", races, mean_diff, 0.0, upper, lower)

    spread = stdev(diffs)
    if spread <= 1e-12:
        # Every race agrees exactly: there is no noise to test against, so
        # decide from the sign of the (unanimous) improvement directly.
        if mean_diff >= min_effect:
            return SequentialEvidence("PROMOTE", races, mean_diff, float("inf"), upper, lower)
        if mean_diff <= 0.0:
            return SequentialEvidence("REJECT", races, mean_diff, float("-inf"), upper, lower)
        return SequentialEvidence("CONTINUE", races, mean_diff, 0.0, upper, lower)

    variance = spread * spread
    mu0, mu1 = 0.0, min_effect
    llr = sum((mu1 - mu0) * (2.0 * d - mu0 - mu1) for d in diffs) / (2.0 * variance)
    if llr >= upper:
        decision = "PROMOTE"
    elif llr <= lower:
        decision = "REJECT"
    else:
        decision = "CONTINUE"
    return SequentialEvidence(decision, races, mean_diff, llr, upper, lower)


class RacingRecursiveImprovementGate:
    """Keep full/simple generations separate and test only genuinely future races."""

    def __init__(
        self,
        mode: Mode,
        *,
        min_future_races: int = 8,
        min_loss_improvement: float = 0.001,
        false_promotion_rate: float = 0.05,
        false_rejection_rate: float = 0.10,
    ) -> None:
        if mode not in ("full", "simple") or min_future_races < 8 or min_loss_improvement <= 0:
            raise ValueError("unsafe racing recursive-improvement gate configuration")
        if not 0.0 < false_promotion_rate < 0.5 or not 0.0 < false_rejection_rate < 0.5:
            raise ValueError("unsafe racing recursive-improvement gate configuration")
        self.mode = mode
        self.min_future_races = int(min_future_races)
        self.min_loss_improvement = float(min_loss_improvement)
        self.false_promotion_rate = float(false_promotion_rate)
        self.false_rejection_rate = float(false_rejection_rate)

    def evaluate(
        self,
        candidate: RacingRsiCandidate,
        observations: Sequence[RacingFutureEvaluation],
        trials: Sequence[RacingFrozenTrial],
    ) -> RacingPromotionReport:
        if candidate.mode != self.mode:
            raise ValueError("candidate belongs to another racing mode")
        rows = sorted(observations, key=lambda item: item.scheduled_start)
        if len(rows) < self.min_future_races:
            raise ValueError("insufficient future races for recursive RSI evaluation")
        if len({row.race_id for row in rows}) != len(rows):
            raise ValueError("duplicate evaluation races are prohibited")
        trials_by_race = {trial.race_id: trial for trial in trials}
        if len(trials_by_race) != len(trials) or set(trials_by_race) != {row.race_id for row in rows}:
            raise ValueError("each evaluation requires exactly one pre-race frozen trial")
        created = _utc(candidate.created_at, "created_at")
        sealed_candidate_hash = candidate_manifest_digest(candidate)
        for row in rows:
            trial = trials_by_race[row.race_id]
            if row.mode != self.mode or row.candidate_id != candidate.candidate_id:
                raise ValueError("full/simple or candidate evaluation mixing is prohibited")
            if trial.mode != self.mode or trial.candidate_id != candidate.candidate_id:
                raise ValueError("full/simple or candidate trial mixing is prohibited")
            if _utc(trial.registered_at, "registered_at") <= created:
                raise ValueError("candidate must be sealed before every registered trial")
            if (
                _utc(row.prediction_frozen_at, "prediction_frozen_at")
                != _utc(trial.prediction_frozen_at, "trial prediction_frozen_at")
                or _utc(row.scheduled_start, "scheduled_start")
                != _utc(trial.scheduled_start, "trial scheduled_start")
            ):
                raise ValueError("evaluation timing does not match frozen trial")
            if _utc(row.prediction_frozen_at, "prediction_frozen_at") <= created:
                raise ValueError("candidate must be sealed before every evaluated prediction")
            if row.candidate_manifest_sha256 != sealed_candidate_hash:
                raise ValueError("evaluation is not bound to the sealed candidate manifest")
            if trial.candidate_manifest_sha256 != sealed_candidate_hash:
                raise ValueError("trial is not bound to the sealed candidate manifest")
            if (
                row.baseline_prediction_sha256 != trial.baseline_prediction_sha256
                or row.candidate_prediction_sha256 != trial.candidate_prediction_sha256
                or row.trial_manifest_sha256 != trial_manifest_digest(trial)
            ):
                raise ValueError("settled evaluation does not match pre-race frozen trial")

        baseline_loss = mean(row.baseline_brier_loss for row in rows)
        candidate_loss = mean(row.candidate_brier_loss for row in rows)
        baseline_top3 = sum(row.baseline_top3_hits for row in rows)
        candidate_top3 = sum(row.candidate_top3_hits for row in rows)
        baseline_recovery = mean(row.baseline_recovery_rate for row in rows)
        candidate_recovery = mean(row.candidate_recovery_rate for row in rows)
        baseline_bug = sum(row.baseline_max_bug_hit for row in rows)
        candidate_bug = sum(row.candidate_max_bug_hit for row in rows)
        improvement = baseline_loss - candidate_loss
        loss_evidence = sequential_loss_improvement_test(
            [row.baseline_brier_loss for row in rows],
            [row.candidate_brier_loss for row in rows],
            min_effect=self.min_loss_improvement,
            alpha=self.false_promotion_rate,
            beta=self.false_rejection_rate,
        )
        gates = {
            "sealed_before_future_predictions": True,
            "pre_race_trial_attested": True,
            "minimum_future_races": True,
            # A Wald SPRT verdict on the paired per-race Brier-loss
            # difference, not a flat mean-improvement threshold: it controls
            # the false-promotion rate explicitly (self.false_promotion_rate)
            # instead of accepting any improvement above an arbitrary bar
            # regardless of race-to-race noise.
            "brier_loss_improved": loss_evidence.decision == "PROMOTE",
            "top3_extraction_not_worse": candidate_top3 >= baseline_top3,
            "recovery_rate_not_worse": candidate_recovery >= baseline_recovery,
            "maximum_bug_detection_not_worse": candidate_bug >= baseline_bug,
            "fixed_core_invariants": True,
            "mode_isolated": True,
        }
        status = "PROMOTION_PROPOSED" if all(gates.values()) else "REJECTED"
        provisional = RacingPromotionReport(
            status=status,
            mode=self.mode,
            candidate_id=candidate.candidate_id,
            generation=candidate.generation,
            evaluated_races=len(rows),
            baseline_mean_brier_loss=baseline_loss,
            candidate_mean_brier_loss=candidate_loss,
            loss_improvement=improvement,
            baseline_top3_hits=baseline_top3,
            candidate_top3_hits=candidate_top3,
            baseline_mean_recovery_rate=baseline_recovery,
            candidate_mean_recovery_rate=candidate_recovery,
            baseline_max_bug_hits=baseline_bug,
            candidate_max_bug_hits=candidate_bug,
            gates=gates,
            candidate_manifest_sha256=sealed_candidate_hash,
            report_sha256="0" * 64,
        )
        report_hash = sha256(_canonical(_promotion_report_unsigned(provisional))).hexdigest()
        return RacingPromotionReport(**{**asdict(provisional), "report_sha256": report_hash})


def validate_successor(
    previous: RacingPromotionReport,
    candidate: RacingRsiCandidate,
    approval: RacingHumanApproval,
) -> None:
    verify_promotion_report(previous)
    if previous.status != "PROMOTION_PROPOSED":
        raise ValueError("a rejected generation cannot become the recursive parent")
    if candidate.mode != previous.mode:
        raise ValueError("full and simple recursive generations cannot be mixed")
    if candidate.generation != previous.generation + 1:
        raise ValueError("recursive candidate generation is not sequential")
    if candidate.parent_version != previous.candidate_id:
        raise ValueError("recursive candidate parent_version mismatch")
    if candidate.parent_report_sha256 != previous.report_sha256:
        raise ValueError("recursive candidate is not chained to the prior report")
    if (
        approval.status != "HUMAN_APPROVED"
        or approval.mode != previous.mode
        or approval.candidate_id != previous.candidate_id
        or approval.generation != previous.generation
        or approval.report_sha256 != previous.report_sha256
        or approval.candidate_manifest_sha256 != previous.candidate_manifest_sha256
    ):
        raise ValueError("recursive successor requires matching human approval")


def freeze_candidate(candidate: RacingRsiCandidate, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = candidate.sealed_payload()
    payload["candidate_manifest_sha256"] = candidate_manifest_digest(candidate)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return destination


def freeze_trial(trial: RacingFrozenTrial, path: str | Path) -> Path:
    """Write the comparison trial once, before the official race result exists."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = trial.sealed_payload()
    payload["trial_manifest_sha256"] = trial_manifest_digest(trial)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return destination


def freeze_promotion_report(report: RacingPromotionReport, path: str | Path) -> Path:
    verify_promotion_report(report)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return destination


def freeze_human_approval(approval: RacingHumanApproval, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(approval.to_dict(), handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return destination
