"""Time-ordered, fail-closed WSI/λ calibration shared by the two racing modes.

The inputs must be *recorded predictions*, not features rebuilt after a race.
The last races are held out when choosing whether a learned WSI contribution
beats the original λ score. This never edits the PCA correlation ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import ceil, isfinite
from typing import Literal, Sequence


Mode = Literal["full", "simple"]


@dataclass(frozen=True)
class WsiBridgeObservation:
    mode: Mode
    race_id: str
    horse_id: str
    field_size: int
    frozen_at: datetime
    scheduled_start: datetime
    result_known_at: datetime
    wsi_trained_until: datetime
    lambda_score: float
    wsi_score: float
    top3: bool

    def __post_init__(self) -> None:
        if self.mode not in ("full", "simple") or not self.race_id or not self.horse_id:
            raise ValueError("mode and race/horse IDs are required")
        if not isinstance(self.top3, bool):
            raise ValueError("top3 label must be a boolean")
        if not isinstance(self.field_size, int) or self.field_size < 5:
            raise ValueError("official field_size must be at least five")
        times = (self.wsi_trained_until, self.frozen_at, self.scheduled_start, self.result_known_at)
        if any(time.tzinfo is None or time.utcoffset() is None for time in times):
            raise ValueError("all bridge timestamps must be timezone-aware")
        if not (times[0] < times[1] < times[2] <= times[3]):
            raise ValueError("WSI training must precede freeze, start, and result")
        if any(not isfinite(float(value)) or not 0 <= value <= 1 for value in
               (self.lambda_score, self.wsi_score)):
            raise ValueError("bridge scores must be finite and in [0, 1]")


@dataclass(frozen=True)
class WsiBridgeSummary:
    mode: Mode
    training_races: int
    validation_races: int
    baseline_validation_mse: float
    candidate_validation_mse: float
    adopted: bool
    weight: float


class AdaptiveWsiBridge:
    """Learn a *score-mixing* weight independently for full or simple racing λ.

    The 0.10/0.90 full-mode PCA *correlation* regularization stays fixed.
    The penalty here shrinks WSI contribution, and a chronological holdout
    rejects any weight that fails to beat λ-only predictions.
    """

    def __init__(self, mode: Mode, *, min_races: int = 8, min_improvement: float = 0.001):
        if mode not in ("full", "simple") or min_races < 8 or min_improvement < 0:
            raise ValueError("invalid bridge configuration")
        self.mode = mode
        self.min_races = min_races
        self.min_improvement = float(min_improvement)

    @staticmethod
    def _loss(rows: Sequence[WsiBridgeObservation], weight: float) -> float:
        return sum(((1 - weight) * row.lambda_score + weight * row.wsi_score -
                    float(row.top3)) ** 2 for row in rows) / len(rows)

    def fit(self, observations: Sequence[WsiBridgeObservation], *, prediction_at: datetime) -> "AdaptiveWsiBridge":
        if prediction_at.tzinfo is None or prediction_at.utcoffset() is None:
            raise ValueError("prediction_at must be timezone-aware")
        if not observations or any(row.mode != self.mode for row in observations):
            raise ValueError("bridge observations must belong to one racing mode")
        races: dict[str, list[WsiBridgeObservation]] = {}
        for row in observations:
            if row.result_known_at >= prediction_at:
                raise ValueError("same-race or future results cannot train WSI bridge")
            races.setdefault(row.race_id, []).append(row)
        if len(races) < self.min_races:
            raise ValueError("insufficient distinct frozen races for WSI bridge")
        for race_id, rows in races.items():
            if (len(rows) != rows[0].field_size or
                    any(row.field_size != len(rows) for row in rows) or
                    len({row.horse_id for row in rows}) != len(rows)):
                raise ValueError(f"{race_id}: a full unique runner list is required")
            if sum(row.top3 for row in rows) != 3:
                raise ValueError(f"{race_id}: exactly three podium labels are required")
            if len({row.scheduled_start for row in rows}) != 1 or len({row.result_known_at for row in rows}) != 1:
                raise ValueError(f"{race_id}: inconsistent race/result timestamps")
        ordered = sorted(races.values(), key=lambda rows: rows[0].scheduled_start)
        if len({rows[0].scheduled_start for rows in ordered}) != len(ordered):
            raise ValueError("race start times must be unique for chronological validation")
        holdout = max(2, ceil(len(ordered) * 0.25))
        train_races, validation_races = ordered[:-holdout], ordered[-holdout:]
        if any(row.result_known_at >= validation_races[0][0].scheduled_start
               for race in train_races for row in race):
            raise ValueError("training results overlap the chronological validation window")
        train = [row for race in train_races for row in race]
        validation = [row for race in validation_races for row in race]
        candidates = [round(index * 0.05, 2) for index in range(11)]
        # Training objective regularizes large WSI weights. Validation decides adoption.
        proposed = min(candidates, key=lambda weight: (self._loss(train, weight) +
                       0.02 * weight * weight, weight))
        baseline = self._loss(validation, 0.0)
        candidate = self._loss(validation, proposed)
        adopted = proposed > 0 and baseline - candidate >= self.min_improvement
        self.weight_ = proposed if adopted else 0.0
        self.trained_through_ = max(row.result_known_at for row in observations)
        self.summary_ = WsiBridgeSummary(
            mode=self.mode, training_races=len(train_races),
            validation_races=len(validation_races),
            baseline_validation_mse=baseline,
            candidate_validation_mse=candidate,
            adopted=adopted, weight=self.weight_,
        )
        return self

    def blend(self, lambda_score: float, wsi_score: float, *, captured_at: datetime) -> float:
        if not hasattr(self, "summary_"):
            raise RuntimeError("WSI bridge must be fit before scoring")
        if captured_at.tzinfo is None or captured_at.utcoffset() is None or captured_at <= self.trained_through_:
            raise ValueError("target input must follow all WSI bridge training results")
        if any(not isfinite(float(value)) or not 0 <= value <= 1 for value in (lambda_score, wsi_score)):
            raise ValueError("bridge scores must be finite and in [0, 1]")
        return (1 - self.weight_) * float(lambda_score) + self.weight_ * float(wsi_score)
