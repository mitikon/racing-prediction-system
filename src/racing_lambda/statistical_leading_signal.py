"""Layer 2: lightweight statistical leading-signal model for current use.

This layer is designed for the present iPhone/iPad workflow.  It consumes
pre-race information that can be supplied without full real-time JRA feeds:
recent-five-race form, monthly condition statistics, current win odds, and
body weight when available within roughly one hour of post time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


def _unit(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


@dataclass(frozen=True)
class StatisticalSignalInput:
    horse_id: str
    recent_form: float
    pace_position_fit: float
    course_distance_surface_fit: float
    going_weather_season_fit: float
    meeting_frequency_fit: float
    jockey_fit: float
    trainer_fit: float
    market_value_gap: float
    body_weight_fit: float | None = None
    data_completeness: float = 1.0

    def __post_init__(self) -> None:
        if not self.horse_id.strip():
            raise ValueError("horse_id is required")
        for name in (
            "recent_form", "pace_position_fit", "course_distance_surface_fit",
            "going_weather_season_fit", "meeting_frequency_fit", "jockey_fit",
            "trainer_fit", "market_value_gap", "data_completeness",
        ):
            _unit(getattr(self, name), name)
        if self.body_weight_fit is not None:
            _unit(self.body_weight_fit, "body_weight_fit")


@dataclass(frozen=True)
class StatisticalSignalResult:
    horse_id: str
    score: float
    raw_score: float
    confidence: float
    contributions: Mapping[str, float]


@dataclass(frozen=True)
class StatisticalSignalWeights:
    recent_form: float = 0.22
    pace_position_fit: float = 0.14
    course_distance_surface_fit: float = 0.16
    going_weather_season_fit: float = 0.11
    meeting_frequency_fit: float = 0.07
    jockey_fit: float = 0.10
    trainer_fit: float = 0.08
    market_value_gap: float = 0.12
    body_weight_fit: float = 0.08


class StatisticalLeadingSignal:
    """Score layer-2 pre-race signals without requiring all-ticket odds feeds."""

    def __init__(self, weights: StatisticalSignalWeights | None = None) -> None:
        self.weights = weights or StatisticalSignalWeights()

    def score(self, item: StatisticalSignalInput) -> StatisticalSignalResult:
        base = {
            "recent_form": item.recent_form,
            "pace_position_fit": item.pace_position_fit,
            "course_distance_surface_fit": item.course_distance_surface_fit,
            "going_weather_season_fit": item.going_weather_season_fit,
            "meeting_frequency_fit": item.meeting_frequency_fit,
            "jockey_fit": item.jockey_fit,
            "trainer_fit": item.trainer_fit,
            "market_value_gap": item.market_value_gap,
        }
        weight_map = {
            name: float(getattr(self.weights, name))
            for name in base
        }

        if item.body_weight_fit is not None:
            base["body_weight_fit"] = item.body_weight_fit
            weight_map["body_weight_fit"] = float(self.weights.body_weight_fit)

        total_weight = sum(weight_map.values())
        if total_weight <= 0:
            raise ValueError("statistical leading-signal weights must sum to a positive value")
        normalized = {name: weight / total_weight for name, weight in weight_map.items()}
        contributions = {name: base[name] * normalized[name] for name in base}
        raw = float(sum(contributions.values()))

        # Missing pre-race fields reduce confidence rather than silently receiving
        # a neutral score.  This prevents sparse inputs from appearing precise.
        body_factor = 1.0 if item.body_weight_fit is not None else 0.90
        confidence = float(np.clip(item.data_completeness * body_factor, 0.0, 1.0))

        # Shrink uncertain estimates toward neutral 0.5.
        score = 0.5 + confidence * (raw - 0.5)
        return StatisticalSignalResult(
            horse_id=item.horse_id,
            score=float(np.clip(score, 0.0, 1.0)),
            raw_score=float(np.clip(raw, 0.0, 1.0)),
            confidence=confidence,
            contributions=contributions,
        )

    def rank(self, items: list[StatisticalSignalInput]) -> list[StatisticalSignalResult]:
        results = [self.score(item) for item in items]
        return sorted(results, key=lambda result: (-result.score, result.horse_id))
