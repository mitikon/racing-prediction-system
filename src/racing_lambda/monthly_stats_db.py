"""Monthly statistics database for the layer-2 racing leading signal.

This module converts pre-race historical aggregate counts into the normalized
``MonthlyConditionStats`` consumed by ``layer2_live_input``.  It never invents
missing aggregates.  A snapshot is buildable only when every required evidence
block is explicitly supplied.

The database is intended to be refreshed at month-end and then frozen for the
following month's race-day predictions.  Results from the target race must not
be used when constructing its snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping

import numpy as np

from .layer2_live_input import MonthlyConditionStats


def _non_negative(value: int, name: str) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class AggregateEvidence:
    """Historical starts and top-three outcomes for one condition slice."""

    starts: int
    wins: int
    seconds: int
    thirds: int

    def __post_init__(self) -> None:
        starts = _non_negative(self.starts, "starts")
        wins = _non_negative(self.wins, "wins")
        seconds = _non_negative(self.seconds, "seconds")
        thirds = _non_negative(self.thirds, "thirds")
        if wins + seconds + thirds > starts:
            raise ValueError("wins + seconds + thirds cannot exceed starts")

    @property
    def top3(self) -> int:
        return self.wins + self.seconds + self.thirds

    @property
    def place_rate(self) -> float:
        if self.starts == 0:
            raise ValueError("place_rate is undefined for zero starts")
        return self.top3 / self.starts


@dataclass(frozen=True)
class HorseMonthlyEvidence:
    """All condition evidence required for one horse's monthly layer-2 input."""

    course_distance_surface: AggregateEvidence
    going_weather_season: AggregateEvidence
    meeting_frequency: AggregateEvidence
    jockey: AggregateEvidence
    trainer: AggregateEvidence


@dataclass(frozen=True)
class MonthlySnapshot:
    """Immutable month-end snapshot keyed by horse id."""

    snapshot_id: str
    cutoff_date: str
    race_id: str
    by_horse: Mapping[str, HorseMonthlyEvidence]
    source_note: str = ""

    def __post_init__(self) -> None:
        if not self.snapshot_id.strip() or not self.cutoff_date.strip() or not self.race_id.strip():
            raise ValueError("snapshot_id, cutoff_date and race_id are required")
        if not self.by_horse:
            raise ValueError("by_horse must not be empty")
        if any(not str(horse_id).strip() for horse_id in self.by_horse):
            raise ValueError("horse ids must not be empty")


@dataclass(frozen=True)
class MonthlyBuildResult:
    horse_id: str
    stats: MonthlyConditionStats
    sample_confidence: float


def _empirical_bayes_fit(
    evidence: AggregateEvidence,
    *,
    prior_rate: float = 0.30,
    prior_strength: float = 8.0,
) -> float:
    """Shrink a place rate toward a neutral historical prior.

    The operation is sample-size aware: a one-off 100% place rate cannot become
    a 1.0 fit, while larger samples are allowed to move farther from the prior.
    ``prior_rate`` is a database-level baseline, not a target-race result.
    """
    if not 0.0 <= prior_rate <= 1.0:
        raise ValueError("prior_rate must be between 0 and 1")
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive")
    if evidence.starts == 0:
        raise ValueError("cannot score an aggregate with zero starts")
    posterior = (evidence.top3 + prior_strength * prior_rate) / (evidence.starts + prior_strength)
    # Convert around the prior into a 0..1 suitability score with 0.5 neutral.
    scale = max(prior_rate, 1.0 - prior_rate)
    centered = 0.5 + 0.5 * (posterior - prior_rate) / scale
    return float(np.clip(centered, 0.0, 1.0))


def _sample_confidence(evidence: HorseMonthlyEvidence) -> float:
    starts = np.asarray(
        [
            evidence.course_distance_surface.starts,
            evidence.going_weather_season.starts,
            evidence.meeting_frequency.starts,
            evidence.jockey.starts,
            evidence.trainer.starts,
        ],
        dtype=float,
    )
    # Saturating confidence; no arbitrary hard minimum is used for scoring.
    parts = 1.0 - np.exp(-starts / 12.0)
    return float(np.clip(np.mean(parts), 0.0, 1.0))


def build_monthly_condition_stats(
    horse_id: str,
    evidence: HorseMonthlyEvidence,
    *,
    prior_rate: float = 0.30,
    prior_strength: float = 8.0,
) -> MonthlyBuildResult:
    blocks = (
        evidence.course_distance_surface,
        evidence.going_weather_season,
        evidence.meeting_frequency,
        evidence.jockey,
        evidence.trainer,
    )
    if any(block.starts == 0 for block in blocks):
        raise ValueError("all monthly evidence blocks require at least one observed start")

    stats = MonthlyConditionStats(
        course_distance_surface_fit=_empirical_bayes_fit(
            evidence.course_distance_surface, prior_rate=prior_rate, prior_strength=prior_strength
        ),
        going_weather_season_fit=_empirical_bayes_fit(
            evidence.going_weather_season, prior_rate=prior_rate, prior_strength=prior_strength
        ),
        meeting_frequency_fit=_empirical_bayes_fit(
            evidence.meeting_frequency, prior_rate=prior_rate, prior_strength=prior_strength
        ),
        jockey_fit=_empirical_bayes_fit(evidence.jockey, prior_rate=prior_rate, prior_strength=prior_strength),
        trainer_fit=_empirical_bayes_fit(evidence.trainer, prior_rate=prior_rate, prior_strength=prior_strength),
    )
    return MonthlyBuildResult(
        horse_id=str(horse_id),
        stats=stats,
        sample_confidence=_sample_confidence(evidence),
    )


def build_snapshot_stats(
    snapshot: MonthlySnapshot,
    *,
    prior_rate: float = 0.30,
    prior_strength: float = 8.0,
) -> dict[str, MonthlyBuildResult]:
    """Build all horse-level normalized stats from one frozen snapshot."""
    return {
        horse_id: build_monthly_condition_stats(
            horse_id,
            evidence,
            prior_rate=prior_rate,
            prior_strength=prior_strength,
        )
        for horse_id, evidence in snapshot.by_horse.items()
    }
