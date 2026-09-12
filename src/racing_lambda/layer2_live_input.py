"""Race-day adapter for the layer-2 statistical leading-signal model.

The adapter converts practical inputs into ``StatisticalSignalInput`` without
requiring the future all-ticket real-time feed.  It is intentionally based on
pre-race data only: last five runs, monthly condition statistics, current win
odds and optional body weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Iterable, Mapping

import numpy as np

from .statistical_leading_signal import StatisticalSignalInput


def _unit(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


@dataclass(frozen=True)
class PastRun:
    finish: int
    field_size: int
    final_section_rank: int | None = None
    first_call_position: int | None = None
    last_call_position: int | None = None
    same_surface: bool = False
    same_distance_band: bool = False
    same_going_family: bool = False

    def __post_init__(self) -> None:
        if self.field_size < 2:
            raise ValueError("field_size must be at least 2")
        if not 1 <= self.finish <= self.field_size:
            raise ValueError("finish must be within field_size")
        for name in ("final_section_rank", "first_call_position", "last_call_position"):
            value = getattr(self, name)
            if value is not None and not 1 <= value <= self.field_size:
                raise ValueError(f"{name} must be within field_size")


@dataclass(frozen=True)
class MonthlyConditionStats:
    """Monthly-refreshed JRA-derived condition fit values, already normalized.

    These values are deliberately explicit so the historical database can be
    rebuilt independently without changing the race-day model.
    """

    course_distance_surface_fit: float
    going_weather_season_fit: float
    meeting_frequency_fit: float
    jockey_fit: float
    trainer_fit: float

    def __post_init__(self) -> None:
        for name in (
            "course_distance_surface_fit",
            "going_weather_season_fit",
            "meeting_frequency_fit",
            "jockey_fit",
            "trainer_fit",
        ):
            _unit(getattr(self, name), name)


@dataclass(frozen=True)
class RaceDayHorseInput:
    horse_id: str
    past_runs: tuple[PastRun, ...]
    monthly_stats: MonthlyConditionStats
    win_odds: float
    body_weight_kg: float | None = None
    recent_body_weights_kg: tuple[float, ...] = ()
    expected_position_fit: float | None = None

    def __post_init__(self) -> None:
        if not self.horse_id.strip():
            raise ValueError("horse_id is required")
        if not 1 <= len(self.past_runs) <= 5:
            raise ValueError("past_runs must contain between one and five races")
        if self.win_odds <= 1.0 or not np.isfinite(self.win_odds):
            raise ValueError("win_odds must be finite decimal odds greater than 1")
        if self.body_weight_kg is not None and self.body_weight_kg <= 0:
            raise ValueError("body_weight_kg must be positive")
        if any(weight <= 0 for weight in self.recent_body_weights_kg):
            raise ValueError("recent body weights must be positive")
        if self.expected_position_fit is not None:
            _unit(self.expected_position_fit, "expected_position_fit")


def _finish_score(run: PastRun) -> float:
    # 1st -> 1.0, last -> 0.0, invariant to field size.
    return 1.0 - (run.finish - 1) / (run.field_size - 1)


def _closing_score(run: PastRun) -> float | None:
    if run.final_section_rank is None:
        return None
    return 1.0 - (run.final_section_rank - 1) / (run.field_size - 1)


def _position_score(run: PastRun) -> float | None:
    positions = [p for p in (run.first_call_position, run.last_call_position) if p is not None]
    if not positions:
        return None
    # Front/midfield capability gets a higher score than consistently deep position.
    percentile = np.mean([(p - 1) / (run.field_size - 1) for p in positions])
    return float(1.0 - percentile)


def recent_form_score(runs: tuple[PastRun, ...]) -> float:
    """Recency-weighted last-five form score with small suitability bonuses."""
    if not runs:
        raise ValueError("runs are required")
    decay = 0.72
    weights = np.asarray([decay**i for i in range(len(runs))], dtype=float)
    weights /= weights.sum()
    values = []
    for run in runs:
        finish = _finish_score(run)
        closing = _closing_score(run)
        base = 0.78 * finish + 0.22 * (closing if closing is not None else finish)
        fit_bonus = 0.04 * sum((run.same_surface, run.same_distance_band, run.same_going_family))
        values.append(min(base + fit_bonus, 1.0))
    return float(np.clip(np.dot(weights, values), 0.0, 1.0))


def pace_position_score(runs: tuple[PastRun, ...], explicit_fit: float | None = None) -> float:
    if explicit_fit is not None:
        return _unit(explicit_fit, "explicit_fit")
    observed = [score for run in runs if (score := _position_score(run)) is not None]
    if not observed:
        return 0.5
    weights = np.asarray([0.72**i for i in range(len(observed))], dtype=float)
    weights /= weights.sum()
    return float(np.clip(np.dot(weights, observed), 0.0, 1.0))


def body_weight_fit(current: float | None, history: tuple[float, ...]) -> float | None:
    if current is None or not history:
        return None
    baseline = float(np.median(np.asarray(history[-5:], dtype=float)))
    relative_change = abs(current - baseline) / baseline
    # No automatic preference for weight gain/loss.  Only large deviations are
    # treated as increasing uncertainty until empirical calibration says more.
    return float(np.clip(exp(-8.0 * relative_change), 0.0, 1.0))


def normalized_market_probabilities(items: Iterable[RaceDayHorseInput]) -> Mapping[str, float]:
    rows = list(items)
    if not rows:
        raise ValueError("at least one horse is required")
    raw = {row.horse_id: 1.0 / row.win_odds for row in rows}
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("market probability total must be positive")
    return {horse_id: value / total for horse_id, value in raw.items()}


def _condition_strength(row: RaceDayHorseInput) -> float:
    stats = row.monthly_stats
    recent = recent_form_score(row.past_runs)
    pace = pace_position_score(row.past_runs, row.expected_position_fit)
    return float(np.clip(
        0.30 * recent
        + 0.15 * pace
        + 0.19 * stats.course_distance_surface_fit
        + 0.13 * stats.going_weather_season_fit
        + 0.06 * stats.meeting_frequency_fit
        + 0.10 * stats.jockey_fit
        + 0.07 * stats.trainer_fit,
        0.0,
        1.0,
    ))


def build_statistical_inputs(items: Iterable[RaceDayHorseInput]) -> list[StatisticalSignalInput]:
    """Convert one race's practical inputs into layer-2 model inputs.

    ``market_value_gap`` compares each horse's condition-strength share against
    normalized current win-market probability.  Positive under-valuation is
    mapped above 0.5; over-valuation maps below 0.5.
    """
    rows = list(items)
    if not rows:
        raise ValueError("at least one horse is required")
    if len({row.horse_id for row in rows}) != len(rows):
        raise ValueError("horse_id values must be unique within a race")

    market = normalized_market_probabilities(rows)
    strengths = {row.horse_id: _condition_strength(row) for row in rows}
    strength_total = sum(strengths.values())
    fair_share = {
        horse_id: (value / strength_total if strength_total > 1e-12 else 1.0 / len(rows))
        for horse_id, value in strengths.items()
    }

    output: list[StatisticalSignalInput] = []
    for row in rows:
        gap = fair_share[row.horse_id] - market[row.horse_id]
        # Smoothly map probability-share difference to [0,1].  The scale is
        # deliberately conservative and can later be calibrated out-of-sample.
        market_gap = float(1.0 / (1.0 + np.exp(-8.0 * gap)))
        bw_fit = body_weight_fit(row.body_weight_kg, row.recent_body_weights_kg)
        completeness_parts = [
            len(row.past_runs) / 5.0,
            1.0,
            1.0 if row.expected_position_fit is not None or any(
                run.first_call_position is not None or run.last_call_position is not None
                for run in row.past_runs
            ) else 0.6,
            1.0 if bw_fit is not None else 0.8,
        ]
        completeness = float(np.clip(np.mean(completeness_parts), 0.0, 1.0))
        stats = row.monthly_stats
        output.append(
            StatisticalSignalInput(
                horse_id=row.horse_id,
                recent_form=recent_form_score(row.past_runs),
                pace_position_fit=pace_position_score(row.past_runs, row.expected_position_fit),
                course_distance_surface_fit=stats.course_distance_surface_fit,
                going_weather_season_fit=stats.going_weather_season_fit,
                meeting_frequency_fit=stats.meeting_frequency_fit,
                jockey_fit=stats.jockey_fit,
                trainer_fit=stats.trainer_fit,
                market_value_gap=market_gap,
                body_weight_fit=bw_fit,
                data_completeness=completeness,
            )
        )
    return output
