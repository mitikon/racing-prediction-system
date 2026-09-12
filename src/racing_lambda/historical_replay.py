"""Leakage-safe historical replay support for racing lambda development.

Historical races are useful only if pre-race facts remain strictly separated
from the official result.  This module stores both sides independently and
reports whether a race has enough confirmed pre-race data to run the current
layer-2 adapter without inventing missing values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class HistoricalHorsePreRace:
    horse_id: str
    horse_name: str
    win_odds: float | None = None
    popularity: int | None = None
    jockey: str | None = None
    assigned_weight_kg: float | None = None
    confirmed_past_runs: int = 0
    body_weight_available: bool = False
    monthly_stats_available: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.horse_id.strip() or not self.horse_name.strip():
            raise ValueError("horse_id and horse_name are required")
        if self.win_odds is not None and self.win_odds <= 1.0:
            raise ValueError("win_odds must be decimal odds greater than 1")
        if self.popularity is not None and self.popularity < 1:
            raise ValueError("popularity must be positive")
        if not 0 <= self.confirmed_past_runs <= 5:
            raise ValueError("confirmed_past_runs must be between 0 and 5")


@dataclass(frozen=True)
class HistoricalRacePreRace:
    race_id: str
    race_name: str
    venue: str
    surface: str
    distance_m: int
    going: str
    weather: str
    horses: tuple[HistoricalHorsePreRace, ...]
    frozen_lambda_ranking: tuple[str, ...] = ()
    frozen_leading_ranking: tuple[str, ...] = ()
    frozen_odds_bug_ranking: tuple[str, ...] = ()
    source_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.race_id.strip() or not self.race_name.strip():
            raise ValueError("race_id and race_name are required")
        ids = [horse.horse_id for horse in self.horses]
        if len(ids) != len(set(ids)):
            raise ValueError("horse ids must be unique")
        if self.distance_m <= 0:
            raise ValueError("distance_m must be positive")


@dataclass(frozen=True)
class HistoricalRaceResult:
    race_id: str
    finishing_order: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.race_id.strip():
            raise ValueError("race_id is required")
        if not self.finishing_order:
            raise ValueError("finishing_order is required")
        if len(self.finishing_order) != len(set(self.finishing_order)):
            raise ValueError("finishing_order must not contain duplicates")


@dataclass(frozen=True)
class ReplayReadiness:
    race_id: str
    horse_count: int
    odds_complete: bool
    past_runs_complete: bool
    monthly_stats_complete: bool
    body_weight_complete: bool
    layer2_ready: bool
    missing_by_horse: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


def replay_readiness(race: HistoricalRacePreRace) -> ReplayReadiness:
    """Check confirmed input completeness without using the official result."""
    missing: dict[str, tuple[str, ...]] = {}
    for horse in race.horses:
        fields: list[str] = []
        if horse.win_odds is None:
            fields.append("win_odds")
        if horse.confirmed_past_runs < 1:
            fields.append("past_runs")
        if not horse.monthly_stats_available:
            fields.append("monthly_stats")
        if not horse.body_weight_available:
            fields.append("body_weight")
        if fields:
            missing[horse.horse_id] = tuple(fields)

    odds_complete = all(horse.win_odds is not None for horse in race.horses)
    past_runs_complete = all(horse.confirmed_past_runs >= 1 for horse in race.horses)
    monthly_stats_complete = all(horse.monthly_stats_available for horse in race.horses)
    body_weight_complete = all(horse.body_weight_available for horse in race.horses)

    # Body weight is optional in layer 2.  Odds, at least one confirmed past run,
    # and monthly condition statistics are mandatory for a faithful replay.
    layer2_ready = odds_complete and past_runs_complete and monthly_stats_complete
    return ReplayReadiness(
        race_id=race.race_id,
        horse_count=len(race.horses),
        odds_complete=odds_complete,
        past_runs_complete=past_runs_complete,
        monthly_stats_complete=monthly_stats_complete,
        body_weight_complete=body_weight_complete,
        layer2_ready=layer2_ready,
        missing_by_horse=missing,
    )


def top_k_hits(prediction: tuple[str, ...], result: HistoricalRaceResult, k: int = 5) -> int:
    """Count result top-k horses present in the frozen pre-race top-k."""
    if k < 1:
        raise ValueError("k must be positive")
    predicted = set(prediction[:k])
    actual = set(result.finishing_order[:k])
    return len(predicted & actual)
