"""Recovery planning for historical layer-2 replay.

The goal is to determine exactly what still has to be recovered before the
current race-day adapter can run faithfully.  This module never substitutes
neutral values for missing historical evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from .chukyo_2yo_recovered_runs import recovered_horse_ids
from .historical_evidence_2026 import CHUKYO_2YO_PAST_RUNS


@dataclass(frozen=True)
class HorseRecoveryGap:
    horse_id: str
    missing: tuple[str, ...]


@dataclass(frozen=True)
class RaceRecoveryPlan:
    race_id: str
    ready_for_full_layer2: bool
    gaps: tuple[HorseRecoveryGap, ...]
    race_level_missing: tuple[str, ...]
    recovered_horses: tuple[str, ...] = ()
    unresolved_horses: tuple[str, ...] = ()


def chukyo_2yo_recovery_plan(*, monthly_snapshot_available: bool = False) -> RaceRecoveryPlan:
    recovered = set(recovered_horse_ids())
    gaps: list[HorseRecoveryGap] = []
    unresolved: list[str] = []

    for horse_id, _runs in sorted(CHUKYO_2YO_PAST_RUNS.items(), key=lambda item: int(item[0])):
        missing: set[str] = set()
        if horse_id not in recovered:
            missing.add("past_run_field_size")
            missing.add("passing_positions")
            unresolved.append(horse_id)
        gaps.append(HorseRecoveryGap(horse_id=horse_id, missing=tuple(sorted(missing))))

    # Horse-level historical race details are now recovered for all nine
    # runners.  Until a frozen month-end snapshot is attached, the three
    # statistical condition families remain explicitly missing.
    race_level_missing = () if monthly_snapshot_available else (
        "monthly_course_distance_surface_stats",
        "monthly_going_weather_season_stats",
        "monthly_meeting_frequency_stats",
    )
    ready = not race_level_missing and all(not gap.missing for gap in gaps)
    return RaceRecoveryPlan(
        race_id="2026-08-30-chukyo-07",
        ready_for_full_layer2=ready,
        gaps=tuple(gaps),
        race_level_missing=race_level_missing,
        recovered_horses=tuple(sorted(recovered, key=int)),
        unresolved_horses=tuple(sorted(unresolved, key=int)),
    )
