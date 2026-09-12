"""Recovered pre-race past-run details for 2026 Chukyo 2yo Stakes replay.

Only information independently recoverable from pre-race/public race records is
stored here.  The official 2026-08-30 Chukyo 2yo Stakes result is never used as
an input feature.
"""

from __future__ import annotations

from dataclasses import dataclass

from .layer2_live_input import PastRun


@dataclass(frozen=True)
class RecoveredRun:
    date: str
    surface: str
    distance_m: int
    going: str
    finish: int
    field_size: int
    first_call_position: int
    last_call_position: int
    final_section: float | None = None
    body_weight_kg: float | None = None

    def to_layer2(self, *, target_surface: str = "芝", target_distance_m: int = 1400, target_going: str = "良") -> PastRun:
        return PastRun(
            finish=self.finish,
            field_size=self.field_size,
            first_call_position=self.first_call_position,
            last_call_position=self.last_call_position,
            same_surface=self.surface == target_surface,
            same_distance_band=abs(self.distance_m - target_distance_m) <= 200,
            same_going_family=self.going == target_going,
        )


# Sources used to recover these fields include archived pre-race racecards plus
# JRA/JBIS/public race records.  Every value describes a race BEFORE the target
# 2026-08-30 event.
CHUKYO_2YO_RECOVERED_RUNS: dict[str, tuple[RecoveredRun, ...]] = {
    "1": (
        RecoveredRun("2026-08-15", "芝", 1200, "良", 7, 7, 5, 5, 36.2, 472),
    ),
    "2": (
        RecoveredRun("2026-08-16", "芝", 1200, "良", 2, 10, 5, 5, 33.2, 470),
        RecoveredRun("2026-07-11", "芝", 1200, "良", 2, 12, 1, 1, 36.1, 468),
        RecoveredRun("2026-06-21", "芝", 1600, "稍重", 6, 9, 2, 2, 35.2, 474),
    ),
    "3": (
        RecoveredRun("2026-07-26", "芝", 1400, "良", 1, 9, 8, 8, 33.7, 460),
        RecoveredRun("2026-06-20", "芝", 1400, "稍重", 6, 9, 5, 3, 36.7, 450),
    ),
    "4": (
        RecoveredRun("2026-06-27", "芝", 1200, "重", 1, 6, 1, 1, 34.0, 454),
        RecoveredRun("2026-06-07", "芝", 1400, "稍重", 2, 11, 3, 3, 35.0, 460),
    ),
    "5": (
        RecoveredRun("2026-07-12", "芝", 1200, "良", 1, 9, 1, 1, 36.0, 458),
    ),
    "6": (
        RecoveredRun("2026-08-08", "ダ", 1200, "良", 1, 10, 1, 1, 37.2, 484),
    ),
    "7": (
        RecoveredRun("2026-06-07", "芝", 1400, "稍重", 1, 11, 6, 6, 34.4, 454),
    ),
    "8": (
        RecoveredRun("2026-08-16", "ダ", 1400, "良", 8, 12, 3, 5, 39.8, 446),
    ),
    "9": (
        RecoveredRun("2026-06-28", "芝", 1200, "良", 1, 14, 3, 3, 33.6, 446),
    ),
}


def recovered_horse_ids() -> tuple[str, ...]:
    return tuple(sorted(CHUKYO_2YO_RECOVERED_RUNS, key=int))


def layer2_past_runs_for(horse_id: str) -> tuple[PastRun, ...]:
    runs = CHUKYO_2YO_RECOVERED_RUNS.get(horse_id)
    if not runs:
        raise KeyError(f"no fully recovered past runs for horse {horse_id}")
    return tuple(run.to_layer2() for run in runs)
