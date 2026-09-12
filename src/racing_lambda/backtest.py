"""Leakage-safe chronological backtest for frozen racing predictions.

The runner accepts only predictions carrying an aware freeze timestamp and an
aware scheduled start timestamp.  A result cannot be evaluated when the
prediction was frozen at or after the start, preventing post-result backfill.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Sequence

from .validation_2026_09_06 import VALIDATION_RECORDS_2026_09_06


@dataclass(frozen=True)
class FrozenRaceCase:
    race_id: str
    frozen_at: str
    scheduled_start: str
    overall_top5: tuple[str, ...]
    expanded_candidates: tuple[str, ...]
    maximum_bug: str | None
    finishing_order: tuple[str, ...]
    stake_yen: int = 0
    return_yen: int | None = None

    def __post_init__(self) -> None:
        if not self.race_id.strip():
            raise ValueError("race_id is required")
        frozen = _aware_datetime(self.frozen_at, "frozen_at")
        start = _aware_datetime(self.scheduled_start, "scheduled_start")
        if frozen >= start:
            raise ValueError("prediction must be frozen before scheduled start")
        if len(self.overall_top5) != len(set(self.overall_top5)):
            raise ValueError("overall_top5 must not contain duplicates")
        if len(self.expanded_candidates) != len(set(self.expanded_candidates)):
            raise ValueError("expanded_candidates must not contain duplicates")
        if len(self.finishing_order) < 5:
            raise ValueError("official finishing order must contain at least five")
        if len(self.finishing_order) != len(set(self.finishing_order)):
            raise ValueError("finishing_order must not contain duplicates")
        if self.stake_yen < 0:
            raise ValueError("stake_yen cannot be negative")
        if self.return_yen is not None and self.stake_yen <= 0:
            raise ValueError("a settled return requires a positive stake")
        if self.return_yen is not None and self.return_yen < 0:
            raise ValueError("return_yen cannot be negative")


@dataclass(frozen=True)
class RaceBacktestRow:
    race_id: str
    frozen_at: str
    scheduled_start: str
    prior_races_available: int
    winner_hit_top5: bool
    top3_hits: int
    top5_hits: int
    expanded_top3_hits: int
    maximum_bug_finish: int | None
    extraction_misses: tuple[str, ...]
    overvalued: tuple[str, ...]
    final_exclusions: tuple[str, ...]
    stake_yen: int
    return_yen: int | None


@dataclass(frozen=True)
class RacingBacktestReport:
    races: int
    rows: tuple[RaceBacktestRow, ...]
    winner_hits: int
    top3_slots_captured: int
    top3_slots_total: int
    top5_slots_captured: int
    top5_slots_total: int
    expanded_top3_slots_captured: int
    maximum_bug_top3_hits: int
    settled_races: int
    total_stake_yen: int
    total_return_yen: int
    recovery_rate: float | None


def run_frozen_backtest(cases: Sequence[FrozenRaceCase]) -> RacingBacktestReport:
    """Evaluate immutable predictions in race-time order without future input."""
    if not cases:
        raise ValueError("at least one frozen race case is required")
    race_ids = [case.race_id for case in cases]
    if len(race_ids) != len(set(race_ids)):
        raise ValueError("race_id values must be unique")

    ordered = sorted(
        cases,
        key=lambda case: _aware_datetime(
            case.scheduled_start, "scheduled_start"
        ),
    )
    rows: list[RaceBacktestRow] = []
    for prior_count, case in enumerate(ordered):
        actual_top3 = set(case.finishing_order[:3])
        actual_top5 = set(case.finishing_order[:5])
        predicted_top5 = set(case.overall_top5)
        expanded = set(case.expanded_candidates)
        finish = {
            horse_id: index + 1
            for index, horse_id in enumerate(case.finishing_order)
        }
        rows.append(
            RaceBacktestRow(
                race_id=case.race_id,
                frozen_at=case.frozen_at,
                scheduled_start=case.scheduled_start,
                prior_races_available=prior_count,
                winner_hit_top5=case.finishing_order[0] in predicted_top5,
                top3_hits=len(actual_top3 & predicted_top5),
                top5_hits=len(actual_top5 & predicted_top5),
                expanded_top3_hits=len(actual_top3 & expanded),
                maximum_bug_finish=(
                    finish.get(case.maximum_bug) if case.maximum_bug else None
                ),
                extraction_misses=tuple(sorted(actual_top5 - expanded)),
                overvalued=tuple(
                    horse_id
                    for horse_id in case.overall_top5
                    if finish.get(horse_id, 999) > 7
                ),
                final_exclusions=tuple(
                    sorted((actual_top5 & expanded) - predicted_top5)
                ),
                stake_yen=case.stake_yen,
                return_yen=case.return_yen,
            )
        )

    settled = [row for row in rows if row.return_yen is not None]
    total_stake = sum(row.stake_yen for row in settled)
    total_return = sum(row.return_yen or 0 for row in settled)
    return RacingBacktestReport(
        races=len(rows),
        rows=tuple(rows),
        winner_hits=sum(row.winner_hit_top5 for row in rows),
        top3_slots_captured=sum(row.top3_hits for row in rows),
        top3_slots_total=len(rows) * 3,
        top5_slots_captured=sum(row.top5_hits for row in rows),
        top5_slots_total=len(rows) * 5,
        expanded_top3_slots_captured=sum(row.expanded_top3_hits for row in rows),
        maximum_bug_top3_hits=sum(
            row.maximum_bug_finish is not None and row.maximum_bug_finish <= 3
            for row in rows
        ),
        settled_races=len(settled),
        total_stake_yen=total_stake,
        total_return_yen=total_return,
        recovery_rate=(total_return / total_stake if total_stake else None),
    )


def builtin_backtest_cases_2026_09_06() -> tuple[FrozenRaceCase, ...]:
    """Return the three timestamp-audited races currently available."""
    scheduled_starts = {
        "2026-09-06_中山11R_紫苑S": "2026-09-06T15:45:00+09:00",
        "2026-09-06_阪神11R_セントウルS": "2026-09-06T15:35:00+09:00",
        "2026-09-06_中山12R_1勝クラス": "2026-09-06T16:30:00+09:00",
    }
    return tuple(
        FrozenRaceCase(
            race_id=prediction.race_id,
            frozen_at=prediction.captured_at,
            scheduled_start=scheduled_starts[prediction.race_id],
            overall_top5=prediction.overall_top5,
            expanded_candidates=prediction.expanded_candidates,
            maximum_bug=prediction.maximum_bug,
            finishing_order=result.finishing_order,
        )
        for prediction, result in VALIDATION_RECORDS_2026_09_06
    )


def _aware_datetime(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed


def main() -> None:
    report = run_frozen_backtest(builtin_backtest_cases_2026_09_06())
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
