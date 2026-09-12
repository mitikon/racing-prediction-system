"""2026-09-06 実戦3レースの漏洩防止付き検証記録。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecordedFrozenPrediction:
    race_id: str
    captured_at: str
    overall_top5: tuple[str, ...]
    expanded_candidates: tuple[str, ...]
    maximum_bug: str | None
    odds_bug_top5: tuple[str, ...]


@dataclass(frozen=True)
class RecordedRaceResult:
    race_id: str
    finishing_order: tuple[str, ...]


@dataclass(frozen=True)
class ThreeRaceValidationReport:
    race_id: str
    winner_hit_top5: bool
    top3_hits: int
    top5_hits: int
    expanded_top3_hits: int
    maximum_bug_finish: int | None
    extraction_misses: tuple[str, ...]
    overvalued: tuple[str, ...]
    final_exclusions: tuple[str, ...]


def validate_record(
    prediction: RecordedFrozenPrediction,
    result: RecordedRaceResult,
) -> ThreeRaceValidationReport:
    if prediction.race_id != result.race_id:
        raise ValueError("race_id mismatch")
    if len(result.finishing_order) < 5:
        raise ValueError("at least the official top five is required")

    actual_top3 = set(result.finishing_order[:3])
    actual_top5 = set(result.finishing_order[:5])
    predicted_top5 = set(prediction.overall_top5)
    expanded = set(prediction.expanded_candidates)
    finish = {
        horse_id: index + 1
        for index, horse_id in enumerate(result.finishing_order)
    }
    return ThreeRaceValidationReport(
        race_id=result.race_id,
        winner_hit_top5=result.finishing_order[0] in predicted_top5,
        top3_hits=len(actual_top3 & predicted_top5),
        top5_hits=len(actual_top5 & predicted_top5),
        expanded_top3_hits=len(actual_top3 & expanded),
        maximum_bug_finish=(
            finish.get(prediction.maximum_bug) if prediction.maximum_bug else None
        ),
        extraction_misses=tuple(sorted(actual_top5 - expanded)),
        overvalued=tuple(
            horse_id
            for horse_id in prediction.overall_top5
            if finish.get(horse_id, 999) > 7
        ),
        final_exclusions=tuple(sorted((actual_top5 & expanded) - predicted_top5)),
    )


VALIDATION_RECORDS_2026_09_06: tuple[
    tuple[RecordedFrozenPrediction, RecordedRaceResult], ...
] = (
    (
        RecordedFrozenPrediction(
            race_id="2026-09-06_中山11R_紫苑S",
            captured_at="2026-09-06T15:10:50+09:00",
            overall_top5=("9", "2", "3", "11", "6"),
            expanded_candidates=("9", "2", "3", "11", "6", "1"),
            maximum_bug="11",
            odds_bug_top5=("11", "2", "1", "6", "7"),
        ),
        RecordedRaceResult(
            race_id="2026-09-06_中山11R_紫苑S",
            finishing_order=("4", "2", "9", "6", "1", "8", "3", "5", "11", "10", "7"),
        ),
    ),
    (
        RecordedFrozenPrediction(
            race_id="2026-09-06_阪神11R_セントウルS",
            captured_at="2026-09-06T13:11:00+09:00",
            overall_top5=("9", "5", "3", "1", "16"),
            expanded_candidates=("9", "5", "3", "1", "16", "12", "8"),
            maximum_bug="16",
            odds_bug_top5=("16", "12", "9", "5", "1"),
        ),
        RecordedRaceResult(
            race_id="2026-09-06_阪神11R_セントウルS",
            finishing_order=(
                "3", "9", "1", "2", "4", "7", "6", "5",
                "10", "14", "11", "13", "15", "16", "8", "12",
            ),
        ),
    ),
    (
        RecordedFrozenPrediction(
            race_id="2026-09-06_中山12R_1勝クラス",
            captured_at="2026-09-06T15:58:11+09:00",
            overall_top5=("6", "3", "4", "10", "7"),
            expanded_candidates=("6", "3", "4", "10", "7", "2", "16", "1"),
            maximum_bug="4",
            odds_bug_top5=("4", "6", "16", "10", "1"),
        ),
        RecordedRaceResult(
            race_id="2026-09-06_中山12R_1勝クラス",
            finishing_order=(
                "3", "7", "16", "1", "2", "11", "5", "6",
                "13", "8", "15", "9", "4", "10", "14", "12",
            ),
        ),
    ),
)


def validation_summary_2026_09_06() -> dict[str, int]:
    reports = [
        validate_record(prediction, result)
        for prediction, result in VALIDATION_RECORDS_2026_09_06
    ]
    return {
        "races": len(reports),
        "winner_hits": sum(report.winner_hit_top5 for report in reports),
        "top3_slots_captured": sum(report.top3_hits for report in reports),
        "top3_slots_total": len(reports) * 3,
        "top5_slots_captured": sum(report.top5_hits for report in reports),
        "top5_slots_total": len(reports) * 5,
        "expanded_top3_slots_captured": sum(
            report.expanded_top3_hits for report in reports
        ),
        "maximum_bug_top3_hits": sum(
            report.maximum_bug_finish is not None
            and report.maximum_bug_finish <= 3
            for report in reports
        ),
    }
