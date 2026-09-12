"""Recovered pre-race evidence from project history.

This module stores only values that were explicitly present in the project
conversation before the official result was used for evaluation.  It is not a
replacement for the future monthly statistics database.  Unknown values stay
unknown so replay code cannot silently invent information.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecordedPastRun:
    date: str | None = None
    surface: str | None = None
    distance_m: int | None = None
    finish: int | None = None
    time: str | None = None
    final_section: float | None = None
    body_weight_kg: float | None = None
    note: str = ""


@dataclass(frozen=True)
class RecordedAggregateStats:
    starts: int
    wins: int
    seconds: int
    thirds: int
    win_rate_pct: float | None = None
    place_rate_pct: float | None = None
    win_roi_pct: float | None = None
    place_roi_pct: float | None = None


# 2026-08-30 中京7R 中京2歳ステークス G3
# Values below were recorded during the pre-race discussion.
CHUKYO_2YO_PAST_RUNS = {
    "1": (
        RecordedPastRun(surface="芝", distance_m=1200, finish=7, time="1:11.8", final_section=36.2, body_weight_kg=472, note="デビュー戦・近走評価不振"),
    ),
    "2": (
        RecordedPastRun(date="2026-08-16", surface="芝", distance_m=1200, finish=2, time="1:08.4", final_section=33.2, body_weight_kg=470),
        RecordedPastRun(date="2026-07-11", surface="芝", distance_m=1200, finish=2, time="1:08.8", final_section=36.1, body_weight_kg=468),
        RecordedPastRun(date="2026-06-21", surface="芝", distance_m=1600, finish=6, time="1:37.8", final_section=35.2, body_weight_kg=474),
    ),
    "3": (
        RecordedPastRun(date="2026-07-26", surface="芝", distance_m=1400, finish=1, time="1:20.7", final_section=33.7, body_weight_kg=460, note="明確な上昇"),
        RecordedPastRun(surface="芝", distance_m=1400, finish=6, time="1:22.9", final_section=36.7, body_weight_kg=450),
    ),
    "4": (
        RecordedPastRun(date="2026-06-27", surface="芝", distance_m=1200, finish=1, time="1:09.0", final_section=34.0, body_weight_kg=454),
        RecordedPastRun(surface="芝", distance_m=1400, finish=2, time="1:22.1", final_section=35.0, body_weight_kg=460),
    ),
    "5": (
        RecordedPastRun(surface="芝", distance_m=1200, finish=1, time="1:09.4", final_section=36.0, body_weight_kg=458),
    ),
    "6": (
        RecordedPastRun(surface="芝", distance_m=1200, finish=1, time="1:13.1", final_section=37.2, body_weight_kg=484, note="時計は遅いが条件影響の可能性"),
    ),
    "7": (
        RecordedPastRun(surface="芝", distance_m=1400, finish=1, time="1:21.9", final_section=34.4, body_weight_kg=454),
    ),
    "8": (
        RecordedPastRun(surface="芝", distance_m=1400, finish=8, time="1:28.5", final_section=39.8, body_weight_kg=446, note="デビュー戦大敗"),
    ),
    "9": (
        RecordedPastRun(surface="芝", distance_m=1200, finish=1, time="1:08.4", final_section=33.6, body_weight_kg=446, note="高速時計"),
    ),
}

CHUKYO_2YO_JOCKEY_STATS = {
    "1": RecordedAggregateStats(30, 2, 0, 1, 7, 10, 89, 40),
    "2": RecordedAggregateStats(63, 3, 6, 9, 5, 29, 58, 157),
    "3": RecordedAggregateStats(59, 6, 12, 4, 10, 37, 55, 74),
    "4": RecordedAggregateStats(78, 6, 5, 9, 8, 26, 59, 72),
    "5": RecordedAggregateStats(23, 0, 2, 0, 0, 9, 0, 35),
    "6": RecordedAggregateStats(22, 0, 1, 2, 0, 14, 0, 202),
    "7": RecordedAggregateStats(33, 5, 4, 1, 15, 30, 182, 105),
    "8": RecordedAggregateStats(47, 5, 3, 3, 11, 23, 51, 58),
    "9": RecordedAggregateStats(67, 8, 9, 5, 12, 33, 45, 53),
}

CHUKYO_2YO_TRAINER_STATS = {
    "1": RecordedAggregateStats(16, 0, 3, 1, 0, 25, 0, 123),
    "2": RecordedAggregateStats(16, 0, 3, 1, 0, 25, 0, 123),
    "3": RecordedAggregateStats(36, 6, 2, 4, 17, 33, 168, 75),
    "4": RecordedAggregateStats(35, 6, 1, 3, 17, 29, 175, 81),
    "5": RecordedAggregateStats(18, 1, 0, 2, 6, 17, 109, 110),
    "6": RecordedAggregateStats(16, 0, 3, 1, 0, 25, 0, 123),
    "7": RecordedAggregateStats(6, 0, 0, 0, 0, 0, None, None),
    "8": RecordedAggregateStats(16, 0, 3, 1, 0, 25, 0, 123),
    "9": RecordedAggregateStats(6, 1, 0, 0, 17, 17, 122, 47),
}


# 2026-08-30 新潟8R 新潟記念 G3
# Only the specific past-run facts that were explicitly preserved are recorded.
NIIGATA_KINEN_RECOVERED_FACTS = {
    "3": ("NHKマイルC G1 7着・1番人気", "NZT G2 7着・1番人気", "1勝クラス勝ち", "距離延長が論点"),
    "4": ("高クラス実績", "59kg", "約11か月休養明け"),
    "5": ("きさらぎ賞G3 1着", "東京スポーツ杯G2 2着", "皐月賞12着", "3歳55kg"),
    "6": ("G1級牝馬", "近況に懸念"),
    "8": ("目黒記念3着", "阪神大賞典3着", "安定型"),
    "9": ("高クラス実績", "59kg", "近走不振"),
    "10": ("新潟大賞典G3 2着・12番人気", "単勝29.5倍", "過小評価候補として事前抽出"),
    "11": ("G1級牝馬", "56kg"),
}


def recovery_summary() -> dict[str, object]:
    """Return a leakage-safe completeness summary for the recovered evidence."""
    return {
        "chukyo_horses_with_past_runs": len(CHUKYO_2YO_PAST_RUNS),
        "chukyo_horses_with_jockey_stats": len(CHUKYO_2YO_JOCKEY_STATS),
        "chukyo_horses_with_trainer_stats": len(CHUKYO_2YO_TRAINER_STATS),
        "chukyo_monthly_condition_db_available": False,
        "niigata_complete_past_run_numeric_data": False,
        "niigata_monthly_condition_db_available": False,
        "result_data_used_for_backfill": False,
    }
