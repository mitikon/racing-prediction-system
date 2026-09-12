import pytest

from racing_lambda.monthly_stats_db import (
    AggregateEvidence,
    HorseMonthlyEvidence,
    MonthlySnapshot,
    build_monthly_condition_stats,
    build_snapshot_stats,
)
from racing_lambda.replay_recovery import chukyo_2yo_recovery_plan


def evidence(starts: int, wins: int, seconds: int, thirds: int) -> AggregateEvidence:
    return AggregateEvidence(starts, wins, seconds, thirds)


def complete_horse_evidence() -> HorseMonthlyEvidence:
    return HorseMonthlyEvidence(
        course_distance_surface=evidence(20, 3, 3, 2),
        going_weather_season=evidence(15, 2, 2, 2),
        meeting_frequency=evidence(18, 2, 3, 2),
        jockey=evidence(30, 5, 4, 3),
        trainer=evidence(25, 4, 3, 2),
    )


def test_aggregate_rejects_impossible_counts():
    with pytest.raises(ValueError):
        AggregateEvidence(2, 2, 1, 0)


def test_monthly_builder_returns_bounded_stats_and_confidence():
    result = build_monthly_condition_stats("7", complete_horse_evidence())
    stats = result.stats
    assert 0.0 <= stats.course_distance_surface_fit <= 1.0
    assert 0.0 <= stats.going_weather_season_fit <= 1.0
    assert 0.0 <= stats.meeting_frequency_fit <= 1.0
    assert 0.0 <= stats.jockey_fit <= 1.0
    assert 0.0 <= stats.trainer_fit <= 1.0
    assert 0.0 < result.sample_confidence < 1.0


def test_small_sample_extreme_is_shrunk():
    small = HorseMonthlyEvidence(
        course_distance_surface=evidence(1, 1, 0, 0),
        going_weather_season=evidence(1, 1, 0, 0),
        meeting_frequency=evidence(1, 1, 0, 0),
        jockey=evidence(1, 1, 0, 0),
        trainer=evidence(1, 1, 0, 0),
    )
    result = build_monthly_condition_stats("1", small)
    assert result.stats.jockey_fit < 0.80


def test_zero_start_block_is_not_silently_filled():
    incomplete = HorseMonthlyEvidence(
        course_distance_surface=evidence(0, 0, 0, 0),
        going_weather_season=evidence(10, 1, 2, 1),
        meeting_frequency=evidence(10, 1, 2, 1),
        jockey=evidence(10, 1, 2, 1),
        trainer=evidence(10, 1, 2, 1),
    )
    with pytest.raises(ValueError):
        build_monthly_condition_stats("1", incomplete)


def test_snapshot_builds_all_horses():
    snapshot = MonthlySnapshot(
        snapshot_id="2026-07-end",
        cutoff_date="2026-07-31",
        race_id="2026-08-30-chukyo-07",
        by_horse={"1": complete_horse_evidence(), "2": complete_horse_evidence()},
    )
    built = build_snapshot_stats(snapshot)
    assert set(built) == {"1", "2"}


def test_chukyo_replay_becomes_ready_only_after_monthly_snapshot():
    before = chukyo_2yo_recovery_plan()
    assert before.ready_for_full_layer2 is False
    assert len(before.race_level_missing) == 3

    after = chukyo_2yo_recovery_plan(monthly_snapshot_available=True)
    assert after.ready_for_full_layer2 is True
    assert after.race_level_missing == ()
    assert after.unresolved_horses == ()
