from racing_lambda.historical_evidence_2026 import (
    CHUKYO_2YO_JOCKEY_STATS,
    CHUKYO_2YO_PAST_RUNS,
    CHUKYO_2YO_TRAINER_STATS,
    recovery_summary,
)
from racing_lambda.replay_recovery import chukyo_2yo_recovery_plan


def test_chukyo_recovered_evidence_covers_all_nine_horses():
    expected = {str(number) for number in range(1, 10)}
    assert set(CHUKYO_2YO_PAST_RUNS) == expected
    assert set(CHUKYO_2YO_JOCKEY_STATS) == expected
    assert set(CHUKYO_2YO_TRAINER_STATS) == expected


def test_recovery_summary_never_claims_result_backfill():
    summary = recovery_summary()
    assert summary["chukyo_horses_with_past_runs"] == 9
    assert summary["result_data_used_for_backfill"] is False
    assert summary["chukyo_monthly_condition_db_available"] is False


def test_chukyo_full_layer2_replay_is_blocked_only_by_monthly_inputs():
    plan = chukyo_2yo_recovery_plan()
    assert plan.ready_for_full_layer2 is False
    assert "monthly_course_distance_surface_stats" in plan.race_level_missing
    assert len(plan.gaps) == 9
    assert plan.recovered_horses == tuple(str(number) for number in range(1, 10))
    assert plan.unresolved_horses == ()
    assert all(not gap.missing for gap in plan.gaps)
