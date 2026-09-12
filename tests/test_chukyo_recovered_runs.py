from racing_lambda.chukyo_2yo_recovered_runs import (
    layer2_past_runs_for,
    recovered_horse_ids,
)
from racing_lambda.replay_recovery import chukyo_2yo_recovery_plan


def test_all_nine_horses_have_recovered_past_run_details():
    assert recovered_horse_ids() == tuple(str(i) for i in range(1, 10))
    for horse_id in recovered_horse_ids():
        runs = layer2_past_runs_for(horse_id)
        assert len(runs) >= 1
        for run in runs:
            assert run.field_size >= 2
            assert run.first_call_position is not None
            assert run.last_call_position is not None


def test_santangelo_recovery_matches_pre_target_history():
    runs = layer2_past_runs_for("3")
    assert len(runs) == 2
    assert runs[0].finish == 1
    assert runs[0].field_size == 9
    assert runs[0].first_call_position == 8
    assert runs[0].last_call_position == 8
    assert runs[1].finish == 6
    assert runs[1].first_call_position == 5
    assert runs[1].last_call_position == 3


def test_recovery_plan_has_only_race_level_monthly_stats_left():
    plan = chukyo_2yo_recovery_plan()
    assert plan.recovered_horses == tuple(str(i) for i in range(1, 10))
    assert plan.unresolved_horses == ()
    assert all(not gap.missing for gap in plan.gaps)
    assert plan.ready_for_full_layer2 is False
    assert plan.race_level_missing == (
        "monthly_course_distance_surface_stats",
        "monthly_going_weather_season_stats",
        "monthly_meeting_frequency_stats",
    )
