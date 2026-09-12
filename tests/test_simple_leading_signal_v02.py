from racing_lambda import (
    BugType,
    Going,
    SimpleHorseFeatures,
    SimpleLeadingSignalLambdaV02,
    SimpleRaceContext,
    VALIDATION_RECORDS_2026_09_06,
    validate_record,
    validation_summary_2026_09_06,
)


def horse(
    horse_id: str,
    position: int,
    odds: float = 10.0,
    **overrides,
) -> SimpleHorseFeatures:
    values = {
        "horse_id": horse_id,
        "horse_name": f"horse-{horse_id}",
        "odds": odds,
        "age": 4,
        "assigned_weight_kg": 55.0,
        "body_weight_kg": 480,
        "body_weight_change_kg": 0,
        "predicted_position": position,
        "recent_top3_count": 2,
        "recent_top5_count": 3,
        "class_score": 0.70,
        "going_score": 0.70,
        "course_score": 0.70,
        "jockey_place_rate": 0.20,
        "jockey_win_return": 1.0,
        "jockey_place_return": 1.0,
        "days_since_last_run": 28,
    }
    values.update(overrides)
    return SimpleHorseFeatures(**values)


def wet_context(front_runners: int = 5) -> SimpleRaceContext:
    return SimpleRaceContext(
        race_id="test-wet",
        surface="芝",
        distance_m=1200,
        going=Going.SOFT,
        opening_week=True,
        rain=True,
        projected_front_runners=front_runners,
    )


def test_wet_congested_pace_prefers_positions_four_to_seven():
    engine = SimpleLeadingSignalLambdaV02()
    leader = horse("1", 1)
    stalker = horse("2", 5)
    stalker_score = engine._pace_position_score(wet_context(), stalker)
    leader_score = engine._pace_position_score(wet_context(), leader)
    assert stalker_score > leader_score


def test_opening_week_front_bias_returns_when_pace_is_not_congested():
    engine = SimpleLeadingSignalLambdaV02()
    leader = horse("1", 1)
    stalker = horse("2", 5)
    context = wet_context(front_runners=2)
    assert engine._pace_position_score(context, leader) > engine._pace_position_score(
        context, stalker
    )


def test_three_year_old_growth_return_is_not_penalised_as_layoff():
    engine = SimpleLeadingSignalLambdaV02()
    growth = horse(
        "3",
        5,
        age=3,
        assigned_weight_kg=55.0,
        body_weight_kg=492,
        body_weight_change_kg=10,
        days_since_last_run=180,
        stakes_top5=True,
    )
    ordinary_layoff = horse("4", 5, days_since_last_run=180)
    assert engine._physical_condition_score(growth) > engine._physical_condition_score(
        ordinary_layoff
    )


def test_maximum_bug_requires_three_independent_axes():
    engine = SimpleLeadingSignalLambdaV02()
    weak = horse(
        "1",
        10,
        odds=60.0,
        recent_top3_count=0,
        recent_top5_count=0,
        class_score=0.30,
        going_score=0.30,
        course_score=0.30,
        jockey_place_rate=0.50,
        jockey_win_return=3.0,
        jockey_place_return=3.0,
    )
    strong = horse(
        "2",
        5,
        odds=20.0,
        age=3,
        assigned_weight_kg=54.0,
        body_weight_kg=500,
        body_weight_change_kg=8,
        class_score=0.90,
        going_score=0.90,
        course_score=0.85,
        jockey_place_rate=0.30,
        jockey_win_return=1.3,
        jockey_place_return=1.3,
        stakes_top5=True,
    )
    neutral = horse("3", 8, odds=2.5, class_score=0.72)
    output = engine.rank(wet_context(), [weak, strong, neutral])
    weak_row = next(row for row in output.lambda_overall_final if row.horse_id == "1")
    assert not weak_row.maximum_bug_eligible
    assert weak_row.bug_type is BugType.NONE
    assert all(
        len(row.corroborating_axes) >= 3
        for row in output.odds_distortion_bug_ranking
    )


def test_three_race_frozen_validation_baseline():
    summary = validation_summary_2026_09_06()
    assert summary == {
        "races": 3,
        "winner_hits": 2,
        "top3_slots_captured": 7,
        "top3_slots_total": 9,
        "top5_slots_captured": 8,
        "top5_slots_total": 15,
        "expanded_top3_slots_captured": 8,
        "maximum_bug_top3_hits": 0,
    }


def test_validation_classifies_known_failures_without_mutating_prediction():
    prediction, result = VALIDATION_RECORDS_2026_09_06[0]
    report = validate_record(prediction, result)
    assert report.extraction_misses == ("4",)
    assert "11" in report.overvalued
    assert report.final_exclusions == ("1",)
    assert prediction.overall_top5 == ("9", "2", "3", "11", "6")
