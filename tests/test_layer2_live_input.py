import pytest

from racing_lambda.layer2_live_input import (
    MonthlyConditionStats,
    PastRun,
    RaceDayHorseInput,
    body_weight_fit,
    build_statistical_inputs,
    normalized_market_probabilities,
    recent_form_score,
)
from racing_lambda.statistical_leading_signal import StatisticalLeadingSignal


def _stats(course=0.8, going=0.7, meeting=0.6, jockey=0.75, trainer=0.7):
    return MonthlyConditionStats(
        course_distance_surface_fit=course,
        going_weather_season_fit=going,
        meeting_frequency_fit=meeting,
        jockey_fit=jockey,
        trainer_fit=trainer,
    )


def _run(finish, field=12, closing=3, first=4, last=4, same=True):
    return PastRun(
        finish=finish,
        field_size=field,
        final_section_rank=closing,
        first_call_position=first,
        last_call_position=last,
        same_surface=same,
        same_distance_band=same,
        same_going_family=same,
    )


def test_recent_form_rewards_better_recent_runs():
    strong = (_run(1), _run(2), _run(3))
    weak = (_run(9), _run(10), _run(8))
    assert recent_form_score(strong) > recent_form_score(weak)


def test_market_probabilities_remove_overround():
    rows = [
        RaceDayHorseInput("1", (_run(2),), _stats(), 2.0),
        RaceDayHorseInput("2", (_run(3),), _stats(), 3.0),
        RaceDayHorseInput("3", (_run(4),), _stats(), 6.0),
    ]
    probs = normalized_market_probabilities(rows)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["1"] > probs["2"] > probs["3"]


def test_body_weight_fit_is_optional_and_penalizes_large_deviation():
    assert body_weight_fit(None, (460.0, 462.0)) is None
    stable = body_weight_fit(461.0, (460.0, 462.0, 461.0))
    large = body_weight_fit(500.0, (460.0, 462.0, 461.0))
    assert stable is not None and large is not None
    assert stable > large


def test_build_statistical_inputs_produces_layer2_ready_values():
    rows = [
        RaceDayHorseInput(
            horse_id="8",
            past_runs=(_run(1), _run(3), _run(2), _run(4), _run(3)),
            monthly_stats=_stats(course=0.88, jockey=0.82),
            win_odds=5.5,
            body_weight_kg=462.0,
            recent_body_weights_kg=(458.0, 460.0, 461.0),
        ),
        RaceDayHorseInput(
            horse_id="5",
            past_runs=(_run(5), _run(6), _run(4), _run(7), _run(5)),
            monthly_stats=_stats(course=0.62, jockey=0.64),
            win_odds=2.5,
            body_weight_kg=450.0,
            recent_body_weights_kg=(448.0, 450.0, 449.0),
        ),
    ]
    inputs = build_statistical_inputs(rows)
    assert {item.horse_id for item in inputs} == {"5", "8"}
    assert all(0.0 <= item.market_value_gap <= 1.0 for item in inputs)
    ranked = StatisticalLeadingSignal().rank(inputs)
    assert len(ranked) == 2
    assert all(0.0 <= row.score <= 1.0 for row in ranked)


def test_duplicate_horse_id_is_rejected():
    rows = [
        RaceDayHorseInput("1", (_run(2),), _stats(), 3.0),
        RaceDayHorseInput("1", (_run(4),), _stats(), 5.0),
    ]
    with pytest.raises(ValueError):
        build_statistical_inputs(rows)
