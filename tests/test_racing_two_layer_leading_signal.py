import pytest

from racing_lambda.statistical_leading_signal import (
    StatisticalLeadingSignal,
    StatisticalSignalInput,
)
from racing_lambda.two_layer_leading_signal import combine_two_layers


def test_statistical_layer_works_without_body_weight():
    item = StatisticalSignalInput(
        horse_id="8",
        recent_form=0.80,
        pace_position_fit=0.75,
        course_distance_surface_fit=0.85,
        going_weather_season_fit=0.70,
        meeting_frequency_fit=0.65,
        jockey_fit=0.80,
        trainer_fit=0.70,
        market_value_gap=0.75,
        body_weight_fit=None,
        data_completeness=0.95,
    )
    result = StatisticalLeadingSignal().score(item)
    assert 0.0 <= result.score <= 1.0
    assert result.confidence == pytest.approx(0.855)
    assert "body_weight_fit" not in result.contributions


def test_two_layer_does_not_penalize_missing_realtime_layer():
    result = combine_two_layers({"8": 0.82, "5": 0.76})
    assert result[0].horse_id == "8"
    assert result[0].combined_score == pytest.approx(0.82)
    assert result[0].mode == "statistical_only"


def test_two_layer_confluence():
    result = combine_two_layers(
        {"8": 0.80, "5": 0.70},
        {"8": 0.85, "5": 0.40},
        market_weight=0.55,
    )
    eight = next(row for row in result if row.horse_id == "8")
    assert eight.confluence is True
    assert eight.mode == "two_layer"
