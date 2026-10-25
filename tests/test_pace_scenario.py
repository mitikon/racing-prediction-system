import pytest

from racing_lambda.pace_scenario import (
    analyze_pace_scenario,
    pace_fit_from_scenario,
    predict_pace_scenario,
)
from racing_lambda.schema import ComponentWeights, HorseEntry
from racing_lambda.scoring import build_prediction


def test_predict_pace_scenario_classifies_many_front_runners_as_fast():
    styles = {"1": "escape", "2": "escape", "3": "front", "4": "front", "5": "closer"}
    assert predict_pace_scenario(styles) == "fast"


def test_predict_pace_scenario_classifies_few_front_runners_as_slow():
    styles = {"1": "escape", "2": "stalk", "3": "stalk", "4": "closer", "5": "closer"}
    assert predict_pace_scenario(styles) == "slow"


def test_predict_pace_scenario_classifies_balanced_field_as_average():
    styles = {
        "1": "escape", "2": "front",
        "3": "stalk", "4": "stalk",
        "5": "closer", "6": "closer",
    }
    # 2 of 6 are front-runners (ratio 0.33), strictly between the slow
    # (<=0.20) and fast (>=0.45) thresholds.
    assert predict_pace_scenario(styles) == "average"


def test_predict_pace_scenario_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        predict_pace_scenario({})


def test_predict_pace_scenario_rejects_unknown_style():
    with pytest.raises(ValueError, match="unknown running style"):
        predict_pace_scenario({"1": "sprint"})  # type: ignore[arg-type]


def test_pace_fit_from_scenario_favors_closers_in_a_fast_scenario():
    closer_fit = pace_fit_from_scenario("closer", "fast")
    escape_fit = pace_fit_from_scenario("escape", "fast")
    assert closer_fit > escape_fit


def test_pace_fit_from_scenario_favors_front_runners_in_a_slow_scenario():
    escape_fit = pace_fit_from_scenario("escape", "slow")
    closer_fit = pace_fit_from_scenario("closer", "slow")
    assert escape_fit > closer_fit


def test_pace_fit_from_scenario_rejects_unknown_style():
    with pytest.raises(ValueError, match="unknown running style"):
        pace_fit_from_scenario("sprint", "fast")  # type: ignore[arg-type]


def test_analyze_pace_scenario_reports_consistent_scenario_and_fit_values():
    styles = {"1": "escape", "2": "escape", "3": "front", "4": "closer"}
    analysis = analyze_pace_scenario(styles)
    assert analysis.scenario == "fast"
    assert analysis.front_runner_ratio == pytest.approx(0.75)
    assert analysis.pace_fit_by_horse["1"] == pace_fit_from_scenario("escape", "fast")
    assert analysis.pace_fit_by_horse["4"] == pace_fit_from_scenario("closer", "fast")


def test_pace_scenario_output_plugs_directly_into_lambda_scoring():
    """The whole point: analyze_pace_scenario's output must be usable as a
    drop-in replacement for HorseEntry.pace_fit, without touching scoring.py."""
    styles = {"1": "escape", "2": "escape", "3": "front", "4": "closer", "5": "closer"}
    analysis = analyze_pace_scenario(styles)
    entries = [
        HorseEntry(
            horse_id=horse_id,
            horse_name=f"horse-{horse_id}",
            ability=0.5,
            pace_fit=analysis.pace_fit_by_horse[horse_id],
            course_draw_fit=0.5,
            weight_track_fit=0.5,
            jockey_skill=0.5,
            latest_information=0.5,
        )
        for horse_id in styles
    ]
    predictions = build_prediction(entries, ComponentWeights())
    assert len(predictions) == 5
    # The closers (favored under the "fast" scenario this field produces)
    # must out-rank the lone front-runner once pace_fit is the only thing
    # distinguishing otherwise-identical horses.
    ranks = {row.horse_id: row.rank for row in predictions}
    assert ranks["4"] < ranks["1"]
    assert ranks["5"] < ranks["2"]
