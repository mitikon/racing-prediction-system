import pytest

from racing_lambda.market_probability import (
    MarketProbability,
    estimate_market_top3_probabilities,
    harville_top3_probabilities,
    implied_win_probabilities,
)


def test_implied_win_probabilities_removes_overround():
    odds = {"1": 2.0, "2": 4.0, "3": 4.0}
    probs = implied_win_probabilities(odds)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["1"] > probs["2"] == pytest.approx(probs["3"])


def test_implied_win_probabilities_rejects_odds_at_or_below_one():
    with pytest.raises(ValueError):
        implied_win_probabilities({"1": 1.0, "2": 3.0, "3": 5.0})


def test_implied_win_probabilities_rejects_empty_input():
    with pytest.raises(ValueError):
        implied_win_probabilities({})


def test_harville_three_horse_field_all_finish_in_top3():
    probs = {"1": 1 / 3, "2": 1 / 3, "3": 1 / 3}
    top3 = harville_top3_probabilities(probs)
    for value in top3.values():
        assert value == pytest.approx(1.0, abs=1e-9)


def test_harville_top3_probabilities_sum_to_three():
    probs = {"1": 0.4, "2": 0.3, "3": 0.2, "4": 0.1}
    top3 = harville_top3_probabilities(probs)
    assert sum(top3.values()) == pytest.approx(3.0)


def test_harville_rejects_fewer_than_three_horses():
    with pytest.raises(ValueError):
        harville_top3_probabilities({"1": 0.5, "2": 0.5})


def test_harville_rejects_probabilities_not_summing_to_one():
    with pytest.raises(ValueError):
        harville_top3_probabilities({"1": 0.5, "2": 0.3, "3": 0.1})


def test_short_favorite_has_higher_top3_probability_than_longshot():
    odds = {"1": 2.0, "2": 10.0, "3": 15.0, "4": 20.0}
    rows = estimate_market_top3_probabilities(odds)
    by_id = {row.horse_id: row for row in rows}
    assert isinstance(by_id["1"], MarketProbability)
    assert by_id["1"].estimated_top3_probability > by_id["4"].estimated_top3_probability
    assert by_id["1"].estimated_top3_probability > 0.5


def test_estimate_market_top3_probabilities_round_trips_odds():
    odds = {"1": 3.0, "2": 5.0, "3": 7.0, "4": 9.0, "5": 11.0}
    rows = estimate_market_top3_probabilities(odds)
    assert {row.horse_id for row in rows} == set(odds)
    assert all(0.0 <= row.estimated_top3_probability <= 1.0 for row in rows)
