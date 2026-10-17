import pytest

from racing_lambda.market_probability import (
    FavoriteGapComparison,
    MarketProbability,
    OddsGapProbability,
    WideMarketConcentration,
    WideMarketDistortion,
    compare_favorite_and_runner_up,
    estimate_favorite_top3_probability,
    estimate_market_top3_probabilities,
    estimate_wide_market_concentration,
    estimate_wide_market_distortion,
    harville_pair_top3_probability,
    harville_top3_probabilities,
    implied_win_probabilities,
    rank_odds_gap_probabilities,
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


def test_estimate_favorite_top3_probability_picks_lowest_odds_horse():
    odds = {"5": 2.2, "9": 9.6, "13": 3.9, "16": 13.9}
    favorite = estimate_favorite_top3_probability(odds)
    assert favorite.horse_id == "5"
    assert favorite.win_odds == 2.2
    assert 0.0 <= favorite.estimated_top3_probability <= 1.0


def test_estimate_favorite_top3_probability_short_price_favorite_is_well_above_half():
    # 混戦ではない「抜けた本命」(単勝1.0倍台〜2倍前半)を想定したケース。
    odds = {"1": 1.8, "2": 8.0, "3": 12.0, "4": 20.0, "5": 25.0}
    favorite = estimate_favorite_top3_probability(odds)
    assert favorite.horse_id == "1"
    assert favorite.estimated_top3_probability > 0.75


def test_estimate_favorite_top3_probability_contested_favorite_is_lower():
    # 混戦の1番人気(単勝4〜5倍程度)を想定したケース。抜けた本命より
    # 3着以内確率が下がることを確認する。
    short_price = estimate_favorite_top3_probability(
        {"1": 1.8, "2": 8.0, "3": 12.0, "4": 20.0, "5": 25.0}
    )
    contested = estimate_favorite_top3_probability(
        {"1": 4.5, "2": 5.0, "3": 6.0, "4": 10.0, "5": 15.0}
    )
    assert contested.estimated_top3_probability < short_price.estimated_top3_probability


def test_estimate_favorite_top3_probability_rejects_empty_input():
    with pytest.raises(ValueError):
        estimate_favorite_top3_probability({})


def test_compare_favorite_and_runner_up_identifies_both_horses():
    odds = {"5": 2.2, "9": 9.6, "13": 3.9, "16": 13.9}
    comparison = compare_favorite_and_runner_up(odds)
    assert isinstance(comparison, FavoriteGapComparison)
    assert comparison.favorite_id == "5"
    assert comparison.favorite_odds == pytest.approx(2.2)
    assert comparison.runner_up_id == "13"
    assert comparison.runner_up_odds == pytest.approx(3.9)
    assert comparison.odds_gap == pytest.approx(1.7)
    assert comparison.probability_gap == pytest.approx(
        comparison.favorite_top3_probability - comparison.runner_up_top3_probability
    )


def test_compare_favorite_and_runner_up_wider_gap_widens_probability_gap():
    # 展開・脚質以外の条件(出走頭数・残りの馬のオッズ分布)を揃え、
    # 1番人気と2番人気のオッズ差だけを広げた場合の比較。
    narrow_gap = compare_favorite_and_runner_up(
        {"1": 3.0, "2": 3.5, "3": 8.0, "4": 12.0, "5": 20.0}
    )
    wide_gap = compare_favorite_and_runner_up(
        {"1": 1.5, "2": 8.0, "3": 12.0, "4": 15.0, "5": 20.0}
    )
    assert wide_gap.odds_gap > narrow_gap.odds_gap
    assert wide_gap.favorite_top3_probability > narrow_gap.favorite_top3_probability
    assert wide_gap.probability_gap > narrow_gap.probability_gap


def test_compare_favorite_and_runner_up_requires_at_least_three_horses():
    with pytest.raises(ValueError):
        compare_favorite_and_runner_up({"1": 2.0, "2": 3.0})


def test_rank_odds_gap_probabilities_covers_every_horse_in_popularity_order():
    odds = {"5": 2.2, "9": 9.6, "13": 3.9, "16": 13.9}
    rows = rank_odds_gap_probabilities(odds)
    assert [row.horse_id for row in rows] == ["5", "13", "9", "16"]
    assert [row.rank for row in rows] == [1, 2, 3, 4]
    assert all(isinstance(row, OddsGapProbability) for row in rows)


def test_rank_odds_gap_probabilities_edge_horses_have_one_sided_gaps():
    odds = {"5": 2.2, "9": 9.6, "13": 3.9, "16": 13.9}
    rows = rank_odds_gap_probabilities(odds)
    favorite = rows[0]
    longshot = rows[-1]
    assert favorite.odds_gap_to_shorter is None
    assert favorite.probability_gap_to_shorter is None
    assert favorite.odds_gap_to_longer is not None
    assert longshot.odds_gap_to_longer is None
    assert longshot.probability_gap_to_longer is None
    assert longshot.odds_gap_to_shorter is not None


def test_rank_odds_gap_probabilities_matches_compare_favorite_and_runner_up():
    odds = {"5": 2.2, "9": 9.6, "13": 3.9, "16": 13.9}
    comparison = compare_favorite_and_runner_up(odds)
    rows = rank_odds_gap_probabilities(odds)
    favorite = rows[0]
    assert favorite.horse_id == comparison.favorite_id
    assert favorite.odds_gap_to_longer == pytest.approx(comparison.odds_gap)
    assert favorite.probability_gap_to_longer == pytest.approx(comparison.probability_gap)


def test_rank_odds_gap_probabilities_middle_horse_has_both_neighbors_consistent():
    odds = {"1": 2.0, "2": 4.0, "3": 6.0, "4": 10.0}
    rows = rank_odds_gap_probabilities(odds)
    by_id = {row.horse_id: row for row in rows}
    second = by_id["2"]
    first = by_id["1"]
    third = by_id["3"]
    assert second.odds_gap_to_shorter == pytest.approx(second.win_odds - first.win_odds)
    assert second.odds_gap_to_longer == pytest.approx(third.win_odds - second.win_odds)
    assert second.probability_gap_to_shorter == pytest.approx(
        first.estimated_top3_probability - second.estimated_top3_probability
    )
    assert second.probability_gap_to_longer == pytest.approx(
        second.estimated_top3_probability - third.estimated_top3_probability
    )


def test_harville_pair_top3_probability_is_certain_in_a_three_horse_field():
    probs = {"1": 0.5, "2": 0.3, "3": 0.2}
    assert harville_pair_top3_probability(probs, "1", "2") == pytest.approx(1.0, abs=1e-9)


def test_harville_pair_top3_probability_never_exceeds_either_marginal():
    probs = {"1": 0.4, "2": 0.3, "3": 0.2, "4": 0.1}
    marginals = harville_top3_probabilities(probs)
    pair_p = harville_pair_top3_probability(probs, "1", "4")
    assert pair_p <= marginals["1"] + 1e-9
    assert pair_p <= marginals["4"] + 1e-9


def test_harville_pair_top3_probability_rejects_same_horse():
    probs = {"1": 0.4, "2": 0.3, "3": 0.2, "4": 0.1}
    with pytest.raises(ValueError):
        harville_pair_top3_probability(probs, "1", "1")


def test_harville_pair_top3_probability_rejects_unknown_horse():
    probs = {"1": 0.4, "2": 0.3, "3": 0.2, "4": 0.1}
    with pytest.raises(ValueError):
        harville_pair_top3_probability(probs, "1", "9")


def test_harville_pair_top3_probability_rejects_fewer_than_three_horses():
    probs = {"1": 0.5, "2": 0.5}
    with pytest.raises(ValueError):
        harville_pair_top3_probability(probs, "1", "2")


def test_estimate_wide_market_distortion_covers_every_pair_with_both_horses():
    odds = {"1": 2.0, "2": 4.0, "3": 6.0, "4": 10.0}
    wide_odds = {"1-2": 1.8, "3-4": 25.0}
    rows = estimate_wide_market_distortion(odds, wide_odds)
    assert {row.horse_a for row in rows} == {"1", "3"}
    assert {row.horse_b for row in rows} == {"2", "4"}
    assert isinstance(rows[0], WideMarketDistortion)


def test_estimate_wide_market_distortion_flags_underpriced_combo_above_one():
    # 3-4は長オッズ同士の組み合わせで、Harville理論値は低い。
    # それでも市場のワイドオッズが極端に低い(=市場織り込み確率が高い)場合、
    # distortion_ratioが1を大きく超えることを確認する。
    odds = {"1": 2.0, "2": 4.0, "3": 6.0, "4": 10.0}
    wide_odds = {"3-4": 1.5}
    rows = estimate_wide_market_distortion(odds, wide_odds)
    assert rows[0].distortion_ratio > 1.0


def test_estimate_wide_market_distortion_rejects_pair_referencing_unknown_horse():
    odds = {"1": 2.0, "2": 4.0, "3": 6.0}
    wide_odds = {"1-9": 5.0}
    with pytest.raises(ValueError, match="references a horse not in win_odds"):
        estimate_wide_market_distortion(odds, wide_odds)


def test_wide_market_concentration_is_near_zero_for_uniform_odds():
    wide_odds = {f"{i}-{i+1}": 10.0 for i in range(1, 10)}
    result = estimate_wide_market_concentration(wide_odds)
    assert isinstance(result, WideMarketConcentration)
    assert result.gini_coefficient == pytest.approx(0.0, abs=1e-9)
    assert result.pair_count == 9


def test_wide_market_concentration_rises_with_a_single_cheap_outlier():
    uniform = {f"{i}-{i+1}": 50.0 for i in range(1, 15)}
    flat_gini = estimate_wide_market_concentration(uniform).gini_coefficient

    skewed = dict(uniform)
    skewed["1-2"] = 1.5  # 一点だけ極端に安い(大口投資が集中したと見立てる)
    skewed_gini = estimate_wide_market_concentration(skewed).gini_coefficient

    assert skewed_gini > flat_gini


def test_wide_market_concentration_identifies_cheapest_pair_and_median():
    wide_odds = {"1-2": 2.0, "3-4": 10.0, "5-6": 40.0}
    result = estimate_wide_market_concentration(wide_odds)
    assert result.cheapest_pair == "1-2"
    assert result.cheapest_odds == pytest.approx(2.0)
    assert result.median_odds == pytest.approx(10.0)


def test_wide_market_concentration_requires_at_least_two_pairs():
    with pytest.raises(ValueError):
        estimate_wide_market_concentration({"1-2": 5.0})


def test_wide_market_concentration_rejects_odds_at_or_below_one():
    with pytest.raises(ValueError):
        estimate_wide_market_concentration({"1-2": 1.0, "3-4": 5.0})
