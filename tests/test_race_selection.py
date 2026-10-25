import pytest

from racing_lambda.race_selection import (
    RaceEdgeCandidacy,
    classify_information_tier,
    lambda_market_divergence,
    rank_races_by_edge_candidacy,
    score_race_edge_candidacy,
)


def test_classify_information_tier_detects_maiden_races():
    assert classify_information_tier("2歳未勝利") == "maiden"
    assert classify_information_tier("3歳新馬") == "maiden"


def test_classify_information_tier_detects_graded_races():
    assert classify_information_tier("神戸新聞杯 G2") == "graded"
    assert classify_information_tier("セントライト記念 GII") == "graded"


def test_classify_information_tier_defaults_to_conditional():
    assert classify_information_tier("3歳上1勝クラス") == "conditional"
    assert classify_information_tier("松戸特別") == "conditional"


def test_lambda_market_divergence_is_zero_when_lambda_just_echoes_the_crowd():
    odds = {"1": 1.5, "2": 3.0, "3": 5.0, "4": 8.0, "5": 12.0, "6": 20.0}
    favorite_top5 = ["1", "2", "3", "4", "5"]
    assert lambda_market_divergence(favorite_top5, odds) == 0


def test_lambda_market_divergence_counts_horses_lambda_adds_beyond_the_crowd():
    odds = {"1": 1.5, "2": 3.0, "3": 5.0, "4": 8.0, "5": 12.0, "6": 20.0}
    # lambda replaces the two weakest favorites (4, 5) with two longshots (6, and a
    # horse not even in the crowd's radar is not possible here, so just 6 plus a
    # repeat is invalid) - use a field where lambda disagrees on two slots.
    lambda_top5 = ["1", "2", "3", "6", "20"]
    odds_with_extra = {**odds, "20": 40.0}
    assert lambda_market_divergence(lambda_top5, odds_with_extra) == 2


def test_lambda_market_divergence_rejects_duplicate_lambda_picks():
    with pytest.raises(ValueError, match="duplicates"):
        lambda_market_divergence(["1", "1", "2"], {"1": 1.5, "2": 2.0})


def test_lambda_market_divergence_against_the_real_hanshin11_snapshot():
    # From the real 2026-09-21-hanshin-11 pre-race data used in
    # test_edge_diagnostics.py: lambda's top5 is ["1","4","7","2","3"].
    odds = {
        "1": 1.4, "2": 16.3, "3": 11.9, "4": 12.8, "5": 52.8,
        "6": 31.4, "7": 13.8, "8": 74.2, "9": 8.1, "10": 34.6, "11": 28.7,
    }
    lambda_top5 = ["1", "4", "7", "2", "3"]
    # Favorite top5 by shortest odds: 1(1.4), 9(8.1), 3(11.9), 4(12.8), 7(13.8).
    # lambda's "2" is the only pick outside that set.
    assert lambda_market_divergence(lambda_top5, odds) == 1


def test_score_race_edge_candidacy_combines_divergence_and_information_tier():
    odds = {"1": 1.5, "2": 3.0, "3": 5.0, "4": 8.0, "5": 12.0, "6": 20.0}
    lambda_top5 = ["1", "2", "3", "6", "20"]
    candidacy = score_race_edge_candidacy(
        race_id="race-x",
        race_name="2歳未勝利",
        lambda_top5=lambda_top5,
        odds_by_horse={**odds, "20": 40.0},
    )
    assert candidacy.information_tier == "maiden"
    assert candidacy.lambda_market_divergence == 2
    assert candidacy.edge_candidacy_score == pytest.approx(3.0)  # 2 divergence + 1.0 maiden bonus


def test_rank_races_by_edge_candidacy_orders_by_score_descending():
    high = RaceEdgeCandidacy(race_id="high", information_tier="maiden", lambda_market_divergence=3, edge_candidacy_score=4.0)
    low = RaceEdgeCandidacy(race_id="low", information_tier="graded", lambda_market_divergence=0, edge_candidacy_score=0.0)
    mid = RaceEdgeCandidacy(race_id="mid", information_tier="conditional", lambda_market_divergence=1, edge_candidacy_score=1.5)
    ranked = rank_races_by_edge_candidacy([low, high, mid])
    assert [c.race_id for c in ranked] == ["high", "mid", "low"]


def test_rank_races_by_edge_candidacy_requires_at_least_one_race():
    with pytest.raises(ValueError, match="at least one"):
        rank_races_by_edge_candidacy([])
