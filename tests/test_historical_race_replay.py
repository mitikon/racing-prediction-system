from racing_lambda.historical_races_2026 import (
    CHUKYO_2YO_STAKES_20260830,
    CHUKYO_2YO_STAKES_20260830_RESULT,
    NIIGATA_KINEN_20260830,
    NIIGATA_KINEN_20260830_RESULT,
)
from racing_lambda.historical_replay import replay_readiness, top_k_hits


def test_niigata_history_preserves_frozen_prediction_and_result_separately():
    assert NIIGATA_KINEN_20260830.frozen_lambda_ranking == ("8", "10", "5", "11", "6")
    assert NIIGATA_KINEN_20260830_RESULT.finishing_order[:5] == ("5", "3", "8", "2", "9")
    assert top_k_hits(
        NIIGATA_KINEN_20260830.frozen_lambda_ranking,
        NIIGATA_KINEN_20260830_RESULT,
        5,
    ) == 2


def test_chukyo_history_preserves_frozen_prediction_and_result_separately():
    assert CHUKYO_2YO_STAKES_20260830.frozen_lambda_ranking == ("7", "3", "4", "9", "6")
    assert CHUKYO_2YO_STAKES_20260830_RESULT.finishing_order[:5] == ("9", "4", "7", "6", "3")
    assert top_k_hits(
        CHUKYO_2YO_STAKES_20260830.frozen_lambda_ranking,
        CHUKYO_2YO_STAKES_20260830_RESULT,
        5,
    ) == 5


def test_replay_readiness_refuses_to_invent_monthly_stats():
    niigata = replay_readiness(NIIGATA_KINEN_20260830)
    chukyo = replay_readiness(CHUKYO_2YO_STAKES_20260830)
    assert niigata.odds_complete is True
    assert chukyo.odds_complete is True
    assert niigata.monthly_stats_complete is False
    assert chukyo.monthly_stats_complete is False
    assert niigata.layer2_ready is False
    assert chukyo.layer2_ready is False


def test_body_weight_is_optional_but_reported_missing():
    readiness = replay_readiness(CHUKYO_2YO_STAKES_20260830)
    assert readiness.body_weight_complete is False
    assert "body_weight" in readiness.missing_by_horse["9"]
