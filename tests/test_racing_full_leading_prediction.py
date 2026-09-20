from datetime import datetime, timezone
import json

import pandas as pd
import pytest

RACE_START = datetime(2026, 9, 9, 4, 0, tzinfo=timezone.utc)

from racing_lambda import (
    FULL_LEADING_PREDICTION_NAME,
    SIMPLE_LEADING_PREDICTION_NAME,
    FullLeadingPredictionLambda,
    SimpleLeadingPredictionLambda,
    SimpleLeadingSignalLambdaV02,
    build_jra_training_frame,
    freeze_snapshot,
    ingest_snapshot,
    load_frozen_snapshot,
    odds_snapshots_from_official,
    OfficialResult,
    RACING_RSI_FEATURE_VERSION,
    calculate_support_rsi,
)


def pre_race(race_id: str, minute: int, supports: list[tuple[str, float, float]]):
    return ingest_snapshot(
        race_id=race_id,
        phase="PRE_RACE",
        source_url="https://www.jra.go.jp/",
        observed_at=datetime(2026, 9, 9, 3, minute, tzinfo=timezone.utc),
        payload={
            "market_support": [
                {
                    "horse_id": horse_id,
                    "support": {"win": win, "place": place},
                }
                for horse_id, win, place in supports
            ]
        },
    )


def race_history(race_id: str, offset: float = 0.0):
    return [
        pre_race(
            race_id,
            0,
            [
                ("1", 0.20 + offset, 0.30 + offset),
                ("2", 0.12 + offset, 0.18 + offset),
                ("3", 0.08 + offset, 0.11 + offset),
                ("4", 0.04 + offset, 0.07 + offset),
            ],
        ),
        pre_race(
            race_id,
            5,
            [
                ("1", 0.26 + offset, 0.36 + offset),
                ("2", 0.11 + offset, 0.20 + offset),
                ("3", 0.10 + offset, 0.14 + offset),
                ("4", 0.05 + offset, 0.08 + offset),
            ],
        ),
    ]


def dense_race_history(race_id: str, winner_bias: int = 1):
    snapshots = []
    for minute in range(25):
        supports = []
        for horse in range(1, 5):
            trend = 0.0025 * minute if horse == winner_bias else -0.0005 * minute
            supports.append((str(horse), 0.08 + horse * 0.02 + trend, 0.12 + horse * 0.02 + trend))
        snapshots.append(pre_race(race_id, minute, supports))
    return snapshots


def test_names_are_explicit_and_backward_compatible():
    assert FULL_LEADING_PREDICTION_NAME == "本格先行予測λ"
    assert SIMPLE_LEADING_PREDICTION_NAME == "簡易式先行予測λ"
    assert SimpleLeadingPredictionLambda is SimpleLeadingSignalLambdaV02
    assert FullLeadingPredictionLambda is not SimpleLeadingPredictionLambda


def test_public_jra_pre_race_snapshots_feed_full_model_features():
    snapshots = race_history("R1")
    rows = odds_snapshots_from_official(snapshots)
    assert len(rows) == 8
    frame = build_jra_training_frame([snapshots])
    assert frame.shape[0] == 4
    assert "win_change" in frame.columns
    assert "place_vs_win" in frame.columns
    assert "rsi_win_5_level" in frame.columns


def test_full_model_fits_and_scores_sparse_free_jra_ticket_history():
    model = FullLeadingPredictionLambda(enabled=True)
    model.fit_from_jra_history(
        recent_races=[race_history("RECENT", 0.00)],
        prior_races=[race_history("PRIOR", 0.01)],
    )
    assert len(model.feature_columns_) >= 2
    results = model.score_jra_race_research(race_history("TARGET", 0.02), scheduled_start=RACE_START)
    assert len(results) == 4
    assert {row.horse_id for row in results} == {"1", "2", "3", "4"}
    assert all(0.0 <= row.anomaly_score <= 1.0 for row in results)
    assert all(row.realtime_ready for row in results)


def test_full_model_requires_fit_and_enable_before_scoring():
    disabled = FullLeadingPredictionLambda(enabled=False)
    with pytest.raises(RuntimeError, match="disabled"):
        disabled.score_jra_race_research(race_history("TARGET"), scheduled_start=RACE_START)

    enabled = FullLeadingPredictionLambda(enabled=True)
    with pytest.raises(RuntimeError, match="fit_from_jra_history"):
        enabled.score_jra_race_research(race_history("TARGET"), scheduled_start=RACE_START)


def test_full_model_rejects_post_start_snapshot_even_without_adaptive_bridge():
    model = FullLeadingPredictionLambda(enabled=True).fit_from_jra_history(
        recent_races=[race_history("RECENT")], prior_races=[race_history("PRIOR", 0.01)]
    )
    with pytest.raises(ValueError, match="before-start"):
        model.score_jra_race_research(race_history("TARGET"), scheduled_start=datetime(2026, 9, 9, 3, 4, tzinfo=timezone.utc))


def test_result_snapshot_can_never_enter_full_leading_prediction():
    result = ingest_snapshot(
        race_id="R1",
        phase="RESULT",
        source_url="https://www.jra.go.jp/",
        observed_at=datetime(2026, 9, 9, 4, 0, tzinfo=timezone.utc),
        payload={
            "official_result": ["1", "2", "3"],
            "market_support": [
                {"horse_id": "1", "support": {"win": 0.30, "place": 0.40}}
            ],
        },
    )
    with pytest.raises(ValueError, match="RESULT snapshots cannot enter"):
        odds_snapshots_from_official([result])


def test_pre_race_ingestion_rejects_result_leakage_before_learning():
    with pytest.raises(ValueError, match="result leakage"):
        ingest_snapshot(
            race_id="R1",
            phase="PRE_RACE",
            source_url="https://www.jra.go.jp/",
            observed_at=datetime(2026, 9, 9, 3, 0, tzinfo=timezone.utc),
            payload={"finish_order": ["1", "2", "3"]},
        )


def test_non_public_or_member_jra_paths_are_rejected():
    with pytest.raises(ValueError, match="public JRA"):
        ingest_snapshot(
            race_id="R1",
            phase="PRE_RACE",
            source_url="https://example.com/race",
            payload={"market_support": []},
        )
    with pytest.raises(ValueError, match="member/ticket"):
        ingest_snapshot(
            race_id="R1",
            phase="PRE_RACE",
            source_url="https://www.jra.go.jp/dento/example",
            payload={"market_support": []},
        )


def test_frozen_jra_snapshot_is_write_once_and_tamper_detected(tmp_path):
    snapshot = race_history("FREEZE")[0]
    first = freeze_snapshot(snapshot, tmp_path)
    second = freeze_snapshot(snapshot, tmp_path)
    assert first == second
    assert load_frozen_snapshot(first) == snapshot

    data = json.loads(first.read_text(encoding="utf-8"))
    data["payload"]["market_support"][0]["support"]["win"] = 0.99
    first.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        load_frozen_snapshot(first)


def test_rsi_self_learning_uses_only_prior_results_and_scores_target_race():
    recent = [dense_race_history("RECENT1", 1), dense_race_history("RECENT2", 2)]
    prior = [dense_race_history("PRIOR1", 3), dense_race_history("PRIOR2", 4)]
    results = [
        OfficialResult(race_id="RECENT1", finishing_order=("1", "2", "3", "4")),
        OfficialResult(race_id="RECENT2", finishing_order=("2", "1", "3", "4")),
        OfficialResult(race_id="PRIOR1", finishing_order=("3", "1", "2", "4")),
        OfficialResult(race_id="PRIOR2", finishing_order=("4", "1", "2", "3")),
    ]
    model = FullLeadingPredictionLambda(enabled=True).fit_from_jra_history(
        recent_races=recent,
        prior_races=prior,
        historical_results=results,
    )
    assert model.rsi_feature_version == RACING_RSI_FEATURE_VERSION
    assert model.rsi_learning_summary_.rows == 16
    scored = model.score_jra_race_research(dense_race_history("TARGET", 1), scheduled_start=RACE_START)
    assert all(row.rsi_self_learning_score is not None for row in scored)
    assert all(0.0 <= row.combined_score <= 1.0 for row in scored)
    assert all(row.rsi_feature_count > 0 for row in scored)


def test_rsi_does_not_change_past_values_when_future_support_arrives():
    support = pd.Series([0.10 + 0.002 * index for index in range(30)])
    original = calculate_support_rsi(support, 14)
    extended = calculate_support_rsi(pd.concat([support, pd.Series([0.01])], ignore_index=True), 14)
    pd.testing.assert_series_equal(original, extended.iloc[:-1], check_names=False)


def test_target_result_cannot_be_supplied_to_pre_race_scoring():
    result = ingest_snapshot(
        race_id="TARGET",
        phase="RESULT",
        source_url="https://www.jra.go.jp/",
        observed_at=datetime(2026, 9, 9, 4, 0, tzinfo=timezone.utc),
        payload={"official_result": ["1", "2", "3", "4"]},
    )
    model = FullLeadingPredictionLambda(enabled=True)
    with pytest.raises(ValueError, match="RESULT snapshots cannot enter"):
        model.score_jra_race_research([result], scheduled_start=RACE_START)


def test_full_lambda_adaptive_rsi_keeps_pca_regularization_and_blocks_old_target():
    from datetime import timedelta
    from racing_lambda.adaptive_rsi_bridge import RsiBridgeObservation
    from racing_lambda.regularized_pca import (
        RECENT_CORRELATION_WEIGHT, PRIOR_CORRELATION_WEIGHT,
    )

    prior = [dense_race_history("PRIOR1", 1), dense_race_history("PRIOR2", 2)]
    recent = [dense_race_history("RECENT1", 3), dense_race_history("RECENT2", 4)]
    results = [OfficialResult(race_id=name, finishing_order=("1", "2", "3", "4"))
               for name in ("PRIOR1", "PRIOR2", "RECENT1", "RECENT2")]
    model = FullLeadingPredictionLambda(enabled=True).fit_from_jra_history(
        recent_races=recent, prior_races=prior, historical_results=results,
        historical_result_known_at={name: datetime(2026, 9, 9, 4, tzinfo=timezone.utc)
                                    for name in ("PRIOR1", "PRIOR2", "RECENT1", "RECENT2")},
    )
    base = datetime(2026, 8, 1, 12, tzinfo=timezone.utc)
    observations = [RsiBridgeObservation(
        mode="full", race_id=f"H{day}", horse_id=str(horse), field_size=5,
        rsi_trained_until=base + timedelta(days=day, hours=-3),
        frozen_at=base + timedelta(days=day, hours=-1),
        scheduled_start=base + timedelta(days=day),
        result_known_at=base + timedelta(days=day, hours=1),
        lambda_score=0.47 if horse <= 3 else 0.53,
        rsi_score=0.90 if horse <= 3 else 0.10,
        top3=horse <= 3,
    ) for day in range(8) for horse in range(1, 6)]
    model.fit_rsi_bridge(observations,
                         prediction_at=datetime(2026, 9, 9, 5, tzinfo=timezone.utc))
    assert model.adaptive_rsi_bridge.weight_ > 0
    target = [ingest_snapshot(
        race_id="TARGET", phase="PRE_RACE", source_url=row.source_url,
        observed_at=datetime.fromisoformat(row.observed_at) + timedelta(days=1),
        payload=row.payload,
    ) for row in dense_race_history("TARGET", 1)]
    scored = model.score_jra_race_research(
        target, scheduled_start=datetime(2026, 9, 10, 4, tzinfo=timezone.utc)
    )
    assert all(row.adaptive_rsi_weight == model.adaptive_rsi_bridge.weight_ for row in scored)
    assert RECENT_CORRELATION_WEIGHT == 0.10
    assert PRIOR_CORRELATION_WEIGHT == 0.90
    with pytest.raises(ValueError, match="training history"):
        model.score_jra_race_research(dense_race_history("TARGET", 1),
                             scheduled_start=datetime(2026, 9, 9, 4, tzinfo=timezone.utc))
