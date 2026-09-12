from datetime import datetime, timezone
import json

import pytest

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


def test_full_model_fits_and_scores_sparse_free_jra_ticket_history():
    model = FullLeadingPredictionLambda(enabled=True)
    model.fit_from_jra_history(
        recent_races=[race_history("RECENT", 0.00)],
        prior_races=[race_history("PRIOR", 0.01)],
    )
    assert len(model.feature_columns_) >= 2
    results = model.score_jra_race(race_history("TARGET", 0.02))
    assert len(results) == 4
    assert {row.horse_id for row in results} == {"1", "2", "3", "4"}
    assert all(0.0 <= row.anomaly_score <= 1.0 for row in results)
    assert all(row.realtime_ready for row in results)


def test_full_model_requires_fit_and_enable_before_scoring():
    disabled = FullLeadingPredictionLambda(enabled=False)
    with pytest.raises(RuntimeError, match="disabled"):
        disabled.score_jra_race(race_history("TARGET"))

    enabled = FullLeadingPredictionLambda(enabled=True)
    with pytest.raises(RuntimeError, match="fit_from_jra_history"):
        enabled.score_jra_race(race_history("TARGET"))


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
