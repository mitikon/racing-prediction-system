from datetime import datetime, timedelta, timezone
from dataclasses import replace

import pytest

from racing_lambda.realtime_market_leading_signal import BetType, OddsSnapshot
from racing_lambda.schema import OfficialResult
from racing_lambda.simple_leading_signal_v02 import (
    Going,
    SimpleHorseFeatures,
    SimpleLeadingSignalLambdaV02,
    SimpleRaceContext,
)
from racing_lambda.simple_realtime_rsi import (
    SimpleRealtimeRsiSignal,
    SimpleThreeSnapshotRsiLearner,
    build_three_snapshot_features,
)


def _race_snapshots(race_id: str, race_shift: float = 0.0):
    base = datetime(2026, 9, 13, 5, 0, tzinfo=timezone.utc)
    rows = []
    for step, minute in enumerate((0, 15, 25)):
        for horse in range(1, 5):
            direction = 1.0 if horse in (1, 3) else -0.45
            support = {}
            for i, bet in enumerate(BetType):
                start = 0.08 + 0.012 * horse + 0.004 * i + race_shift
                value = start * (1.0 + direction * (0.05 * step + 0.008 * i))
                support[bet] = value
            rows.append(
                OddsSnapshot(
                    race_id=race_id,
                    horse_id=str(horse),
                    captured_at=base + timedelta(minutes=minute),
                    implied_support=support,
                )
            )
    return rows


def _horses():
    return [
        SimpleHorseFeatures(
            horse_id=str(i),
            horse_name=f"H{i}",
            odds=2.5 + i,
            age=4,
            assigned_weight_kg=56.0,
            body_weight_kg=480 + i * 4,
            body_weight_change_kg=0,
            predicted_position=i,
            recent_top3_count=2 if i <= 2 else 1,
            recent_top5_count=4,
            class_score=0.72 - i * 0.02,
            going_score=0.70,
            course_score=0.68,
            jockey_place_rate=0.22,
            jockey_win_return=1.0,
            jockey_place_return=1.0,
            days_since_last_run=28,
        )
        for i in range(1, 5)
    ]


def test_three_snapshot_features_require_exactly_three_points_per_horse():
    rows = _race_snapshots("R1")
    features = build_three_snapshot_features(rows)
    assert set(features.index) == {"1", "2", "3", "4"}
    assert "simple_rsi_win_change_30_to_5" in features.columns
    assert "simple_rsi2_win_level" in features.columns
    assert 0.0 <= features.loc["1", "simple_rsi2_win_level"] <= 1.0
    assert "simple_rsi_trifecta_acceleration" in features.columns
    assert "simple_rsi_cross_ticket_change_std" in features.columns


def test_three_snapshot_learner_uses_only_completed_historical_labels():
    races = [
        _race_snapshots("R1", 0.000),
        _race_snapshots("R2", 0.003),
        _race_snapshots("R3", 0.006),
    ]
    results = [
        OfficialResult("R1", ("1", "3", "2", "4")),
        OfficialResult("R2", ("3", "1", "4", "2")),
        OfficialResult("R3", ("1", "2", "3", "4")),
    ]
    learner = SimpleThreeSnapshotRsiLearner(ridge=1.0).fit(races, results)
    scored = learner.score(_race_snapshots("LIVE", 0.004))
    assert len(scored) == 4
    assert all(0.0 <= row.top3_probability <= 1.0 for row in scored)
    assert all(0.0 <= row.bug_score <= 1.0 for row in scored)
    assert learner.summary_.rows == 12


def test_simple_lambda_accepts_three_snapshot_rsi_as_optional_market_layer():
    context = SimpleRaceContext(
        race_id="LIVE",
        surface="芝",
        distance_m=1600,
        going=Going.FIRM,
        opening_week=False,
        rain=False,
        projected_front_runners=3,
    )
    baseline = SimpleLeadingSignalLambdaV02().rank(context, _horses())
    live_signals = [
        SimpleRealtimeRsiSignal("1", 0.88, 0.91, 12),
        SimpleRealtimeRsiSignal("2", 0.30, 0.20, 12),
        SimpleRealtimeRsiSignal("3", 0.75, 0.80, 12),
        SimpleRealtimeRsiSignal("4", 0.15, 0.10, 12),
    ]
    enriched = SimpleLeadingSignalLambdaV02().rank(
        context, _horses(), realtime_rsi_signals=live_signals
    )
    assert not any(row.realtime_rsi_used for row in baseline.lambda_overall_final)
    assert all(row.realtime_rsi_used for row in enriched.lambda_overall_final)
    one = next(row for row in enriched.lambda_overall_final if row.horse_id == "1")
    assert one.realtime_rsi_bug_score == 0.91
    assert "3時点全券種オッズ変動" in one.corroborating_axes


def test_dated_simple_rsi_training_flows_directly_to_lambda_without_future_result():
    races = [_race_snapshots(f"R{i}", 0.003 * i) for i in range(3)]
    results = [OfficialResult(f"R{i}", ("1", "2", "3", "4")) for i in range(3)]
    known_at = datetime(2026, 9, 13, 6, tzinfo=timezone.utc)
    learner = SimpleThreeSnapshotRsiLearner().fit(
        races, results,
        result_known_at={f"R{i}": known_at for i in range(3)},
    )
    tomorrow = [replace(row, race_id="TARGET",
                        captured_at=row.captured_at + timedelta(days=1))
                for row in _race_snapshots("TARGET")]
    signals = learner.score(tomorrow)
    assert all(row.trained_until == known_at and row.snapshot_count == 3
               for row in signals)
    with pytest.raises(ValueError, match="dated RSI training results"):
        learner.score(_race_snapshots("TARGET"))
    with pytest.raises(ValueError, match="target race"):
        learner.score(_race_snapshots("R0"))
