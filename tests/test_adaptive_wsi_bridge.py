from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from racing_lambda.adaptive_wsi_bridge import AdaptiveWsiBridge, WsiBridgeObservation
from racing_lambda.simple_leading_signal_v02 import (
    Going, SimpleHorseFeatures, SimpleLeadingSignalLambdaV02, SimpleRaceContext,
)
from racing_lambda.simple_realtime_wsi import SimpleRealtimeWsiSignal


BASE = datetime(2026, 8, 1, 12, tzinfo=timezone.utc)


def frozen_history(mode="simple", *, helpful=True):
    rows = []
    for day in range(8):
        start = BASE + timedelta(days=day)
        for number in range(1, 7):
            placed = number <= 3
            rows.append(WsiBridgeObservation(
                mode=mode, race_id=f"R{day}", horse_id=str(number), field_size=6,
                wsi_trained_until=start - timedelta(hours=3),
                frozen_at=start - timedelta(hours=1),
                scheduled_start=start,
                result_known_at=start + timedelta(hours=1),
                lambda_score=0.47 if placed else 0.53,
                wsi_score=(0.9 if placed else 0.1) if helpful else
                          (0.1 if placed else 0.9),
                top3=placed,
            ))
    return rows


TARGET = BASE + timedelta(days=9)


def test_separate_modes_adopt_only_improving_wsi_from_chronological_races():
    simple = AdaptiveWsiBridge("simple").fit(frozen_history(), prediction_at=TARGET)
    full = AdaptiveWsiBridge("full").fit(frozen_history("full"), prediction_at=TARGET)
    assert simple.summary_.adopted and full.summary_.adopted
    assert simple.summary_.training_races == 6
    assert simple.summary_.validation_races == 2
    assert 0 < simple.weight_ <= 0.5
    assert simple.blend(0.47, 0.9, captured_at=TARGET) > 0.47
    assert full.summary_.mode == "full"


def test_harmful_wsi_is_regularized_away():
    bridge = AdaptiveWsiBridge("simple").fit(
        frozen_history(helpful=False), prediction_at=TARGET
    )
    assert bridge.weight_ == 0
    assert not bridge.summary_.adopted
    assert bridge.blend(0.47, 0.1, captured_at=TARGET) == 0.47


def test_after_result_predictions_and_future_results_are_rejected():
    rows = frozen_history()
    rows[0] = replace(rows[0], result_known_at=TARGET + timedelta(hours=1))
    with pytest.raises(ValueError, match="future results"):
        AdaptiveWsiBridge("simple").fit(rows, prediction_at=TARGET)
    with pytest.raises(ValueError, match="training must precede freeze"):
        replace(frozen_history()[0], frozen_at=BASE + timedelta(hours=2))
    with pytest.raises(ValueError, match="insufficient distinct"):
        AdaptiveWsiBridge("simple").fit(frozen_history()[:42], prediction_at=TARGET)
    with pytest.raises(ValueError, match="one racing mode"):
        AdaptiveWsiBridge("full").fit(frozen_history(), prediction_at=TARGET)
    rows = frozen_history()
    rows.pop(0)
    with pytest.raises(ValueError, match="full unique runner list"):
        AdaptiveWsiBridge("simple").fit(rows, prediction_at=TARGET)


def test_simple_lambda_joins_wsi_only_after_training_and_before_next_start():
    horses = [SimpleHorseFeatures(
        horse_id=str(i), horse_name=f"Horse {i}", odds=3.0 + i,
        age=3, assigned_weight_kg=57.0, body_weight_kg=490,
        body_weight_change_kg=0, predicted_position=i,
        recent_top3_count=2, recent_top5_count=3,
        class_score=0.70, going_score=0.70, course_score=0.70,
        jockey_place_rate=0.2, jockey_win_return=1.0, jockey_place_return=1.0,
        days_since_last_run=28,
    ) for i in range(1, 7)]
    context = SimpleRaceContext("TARGET", "芝", 2200, Going.FIRM,
                                False, False, 2, TARGET + timedelta(hours=2))
    signals = [SimpleRealtimeWsiSignal(str(i), 0.9 if i == 1 else 0.1,
                                       0.8 if i == 1 else 0.2, 6,
                                       captured_at=TARGET,
                                       trained_until=TARGET - timedelta(hours=1))
               for i in range(1, 7)]
    model = SimpleLeadingSignalLambdaV02().fit_wsi_bridge(
        frozen_history(), prediction_at=TARGET
    )
    output = model.rank_research(context, horses, signals, captured_at=TARGET)
    assert all(row.adaptive_wsi_weight == model.adaptive_wsi_bridge.weight_
               for row in output.lambda_overall_final)
    assert all(row.realtime_wsi_used for row in output.lambda_overall_final)
    with pytest.raises(ValueError, match="all runners"):
        model.rank_research(context, horses, signals[:-1], captured_at=TARGET)
    with pytest.raises(ValueError, match="future scheduled_start"):
        model.rank_research(replace(context, scheduled_start=TARGET), horses,
                   signals, captured_at=TARGET)
