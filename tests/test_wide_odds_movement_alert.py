import pytest

from racing_lambda.wide_odds_movement_alert import (
    SUDDEN_WIDE_MOVEMENT_THRESHOLD,
    WideOddsMovementAlert,
    detect_sudden_wide_odds_movement,
    scan_wide_odds_for_sudden_movement,
)


def test_detect_sudden_wide_odds_movement_flags_large_shortening():
    alert = detect_sudden_wide_odds_movement("9-13", 8.0, 5.0)
    assert isinstance(alert, WideOddsMovementAlert)
    assert alert.direction == "shorten"
    assert alert.triggered is True
    assert alert.change_ratio == pytest.approx(-0.375)


def test_detect_sudden_wide_odds_movement_ignores_small_change():
    alert = detect_sudden_wide_odds_movement("9-13", 8.0, 7.6)
    assert alert.triggered is False


def test_detect_sudden_wide_odds_movement_flags_drift_up():
    alert = detect_sudden_wide_odds_movement("9-13", 5.0, 8.0)
    assert alert.direction == "drift_up"
    assert alert.triggered is True


def test_detect_sudden_wide_odds_movement_rejects_non_positive_odds():
    with pytest.raises(ValueError):
        detect_sudden_wide_odds_movement("9-13", 1.0, 5.0)


def test_detect_sudden_wide_odds_movement_respects_custom_threshold():
    alert = detect_sudden_wide_odds_movement("9-13", 8.0, 7.6, threshold=0.04)
    assert alert.triggered is True


def test_scan_wide_odds_for_sudden_movement_only_compares_common_pairs():
    # 締切直前は全通りではなく一部の組み合わせだけ共有されることを想定する
    before = {"9-13": 8.0, "9-15": 7.0, "1-9": 6.0}
    after = {"9-13": 4.5, "1-9": 6.1}  # 9-15は2回目に含まれていない
    alerts = scan_wide_odds_for_sudden_movement(before, after)
    assert {alert.pair_key for alert in alerts} == {"9-13"}
    assert alerts[0].triggered is True


def test_scan_wide_odds_for_sudden_movement_sorts_by_magnitude():
    before = {"1-2": 10.0, "3-4": 10.0}
    after = {"1-2": 5.0, "3-4": 2.0}  # 3-4の方が変動率が大きい(-80% vs -50%)
    alerts = scan_wide_odds_for_sudden_movement(before, after)
    assert [alert.pair_key for alert in alerts] == ["3-4", "1-2"]


def test_scan_wide_odds_for_sudden_movement_returns_empty_when_nothing_triggers():
    before = {"1-2": 10.0}
    after = {"1-2": 10.3}
    assert scan_wide_odds_for_sudden_movement(before, after) == []


def test_default_threshold_matches_win_odds_alert_convention():
    assert SUDDEN_WIDE_MOVEMENT_THRESHOLD == 0.10
