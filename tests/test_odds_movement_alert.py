import pytest

from racing_lambda.odds_movement_alert import (
    SUDDEN_MOVEMENT_THRESHOLD,
    detect_sudden_odds_movement,
    scan_field_for_sudden_movement,
)


def test_threshold_is_ten_percent():
    assert SUDDEN_MOVEMENT_THRESHOLD == pytest.approx(0.10)


def test_drift_up_past_threshold_is_triggered():
    alert = detect_sudden_odds_movement("12", odds_before=130.0, odds_after=248.9)
    assert alert.direction == "drift_up"
    assert alert.triggered is True
    assert alert.change_ratio > 0.10


def test_movement_under_threshold_is_not_triggered():
    alert = detect_sudden_odds_movement("3", odds_before=10.0, odds_after=10.5)
    assert alert.triggered is False


def test_shorten_past_threshold_is_triggered():
    alert = detect_sudden_odds_movement("8", odds_before=20.0, odds_after=15.0)
    assert alert.direction == "shorten"
    assert alert.triggered is True


def test_decimal_odds_must_exceed_one():
    with pytest.raises(ValueError):
        detect_sudden_odds_movement("1", odds_before=1.0, odds_after=2.0)


def test_scan_field_returns_only_triggered_sorted_by_magnitude():
    before = {"1": 10.0, "2": 50.0, "3": 5.0}
    after = {"1": 10.2, "2": 90.0, "3": 4.0}
    alerts = scan_field_for_sudden_movement(before, after)
    assert [alert.horse_id for alert in alerts] == ["2", "3"]
    assert alerts[0].direction == "drift_up"


def test_scan_field_requires_matching_horse_sets():
    with pytest.raises(ValueError):
        scan_field_for_sudden_movement({"1": 10.0}, {"2": 10.0})
