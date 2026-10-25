import pytest

from racing_lambda.axis_formation import (
    AxisFormationPrediction,
    axis_hit,
    chance_sanrenpuku_formation_hit_probability,
    chance_wide_or_umaren_hit_probability,
    evaluate_axis_formation,
    freeze_axis_formation,
    load_axis_formation,
    sanrenpuku_formation_hit,
    wide_or_umaren_hit,
)


def _prediction(**overrides):
    defaults = dict(
        race_id="2026-10-25-test-1",
        scheduled_start="2026-10-25T15:40:00+09:00",
        frozen_at="2026-10-25T15:10:00+09:00",
        axis_horse_id="1",
        contender_horse_ids=("2", "3", "4"),
    )
    defaults.update(overrides)
    return AxisFormationPrediction(**defaults)


def test_rejects_axis_also_listed_as_contender():
    with pytest.raises(ValueError, match="must not also be a contender"):
        _prediction(axis_horse_id="2", contender_horse_ids=("2", "3", "4"))


def test_rejects_duplicate_contenders():
    with pytest.raises(ValueError, match="exactly three unique"):
        _prediction(contender_horse_ids=("2", "2", "4"))


def test_rejects_prediction_frozen_after_post_time():
    with pytest.raises(ValueError, match="before scheduled start"):
        _prediction(frozen_at="2026-10-25T15:50:00+09:00")


def test_rejects_naive_datetimes():
    with pytest.raises(ValueError, match="timezone-aware"):
        _prediction(scheduled_start="2026-10-25T15:40:00")


def test_all_four_includes_axis_first():
    prediction = _prediction()
    assert prediction.all_four == ("1", "2", "3", "4")


def test_axis_hit_true_when_axis_places():
    prediction = _prediction()
    assert axis_hit(prediction, ["5", "1", "6"]) is True
    assert axis_hit(prediction, ["5", "6", "7"]) is False


def test_sanrenpuku_formation_hit_requires_axis_and_full_top3_coverage():
    prediction = _prediction()  # axis=1, contenders={2,3,4}
    # Axis places, and the other two finishers are both contenders: hit.
    assert sanrenpuku_formation_hit(prediction, ["1", "2", "3"]) is True
    # Axis places, but one finisher (9) is outside the four picked horses: miss.
    assert sanrenpuku_formation_hit(prediction, ["1", "2", "9"]) is False
    # Axis itself misses the podium entirely: miss, even if all three
    # finishers happen to be contenders (no ticket pays without the axis).
    assert sanrenpuku_formation_hit(prediction, ["2", "3", "4"]) is False


def test_wide_or_umaren_hit_only_needs_axis_plus_one_contender():
    prediction = _prediction()
    assert wide_or_umaren_hit(prediction, ["1", "2", "9"]) is True
    assert wide_or_umaren_hit(prediction, ["1", "8", "9"]) is False
    assert wide_or_umaren_hit(prediction, ["2", "3", "4"]) is False


def test_evaluate_axis_formation_rejects_malformed_actual_top3():
    prediction = _prediction()
    with pytest.raises(ValueError, match="exactly three unique"):
        evaluate_axis_formation(prediction, ["1", "1", "2"])


def test_evaluate_axis_formation_reports_all_three_hit_flags_together():
    prediction = _prediction()
    report = evaluate_axis_formation(prediction, ["1", "2", "3"])
    assert report.axis_hit is True
    assert report.sanrenpuku_formation_hit is True
    assert report.wide_or_umaren_hit is True
    assert report.actual_top3 == ("1", "2", "3")


def test_chance_sanrenpuku_probability_at_minimum_field_size_is_three_quarters():
    # With exactly 4 runners, "axis + 3 contenders" is the whole field, so
    # the top-3 is trivially covered; only which of the 4 is "axis" matters.
    assert chance_sanrenpuku_formation_hit_probability(4) == pytest.approx(0.75)


def test_chance_sanrenpuku_probability_decreases_as_field_grows():
    small = chance_sanrenpuku_formation_hit_probability(8)
    large = chance_sanrenpuku_formation_hit_probability(18)
    assert small > large > 0


def test_chance_sanrenpuku_probability_rejects_tiny_fields():
    with pytest.raises(ValueError, match="at least 4"):
        chance_sanrenpuku_formation_hit_probability(3)


def test_chance_wide_or_umaren_probability_is_higher_than_sanrenpuku():
    # Wide/umaren only needs axis + 1 of 3 contenders, a strictly easier
    # bar than sanrenpuku's "axis + both remaining podium horses covered".
    field_size = 14
    wide = chance_wide_or_umaren_hit_probability(field_size)
    sanrenpuku = chance_sanrenpuku_formation_hit_probability(field_size)
    assert wide > sanrenpuku


def test_chance_wide_or_umaren_probability_handles_minimum_field_size():
    # field_size=4: remaining=3, so "none of 3 contenders hit" is
    # impossible (only 1 non-podium horse remains) - must not raise.
    probability = chance_wide_or_umaren_hit_probability(4)
    assert 0.0 < probability <= 1.0


def test_freeze_then_load_round_trips_and_rejects_overwrite(tmp_path):
    prediction = _prediction()
    path = freeze_axis_formation(tmp_path, prediction)
    loaded = load_axis_formation(path)
    assert loaded == prediction

    with pytest.raises(FileExistsError, match="immutable"):
        freeze_axis_formation(tmp_path, prediction)


def test_load_axis_formation_detects_tampering(tmp_path):
    prediction = _prediction()
    path = freeze_axis_formation(tmp_path, prediction)
    payload = path.read_text(encoding="utf-8").replace('"axis_horse_id": "1"', '"axis_horse_id": "9"')
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_axis_formation(path)
