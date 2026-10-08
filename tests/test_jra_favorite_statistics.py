import pytest

from racing_lambda.jra_favorite_statistics import (
    FAVORITE_OVERALL_PLACE_RATE_RANGE,
    FAVORITE_OVERALL_WIN_RATE_RANGE,
    FavoritePlaceRateReference,
    lookup_favorite_place_rate_reference,
)


def test_short_price_favorite_has_a_high_place_rate_reference():
    reference = lookup_favorite_place_rate_reference(1.2)
    assert isinstance(reference, FavoritePlaceRateReference)
    assert reference.low_estimate >= 0.8
    assert "JRA-VAN" in reference.source_note


def test_mid_price_favorite_has_a_moderate_place_rate_reference():
    reference = lookup_favorite_place_rate_reference(3.5)
    assert reference is not None
    assert reference.low_estimate == pytest.approx(0.50)
    assert reference.high_estimate == pytest.approx(0.50)


def test_band_with_no_sourced_data_returns_none_instead_of_guessing():
    # 2.0-2.9倍帯は1番人気限定の実測出典が見つかっていないため、
    # 補間せずNoneを返す。
    assert lookup_favorite_place_rate_reference(2.5) is None


def test_odds_at_or_below_one_is_rejected():
    with pytest.raises(ValueError):
        lookup_favorite_place_rate_reference(1.0)


def test_overall_reference_ranges_are_plausible_fractions():
    for low, high in (FAVORITE_OVERALL_WIN_RATE_RANGE, FAVORITE_OVERALL_PLACE_RATE_RANGE):
        assert 0.0 < low <= high < 1.0
