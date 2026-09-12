import pytest

from racing_lambda import (
    FrozenRaceCase,
    builtin_backtest_cases_2026_09_06,
    run_frozen_backtest,
)


def test_builtin_backtest_is_chronological_and_matches_frozen_baseline():
    report = run_frozen_backtest(builtin_backtest_cases_2026_09_06())
    assert report.races == 3
    assert report.winner_hits == 2
    assert report.top3_slots_captured == 7
    assert report.top3_slots_total == 9
    assert report.top5_slots_captured == 8
    assert report.top5_slots_total == 15
    assert report.expanded_top3_slots_captured == 8
    assert report.maximum_bug_top3_hits == 0
    assert [row.prior_races_available for row in report.rows] == [0, 1, 2]
    assert report.recovery_rate is None


def test_backtest_rejects_prediction_frozen_at_start():
    with pytest.raises(ValueError, match="frozen before"):
        FrozenRaceCase(
            race_id="leakage",
            frozen_at="2026-09-06T15:45:00+09:00",
            scheduled_start="2026-09-06T15:45:00+09:00",
            overall_top5=("1", "2", "3", "4", "5"),
            expanded_candidates=("1", "2", "3", "4", "5"),
            maximum_bug=None,
            finishing_order=("1", "2", "3", "4", "5"),
        )


def test_backtest_requires_timezone_for_auditable_freeze():
    with pytest.raises(ValueError, match="timezone"):
        FrozenRaceCase(
            race_id="no-timezone",
            frozen_at="2026-09-06T15:00:00",
            scheduled_start="2026-09-06T15:45:00+09:00",
            overall_top5=("1", "2", "3", "4", "5"),
            expanded_candidates=("1", "2", "3", "4", "5"),
            maximum_bug=None,
            finishing_order=("1", "2", "3", "4", "5"),
        )


def test_backtest_calculates_recovery_only_for_settled_bets():
    source = builtin_backtest_cases_2026_09_06()[0]
    settled = FrozenRaceCase(
        race_id=source.race_id,
        frozen_at=source.frozen_at,
        scheduled_start=source.scheduled_start,
        overall_top5=source.overall_top5,
        expanded_candidates=source.expanded_candidates,
        maximum_bug=source.maximum_bug,
        finishing_order=source.finishing_order,
        stake_yen=1_000,
        return_yen=1_250,
    )
    report = run_frozen_backtest((settled,))
    assert report.settled_races == 1
    assert report.total_stake_yen == 1_000
    assert report.total_return_yen == 1_250
    assert report.recovery_rate == 1.25
