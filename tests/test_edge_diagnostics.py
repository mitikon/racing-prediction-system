import math

import pytest

from racing_lambda.edge_diagnostics import (
    GradedRace,
    chance_expected_top3_hits,
    load_graded_races,
    load_odds_by_race,
    odds_favorite_order,
    summarize_vs_chance,
    summarize_vs_odds_baseline,
)


def test_chance_expected_top3_hits_is_the_hypergeometric_mean():
    assert chance_expected_top3_hits(field_size=10, predicted_depth=5) == pytest.approx(1.5)
    assert chance_expected_top3_hits(field_size=12, predicted_depth=3) == pytest.approx(0.75)


def test_graded_race_top3_hits_counts_overlap_within_predicted_depth():
    race = GradedRace(
        race_id="x",
        field_size=10,
        predicted_top5=("1", "2", "3", "4", "5"),
        actual_top3=("5", "6", "7"),
    )
    assert race.top3_hits() == 1
    # A baseline ranking is graded to the same depth lambda itself used.
    assert race.top3_hits(("5", "6", "7", "8", "9")) == 3


def test_graded_race_rejects_malformed_input():
    with pytest.raises(ValueError, match="exactly 3"):
        GradedRace(race_id="x", field_size=10, predicted_top5=("1",), actual_top3=("1", "2"))
    with pytest.raises(ValueError, match="duplicates"):
        GradedRace(race_id="x", field_size=10, predicted_top5=("1", "1"), actual_top3=("1", "2", "3"))


def test_load_graded_races_reads_only_the_two_real_verification_days():
    races = load_graded_races()
    # 2026-09-20 (8 races) + 2026-09-21 (7 races) = 15. 2026-09-13 (a
    # post-hoc virtual exercise) and 2026-09-21-hanshin-08 (missing freeze
    # evidence, listed under non_evaluable_results) must not appear.
    assert len(races) == 15
    race_ids = {race.race_id for race in races}
    assert "2026-09-13-nakayama-7" not in race_ids
    assert "2026-09-21-hanshin-08" not in race_ids
    assert "2026-09-20-nakayama-6" in race_ids
    assert "2026-09-21-hanshin-11" in race_ids


def test_load_graded_races_truncates_full_field_orders_to_published_top5():
    races = {race.race_id: race for race in load_graded_races()}
    # hanshin-07's source file stores the full 18-horse finishing order
    # under pre_race_order; only the first 5 were ever actually published
    # as lambda's Top5, and the diagnostic must grade against that, not
    # the full field (which would trivially inflate the hit rate).
    race = races["2026-09-21-hanshin-07"]
    assert race.predicted_depth == 5
    assert race.predicted_top5 == ("6", "12", "17", "2", "4")


def test_load_graded_races_keeps_a_published_depth_of_three_as_three():
    races = {race.race_id: race for race in load_graded_races()}
    race = races["2026-09-20-hanshin-10"]
    assert race.predicted_top5 == ("3", "5", "12")
    assert race.predicted_depth == 3


def test_real_track_record_matches_the_published_markdown_totals():
    races = load_graded_races()
    total_hits = sum(race.top3_hits() for race in races)
    # docs/racing/validation-2026-09-21.md: "Top5による馬券圏捕捉10/21" (7 races).
    sept_21_hits = sum(race.top3_hits() for race in races if race.race_id.startswith("2026-09-21"))
    assert sept_21_hits == 10
    # The two days combined; this is a fixed historical fact, not a moving target.
    assert total_hits == 26


def test_odds_favorite_order_ranks_shortest_odds_first():
    order = odds_favorite_order({"1": 1.4, "2": 16.3, "3": 11.9})
    assert order == ("1", "3", "2")


def test_load_odds_by_race_only_covers_races_with_a_committed_snapshot():
    odds_by_race = load_odds_by_race()
    assert "2026-09-21-hanshin-11" in odds_by_race
    assert odds_by_race["2026-09-21-hanshin-11"]["1"] == pytest.approx(1.4)
    # Most graded races have no committed full-field odds snapshot yet.
    assert "2026-09-20-nakayama-6" not in odds_by_race


def test_summarize_vs_chance_flags_small_samples_as_insufficient():
    races = load_graded_races()
    result = summarize_vs_chance(races, n_bootstrap=500, seed=0)
    assert result.races == 15
    assert result.sufficient_sample is False
    assert "monitoring readout" in result.note
    # The real excess is positive here, but with n=15 that alone proves nothing;
    # the CI must not be mistaken for a verdict either way.
    assert result.bootstrap_ci_low <= result.mean_excess_per_race <= result.bootstrap_ci_high


def test_summarize_vs_chance_requires_at_least_one_race():
    with pytest.raises(ValueError, match="at least one"):
        summarize_vs_chance([])


def test_summarize_vs_odds_baseline_only_uses_races_with_a_committed_snapshot():
    races = load_graded_races()
    odds_by_race = load_odds_by_race()
    result = summarize_vs_odds_baseline(races, odds_by_race, n_bootstrap=500, seed=0)
    # Only hanshin-09, hanshin-10, hanshin-11, hanshin-12 (2026-09-21) currently
    # have a committed full-field odds snapshot.
    assert result.races == 4
    assert result.sufficient_sample is False


def test_summarize_vs_odds_baseline_raises_when_no_race_has_odds():
    races = load_graded_races()
    with pytest.raises(ValueError, match="no graded race"):
        summarize_vs_odds_baseline(races, {})


def test_bootstrap_ci_is_reproducible_for_a_fixed_seed():
    races = load_graded_races()
    first = summarize_vs_chance(races, n_bootstrap=1000, seed=42)
    second = summarize_vs_chance(races, n_bootstrap=1000, seed=42)
    assert first.bootstrap_ci_low == second.bootstrap_ci_low
    assert first.bootstrap_ci_high == second.bootstrap_ci_high
