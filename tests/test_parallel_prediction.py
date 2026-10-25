import pytest

from racing_lambda.parallel_prediction import (
    ParallelPredictionBundle,
    RaceIntake,
    build_parallel_prediction,
    freeze_parallel_prediction,
    load_parallel_prediction,
)
from racing_lambda.schema import HorseEntry


def _horse(horse_id: str, ability: float = 0.5, pace_fit: float = 0.5) -> HorseEntry:
    return HorseEntry(
        horse_id=horse_id,
        horse_name=f"horse-{horse_id}",
        ability=ability,
        pace_fit=pace_fit,
        course_draw_fit=0.5,
        weight_track_fit=0.5,
        jockey_skill=0.5,
        latest_information=0.5,
    )


def _intake(**overrides) -> RaceIntake:
    defaults = dict(
        race_id="2026-10-25-test-1",
        race_name="3歳上1勝クラス",
        scheduled_start="2026-10-25T15:40:00+09:00",
        frozen_at="2026-10-25T15:10:00+09:00",
        horses=tuple(_horse(str(i), ability=0.9 - i * 0.1) for i in range(1, 7)),
    )
    defaults.update(overrides)
    return RaceIntake(**defaults)


def test_intake_rejects_fewer_than_four_horses():
    with pytest.raises(ValueError, match="at least four horses"):
        _intake(horses=tuple(_horse(str(i)) for i in range(1, 4)))


def test_intake_rejects_duplicate_horse_ids():
    with pytest.raises(ValueError, match="unique"):
        _intake(horses=(_horse("1"), _horse("1"), _horse("2"), _horse("3")))


def test_intake_rejects_partial_running_styles():
    horses = tuple(_horse(str(i)) for i in range(1, 7))
    with pytest.raises(ValueError, match="exactly the same horses"):
        _intake(horses=horses, running_styles={"1": "escape", "2": "closer"})


def test_build_parallel_prediction_without_running_styles_or_odds_skips_those_angles():
    bundle = build_parallel_prediction(_intake())
    assert len(bundle.base_prediction) == 6
    assert bundle.pace_scenario_analysis is None
    assert bundle.pace_adjusted_prediction is None
    assert bundle.edge_candidacy is None
    # Axis formation still gets built, from the base ranking.
    assert bundle.axis_formation.rationale == "base lambda rank 1-4"
    assert bundle.primary_prediction == bundle.base_prediction


def test_build_parallel_prediction_derives_axis_formation_from_primary_ranking():
    bundle = build_parallel_prediction(_intake())
    ranked = sorted(bundle.base_prediction, key=lambda row: row.rank)
    expected_axis = ranked[0].horse_id
    expected_contenders = tuple(row.horse_id for row in ranked[1:4])
    assert bundle.axis_formation.axis_horse_id == expected_axis
    assert bundle.axis_formation.contender_horse_ids == expected_contenders


def test_build_parallel_prediction_with_running_styles_produces_pace_adjusted_ranking():
    horses = tuple(_horse(str(i), ability=0.5, pace_fit=0.5) for i in range(1, 7))
    # Heavy front-runner bias -> fast-pace scenario -> closers should rank higher
    # once pace_fit is the only differentiating input.
    styles = {"1": "escape", "2": "escape", "3": "front", "4": "closer", "5": "closer", "6": "stalk"}
    bundle = build_parallel_prediction(_intake(horses=horses, running_styles=styles))
    assert bundle.pace_scenario_analysis is not None
    assert bundle.pace_scenario_analysis.scenario == "fast"
    assert bundle.pace_adjusted_prediction is not None
    assert bundle.primary_prediction == bundle.pace_adjusted_prediction
    assert bundle.axis_formation.rationale == "pace-adjusted lambda rank 1-4"
    ranks = {row.horse_id: row.rank for row in bundle.pace_adjusted_prediction}
    assert ranks["4"] < ranks["1"]  # closer out-ranks escape under a fast scenario


def test_build_parallel_prediction_with_odds_produces_edge_candidacy():
    bundle = build_parallel_prediction(
        _intake(odds_by_horse={"1": 1.5, "2": 3.0, "3": 5.0, "4": 8.0, "5": 12.0, "6": 20.0})
    )
    assert bundle.edge_candidacy is not None
    assert bundle.edge_candidacy.race_id == "2026-10-25-test-1"
    assert bundle.edge_candidacy.information_tier == "conditional"


def test_same_intake_feeds_every_angle_identically():
    """The entire point: one RaceIntake must drive every active angle without
    re-entering the data, and every angle must agree on race_id/timestamps."""
    bundle = build_parallel_prediction(
        _intake(
            running_styles={
                "1": "escape", "2": "front", "3": "stalk",
                "4": "stalk", "5": "closer", "6": "closer",
            },
            odds_by_horse={"1": 1.5, "2": 3.0, "3": 5.0, "4": 8.0, "5": 12.0, "6": 20.0},
        )
    )
    assert bundle.axis_formation.race_id == bundle.race_id == "2026-10-25-test-1"
    assert bundle.axis_formation.scheduled_start == bundle.scheduled_start
    assert bundle.edge_candidacy.race_id == bundle.race_id
    assert bundle.pace_scenario_analysis is not None


def test_freeze_then_load_round_trips_and_detects_tampering(tmp_path):
    bundle = build_parallel_prediction(
        _intake(odds_by_horse={"1": 1.5, "2": 3.0, "3": 5.0, "4": 8.0, "5": 12.0, "6": 20.0})
    )
    path = freeze_parallel_prediction(tmp_path, bundle)
    loaded = load_parallel_prediction(path)
    assert loaded["race_id"] == bundle.race_id
    assert loaded["axis_formation"]["axis_horse_id"] == bundle.axis_formation.axis_horse_id
    assert len(loaded["base_prediction"]) == 6

    with pytest.raises(FileExistsError, match="immutable"):
        freeze_parallel_prediction(tmp_path, bundle)

    tampered = path.read_text(encoding="utf-8").replace(
        f'"race_id": "{bundle.race_id}"', '"race_id": "tampered"'
    )
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_parallel_prediction(path)
