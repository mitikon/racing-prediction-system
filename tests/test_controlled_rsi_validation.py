from datetime import datetime, timedelta, timezone

import pytest

from racing_lambda import (
    ControlledPrediction,
    ControlledRsiValidationLoop,
    FullLeadingPredictionLambda,
    Going,
    OfficialResult,
    RacingRsiCandidate,
    SimpleHorseFeatures,
    SimpleLeadingSignalLambdaV02,
    SimpleRealtimeWsiSignal,
    SimpleRaceContext,
    TrialMetrics,
    ingest_snapshot,
    parameter_manifest_digest,
)


UTC = timezone.utc
CREATED = datetime(2026, 9, 1, tzinfo=UTC)


def _candidate(mode: str, *, parameters: dict | None = None):
    parameters = parameters if parameters is not None else {"wsi_weight": 0.2}
    return RacingRsiCandidate(
        mode=mode,
        candidate_id=f"{mode}-g1",
        parent_version=f"{mode}-current",
        generation=1,
        created_at=CREATED,
        source_commit="a" * 40,
        parameter_manifest_sha256=parameter_manifest_digest(parameters),
        parameters=parameters,
    )


def _full_history(race_id: str, offset: float = 0.0):
    return [
        ingest_snapshot(
            race_id=race_id,
            phase="PRE_RACE",
            source_url="https://www.jra.go.jp/",
            observed_at=datetime(2026, 9, 9, 3, minute, tzinfo=UTC),
            payload={
                "market_support": [
                    {
                        "horse_id": str(horse),
                        "support": {
                            "win": .08 + horse * .02 + offset
                            + (.0025 * minute if horse == 1 else -.0005 * minute),
                            "place": .12 + horse * .02 + offset
                            + (.0015 * minute if horse == 2 else -.0003 * minute),
                        },
                    }
                    for horse in range(1, 5)
                ]
            },
        )
        for minute in range(25)
    ]


def _full_model(wsi_weight: float = 0.2):
    return FullLeadingPredictionLambda(enabled=True, wsi_weight=wsi_weight).fit_from_jra_history(
        recent_races=[_full_history("RECENT")],
        prior_races=[_full_history("PRIOR", .01)],
        historical_results=[
            OfficialResult("RECENT", ("1", "2", "3", "4")),
            OfficialResult("PRIOR", ("2", "1", "3", "4")),
        ],
    )


def _simple_horses():
    return [
        SimpleHorseFeatures(
            horse_id=str(index), horse_name=f"horse-{index}", odds=2.0 + index,
            age=4, assigned_weight_kg=55.0, body_weight_kg=480,
            body_weight_change_kg=0, predicted_position=index,
            recent_top3_count=2, recent_top5_count=3, class_score=.7,
            going_score=.7, course_score=.7, jockey_place_rate=.2,
            jockey_win_return=1.0, jockey_place_return=1.0,
            days_since_last_run=28,
        )
        for index in range(1, 4)
    ]


def _simple_signals(captured: datetime):
    return [
        SimpleRealtimeWsiSignal(
            horse_id=str(index), top3_probability=.9 - index * .1,
            bug_score=.8 - index * .1, feature_count=12,
            captured_at=captured, trained_until=captured - timedelta(days=1),
        )
        for index in range(1, 4)
    ]


def test_operational_loop_keeps_modes_separate_and_writes_pre_race_first(tmp_path):
    full = ControlledRsiValidationLoop(tmp_path, _candidate("full"))
    simple = ControlledRsiValidationLoop(tmp_path, _candidate("simple"))
    start = CREATED + timedelta(days=2)
    full_trial = full.register_pre_race(
        race_id="R1", registered_at=start - timedelta(hours=2),
        prediction_frozen_at=start - timedelta(hours=1), scheduled_start=start,
        input_payload={"odds": [1, 2]}, baseline_prediction=["1", "2"],
        candidate_prediction=["2", "1"],
    )
    simple_trial = simple.register_pre_race(
        race_id="R1", registered_at=start - timedelta(hours=2),
        prediction_frozen_at=start - timedelta(hours=1), scheduled_start=start,
        input_payload={"odds": [1, 2]}, baseline_prediction=["1", "2"],
        candidate_prediction=["2", "1"],
    )
    assert full_trial.mode == "full" and simple_trial.mode == "simple"
    assert (tmp_path / "full" / "full-g1" / "R1" / "PRE_RACE" / "trial.json").exists()
    assert (tmp_path / "simple" / "simple-g1" / "R1" / "PRE_RACE" / "trial.json").exists()


def test_operational_loop_settles_result_without_rewriting_trial(tmp_path):
    loop = ControlledRsiValidationLoop(tmp_path, _candidate("full"))
    start = CREATED + timedelta(days=2)
    trial = loop.register_pre_race(
        race_id="R1", registered_at=start - timedelta(hours=2),
        prediction_frozen_at=start - timedelta(hours=1), scheduled_start=start,
        input_payload={}, baseline_prediction=["1"], candidate_prediction=["2"],
    )
    evaluation = loop.settle_result(
        trial, result_known_at=start + timedelta(hours=1),
        metrics=TrialMetrics(0.2, 0.1, 2, 3, 0.8, 1.0, False, True),
    )
    assert evaluation.race_id == "R1"
    assert (tmp_path / "full" / "full-g1" / "R1" / "RESULT" / "evaluation.json").exists()


def test_full_official_entrypoint_cannot_return_before_pre_race_freeze(tmp_path):
    loop = ControlledRsiValidationLoop(tmp_path, _candidate("full"))
    baseline = _full_model(.0)
    candidate = _full_model(.2)
    snapshots = _full_history("TARGET", .02)
    start = datetime(2026, 9, 9, 4, tzinfo=UTC)

    output = candidate.score_jra_race(
        snapshots,
        scheduled_start=start,
        validation_loop=loop,
        baseline_model=baseline,
        prediction_frozen_at=start - timedelta(minutes=30),
    )

    assert isinstance(output, ControlledPrediction)
    assert output.mode == "full"
    assert output.official_prediction == baseline.score_jra_race_research(
        snapshots, scheduled_start=start
    )
    assert (tmp_path / "full" / "full-g1" / "TARGET" / "PRE_RACE" / "trial.json").exists()

    with pytest.raises(FileExistsError):
        candidate.score_jra_race(
            snapshots,
            scheduled_start=start,
            validation_loop=loop,
            baseline_model=baseline,
            prediction_frozen_at=start - timedelta(minutes=20),
        )


def test_pre_rename_rsi_weight_candidate_still_validates_against_current_model(tmp_path):
    """A candidate sealed before the rsi_weight -> wsi_weight rename must keep
    working: its hash-locked parameters dict is never rewritten in place, so
    the model's current (wsi_weight-only) attestation must still match it via
    canonicalize_parameter_keys.
    """
    loop = ControlledRsiValidationLoop(tmp_path, _candidate("full", parameters={"rsi_weight": 0.2}))
    baseline = _full_model(.0)
    candidate = _full_model(.2)
    snapshots = _full_history("TARGET", .02)
    start = datetime(2026, 9, 9, 4, tzinfo=UTC)

    output = candidate.score_jra_race(
        snapshots,
        scheduled_start=start,
        validation_loop=loop,
        baseline_model=baseline,
        prediction_frozen_at=start - timedelta(minutes=30),
    )

    assert isinstance(output, ControlledPrediction)
    assert output.mode == "full"


def test_full_official_entrypoint_rejects_snapshot_newer_than_freeze(tmp_path):
    loop = ControlledRsiValidationLoop(tmp_path, _candidate("full"))
    start = datetime(2026, 9, 9, 4, tzinfo=UTC)
    with pytest.raises(ValueError, match="latest snapshot"):
        _full_model().score_jra_race(
            _full_history("TARGET"),
            scheduled_start=start,
            validation_loop=loop,
            baseline_model=_full_model(),
            prediction_frozen_at=datetime(2026, 9, 9, 3, 4, tzinfo=UTC),
        )
    assert not (tmp_path / "full" / "full-g1" / "TARGET").exists()


def test_official_entrypoint_rejects_model_not_matching_frozen_candidate(tmp_path):
    loop = ControlledRsiValidationLoop(tmp_path, _candidate("full"))
    start = datetime(2026, 9, 9, 4, tzinfo=UTC)
    with pytest.raises(ValueError, match="frozen parameter"):
        _full_model(.4).score_jra_race(
            _full_history("TARGET"), scheduled_start=start,
            validation_loop=loop, baseline_model=_full_model(.0),
            prediction_frozen_at=start - timedelta(minutes=30),
        )


def test_full_official_entrypoint_rejects_fake_validation_loop():
    class FakeLoop:
        def run_full_prediction(self, **_kwargs):
            return "unfrozen prediction"

    start = datetime(2026, 9, 9, 4, tzinfo=UTC)
    with pytest.raises(TypeError, match="ControlledRsiValidationLoop"):
        _full_model(.2).score_jra_race(
            _full_history("TARGET"), scheduled_start=start,
            validation_loop=FakeLoop(), baseline_model=_full_model(.0),
            prediction_frozen_at=start - timedelta(minutes=30),
        )


def test_simple_official_entrypoint_returns_only_attested_prediction(tmp_path):
    loop = ControlledRsiValidationLoop(tmp_path, _candidate("simple"))
    baseline = SimpleLeadingSignalLambdaV02()
    candidate = SimpleLeadingSignalLambdaV02(wsi_weight=.2)
    start = CREATED + timedelta(days=2)
    captured = start - timedelta(minutes=5)
    context = SimpleRaceContext(
        race_id="SIMPLE-TARGET", surface="芝", distance_m=1200,
        going=Going.FIRM, opening_week=True, rain=False,
        projected_front_runners=2, scheduled_start=start,
    )

    output = candidate.rank(
        context,
        _simple_horses(),
        _simple_signals(captured),
        captured_at=captured,
        validation_loop=loop,
        baseline_model=baseline,
        prediction_frozen_at=captured,
    )

    assert isinstance(output, ControlledPrediction)
    assert output.mode == "simple"
    assert output.official_prediction == baseline.rank_research(
        context, _simple_horses(), captured_at=captured
    )
    assert (tmp_path / "simple" / "simple-g1" / "SIMPLE-TARGET" / "PRE_RACE" / "trial.json").exists()


def test_simple_official_entrypoint_rejects_capture_after_freeze(tmp_path):
    loop = ControlledRsiValidationLoop(tmp_path, _candidate("simple"))
    start = CREATED + timedelta(days=2)
    context = SimpleRaceContext(
        race_id="SIMPLE-LATE", surface="芝", distance_m=1200,
        going=Going.FIRM, opening_week=True, rain=False,
        projected_front_runners=2, scheduled_start=start,
    )
    with pytest.raises(ValueError, match="capture and freeze"):
        SimpleLeadingSignalLambdaV02(wsi_weight=.2).rank(
            context,
            _simple_horses(),
            captured_at=start - timedelta(minutes=5),
            validation_loop=loop,
            baseline_model=SimpleLeadingSignalLambdaV02(),
            prediction_frozen_at=start - timedelta(minutes=10),
        )
    assert not (tmp_path / "simple" / "simple-g1" / "SIMPLE-LATE").exists()
