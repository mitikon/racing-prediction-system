from datetime import datetime, timedelta, timezone

import pytest

from racing_lambda import (
    Going,
    RacingRsiCandidate,
    SimpleHorseFeatures,
    SimpleRaceContext,
    SimpleRealtimeRsiSignal,
    TrialMetrics,
    approve_promotion,
    approved_parameters,
)
from racing_lambda.recursive_runtime import (
    DEFAULT_MIN_EARLY_REJECTION_RACES,
    DEFAULT_MIN_FUTURE_RACES,
    PARALLEL_CANDIDATES,
    apply_human_approved_promotion,
    bootstrap_state,
    ensure_candidate_slots,
    evaluate_and_rotate_candidates,
    load_state,
    register_simple_race_trials,
    settle_simple_race_result,
    write_state,
)


UTC = timezone.utc
COMMIT = "a" * 40
CREATED = datetime(2026, 9, 15, tzinfo=UTC)


def _context(race_id: str, start: datetime) -> SimpleRaceContext:
    return SimpleRaceContext(
        race_id=race_id, surface="芝", distance_m=1600, going=Going.FIRM,
        opening_week=False, rain=False, projected_front_runners=2,
        scheduled_start=start,
    )


def _horses():
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


def _signals(captured: datetime):
    return [
        SimpleRealtimeRsiSignal(
            horse_id=str(index), top3_probability=.9 - index * .1,
            bug_score=.8 - index * .1, feature_count=12,
            captured_at=captured, trained_until=captured - timedelta(days=1),
        )
        for index in range(1, 4)
    ]


def _register_and_settle_round(state, root, index, *, candidate_losses, top3=2):
    """Register + settle one race across every parallel slot."""
    start = CREATED + timedelta(days=index + 1, hours=6)
    captured = start - timedelta(minutes=5)
    context = _context(f"R{index}", start)
    horses = _horses()
    register_simple_race_trials(
        state, root,
        context=context, horses=horses,
        baseline_rsi_signals=_signals(captured), candidate_rsi_signals=_signals(captured),
        captured_at=captured, prediction_frozen_at=captured,
    )
    metrics = [
        TrialMetrics(0.20, loss, top3, top3, 0.90, 0.90, False, False)
        for loss in candidate_losses
    ]
    evaluations = settle_simple_race_result(
        state, root, result_known_at=start + timedelta(hours=1), metrics_by_slot=metrics,
    )
    assert len(evaluations) == len(state["candidate_slots"])


def test_state_is_hash_chained_and_tampering_is_rejected(tmp_path):
    state = bootstrap_state("simple", {"rsi_weight": 0.0}, COMMIT, CREATED)
    first = write_state(state, tmp_path / "state.json")
    loaded = load_state(first)
    assert loaded["active_model"]["generation"] == 0
    assert loaded["mode"] == "simple"
    assert len(loaded["candidate_slots"]) == PARALLEL_CANDIDATES
    assert all(slot["candidate"]["generation"] == 1 for slot in loaded["candidate_slots"])

    import json

    payload = json.loads(first.read_text())
    payload["active_model"]["parameters"]["rsi_weight"] = 0.9
    first.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="state hash mismatch"):
        load_state(first)


def test_ensure_candidate_slots_tops_up_without_disturbing_existing_slots():
    state = bootstrap_state("simple", {"rsi_weight": 0.0}, COMMIT, CREATED, parallel_candidates=1)
    assert len(state["candidate_slots"]) == 1
    kept = state["candidate_slots"][0]["candidate"]["candidate_id"]
    ensure_candidate_slots(state, COMMIT, CREATED, parallel_candidates=PARALLEL_CANDIDATES)
    assert len(state["candidate_slots"]) == PARALLEL_CANDIDATES
    assert state["candidate_slots"][0]["candidate"]["candidate_id"] == kept


def test_parallel_candidates_are_evaluated_independently_after_full_window(tmp_path):
    state = bootstrap_state("simple", {"rsi_weight": 0.0}, COMMIT, CREATED)
    starting_ids = [slot["candidate"]["candidate_id"] for slot in state["candidate_slots"]]
    # Slot 0 clearly better, slot 1 clearly worse, slot 2 ambiguous.
    for index in range(DEFAULT_MIN_FUTURE_RACES):
        _register_and_settle_round(state, tmp_path, index, candidate_losses=[0.05, 0.35, 0.20])

    results = evaluate_and_rotate_candidates(state, tmp_path, COMMIT, CREATED + timedelta(days=30))
    assert len(results) == 3
    statuses = {entry["candidate_id"]: entry["status"] for entry in results}
    assert statuses[starting_ids[0]] == "PROMOTION_PROPOSED"
    assert statuses[starting_ids[1]] == "REJECTED"

    # active_model must never be touched by search alone: promotion needs a
    # separately recorded human approval.
    assert state["active_model"]["generation"] == 0
    assert state["active_model"]["parameters"] == {"rsi_weight": 0.0}

    for slot in state["candidate_slots"]:
        assert slot["candidate"]["candidate_id"] not in starting_ids
        # active_model is still generation 0 (no autonomous promotion), so
        # every reseeded candidate is another attempt at generation 1.
        assert slot["candidate"]["generation"] == 1
        assert slot["trials"] == []
        assert slot["evaluations"] == []
        assert slot["pending_trial"] is None

    assert (tmp_path / "simple" / starting_ids[0] / "PROMOTION" / "proposal.json").exists()
    assert (tmp_path / "simple" / starting_ids[1] / "PROMOTION" / "proposal.json").exists()


def test_clearly_worse_candidate_is_abandoned_before_the_full_window(tmp_path):
    state = bootstrap_state("simple", {"rsi_weight": 0.0}, COMMIT, CREATED)
    starting_ids = [slot["candidate"]["candidate_id"] for slot in state["candidate_slots"]]

    for index in range(DEFAULT_MIN_EARLY_REJECTION_RACES):
        _register_and_settle_round(state, tmp_path, index, candidate_losses=[0.40, 0.40, 0.40])
        if index < DEFAULT_MIN_EARLY_REJECTION_RACES - 1:
            assert evaluate_and_rotate_candidates(state, tmp_path, COMMIT, CREATED) == []

    results = evaluate_and_rotate_candidates(state, tmp_path, COMMIT, CREATED + timedelta(days=10))
    assert len(results) == 3
    assert all(entry["status"] == "EARLY_REJECTED" for entry in results)
    assert all(entry["evaluated_races"] == DEFAULT_MIN_EARLY_REJECTION_RACES for entry in results)
    assert state["active_model"]["generation"] == 0
    for slot in state["candidate_slots"]:
        assert slot["candidate"]["candidate_id"] not in starting_ids
        assert slot["candidate"]["generation"] == 1


def test_ambiguous_candidates_keep_accumulating_past_the_early_floor(tmp_path):
    state = bootstrap_state("simple", {"rsi_weight": 0.0}, COMMIT, CREATED)
    noisy_losses = [0.10, 0.28, 0.14, 0.24, 0.11, 0.23]
    for index, loss in enumerate(noisy_losses):
        _register_and_settle_round(state, tmp_path, index, candidate_losses=[loss, loss, loss])

    assert evaluate_and_rotate_candidates(state, tmp_path, COMMIT, CREATED + timedelta(days=10)) == []
    assert all(len(slot["evaluations"]) == len(noisy_losses) for slot in state["candidate_slots"])


def test_register_rejects_when_a_trial_is_already_pending(tmp_path):
    state = bootstrap_state("simple", {"rsi_weight": 0.0}, COMMIT, CREATED)
    start = CREATED + timedelta(days=1, hours=6)
    captured = start - timedelta(minutes=5)
    register_simple_race_trials(
        state, tmp_path, context=_context("R0", start), horses=_horses(),
        baseline_rsi_signals=_signals(captured), candidate_rsi_signals=_signals(captured),
        captured_at=captured, prediction_frozen_at=captured,
    )
    with pytest.raises(ValueError, match="already awaiting"):
        next_start = start + timedelta(days=1)
        next_captured = next_start - timedelta(minutes=5)
        register_simple_race_trials(
            state, tmp_path, context=_context("R1", next_start), horses=_horses(),
            baseline_rsi_signals=_signals(next_captured), candidate_rsi_signals=_signals(next_captured),
            captured_at=next_captured, prediction_frozen_at=next_captured,
        )


def test_full_mode_state_rejects_simple_only_helpers(tmp_path):
    state = bootstrap_state("full", {"rsi_weight": 0.25}, COMMIT, CREATED)
    with pytest.raises(ValueError, match="simple-mode"):
        register_simple_race_trials(
            state, tmp_path, context=_context("R0", CREATED + timedelta(days=1)), horses=_horses(),
            baseline_rsi_signals=None, candidate_rsi_signals=None,
            captured_at=CREATED, prediction_frozen_at=CREATED,
        )


def test_apply_human_approved_promotion_is_the_only_path_into_active_model(tmp_path):
    state = bootstrap_state("simple", {"rsi_weight": 0.0}, COMMIT, CREATED)
    winning_id = state["candidate_slots"][0]["candidate"]["candidate_id"]
    for index in range(DEFAULT_MIN_FUTURE_RACES):
        _register_and_settle_round(state, tmp_path, index, candidate_losses=[0.05, 0.30, 0.30])
    results = evaluate_and_rotate_candidates(state, tmp_path, COMMIT, CREATED + timedelta(days=30))
    proposed = next(entry for entry in results if entry["candidate_id"] == winning_id)
    assert proposed["status"] == "PROMOTION_PROPOSED"

    # Rehydrate the frozen report and candidate manifests to complete the
    # human-approval chain, exactly as an offline reviewer would.
    import json

    from racing_lambda.recursive_runtime import _candidate_from_payload
    from racing_lambda.recursive_self_improvement import RacingPromotionReport

    proposal_path = tmp_path / "simple" / winning_id / "PROMOTION" / "proposal.json"
    stored = json.loads(proposal_path.read_text(encoding="utf-8"))
    report = RacingPromotionReport(
        status=stored["status"], mode=stored["mode"], candidate_id=stored["candidate_id"],
        generation=stored["generation"], evaluated_races=stored["evaluated_races"],
        baseline_mean_brier_loss=stored["baseline_mean_brier_loss"],
        candidate_mean_brier_loss=stored["candidate_mean_brier_loss"],
        loss_improvement=stored["loss_improvement"],
        baseline_top3_hits=stored["baseline_top3_hits"], candidate_top3_hits=stored["candidate_top3_hits"],
        baseline_mean_recovery_rate=stored["baseline_mean_recovery_rate"],
        candidate_mean_recovery_rate=stored["candidate_mean_recovery_rate"],
        baseline_max_bug_hits=stored["baseline_max_bug_hits"],
        candidate_max_bug_hits=stored["candidate_max_bug_hits"],
        gates=stored["gates"], candidate_manifest_sha256=stored["candidate_manifest_sha256"],
        report_sha256=stored["report_sha256"],
    )
    candidate_path = tmp_path / "simple" / winning_id / "CANDIDATE" / "candidate.json"
    sealed = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate = _candidate_from_payload(sealed)

    approval = approve_promotion(
        report, candidate, approver="human-reviewer", approved_at=CREATED + timedelta(days=40)
    )
    assert approved_parameters(candidate, report, approval) == candidate.parameters

    apply_human_approved_promotion(state, candidate, report, approval)
    assert state["active_model"]["generation"] == 1
    assert state["active_model"]["model_id"] == winning_id
    assert state["active_model"]["parameters"] == candidate.parameters
    assert state["active_model"]["promotion_report_sha256"] == report.report_sha256


def test_apply_human_approved_promotion_rejects_mode_mismatch():
    state = bootstrap_state("simple", {"rsi_weight": 0.0}, COMMIT, CREATED)
    from racing_lambda import parameter_manifest_digest

    params = {"rsi_weight": 0.5}
    full_candidate = RacingRsiCandidate(
        mode="full", candidate_id="full-rsi-g1-a1", parent_version="full-current", generation=1,
        created_at=CREATED, source_commit=COMMIT,
        parameter_manifest_sha256=parameter_manifest_digest(params), parameters=params,
    )
    with pytest.raises(ValueError, match="another racing mode"):
        apply_human_approved_promotion(state, full_candidate, None, None)  # type: ignore[arg-type]
