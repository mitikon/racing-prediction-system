from datetime import datetime, timedelta, timezone
import hashlib

import pytest

from racing_lambda import (
    RacingFutureEvaluation,
    RacingFrozenTrial,
    RacingRecursiveImprovementGate,
    RacingRsiCandidate,
    candidate_manifest_digest,
    freeze_recursive_rsi_candidate,
    freeze_recursive_rsi_report,
    freeze_recursive_rsi_trial,
    parameter_manifest_digest,
    trial_manifest_digest,
    validate_recursive_rsi_successor,
)


UTC = timezone.utc
CREATED = datetime(2026, 9, 14, 0, 0, tzinfo=UTC)
COMMIT = "a" * 40


def candidate(mode="full", **changes):
    params = {"rsi_weight": 0.20}
    values = {
        "mode": mode,
        "candidate_id": f"{mode}-rsi-g1",
        "parent_version": f"{mode}-current",
        "generation": 1,
        "created_at": CREATED,
        "source_commit": COMMIT,
        "parameters": params,
    }
    values.update(changes)
    values.setdefault("parameter_manifest_sha256", parameter_manifest_digest(values["parameters"]))
    return RacingRsiCandidate(**values)


def trial_rows(mode="full"):
    sealed = candidate_manifest_digest(candidate(mode))
    rows = []
    for index in range(8):
        start = CREATED + timedelta(days=index + 1, hours=3)
        rows.append(RacingFrozenTrial(
            mode=mode,
            candidate_id=f"{mode}-rsi-g1",
            race_id=f"R{index}",
            registered_at=start - timedelta(hours=2),
            prediction_frozen_at=start - timedelta(hours=1),
            scheduled_start=start,
            baseline_prediction_sha256=hashlib.sha256(f"baseline-{index}".encode()).hexdigest(),
            candidate_prediction_sha256=hashlib.sha256(f"candidate-{index}".encode()).hexdigest(),
            candidate_manifest_sha256=sealed,
            input_sha256=hashlib.sha256(f"input-{index}".encode()).hexdigest(),
        ))
    return rows


def observations(mode="full", *, better=True):
    sealed = candidate_manifest_digest(candidate(mode))
    rows = []
    for index, trial in enumerate(trial_rows(mode)):
        start = trial.scheduled_start
        rows.append(RacingFutureEvaluation(
            mode=mode,
            candidate_id=f"{mode}-rsi-g1",
            race_id=f"R{index}",
            prediction_frozen_at=start - timedelta(hours=1),
            scheduled_start=start,
            result_known_at=start + timedelta(hours=1),
            baseline_prediction_sha256=hashlib.sha256(f"baseline-{index}".encode()).hexdigest(),
            candidate_prediction_sha256=hashlib.sha256(f"candidate-{index}".encode()).hexdigest(),
            candidate_manifest_sha256=sealed,
            trial_manifest_sha256=trial_manifest_digest(trial),
            baseline_brier_loss=0.20,
            candidate_brier_loss=0.15 if better else 0.25,
            baseline_top3_hits=2,
            candidate_top3_hits=3 if better else 1,
            baseline_recovery_rate=0.90,
            candidate_recovery_rate=1.05 if better else 0.70,
            baseline_max_bug_hit=index < 2,
            candidate_max_bug_hit=index < (3 if better else 1),
        ))
    return rows


@pytest.mark.parametrize("mode", ["full", "simple"])
def test_each_mode_can_propose_only_future_improvement(mode):
    report = RacingRecursiveImprovementGate(mode).evaluate(candidate(mode), observations(mode), trial_rows(mode))
    assert report.status == "PROMOTION_PROPOSED"
    assert all(report.gates.values())
    assert report.to_dict()["human_approval_required"] is True
    assert report.to_dict()["betting_authority"] is False


def test_harmful_candidate_is_rejected():
    report = RacingRecursiveImprovementGate("simple").evaluate(
        candidate("simple"), observations("simple", better=False), trial_rows("simple")
    )
    assert report.status == "REJECTED"
    assert not report.gates["brier_loss_improved"]


def test_post_hoc_candidate_and_mode_mixing_are_blocked():
    rows = observations("full")
    with pytest.raises(ValueError, match="sealed before"):
        RacingRecursiveImprovementGate("full").evaluate(
            candidate("full", created_at=rows[0].prediction_frozen_at),
            rows,
            trial_rows("full"),
        )
    with pytest.raises(ValueError, match="another racing mode"):
        RacingRecursiveImprovementGate("simple").evaluate(candidate("full"), rows, trial_rows("full"))


def test_fixed_core_and_non_allow_listed_changes_are_blocked():
    with pytest.raises(ValueError, match="fixed racing PCA"):
        candidate(recent_correlation_weight=0.20)
    params = {"automatic_betting": True}
    with pytest.raises(ValueError, match="non-allow-listed"):
        candidate(parameters=params, parameter_manifest_sha256=parameter_manifest_digest(params))


def test_candidate_and_report_are_write_once(tmp_path):
    item = candidate()
    report = RacingRecursiveImprovementGate("full").evaluate(item, observations(), trial_rows())
    manifest = freeze_recursive_rsi_candidate(item, tmp_path / "candidate.json")
    proposal = freeze_recursive_rsi_report(report, tmp_path / "proposal.json")
    frozen_trial = freeze_recursive_rsi_trial(trial_rows()[0], tmp_path / "trial.json")
    assert manifest.exists() and proposal.exists() and frozen_trial.exists()
    with pytest.raises(FileExistsError):
        freeze_recursive_rsi_candidate(item, manifest)
    with pytest.raises(FileExistsError):
        freeze_recursive_rsi_trial(trial_rows()[0], frozen_trial)


def test_settlement_cannot_replace_a_frozen_prediction():
    rows = observations()
    row = rows[0]
    rows[0] = RacingFutureEvaluation(
        **{**row.__dict__, "candidate_prediction_sha256": "f" * 64}
    )
    with pytest.raises(ValueError, match="does not match pre-race"):
        RacingRecursiveImprovementGate("full").evaluate(candidate(), rows, trial_rows())


def test_recursive_successor_must_chain_and_remain_in_mode():
    report = RacingRecursiveImprovementGate("full").evaluate(candidate(), observations(), trial_rows())
    params = {"rsi_weight": 0.25}
    child = candidate(
        candidate_id="full-rsi-g2",
        parent_version=report.candidate_id,
        generation=2,
        parent_report_sha256=report.report_sha256,
        parameters=params,
        parameter_manifest_sha256=parameter_manifest_digest(params),
    )
    validate_recursive_rsi_successor(report, child)
    with pytest.raises(ValueError, match="cannot be mixed"):
        validate_recursive_rsi_successor(report, candidate(
            "simple", candidate_id="simple-g2", generation=2,
            parent_version=report.candidate_id,
            parent_report_sha256=report.report_sha256,
        ))
