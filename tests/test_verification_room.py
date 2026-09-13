from datetime import datetime, timedelta, timezone

import pytest

from racing_lambda.verification_room import (
    VerificationPrediction,
    VerificationResult,
    evaluate_verification,
    freeze_verification_prediction,
    settle_verification_result,
)


def _prediction():
    start = datetime(2026, 9, 13, 15, 45, tzinfo=timezone(timedelta(hours=9)))
    return VerificationPrediction(
        race_id="20260913-NAKAYAMA-11",
        race_name="検証レース",
        venue="中山",
        scheduled_start=start.isoformat(),
        frozen_at=(start - timedelta(minutes=5)).isoformat(),
        top5=("8", "3", "5", "2", "11"),
        full_leading_ranking=("8", "5", "3", "11", "2"),
        simple_leading_ranking=("8", "3", "2", "5", "11"),
        odds_bug_ranking=("11", "8", "5"),
        must_keep_top3=("8",),
        axis_recommendation="8",
        torigami_warning={"trio_box": {"warning": False}},
    )


def test_prediction_requires_exactly_five_unique_horses():
    p = _prediction()
    with pytest.raises(ValueError):
        VerificationPrediction(
            **{**p.__dict__, "top5": ("1", "2", "3", "4", "4")}
        )


def test_prediction_must_be_frozen_before_start():
    p = _prediction()
    with pytest.raises(ValueError):
        VerificationPrediction(
            **{**p.__dict__, "frozen_at": p.scheduled_start}
        )


def test_write_once_prevents_posthoc_prediction_overwrite(tmp_path):
    p = _prediction()
    path = freeze_verification_prediction(tmp_path, p)
    assert path.exists()
    with pytest.raises(FileExistsError):
        freeze_verification_prediction(tmp_path, p)


def test_result_is_separate_and_evaluation_detects_top3_coverage(tmp_path):
    p = _prediction()
    freeze_verification_prediction(tmp_path, p)
    result = VerificationResult(
        race_id=p.race_id,
        finishing_order=("8", "5", "3", "11", "2"),
        settled_at=datetime.now(timezone.utc).isoformat(),
        stake_yen=1000,
        return_yen=1450,
    )
    result_path, eval_path = settle_verification_result(tmp_path, p, result)
    assert result_path.exists()
    assert eval_path.exists()
    evaluation = evaluate_verification(p, result)
    assert evaluation.top5_hits_in_actual_top3 == 3
    assert evaluation.top5_contains_all_top3 is True
    assert evaluation.winner_in_top5 is True
    assert evaluation.recovery_rate == 1.45


def test_missing_actual_top3_is_classified_as_extraction_failure():
    p = _prediction()
    result = VerificationResult(
        race_id=p.race_id,
        finishing_order=("14", "8", "3"),
        settled_at=datetime.now(timezone.utc).isoformat(),
    )
    evaluation = evaluate_verification(p, result)
    assert evaluation.top5_hits_in_actual_top3 == 2
    assert "抽出漏れ" in evaluation.failure_categories
