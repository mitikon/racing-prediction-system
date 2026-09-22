from datetime import datetime, timedelta, timezone
from hashlib import sha256

import pytest

from racing_lambda.automated_learning_loop import (
    freeze_selected_race,
    settle_and_feedback,
)


START = datetime(2026, 9, 27, 15, 40, tzinfo=timezone(timedelta(hours=9)))


def _prediction():
    return {
        "input_provenance": {"kind": "photo_ocr", "sha256": "a" * 64},
        "prediction": {
            "race_id": "20260927-NAKAYAMA-11",
            "race_name": "検証重賞",
            "venue": "中山",
            "scheduled_start": START.isoformat(),
            "frozen_at": (START - timedelta(minutes=5)).isoformat(),
            "top5": ["8", "3", "5", "2", "11"],
            "simple_leading_ranking": ["8", "3", "5", "2", "11"],
            "odds_bug_ranking": ["11", "8", "5"],
        },
    }


def _result():
    raw = {
        "race_id": "20260927-NAKAYAMA-11",
        "settled_at": (START + timedelta(minutes=10)).isoformat(),
        "source_url": "https://www.jra.go.jp/JRADB/accessS.html",
        "race_time": "1:33.4",
        "finishing_order": ["8", "14", "3", "5", "2", "11"],
        "passing_order": ["2-8-3", "8-3-14", "8-14-3"],
        "final_odds": {"8": 3.1, "14": 22.4, "3": 5.0, "5": 7.8, "2": 9.1, "11": 12.0},
        "field_size": 6,
    }
    raw["source_payload_sha256"] = sha256(str(raw).encode()).hexdigest()
    return {"official_result": raw}


def test_complete_loop_is_append_only_and_creates_next_race_feedback(tmp_path):
    frozen = freeze_selected_race(tmp_path, _prediction())
    before = frozen.read_bytes()
    paths = settle_and_feedback(tmp_path, _prediction(), _result())
    assert frozen.read_bytes() == before
    assert all(path.exists() for path in paths.values())
    feedback = paths["feedback"].read_text(encoding="utf-8")
    assert '"displayed_five_score": "2/5"' in feedback
    assert '"podium_coverage": "2/3"' in feedback
    assert '"effective_from": "next_race_only"' in feedback
    with pytest.raises(FileExistsError):
        settle_and_feedback(tmp_path, _prediction(), _result())


def test_settlement_rejects_prediction_changed_after_freeze(tmp_path):
    freeze_selected_race(tmp_path, _prediction())
    changed = _prediction()
    changed["prediction"]["top5"] = ["8", "14", "3", "5", "2"]
    with pytest.raises(ValueError, match="differs from immutable"):
        settle_and_feedback(tmp_path, changed, _result())


def test_result_requires_all_finishers_and_odds(tmp_path):
    freeze_selected_race(tmp_path, _prediction())
    bad = _result()
    del bad["official_result"]["final_odds"]["11"]
    with pytest.raises(ValueError, match="final_odds"):
        settle_and_feedback(tmp_path, _prediction(), bad)
