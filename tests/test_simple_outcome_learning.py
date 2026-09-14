import copy
import json
from pathlib import Path

import pytest

from racing_lambda.simple_outcome_learning import audit_outcomes


DATA = Path(__file__).resolve().parents[1] / "data/verification/2026-09-13-simple-outcomes.json"


def _races():
    return json.loads(DATA.read_text(encoding="utf-8"))["races"]


def test_september_13_audit_counts_exactly_three_excluded_podium_horses():
    races = _races()
    original = copy.deepcopy(races)
    audit = audit_outcomes(races)

    assert races == original  # RESULT cannot rewrite PRE_RACE evidence
    assert audit["total"] == {
        "race_count": 5, "winner_rank1": 1, "winner_in_top5": 4,
        "top3_hits": 6, "top5_hits": 12,
    }
    assert [
        (row["race_id"], row["horse_id"], row["actual_place"], row["frozen_rank"])
        for row in audit["excluded_podium_horses"]
    ] == [
        ("2026-09-13-nakayama-7", "2", 1, 12),
        ("2026-09-13-nakayama-8", "13", 2, 6),
        ("2026-09-13-hanshin-10", "5", 2, 10),
    ]
    assert all(row["cause_status"] == "unverified" for row in audit["excluded_podium_horses"])
    assert len(audit["next_race_training_labels"]) == sum(len(r["pre_race_order"]) for r in races)
    assert sum(row["top3_result_label"] for row in audit["next_race_training_labels"]) == 15
    assert audit["rsi_training"]["performed"] is False
    assert audit["weight_changes"] == []


def test_cannot_mislabel_an_unknown_result_or_duplicate_prediction():
    races = _races()
    races[0]["actual_top3"][0] = "99"
    with pytest.raises(ValueError, match="actual horses"):
        audit_outcomes(races)
    races = _races()
    races[1]["pre_race_order"][5] = "16"
    with pytest.raises(ValueError, match="unique"):
        audit_outcomes(races)


def test_after_race_information_cannot_be_substituted_for_pre_race_evidence():
    races = _races()
    races[4]["notes"]["5"]["causal_explanation"] = "post-hoc"
    with pytest.raises(ValueError, match="separate pre/post-race"):
        audit_outcomes(races)
