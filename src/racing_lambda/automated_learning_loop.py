"""Auditable automation loop for racing prediction and RSI feedback.

The loop deliberately separates PRE_RACE, RESULT and LEARNING.  It does not
scrape JRA pages and it never edits a frozen prediction.  A permitted importer
or a human supplies a normalized official-result payload after the race.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .verification_room import (
    VerificationPrediction,
    VerificationResult,
    evaluate_verification,
    freeze_verification_prediction,
    settle_verification_result,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def _read_json(path: str | Path) -> Mapping[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("input must be a regular non-symlink JSON file")
    if source.suffix.lower() != ".json" or source.stat().st_size > 25_000_000:
        raise ValueError("input must be JSON and no larger than 25 MB")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("JSON root must be an object")
    return value


def _write_once(path: Path, payload: Mapping[str, Any]) -> Path:
    if path.exists():
        raise FileExistsError(f"immutable loop record already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["checksum_sha256"] = sha256(_canonical(body)).hexdigest()
    with path.open("x", encoding="utf-8") as handle:
        json.dump(body, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return path


@dataclass(frozen=True)
class OfficialRaceResult:
    race_id: str
    settled_at: str
    source_url: str
    race_time: str
    finishing_order: tuple[str, ...]
    passing_order: tuple[str, ...]
    final_odds: Mapping[str, float]
    field_size: int
    source_payload_sha256: str

    def __post_init__(self) -> None:
        when = datetime.fromisoformat(self.settled_at)
        if when.tzinfo is None or when.utcoffset() is None:
            raise ValueError("settled_at must be timezone-aware")
        if not self.source_url.startswith("https://"):
            raise ValueError("official result source must use HTTPS")
        if self.field_size < 5 or len(self.finishing_order) != self.field_size:
            raise ValueError("finishing_order must contain every starter")
        if len(set(self.finishing_order)) != len(self.finishing_order):
            raise ValueError("finishing_order must contain unique horses")
        if set(self.final_odds) != set(self.finishing_order):
            raise ValueError("final_odds must cover every finisher")
        if len(self.source_payload_sha256) != 64:
            raise ValueError("source_payload_sha256 must be a SHA-256 digest")


def prediction_from_payload(payload: Mapping[str, Any]) -> VerificationPrediction:
    """Normalize photo-OCR or prompt extraction without making it authoritative."""
    prediction = payload.get("prediction", payload)
    if not isinstance(prediction, Mapping):
        raise ValueError("prediction must be an object")
    provenance = payload.get("input_provenance", {})
    if provenance and not isinstance(provenance, Mapping):
        raise ValueError("input_provenance must be an object")
    source_note = str(prediction.get("source_note", "競馬予想検証室"))
    if provenance:
        kind = str(provenance.get("kind", "prompt"))
        digest = str(provenance.get("sha256", ""))
        if kind not in {"photo_ocr", "prompt", "normalized_json"}:
            raise ValueError("unsupported input provenance kind")
        if digest and len(digest) != 64:
            raise ValueError("input provenance sha256 must be a SHA-256 digest")
        source_note = f"{source_note}; input={kind}; sha256={digest or 'not-supplied'}"
    return VerificationPrediction(
        race_id=str(prediction["race_id"]),
        race_name=str(prediction["race_name"]),
        venue=str(prediction["venue"]),
        scheduled_start=str(prediction["scheduled_start"]),
        frozen_at=str(prediction["frozen_at"]),
        top5=tuple(str(x) for x in prediction["top5"]),
        full_leading_ranking=tuple(str(x) for x in prediction.get("full_leading_ranking", ())),
        simple_leading_ranking=tuple(str(x) for x in prediction.get("simple_leading_ranking", ())),
        odds_bug_ranking=tuple(str(x) for x in prediction.get("odds_bug_ranking", ())),
        must_keep_top3=tuple(str(x) for x in prediction.get("must_keep_top3", ())),
        axis_recommendation=prediction.get("axis_recommendation"),
        torigami_warning=prediction.get("torigami_warning"),
        source_note=source_note,
        schema_version=2,
    )


def official_result_from_payload(payload: Mapping[str, Any]) -> OfficialRaceResult:
    raw = payload.get("official_result", payload)
    if not isinstance(raw, Mapping):
        raise ValueError("official_result must be an object")
    source_digest = str(raw.get("source_payload_sha256") or sha256(_canonical(raw)).hexdigest())
    return OfficialRaceResult(
        race_id=str(raw["race_id"]),
        settled_at=str(raw["settled_at"]),
        source_url=str(raw["source_url"]),
        race_time=str(raw["race_time"]),
        finishing_order=tuple(str(x) for x in raw["finishing_order"]),
        passing_order=tuple(str(x) for x in raw["passing_order"]),
        final_odds={str(k): float(v) for k, v in raw["final_odds"].items()},
        field_size=int(raw["field_size"]),
        source_payload_sha256=source_digest,
    )


def freeze_selected_race(root: str | Path, payload: Mapping[str, Any]) -> Path:
    """Freeze the displayed five horses before the scheduled start."""
    return freeze_verification_prediction(root, prediction_from_payload(payload))


def settle_and_feedback(
    root: str | Path,
    prediction_payload: Mapping[str, Any],
    result_payload: Mapping[str, Any],
) -> dict[str, Path]:
    """Settle one race and emit append-only next-race RSI learning evidence."""
    prediction = prediction_from_payload(prediction_payload)
    official = official_result_from_payload(result_payload)
    if prediction.race_id != official.race_id:
        raise ValueError("prediction/result race_id mismatch")
    start = datetime.fromisoformat(prediction.scheduled_start)
    settled = datetime.fromisoformat(official.settled_at)
    if settled < start:
        raise ValueError("official result cannot predate scheduled start")

    base = Path(root) / prediction.race_id
    frozen_path = base / "PRE_RACE" / "prediction.json"
    if not frozen_path.exists():
        raise FileNotFoundError("PRE_RACE must be frozen before RESULT processing")
    stored = json.loads(frozen_path.read_text(encoding="utf-8"))
    stored_checksum = stored.pop("checksum_sha256", None)
    if stored_checksum != sha256(_canonical(stored)).hexdigest():
        raise ValueError("frozen PRE_RACE checksum mismatch")
    if stored_checksum != sha256(_canonical(asdict(prediction))).hexdigest():
        raise ValueError("supplied prediction differs from immutable PRE_RACE")

    official_path = _write_once(base / "RESULT" / "jra_official.json", asdict(official))
    result = VerificationResult(
        race_id=official.race_id,
        finishing_order=official.finishing_order,
        settled_at=official.settled_at,
    )
    result_path, evaluation_path = settle_verification_result(root, prediction, result)
    evaluation = evaluate_verification(prediction, result)
    actual_top3 = set(official.finishing_order[:3])
    top5_labels = [
        {
            "horse_id": horse_id,
            "display_rank": rank,
            "top3_result_label": int(horse_id in actual_top3),
            "finish_position": official.finishing_order.index(horse_id) + 1,
            "final_odds": official.final_odds[horse_id],
        }
        for rank, horse_id in enumerate(prediction.top5, 1)
    ]
    feedback = {
        "schema": "racing-rsi-feedback-v1",
        "race_id": prediction.race_id,
        "prediction_checksum_sha256": stored_checksum,
        "official_result_checksum_sha256": sha256(_canonical(asdict(official))).hexdigest(),
        "effective_from": "next_race_only",
        "posthoc_prediction_mutation": False,
        "displayed_five_score": f"{evaluation.top5_hits_in_actual_top3}/5",
        "podium_coverage": f"{evaluation.top5_hits_in_actual_top3}/3",
        "winner_hit": evaluation.winner_in_top5,
        "all_podium_in_displayed_five": evaluation.top5_contains_all_top3,
        "failure_categories": list(evaluation.failure_categories),
        "labels": top5_labels,
        "training_gate": {
            "automatic_feedback_recorded": True,
            "parameter_update_performed": False,
            "reason": "候補世代は8レース以上の将来検証と人間承認後のみ昇格可能",
        },
    }
    feedback_path = _write_once(base / "LEARNING" / "rsi_feedback.json", feedback)
    return {
        "official_result": official_path,
        "result": result_path,
        "evaluation": evaluation_path,
        "feedback": feedback_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="競馬予想→結果→RSIフィードバック自動ループ")
    parser.add_argument("--root", type=Path, default=Path("data/automated_loop"))
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="発走前予想を変更不能で保存")
    freeze.add_argument("prediction", type=Path)
    settle = commands.add_parser("settle", help="公式結果を照合し次回学習証拠を生成")
    settle.add_argument("prediction", type=Path)
    settle.add_argument("result", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        print(freeze_selected_race(args.root, _read_json(args.prediction)))
    else:
        paths = settle_and_feedback(
            args.root, _read_json(args.prediction), _read_json(args.result)
        )
        print(json.dumps({k: str(v) for k, v in paths.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
