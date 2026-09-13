"""競馬予想検証室 ↔ GitHub 実戦検証ループ。

目的:
1. 発走前の予想を固定保存する
2. 同一レースの結果で予想を書き換えない
3. 結果確定後に評価レコードを追加保存する
4. 改善材料は次レース以降にのみ使用する

このモジュールはChatGPT/人間が作成した予想結果を監査可能なJSONとして
GitHubへ保存するための境界であり、予想ロジック自体は変更しない。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


@dataclass(frozen=True)
class VerificationPrediction:
    race_id: str
    race_name: str
    venue: str
    scheduled_start: str
    frozen_at: str
    top5: tuple[str, ...]
    full_leading_ranking: tuple[str, ...] = ()
    simple_leading_ranking: tuple[str, ...] = ()
    odds_bug_ranking: tuple[str, ...] = ()
    must_keep_top3: tuple[str, ...] = ()
    axis_recommendation: str | None = None
    torigami_warning: Mapping[str, Any] | None = None
    source_note: str = "競馬予想検証室"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.race_id.strip():
            raise ValueError("race_id is required")
        if len(self.top5) != 5 or len(set(self.top5)) != 5:
            raise ValueError("top5 must contain exactly five unique horses")
        start = datetime.fromisoformat(self.scheduled_start)
        frozen = datetime.fromisoformat(self.frozen_at)
        if start.tzinfo is None or frozen.tzinfo is None:
            raise ValueError("scheduled_start and frozen_at must be timezone-aware")
        if frozen >= start:
            raise ValueError("prediction must be frozen before scheduled start")


@dataclass(frozen=True)
class VerificationResult:
    race_id: str
    finishing_order: tuple[str, ...]
    settled_at: str
    payouts: Mapping[str, float] | None = None
    stake_yen: int | None = None
    return_yen: int | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.race_id.strip():
            raise ValueError("race_id is required")
        if len(self.finishing_order) < 3:
            raise ValueError("finishing_order requires at least top3")
        settled = datetime.fromisoformat(self.settled_at)
        if settled.tzinfo is None:
            raise ValueError("settled_at must be timezone-aware")
        if self.stake_yen is not None and self.stake_yen < 0:
            raise ValueError("stake_yen cannot be negative")
        if self.return_yen is not None and self.return_yen < 0:
            raise ValueError("return_yen cannot be negative")


@dataclass(frozen=True)
class VerificationEvaluation:
    race_id: str
    top5_hits_in_actual_top3: int
    top5_contains_all_top3: bool
    top3_exact_set_hit: bool
    winner_in_top5: bool
    must_keep_top3_hits: int
    failure_categories: tuple[str, ...]
    recovery_rate: float | None


def evaluate_verification(
    prediction: VerificationPrediction,
    result: VerificationResult,
) -> VerificationEvaluation:
    if prediction.race_id != result.race_id:
        raise ValueError("prediction/result race_id mismatch")
    actual_top3 = tuple(result.finishing_order[:3])
    actual_set = set(actual_top3)
    predicted_set = set(prediction.top5)
    hits = len(predicted_set & actual_set)
    top3_exact_set_hit = set(prediction.top5[:3]) == actual_set
    failures: list[str] = []

    missing = [horse for horse in actual_top3 if horse not in predicted_set]
    if missing:
        failures.append("抽出漏れ")
    if not top3_exact_set_hit and not missing:
        failures.append("順位配分誤差")
    predicted_top3 = prediction.top5[:3]
    if any(horse not in actual_set for horse in predicted_top3):
        failures.append("過大評価")
    if any(horse in actual_set for horse in prediction.top5[3:]):
        failures.append("最終除外")
    if prediction.must_keep_top3:
        warned = set(prediction.must_keep_top3)
        missed_warning = [horse for horse in warned if horse not in actual_set]
        if missed_warning:
            failures.append("3着以内濃厚判定過大")

    recovery = None
    if result.stake_yen is not None and result.return_yen is not None:
        recovery = (
            float(result.return_yen) / float(result.stake_yen)
            if result.stake_yen > 0
            else None
        )

    return VerificationEvaluation(
        race_id=prediction.race_id,
        top5_hits_in_actual_top3=hits,
        top5_contains_all_top3=hits == 3,
        top3_exact_set_hit=top3_exact_set_hit,
        winner_in_top5=result.finishing_order[0] in predicted_set,
        must_keep_top3_hits=len(set(prediction.must_keep_top3) & actual_set),
        failure_categories=tuple(dict.fromkeys(failures)),
        recovery_rate=recovery,
    )


def _write_once(path: Path, payload: Mapping[str, Any]) -> Path:
    if path.exists():
        raise FileExistsError(f"immutable verification record already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["checksum_sha256"] = sha256(_canonical(body)).hexdigest()
    with path.open("x", encoding="utf-8") as handle:
        json.dump(body, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return path


def freeze_verification_prediction(
    root: str | Path,
    prediction: VerificationPrediction,
) -> Path:
    return _write_once(
        Path(root) / prediction.race_id / "PRE_RACE" / "prediction.json",
        asdict(prediction),
    )


def settle_verification_result(
    root: str | Path,
    prediction: VerificationPrediction,
    result: VerificationResult,
) -> tuple[Path, Path]:
    """Append RESULT and EVALUATION without touching the frozen prediction."""
    if prediction.race_id != result.race_id:
        raise ValueError("prediction/result race_id mismatch")
    base = Path(root) / result.race_id
    result_path = _write_once(base / "RESULT" / "result.json", asdict(result))
    evaluation = evaluate_verification(prediction, result)
    evaluation_path = _write_once(
        base / "EVALUATION" / "evaluation.json",
        asdict(evaluation),
    )
    return result_path, evaluation_path
