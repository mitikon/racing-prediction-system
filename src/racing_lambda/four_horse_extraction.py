"""写真・出馬表・プロンプトを起点とする予想4頭抽出方式。

部分空間正則化PCA（`regularized_pca.py`のC_reg = 0.10*C_recent + 0.90*C_prior）を
中核とする本格・簡易式先行予測λとは完全に独立した、新しい予想方式。

## 位置づけ
- 入力: 出馬表の写真・成績欄の写真、または対象レースを指定したプロンプト
- 予想の生成: 定量式（PCA等）ではなく、写真・プロンプトを読んだ人間またはLLMによる
  直接判断で4頭を抽出する。このモジュール自体はスコアを計算しない。
- このモジュールの役割: その4頭抽出予想を発走前に凍結保存し、結果判明後に
  馬券種別（馬連・ワイド・三連複・三連単）ごとの的中を機械的に判定し、
  「1着4着」のような僅差の的中漏れを定量記録すること。

既存の`verification_room.py`（本格・簡易式λのTop5方式）とは別スキーマ・別ファイル名で
並行運用し、どちらの方式が実戦で成績が良いかを同じレースで比較できるようにする。
本格・簡易式λの既存コード・固定係数は一切変更しない。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

EXTRACTION_METHOD = "photo_prompt_manual_extraction"
EXTRACTION_COUNT = 4


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


@dataclass(frozen=True)
class FourHorseExtraction:
    race_id: str
    race_name: str
    venue: str
    scheduled_start: str
    frozen_at: str
    horses: tuple[str, ...]
    evidence_notes: tuple[str, ...] = ()
    source_document_sha256: tuple[str, ...] = ()
    method: str = EXTRACTION_METHOD
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.race_id.strip():
            raise ValueError("race_id is required")
        if len(self.horses) != EXTRACTION_COUNT or len(set(self.horses)) != EXTRACTION_COUNT:
            raise ValueError(f"horses must contain exactly {EXTRACTION_COUNT} unique picks")
        if self.method != EXTRACTION_METHOD:
            raise ValueError(f"method must remain {EXTRACTION_METHOD!r}")
        start = datetime.fromisoformat(self.scheduled_start)
        frozen = datetime.fromisoformat(self.frozen_at)
        if start.tzinfo is None or frozen.tzinfo is None:
            raise ValueError("scheduled_start and frozen_at must be timezone-aware")
        if frozen >= start:
            raise ValueError("prediction must be frozen before scheduled start")


@dataclass(frozen=True)
class ExtractionResult:
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
class ExtractionEvaluation:
    race_id: str
    winner_captured: bool
    top3_hits: int
    exact_top3_set_hit: bool
    umaren_hit: bool
    wide_hit: bool
    sanrenpuku_hit: bool
    sanrentan_hit: bool
    near_miss_horses: tuple[str, ...]
    failure_categories: tuple[str, ...]
    recovery_rate: float | None


def evaluate_four_horse_extraction(
    prediction: FourHorseExtraction,
    result: ExtractionResult,
) -> ExtractionEvaluation:
    """馬券種別ごとに機械的判定する。

    「1着4着」「3着4着」のような僅差の取りこぼしを、感覚ではなく
    馬連・ワイド・三連複・三連単それぞれの的中有無として記録する。
    """
    if prediction.race_id != result.race_id:
        raise ValueError("prediction/result race_id mismatch")

    predicted_top2 = prediction.horses[:2]
    predicted_top3 = prediction.horses[:3]
    predicted_set = set(prediction.horses)
    actual_top3 = tuple(result.finishing_order[:3])
    actual_top2 = set(result.finishing_order[:2])
    actual_set = set(actual_top3)

    top3_hits = len(set(predicted_top3) & actual_set)
    exact_top3_set_hit = set(predicted_top3) == actual_set
    umaren_hit = set(predicted_top2) == actual_top2
    wide_hit = len(set(predicted_top2) & actual_set) == 2
    sanrentan_hit = predicted_top3 == actual_top3

    failures: list[str] = []
    missing = [horse for horse in actual_top3 if horse not in predicted_set]
    if missing:
        failures.append("抽出漏れ")

    # 僅差の取りこぼし: 予想した4頭のうち券外に外れた馬が、実際には4着
    # （ユーザー報告の「1着4着」「3着4着」パターン）だった場合。3着以内は
    # 捕まえていても券種によっては不的中になるため、通常の的中判定とは
    # 別に「あと一頭分」の精度課題として記録する。
    near_miss = tuple(
        horse
        for horse in prediction.horses
        if horse not in actual_set
        and horse in result.finishing_order
        and result.finishing_order.index(horse) == 3
    )
    if near_miss:
        failures.append("着順僅差の取りこぼし")

    if not exact_top3_set_hit and not missing:
        failures.append("順位配分誤差")
    if any(horse not in actual_set for horse in predicted_top3):
        failures.append("過大評価")
    if prediction.horses[3] in actual_set:
        failures.append("最終除外候補が的中圏")

    recovery = None
    if result.stake_yen is not None and result.return_yen is not None:
        recovery = (
            float(result.return_yen) / float(result.stake_yen)
            if result.stake_yen > 0
            else None
        )

    return ExtractionEvaluation(
        race_id=prediction.race_id,
        winner_captured=result.finishing_order[0] in predicted_set,
        top3_hits=top3_hits,
        exact_top3_set_hit=exact_top3_set_hit,
        umaren_hit=umaren_hit,
        wide_hit=wide_hit,
        sanrenpuku_hit=exact_top3_set_hit,
        sanrentan_hit=sanrentan_hit,
        near_miss_horses=near_miss,
        failure_categories=tuple(dict.fromkeys(failures)),
        recovery_rate=recovery,
    )


def _write_once(path: Path, payload: Mapping[str, Any]) -> Path:
    if path.exists():
        raise FileExistsError(f"immutable four-horse-extraction record already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["checksum_sha256"] = sha256(_canonical(body)).hexdigest()
    with path.open("x", encoding="utf-8") as handle:
        json.dump(body, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return path


def freeze_four_horse_extraction(
    root: str | Path,
    prediction: FourHorseExtraction,
) -> Path:
    return _write_once(
        Path(root) / prediction.race_id / "PRE_RACE" / "four_horse_extraction.json",
        asdict(prediction),
    )


def settle_four_horse_extraction(
    root: str | Path,
    prediction: FourHorseExtraction,
    result: ExtractionResult,
) -> tuple[Path, Path]:
    """RESULTとEVALUATIONを追記専用で保存する。凍結済みの予想は書き換えない。"""
    if prediction.race_id != result.race_id:
        raise ValueError("prediction/result race_id mismatch")
    base = Path(root) / result.race_id
    result_path = _write_once(base / "RESULT" / "four_horse_result.json", asdict(result))
    evaluation = evaluate_four_horse_extraction(prediction, result)
    evaluation_path = _write_once(
        base / "EVALUATION" / "four_horse_evaluation.json",
        asdict(evaluation),
    )
    return result_path, evaluation_path
