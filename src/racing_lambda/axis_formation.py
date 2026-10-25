"""軸1頭+相手3頭フォーメーション - Top5ランキングとは別の予想構築方式。

Top5ランキング(full_leading_prediction_lambda.py等)は「1〜3着をどれだけ
網羅できたか」という再現率を測る形式だが、実際の馬券(三連複・ワイドの
軸流し)は軸1頭の的中を前提に相手3頭を組み合わせる。的中条件が異なる
ため、Top5の部分集合として扱わず、この専用モジュールで固定保存・評価
する。

このモジュールはλの予測ロジック自体を変更しない。既存のTop5予想・λ
スコアから軸・相手を選ぶことも、オッズ人気順など全く別の基準で選ぶ
ことも許容する入出力の境界である。

軸・相手の選び方自体はここでは決めない。まだ実検証データが0件のため、
固定保存(freeze)・評価(evaluate)・ランダム選択時の期待値比較
(chance_*_probability)という、edge_diagnostics.pyと同じ検証の足場だけ
を先に用意する。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
from math import comb
from pathlib import Path
from typing import Any, Mapping, Sequence


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def _write_once(path: Path, payload: Mapping[str, Any]) -> Path:
    if path.exists():
        raise FileExistsError(f"immutable axis-formation record already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["checksum_sha256"] = sha256(_canonical(body)).hexdigest()
    with path.open("x", encoding="utf-8") as handle:
        json.dump(body, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return path


def _safe_comb(n: int, k: int) -> int:
    if n < 0 or k < 0 or n < k:
        return 0
    return comb(n, k)


@dataclass(frozen=True)
class AxisFormationPrediction:
    """軸1頭+相手3頭=計4頭の発走前固定予想。"""

    race_id: str
    scheduled_start: str
    frozen_at: str
    axis_horse_id: str
    contender_horse_ids: tuple[str, str, str]
    rationale: str = ""
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.race_id.strip():
            raise ValueError("race_id is required")
        axis = str(self.axis_horse_id).strip()
        if not axis:
            raise ValueError("axis_horse_id is required")
        object.__setattr__(self, "axis_horse_id", axis)
        contenders = tuple(str(horse) for horse in self.contender_horse_ids)
        if len(contenders) != 3 or len(set(contenders)) != 3:
            raise ValueError("contender_horse_ids must contain exactly three unique horses")
        if axis in contenders:
            raise ValueError("axis_horse_id must not also be a contender")
        object.__setattr__(self, "contender_horse_ids", contenders)
        start = datetime.fromisoformat(self.scheduled_start)
        frozen = datetime.fromisoformat(self.frozen_at)
        if start.tzinfo is None or frozen.tzinfo is None:
            raise ValueError("scheduled_start and frozen_at must be timezone-aware")
        if frozen >= start:
            raise ValueError("prediction must be frozen before scheduled start")

    @property
    def all_four(self) -> tuple[str, ...]:
        return (self.axis_horse_id, *self.contender_horse_ids)


@dataclass(frozen=True)
class AxisFormationEvaluation:
    race_id: str
    axis_hit: bool
    sanrenpuku_formation_hit: bool
    wide_or_umaren_hit: bool
    actual_top3: tuple[str, ...]


def _validated_top3(actual_top3: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(horse) for horse in actual_top3)
    if len(values) != 3 or len(set(values)) != 3:
        raise ValueError("actual_top3 must contain exactly three unique horses")
    return values


def axis_hit(prediction: AxisFormationPrediction, actual_top3: Sequence[str]) -> bool:
    """軸馬が実際の1〜3着に入ったか。"""
    return prediction.axis_horse_id in set(_validated_top3(actual_top3))


def sanrenpuku_formation_hit(prediction: AxisFormationPrediction, actual_top3: Sequence[str]) -> bool:
    """三連複の軸流し(軸+相手3頭から2頭)が的中したか。

    的中条件: 軸が1〜3着に入り、かつ実際の1〜3着が{軸}∪{相手3頭}の
    部分集合であること(軸+相手のうち2頭で残り2着分を埋められる場合)。
    """
    actual = set(_validated_top3(actual_top3))
    candidates = set(prediction.all_four)
    return prediction.axis_horse_id in actual and actual.issubset(candidates)


def wide_or_umaren_hit(prediction: AxisFormationPrediction, actual_top3: Sequence[str]) -> bool:
    """ワイド・馬連の軸流し(軸+相手3頭のいずれか1頭)が的中したか。

    的中条件: 軸が1〜3着に入り、かつ相手3頭のうち少なくとも1頭も
    1〜3着に入ること。
    """
    actual = set(_validated_top3(actual_top3))
    if prediction.axis_horse_id not in actual:
        return False
    return bool(set(prediction.contender_horse_ids) & actual)


def evaluate_axis_formation(
    prediction: AxisFormationPrediction, actual_top3: Sequence[str]
) -> AxisFormationEvaluation:
    top3 = _validated_top3(actual_top3)
    return AxisFormationEvaluation(
        race_id=prediction.race_id,
        axis_hit=axis_hit(prediction, top3),
        sanrenpuku_formation_hit=sanrenpuku_formation_hit(prediction, top3),
        wide_or_umaren_hit=wide_or_umaren_hit(prediction, top3),
        actual_top3=top3,
    )


def chance_sanrenpuku_formation_hit_probability(field_size: int) -> float:
    """軸+相手3頭を無作為に選んだ場合の、三連複軸流し的中確率。

    4頭(軸1+相手3)を無作為に選んだとき、実際の1〜3着が選んだ4頭の
    部分集合であり、かつそのうち軸役が実際の1〜3着に入る確率。4頭の
    うちどれが軸かも無作為だとすれば、1〜3着が4頭に含まれている前提で
    軸がその1〜3着側に入る条件付き確率は3/4(4頭のうち3頭が1〜3着馬)
    である。
    """
    if field_size < 4:
        raise ValueError("field_size must be at least 4")
    p_top3_subset_of_four = _safe_comb(field_size - 3, 1) / comb(field_size, 4)
    return p_top3_subset_of_four * 0.75


def chance_wide_or_umaren_hit_probability(field_size: int) -> float:
    """軸+相手3頭を無作為に選んだ場合の、ワイド・馬連軸流し的中確率。

    軸が1〜3着に入る確率(3/field_size)と、軸的中を前提に相手3頭の
    うち少なくとも1頭が残り2着分に入る確率(超幾何分布)の積。
    """
    if field_size < 4:
        raise ValueError("field_size must be at least 4")
    p_axis_hit = 3.0 / field_size
    remaining = field_size - 1
    p_none_of_3_contenders_hit = _safe_comb(remaining - 2, 3) / comb(remaining, 3)
    return p_axis_hit * (1.0 - p_none_of_3_contenders_hit)


def freeze_axis_formation(root: str | Path, prediction: AxisFormationPrediction) -> Path:
    return _write_once(
        Path(root) / prediction.race_id / "AXIS_FORMATION" / "prediction.json",
        asdict(prediction),
    )


def load_axis_formation(path: str | Path) -> AxisFormationPrediction:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("axis-formation prediction must be a regular non-symlink file")
    payload = json.loads(source.read_text(encoding="utf-8"))
    checksum = payload.pop("checksum_sha256")
    actual = sha256(_canonical(payload)).hexdigest()
    if checksum != actual:
        raise ValueError("axis-formation prediction checksum mismatch: file may have been tampered with")
    payload["contender_horse_ids"] = tuple(payload["contender_horse_ids"])
    return AxisFormationPrediction(**payload)
