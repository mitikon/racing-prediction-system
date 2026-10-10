"""単勝オッズの急変動検知（仮説段階の検討材料）。

大衆と同じ公開オッズだけを使い、発走前から最終（投票締切時）にかけて
±10%以上動いた馬を機械的に検出する。これはまだ「当たる根拠」として
確立していない仮説であり、`four_horse_extraction.py`の凍結予測とは
別に、検討材料として都度記録・報告するためのものである。十分な件数の
実レースで勝敗との関係を検証するまでは、予測ロジックへ自動接続しない。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


SUDDEN_MOVEMENT_THRESHOLD = 0.10


@dataclass(frozen=True)
class OddsMovementAlert:
    horse_id: str
    odds_before: float
    odds_after: float
    change_ratio: float
    direction: str
    triggered: bool


def detect_sudden_odds_movement(
    horse_id: str,
    odds_before: float,
    odds_after: float,
    *,
    threshold: float = SUDDEN_MOVEMENT_THRESHOLD,
) -> OddsMovementAlert:
    if not horse_id.strip():
        raise ValueError("horse_id is required")
    if odds_before <= 1.0 or odds_after <= 1.0:
        raise ValueError("decimal win odds must be greater than 1")
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    change_ratio = (odds_after - odds_before) / odds_before
    direction = "drift_up" if change_ratio > 0 else "shorten" if change_ratio < 0 else "unchanged"
    return OddsMovementAlert(
        horse_id=horse_id,
        odds_before=float(odds_before),
        odds_after=float(odds_after),
        change_ratio=round(change_ratio, 6),
        direction=direction,
        triggered=abs(change_ratio) >= threshold,
    )


def scan_field_for_sudden_movement(
    odds_before: Mapping[str, float],
    odds_after: Mapping[str, float],
    *,
    threshold: float = SUDDEN_MOVEMENT_THRESHOLD,
) -> list[OddsMovementAlert]:
    """出走全馬のオッズ推移を確認し、急変動した馬だけを変動幅の大きい順に返す。"""
    if set(odds_before) != set(odds_after):
        raise ValueError("odds_before and odds_after must cover the same horses")
    alerts = [
        detect_sudden_odds_movement(horse_id, odds_before[horse_id], odds_after[horse_id], threshold=threshold)
        for horse_id in odds_before
    ]
    triggered = [alert for alert in alerts if alert.triggered]
    return sorted(triggered, key=lambda alert: abs(alert.change_ratio), reverse=True)
