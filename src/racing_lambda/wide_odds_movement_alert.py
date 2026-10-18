"""ワイドオッズの急変動検知（仮説段階の検討材料）。

単勝オッズの急変動検知(`odds_movement_alert.py`)と同じ考え方を、
馬番の組み合わせ(ワイド)オッズに適用する。大口投資が発走直前に
的中濃厚と判断した組み合わせへ資金を投入すると、他の参加者より先に
確信を持って買われたことで締切直前にその組み合わせのオッズが急激に
下がる、という仮説を機械的な変動率検出として定量化する。

これは「当たる根拠」として確立した指標ではなく、`four_horse_extraction.py`
の凍結予測とは別に、検討材料として都度記録・報告するためのもの。十分な
件数の実レースで勝敗との関係を検証するまでは、予測ロジックへ自動接続
しない。

運用上の制約: この仕組みはオッズを自動監視するものではない。ユーザーが
共有した時点のワイドオッズボード(画像)を2回分(早い時点・締切直前)
比較することで変動を検出する。締切直前に画像が共有された場合、比較・
報告が投票締切に間に合うとは限らない。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

SUDDEN_WIDE_MOVEMENT_THRESHOLD = 0.10


@dataclass(frozen=True)
class WideOddsMovementAlert:
    pair_key: str
    odds_before: float
    odds_after: float
    change_ratio: float
    direction: str
    triggered: bool


def detect_sudden_wide_odds_movement(
    pair_key: str,
    odds_before: float,
    odds_after: float,
    *,
    threshold: float = SUDDEN_WIDE_MOVEMENT_THRESHOLD,
) -> WideOddsMovementAlert:
    if not pair_key.strip():
        raise ValueError("pair_key is required")
    if odds_before <= 1.0 or odds_after <= 1.0:
        raise ValueError("decimal wide odds must be greater than 1")
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    change_ratio = (odds_after - odds_before) / odds_before
    direction = "drift_up" if change_ratio > 0 else "shorten" if change_ratio < 0 else "unchanged"
    return WideOddsMovementAlert(
        pair_key=pair_key,
        odds_before=float(odds_before),
        odds_after=float(odds_after),
        change_ratio=round(change_ratio, 6),
        direction=direction,
        triggered=abs(change_ratio) >= threshold,
    )


def scan_wide_odds_for_sudden_movement(
    odds_before: Mapping[str, float],
    odds_after: Mapping[str, float],
    *,
    threshold: float = SUDDEN_WIDE_MOVEMENT_THRESHOLD,
) -> list[WideOddsMovementAlert]:
    """2時点のワイドオッズを比較し、急変動した組み合わせだけを変動幅の大きい順に返す。

    全通りの組み合わせが両方の時点で揃っている必要はなく、両方に記録
    されている組み合わせ(積集合)だけを比較対象とする。締切直前の
    確認では全通りを撮り直す時間がない想定のため、一部の組み合わせ
    だけ(例えば選出4頭の6点だけ)を2回目として共有された場合でも
    比較できるようにしている。
    """
    common_pairs = set(odds_before) & set(odds_after)
    alerts = [
        detect_sudden_wide_odds_movement(
            pair_key, odds_before[pair_key], odds_after[pair_key], threshold=threshold
        )
        for pair_key in common_pairs
    ]
    triggered = [alert for alert in alerts if alert.triggered]
    return sorted(triggered, key=lambda alert: abs(alert.change_ratio), reverse=True)
