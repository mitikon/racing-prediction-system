"""展開(ペース)予想 - 個々の馬の能力ではなく、レースの流れを予測してλへ渡す。

Top5ランキング(scoring.py)は`HorseEntry.pace_fit`を入力の一つとして使うが、
これまでその値は人手で付けた0-1の判断値だった。このモジュールは、出走馬の
脚質構成からレースのペース想定(ハイペース/ミドル/スローペース)を予測し、
脚質とペース想定の組み合わせから`pace_fit`を導出する。

導出したpace_fitは`scoring.build_prediction`へそのまま渡せる形にし、
λの予測ロジック(重み付け・ジョッキー縮小推定等)自体は変更しない。あくまで
一つの入力成分を、人手の勘ではなく明示的なペース理論から計算する。

脚質分類(逃げ/先行/差し/追込)自体は、まだこのプロジェクトのPRE_RACE記録に
含まれていない。公開される出馬表・過去走には通常印字されているため、入力
として外部から与える前提とする。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

RunningStyle = Literal["escape", "front", "stalk", "closer"]
PaceScenario = Literal["fast", "average", "slow"]

_VALID_STYLES: tuple[RunningStyle, ...] = ("escape", "front", "stalk", "closer")

# 逃げ・先行(早め先頭集団)の頭数が全体に占める比率がこれ以上ならハイペース、
# これ以下ならスローペースと見なす。中間はミドルペース。この閾値は固定的な
# ハンディキャッピング経験則であり、実データで較正したものではない -
# edge_diagnostics.pyと同じ手法で、実際のレース結果が蓄積した時点で
# この閾値自体を検証すべきである。
FAST_PACE_FRONT_RATIO_THRESHOLD = 0.45
SLOW_PACE_FRONT_RATIO_THRESHOLD = 0.20


def _validate_styles(styles: Mapping[str, RunningStyle]) -> None:
    if not styles:
        raise ValueError("at least one horse's running style is required")
    for horse_id, style in styles.items():
        if style not in _VALID_STYLES:
            raise ValueError(f"unknown running style {style!r} for horse {horse_id!r}")


def predict_pace_scenario(styles: Mapping[str, RunningStyle]) -> PaceScenario:
    """出走馬の脚質構成から、レースのペース想定を分類する。

    逃げ・先行馬の比率が高いほど、先頭集団を取り合う形になりペースが速く
    なりやすい(ハイペース)。比率が低いと、誰も先を急がずペースが落ち着き
    やすい(スローペース)。
    """
    _validate_styles(styles)
    total = len(styles)
    front_runners = sum(1 for style in styles.values() if style in ("escape", "front"))
    ratio = front_runners / total
    if ratio >= FAST_PACE_FRONT_RATIO_THRESHOLD:
        return "fast"
    if ratio <= SLOW_PACE_FRONT_RATIO_THRESHOLD:
        return "slow"
    return "average"


# scenario -> style -> pace_fit (0-1)。ハイペースは差し・追込に有利(前が
# 止まりやすい)、スローペースは逃げ・先行に有利(そのまま残りやすい)という、
# 競馬の基本的なペース理論に基づく対応表。これも固定値であり、実データで
# 較正されたものではない。
_PACE_FIT_TABLE: Mapping[PaceScenario, Mapping[RunningStyle, float]] = {
    "fast": {"escape": 0.25, "front": 0.35, "stalk": 0.65, "closer": 0.80},
    "average": {"escape": 0.50, "front": 0.55, "stalk": 0.50, "closer": 0.45},
    "slow": {"escape": 0.80, "front": 0.70, "stalk": 0.40, "closer": 0.25},
}


def pace_fit_from_scenario(style: RunningStyle, scenario: PaceScenario) -> float:
    """脚質とペース想定の組み合わせから、scoring.pyへ渡すpace_fitを返す。"""
    if style not in _VALID_STYLES:
        raise ValueError(f"unknown running style {style!r}")
    return _PACE_FIT_TABLE[scenario][style]


@dataclass(frozen=True)
class PaceScenarioAnalysis:
    scenario: PaceScenario
    front_runner_ratio: float
    pace_fit_by_horse: Mapping[str, float]


def analyze_pace_scenario(styles: Mapping[str, RunningStyle]) -> PaceScenarioAnalysis:
    """脚質構成からペース想定を分類し、各馬のpace_fitを一括算出する。

    返り値の`pace_fit_by_horse`を、既存の`HorseEntry.pace_fit`に差し替えて
    `scoring.build_prediction`へ渡せば、λの重み付けロジックはそのまま
    ペース理論に基づくpace_fitで予測を行える。
    """
    _validate_styles(styles)
    scenario = predict_pace_scenario(styles)
    total = len(styles)
    front_runners = sum(1 for style in styles.values() if style in ("escape", "front"))
    return PaceScenarioAnalysis(
        scenario=scenario,
        front_runner_ratio=front_runners / total,
        pace_fit_by_horse={
            horse_id: pace_fit_from_scenario(style, scenario) for horse_id, style in styles.items()
        },
    )
