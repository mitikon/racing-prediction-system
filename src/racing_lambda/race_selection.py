"""λの出力を使って「どのレースに力を注ぐべきか」を選別する。

これまでのλ(Top5ランキング)・軸形成(axis_formation.py)は、与えられた
1レースをどう予想するかを決める。このモジュールは逆に、λ自身の出力を
使って複数のレースの中から「市場の非効率が残っていそうなレース」を
選び出す - 個々のレースの予想精度ではなく、どこに予想労力・検証を
集中させるかという選別の角度である。

2つの独立した手がかりを使う。

1. λとオッズ人気順との乖離 - λのTop5と単勝人気Top5が大きく異なる
   レースほど、λが市場の価格付けと異なる情報を持っている(本当に
   優れた情報か、単なるノイズかは別問題)可能性が高い。
2. レースのクラス - 未勝利・新馬戦は公開情報が少なく市場参加者の
   分析精度も粗いため非効率が残りやすいとされる(ハンディキャッピング
   業界で広く言われる経験則)。重賞(G1-G3)は参加者が多く分析も厚いため
   最も効率的に価格付けされる傾向がある。

いずれも経験則であり、このプロジェクトの実データで検証されたものでは
ない。edge_diagnostics.pyと同じ手法で、選別したレースが実際に高い
的中率・回収率を示すかどうかは、レースが蓄積してから別途検証する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from .edge_diagnostics import odds_favorite_order

RaceInformationTier = Literal["maiden", "conditional", "graded"]

_MAIDEN_KEYWORDS = ("未勝利", "新馬")
_GRADED_KEYWORDS = ("G1", "G2", "G3", "GI", "GII", "GIII")

# レース情報階層ごとの非効率ボーナス。重賞(graded)は最も効率的に価格付け
# されるためボーナス無し、未勝利・新馬は情報が薄く非効率が残りやすいため
# 最大のボーナスを与える。固定値であり、実データでの較正はまだ無い。
_INFORMATION_TIER_BONUS: Mapping[RaceInformationTier, float] = {
    "graded": 0.0,
    "conditional": 0.5,
    "maiden": 1.0,
}


def classify_information_tier(race_name: str) -> RaceInformationTier:
    """レース名から情報階層を分類する(重賞 > 条件戦 > 未勝利・新馬の順に
    市場が効率的と見なす経験則)。"""
    name = race_name.upper()
    if any(keyword.upper() in name for keyword in _GRADED_KEYWORDS):
        return "graded"
    if any(keyword in race_name for keyword in _MAIDEN_KEYWORDS):
        return "maiden"
    return "conditional"


def lambda_market_divergence(
    lambda_top5: Sequence[str],
    odds_by_horse: Mapping[str, float],
    depth: int = 5,
) -> int:
    """λのTop(depth)と単勝人気Top(depth)のうち、両方に無い馬の数。

    0ならλは市場の人気順をそのまま再現しているだけで、新しい情報を
    加えていないことを意味する。depthに近いほど、λと市場の見方が
    大きく異なっている。
    """
    lambda_set = set(lambda_top5[:depth])
    if len(lambda_set) != min(depth, len(lambda_top5)):
        raise ValueError("lambda_top5 must not contain duplicates")
    favorite_set = set(odds_favorite_order(odds_by_horse)[:depth])
    return len(lambda_set - favorite_set)


@dataclass(frozen=True)
class RaceEdgeCandidacy:
    race_id: str
    information_tier: RaceInformationTier
    lambda_market_divergence: int
    edge_candidacy_score: float


def score_race_edge_candidacy(
    race_id: str,
    race_name: str,
    lambda_top5: Sequence[str],
    odds_by_horse: Mapping[str, float],
    depth: int = 5,
) -> RaceEdgeCandidacy:
    """λと市場の乖離・レースの情報階層から、このレースを優先検証すべき
    度合いを1つのスコアにまとめる。

    score = 乖離数(0〜depth) + 情報階層ボーナス。乖離とボーナスは全く別の
    根拠を持つ経験則であり、重みを揃えて単純加算する以上の精緻化は、
    実データでの較正を経るまで行わない。
    """
    divergence = lambda_market_divergence(lambda_top5, odds_by_horse, depth=depth)
    tier = classify_information_tier(race_name)
    score = divergence + _INFORMATION_TIER_BONUS[tier]
    return RaceEdgeCandidacy(
        race_id=race_id,
        information_tier=tier,
        lambda_market_divergence=divergence,
        edge_candidacy_score=score,
    )


def rank_races_by_edge_candidacy(
    candidacies: Sequence[RaceEdgeCandidacy],
) -> tuple[RaceEdgeCandidacy, ...]:
    """edge_candidacy_scoreの降順(同点はrace_id昇順)に並べる。"""
    if not candidacies:
        raise ValueError("at least one race candidacy is required")
    return tuple(sorted(candidacies, key=lambda item: (-item.edge_candidacy_score, item.race_id)))
