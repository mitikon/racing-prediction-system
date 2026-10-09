"""単勝オッズから市場が織り込む3着以内確率を推定する（仮説段階の検討材料）。

JRA統計で一般に知られる傾向として、同じ「1番人気」でも単勝オッズの
水準によって複勝率は大きく異なる（抜けた本命ほど複勝率は高く、混戦の
1番人気ほど複勝率は下がる）。この一般的な傾向を特定のレース結果に
合わせて作った係数ではなく、単勝オッズそのものを確率に変換する
標準的な統計モデル(Harville, 1973)で定量化する。

出走全馬の単勝オッズから控除率を除いた勝率を求め(`implied_win_probabilities`)、
その勝率を基に「ある馬が勝ち抜けた後も残りの馬の相対的な強さの比率は
変わらない」という単純化した仮定のもとで、3着以内確率を推定する
(`harville_top3_probabilities`)。展開・脚質・馬場適性などは一切
考慮しないため、予想4頭抽出方式には自動接続しない。全頭レビュー・
脚質レビューと並ぶ検討材料として都度参照する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class MarketProbability:
    horse_id: str
    win_odds: float
    implied_win_probability: float
    estimated_top3_probability: float


def implied_win_probabilities(win_odds: Mapping[str, float]) -> dict[str, float]:
    """単勝オッズの逆数を控除率(オーバーラウンド)で正規化し、市場が織り込む勝率を返す。"""
    if not win_odds:
        raise ValueError("win_odds must not be empty")
    if any(odds <= 1.0 for odds in win_odds.values()):
        raise ValueError("decimal win odds must be greater than 1")
    raw = {horse_id: 1.0 / float(odds) for horse_id, odds in win_odds.items()}
    overround = sum(raw.values())
    return {horse_id: value / overround for horse_id, value in raw.items()}


def harville_top3_probabilities(win_probabilities: Mapping[str, float]) -> dict[str, float]:
    """Harville(1973)モデルで、各馬の3着以内確率を勝率から推定する。

    出走全馬の勝率(合計1に正規化済み)を受け取り、各馬が1着・2着・3着の
    いずれかになる確率の和を返す。「ある馬が勝ち抜けた後、残りの馬の
    相対的な強さの比率は変わらない」という単純化した仮定(Plackett-Luce
    モデル)に基づく近似であり、特定レースの結果に合わせて調整した値では
    ない。
    """
    if len(win_probabilities) < 3:
        raise ValueError("harville_top3_probabilities requires at least 3 horses")
    total = sum(win_probabilities.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError("win_probabilities must sum to 1")

    horses = list(win_probabilities)
    p = win_probabilities
    result = {horse_id: 0.0 for horse_id in horses}

    for horse_id in horses:
        result[horse_id] += p[horse_id]

    for first in horses:
        remaining_after_first = 1.0 - p[first]
        if remaining_after_first <= 0.0:
            continue
        for second in horses:
            if second == first:
                continue
            result[second] += p[first] * (p[second] / remaining_after_first)

    for first in horses:
        remaining_after_first = 1.0 - p[first]
        if remaining_after_first <= 0.0:
            continue
        for second in horses:
            if second == first:
                continue
            p_second_given_first = p[second] / remaining_after_first
            remaining_after_two = remaining_after_first - p[second]
            if remaining_after_two <= 0.0:
                continue
            for third in horses:
                if third == first or third == second:
                    continue
                result[third] += (
                    p[first] * p_second_given_first * (p[third] / remaining_after_two)
                )

    return result


def estimate_market_top3_probabilities(win_odds: Mapping[str, float]) -> list[MarketProbability]:
    """単勝オッズだけから、各馬の3着以内確率をHarvilleモデルで推定する。"""
    win_probs = implied_win_probabilities(win_odds)
    top3 = harville_top3_probabilities(win_probs)
    return [
        MarketProbability(
            horse_id=horse_id,
            win_odds=float(win_odds[horse_id]),
            implied_win_probability=round(win_probs[horse_id], 6),
            estimated_top3_probability=round(top3[horse_id], 6),
        )
        for horse_id in win_odds
    ]


def estimate_favorite_top3_probability(win_odds: Mapping[str, float]) -> MarketProbability:
    """出走全馬の単勝オッズから1番人気(最低オッズ)を特定し、その馬の
    3着以内確率をHarvilleモデルで推定する。

    「1番人気だから3着以内に来るはず」という決めつけを避けるため、
    1番人気というラベルではなく、そのオッズ水準から実際に見込まれる
    3着以内確率を数値で確認する用途を想定する。
    """
    if not win_odds:
        raise ValueError("win_odds must not be empty")
    favorite_id = min(win_odds, key=lambda horse_id: win_odds[horse_id])
    rows = estimate_market_top3_probabilities(win_odds)
    return next(row for row in rows if row.horse_id == favorite_id)


@dataclass(frozen=True)
class FavoriteGapComparison:
    favorite_id: str
    favorite_odds: float
    favorite_top3_probability: float
    runner_up_id: str
    runner_up_odds: float
    runner_up_top3_probability: float
    odds_gap: float
    probability_gap: float


def compare_favorite_and_runner_up(win_odds: Mapping[str, float]) -> FavoriteGapComparison:
    """1番人気と2番人気の単勝オッズ差から、両者の3着以内確率をHarville
    モデルで比較する。

    「1番人気と2番人気のオッズ差が大きい(=市場が1番人気に強い確信を
    持っている)ほど、1番人気の3着以内確率が高く、2番人気との差も
    開きやすい」という一般的な傾向を、特定レースの結果に合わせた係数
    ではなく、実際のレースの全馬オッズそのものから計算して確認する。
    出走馬が2頭以下の場合はHarvilleモデル自体が定義できないため
    `ValueError`になる。
    """
    rows = estimate_market_top3_probabilities(win_odds)
    sorted_rows = sorted(rows, key=lambda row: row.win_odds)
    favorite, runner_up = sorted_rows[0], sorted_rows[1]
    return FavoriteGapComparison(
        favorite_id=favorite.horse_id,
        favorite_odds=favorite.win_odds,
        favorite_top3_probability=favorite.estimated_top3_probability,
        runner_up_id=runner_up.horse_id,
        runner_up_odds=runner_up.win_odds,
        runner_up_top3_probability=runner_up.estimated_top3_probability,
        odds_gap=round(runner_up.win_odds - favorite.win_odds, 6),
        probability_gap=round(
            favorite.estimated_top3_probability - runner_up.estimated_top3_probability, 6
        ),
    )
