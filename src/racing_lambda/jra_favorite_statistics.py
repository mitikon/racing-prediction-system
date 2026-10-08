"""JRA統計に基づく「1番人気」のオッズ帯別・複勝率の参考値（出典明記）。

`market_probability.py`のHarvilleモデルが理論値であるのに対し、こちらは
一般公開されている集計記事から収集した**実測値の参考レンジ**である。
2026-10-08にWeb検索で収集した。集計期間・対象レース(全レース/G1限定)が
出典ごとに異なり、単一の正確な統計として扱えないため、値ではなく
「範囲」として保持し、該当する実測データがないオッズ帯は無理に
補間せず`None`を返す。

出典(2026-10-08確認、検索結果の抜粋による。原文ページは直接確認できて
いないため、数値は検索結果の要約を経由した値である点に留意すること):
- JRA-VANコラム「DATA DRIVEN DERBY」(jra-van.jp/fun/ddd/20190513.html等):
  G1限定・過去10年の集計で、単勝1.0-1.4倍の1番人気の複勝率81.3%
  (複勝回収率89%)、1.5-1.9倍で複勝率73.8%(複勝回収率83%)
- うまめし(umameshi.com)の1番人気・複勝に関する集計記事:
  1番人気全体で複勝率約63%(勝率32%・連対率51%)、単勝1.9倍以下の
  1番人気で複勝率8割超、単勝3倍台の1番人気で複勝率約50%
- 東洋経済オンライン: 中央競馬過去5年、1番人気の1着率30.2%・2着率19.7%
- 現代ビジネス: 2010-2019年の10年間、1番人気の勝率32.1%

これらはあくまで「1番人気というレンジの過去実績」であり、個別レースの
展開・脚質・馬場適性を一切反映しない。予想4頭抽出方式には自動接続せず、
Harvilleモデルの理論値と併記して参照する検討材料として位置づける。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FavoritePlaceRateReference:
    odds_lower_inclusive: float
    odds_upper_inclusive: float
    low_estimate: float
    high_estimate: float
    source_note: str


# 1番人気(=単勝最低オッズの馬)に限定した実測複勝率の参考レンジ。
# 出典ごとに集計期間・対象レースが異なるため、幅を持たせている。
# 該当する実測データが見つからなかったオッズ帯は、ここに含めない
# (lookup_favorite_place_rate_referenceがNoneを返す)。
FAVORITE_PLACE_RATE_REFERENCES: tuple[FavoritePlaceRateReference, ...] = (
    FavoritePlaceRateReference(
        odds_lower_inclusive=1.0,
        odds_upper_inclusive=1.4,
        low_estimate=0.813,
        high_estimate=0.897,
        source_note="JRA-VAN「DATA DRIVEN DERBY」G1限定過去10年(81.3%) / うまめし集計(89.7%、集計期間不明)",
    ),
    FavoritePlaceRateReference(
        odds_lower_inclusive=1.5,
        odds_upper_inclusive=1.9,
        low_estimate=0.738,
        high_estimate=0.80,
        source_note="JRA-VAN「DATA DRIVEN DERBY」G1限定過去10年(73.8%) / うまめし「単勝1.9倍以下」集計(8割超)",
    ),
    FavoritePlaceRateReference(
        odds_lower_inclusive=3.0,
        odds_upper_inclusive=3.9,
        low_estimate=0.50,
        high_estimate=0.50,
        source_note="うまめし「単勝3倍台の1番人気」集計(約50%、集計期間不明)",
    ),
)

# 1番人気全体(オッズ帯を区切らない平均)の参考値。
FAVORITE_OVERALL_WIN_RATE_RANGE = (0.302, 0.321)
FAVORITE_OVERALL_PLACE_RATE_RANGE = (0.63, 0.63)


def lookup_favorite_place_rate_reference(odds: float) -> FavoritePlaceRateReference | None:
    """1番人気の単勝オッズから、該当する実測複勝率レンジを探す。

    対応するオッズ帯の実測データが見つかっていない場合はNoneを返す。
    理論モデル(`market_probability.estimate_favorite_top3_probability`)で
    補うか、「この帯の実測1番人気限定データはまだ見つかっていない」と
    明示したうえで判断する。
    """
    if odds <= 1.0:
        raise ValueError("decimal win odds must be greater than 1")
    for reference in FAVORITE_PLACE_RATE_REFERENCES:
        if reference.odds_lower_inclusive <= odds <= reference.odds_upper_inclusive:
            return reference
    return None
