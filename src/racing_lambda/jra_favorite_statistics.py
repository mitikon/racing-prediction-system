"""JRA統計に基づく「1番人気」のオッズ帯別・複勝率の参考値（出典明記）。

`market_probability.py`のHarvilleモデルが理論値であるのに対し、こちらは
一般公開されている集計記事から収集した**実測値の参考レンジ**である。
集計期間・対象レース(全レース/G1限定)が出典ごとに異なり、単一の正確な
統計として扱えないため、値ではなく「範囲」として保持する。

各エントリは`verified`フラグで出典の確度を区別する。
- `verified=True`: JRA-VAN・うまめし等、個別のURL・記事名まで特定できる
  出典に基づく値(2026-10-08にWeb検索で収集。原文ページは直接確認でき
  ていないため、検索結果の要約を経由した値である点に留意)
- `verified=False`: Gemini(Google検索)による要約から得た値。ユーザーが
  Geminiに検索させた結果を提供したもので、個別のURL出典までは辿れて
  いない。既存の確認済みの帯(1.0-1.9倍・3.0-3.9倍)とおおむね方向性が
  一致するため収録するが、一次出典が未確認である点は`verified=False`
  で明示する

出典(verified=Trueの帯):
- JRA-VANコラム「DATA DRIVEN DERBY」(jra-van.jp/fun/ddd/20190513.html等):
  G1限定・過去10年の集計で、単勝1.0-1.4倍の1番人気の複勝率81.3%
  (複勝回収率89%)、1.5-1.9倍で複勝率73.8%(複勝回収率83%)
- うまめし(umameshi.com)の1番人気・複勝に関する集計記事:
  1番人気全体で複勝率約63%(勝率32%・連対率51%)、単勝1.9倍以下の
  1番人気で複勝率8割超、単勝3倍台の1番人気で複勝率約50%
- 東洋経済オンライン: 中央競馬過去5年、1番人気の1着率30.2%・2着率19.7%
- 現代ビジネス: 2010-2019年の10年間、1番人気の勝率32.1%

出典(verified=Falseの帯、2026-10-08にユーザー提供、Gemini/Google検索の
要約。一次URLは未確認):
- 単勝2.0-2.9倍: 勝率約30-33%、連対率約50-52%、複勝率約65-67%
- 単勝4.0倍以上: 勝率約15-20%、連対率約28-32%、複勝率約40-45%

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
    verified: bool


# 1番人気(=単勝最低オッズの馬)に限定した複勝率の参考レンジ。
# 出典ごとに集計期間・対象レースが異なるため、幅を持たせている。
# 該当する参考値が見つからなかったオッズ帯は、ここに含めない
# (lookup_favorite_place_rate_referenceがNoneを返す)。
FAVORITE_PLACE_RATE_REFERENCES: tuple[FavoritePlaceRateReference, ...] = (
    FavoritePlaceRateReference(
        odds_lower_inclusive=1.0,
        odds_upper_inclusive=1.4,
        low_estimate=0.813,
        high_estimate=0.897,
        source_note="JRA-VAN「DATA DRIVEN DERBY」G1限定過去10年(81.3%) / うまめし集計(89.7%、集計期間不明)",
        verified=True,
    ),
    FavoritePlaceRateReference(
        odds_lower_inclusive=1.5,
        odds_upper_inclusive=1.9,
        low_estimate=0.738,
        high_estimate=0.80,
        source_note="JRA-VAN「DATA DRIVEN DERBY」G1限定過去10年(73.8%) / うまめし「単勝1.9倍以下」集計(8割超)",
        verified=True,
    ),
    FavoritePlaceRateReference(
        odds_lower_inclusive=2.0,
        odds_upper_inclusive=2.9,
        low_estimate=0.65,
        high_estimate=0.67,
        source_note="Gemini(Google検索)による要約値(2026-10-08提供)。一次URL出典は未確認。複数回の独自検索でもこの帯の確認済み出典は見つからなかった",
        verified=False,
    ),
    FavoritePlaceRateReference(
        odds_lower_inclusive=3.0,
        odds_upper_inclusive=3.9,
        low_estimate=0.50,
        high_estimate=0.50,
        source_note="うまめし「単勝3倍台の1番人気」集計(約50%、集計期間不明)。Gemini要約値(約50-55%)もおおむね近い水準",
        verified=True,
    ),
    FavoritePlaceRateReference(
        odds_lower_inclusive=4.0,
        odds_upper_inclusive=999.0,
        low_estimate=0.40,
        high_estimate=0.45,
        source_note="Gemini(Google検索)による要約値(2026-10-08提供)。一次URL出典は未確認",
        verified=False,
    ),
)

# 1番人気全体(オッズ帯を区切らない平均)の参考値。
FAVORITE_OVERALL_WIN_RATE_RANGE = (0.302, 0.321)
FAVORITE_OVERALL_PLACE_RATE_RANGE = (0.63, 0.63)


def lookup_favorite_place_rate_reference(odds: float) -> FavoritePlaceRateReference | None:
    """1番人気の単勝オッズから、該当する複勝率の参考レンジを探す。

    返り値の`verified`が`False`の場合、個別のURL出典までは辿れていない
    (Gemini等のAI要約を経由した)値であることを示す。`verified=True`の
    帯より慎重に扱うこと。対応するオッズ帯の参考値が一切見つかっていない
    場合はNoneを返す。
    """
    if odds <= 1.0:
        raise ValueError("decimal win odds must be greater than 1")
    for reference in FAVORITE_PLACE_RATE_REFERENCES:
        if reference.odds_lower_inclusive <= odds <= reference.odds_upper_inclusive:
            return reference
    return None
