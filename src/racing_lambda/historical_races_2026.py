"""Confirmed project-history race snapshots for replay development.

Only values explicitly fixed before the race or explicitly recorded as official
results are stored here.  Unknown layer-2 fields remain unknown; they are not
backfilled from the result.
"""

from __future__ import annotations

from .historical_replay import (
    HistoricalHorsePreRace,
    HistoricalRacePreRace,
    HistoricalRaceResult,
)


NIIGATA_KINEN_20260830 = HistoricalRacePreRace(
    race_id="2026-08-30-niigata-08",
    race_name="新潟記念 G3",
    venue="新潟",
    surface="芝",
    distance_m=2000,
    going="稍重",
    weather="曇",
    horses=(
        HistoricalHorsePreRace("1", "ボーンディスウェイ", 58.4, 11, "丸山元気", 57.0, notes=("近走不振",)),
        HistoricalHorsePreRace("2", "サヴォーナ", 30.2, 9, "池添謙一", 57.0, notes=("近走不振",)),
        HistoricalHorsePreRace("3", "ロデオドライブ", 4.2, 3, "ルメール", 57.0, confirmed_past_runs=3, notes=("3歳", "距離延長が論点",)),
        HistoricalHorsePreRace("4", "ドゥレッツァ", 9.1, 4, "田辺裕信", 59.0, notes=("高クラス", "約11か月休養明け",)),
        HistoricalHorsePreRace("5", "ゾロアストロ", 3.3, 1, "岩田望来", 55.0, confirmed_past_runs=3, notes=("きさらぎ賞1着", "東京スポーツ杯2着", "皐月賞12着",)),
        HistoricalHorsePreRace("6", "チェルヴィニア", 12.8, 6, "津村明秀", 56.0, notes=("G1級牝馬",)),
        HistoricalHorsePreRace("7", "ジュンブロッサム", 42.2, 10, "杉原誠人", 58.0, notes=("近走不振",)),
        HistoricalHorsePreRace("8", "ダノンシーマ", 3.7, 2, "川田将雅", 57.0, confirmed_past_runs=2, notes=("目黒記念3着", "阪神大賞典3着",)),
        HistoricalHorsePreRace("9", "アーバンシック", 14.6, 7, "三浦皇成", 59.0, notes=("高クラス", "近走不振",)),
        HistoricalHorsePreRace("10", "バルエマスター", 29.5, 8, "菊沢一樹", 57.0, confirmed_past_runs=1, notes=("新潟大賞典2着",)),
        HistoricalHorsePreRace("11", "ステレンボッシュ", 11.7, 5, "戸崎圭太", 56.0, notes=("G1級牝馬",)),
    ),
    frozen_lambda_ranking=("8", "10", "5", "11", "6"),
    frozen_leading_ranking=("10", "8", "11", "6", "5"),
    frozen_odds_bug_ranking=("10", "11", "6", "8", "5"),
    source_notes=(
        "発走前固定。稍重補正後の正式保存順位。",
        "旧予想の3番は稍重補正前に候補だったが最終除外。",
        "月次統計DBと全馬の完全な過去5走数値は当時の履歴から未復元。",
        "馬体重はこの登録では未確認扱い。",
    ),
)

NIIGATA_KINEN_20260830_RESULT = HistoricalRaceResult(
    race_id=NIIGATA_KINEN_20260830.race_id,
    finishing_order=("5", "3", "8", "2", "9", "1", "7", "6", "4", "10", "11"),
)


CHUKYO_2YO_STAKES_20260830 = HistoricalRacePreRace(
    race_id="2026-08-30-chukyo-07",
    race_name="中京2歳ステークス G3",
    venue="中京",
    surface="芝",
    distance_m=1400,
    going="良",
    weather="曇",
    horses=(
        HistoricalHorsePreRace("1", "ピコキング", 65.5, 9, "田口貫太", 55.0, confirmed_past_runs=1),
        HistoricalHorsePreRace("2", "バクソウシャチョウ", 15.2, 6, "松若風馬", 55.0, confirmed_past_runs=3),
        HistoricalHorsePreRace("3", "サタンジェロ", 4.7, 3, "西村淳也", 55.0, confirmed_past_runs=2, notes=("前走1400m 1着", "明確な上昇",)),
        HistoricalHorsePreRace("4", "シスキンブルーム", 9.2, 4, "団野大成", 55.0, confirmed_past_runs=2),
        HistoricalHorsePreRace("5", "マルモリムゾウ", 10.3, 5, "泉谷楓真", 55.0, confirmed_past_runs=1),
        HistoricalHorsePreRace("6", "ジャスパートレノ", 17.3, 7, "中井裕二", 55.0, confirmed_past_runs=1),
        HistoricalHorsePreRace("7", "ビスケットサンド", 4.0, 2, "北村友一", 55.0, confirmed_past_runs=1),
        HistoricalHorsePreRace("8", "ピコジャック", 31.5, 8, "吉村誠之", 55.0, confirmed_past_runs=1),
        HistoricalHorsePreRace("9", "ジーティーマイカ", 2.4, 1, "松山弘平", 55.0, confirmed_past_runs=1),
    ),
    frozen_lambda_ranking=("7", "3", "4", "9", "6"),
    frozen_leading_ranking=("4", "7", "3", "6", "2"),
    frozen_odds_bug_ranking=("4", "6", "2", "7", "3"),
    source_notes=(
        "発走前固定。馬体重未発表時点。",
        "月次統計DBは当時未構築のため未確認扱い。",
    ),
)

CHUKYO_2YO_STAKES_20260830_RESULT = HistoricalRaceResult(
    race_id=CHUKYO_2YO_STAKES_20260830.race_id,
    finishing_order=("9", "4", "7", "6", "3"),
)


HISTORICAL_RACES = {
    NIIGATA_KINEN_20260830.race_id: (NIIGATA_KINEN_20260830, NIIGATA_KINEN_20260830_RESULT),
    CHUKYO_2YO_STAKES_20260830.race_id: (CHUKYO_2YO_STAKES_20260830, CHUKYO_2YO_STAKES_20260830_RESULT),
}
