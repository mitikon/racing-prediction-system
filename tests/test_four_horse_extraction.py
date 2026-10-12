from datetime import datetime, timedelta, timezone

import pytest

from racing_lambda.four_horse_extraction import (
    EXTRACTION_METHOD,
    ExtractionResult,
    FourHorseExtraction,
    WideOddsSnapshot,
    evaluate_four_horse_extraction,
    freeze_four_horse_extraction,
    freeze_wide_odds_snapshot,
    settle_four_horse_extraction,
    wide_pair_key,
    winning_wide_combo_odds,
)


def _prediction(**overrides):
    start = datetime(2026, 9, 27, 15, 45, tzinfo=timezone(timedelta(hours=9)))
    defaults = dict(
        race_id="20260927-HANSHIN-11",
        race_name="検証レース",
        venue="阪神",
        scheduled_start=start.isoformat(),
        frozen_at=(start - timedelta(minutes=5)).isoformat(),
        horses=("8", "3", "5", "2"),
        field_size=8,
        reviewed_horse_numbers=("1", "2", "3", "4", "5", "6", "7", "8"),
        running_styles=(
            ("1", "逃げ"),
            ("2", "先行"),
            ("3", "差し"),
            ("4", "追込"),
            ("5", "先行"),
            ("6", "差し"),
            ("7", "逃げ"),
            ("8", "先行"),
        ),
        pace_scenario="ハイペース想定",
        evidence_notes=("出馬表画像より直近上昇度を最重視", "距離短縮を減点"),
        source_document_sha256=("a" * 64,),
    )
    defaults.update(overrides)
    return FourHorseExtraction(**defaults)


def test_prediction_requires_exactly_four_unique_horses():
    with pytest.raises(ValueError):
        _prediction(horses=("1", "2", "3", "3"))
    with pytest.raises(ValueError):
        _prediction(horses=("1", "2", "3"))


def test_prediction_must_be_frozen_before_start():
    p = _prediction()
    with pytest.raises(ValueError):
        FourHorseExtraction(**{**p.__dict__, "frozen_at": p.scheduled_start})


def test_method_field_cannot_be_repurposed_as_pca():
    with pytest.raises(ValueError):
        _prediction(method="regularized_pca")


def test_partial_field_review_is_rejected():
    """一部の目立つ馬だけを見て4頭を選ぶことを禁止する。"""
    with pytest.raises(ValueError, match="every starter must be reviewed"):
        _prediction(field_size=8, reviewed_horse_numbers=("1", "2", "3", "8"))


def test_duplicate_reviewed_horse_numbers_are_rejected():
    with pytest.raises(ValueError, match="duplicates"):
        _prediction(
            field_size=8,
            reviewed_horse_numbers=("1", "2", "3", "4", "5", "6", "7", "7"),
        )


def test_partial_running_style_review_is_rejected():
    """人気馬だけ脚質を記録して中位人気の先行馬を見落とすことを禁止する。"""
    with pytest.raises(ValueError, match="running_styles must record every reviewed starter"):
        _prediction(
            running_styles=(
                ("1", "逃げ"),
                ("2", "先行"),
                ("3", "差し"),
            ),
        )


def test_duplicate_running_style_horse_numbers_are_rejected():
    with pytest.raises(ValueError, match="running_styles must not contain duplicate"):
        _prediction(
            running_styles=(
                ("1", "逃げ"),
                ("1", "先行"),
                ("2", "先行"),
                ("3", "差し"),
                ("4", "追込"),
                ("5", "先行"),
                ("6", "差し"),
                ("7", "逃げ"),
            ),
        )


def test_empty_running_style_value_is_rejected():
    with pytest.raises(ValueError, match="running_styles values must not be empty"):
        _prediction(
            running_styles=(
                ("1", "逃げ"),
                ("2", ""),
                ("3", "差し"),
                ("4", "追込"),
                ("5", "先行"),
                ("6", "差し"),
                ("7", "逃げ"),
                ("8", "先行"),
            ),
        )


def test_pace_scenario_is_required():
    with pytest.raises(ValueError, match="pace_scenario is required"):
        _prediction(pace_scenario="  ")


def test_win_odds_is_optional_and_defaults_empty():
    prediction = _prediction()
    assert prediction.win_odds == {}


def test_win_odds_must_cover_every_reviewed_starter():
    with pytest.raises(ValueError, match="win_odds, once recorded, must cover every reviewed starter"):
        _prediction(win_odds={"1": 2.0, "2": 3.0})


def test_win_odds_rejects_odds_at_or_below_one():
    with pytest.raises(ValueError, match="decimal win odds in win_odds must be greater than 1"):
        _prediction(
            win_odds={
                "1": 1.0, "2": 3.0, "3": 4.0, "4": 5.0,
                "5": 6.0, "6": 7.0, "7": 8.0, "8": 9.0,
            }
        )


def test_win_odds_accepts_full_field_coverage():
    prediction = _prediction(
        win_odds={
            "1": 10.0, "2": 2.0, "3": 5.0, "4": 20.0,
            "5": 4.0, "6": 8.0, "7": 15.0, "8": 3.0,
        }
    )
    assert prediction.win_odds["2"] == 2.0


def test_wide_pair_key_normalizes_by_numeric_order():
    assert wide_pair_key("11", "4") == "4-11"
    assert wide_pair_key("4", "11") == "4-11"
    assert wide_pair_key("8", "3") == "3-8"


def test_wide_odds_is_optional_and_defaults_empty():
    prediction = _prediction()
    assert prediction.wide_odds == {}


def test_wide_odds_must_cover_all_six_pairs_of_selected_horses():
    # horses=("8", "3", "5", "2") の6通りのうち1組だけ欠けている
    with pytest.raises(ValueError, match="wide_odds, once recorded, must include at least the 6 pairs"):
        _prediction(
            wide_odds={
                "3-8": 6.5, "5-8": 7.0, "2-8": 8.0, "3-5": 5.5, "2-3": 9.0,
            }
        )


def test_wide_odds_accepts_extra_pairs_beyond_the_required_six():
    # 全通りのオッズボードを添付された場合、6点以外の組み合わせも保存できる
    prediction = _prediction(
        wide_odds={
            "3-8": 6.5, "5-8": 7.0, "2-8": 8.0, "3-5": 5.5, "2-3": 9.0, "2-5": 6.0,
            "1-8": 12.0,  # 選出4頭に含まれない馬との組み合わせ
        }
    )
    assert prediction.wide_odds["1-8"] == 12.0


def test_wide_odds_rejects_odds_at_or_below_one():
    with pytest.raises(ValueError, match="decimal wide odds in wide_odds must be greater than 1"):
        _prediction(
            wide_odds={
                "3-8": 1.0, "5-8": 7.0, "2-8": 8.0, "3-5": 5.5, "2-3": 9.0, "2-5": 6.0,
            }
        )


def test_wide_odds_accepts_all_six_pairs_of_selected_horses():
    prediction = _prediction(
        wide_odds={
            "3-8": 6.5, "5-8": 7.0, "2-8": 8.0, "3-5": 5.5, "2-3": 9.0, "2-5": 6.0,
        }
    )
    assert prediction.wide_odds[wide_pair_key("8", "3")] == 6.5


def test_selected_horses_must_be_among_reviewed_starters():
    with pytest.raises(ValueError, match="reviewed starters"):
        _prediction(
            horses=("8", "3", "5", "9"),
            field_size=8,
            reviewed_horse_numbers=("1", "2", "3", "4", "5", "6", "7", "8"),
        )


def test_write_once_prevents_posthoc_prediction_overwrite(tmp_path):
    p = _prediction()
    path = freeze_four_horse_extraction(tmp_path, p)
    assert path.exists()
    with pytest.raises(FileExistsError):
        freeze_four_horse_extraction(tmp_path, p)


def _wide_odds_snapshot(prediction, **overrides):
    start = datetime.fromisoformat(prediction.scheduled_start)
    defaults = dict(
        race_id=prediction.race_id,
        horses=prediction.horses,
        captured_at=(start - timedelta(hours=1)).isoformat(),
        wide_odds={
            "3-8": 6.5, "5-8": 7.0, "2-8": 8.0, "3-5": 5.5, "2-3": 9.0, "2-5": 6.0,
        },
    )
    defaults.update(overrides)
    return WideOddsSnapshot(**defaults)


def test_wide_odds_snapshot_requires_existing_frozen_prediction(tmp_path):
    p = _prediction()
    with pytest.raises(FileNotFoundError):
        freeze_wide_odds_snapshot(tmp_path, _wide_odds_snapshot(p))


def test_wide_odds_snapshot_must_match_frozen_horses(tmp_path):
    p = _prediction()
    freeze_four_horse_extraction(tmp_path, p)
    snapshot = _wide_odds_snapshot(
        p,
        horses=("1", "2", "3", "4"),
        wide_odds={
            "1-2": 6.5, "1-3": 7.0, "1-4": 8.0, "2-3": 5.5, "2-4": 9.0, "3-4": 6.0,
        },
    )
    with pytest.raises(ValueError, match="must match the frozen PRE_RACE selection"):
        freeze_wide_odds_snapshot(tmp_path, snapshot)


def test_wide_odds_snapshot_must_be_captured_before_scheduled_start(tmp_path):
    p = _prediction()
    freeze_four_horse_extraction(tmp_path, p)
    start = datetime.fromisoformat(p.scheduled_start)
    snapshot = _wide_odds_snapshot(p, captured_at=start.isoformat())
    with pytest.raises(ValueError, match="must be captured before scheduled start"):
        freeze_wide_odds_snapshot(tmp_path, snapshot)


def test_wide_odds_snapshot_does_not_touch_frozen_prediction(tmp_path):
    p = _prediction()
    pre_race_path = freeze_four_horse_extraction(tmp_path, p)
    before = pre_race_path.read_text(encoding="utf-8")
    snapshot_path = freeze_wide_odds_snapshot(tmp_path, _wide_odds_snapshot(p))
    assert snapshot_path != pre_race_path
    assert pre_race_path.read_text(encoding="utf-8") == before


def test_wide_odds_snapshot_allows_multiple_timed_captures_per_race(tmp_path):
    # 早い時点・締切直前など、1レースにつき複数回の撮影を想定する
    p = _prediction()
    freeze_four_horse_extraction(tmp_path, p)
    first = freeze_wide_odds_snapshot(tmp_path, _wide_odds_snapshot(p))
    second = freeze_wide_odds_snapshot(
        tmp_path,
        _wide_odds_snapshot(
            p,
            captured_at=(
                datetime.fromisoformat(p.scheduled_start) - timedelta(minutes=5)
            ).isoformat(),
            wide_odds={
                "3-8": 4.0, "5-8": 7.0, "2-8": 8.0, "3-5": 5.5, "2-3": 9.0, "2-5": 6.0,
            },
        ),
    )
    assert first.name == "wide_odds_snapshot.json"
    assert second.name == "wide_odds_snapshot_2.json"
    assert first.exists() and second.exists()
    assert first.read_text(encoding="utf-8") != second.read_text(encoding="utf-8")


def test_wide_odds_snapshot_accepts_full_board_beyond_the_required_six(tmp_path):
    p = _prediction()
    freeze_four_horse_extraction(tmp_path, p)
    snapshot = _wide_odds_snapshot(
        p,
        wide_odds={
            "3-8": 6.5, "5-8": 7.0, "2-8": 8.0, "3-5": 5.5, "2-3": 9.0, "2-5": 6.0,
            "1-4": 29.6,  # 選出4頭に含まれない馬同士の組み合わせ(全通り添付を想定)
        },
    )
    path = freeze_wide_odds_snapshot(tmp_path, snapshot)
    assert path.exists()


def test_winning_wide_combo_odds_returns_odds_for_actual_top3_pairs():
    # 2026-10-12京都11Rの実例: 予想4頭は{15,9,13,2}だが的中は1-4-9
    finishing_order = ("1", "4", "9", "8", "15")
    wide_odds = {
        wide_pair_key("1", "4"): 29.6,
        wide_pair_key("1", "9"): 5.6,
        wide_pair_key("4", "9"): 11.9,
        wide_pair_key("9", "15"): 7.4,  # 的中以外の組み合わせは無視される
    }
    result = winning_wide_combo_odds(finishing_order, wide_odds)
    assert result == {
        wide_pair_key("1", "4"): 29.6,
        wide_pair_key("1", "9"): 5.6,
        wide_pair_key("4", "9"): 11.9,
    }


def test_winning_wide_combo_odds_omits_pairs_not_recorded():
    finishing_order = ("1", "4", "9")
    wide_odds = {wide_pair_key("1", "9"): 5.6}  # 1-4, 4-9は未記録
    result = winning_wide_combo_odds(finishing_order, wide_odds)
    assert result == {wide_pair_key("1", "9"): 5.6}


def test_winning_wide_combo_odds_requires_at_least_top3():
    with pytest.raises(ValueError):
        winning_wide_combo_odds(("1", "4"), {})


def test_clean_top3_hit_scores_all_bet_types(tmp_path):
    p = _prediction(horses=("8", "3", "5", "2"))
    freeze_four_horse_extraction(tmp_path, p)
    result = ExtractionResult(
        race_id=p.race_id,
        finishing_order=("8", "3", "5"),
        settled_at=datetime.now(timezone.utc).isoformat(),
        stake_yen=1000,
        return_yen=1450,
    )
    result_path, eval_path = settle_four_horse_extraction(tmp_path, p, result)
    assert result_path.exists()
    assert eval_path.exists()
    evaluation = evaluate_four_horse_extraction(p, result)
    assert evaluation.winner_captured is True
    assert evaluation.top3_hits == 3
    assert evaluation.exact_top3_set_hit is True
    assert evaluation.umaren_hit is True
    assert evaluation.wide_hit is True
    assert evaluation.sanrenpuku_hit is True
    assert evaluation.sanrentan_hit is True
    assert evaluation.failure_categories == ()
    assert evaluation.recovery_rate == 1.45


def test_sanrentan_hit_survives_json_round_trip_with_list_typed_horses():
    """settle_four_horse_extractionの実運用では、凍結済みPRE_RACEをjson.loads()で
    読み戻してFourHorseExtraction(**body)で再構築する(本ファイルの他テストと違い
    タプルのリテラルを直接渡さない)。dataclassは型ヒントを実行時に強制しないため、
    JSON由来のhorsesはlistのまま保持され、tuple化し忘れるとsanrentan_hitが
    list != tupleで常にFalseになる回帰バグが2026-10-11に発覚した。"""
    import json as _json
    from dataclasses import asdict as _asdict

    p = _prediction(horses=("8", "3", "5", "2"))
    reloaded = FourHorseExtraction(**_json.loads(_json.dumps(_asdict(p), default=str)))
    assert isinstance(reloaded.horses, list)  # JSON往復でlistになっていることの前提確認

    result = ExtractionResult(
        race_id=reloaded.race_id,
        finishing_order=("8", "3", "5"),
        settled_at=datetime.now(timezone.utc).isoformat(),
    )
    evaluation = evaluate_four_horse_extraction(reloaded, result)
    assert evaluation.sanrentan_hit is True


def test_second_pick_finishing_fourth_fails_umaren_and_wide_despite_capturing_all_top3():
    """ユーザー報告の「1着4着」パターン: 馬連の相手候補が実際は4着で券外になり、
    3着以内馬はすべて予想4頭の中にいたのに主要券種が軒並み不的中になる。"""
    p = _prediction(horses=("8", "2", "3", "5"))
    result = ExtractionResult(
        race_id=p.race_id,
        finishing_order=("8", "3", "5", "2"),
        settled_at=datetime.now(timezone.utc).isoformat(),
    )
    evaluation = evaluate_four_horse_extraction(p, result)
    assert evaluation.winner_captured is True
    assert evaluation.top3_hits == 2
    assert evaluation.umaren_hit is False
    assert evaluation.wide_hit is False
    assert evaluation.sanrenpuku_hit is False
    assert evaluation.sanrentan_hit is False
    assert "抽出漏れ" not in evaluation.failure_categories
    assert "着順僅差の取りこぼし" in evaluation.failure_categories
    assert evaluation.near_miss_horses == ("2",)


def test_missing_actual_top3_is_classified_as_extraction_failure():
    p = _prediction(horses=("8", "3", "5", "2"))
    result = ExtractionResult(
        race_id=p.race_id,
        finishing_order=("14", "8", "3"),
        settled_at=datetime.now(timezone.utc).isoformat(),
    )
    evaluation = evaluate_four_horse_extraction(p, result)
    assert evaluation.top3_hits == 2
    assert "抽出漏れ" in evaluation.failure_categories


def test_extraction_method_is_fixed_and_distinct_from_pca():
    p = _prediction()
    assert p.method == EXTRACTION_METHOD == "photo_prompt_manual_extraction"
