from datetime import datetime, timedelta, timezone

import pytest

from racing_lambda.four_horse_extraction import (
    EXTRACTION_METHOD,
    ExtractionResult,
    FourHorseExtraction,
    evaluate_four_horse_extraction,
    freeze_four_horse_extraction,
    settle_four_horse_extraction,
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
