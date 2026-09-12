from datetime import datetime, timedelta, timezone

from racing_lambda import (
    HorseEntry, LeadingSignalPolicy, OfficialResult, RaceContext,
    build_prediction, evaluate_prediction, freeze_prediction,
    load_frozen_prediction, rank_odds_distortion,
)
from racing_lambda.formatter import format_japanese_report


def entries():
    return [
        HorseEntry("1", "Alpha", .90, .80, .70, .75, .95, .65, 2.5, 2),
        HorseEntry("2", "Beta", .75, .85, .82, .70, .65, .80, 4.0, 80),
        HorseEntry("3", "Gamma", .60, .62, .65, .85, .70, .72, 8.0, 30),
    ]


def test_jockey_small_sample_is_shrunk():
    rows = build_prediction(entries())
    alpha = next(row for row in rows if row.horse_id == "1")
    assert alpha.component_scores["jockey_skill"] < .60


def test_leading_signal_is_disabled_by_default():
    rows = build_prediction(entries(), leading=LeadingSignalPolicy(values={"3": 1.0}))
    assert not any(row.leading_signal_used for row in rows)


def test_enabled_leading_signal_is_auditable():
    policy = LeadingSignalPolicy(enabled=True, weight=.25, values={"3": 1.0}, reason="test")
    rows = build_prediction(entries(), leading=policy)
    assert next(row for row in rows if row.horse_id == "3").leading_signal_used


def test_odds_ranking_is_separate_and_normalized():
    odds = rank_odds_distortion(entries(), build_prediction(entries()))
    assert [row.rank for row in odds] == [1, 2, 3]
    assert abs(sum(row.market_probability for row in odds) - 1.0) < 1e-7


def test_freeze_refuses_post_time_and_overwrite(tmp_path):
    now = datetime.now(timezone.utc)
    race = RaceContext("R1", "Test", now + timedelta(hours=1))
    rows = build_prediction(entries())
    odds = rank_odds_distortion(entries(), rows)
    path = freeze_prediction(tmp_path / "R1.json", race, rows, odds, frozen_at=now)
    assert load_frozen_prediction(path).race["race_id"] == "R1"
    try:
        freeze_prediction(path, race, rows, odds, frozen_at=now)
    except FileExistsError:
        pass
    else:
        raise AssertionError("frozen record must not be overwritten")


def test_tampering_is_detected(tmp_path):
    now = datetime.now(timezone.utc)
    race = RaceContext("R2", "Test", now + timedelta(hours=1))
    rows = build_prediction(entries())
    odds = rank_odds_distortion(entries(), rows)
    path = freeze_prediction(tmp_path / "R2.json", race, rows, odds, frozen_at=now)
    text = path.read_text(encoding="utf-8").replace("Alpha", "Altered")
    path.write_text(text, encoding="utf-8")
    try:
        load_frozen_prediction(path)
    except ValueError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("tampering must be detected")


def test_evaluation_assigns_failure_category():
    rows = build_prediction(entries())
    odds = rank_odds_distortion(entries(), rows)
    report = evaluate_prediction(rows, odds, OfficialResult("R1", ("3", "2", "1")))
    assert report.failure_category in {"過大評価", "最終除外ミス"}
    assert report.top3_hits == 3


def test_report_keeps_two_rankings_separate():
    rows = build_prediction(entries())
    odds = rank_odds_distortion(entries(), rows)
    report = format_japanese_report(rows, odds)
    assert "## λ総合最終予想" in report
    assert "## オッズ歪み／最大バグ順位予想" in report
