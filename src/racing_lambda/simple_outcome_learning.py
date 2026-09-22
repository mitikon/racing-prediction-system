"""簡易式先行予測λの発走前順位と事後結果の監査。

これは結果からモデルを再学習する前段のラベル生成である。WSI（Wilder Strength Index）用の
3時点オッズがないレースを、WSI学習済みと表示しない。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class MissedHorse:
    race_id: str
    horse_id: str
    actual_place: int
    frozen_rank: int | None
    category: str
    pre_race_observation: str | None
    post_race_observation: str | None
    cause_status: str = "unverified"


def audit_outcomes(races: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit immutable PRE_RACE ranks; never re-rank using RESULT data.

    Missing pre-race features or three chronological odds snapshots explicitly
    block WSI parameter training. Descriptions supplied by a human are retained
    as observations, never promoted to proved causal explanations.
    """
    if not races:
        raise ValueError("at least one race is required")
    identifiers: set[str] = set()
    details: list[dict[str, Any]] = []
    misses: list[MissedHorse] = []
    labels: list[dict[str, Any]] = []
    for race in races:
        race_id = str(race["race_id"])
        if not race_id or race_id in identifiers:
            raise ValueError("race IDs must be non-empty and unique")
        identifiers.add(race_id)
        if not race.get("pre_race_source") or not race.get("official_result_source"):
            raise ValueError(f"{race_id}: both provenance sources are required")
        ranking = tuple(str(item) for item in race["pre_race_order"])
        actual = tuple(str(item) for item in race["actual_top3"])
        if len(ranking) < 5 or len(ranking) != len(set(ranking)):
            raise ValueError(f"{race_id}: full pre-race ranking must be unique")
        if len(actual) != 3 or len(set(actual)) != 3:
            raise ValueError(f"{race_id}: actual top three must be unique")
        if not set(actual).issubset(ranking):
            raise ValueError(f"{race_id}: actual horses must appear in the frozen full ranking")
        notes = race.get("notes", {})
        if not isinstance(notes, dict) or any(str(key) not in actual for key in notes):
            raise ValueError(f"{race_id}: notes must be keyed by actual top-three horse ID")
        for horse_id, note in notes.items():
            if not isinstance(note, dict) or set(note) - {
                "pre_race_observation", "post_race_observation"
            }:
                raise ValueError(f"{race_id}: notes must separate pre/post-race observations")
        hit3 = len(set(ranking[:3]) & set(actual))
        hit5 = len(set(ranking[:5]) & set(actual))
        labels.extend({
            "race_id": race_id,
            "horse_id": horse_id,
            "frozen_rank": rank,
            "top3_result_label": int(horse_id in actual),
        } for rank, horse_id in enumerate(ranking, 1))
        for place, horse_id in enumerate(actual, 1):
            frozen_rank = ranking.index(horse_id) + 1
            if frozen_rank <= 5:
                continue
            note = notes.get(horse_id, {})
            misses.append(MissedHorse(
                race_id=race_id,
                horse_id=horse_id,
                actual_place=place,
                frozen_rank=frozen_rank,
                category="top5_exclusion",
                pre_race_observation=note.get("pre_race_observation"),
                post_race_observation=note.get("post_race_observation"),
            ))
        details.append({
            "race_id": race_id,
            "winner_rank": ranking.index(actual[0]) + 1,
            "top3_hits": hit3,
            "top5_hits": hit5,
            "overrated_top3": [item for item in ranking[:3] if item not in actual],
            "odds_bug_top_candidate": str(race["odds_bug_top_candidate"])
            if race.get("odds_bug_top_candidate") is not None else None,
            "odds_bug_top_candidate_placed": (
                str(race["odds_bug_top_candidate"]) in actual
                if race.get("odds_bug_top_candidate") is not None else None
            ),
        })
    return {
        "schema": "simple-outcome-audit-v1",
        "model": "簡易式先行予測λ",
        "races": details,
        "total": {
            "race_count": len(details),
            "winner_rank1": sum(row["winner_rank"] == 1 for row in details),
            "winner_in_top5": sum(row["winner_rank"] <= 5 for row in details),
            "top3_hits": sum(row["top3_hits"] for row in details),
            "top5_hits": sum(row["top5_hits"] for row in details),
        },
        "excluded_podium_horses": [asdict(item) for item in misses],
        "next_race_training_labels": labels,
        "wsi_training": {
            "performed": False,
            "reason": "3時点の発走前オッズと全馬の定量特徴量が保存されていない",
            "next_step": "時点固定データを蓄積してから、後続レースだけで検証する",
        },
        "weight_changes": [],
    }


def audit_published_outcomes(races: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit a frozen ranking when only its published prefix was preserved.

    The unlisted horses deliberately receive no synthetic rank or negative
    training label.  Their official podium finishes are exclusion evidence for
    the next race only; they cannot be used to reconstruct the old prediction.
    """
    if not races:
        raise ValueError("at least one race is required")
    identifiers: set[str] = set()
    details: list[dict[str, Any]] = []
    misses: list[MissedHorse] = []
    labels: list[dict[str, Any]] = []
    for race in races:
        race_id = str(race["race_id"])
        if not race_id or race_id in identifiers:
            raise ValueError("race IDs must be non-empty and unique")
        identifiers.add(race_id)
        if not race.get("pre_race_source") or not race.get("official_result_source"):
            raise ValueError(f"{race_id}: both provenance sources are required")
        published = tuple(str(item) for item in race["published_order"])
        actual = tuple(str(item) for item in race["actual_top3"])
        field_size = int(race["field_size"])
        if not published or len(published) != len(set(published)):
            raise ValueError(f"{race_id}: published ranking must be non-empty and unique")
        if field_size < len(published) or field_size < 5:
            raise ValueError(f"{race_id}: field_size cannot be smaller than the published ranking")
        if len(actual) != 3 or len(actual) != len(set(actual)):
            raise ValueError(f"{race_id}: actual top three must be unique")
        if any(not horse_id for horse_id in (*published, *actual)):
            raise ValueError(f"{race_id}: horse IDs must be non-empty")
        bug_order = tuple(str(item) for item in race.get("odds_bug_order", ()))
        if len(bug_order) != len(set(bug_order)):
            raise ValueError(f"{race_id}: odds bug ranking must be unique")

        labels.extend({
            "race_id": race_id,
            "horse_id": horse_id,
            "frozen_rank": rank,
            "top3_result_label": int(horse_id in actual),
            "scope": "published_horses_only",
        } for rank, horse_id in enumerate(published, 1))
        for place, horse_id in enumerate(actual, 1):
            if horse_id in published:
                continue
            misses.append(MissedHorse(
                race_id=race_id,
                horse_id=horse_id,
                actual_place=place,
                frozen_rank=None,
                category="published_ranking_exclusion",
                pre_race_observation=None,
                post_race_observation=None,
            ))

        winner_rank = published.index(actual[0]) + 1 if actual[0] in published else None
        published_hits = len(set(published) & set(actual))
        top3_hits = len(set(published[:3]) & set(actual))
        if winner_rank is None:
            winner_classification = "winner_extraction_omission"
        elif winner_rank == 1:
            winner_classification = "winner_exact"
        else:
            winner_classification = "winner_ranking_error"
        details.append({
            "race_id": race_id,
            "published_depth": len(published),
            "winner_rank": winner_rank,
            "winner_classification": winner_classification,
            "top3_hits": top3_hits,
            "published_podium_hits": published_hits,
            "published_podium_complete": published_hits == 3,
            "overrated_top3": [item for item in published[:3] if item not in actual],
            "odds_bug_podium_hits": [item for item in bug_order if item in actual],
            "manual_formation_recovered": bool(race.get("manual_formation_recovered", False)),
        })

    return {
        "schema": "simple-published-outcome-audit-v1",
        "model": "簡易式先行予測λ",
        "freeze_policy": "PRE_RACE順位は不変。RESULTは別記録で、次レース以降にのみ利用する",
        "races": details,
        "total": {
            "race_count": len(details),
            "winner_rank1": sum(row["winner_rank"] == 1 for row in details),
            "winner_in_published": sum(row["winner_rank"] is not None for row in details),
            "winner_extraction_omissions": sum(
                row["winner_classification"] == "winner_extraction_omission" for row in details
            ),
            "top3_hits": sum(row["top3_hits"] for row in details),
            "published_podium_hits": sum(row["published_podium_hits"] for row in details),
            "published_podium_complete_races": sum(
                row["published_podium_complete"] for row in details
            ),
        },
        "excluded_podium_horses": [asdict(item) for item in misses],
        "next_race_training_labels": labels,
        "learning_evidence": {
            "accepted": True,
            "effective_from": "next_race_only",
            "scope": "published_horses_and_podium_exclusions",
        },
        "wsi_training": {
            "performed": False,
            "reason": "全馬の定量特徴量と30/15/5分の3時点オッズが保存されていない",
            "next_step": "同形式のPRE_RACE特徴量を8レース以上固定し、後続レースで候補を比較する",
        },
        "promotion_status": "NOT_EVALUABLE",
        "weight_changes": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit simple racing λ outcome labels")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = (
        audit_published_outcomes(payload["races"])
        if payload.get("frozen_scope") == "published_ranking_only"
        else audit_outcomes(payload["races"])
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as destination:
            destination.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
