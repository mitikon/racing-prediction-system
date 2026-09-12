from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .odds import OddsDistortion
from .schema import OfficialResult, PredictionRow


@dataclass(frozen=True)
class EvaluationReport:
    race_id: str
    winner_hit: bool
    top3_hits: int
    top3_recall: float
    distortion_winner_hit: bool
    return_amount: float | None
    failure_category: str | None


def evaluate_prediction(
    prediction: Sequence[PredictionRow],
    distortion: Sequence[OddsDistortion],
    result: OfficialResult,
    stake: float | None = None,
) -> EvaluationReport:
    if not result.finishing_order:
        raise ValueError("official finishing order is required")
    predicted = [row.horse_id for row in sorted(prediction, key=lambda row: row.rank)]
    if not predicted:
        raise ValueError("prediction is required")
    actual_top3 = set(result.finishing_order[:3])
    top3_hits = len(set(predicted[:3]) & actual_top3)
    winner_hit = predicted[0] == result.finishing_order[0]
    distortion_first = min(distortion, key=lambda row: row.rank).horse_id if distortion else None

    failure = None
    if not winner_hit:
        winner = result.finishing_order[0]
        if winner not in predicted:
            failure = "抽出漏れ"
        elif predicted.index(winner) >= 3:
            failure = "最終除外ミス"
        else:
            failure = "過大評価"

    return_amount = None
    if stake is not None:
        if stake < 0:
            raise ValueError("stake cannot be negative")
        return_amount = float(result.payouts.get(predicted[0], 0.0))

    return EvaluationReport(
        race_id=result.race_id,
        winner_hit=winner_hit,
        top3_hits=top3_hits,
        top3_recall=top3_hits / min(3, len(actual_top3)),
        distortion_winner_hit=distortion_first == result.finishing_order[0],
        return_amount=return_amount,
        failure_category=failure,
    )
