from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .schema import HorseEntry, PredictionRow


@dataclass(frozen=True)
class OddsDistortion:
    rank: int
    horse_id: str
    horse_name: str
    model_probability: float
    market_probability: float
    distortion: float


def _softmax_scores(rows: Sequence[PredictionRow]) -> dict[str, float]:
    # Scores are bounded; a fixed temperature creates stable relative probabilities.
    exponents = {row.horse_id: pow(2.718281828459045, row.score / 0.12) for row in rows}
    total = sum(exponents.values())
    return {key: value / total for key, value in exponents.items()}


def rank_odds_distortion(
    entries: Iterable[HorseEntry], prediction: Sequence[PredictionRow]
) -> list[OddsDistortion]:
    entries = list(entries)
    odds_entries = [entry for entry in entries if entry.win_odds is not None]
    if len(odds_entries) != len(entries):
        raise ValueError("win odds are required for every horse")
    predicted_ids = {row.horse_id for row in prediction}
    if predicted_ids != {entry.horse_id for entry in entries}:
        raise ValueError("prediction and odds entries must contain the same horses")

    raw_market = {entry.horse_id: 1.0 / float(entry.win_odds) for entry in entries}
    overround = sum(raw_market.values())
    market = {key: value / overround for key, value in raw_market.items()}
    model = _softmax_scores(prediction)
    names = {entry.horse_id: entry.horse_name for entry in entries}
    ranked = sorted(entries, key=lambda entry: (-(model[entry.horse_id] - market[entry.horse_id]), entry.horse_id))
    return [
        OddsDistortion(
            rank=index,
            horse_id=entry.horse_id,
            horse_name=names[entry.horse_id],
            model_probability=round(model[entry.horse_id], 8),
            market_probability=round(market[entry.horse_id], 8),
            distortion=round(model[entry.horse_id] - market[entry.horse_id], 8),
        )
        for index, entry in enumerate(ranked, start=1)
    ]
