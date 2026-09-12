"""Layer 1: real-time all-ticket odds leading-signal model.

This module is dormant until a reliable JRA real-time feed is connected.  It
already defines the data contract and feature extraction path so that the
collector can be plugged in later without redesigning the prediction layer.

The PCA core uses the fixed 0.1 recent / 0.9 prior regularization implemented
in :mod:`racing_lambda.regularized_pca`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .regularized_pca import RacingRegularizedPCA


class BetType(str, Enum):
    WIN = "win"
    PLACE = "place"
    QUINELLA = "quinella"
    EXACTA = "exacta"
    WIDE = "wide"
    TRIO = "trio"
    TRIFECTA = "trifecta"


ALL_BET_TYPES = tuple(BetType)


@dataclass(frozen=True)
class OddsSnapshot:
    race_id: str
    horse_id: str
    captured_at: datetime
    implied_support: Mapping[BetType, float]

    def __post_init__(self) -> None:
        if not self.race_id.strip() or not self.horse_id.strip():
            raise ValueError("race_id and horse_id are required")
        if self.captured_at.tzinfo is None:
            raise ValueError("captured_at must include a timezone")
        for bet_type, value in self.implied_support.items():
            if bet_type not in ALL_BET_TYPES:
                raise ValueError(f"unsupported bet type: {bet_type}")
            if not np.isfinite(value) or value < 0:
                raise ValueError("implied_support must contain finite non-negative values")


@dataclass(frozen=True)
class MarketSignalResult:
    horse_id: str
    anomaly_score: float
    feature_count: int
    snapshot_count: int
    realtime_ready: bool


def _safe_relative_change(first: float, last: float) -> float:
    if first <= 1e-12:
        return 0.0
    return float((last - first) / first)


def extract_market_features(snapshots: Iterable[OddsSnapshot]) -> pd.DataFrame:
    """Build per-horse all-ticket temporal features from chronological snapshots.

    Features include level, start-to-latest change, last-step change, volatility,
    and disagreement versus win support for every available ticket type.
    """
    rows = list(snapshots)
    if not rows:
        raise ValueError("at least one odds snapshot is required")
    race_ids = {row.race_id for row in rows}
    if len(race_ids) != 1:
        raise ValueError("all snapshots must belong to one race")

    by_horse: dict[str, list[OddsSnapshot]] = {}
    for row in rows:
        by_horse.setdefault(row.horse_id, []).append(row)

    records: dict[str, dict[str, float]] = {}
    for horse_id, horse_rows in by_horse.items():
        horse_rows.sort(key=lambda item: item.captured_at)
        feature_row: dict[str, float] = {}
        for bet_type in ALL_BET_TYPES:
            series = [float(row.implied_support.get(bet_type, np.nan)) for row in horse_rows]
            available = np.asarray([value for value in series if np.isfinite(value)], dtype=float)
            if available.size == 0:
                level = change = last_step = volatility = 0.0
            else:
                level = float(available[-1])
                change = _safe_relative_change(float(available[0]), float(available[-1]))
                last_step = (
                    _safe_relative_change(float(available[-2]), float(available[-1]))
                    if available.size >= 2 else 0.0
                )
                volatility = float(np.std(available, ddof=0))
            prefix = bet_type.value
            feature_row[f"{prefix}_level"] = level
            feature_row[f"{prefix}_change"] = change
            feature_row[f"{prefix}_last_step"] = last_step
            feature_row[f"{prefix}_volatility"] = volatility

        win_level = feature_row["win_level"]
        for bet_type in ALL_BET_TYPES:
            if bet_type is BetType.WIN:
                continue
            feature_row[f"{bet_type.value}_vs_win"] = feature_row[f"{bet_type.value}_level"] - win_level

        non_win_changes = [
            feature_row[f"{bet_type.value}_change"]
            for bet_type in ALL_BET_TYPES if bet_type is not BetType.WIN
        ]
        feature_row["cross_ticket_change_mean"] = float(np.mean(non_win_changes))
        feature_row["cross_ticket_change_std"] = float(np.std(non_win_changes, ddof=0))
        feature_row["market_breadth"] = float(
            np.mean([1.0 if change > 0 else 0.0 for change in non_win_changes])
        )
        records[horse_id] = feature_row

    return pd.DataFrame.from_dict(records, orient="index").sort_index(axis=1)


class RealtimeMarketLeadingSignal:
    """All-ticket layer using regularized PCA anomaly detection.

    Keep ``enabled=False`` until the production collector can supply sufficiently
    dense, timestamped JRA market snapshots.
    """

    def __init__(self, enabled: bool = False, variance_target: float = 0.90) -> None:
        self.enabled = bool(enabled)
        self.model = RacingRegularizedPCA(variance_target=variance_target)

    def fit(self, recent_features: pd.DataFrame, prior_features: pd.DataFrame) -> "RealtimeMarketLeadingSignal":
        self.model.fit(recent_features, prior_features)
        return self

    def score(self, snapshots: Iterable[OddsSnapshot]) -> list[MarketSignalResult]:
        rows = list(snapshots)
        if not self.enabled:
            raise RuntimeError("realtime market leading-signal layer is disabled")
        features = extract_market_features(rows)
        anomaly = self.model.anomaly_score(features)
        counts: dict[str, int] = {}
        for row in rows:
            counts[row.horse_id] = counts.get(row.horse_id, 0) + 1
        results = [
            MarketSignalResult(
                horse_id=horse_id,
                anomaly_score=float(anomaly.loc[horse_id]),
                feature_count=int(features.shape[1]),
                snapshot_count=counts[horse_id],
                realtime_ready=True,
            )
            for horse_id in features.index
        ]
        return sorted(results, key=lambda result: (-result.anomaly_score, result.horse_id))
