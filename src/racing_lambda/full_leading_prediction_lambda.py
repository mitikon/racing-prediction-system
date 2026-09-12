"""本格先行予測λ — JRA公式PRE_RACEスナップショット学習経路。

This module is the full horse-racing leading-prediction path.
It is deliberately separate from:
- 簡易式先行予測λ (``SimpleLeadingSignalLambdaV02``)
- 先行シグナル予測λ / 部分空間正則化PCA (``src/leading_signal_lambda``)

The model never fetches JRA pages itself. It only consumes immutable
``OfficialSnapshot`` objects created from public JRA pages by a permitted
collection/manual-save path. RESULT snapshots are rejected from model input,
so same-race hindsight cannot enter PRE_RACE learning or scoring.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Mapping

import pandas as pd

from .jra_official_free_ingestion import OfficialSnapshot
from .realtime_market_leading_signal import (
    BetType,
    MarketSignalResult,
    OddsSnapshot,
    RealtimeMarketLeadingSignal,
    extract_market_features,
)

PUBLIC_JRA_MARKET_KEY = "market_support"
_MIN_VARIANCE = 1e-12


def _bet_type(value: str) -> BetType:
    try:
        return BetType(value)
    except ValueError as exc:
        raise ValueError(f"unsupported bet type in JRA snapshot: {value}") from exc


def odds_snapshots_from_official(
    snapshots: Sequence[OfficialSnapshot],
) -> list[OddsSnapshot]:
    """Convert normalized public-JRA PRE_RACE snapshots to market snapshots.

    Expected normalized payload shape for every observation::

        {
          "market_support": [
            {"horse_id": "1", "support": {"win": 0.12, "place": 0.18}},
            ...
          ]
        }

    Missing ticket types are allowed because free public pages may not expose
    every field in every captured observation. Values are support measures,
    not required to sum to one.
    """
    if not snapshots:
        raise ValueError("at least one PRE_RACE JRA snapshot is required")
    race_ids = {snapshot.race_id for snapshot in snapshots}
    if len(race_ids) != 1:
        raise ValueError("all snapshots in one conversion must belong to one race")

    rows: list[OddsSnapshot] = []
    for snapshot in snapshots:
        if snapshot.phase != "PRE_RACE":
            raise ValueError("RESULT snapshots cannot enter 本格先行予測λ")
        captured_at = datetime.fromisoformat(snapshot.observed_at)
        if captured_at.tzinfo is None:
            raise ValueError("official snapshot observed_at must be timezone-aware")
        market_rows = snapshot.payload.get(PUBLIC_JRA_MARKET_KEY)
        if not isinstance(market_rows, list) or not market_rows:
            raise ValueError(
                f"PRE_RACE payload requires non-empty {PUBLIC_JRA_MARKET_KEY!r}"
            )
        for row in market_rows:
            if not isinstance(row, Mapping):
                raise ValueError("market_support rows must be mappings")
            horse_id = str(row.get("horse_id", "")).strip()
            if not horse_id:
                raise ValueError("market_support horse_id is required")
            support = row.get("support")
            if not isinstance(support, Mapping) or not support:
                raise ValueError("market_support support mapping is required")
            implied_support: dict[BetType, float] = {}
            for key, value in support.items():
                bet = _bet_type(str(key))
                implied_support[bet] = float(value)
            rows.append(
                OddsSnapshot(
                    race_id=snapshot.race_id,
                    horse_id=horse_id,
                    captured_at=captured_at,
                    implied_support=implied_support,
                )
            )
    return rows


def build_jra_training_frame(
    races: Iterable[Sequence[OfficialSnapshot]],
) -> pd.DataFrame:
    """Build a multi-race feature frame from immutable PRE_RACE history."""
    frames: list[pd.DataFrame] = []
    for snapshots in races:
        odds_rows = odds_snapshots_from_official(snapshots)
        features = extract_market_features(odds_rows)
        race_id = odds_rows[0].race_id
        features.index = [f"{race_id}:{horse_id}" for horse_id in features.index]
        frames.append(features)
    if not frames:
        raise ValueError("at least one race is required for JRA learning")
    frame = pd.concat(frames, axis=0)
    if frame.index.has_duplicates:
        raise ValueError("duplicate race/horse rows in JRA training history")
    return frame.sort_index(axis=1)


class FullLeadingPredictionLambda:
    """本格先行予測λ.

    The existing all-ticket regularized-PCA market model is retained as the
    prediction core. This facade gives it an unambiguous name and adds a
    guarded learning path from free official JRA PRE_RACE observations.
    """

    def __init__(self, *, enabled: bool = False, variance_target: float = 0.90) -> None:
        self._core = RealtimeMarketLeadingSignal(
            enabled=enabled,
            variance_target=variance_target,
        )

    @property
    def enabled(self) -> bool:
        return self._core.enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._core.enabled = bool(value)

    @property
    def model(self):
        return self._core.model

    def fit_from_jra_history(
        self,
        *,
        recent_races: Iterable[Sequence[OfficialSnapshot]],
        prior_races: Iterable[Sequence[OfficialSnapshot]],
    ) -> "FullLeadingPredictionLambda":
        """Learn only from frozen PRE_RACE JRA observations.

        RESULT data is structurally rejected before feature extraction.
        The underlying racing PCA keeps its fixed 0.10 recent / 0.90 prior
        correlation regularization; this method does not alter that logic.

        Public snapshots can legitimately omit ticket types. Features that are
        constant in either training window are removed before PCA so missing
        ticket types do not create zero-variance failures.
        """
        recent = build_jra_training_frame(recent_races)
        prior = build_jra_training_frame(prior_races)
        if list(recent.columns) != list(prior.columns):
            raise ValueError("recent and prior JRA history must use identical features")

        usable_columns = [
            column
            for column in recent.columns
            if float(recent[column].std(ddof=0)) > _MIN_VARIANCE
            and float(prior[column].std(ddof=0)) > _MIN_VARIANCE
        ]
        if len(usable_columns) < 2:
            raise ValueError(
                "JRA learning requires at least two non-constant market features"
            )
        self.feature_columns_ = tuple(usable_columns)
        self._core.fit(recent.loc[:, usable_columns], prior.loc[:, usable_columns])
        return self

    def score_jra_race(
        self,
        snapshots: Sequence[OfficialSnapshot],
    ) -> list[MarketSignalResult]:
        """Score one race using PRE_RACE observations only."""
        if not self.enabled:
            raise RuntimeError("本格先行予測λ is disabled")
        if not hasattr(self, "feature_columns_"):
            raise RuntimeError("fit_from_jra_history must be called before scoring")

        rows = odds_snapshots_from_official(snapshots)
        features = extract_market_features(rows)
        selected = features.reindex(columns=list(self.feature_columns_), fill_value=0.0)
        anomaly = self.model.anomaly_score(selected)
        counts: dict[str, int] = {}
        for row in rows:
            counts[row.horse_id] = counts.get(row.horse_id, 0) + 1
        results = [
            MarketSignalResult(
                horse_id=horse_id,
                anomaly_score=float(anomaly.loc[horse_id]),
                feature_count=len(self.feature_columns_),
                snapshot_count=counts[horse_id],
                realtime_ready=True,
            )
            for horse_id in selected.index
        ]
        return sorted(results, key=lambda result: (-result.anomaly_score, result.horse_id))


FULL_LEADING_PREDICTION_NAME = "本格先行予測λ"
SIMPLE_LEADING_PREDICTION_NAME = "簡易式先行予測λ"
