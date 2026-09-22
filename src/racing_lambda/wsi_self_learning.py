"""競馬予想システム開発・本格先行予測λ専用のWSI（Wilder Strength Index）自己学習部品。

WSI is the Wilder Strength Index (the classic Relative Strength Index
formula), never RSI (Recursive Self-Improvement, see
recursive_self_improvement.py)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .schema import OfficialResult


RACING_WSI_PERIODS = (5, 7, 14, 21)
RACING_WSI_FEATURE_VERSION = "racing-wsi-self-learning-v1"


def calculate_support_wsi(values: pd.Series, period: int) -> pd.Series:
    """Calculate the Wilder Strength Index from chronological implied-support observations."""
    if period < 2:
        raise ValueError("WSI period must be at least 2")
    numeric = pd.to_numeric(values, errors="coerce")
    delta = numeric.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    average_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    average_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    relative_strength = average_gain / average_loss
    wsi = 100.0 - 100.0 / (1.0 + relative_strength)
    wsi = wsi.mask((average_gain == 0.0) & (average_loss == 0.0), 50.0)
    wsi = wsi.mask((average_gain > 0.0) & (average_loss == 0.0), 100.0)
    return wsi


def latest_wsi_state(values: Iterable[float], period: int) -> dict[str, float]:
    """Return outcome-neutral WSI state; 30/70 never directly means buy/sell."""
    series = pd.Series(list(values), dtype=float).dropna().reset_index(drop=True)
    wsi = calculate_support_wsi(series, period).dropna()
    if wsi.empty:
        return {
            "level": 0.0,
            "velocity3": 0.0,
            "cross50": 0.0,
            "extreme_state": 0.0,
            "available": 0.0,
        }
    latest = float(wsi.iloc[-1])
    previous = float(wsi.iloc[-2]) if len(wsi) >= 2 else latest
    velocity = (latest - float(wsi.iloc[-4])) / 100.0 if len(wsi) >= 4 else 0.0
    cross50 = 1.0 if latest >= 50.0 > previous else -1.0 if latest <= 50.0 < previous else 0.0
    if latest >= 70.0 and previous >= 70.0:
        extreme = 2.0
    elif latest <= 30.0 and previous <= 30.0:
        extreme = -2.0
    elif latest < 70.0 <= previous:
        extreme = -1.0
    elif latest > 30.0 >= previous:
        extreme = 1.0
    elif latest >= 70.0:
        extreme = 0.5
    elif latest <= 30.0:
        extreme = -0.5
    else:
        extreme = 0.0
    return {
        "level": (latest - 50.0) / 50.0,
        "velocity3": velocity,
        "cross50": cross50,
        "extreme_state": extreme,
        "available": 1.0,
    }


def build_result_labels(
    index: pd.Index,
    results: Iterable[OfficialResult],
) -> pd.Series:
    """Create prior-race top-three labels without modifying PRE_RACE features."""
    result_map = {result.race_id: result for result in results}
    labels: dict[str, float] = {}
    for value in index.astype(str):
        race_id, horse_id = value.rsplit(":", 1)
        result = result_map.get(race_id)
        if result is None or horse_id not in result.finishing_order:
            continue
        labels[value] = 1.0 if horse_id in result.finishing_order[:3] else 0.0
    return pd.Series(labels, dtype=float, name="top3_result").reindex(index)


@dataclass(frozen=True)
class WsiLearningSummary:
    rows: int
    positive_rows: int
    feature_count: int


class RacingWsiOutcomeLearner:
    """Ridge learner that links past PRE_RACE WSI states to later results."""

    def __init__(self, ridge: float = 1.0) -> None:
        if ridge <= 0.0:
            raise ValueError("ridge must be positive")
        self.ridge = float(ridge)

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "RacingWsiOutcomeLearner":
        wsi_columns = [column for column in features if column.startswith("wsi_")]
        aligned = features.loc[:, wsi_columns].replace([np.inf, -np.inf], np.nan)
        valid = labels.notna() & aligned.notna().all(axis=1)
        aligned = aligned.loc[valid]
        target = labels.loc[valid].astype(float)
        if len(aligned) < 6 or target.nunique() < 2:
            raise ValueError("WSI result learning needs at least six labeled rows and two classes")
        variable = aligned.std(axis=0, ddof=0) > 1e-12
        aligned = aligned.loc[:, variable]
        if aligned.shape[1] < 1:
            raise ValueError("WSI result learning needs at least one variable WSI feature")
        self.columns_ = tuple(aligned.columns)
        self.mean_ = aligned.mean(axis=0)
        self.std_ = aligned.std(axis=0, ddof=0)
        z = (aligned - self.mean_) / self.std_
        design = np.column_stack([np.ones(len(z)), z.to_numpy(dtype=float)])
        penalty = np.eye(design.shape[1]) * self.ridge
        penalty[0, 0] = 0.0
        self.coefficients_ = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ target.to_numpy(dtype=float),
        )
        self.summary_ = WsiLearningSummary(
            rows=len(target),
            positive_rows=int(target.sum()),
            feature_count=len(self.columns_),
        )
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not hasattr(self, "coefficients_"):
            raise RuntimeError("fit must be called before WSI prediction")
        selected = features.reindex(columns=list(self.columns_), fill_value=0.0)
        if not np.isfinite(selected.to_numpy(dtype=float)).all():
            raise ValueError("WSI prediction features must be finite")
        z = (selected - self.mean_) / self.std_
        design = np.column_stack([np.ones(len(z)), z.to_numpy(dtype=float)])
        prediction = np.clip(design @ self.coefficients_, 0.0, 1.0)
        return pd.Series(prediction, index=features.index, name="wsi_top3_probability")
