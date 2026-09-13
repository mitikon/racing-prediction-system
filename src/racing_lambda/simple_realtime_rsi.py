"""簡易式先行予測λ向け・3時点全券種オッズRSI自己学習レイヤー。

想定時点:
- 発走30分前
- 発走15分前
- 発走5分前

JRA公式のPRE_RACE公開データを3回だけ取得した場合でも、
単勝だけでなく全券種の変化率・直近変化・加速度・券種間乖離を
簡易式先行予測λへ渡せるようにする。

RESULTは学習ラベルにのみ使用し、同一レースのPRE_RACE特徴量へ
混入させない。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .realtime_market_leading_signal import (
    ALL_BET_TYPES,
    BetType,
    OddsSnapshot,
)
from .schema import OfficialResult


EXPECTED_MINUTES_BEFORE_START = (30, 15, 5)
SIMPLE_RSI_THREE_SNAPSHOT_VERSION = "simple-leading-rsi-3snapshot-v1"


def _safe_relative_change(first: float, last: float) -> float:
    if first <= 1e-12:
        return 0.0
    return float((last - first) / first)


def build_three_snapshot_features(
    snapshots: Sequence[OddsSnapshot],
) -> pd.DataFrame:
    """Create per-horse dynamic features from exactly three PRE_RACE snapshots.

    The caller is responsible for capturing the observations at roughly
    30/15/5 minutes before post time. This function requires exactly three
    chronological observations per horse so accidental one-point fallbacks
    cannot masquerade as real-time learning.
    """
    rows = list(snapshots)
    if not rows:
        raise ValueError("three PRE_RACE odds snapshots are required")
    race_ids = {row.race_id for row in rows}
    if len(race_ids) != 1:
        raise ValueError("all snapshots must belong to one race")

    by_horse: dict[str, list[OddsSnapshot]] = {}
    for row in rows:
        by_horse.setdefault(row.horse_id, []).append(row)

    records: dict[str, dict[str, float]] = {}
    for horse_id, horse_rows in by_horse.items():
        horse_rows.sort(key=lambda item: item.captured_at)
        if len(horse_rows) != 3:
            raise ValueError(
                f"horse {horse_id} requires exactly three chronological snapshots"
            )
        if len({row.captured_at for row in horse_rows}) != 3:
            raise ValueError(f"horse {horse_id} snapshot timestamps must be unique")

        feature_row: dict[str, float] = {}
        for bet_type in ALL_BET_TYPES:
            values = np.asarray(
                [
                    float(row.implied_support.get(bet_type, np.nan))
                    for row in horse_rows
                ],
                dtype=float,
            )
            available = np.isfinite(values)
            prefix = bet_type.value
            feature_row[f"simple_rsi_{prefix}_coverage"] = float(available.mean())
            if int(available.sum()) != 3:
                feature_row[f"simple_rsi_{prefix}_level"] = 0.0
                feature_row[f"simple_rsi_{prefix}_change_30_to_5"] = 0.0
                feature_row[f"simple_rsi_{prefix}_change_30_to_15"] = 0.0
                feature_row[f"simple_rsi_{prefix}_change_15_to_5"] = 0.0
                feature_row[f"simple_rsi_{prefix}_acceleration"] = 0.0
                feature_row[f"simple_rsi_{prefix}_volatility"] = 0.0
                continue

            v30, v15, v5 = [float(x) for x in values]
            d1 = _safe_relative_change(v30, v15)
            d2 = _safe_relative_change(v15, v5)
            feature_row[f"simple_rsi_{prefix}_level"] = v5
            feature_row[f"simple_rsi_{prefix}_change_30_to_5"] = _safe_relative_change(v30, v5)
            feature_row[f"simple_rsi_{prefix}_change_30_to_15"] = d1
            feature_row[f"simple_rsi_{prefix}_change_15_to_5"] = d2
            feature_row[f"simple_rsi_{prefix}_acceleration"] = d2 - d1
            feature_row[f"simple_rsi_{prefix}_volatility"] = float(np.std(values, ddof=0))

        win_last = feature_row.get("simple_rsi_win_level", 0.0)
        ticket_last_levels: list[float] = []
        ticket_last_changes: list[float] = []
        for bet_type in ALL_BET_TYPES:
            if bet_type is BetType.WIN:
                continue
            level = feature_row[f"simple_rsi_{bet_type.value}_level"]
            change = feature_row[f"simple_rsi_{bet_type.value}_change_15_to_5"]
            feature_row[f"simple_rsi_{bet_type.value}_vs_win"] = level - win_last
            ticket_last_levels.append(level)
            ticket_last_changes.append(change)

        feature_row["simple_rsi_cross_ticket_change_mean"] = float(
            np.mean(ticket_last_changes)
        )
        feature_row["simple_rsi_cross_ticket_change_std"] = float(
            np.std(ticket_last_changes, ddof=0)
        )
        feature_row["simple_rsi_market_breadth"] = float(
            np.mean([1.0 if value > 0 else 0.0 for value in ticket_last_changes])
        )
        feature_row["simple_rsi_cross_ticket_level_std"] = float(
            np.std(ticket_last_levels, ddof=0)
        )
        records[horse_id] = feature_row

    return pd.DataFrame.from_dict(records, orient="index").sort_index(axis=1)


def build_three_snapshot_training_frame(
    races: Iterable[Sequence[OddsSnapshot]],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for snapshots in races:
        snapshots = list(snapshots)
        if not snapshots:
            continue
        frame = build_three_snapshot_features(snapshots)
        race_id = snapshots[0].race_id
        frame.index = [f"{race_id}:{horse_id}" for horse_id in frame.index]
        frames.append(frame)
    if not frames:
        raise ValueError("at least one three-snapshot race is required")
    result = pd.concat(frames, axis=0)
    if result.index.has_duplicates:
        raise ValueError("duplicate race/horse rows in three-snapshot history")
    return result.sort_index(axis=1)


def build_top3_labels(index: pd.Index, results: Iterable[OfficialResult]) -> pd.Series:
    result_map = {result.race_id: result for result in results}
    labels: dict[str, float] = {}
    for key in index.astype(str):
        race_id, horse_id = key.rsplit(":", 1)
        result = result_map.get(race_id)
        if result is None:
            continue
        labels[key] = 1.0 if horse_id in result.finishing_order[:3] else 0.0
    return pd.Series(labels, dtype=float, name="top3_result").reindex(index)


@dataclass(frozen=True)
class SimpleRsiLearningSummary:
    rows: int
    positive_rows: int
    feature_count: int


@dataclass(frozen=True)
class SimpleRealtimeRsiSignal:
    horse_id: str
    top3_probability: float
    bug_score: float
    feature_count: int
    snapshot_count: int = 3


class SimpleThreeSnapshotRsiLearner:
    """Outcome learner for the simple mode's 30/15/5-minute market motion.

    It learns only from already completed historical races. Current-race
    result data never enters :meth:`score`.
    """

    def __init__(self, ridge: float = 1.0) -> None:
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        self.ridge = float(ridge)

    def fit(
        self,
        historical_races: Iterable[Sequence[OddsSnapshot]],
        historical_results: Iterable[OfficialResult],
    ) -> "SimpleThreeSnapshotRsiLearner":
        features = build_three_snapshot_training_frame(historical_races)
        labels = build_top3_labels(features.index, historical_results)
        aligned = features.replace([np.inf, -np.inf], np.nan)
        valid = labels.notna() & aligned.notna().all(axis=1)
        aligned = aligned.loc[valid]
        target = labels.loc[valid].astype(float)
        if len(aligned) < 8 or target.nunique() < 2:
            raise ValueError(
                "three-snapshot RSI learning needs at least eight labeled rows and two classes"
            )
        variable = aligned.std(axis=0, ddof=0) > 1e-12
        aligned = aligned.loc[:, variable]
        if aligned.shape[1] < 2:
            raise ValueError("at least two variable market-motion features are required")

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
        self.summary_ = SimpleRsiLearningSummary(
            rows=len(target),
            positive_rows=int(target.sum()),
            feature_count=len(self.columns_),
        )
        return self

    def score(self, snapshots: Sequence[OddsSnapshot]) -> list[SimpleRealtimeRsiSignal]:
        if not hasattr(self, "coefficients_"):
            raise RuntimeError("fit must be called before three-snapshot RSI scoring")
        features = build_three_snapshot_features(snapshots)
        selected = features.reindex(columns=list(self.columns_), fill_value=0.0)
        if not np.isfinite(selected.to_numpy(dtype=float)).all():
            raise ValueError("three-snapshot scoring features must be finite")
        z = (selected - self.mean_) / self.std_
        design = np.column_stack([np.ones(len(z)), z.to_numpy(dtype=float)])
        probability = np.clip(design @ self.coefficients_, 0.0, 1.0)

        # Bug score emphasizes both learned top3 probability and unusual
        # late cross-ticket movement. It is still only a ranking signal.
        motion = features["simple_rsi_cross_ticket_change_std"].reindex(
            selected.index
        ).to_numpy(dtype=float)
        motion_unit = 1.0 - np.exp(-np.maximum(motion, 0.0))
        bug = np.clip(0.75 * probability + 0.25 * motion_unit, 0.0, 1.0)

        results = [
            SimpleRealtimeRsiSignal(
                horse_id=horse_id,
                top3_probability=float(probability[i]),
                bug_score=float(bug[i]),
                feature_count=len(self.columns_),
            )
            for i, horse_id in enumerate(selected.index)
        ]
        return sorted(results, key=lambda row: (-row.bug_score, row.horse_id))
