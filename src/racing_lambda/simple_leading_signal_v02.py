"""簡易式先行シグナル予測λ v0.2.

2026-09-06 の実戦検証から得た規則を、既存の競馬予想パッケージへ
接続するための独立スコアラー。研究上の PCA 正則化 λ とは別の、
暫定的な運用配分である。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import exp
from typing import Mapping, Sequence


class Going(str, Enum):
    FIRM = "良"
    GOOD = "稍重"
    SOFT = "重"
    HEAVY = "不良"


class BugType(str, Enum):
    WIN = "勝利バグ"
    PLACE = "複勝バグ"
    NONE = "認定なし"


@dataclass(frozen=True)
class SimpleRaceContext:
    race_id: str
    surface: str
    distance_m: int
    going: Going
    opening_week: bool
    rain: bool
    projected_front_runners: int

    def __post_init__(self) -> None:
        if not self.race_id.strip():
            raise ValueError("race_id is required")
        if self.distance_m <= 0:
            raise ValueError("distance_m must be positive")
        if self.projected_front_runners < 0:
            raise ValueError("projected_front_runners cannot be negative")

    @property
    def wet(self) -> bool:
        return self.going in {Going.SOFT, Going.HEAVY}

    @property
    def wet_congested_pace(self) -> bool:
        return self.wet and self.projected_front_runners >= 4


@dataclass(frozen=True)
class SimpleHorseFeatures:
    horse_id: str
    horse_name: str
    odds: float
    age: int
    assigned_weight_kg: float
    body_weight_kg: int
    body_weight_change_kg: int
    predicted_position: int
    recent_top3_count: int
    recent_top5_count: int
    class_score: float
    going_score: float
    course_score: float
    jockey_place_rate: float
    jockey_win_return: float
    jockey_place_return: float
    days_since_last_run: int
    stakes_top5: bool = False
    is_front_runner: bool = False

    def __post_init__(self) -> None:
        if not self.horse_id.strip() or not self.horse_name.strip():
            raise ValueError("horse_id and horse_name are required")
        if self.odds <= 1.0:
            raise ValueError("odds must be greater than 1.0")
        if self.age <= 0 or self.body_weight_kg <= 0:
            raise ValueError("age and body weight must be positive")
        for key in ("class_score", "going_score", "course_score"):
            value = float(getattr(self, key))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{key} must be between 0 and 1")
        if self.predicted_position < 1:
            raise ValueError("predicted_position must be positive")


@dataclass(frozen=True)
class SimpleScoreBreakdown:
    horse_id: str
    horse_name: str
    race_ability: float
    pace_position: float
    going_course: float
    physical_condition: float
    odds_signal: float
    overall: float
    fair_win_probability: float
    market_win_probability: float
    value_gap: float
    corroborating_axes: tuple[str, ...]
    bug_type: BugType
    maximum_bug_eligible: bool


@dataclass(frozen=True)
class SimplePredictionOutput:
    lambda_overall_final: tuple[SimpleScoreBreakdown, ...]
    odds_distortion_bug_ranking: tuple[SimpleScoreBreakdown, ...]
    win_bug_ranking: tuple[SimpleScoreBreakdown, ...]
    place_bug_ranking: tuple[SimpleScoreBreakdown, ...]


class SimpleLeadingSignalLambdaV02:
    """実戦3レース検証後の暫定重みを使うランキングエンジン。"""

    WEIGHTS: Mapping[str, float] = {
        "race_ability": 0.35,
        "pace_position": 0.25,
        "going_course": 0.20,
        "physical_condition": 0.10,
        "odds_signal": 0.10,
    }

    def rank(
        self,
        context: SimpleRaceContext,
        horses: Sequence[SimpleHorseFeatures],
    ) -> SimplePredictionOutput:
        if len(horses) < 2:
            raise ValueError("at least two horses are required")
        horse_ids = [horse.horse_id for horse in horses]
        if len(horse_ids) != len(set(horse_ids)):
            raise ValueError("horse_id values must be unique")

        raw = [self._fundamental_scores(context, horse) for horse in horses]
        fair_probabilities = self._softmax_probabilities(
            [self._fundamental_total(item) for item in raw]
        )
        market_probabilities = self._normalised_market_probabilities(horses)

        scored: list[SimpleScoreBreakdown] = []
        for horse, parts, fair_probability in zip(horses, raw, fair_probabilities):
            market_probability = market_probabilities[horse.horse_id]
            value_gap = fair_probability - market_probability
            odds_signal = self._odds_signal(value_gap, market_probability)
            axes = self._corroborating_axes(horse, parts, value_gap)
            maximum_eligible = value_gap > 0 and len(axes) >= 3
            bug_type = self._bug_type(
                maximum_eligible, fair_probability, parts["race_ability"]
            )
            overall = (
                self.WEIGHTS["race_ability"] * parts["race_ability"]
                + self.WEIGHTS["pace_position"] * parts["pace_position"]
                + self.WEIGHTS["going_course"] * parts["going_course"]
                + self.WEIGHTS["physical_condition"]
                * parts["physical_condition"]
                + self.WEIGHTS["odds_signal"] * odds_signal
            )
            scored.append(
                SimpleScoreBreakdown(
                    horse_id=horse.horse_id,
                    horse_name=horse.horse_name,
                    race_ability=round(parts["race_ability"], 8),
                    pace_position=round(parts["pace_position"], 8),
                    going_course=round(parts["going_course"], 8),
                    physical_condition=round(parts["physical_condition"], 8),
                    odds_signal=round(odds_signal, 8),
                    overall=round(overall, 8),
                    fair_win_probability=round(fair_probability, 8),
                    market_win_probability=round(market_probability, 8),
                    value_gap=round(value_gap, 8),
                    corroborating_axes=axes,
                    bug_type=bug_type,
                    maximum_bug_eligible=maximum_eligible,
                )
            )

        overall_ranking = tuple(
            sorted(scored, key=lambda item: (-item.overall, item.horse_id))
        )
        bug_ranking = tuple(
            sorted(
                (item for item in scored if item.maximum_bug_eligible),
                key=lambda item: (-item.value_gap, -item.overall, item.horse_id),
            )
        )
        return SimplePredictionOutput(
            lambda_overall_final=overall_ranking,
            odds_distortion_bug_ranking=bug_ranking,
            win_bug_ranking=tuple(
                item for item in bug_ranking if item.bug_type is BugType.WIN
            ),
            place_bug_ranking=tuple(
                item for item in bug_ranking if item.bug_type is BugType.PLACE
            ),
        )

    def _fundamental_scores(
        self, context: SimpleRaceContext, horse: SimpleHorseFeatures
    ) -> dict[str, float]:
        form = min(
            1.0,
            0.16 * horse.recent_top3_count + 0.08 * horse.recent_top5_count,
        )
        return {
            "race_ability": self._clip(0.62 * horse.class_score + 0.38 * form),
            "pace_position": self._pace_position_score(context, horse),
            "going_course": self._clip(
                0.62 * horse.going_score + 0.38 * horse.course_score
            ),
            "physical_condition": self._physical_condition_score(horse),
        }

    def _pace_position_score(
        self, context: SimpleRaceContext, horse: SimpleHorseFeatures
    ) -> float:
        position = horse.predicted_position
        if context.wet_congested_pace:
            # 中山12R検証: 重馬場ハイペースでは先頭1～3番手より
            # 4～7番手の好位後方～中団前が有利だった。
            if 4 <= position <= 7:
                return 0.94
            if position == 8:
                return 0.78
            if position <= 3:
                return 0.38
            return 0.48
        if context.opening_week:
            if position <= 3:
                return 0.90
            if position <= 6:
                return 0.76
            return 0.48
        if position <= 5:
            return 0.76
        if position <= 9:
            return 0.65
        return 0.48

    def _physical_condition_score(self, horse: SimpleHorseFeatures) -> float:
        score = 0.58
        if horse.assigned_weight_kg <= 54.0:
            score += 0.10
        elif horse.assigned_weight_kg >= 58.0:
            score -= 0.08
        if horse.body_weight_kg >= 490:
            score += 0.06

        long_layoff = horse.days_since_last_run >= 120
        growth_return = (
            horse.age == 3
            and long_layoff
            and 6 <= horse.body_weight_change_kg <= 12
            and horse.stakes_top5
        )
        if growth_return:
            # 紫苑S④型: 3歳秋の成長分を休養明け減点より優先。
            score += 0.16
        elif long_layoff:
            score -= 0.10
        elif horse.body_weight_change_kg <= -16:
            score -= 0.08
        return self._clip(score)

    def _corroborating_axes(
        self,
        horse: SimpleHorseFeatures,
        parts: Mapping[str, float],
        value_gap: float,
    ) -> tuple[str, ...]:
        axes: list[str] = []
        if (
            parts["race_ability"] >= 0.62
            or horse.recent_top3_count >= 2
            or horse.stakes_top5
        ):
            axes.append("近走・クラス")
        if (
            parts["pace_position"] >= 0.72
            and parts["going_course"] >= 0.58
        ):
            axes.append("馬場・コース・展開")
        jockey_support = (
            horse.jockey_place_rate >= 0.20
            or horse.jockey_win_return >= 1.20
            or horse.jockey_place_return >= 1.20
        )
        physical_support = (
            horse.assigned_weight_kg <= 54.0
            or horse.body_weight_kg >= 490
            or parts["physical_condition"] >= 0.70
        )
        # 騎手回収率だけで最大バグへ昇格させない。
        if jockey_support and physical_support:
            axes.append("斤量・馬体・騎手")
        if value_gap >= 0.012:
            axes.append("最終オッズ売れ不足")
        return tuple(axes)

    @staticmethod
    def _bug_type(
        eligible: bool, fair_probability: float, race_ability: float
    ) -> BugType:
        if not eligible:
            return BugType.NONE
        if fair_probability >= 0.115 and race_ability >= 0.68:
            return BugType.WIN
        return BugType.PLACE

    @staticmethod
    def _softmax_probabilities(values: Sequence[float]) -> list[float]:
        maximum = max(values)
        exponentials = [exp((value - maximum) * 4.2) for value in values]
        total = sum(exponentials)
        return [value / total for value in exponentials]

    @staticmethod
    def _normalised_market_probabilities(
        horses: Sequence[SimpleHorseFeatures],
    ) -> dict[str, float]:
        raw = {horse.horse_id: 1.0 / horse.odds for horse in horses}
        overround = sum(raw.values())
        return {horse_id: value / overround for horse_id, value in raw.items()}

    def _fundamental_total(self, parts: Mapping[str, float]) -> float:
        return (
            self.WEIGHTS["race_ability"] * parts["race_ability"]
            + self.WEIGHTS["pace_position"] * parts["pace_position"]
            + self.WEIGHTS["going_course"] * parts["going_course"]
            + self.WEIGHTS["physical_condition"] * parts["physical_condition"]
        ) / 0.90

    @staticmethod
    def _odds_signal(value_gap: float, market_probability: float) -> float:
        relative_gap = value_gap / max(market_probability, 0.01)
        return SimpleLeadingSignalLambdaV02._clip(0.5 + 0.35 * relative_gap)

    @staticmethod
    def _clip(value: float) -> float:
        return min(1.0, max(0.0, value))
