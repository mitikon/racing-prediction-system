from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Mapping


def _unit(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


@dataclass(frozen=True)
class RaceContext:
    race_id: str
    race_name: str
    post_time: datetime
    venue: str = ""
    surface: str = ""
    distance_m: int | None = None

    def __post_init__(self) -> None:
        if not self.race_id.strip():
            raise ValueError("race_id is required")
        if self.post_time.tzinfo is None:
            raise ValueError("post_time must include a timezone")


@dataclass(frozen=True)
class HorseEntry:
    horse_id: str
    horse_name: str
    ability: float
    pace_fit: float
    course_draw_fit: float
    weight_track_fit: float
    jockey_skill: float
    latest_information: float
    win_odds: float | None = None
    jockey_sample_size: int = 0
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.horse_id.strip() or not self.horse_name.strip():
            raise ValueError("horse_id and horse_name are required")
        for name in (
            "ability", "pace_fit", "course_draw_fit", "weight_track_fit",
            "jockey_skill", "latest_information",
        ):
            _unit(getattr(self, name), name)
        if self.win_odds is not None and self.win_odds <= 1.0:
            raise ValueError("decimal win_odds must be greater than 1")
        if self.jockey_sample_size < 0:
            raise ValueError("jockey_sample_size cannot be negative")


@dataclass(frozen=True)
class ComponentWeights:
    """Provisional operational weights; these are not the research lambda."""

    ability: float = 0.30
    pace_fit: float = 0.20
    course_draw_fit: float = 0.15
    weight_track_fit: float = 0.15
    jockey_skill: float = 0.10
    latest_information: float = 0.10

    def normalized(self) -> Mapping[str, float]:
        values = asdict(self)
        if any(value < 0 for value in values.values()):
            raise ValueError("weights cannot be negative")
        total = sum(values.values())
        if total <= 0:
            raise ValueError("at least one weight must be positive")
        return {key: value / total for key, value in values.items()}


@dataclass(frozen=True)
class LeadingSignalPolicy:
    """Gate for experimental signals observed before the race outcome."""

    enabled: bool = False
    weight: float = 0.0
    values: Mapping[str, float] = field(default_factory=dict)
    reason: str = "temporarily paused pending out-of-sample validation"

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("leading-signal weight must be between 0 and 1")
        for horse_id, value in self.values.items():
            _unit(value, f"leading signal for {horse_id}")
        if self.enabled and self.weight <= 0:
            raise ValueError("enabled leading signal requires a positive weight")


@dataclass(frozen=True)
class PredictionRow:
    rank: int
    horse_id: str
    horse_name: str
    score: float
    component_scores: Mapping[str, float]
    leading_signal_used: bool


@dataclass(frozen=True)
class OfficialResult:
    race_id: str
    finishing_order: tuple[str, ...]
    payouts: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FrozenPrediction:
    schema_version: int
    frozen_at: str
    race: Mapping[str, object]
    lambda_ranking: tuple[Mapping[str, object], ...]
    odds_distortion_ranking: tuple[Mapping[str, object], ...]
    leading_signal_policy: Mapping[str, object]
    checksum_sha256: str


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
