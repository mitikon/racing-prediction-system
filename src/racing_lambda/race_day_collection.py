"""Bounded race-day collection policy for official result ingestion.

This module decides *what* may be collected.  Network access and HTML parsing
remain in an authorised provider adapter so a page-layout change cannot silently
corrupt RSI learning data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Iterable, Literal
from zoneinfo import ZoneInfo


JST = ZoneInfo("Asia/Tokyo")
RaceType = Literal["flat", "jump"]


@dataclass(frozen=True)
class ScheduledRace:
    race_id: str
    venue: str
    race_number: int
    scheduled_start: datetime
    race_type: RaceType
    result_status: Literal["not_started", "provisional", "official"] = "not_started"

    def __post_init__(self) -> None:
        if not self.race_id or not self.venue:
            raise ValueError("race_id and venue are required")
        if not 1 <= self.race_number <= 12:
            raise ValueError("race_number must be between 1 and 12")
        if self.scheduled_start.tzinfo is None or self.scheduled_start.utcoffset() is None:
            raise ValueError("scheduled_start must be timezone-aware")


@dataclass(frozen=True)
class RaceDayCollectionPolicy:
    """Collect each eligible official result once, only during the meeting day."""

    window_start: time = time(9, 0)
    window_end: time = time(17, 30)
    result_grace: timedelta = timedelta(minutes=5)
    exclude_jump_races: bool = True

    def eligible(
        self,
        race: ScheduledRace,
        *,
        now: datetime,
        already_collected: Iterable[str] = (),
    ) -> bool:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        local_now = now.astimezone(JST)
        local_start = race.scheduled_start.astimezone(JST)
        if local_now.date() != local_start.date():
            return False
        if not self.window_start <= local_now.time() <= self.window_end:
            return False
        if self.exclude_jump_races and race.race_type == "jump":
            return False
        if race.race_id in set(already_collected):
            return False
        if local_now < local_start + self.result_grace:
            return False
        return race.result_status == "official"

    def due_races(
        self,
        races: Iterable[ScheduledRace],
        *,
        now: datetime,
        already_collected: Iterable[str] = (),
    ) -> tuple[ScheduledRace, ...]:
        collected = frozenset(already_collected)
        return tuple(
            race for race in races
            if self.eligible(race, now=now, already_collected=collected)
        )
