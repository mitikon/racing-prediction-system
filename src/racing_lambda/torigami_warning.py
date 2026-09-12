"""Post-prediction torigami (negative-return) warning for racing Top5.

This module does not alter λ ranking or prediction logic.  It runs after a
Top5 prediction has been fixed and evaluates ticket economics from supplied
current odds.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Mapping, Sequence


TicketKey = tuple[str, str, str]


@dataclass(frozen=True)
class TorigamiReport:
    bet_type: str
    strategy: str
    selections: tuple[str, ...]
    axis: str | None
    ticket_count: int
    stake_per_ticket: int
    total_stake: int
    odds_available: int
    torigami_count: int
    torigami_rate: float
    break_even_odds: float
    missing_odds_count: int
    warning: bool

    @property
    def warning_text(self) -> str:
        if self.missing_odds_count:
            suffix = f" / オッズ未取得 {self.missing_odds_count}点"
        else:
            suffix = ""
        if not self.warning:
            return f"トリガミ警告なし（{self.torigami_count}/{self.odds_available}点）{suffix}"
        return (
            f"⚠️ トリガミ警告: {self.torigami_count}/{self.odds_available}点 "
            f"({self.torigami_rate:.1%}) が総投資額 {self.total_stake:,}円未満の払戻見込み{suffix}"
        )


def _validate_top5(top5: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(x) for x in top5)
    if len(values) != 5 or len(set(values)) != 5:
        raise ValueError("prediction must contain exactly five unique horses")
    return values


def ticket_keys(top5: Sequence[str], *, bet_type: str, axis: str | None = None) -> tuple[TicketKey, ...]:
    """Return tickets for 5-horse BOX or one-axis flow.

    trifecta: 三連単. trio: 三連複.
    axis=None means 5-horse BOX.
    axis=<horse> means that horse must be included, with the other four as flows.
    """
    horses = _validate_top5(top5)
    if bet_type not in {"trifecta", "trio"}:
        raise ValueError("bet_type must be 'trifecta' or 'trio'")
    if axis is not None:
        axis = str(axis)
        if axis not in horses:
            raise ValueError("axis horse must be one of prediction Top5")

    if bet_type == "trifecta":
        tickets = tuple(permutations(horses, 3))
        if axis is not None:
            tickets = tuple(t for t in tickets if axis in t)
        return tickets

    tickets = tuple(combinations(horses, 3))
    if axis is not None:
        tickets = tuple(t for t in tickets if axis in t)
    return tickets


def evaluate_torigami(
    top5: Sequence[str],
    *,
    bet_type: str,
    odds: Mapping[TicketKey, float],
    stake_per_ticket: int = 100,
    axis: str | None = None,
) -> TorigamiReport:
    """Warn when a winning ticket would return less than the strategy total stake.

    JRA odds are payout multiples for a 100-yen-style unit comparison.  With an
    equal stake on every ticket, break-even odds are total_stake/stake_per_ticket,
    i.e. simply the number of tickets.  A ticket is torigami when
    odds * stake_per_ticket < total_stake.
    """
    if stake_per_ticket <= 0:
        raise ValueError("stake_per_ticket must be positive")
    tickets = ticket_keys(top5, bet_type=bet_type, axis=axis)
    total_stake = len(tickets) * stake_per_ticket
    torigami = 0
    available = 0
    missing = 0
    for ticket in tickets:
        key = ticket if bet_type == "trifecta" else tuple(sorted(ticket))
        value = odds.get(key)
        if value is None:
            missing += 1
            continue
        value = float(value)
        if value <= 0:
            raise ValueError("odds must be positive")
        available += 1
        if value * stake_per_ticket < total_stake:
            torigami += 1
    return TorigamiReport(
        bet_type=bet_type,
        strategy="axis" if axis is not None else "box",
        selections=tuple(str(x) for x in top5),
        axis=axis,
        ticket_count=len(tickets),
        stake_per_ticket=stake_per_ticket,
        total_stake=total_stake,
        odds_available=available,
        torigami_count=torigami,
        torigami_rate=(torigami / available) if available else 0.0,
        break_even_odds=total_stake / stake_per_ticket,
        missing_odds_count=missing,
        warning=torigami > 0,
    )


def evaluate_top5_bet_warnings(
    top5: Sequence[str],
    *,
    trifecta_odds: Mapping[TicketKey, float],
    trio_odds: Mapping[TicketKey, float],
    axis: str,
    stake_per_ticket: int = 100,
) -> tuple[TorigamiReport, ...]:
    """Standard post-prediction warning set required by the racing workflow."""
    return (
        evaluate_torigami(top5, bet_type="trifecta", odds=trifecta_odds, stake_per_ticket=stake_per_ticket),
        evaluate_torigami(top5, bet_type="trio", odds=trio_odds, stake_per_ticket=stake_per_ticket),
        evaluate_torigami(top5, bet_type="trifecta", odds=trifecta_odds, stake_per_ticket=stake_per_ticket, axis=axis),
        evaluate_torigami(top5, bet_type="trio", odds=trio_odds, stake_per_ticket=stake_per_ticket, axis=axis),
    )
