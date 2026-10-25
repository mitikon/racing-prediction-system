"""Statistical edge diagnostics for racing_lambda's real, verified predictions.

Applies the same discipline mitikonpro/etf-core-satellite established for
the US equities project - compare the real track record against a
model-free baseline, with a bootstrap confidence interval, rather than
trusting a raw hit count - to this project's own real, frozen PRE_RACE
predictions and their official JRA results.

Unlike that project's daily-return time series, each race is a materially
independent trial with no serial correlation to account for, so a plain
i.i.d. bootstrap over races (not the block bootstrap used for
autocorrelated daily returns) is the statistically appropriate resampling
unit here.

As of 2026-10-25 there are only 15 real, auditable races (2026-09-20 and
2026-09-21; 2026-09-13 is excluded per README/docs - a post-hoc virtual
exercise created after seeing result images, not a real-time prediction).
That is far too small a sample for either a positive or a negative
verdict. MIN_RACES_FOR_VERDICT documents the floor this module requires
before it reports anything beyond "insufficient sample, monitoring only" -
re-run this as more real races accumulate; nothing here promotes a
candidate or changes prediction logic by itself.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

# Below this many graded races, any bootstrap CI is too wide to mean
# anything (3 top-3 slots per race is a thin signal even before counting
# overdispersion across very different field sizes). This mirrors
# satellite_gate.py's MIN_FUTURE_SESSIONS discipline in etf-core-satellite:
# a documented floor the diagnostic itself refuses to pretend past.
MIN_RACES_FOR_VERDICT = 50

DEFAULT_N_BOOTSTRAP = 5000
DEFAULT_SEED = 0
PREDICTED_DEPTH = 5

# 2026-09-13 is a post-hoc virtual exercise created after seeing result
# images (see docs/racing/retrospective-2026-09-13-nakayama11.md) - not a
# real-time prediction - and is explicitly excluded from backtests by this
# project's own README. Only these two files hold real, frozen-before-the-
# race predictions with an official result.
KNOWN_GOOD_OUTCOME_FILES = (
    "2026-09-20-simple-outcomes.json",
    "2026-09-21-simple-outcomes.json",
)


@dataclass(frozen=True)
class GradedRace:
    race_id: str
    field_size: int
    predicted_top5: tuple[str, ...]
    actual_top3: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.field_size < 3:
            raise ValueError("field_size must be at least 3")
        if not self.predicted_top5:
            raise ValueError("predicted_top5 must not be empty")
        if len(self.actual_top3) != 3:
            raise ValueError("actual_top3 must contain exactly 3 horses")
        if len(set(self.predicted_top5)) != len(self.predicted_top5):
            raise ValueError("predicted_top5 must not contain duplicates")

    @property
    def predicted_depth(self) -> int:
        return len(self.predicted_top5)

    def top3_hits(self, order: Sequence[str] | None = None) -> int:
        """How many real top-3 finishers appear in a ranking's leading
        `predicted_depth` entries (defaults to this race's own lambda
        prediction, truncated the same way it was actually published)."""
        candidate = tuple(order) if order is not None else self.predicted_top5
        return len(set(candidate[: self.predicted_depth]) & set(self.actual_top3))


def load_graded_races(verification_dir: str | Path = "data/verification") -> tuple[GradedRace, ...]:
    """Load every real, frozen PRE_RACE prediction with a known official result.

    Reads only the whitelisted outcome files (KNOWN_GOOD_OUTCOME_FILES).
    A race listed under a file's own `non_evaluable_results` key (missing
    freeze evidence) is not in `races` and is therefore already excluded,
    matching this project's own after-the-fact-change ban. Some source
    files record the full finishing field under `pre_race_order`; this
    truncates to the first `PREDICTED_DEPTH` entries in every case, which
    is what was actually published as the Top5 (or the smaller published
    subset on the one day only 3 were published).
    """
    directory = Path(verification_dir)
    races: list[GradedRace] = []
    for filename in KNOWN_GOOD_OUTCOME_FILES:
        payload = json.loads((directory / filename).read_text(encoding="utf-8"))
        for entry in payload["races"]:
            order = entry.get("pre_race_order") or entry.get("published_order")
            races.append(
                GradedRace(
                    race_id=entry["race_id"],
                    field_size=int(entry["field_size"]),
                    predicted_top5=tuple(order[:PREDICTED_DEPTH]),
                    actual_top3=tuple(entry["actual_top3"]),
                )
            )
    return tuple(races)


def load_odds_by_race(verification_dir: str | Path = "data/verification") -> Mapping[str, Mapping[str, float]]:
    """Load win odds for every race that has a committed *-pre-race.json snapshot.

    Only a handful of races currently have a full-field odds snapshot
    committed; races without one are simply absent from the returned
    mapping, and the odds-baseline comparison skips them.
    """
    directory = Path(verification_dir)
    odds_by_race: dict[str, Mapping[str, float]] = {}
    for path in sorted(directory.glob("*-pre-race.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        odds = payload.get("odds_by_horse_number")
        if odds:
            odds_by_race[payload["race_id"]] = {str(horse): float(value) for horse, value in odds.items()}
    return odds_by_race


def chance_expected_top3_hits(field_size: int, predicted_depth: int) -> float:
    """Expected top-3 hits from a uniformly random guess of `predicted_depth`
    horses out of `field_size` runners (the hypergeometric mean)."""
    return predicted_depth * 3.0 / field_size


def odds_favorite_order(odds_by_horse: Mapping[str, float]) -> tuple[str, ...]:
    """Rank horses by shortest win odds first - the betting crowd's own order."""
    return tuple(sorted(odds_by_horse, key=lambda horse_id: odds_by_horse[horse_id]))


@dataclass(frozen=True)
class EdgeDiagnosticResult:
    races: int
    observed_hits: int
    baseline_hits_or_expectation: float
    mean_excess_per_race: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    sufficient_sample: bool
    note: str


def _bootstrap_mean_excess(per_race_excess: Sequence[float], n_bootstrap: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(per_race_excess, dtype=float)
    n = len(values)
    draws = rng.integers(0, n, size=(n_bootstrap, n))
    resampled_means = values[draws].mean(axis=1)
    return float(np.percentile(resampled_means, 2.5)), float(np.percentile(resampled_means, 97.5))


def _insufficiency_note(n: int, context: str) -> str:
    if n >= MIN_RACES_FOR_VERDICT:
        return "sample size meets MIN_RACES_FOR_VERDICT; still confirm against the odds-favorite baseline before trusting a chance-only comparison."
    return (
        f"n={n} {context}, below MIN_RACES_FOR_VERDICT={MIN_RACES_FOR_VERDICT}: "
        "this CI is a monitoring readout, not a verdict. Re-run as more real races accumulate."
    )


def summarize_vs_chance(
    races: Sequence[GradedRace],
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> EdgeDiagnosticResult:
    """Compare lambda's real top-3 hit rate against pure random guessing.

    This is the weakest possible bar (mitikonpro/satellite's "beat the
    no-skill baseline" check) - clearing it says nothing about beating the
    betting crowd, only that lambda is not indistinguishable from chance.
    """
    if not races:
        raise ValueError("at least one graded race is required")
    observed = [race.top3_hits() for race in races]
    expected = [chance_expected_top3_hits(race.field_size, race.predicted_depth) for race in races]
    excess = [obs - exp for obs, exp in zip(observed, expected)]
    ci_low, ci_high = _bootstrap_mean_excess(excess, n_bootstrap, seed)
    return EdgeDiagnosticResult(
        races=len(races),
        observed_hits=sum(observed),
        baseline_hits_or_expectation=sum(expected),
        mean_excess_per_race=float(np.mean(excess)),
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        sufficient_sample=len(races) >= MIN_RACES_FOR_VERDICT,
        note=_insufficiency_note(len(races), "races"),
    )


def summarize_vs_odds_baseline(
    races: Sequence[GradedRace],
    odds_by_race: Mapping[str, Mapping[str, float]],
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> EdgeDiagnosticResult:
    """Compare lambda's top-3 hit rate against simply backing the market
    favorites (shortest odds), to the same depth lambda itself predicted.

    This is the real bar: the Hong Kong precedent (Benter et al.) beat the
    pooled crowd's own odds, not a coin flip. Only races with a committed
    full-field odds snapshot are included.
    """
    paired = [race for race in races if race.race_id in odds_by_race]
    if not paired:
        raise ValueError("no graded race currently has a committed odds snapshot")
    observed = [race.top3_hits() for race in paired]
    baseline = [race.top3_hits(odds_favorite_order(odds_by_race[race.race_id])) for race in paired]
    excess = [obs - base for obs, base in zip(observed, baseline)]
    ci_low, ci_high = _bootstrap_mean_excess(excess, n_bootstrap, seed)
    return EdgeDiagnosticResult(
        races=len(paired),
        observed_hits=sum(observed),
        baseline_hits_or_expectation=float(sum(baseline)),
        mean_excess_per_race=float(np.mean(excess)),
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        sufficient_sample=len(paired) >= MIN_RACES_FOR_VERDICT,
        note=_insufficiency_note(len(paired), "races with a committed odds snapshot"),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Statistical edge diagnostics: lambda's real top-3 hit rate vs "
        "chance and vs the odds-favorite baseline, over every real graded race. "
        "Diagnostic only - promotes nothing and changes no prediction logic."
    )
    parser.add_argument("--verification-dir", default="data/verification")
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    races = load_graded_races(args.verification_dir)
    odds_by_race = load_odds_by_race(args.verification_dir)

    vs_chance = summarize_vs_chance(races, args.n_bootstrap, args.seed)
    print("=== vs chance (random guess) ===")
    print(json.dumps(asdict(vs_chance), ensure_ascii=False, indent=2))

    try:
        vs_odds = summarize_vs_odds_baseline(races, odds_by_race, args.n_bootstrap, args.seed)
        print("=== vs odds-favorite baseline ===")
        print(json.dumps(asdict(vs_odds), ensure_ascii=False, indent=2))
    except ValueError as exc:
        print(f"vs odds-favorite baseline: skipped ({exc})")


if __name__ == "__main__":
    main()
