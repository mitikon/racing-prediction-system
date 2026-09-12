"""Coordinator for the two-layer racing leading-signal architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TwoLayerSignal:
    horse_id: str
    statistical_score: float
    market_score: float | None
    combined_score: float
    confluence: bool
    mode: str


def combine_two_layers(
    statistical_scores: Mapping[str, float],
    market_scores: Mapping[str, float] | None = None,
    market_weight: float = 0.55,
    confluence_threshold: float = 0.65,
) -> list[TwoLayerSignal]:
    """Combine layers only when the real-time market layer is available.

    Until then, the statistical layer remains the complete score rather than
    being diluted by a missing market signal.  ``market_weight`` is an
    operational ensemble weight and is explicitly unrelated to the research
    0.1:0.9 PCA regularization ratio.
    """
    if not 0.0 <= market_weight <= 1.0:
        raise ValueError("market_weight must be between 0 and 1")
    if not 0.0 <= confluence_threshold <= 1.0:
        raise ValueError("confluence_threshold must be between 0 and 1")

    results: list[TwoLayerSignal] = []
    for horse_id, statistical in statistical_scores.items():
        statistical = float(np.clip(statistical, 0.0, 1.0))
        market = None if market_scores is None else market_scores.get(horse_id)
        if market is None:
            combined = statistical
            confluence = False
            mode = "statistical_only"
        else:
            market = float(np.clip(market, 0.0, 1.0))
            combined = (1.0 - market_weight) * statistical + market_weight * market
            confluence = statistical >= confluence_threshold and market >= confluence_threshold
            mode = "two_layer"
        results.append(
            TwoLayerSignal(
                horse_id=horse_id,
                statistical_score=statistical,
                market_score=market,
                combined_score=float(np.clip(combined, 0.0, 1.0)),
                confluence=confluence,
                mode=mode,
            )
        )
    return sorted(results, key=lambda result: (-result.combined_score, result.horse_id))
