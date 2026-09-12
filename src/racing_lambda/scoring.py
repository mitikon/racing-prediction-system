from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

from .schema import ComponentWeights, HorseEntry, LeadingSignalPolicy, PredictionRow


COMPONENTS = (
    "ability", "pace_fit", "course_draw_fit", "weight_track_fit",
    "jockey_skill", "latest_information",
)


def _shrunk_jockey_score(entry: HorseEntry, prior: float = 0.5, prior_strength: int = 20) -> float:
    """Prevent small jockey samples from causing a large rank jump."""
    n = entry.jockey_sample_size
    return (n * entry.jockey_skill + prior_strength * prior) / (n + prior_strength)


def build_prediction(
    entries: Iterable[HorseEntry],
    weights: ComponentWeights | None = None,
    leading: LeadingSignalPolicy | None = None,
) -> list[PredictionRow]:
    entries = list(entries)
    if len(entries) < 2:
        raise ValueError("at least two horses are required")
    if len({entry.horse_id for entry in entries}) != len(entries):
        raise ValueError("horse_id values must be unique")

    normalized_weights = (weights or ComponentWeights()).normalized()
    leading = leading or LeadingSignalPolicy()
    scored: list[tuple[HorseEntry, float, dict[str, float], bool]] = []

    for entry in entries:
        components = {name: float(getattr(entry, name)) for name in COMPONENTS}
        components["jockey_skill"] = _shrunk_jockey_score(entry)
        base = sum(normalized_weights[name] * components[name] for name in COMPONENTS)
        use_leading = leading.enabled and entry.horse_id in leading.values
        score = (
            (1.0 - leading.weight) * base + leading.weight * leading.values[entry.horse_id]
            if use_leading else base
        )
        scored.append((entry, score, components, use_leading))

    scored.sort(key=lambda item: (-item[1], item[0].horse_id))
    return [
        PredictionRow(
            rank=index,
            horse_id=entry.horse_id,
            horse_name=entry.horse_name,
            score=round(score, 8),
            component_scores=components,
            leading_signal_used=used,
        )
        for index, (entry, score, components, used) in enumerate(scored, start=1)
    ]
