from __future__ import annotations

from typing import Sequence

from .odds import OddsDistortion
from .schema import PredictionRow


def format_japanese_report(
    prediction: Sequence[PredictionRow], distortion: Sequence[OddsDistortion]
) -> str:
    lines = ["## λ総合最終予想", ""]
    lines.extend(f"{row.rank}. {row.horse_name}（score={row.score:.4f}）" for row in prediction)
    lines.extend(["", "## オッズ歪み／最大バグ順位予想", ""])
    lines.extend(
        f"{row.rank}. {row.horse_name}（歪み={row.distortion:+.4f}）" for row in distortion
    )
    return "\n".join(lines)
