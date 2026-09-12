from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .odds import OddsDistortion
from .schema import FrozenPrediction, LeadingSignalPolicy, PredictionRow, RaceContext, utc_now


def _canonical(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def freeze_prediction(
    path: str | Path,
    race: RaceContext,
    prediction: Sequence[PredictionRow],
    odds_distortion: Sequence[OddsDistortion],
    leading: LeadingSignalPolicy | None = None,
    frozen_at: datetime | None = None,
) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError("a frozen prediction cannot be overwritten")
    frozen_at = frozen_at or utc_now()
    if frozen_at.tzinfo is None:
        raise ValueError("frozen_at must include a timezone")
    if frozen_at >= race.post_time:
        raise ValueError("prediction must be frozen before post time")
    leading = leading or LeadingSignalPolicy()
    payload = {
        "schema_version": 1,
        "frozen_at": frozen_at.isoformat(),
        "race": {**asdict(race), "post_time": race.post_time.isoformat()},
        "lambda_ranking": [asdict(row) for row in prediction],
        "odds_distortion_ranking": [asdict(row) for row in odds_distortion],
        "leading_signal_policy": asdict(leading),
    }
    payload["checksum_sha256"] = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_frozen_prediction(path: str | Path) -> FrozenPrediction:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    checksum = payload.pop("checksum_sha256")
    actual = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    if checksum != actual:
        raise ValueError("frozen prediction checksum mismatch")
    return FrozenPrediction(checksum_sha256=checksum, **payload)
