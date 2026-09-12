"""Guarded ingestion foundation for public JRA pages used by 本格先行予測λ.

Naming boundary:
- 本格先行予測λ: the full horse-racing leading-prediction path.
- 簡易式先行予測λ: the lightweight current-input model and is separate.
- 先行シグナル予測λ: the separate 部分空間正則化PCA market project and is never handled here.

Safety / validation boundary:
- PRE_RACE observations are immutable once frozen.
- RESULT data is stored separately and can never mutate PRE_RACE.
- Network fetching is deliberately not implemented here until JRA permits the
  intended automated access pattern. Callers may supply manually downloaded or
  otherwise permitted public-page snapshots to ``ingest_snapshot``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Literal, Mapping
from urllib.parse import urlparse

Phase = Literal["PRE_RACE", "RESULT"]
_ALLOWED_HOSTS = {"www.jra.go.jp", "jra.go.jp", "jra.jp", "sp.jra.jp"}


@dataclass(frozen=True)
class OfficialSnapshot:
    race_id: str
    phase: Phase
    source_url: str
    observed_at: str
    payload_sha256: str
    code_commit_sha: str | None
    payload: Mapping[str, Any]


def _validate_public_jra_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_HOSTS:
        raise ValueError("source_url must be an HTTPS public JRA page")
    if "/dento/" in parsed.path or "/ticket/" in parsed.path:
        raise ValueError("member/ticket services are outside this ingestion module")


def _canonical_payload(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def ingest_snapshot(
    *,
    race_id: str,
    phase: Phase,
    source_url: str,
    payload: Mapping[str, Any],
    observed_at: datetime | None = None,
    code_commit_sha: str | None = None,
) -> OfficialSnapshot:
    """Create a provenance-bearing immutable observation."""
    if not race_id.strip():
        raise ValueError("race_id is required")
    _validate_public_jra_url(source_url)
    if phase not in ("PRE_RACE", "RESULT"):
        raise ValueError("phase must be PRE_RACE or RESULT")
    if phase == "PRE_RACE":
        forbidden = {"official_result", "finish_order", "result", "payout"}
        collision = forbidden.intersection(payload.keys())
        if collision:
            raise ValueError(f"result leakage into PRE_RACE: {sorted(collision)}")
    raw = _canonical_payload(payload)
    when = observed_at or datetime.now(timezone.utc)
    if when.tzinfo is None:
        raise ValueError("observed_at must be timezone-aware")
    return OfficialSnapshot(
        race_id=race_id,
        phase=phase,
        source_url=source_url,
        observed_at=when.isoformat(),
        payload_sha256=sha256(raw).hexdigest(),
        code_commit_sha=code_commit_sha,
        payload=dict(payload),
    )


def freeze_snapshot(snapshot: OfficialSnapshot, root: str | Path) -> Path:
    """Write once. Existing snapshots are never overwritten."""
    root = Path(root)
    directory = root / snapshot.race_id / snapshot.phase
    directory.mkdir(parents=True, exist_ok=True)
    safe_time = snapshot.observed_at.replace(":", "-")
    target = directory / f"{safe_time}_{snapshot.payload_sha256[:12]}.json"
    if target.exists():
        return target
    with target.open("x", encoding="utf-8") as fh:
        json.dump(asdict(snapshot), fh, ensure_ascii=False, sort_keys=True, indent=2)
        fh.write("\n")
    return target


def load_frozen_snapshot(path: str | Path) -> OfficialSnapshot:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    snapshot = OfficialSnapshot(**data)
    expected = sha256(_canonical_payload(snapshot.payload)).hexdigest()
    if expected != snapshot.payload_sha256:
        raise ValueError("snapshot integrity check failed")
    return snapshot
