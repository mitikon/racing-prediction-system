"""Quarantine-first defenses for files and payloads entering the racing system."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pandas as pd


class MalwareStatus(str, Enum):
    CLEAN = "CLEAN"
    INFECTED = "INFECTED"
    UNAVAILABLE = "UNAVAILABLE"
    ERROR = "ERROR"


@dataclass(frozen=True)
class MalwareScan:
    status: MalwareStatus
    scanner: str | None
    detail: str


@dataclass(frozen=True)
class DataInspection:
    path: str
    sha256: str
    size_bytes: int
    accepted: bool
    reasons: tuple[str, ...]
    malware: MalwareScan


class RacingExternalDataGuard:
    ALLOWED_EXTENSIONS = frozenset({".json", ".csv", ".html", ".htm"})
    EXECUTABLE_MAGIC = (b"MZ", b"\x7fELF", b"#!")

    def __init__(self, max_bytes: int = 25_000_000) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = int(max_bytes)

    @staticmethod
    def scan_malware(path: Path) -> MalwareScan:
        scanner = shutil.which("clamscan")
        if scanner is None:
            return MalwareScan(MalwareStatus.UNAVAILABLE, None, "no supported scanner is installed")
        try:
            completed = subprocess.run(
                [scanner, "--no-summary", str(path)],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return MalwareScan(MalwareStatus.ERROR, "clamscan", type(exc).__name__)
        detail = (completed.stdout + completed.stderr).strip()[-2000:]
        if completed.returncode == 0:
            return MalwareScan(MalwareStatus.CLEAN, "clamscan", detail)
        if completed.returncode == 1:
            return MalwareScan(MalwareStatus.INFECTED, "clamscan", detail)
        return MalwareScan(MalwareStatus.ERROR, "clamscan", detail)

    def inspect(self, path: str | Path, *, require_antivirus: bool = True) -> DataInspection:
        source = Path(path)
        if source.is_symlink():
            raise ValueError("symbolic links are prohibited")
        if not source.is_file():
            raise ValueError("external data path must be a regular file")
        size = source.stat().st_size
        if size > self.max_bytes:
            return DataInspection(
                str(source), "", size, False, ("file exceeds configured size limit",),
                MalwareScan(MalwareStatus.UNAVAILABLE, None, "rejected before reading"),
            )
        reasons: list[str] = []
        if source.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            reasons.append("file extension is not allow-listed")
        raw = source.read_bytes()
        if b"\x00" in raw:
            reasons.append("NUL byte detected")
        if raw.startswith(self.EXECUTABLE_MAGIC):
            reasons.append("executable content signature detected")
        if not reasons:
            try:
                if source.suffix.lower() == ".json":
                    value = json.loads(raw.decode("utf-8"))
                    if not isinstance(value, Mapping):
                        raise ValueError("JRA JSON root must be an object")
                    validate_jra_payload(value)
                elif source.suffix.lower() == ".csv":
                    pd.read_csv(source, nrows=10)
                else:
                    raw.decode("utf-8")
            except (UnicodeDecodeError, json.JSONDecodeError, pd.errors.ParserError, ValueError) as exc:
                reasons.append(f"content validation failed: {type(exc).__name__}")
        malware = self.scan_malware(source)
        if malware.status in {MalwareStatus.INFECTED, MalwareStatus.ERROR}:
            reasons.append(f"malware scan status is {malware.status.value}")
        if require_antivirus and malware.status is MalwareStatus.UNAVAILABLE:
            reasons.append("mandatory antivirus scanner is unavailable")
        return DataInspection(str(source), sha256(raw).hexdigest(), size, not reasons, tuple(reasons), malware)

    @staticmethod
    def quarantine(path: str | Path, quarantine_root: str | Path) -> Path:
        source = Path(path)
        if source.is_symlink() or not source.is_file():
            raise ValueError("quarantine source must be a regular non-symlink file")
        root = Path(quarantine_root)
        root.mkdir(parents=True, exist_ok=True)
        root.chmod(0o700)
        digest = sha256(source.read_bytes()).hexdigest()
        destination = root / f"{digest}{source.suffix.lower()}.quarantine"
        if destination.exists():
            raise FileExistsError(f"quarantine target already exists: {destination}")
        moved = Path(shutil.move(str(source), str(destination)))
        moved.chmod(0o600)
        return moved


def validate_jra_payload(payload: Mapping[str, Any]) -> None:
    """Reject oversized, non-JSON, non-finite, or structurally hostile payloads."""
    entries = 0
    prohibited_keys = {"__proto__", "constructor", "prototype"}

    def visit(value: Any, depth: int) -> None:
        nonlocal entries
        entries += 1
        if entries > 100_000:
            raise ValueError("JRA payload contains too many values")
        if depth > 16:
            raise ValueError("JRA payload nesting is too deep")
        if value is None or isinstance(value, (bool, str)):
            if isinstance(value, str) and (len(value) > 1_000_000 or "\x00" in value):
                raise ValueError("JRA payload contains an unsafe string")
            return
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("JRA payload contains a non-finite number")
            return
        if isinstance(value, Mapping):
            for key, child in value.items():
                if not isinstance(key, str) or not key or len(key) > 256 or key in prohibited_keys:
                    raise ValueError("JRA payload contains an unsafe key")
                visit(child, depth + 1)
            return
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            for child in value:
                visit(child, depth + 1)
            return
        raise ValueError(f"JRA payload contains unsupported type: {type(value).__name__}")

    visit(payload, 0)
