import json
from pathlib import Path

import pytest

from racing_maintenance_rsi import (
    MalwareScan,
    MalwareStatus,
    RacingExternalDataGuard,
    audit_repository,
    validate_jra_payload,
)
from racing_maintenance_rsi.audit import _check_sources_and_secrets
from racing_lambda import ingest_snapshot_file


def test_current_repository_passes_racing_maintenance_controls():
    report = audit_repository(Path(__file__).resolve().parents[1])
    assert report.status == "PASS", [finding.to_dict() for finding in report.findings]
    payload = report.to_dict()
    assert payload["betting_authority"] is False
    assert payload["autonomous_main_merge"] is False
    assert payload["investment_system_access"] is False


def test_valid_jra_json_is_accepted_after_clean_scan(tmp_path, monkeypatch):
    source = tmp_path / "pre_race.json"
    source.write_text(json.dumps({"race_id": "202609130911", "market_support": []}), encoding="utf-8")
    monkeypatch.setattr(
        RacingExternalDataGuard,
        "scan_malware",
        staticmethod(lambda path: MalwareScan(MalwareStatus.CLEAN, "test", "clean")),
    )
    result = RacingExternalDataGuard().inspect(source)
    assert result.accepted
    assert len(result.sha256) == 64


def test_disguised_executable_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "odds.csv"
    source.write_bytes(b"MZmalicious")
    monkeypatch.setattr(
        RacingExternalDataGuard,
        "scan_malware",
        staticmethod(lambda path: MalwareScan(MalwareStatus.CLEAN, "test", "clean")),
    )
    result = RacingExternalDataGuard().inspect(source)
    assert not result.accepted
    assert "executable content signature detected" in result.reasons


def test_missing_antivirus_fails_closed(tmp_path, monkeypatch):
    source = tmp_path / "odds.json"
    source.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        RacingExternalDataGuard,
        "scan_malware",
        staticmethod(lambda path: MalwareScan(MalwareStatus.UNAVAILABLE, None, "missing")),
    )
    assert not RacingExternalDataGuard().inspect(source).accepted


def test_non_object_json_is_rejected_even_after_clean_scan(tmp_path, monkeypatch):
    source = tmp_path / "odds.json"
    source.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        RacingExternalDataGuard,
        "scan_malware",
        staticmethod(lambda path: MalwareScan(MalwareStatus.CLEAN, "test", "clean")),
    )
    result = RacingExternalDataGuard().inspect(source)
    assert not result.accepted
    assert any("ValueError" in reason for reason in result.reasons)


def test_symlink_is_rejected_without_reading_target(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "input.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic links"):
        RacingExternalDataGuard().inspect(link)


def test_hostile_or_nonfinite_payload_is_rejected():
    with pytest.raises(ValueError, match="unsafe key"):
        validate_jra_payload({"__proto__": {"polluted": True}})
    with pytest.raises(ValueError, match="non-finite"):
        validate_jra_payload({"odds": float("nan")})


def test_quarantine_removes_source_and_restricts_permissions(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text("{}", encoding="utf-8")
    destination = RacingExternalDataGuard.quarantine(source, tmp_path / "quarantine")
    assert destination.exists()
    assert not source.exists()
    assert destination.stat().st_mode & 0o777 == 0o600


def test_renamed_pickle_loads_call_is_detected_by_ast_scan(tmp_path):
    source_dir = tmp_path / "src" / "racing_lambda"
    source_dir.mkdir(parents=True)
    (source_dir / "evil.py").write_text(
        "import pickle\n"
        "loads = pickle.loads\n"
        "def run(data):\n"
        "    return pickle.loads(data)\n",
        encoding="utf-8",
    )
    findings = _check_sources_and_secrets(tmp_path)
    assert any(finding.code == "DANGEROUS_PATTERN" for finding in findings)


def test_maintenance_rsi_package_itself_is_in_scan_scope(tmp_path):
    source_dir = tmp_path / "src" / "racing_maintenance_rsi"
    source_dir.mkdir(parents=True)
    (source_dir / "evil.py").write_text(
        "import yaml\ndef run(text):\n    return yaml.unsafe_load(text)\n",
        encoding="utf-8",
    )
    findings = _check_sources_and_secrets(tmp_path)
    assert any(finding.code == "DANGEROUS_PATTERN" for finding in findings)


def test_inode_swap_between_read_and_malware_scan_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "odds.json"
    source.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        RacingExternalDataGuard,
        "scan_malware",
        staticmethod(lambda path: MalwareScan(MalwareStatus.CLEAN, "test", "clean")),
    )
    real_lstat = Path.lstat

    class _FakeStat:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            if name == "st_ino":
                return self._real.st_ino + 1
            return getattr(self._real, name)

    def fake_lstat(self):
        return _FakeStat(real_lstat(self))

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    with pytest.raises(ValueError, match="changed identity"):
        RacingExternalDataGuard().inspect(source)


def test_guarded_json_file_is_connected_to_official_snapshot_ingestion(tmp_path, monkeypatch):
    source = tmp_path / "jra_pre_race.json"
    source.write_text(json.dumps({"market_support": [{"horse_id": "1", "support": {"win": 0.2}}]}), encoding="utf-8")
    monkeypatch.setattr(
        RacingExternalDataGuard,
        "scan_malware",
        staticmethod(lambda path: MalwareScan(MalwareStatus.CLEAN, "test", "clean")),
    )
    snapshot, inspection = ingest_snapshot_file(
        path=source,
        race_id="202609130911",
        phase="PRE_RACE",
        source_url="https://www.jra.go.jp/",
    )
    assert inspection.accepted
    assert snapshot.phase == "PRE_RACE"
    assert snapshot.payload_sha256
