"""Fail-closed repository audit for the racing maintenance-only RSI."""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import IntEnum
import json
from pathlib import Path
import re


MAINTENANCE_RSI_VERSION = "racing-maintenance-rsi-v1"


class Severity(IntEnum):
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(frozen=True)
class AuditFinding:
    code: str
    severity: Severity
    message: str
    path: str

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["severity"] = self.severity.name
        return row


@dataclass(frozen=True)
class AuditReport:
    version: str
    generated_at_utc: str
    status: str
    findings: tuple[AuditFinding, ...]
    checked_controls: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "generated_at_utc": self.generated_at_utc,
            "status": self.status,
            "findings": [finding.to_dict() for finding in self.findings],
            "checked_controls": list(self.checked_controls),
            "autonomous_source_edits": False,
            "autonomous_main_merge": False,
            "betting_authority": False,
            "investment_system_access": False,
        }


_PINNED_ACTION = re.compile(
    r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*@[0-9a-f]{40}(?:\s*#.*)?$"
)
_BANNED_QUALIFIED_CALLS = {
    ("pickle", "load"): "untrusted pickle deserialization is prohibited",
    ("pickle", "loads"): "untrusted pickle deserialization is prohibited",
    ("yaml", "load"): "unsafe YAML loading is prohibited",
    ("yaml", "unsafe_load"): "unsafe YAML loading is prohibited",
    ("yaml", "full_load"): "unsafe YAML loading is prohibited",
    ("marshal", "loads"): "untrusted marshal deserialization is prohibited",
}
_SECRET_PATTERNS = {
    "AWS_ACCESS_KEY": re.compile(r"AKIA[0-9A-Z]{16}"),
    "GITHUB_TOKEN": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,255}"),
    "PRIVATE_KEY": re.compile("-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def _number(node: ast.AST | None) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    return None


def _module_numbers(path: Path) -> dict[str, float]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, float] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            value = _number(node.value)
            if value is not None:
                values[node.targets[0].id] = value
    return values


def _check_core(root: Path) -> list[AuditFinding]:
    """Protect the current prediction core: the PCA-independent four-horse
    extraction method plus the shared freeze/ingestion/guard infrastructure.

    本格・簡易式先行予測λと部分空間正則化PCA一式は2026-09-23に削除した
    （実戦検証で成績が悪化し、新方式の予測には無関係と判断したため）。
    この関数はその削除後の中核だけを保護する。
    """
    findings: list[AuditFinding] = []
    required = (
        root / "src/racing_lambda/freeze.py",
        root / "src/racing_lambda/jra_official_free_ingestion.py",
        root / "src/racing_lambda/four_horse_extraction.py",
        root / "src/racing_maintenance_rsi/data_guard.py",
    )
    for path in required:
        if not path.is_file():
            findings.append(AuditFinding("CORE_FILE_MISSING", Severity.CRITICAL, "required racing core file is missing", str(path)))
    if findings:
        return findings

    freeze_text = required[0].read_text(encoding="utf-8")
    snapshot_text = required[1].read_text(encoding="utf-8")
    extraction_text = required[2].read_text(encoding="utf-8")
    if 'path.open("x"' not in freeze_text:
        findings.append(AuditFinding("FROZEN_WRITE_WEAKENED", Severity.CRITICAL, "prediction freeze must use exclusive creation", str(required[0])))
    if 'target.open("x"' not in snapshot_text:
        findings.append(AuditFinding("SNAPSHOT_WRITE_WEAKENED", Severity.CRITICAL, "snapshot freeze must use exclusive creation", str(required[1])))
    if 'path.open("x"' not in extraction_text:
        findings.append(AuditFinding("EXTRACTION_WRITE_WEAKENED", Severity.CRITICAL, "four-horse extraction freeze must use exclusive creation", str(required[2])))

    constants = _module_numbers(required[2])
    if constants.get("EXTRACTION_COUNT") != 4:
        findings.append(
            AuditFinding("CORE_INVARIANT_CHANGED", Severity.CRITICAL, "EXTRACTION_COUNT must remain 4", str(required[2]))
        )
    if 'EXTRACTION_METHOD = "photo_prompt_manual_extraction"' not in extraction_text:
        findings.append(
            AuditFinding(
                "CORE_INVARIANT_CHANGED",
                Severity.CRITICAL,
                "EXTRACTION_METHOD must remain photo_prompt_manual_extraction",
                str(required[2]),
            )
        )

    if "inspect_with_content" not in required[3].read_text(encoding="utf-8"):
        findings.append(AuditFinding("TOCTOU_GUARD_MISSING", Severity.CRITICAL, "inspected bytes must flow directly into ingestion", str(required[3])))
    return findings


def _source_candidates(root: Path) -> list[Path]:
    return sorted(
        {
            *(root / "src").rglob("*.py"),
            *(root / "tests").rglob("*.py"),
            *(root / ".github").rglob("*.yml"),
            *(root / ".github").rglob("*.yaml"),
            root / "pyproject.toml",
            root / "README.md",
        }
    )


def _dangerous_call_findings(tree: ast.AST, path: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    module_aliases: dict[str, str] = {}
    imported_calls: dict[str, tuple[str, str]] = {}
    shell_kwargs_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                module_aliases[item.asname or item.name] = item.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                imported_calls[item.asname or item.name] = (node.module, item.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if isinstance(value, ast.Dict) and any(
                isinstance(key, ast.Constant) and key.value == "shell"
                and isinstance(item, ast.Constant) and item.value is True
                for key, item in zip(value.keys, value.values)
            ):
                shell_kwargs_names.update(
                    target.id for target in targets if isinstance(target, ast.Name)
                )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            module = module_aliases.get(func.value.id, func.value.id)
            message = _BANNED_QUALIFIED_CALLS.get((module, func.attr))
            if message is not None:
                findings.append(AuditFinding("DANGEROUS_PATTERN", Severity.HIGH, message, str(path)))
        elif isinstance(func, ast.Name):
            message = _BANNED_QUALIFIED_CALLS.get(imported_calls.get(func.id, ("", "")))
            if message is not None:
                findings.append(AuditFinding("DANGEROUS_PATTERN", Severity.HIGH, message, str(path)))
        for keyword in node.keywords:
            if (
                keyword.arg == "shell"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
            ):
                findings.append(
                    AuditFinding("DANGEROUS_PATTERN", Severity.HIGH, "shell=True is prohibited", str(path))
                )
            elif (
                keyword.arg is None
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id in shell_kwargs_names
            ):
                findings.append(
                    AuditFinding("DANGEROUS_PATTERN", Severity.HIGH, "indirect shell=True is prohibited", str(path))
                )
    return findings


def _check_sources_and_secrets(root: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    dangerous_pattern_roots = (root / "src/racing_lambda", root / "src/racing_maintenance_rsi")
    for path in _source_candidates(root):
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        tree: ast.AST | None = None
        if path.suffix == ".py":
            try:
                tree = ast.parse(text, filename=str(path))
            except SyntaxError as exc:
                findings.append(AuditFinding("PYTHON_SYNTAX_ERROR", Severity.CRITICAL, str(exc), str(path)))
        if tree is not None and any(source_root in path.parents for source_root in dangerous_pattern_roots):
            findings.extend(_dangerous_call_findings(tree, path))
        for secret_type, pattern in _SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(AuditFinding("SECRET_LEAK", Severity.CRITICAL, f"high-confidence {secret_type} signature detected", str(path)))
    return findings


def _check_workflows(root: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    workflow_root = root / ".github/workflows"
    for path in sorted((*workflow_root.glob("*.yml"), *workflow_root.glob("*.yaml"))):
        text = path.read_text(encoding="utf-8")
        if "pull_request_target:" in text:
            findings.append(AuditFinding("PRIVILEGED_PR_TRIGGER", Severity.CRITICAL, "pull_request_target is prohibited", str(path)))
        if re.search(r"(?m)^permissions:\s*write-all\s*$", text):
            findings.append(AuditFinding("WRITE_ALL_PERMISSION", Severity.CRITICAL, "write-all is prohibited", str(path)))
        for match in re.finditer(r"(?m)^\s*-?\s*uses:\s*([^\s]+(?:\s*#.*)?)$", text):
            action = match.group(1).strip()
            if not action.startswith("./") and not _PINNED_ACTION.match(action):
                findings.append(AuditFinding("UNPINNED_ACTION", Severity.HIGH, f"action must use a full commit SHA: {action}", str(path)))
    return findings


def audit_repository(root: str | Path) -> AuditReport:
    repository = Path(root).resolve()
    findings = [*_check_core(repository), *_check_sources_and_secrets(repository), *_check_workflows(repository)]
    status = "BLOCK" if max((item.severity for item in findings), default=Severity.INFO) >= Severity.HIGH else "PASS"
    return AuditReport(
        version=MAINTENANCE_RSI_VERSION,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        status=status,
        findings=tuple(findings),
        checked_controls=(
            "four_horse_extraction_fixed_method_and_count",
            "exclusive_frozen_writes",
            "python_syntax",
            "unsafe_deserialization",
            "high_confidence_secret_scan",
            "workflow_least_privilege",
            "action_sha_pinning",
            "same_bytes_ingestion",
            "ast_import_alias_resolution",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run racing maintenance-only RSI controls")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit_repository(args.root)
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    print(payload)
    raise SystemExit(1 if report.status == "BLOCK" else 0)


if __name__ == "__main__":
    main()
