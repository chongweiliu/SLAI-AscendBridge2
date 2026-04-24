#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REMOTE_PROJECT_ROOT_ENV_KEY = "SLAI_REMOTE_PROJECT_ROOT"
REMOTE_PROJECT_ROOT_PLACEHOLDER = f"${REMOTE_PROJECT_ROOT_ENV_KEY}"
REMOTE_BASE_PREFIX = "/workspace/"
REPO_ROOT_PREFIX = f"{PROJECT_ROOT.resolve().as_posix()}/"
REMOTE_REPO_MARKER = f"/{PROJECT_ROOT.name}"
REMOTE_REPO_PATTERN = re.compile(rf"{re.escape(REMOTE_BASE_PREFIX.rstrip('/'))}(?:/[^/]+)?/{re.escape(PROJECT_ROOT.name)}")


def _local_repo_relative(text: str) -> str:
    repo_root = PROJECT_ROOT.resolve().as_posix()
    updated = text.replace(REPO_ROOT_PREFIX, "")
    return "." if updated == repo_root else updated


def _remote_placeholder(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        return REMOTE_PROJECT_ROOT_PLACEHOLDER

    return REMOTE_REPO_PATTERN.sub(_replace, text)


def _sanitize_string(text: str) -> str:
    return _remote_placeholder(_local_repo_relative(text))


def _sanitize_json(value):
    if isinstance(value, dict):
        return {key: _sanitize_json(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, str):
        return _sanitize_string(value)
    return value


def _sanitize_business_config(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return False
    original = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    payload = _sanitize_json(payload)

    dataset_local_path = str(payload.get("dataset_local_path") or "").strip()
    if dataset_local_path.startswith("datasets/"):
        payload["dataset_local_path"] = (Path("../../") / dataset_local_path).as_posix()

    remote_project_root = str(payload.get("remote_project_root") or "").strip()
    if not remote_project_root or remote_project_root.startswith(REMOTE_PROJECT_ROOT_PLACEHOLDER):
        payload["remote_project_root"] = REMOTE_PROJECT_ROOT_PLACEHOLDER

    updated = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if updated != original:
        try:
            path.write_text(updated, encoding="utf-8")
        except PermissionError:
            print(f"[sanitize-repo-paths] skip unwritable file: {path.relative_to(PROJECT_ROOT)}")
            return False
        return True
    return False


def _sanitize_json_file(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    original = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    sanitized = _sanitize_json(payload)
    updated = json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n"
    if updated != original:
        try:
            path.write_text(updated, encoding="utf-8")
        except PermissionError:
            print(f"[sanitize-repo-paths] skip unwritable file: {path.relative_to(PROJECT_ROOT)}")
            return False
        return True
    return False


def main() -> int:
    changed = 0
    for path in PROJECT_ROOT.glob("adaptations/*/business_benchmark_config*.json"):
        if _sanitize_business_config(path):
            changed += 1
    for path in PROJECT_ROOT.glob("adaptations/*/business_summary*.json"):
        if _sanitize_json_file(path):
            changed += 1
    for path in PROJECT_ROOT.glob("adaptations/*/compare_result*.json"):
        if _sanitize_json_file(path):
            changed += 1
    for path in PROJECT_ROOT.glob("adaptations/**/*.trace.json"):
        if _sanitize_json_file(path):
            changed += 1
    compare_report = PROJECT_ROOT / "benchmark" / "reports" / "compare_report.json"
    if compare_report.exists() and _sanitize_json_file(compare_report):
        changed += 1
    print(f"[sanitize-repo-paths] updated {changed} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
