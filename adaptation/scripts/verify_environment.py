#!/usr/bin/env python3
"""Rebuild an adaptation environment in-project and run its smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import tomllib

REPORT_NAME = "environment_validation.json"
LOG_NAME = "environment_validation.log"
_EXCLUDED_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".validation",
    ".venv",
    "__pycache__",
    REPORT_NAME,
    LOG_NAME,
}
_FINGERPRINT_EXCLUDED_DIRS = _EXCLUDED_NAMES | {"models", "profiling"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _select_extra(adapt_dir: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    status_path = adapt_dir / ".status.json"
    try:
        dry_run = json.loads(status_path.read_text(encoding="utf-8"))["stages"]["dry_run"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法从 .status.json 判断硬件 extra: {exc}") from exc
    device = str(dry_run.get("device", "")).lower()
    if dry_run.get("npu_detected") or device.startswith("npu"):
        return "ascend"
    if dry_run.get("cuda_detected") or device.startswith("cuda"):
        return "cuda"
    raise ValueError(".status.json 的 dry_run 未记录 NPU 或 CUDA")


def _ignore_copy(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in _EXCLUDED_NAMES or name == "models"}


def _prepare_workspace(adapt_dir: Path, workspace: Path) -> None:
    shutil.copytree(adapt_dir, workspace, ignore=_ignore_copy, symlinks=True)
    models_dir = adapt_dir / "models"
    if models_dir.exists():
        (workspace / "models").symlink_to(models_dir.resolve(), target_is_directory=True)


def _validate_portable_sources(adapt_dir: Path) -> None:
    for path in adapt_dir.rglob("*"):
        relative = path.relative_to(adapt_dir)
        if any(part in _FINGERPRINT_EXCLUDED_DIRS for part in relative.parts):
            continue
        if path.is_symlink():
            try:
                path.resolve().relative_to(adapt_dir)
            except ValueError as exc:
                raise ValueError(f"源码软链接指向 adaptation 目录外: {relative} -> {path.resolve()}") from exc

    pyproject = tomllib.loads((adapt_dir / "pyproject.toml").read_text(encoding="utf-8"))
    sources = pyproject.get("tool", {}).get("uv", {}).get("sources", {})
    if not isinstance(sources, dict):
        raise TypeError("pyproject.toml 的 [tool.uv.sources] 必须是表")
    for package, entries in sources.items():
        candidates = entries if isinstance(entries, list) else [entries]
        for entry in candidates:
            if not isinstance(entry, dict) or "path" not in entry:
                continue
            source_path = Path(str(entry["path"]))
            resolved = source_path.resolve() if source_path.is_absolute() else (adapt_dir / source_path).resolve()
            try:
                resolved.relative_to(adapt_dir)
            except ValueError as exc:
                raise ValueError(f"本地依赖 {package} 指向 adaptation 目录外: {source_path}") from exc


def _fingerprint(adapt_dir: Path) -> tuple[str, list[dict[str, str]]]:
    manifest: list[dict[str, str]] = []
    for path in sorted(adapt_dir.rglob("*")):
        relative = path.relative_to(adapt_dir)
        if any(part in _FINGERPRINT_EXCLUDED_DIRS for part in relative.parts):
            continue
        if not path.is_file() or path.name.startswith(("output", "benchmark_metrics_", "trace_")):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append({"path": relative.as_posix(), "sha256": digest})
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), manifest


def _clean_environment(source: dict[str, str], run_dir: Path, workspace: Path) -> dict[str, str]:
    env = source.copy()
    for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX", "UV_PROJECT_ENVIRONMENT"):
        env.pop(key, None)
    path_parts = [part for part in env.get("PATH", "").split(os.pathsep) if ".venv" not in part]
    env.update(
        {
            "PATH": os.pathsep.join(path_parts),
            "PYTHONNOUSERSITE": "1",
            "UV_PROJECT_ENVIRONMENT": str(workspace / ".venv"),
            "UV_CACHE_DIR": str(run_dir / "uv-cache"),
            "XDG_CACHE_HOME": str(run_dir / "xdg-cache"),
        }
    )
    return env


def _run(command: list[str], cwd: Path, env: dict[str, str], log, timeout: int) -> tuple[int, str]:
    printable = " ".join(command)
    log.write(f"\n$ {printable}\n")
    log.flush()
    try:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=timeout, check=False)
        return result.returncode, ""
    except subprocess.TimeoutExpired:
        log.write(f"\n[validation] command timed out after {timeout}s\n")
        return 124, f"命令超时（{timeout}s）: {printable}"
    except OSError as exc:
        log.write(f"\n[validation] failed to start command: {exc}\n")
        return 127, f"命令无法启动: {exc}"


def verify_environment(adapt_dir: Path, extra: str = "auto", sync_timeout: int = 1800, smoke_timeout: int = 3600) -> tuple[bool, dict]:
    adapt_dir = adapt_dir.resolve()
    selected_extra = _select_extra(adapt_dir, extra)
    _validate_portable_sources(adapt_dir)
    validation_root = adapt_dir / ".validation"
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    run_dir = validation_root / run_id
    workspace = run_dir / "workspace"
    run_dir.mkdir(parents=True)
    _prepare_workspace(adapt_dir, workspace)
    fingerprint, manifest = _fingerprint(adapt_dir)
    uv = shutil.which("uv") or "uv"
    env = _clean_environment(os.environ, run_dir, workspace)
    commands = [
        [uv, "sync", "--locked", "--extra", selected_extra, "--no-install-project"],
        [uv, "run", "--no-sync", "--extra", selected_extra, "python", "demo.py", "--smoke-test"],
    ]
    started_at = _utc_now()
    result = {
        "schema_version": 1,
        "status": "failed",
        "validation_level": "python_isolated",
        "run_id": run_id,
        "started_at": started_at,
        "extra": selected_extra,
        "source_fingerprint": fingerprint,
        "source_manifest": manifest,
        "commands": commands,
        "temporary_directory": str(run_dir.relative_to(adapt_dir)),
        "failed_stage": None,
        "failure_reason": "",
    }
    log_path = adapt_dir / LOG_NAME
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[validation] run_id={run_id} extra={selected_extra} started_at={started_at}\n")
        for stage, command, timeout in (
            ("sync", commands[0], sync_timeout),
            ("smoke_test", commands[1], smoke_timeout),
        ):
            returncode, reason = _run(command, workspace, env, log, timeout)
            if returncode != 0:
                result["failed_stage"] = stage
                result["failure_reason"] = reason or f"{stage} 退出码为 {returncode}"
                result["exit_code"] = returncode
                break
        else:
            freeze_command = [uv, "pip", "freeze"]
            log.write("\n$ " + " ".join(freeze_command) + "\n")
            freeze = subprocess.run(freeze_command, cwd=workspace, env=env, capture_output=True, text=True, check=False)
            log.write(freeze.stdout)
            if freeze.stderr:
                log.write(freeze.stderr)
            result["packages"] = freeze.stdout.splitlines() if freeze.returncode == 0 else []
            result["status"] = "passed"
            result["exit_code"] = 0
        result["finished_at"] = _utc_now()
        log.write(f"\n[validation] status={result['status']} finished_at={result['finished_at']}\n")
    (adapt_dir / REPORT_NAME).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if result["status"] == "passed":
        shutil.rmtree(validation_root)
    return result["status"] == "passed", result


def main() -> int:
    parser = argparse.ArgumentParser(description="在 adaptation 目录内重建隔离环境并执行最小真实推理")
    parser.add_argument("--adapt", required=True, help="adaptations/ 下的目录名")
    parser.add_argument("--base-dir", default="adaptations")
    parser.add_argument("--extra", choices=("auto", "ascend", "cuda"), default="auto")
    parser.add_argument("--sync-timeout", type=int, default=1800)
    parser.add_argument("--smoke-timeout", type=int, default=3600)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent.parent
    base_dir = (project_root / args.base_dir).resolve()
    adapt_dir = (base_dir / args.adapt).resolve()
    try:
        adapt_dir.relative_to(base_dir)
    except ValueError:
        print(f"[validation] 非法 adaptation 路径: {adapt_dir}", file=sys.stderr)
        return 1
    if not adapt_dir.is_dir():
        print(f"[validation] adaptation 目录不存在: {adapt_dir}", file=sys.stderr)
        return 1
    try:
        passed, report = verify_environment(adapt_dir, args.extra, args.sync_timeout, args.smoke_timeout)
    except (OSError, TypeError, ValueError) as exc:
        print(f"[validation] 初始化失败: {exc}", file=sys.stderr)
        return 1
    if passed:
        print(f"[validation] PASS: {adapt_dir.relative_to(project_root)} ({report['extra']})")
        return 0
    print(f"[validation] FAIL: {report['failed_stage']}: {report['failure_reason']}", file=sys.stderr)
    print(f"[validation] 日志: {adapt_dir / LOG_NAME}", file=sys.stderr)
    print(f"[validation] 失败现场: {adapt_dir / report['temporary_directory']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
