#!/usr/bin/env python3
"""
对 board.db 中所有 adaptation_status=completed 的模型，依次运行 demo.py 以触发模型下载。
"""

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.adaptation_utils import model_id_to_adaptation_path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or _PROJECT_ROOT)
DB_PATH = PROJECT_ROOT / "board.db"
RECORD_FILE = PROJECT_ROOT / "adaptation" / "scripts" / "run_completed_adaptations.log"


def get_completed_models():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT model_id, adaptation_path FROM models WHERE adaptation_status = 'completed' ORDER BY model_id")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def parse_succeeded_from_record(record_path: str | Path) -> set[str]:
    path = Path(record_path)
    if not path.exists():
        return set()
    lines = path.read_text(encoding="utf-8").splitlines()
    succeeded: set[str] = set()
    in_succeeded = False
    for line in lines:
        stripped = line.strip()
        if stripped == "Succeeded:":
            in_succeeded = True
            continue
        if in_succeeded:
            if stripped.startswith("- "):
                model_id = stripped[2:].strip()
                if model_id and model_id != "(none)":
                    succeeded.add(model_id)
            elif stripped and not line.startswith("  ") and not line.startswith("\t"):
                break
    return succeeded


def resolve_adaptation_path(model_id: str, adaptation_path: str | None) -> Path:
    path_value = (adaptation_path or "").strip()
    if path_value:
        candidate = PROJECT_ROOT / path_value
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / model_id_to_adaptation_path(model_id)


def resolve_cache_dir(project_root: Path, work_dir: Path) -> Path:
    project_root = project_root.resolve()
    work_dir = work_dir.resolve()
    cache_dir = work_dir / "models"
    root_cache_dir = project_root / "models"
    if cache_dir == root_cache_dir:
        raise ValueError(f"拒绝使用项目根缓存目录: {cache_dir}")
    return cache_dir


def detail_to_lines(detail: dict) -> list[str]:
    lines = [
        f"  [{detail['status']}] {detail['model_id']}",
        f"    work_dir: {detail.get('work_dir', '')}",
    ]
    cache_dir = detail.get("cache_dir", "")
    if cache_dir:
        lines.append(f"    cache_dir: {cache_dir}")
    lines.append(f"    started: {detail.get('started', '')}  ended: {detail.get('ended', '')}  duration: {detail.get('duration_s', 0)}s")
    reason = detail.get("reason", "")
    if reason:
        lines.append(f"    reason: {reason}")
    lines.append("")
    return lines


def is_http_416_error(exc: Exception) -> bool:
    msg = str(exc)
    return "416 Requested Range Not Satisfiable" in msg or "Requested Range Not Satisfiable" in msg


def cleanup_model_cache(cache_dir: Path, model_id: str) -> None:
    cache_dir = Path(cache_dir)
    model_cache_name = f"models--{model_id.replace('/', '--')}"
    model_cache_dir = cache_dir / model_cache_name
    lock_dir = cache_dir / ".locks" / model_cache_name
    if model_cache_dir.exists():
        shutil.rmtree(model_cache_dir)
    if lock_dir.exists():
        shutil.rmtree(lock_dir)


def download_model_snapshot(
    model_id: str,
    cache_dir: Path,
    max_workers: int,
    snapshot_download_fn=None,
    cleanup_fn=None,
    revision: str | None = None,
):
    if snapshot_download_fn is None:
        from huggingface_hub import snapshot_download

        snapshot_download_fn = snapshot_download
    if cleanup_fn is None:
        cleanup_fn = cleanup_model_cache
    try:
        kwargs = {
            "repo_id": model_id,
            "cache_dir": str(cache_dir),
            "max_workers": max_workers,
        }
        if revision:
            kwargs["revision"] = revision
        return snapshot_download_fn(**kwargs)
    except Exception as exc:
        if not is_http_416_error(exc):
            raise
        print(f"  -> 检测到 416，清理损坏缓存后重试一次: {cache_dir}")
        cleanup_fn(cache_dir, model_id)
        return snapshot_download_fn(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="对 board.db 中 adaptation_status=completed 的模型运行 demo.py 以下载模型")
    parser.add_argument("--dry-run", action="store_true", help="仅打印将要执行的命令，不实际运行")
    parser.add_argument("--download-only", action="store_true", help="用 huggingface_hub 直接下载，无需 NPU/CUDA")
    parser.add_argument("--max-workers", type=int, default=8, help="--download-only 时并行下载文件数")
    args = parser.parse_args()

    all_completed = get_completed_models()
    models = all_completed
    if not models:
        print("[run_completed_adaptations] 没有 adaptation_status=completed 的模型")
        return 0

    skip_set = parse_succeeded_from_record(RECORD_FILE)
    if skip_set:
        print(f"[run_completed_adaptations] 从记录跳过 {len(skip_set)} 个: {RECORD_FILE.name}")
        models = [model for model in models if model["model_id"] not in skip_set]
    skipped_from_record = len(all_completed) - len(models)
    if not models:
        print("[run_completed_adaptations] 无待处理模型（已全部跳过或完成）")
        return 0

    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []
    details: list[dict] = []
    start_time = datetime.now()

    for index, row in enumerate(models, 1):
        model_id = row["model_id"]
        work_dir = resolve_adaptation_path(model_id, row.get("adaptation_path"))
        demo_py = work_dir / "demo.py"
        item_start = datetime.now()
        if not args.download_only and not demo_py.exists():
            failed.append((model_id, "demo.py 不存在"))
            details.append({"model_id": model_id, "status": "skipped", "reason": "demo.py 不存在", "work_dir": str(work_dir), "started": item_start.strftime("%Y-%m-%d %H:%M:%S"), "duration_s": 0})
            print(f"[{index}/{len(models)}] 跳过 {model_id}: demo.py 不存在 ({work_dir})")
            continue

        try:
            cache_dir = resolve_cache_dir(PROJECT_ROOT, work_dir)
        except ValueError as exc:
            failed.append((model_id, str(exc)))
            details.append({"model_id": model_id, "status": "failed", "reason": str(exc), "work_dir": str(work_dir), "cache_dir": str((work_dir / 'models').resolve()), "started": item_start.strftime("%Y-%m-%d %H:%M:%S"), "ended": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "duration_s": round((datetime.now() - item_start).total_seconds(), 1)})
            print(f"[{index}/{len(models)}] 安全检查失败 {model_id}: {exc}")
            break

        if args.download_only:
            print(f"[{index}/{len(models)}] 下载 {model_id} -> {cache_dir} ...")
            if args.dry_run:
                print(f"  -> snapshot_download({model_id}, cache_dir={cache_dir}) [HF_ENDPOINT=hf-mirror.com]")
                continue
            try:
                os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                download_model_snapshot(model_id, cache_dir, args.max_workers)
                succeeded.append(model_id)
                details.append({"model_id": model_id, "status": "success", "work_dir": str(work_dir), "cache_dir": str(cache_dir), "started": item_start.strftime("%Y-%m-%d %H:%M:%S"), "ended": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "duration_s": round((datetime.now() - item_start).total_seconds(), 1)})
                print("  -> 完成")
            except Exception as exc:
                failed.append((model_id, str(exc)))
                details.append({"model_id": model_id, "status": "failed", "reason": str(exc), "work_dir": str(work_dir), "cache_dir": str(cache_dir), "started": item_start.strftime("%Y-%m-%d %H:%M:%S"), "ended": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "duration_s": round((datetime.now() - item_start).total_seconds(), 1)})
                print(f"  -> 失败: {exc}")
            continue

        cmd = ["uv", "run", "python", "demo.py"]
        print(f"[{index}/{len(models)}] 运行 {model_id} -> {cache_dir} ...")
        if args.dry_run:
            print(f"  -> cd {work_dir} && {' '.join(cmd)}")
            continue
        try:
            env = os.environ.copy()
            env.pop("VIRTUAL_ENV", None)
            env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
            result = subprocess.run(cmd, cwd=work_dir, timeout=3600, env=env)
            item_end = datetime.now()
            duration_s = (item_end - item_start).total_seconds()
            if result.returncode != 0:
                reason = f"exit={result.returncode}"
                failed.append((model_id, reason))
                details.append({"model_id": model_id, "status": "failed", "reason": reason, "work_dir": str(work_dir), "cache_dir": str(cache_dir), "started": item_start.strftime("%Y-%m-%d %H:%M:%S"), "ended": item_end.strftime("%Y-%m-%d %H:%M:%S"), "duration_s": round(duration_s, 1)})
                print(f"  -> 失败 exit={result.returncode}")
            else:
                succeeded.append(model_id)
                details.append({"model_id": model_id, "status": "success", "work_dir": str(work_dir), "cache_dir": str(cache_dir), "started": item_start.strftime("%Y-%m-%d %H:%M:%S"), "ended": item_end.strftime("%Y-%m-%d %H:%M:%S"), "duration_s": round(duration_s, 1)})
                print("  -> 完成")
        except subprocess.TimeoutExpired:
            item_end = datetime.now()
            duration_s = (item_end - item_start).total_seconds()
            failed.append((model_id, "timeout"))
            details.append({"model_id": model_id, "status": "timeout", "reason": "timeout", "work_dir": str(work_dir), "cache_dir": str(cache_dir), "started": item_start.strftime("%Y-%m-%d %H:%M:%S"), "ended": item_end.strftime("%Y-%m-%d %H:%M:%S"), "duration_s": round(duration_s, 1)})
            print("  -> 超时")
        except Exception as exc:
            item_end = datetime.now()
            duration_s = (item_end - item_start).total_seconds()
            failed.append((model_id, str(exc)))
            details.append({"model_id": model_id, "status": "error", "reason": str(exc), "work_dir": str(work_dir), "cache_dir": str(cache_dir), "started": item_start.strftime("%Y-%m-%d %H:%M:%S"), "ended": item_end.strftime("%Y-%m-%d %H:%M:%S"), "duration_s": round(duration_s, 1)})
            print(f"  -> 异常: {exc}")

    if not args.dry_run:
        duration = (datetime.now() - start_time).total_seconds()
        end_time = datetime.now()
        lines = [
            f"=== run_completed_adaptations {start_time.strftime('%Y-%m-%d %H:%M:%S')} ===",
            f"ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"duration: {duration:.1f}s",
            f"skipped_from_record: {skipped_from_record}",
            f"total: {len(models)}",
            f"succeeded: {len(succeeded)}",
            f"failed: {len(failed)}",
            "",
            "Succeeded:",
            *([f"  - {model_id}" for model_id in succeeded] if succeeded else ["  (none)"]),
            "",
            "Failed:",
            *([f"  - {model_id}: {reason}" for model_id, reason in failed] if failed else ["  (none)"]),
            "",
            "--- Details (per model) ---",
        ]
        for detail in details:
            lines.extend(detail_to_lines(detail))
        RECORD_FILE.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n[run_completed_adaptations] 记录已写入: {RECORD_FILE}")

    if failed:
        print(f"\n[run_completed_adaptations] 失败 {len(failed)} 个:")
        for model_id, reason in failed:
            print(f"  - {model_id}: {reason}")
        return 1

    print("\n[run_completed_adaptations] 全部完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
