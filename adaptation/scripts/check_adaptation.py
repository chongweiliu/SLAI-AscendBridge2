#!/usr/bin/env python3
"""
强制检查 adaptation 目录（demo.py 等）是否符合适配完成规范。

用法:
    uv run python adaptation/scripts/check_adaptation.py
    uv run python adaptation/scripts/check_adaptation.py --adapt xxx
    uv run python adaptation/scripts/check_adaptation.py --skip-status
    uv run python adaptation/scripts/check_adaptation.py --db-only
"""

import json
import re
import sqlite3
import sys
from pathlib import Path


def get_completed_adapt_names(project_root: Path) -> list[str]:
    """从 board.db 获取 adaptation_status=completed 的 adaptation 目录名列表。"""
    db_path = project_root / "board.db"
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        """
        SELECT adaptation_path
        FROM models
        WHERE adaptation_status = ?
          AND adaptation_path IS NOT NULL
          AND adaptation_path != ''
        """,
        ("completed",),
    )
    paths = [r[0] for r in cur.fetchall()]
    conn.close()
    names = []
    for path_value in paths:
        normalized = path_value.strip().strip("/")
        if normalized.startswith("adaptations/"):
            names.append(normalized.replace("adaptations/", ""))
        else:
            names.append(normalized)
    return sorted(set(names))


def _check_dangerous_cache_patterns(content: str) -> list[str]:
    errors: list[str] = []
    relative_models_patterns = [
        r'(?:HF_HOME|TRANSFORMERS_CACHE|CACHE_DIR|cache_dir)\s*=\s*["\'](?:\.\/)?models["\']',
        r'Path\s*\(\s*["\']models["\']\s*\)',
    ]
    if any(re.search(pattern, content) for pattern in relative_models_patterns):
        errors.append("缓存目录禁止写成相对路径 models/；应固定到 Path(__file__).resolve().parent / 'models'")
    if re.search(r'Path\s*\(\s*__file__\s*\)\.resolve\(\)\.parent\.parent\s*/\s*["\']models["\']', content):
        errors.append("缓存目录禁止使用 Path(__file__).resolve().parent.parent / 'models'；这会指向项目根 models/")
    if "models" in content and ("Path.cwd()" in content or "os.getcwd()" in content or "getcwd()" in content):
        errors.append("缓存目录禁止基于 cwd/getcwd() 推导 models/；必须固定到 adaptation_path/models/")
    return errors


def check_adaptation(adapt_dir: Path, skip_status: bool = False) -> list[str]:
    """检查单个 adaptation 目录，返回违规列表。"""
    errors: list[str] = []
    adapt_name = adapt_dir.name
    adapt_lower = adapt_name.lower()
    adult_keywords = ["nsfw", "porn", "xxx", "adult", "hentai", "erotic", "nude", "sex", "sexy", "fetish", "onlyfans", "playboy"]
    if any(keyword in adapt_lower for keyword in adult_keywords):
        errors.append("模型包含成人向内容关键词")
        return errors

    if not adapt_dir.exists():
        errors.append(f"目录不存在 {adapt_dir}")
        return errors

    demo_path = adapt_dir / "demo.py"
    if not demo_path.exists():
        errors.append("demo.py 不存在")
        return errors

    demo_content = demo_path.read_text(encoding="utf-8")
    if len(demo_content.strip().split("\n")) < 20:
        errors.append("demo.py 内容过少（至少 20 行）")
    required_keywords = ["import torch", "device", "--dry-run"]
    missing = [keyword for keyword in required_keywords if keyword not in demo_content]
    if missing:
        errors.append(f"demo.py 缺少关键词: {missing}")
    errors.extend(_check_dangerous_cache_patterns(demo_content))

    pyproject_path = adapt_dir / "pyproject.toml"
    if not pyproject_path.exists():
        errors.append("pyproject.toml 不存在")
    else:
        py_content = pyproject_path.read_text(encoding="utf-8")
        if "cuda" not in py_content or "ascend" not in py_content:
            errors.append("pyproject.toml 应包含 optional-dependencies cuda 与 ascend")

    if not (adapt_dir / "README.md").exists():
        errors.append("README.md 不存在")

    lock_file = adapt_dir / "uv.lock"
    if not lock_file.exists():
        errors.append("uv.lock 不存在（需运行 uv sync --extra ascend）")
    else:
        lock_content = lock_file.read_text(encoding="utf-8").lower()
        has_ascend = "ascend" in lock_content or "torch_npu" in lock_content
        has_cuda = "cuda" in lock_content
        if not has_ascend and not has_cuda:
            errors.append("uv.lock 未包含 ascend 或 cuda 相关依赖")

    if not skip_status:
        status_file = adapt_dir / ".status.json"
        if not status_file.exists():
            errors.append(".status.json 不存在（可用 --skip-status 跳过）")
        else:
            try:
                status_data = json.loads(status_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f".status.json 格式错误: {exc}")
            else:
                if status_data.get("status") != "completed":
                    errors.append(f".status.json 状态不是 completed，当前为 {status_data.get('status')}")
                dry_run = status_data.get("stages", {}).get("dry_run", {})
                if not dry_run:
                    errors.append(".status.json 缺少 dry_run 阶段记录")
                else:
                    device = dry_run.get("device", "")
                    npu_ok = dry_run.get("npu_detected")
                    cuda_ok = dry_run.get("cuda_detected") or str(device).startswith("cuda")
                    if not npu_ok and not cuda_ok:
                        errors.append(".status.json 显示 dry_run 未检测到 NPU 或 CUDA")

    output_path = adapt_dir / "output.txt"
    if not output_path.exists():
        errors.append("output.txt 不存在（需运行 uv run python demo.py --dry-run > output.txt 2>&1）")
    else:
        output_content = output_path.read_text(encoding="utf-8", errors="ignore")
        project_root = adapt_dir.parent.parent if adapt_dir.parent.name == "adaptations" else None
        if project_root is not None and str(project_root / "models") in output_content:
            errors.append("output.txt 显示模型缓存落到项目根 models/，必须改为 adaptation_path/models/")

    return errors


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="检查 adaptation 目录是否符合适配完成规范")
    parser.add_argument("--adapt", default=None, help="仅检查指定 adaptation 目录名")
    parser.add_argument("--skip-status", action="store_true", help="跳过 .status.json 检查")
    parser.add_argument("--base-dir", default="adaptations", help="适配目录根路径")
    parser.add_argument("--warn-only", action="store_true", help="仅警告不退出码 1")
    parser.add_argument("--db-only", action="store_true", help="仅检查 board.db 中 adaptation_status=completed 的 adaptation")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    adaptations_dir = project_root / args.base_dir

    if args.adapt:
        paths = [adaptations_dir / args.adapt]
        if not paths[0].exists():
            print(f"[check] 目录不存在: {paths[0]}")
            return 1
    elif args.db_only:
        completed_names = get_completed_adapt_names(project_root)
        if not completed_names:
            print("[check] board.db 中无 adaptation_status=completed 的 adaptation，跳过检查")
            return 0
        paths = [adaptations_dir / name for name in completed_names]
        paths = [path for path in paths if path.exists()]
        print(f"[check] 仅检查 db 中 {len(completed_names)} 个 completed adaptation（存在 {len(paths)} 个）")
    else:
        paths = sorted(path for path in adaptations_dir.iterdir() if path.is_dir())

    total_errors = 0
    for path in paths:
        rel = path.relative_to(project_root)
        errors = check_adaptation(path, skip_status=args.skip_status)
        if errors:
            total_errors += len(errors)
            print(f"\n[check] ❌ {rel}")
            for error in errors:
                print(f"       - {error}")
        else:
            print(f"[check] ✅ {rel}")

    if total_errors > 0:
        print(f"\n[check] 共 {total_errors} 项违规")
        return 0 if args.warn_only else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
