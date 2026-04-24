#!/usr/bin/env python3
"""
Adaptation Manager - 第一阶段运行与产出管理工具。
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from adaptation.scripts import check_adaptation as check_adaptation_module
from adaptation.scripts import package_adaptations as package_adaptations_module
from adaptation.scripts import run_completed_adaptations as run_completed_adaptations_module
from scripts.adaptation_utils import model_id_to_adaptation_path, model_id_to_safe_name

PROJECT_ROOT = _PROJECT_ROOT
DB_PATH = PROJECT_ROOT / "board.db"
ADAPTATIONS_DIR = PROJECT_ROOT / "adaptations"
ARTIFACTS_DIR = PROJECT_ROOT / "dashboard" / "artifacts"


def get_db_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def get_models(status: str | None = None) -> list[dict]:
    conn = get_db_connection()
    cursor = conn.cursor()
    if status:
        cursor.execute(
            """
            SELECT model_id, adaptation_status, adaptation_path, adaptation_owner, adaptation_notes
            FROM models
            WHERE adaptation_status = ?
            ORDER BY model_id
            """,
            (status,),
        )
    else:
        cursor.execute(
            """
            SELECT model_id, adaptation_status, adaptation_path, adaptation_owner, adaptation_notes
            FROM models
            WHERE adaptation_status != '' AND adaptation_status IS NOT NULL
            ORDER BY model_id
            """
        )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _resolve_adaptation_dir(adaptation_path: str | None, model_id: str) -> Path:
    path_value = (adaptation_path or "").strip()
    if path_value:
        return PROJECT_ROOT / path_value
    return PROJECT_ROOT / model_id_to_adaptation_path(model_id)


def cmd_list(args):
    status = args.status or "completed"
    models = get_models(status)
    print(f"\nAdaptation 模型列表 (adaptation_status={status})\n")
    if not models:
        print("没有找到符合条件的模型\n")
        return
    for index, model in enumerate(models, 1):
        print(f"{index}. {model['model_id']}")
        print(f"   adaptation={model['adaptation_status']} owner={model['adaptation_owner']} path={model['adaptation_path']}")
        if model.get("adaptation_notes"):
            print(f"   notes={model['adaptation_notes']}")


def cmd_check(args):
    exit_code = 0
    if args.model:
        conn = get_db_connection()
        row = conn.execute("SELECT adaptation_path FROM models WHERE model_id = ?", (args.model,)).fetchone()
        conn.close()
        if not row:
            raise SystemExit(f"模型不存在: {args.model}")
        target_dir = _resolve_adaptation_dir(row["adaptation_path"], args.model)
        errors = check_adaptation_module.check_adaptation(target_dir, skip_status=args.skip_status)
        if errors:
            exit_code = 1
            print(f"[check] ❌ {target_dir.relative_to(PROJECT_ROOT)}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"[check] ✅ {target_dir.relative_to(PROJECT_ROOT)}")
    else:
        argv = []
        if args.skip_status:
            argv.append("--skip-status")
        if args.db_only:
            argv.append("--db-only")
        original_argv = sys.argv
        try:
            sys.argv = ["check_adaptation.py", *argv]
            exit_code = check_adaptation_module.main()
        finally:
            sys.argv = original_argv
    raise SystemExit(exit_code)


def cmd_run(args):
    import sys

    argv = ["run_completed_adaptations.py"]
    if args.dry_run:
        argv.append("--dry-run")
    if args.download_only:
        argv.append("--download-only")
    if args.max_workers is not None:
        argv.extend(["--max-workers", str(args.max_workers)])
    original_argv = sys.argv
    try:
        sys.argv = argv
        raise SystemExit(run_completed_adaptations_module.main())
    finally:
        sys.argv = original_argv


def cmd_artifacts(args):
    target_dir = ARTIFACTS_DIR
    if not target_dir.exists():
        print("dashboard/artifacts 不存在")
        return
    if args.model:
        zip_name_prefix = model_id_to_safe_name(args.model)
        candidates = sorted(path for path in target_dir.glob("*.zip") if path.stem == zip_name_prefix or path.stem.endswith(zip_name_prefix))
    else:
        candidates = sorted(target_dir.glob("*.zip"))
    if not candidates:
        print("没有找到 adaptation 打包产物")
        return
    for path in candidates:
        print(path.relative_to(PROJECT_ROOT))


def cmd_pack(args):
    package_adaptations_module.package_adaptations(board_db_path=str(DB_PATH), output_dir=str(ARTIFACTS_DIR), incremental=not args.no_incremental)


def cmd_clean(args):
    if not ARTIFACTS_DIR.exists():
        return
    removed = 0
    if args.model:
        stem = model_id_to_safe_name(args.model)
        candidates = [path for path in ARTIFACTS_DIR.glob("*.zip") if path.stem == stem or path.stem.endswith(stem)]
    else:
        candidates = list(ARTIFACTS_DIR.glob("*.zip"))
    for path in candidates:
        path.unlink(missing_ok=True)
        removed += 1
    print(f"已删除 {removed} 个 adaptation zip 产物")


def main():
    parser = argparse.ArgumentParser(description="Adaptation Manager")
    sub = parser.add_subparsers(dest="cmd", required=True)

    list_parser = sub.add_parser("list", help="列出 adaptation 模型")
    list_parser.add_argument("--status", help="过滤 adaptation_status")
    list_parser.set_defaults(func=cmd_list)

    check_parser = sub.add_parser("check", help="运行 adaptation 规范检查")
    check_parser.add_argument("--model", help="只检查指定 model_id")
    check_parser.add_argument("--skip-status", action="store_true")
    check_parser.add_argument("--db-only", action="store_true")
    check_parser.set_defaults(func=cmd_check)

    run_parser = sub.add_parser("run", help="运行 completed adaptation 的 demo.py")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--download-only", action="store_true")
    run_parser.add_argument("--max-workers", type=int, default=8)
    run_parser.set_defaults(func=cmd_run)

    artifacts_parser = sub.add_parser("artifacts", help="列出 adaptation zip 产物")
    artifacts_parser.add_argument("--model", help="只显示指定 model_id")
    artifacts_parser.set_defaults(func=cmd_artifacts)

    pack_parser = sub.add_parser("pack", help="打包 adaptation 产物")
    pack_parser.add_argument("--no-incremental", action="store_true")
    pack_parser.set_defaults(func=cmd_pack)

    clean_parser = sub.add_parser("clean", help="删除 adaptation zip 产物")
    clean_parser.add_argument("--model", help="只删除指定 model_id")
    clean_parser.set_defaults(func=cmd_clean)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
