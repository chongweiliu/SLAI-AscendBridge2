#!/usr/bin/env python3
"""
Optimization Manager - 整合 NPU 优化运行、产出管理工具。

功能：
  list:      列出模型（从 board.db，按 optimization_status）
  run:       运行 accuracy_run_perf.py
  artifacts: 列出 perf 产出文件
  pack:      打包 perf 产出为 zip
  unpack:    解包还原
  clean:     清空 perf 产出文件

Usage:
  uv run python optimization/scripts/optimization_manager.py list [--status STATUS]
  uv run python optimization/scripts/optimization_manager.py run [--model MODEL_ID] [--use-pretrained] [--max-samples N]
  uv run python optimization/scripts/optimization_manager.py artifacts [MODEL_ID]
  uv run python optimization/scripts/optimization_manager.py pack [--output FILE.zip] [--model MODEL_ID]
  uv run python optimization/scripts/optimization_manager.py unpack --input FILE.zip
  uv run python optimization/scripts/optimization_manager.py clean [--model MODEL_ID]
"""

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "board.db"
ADAPTATIONS_DIR = PROJECT_ROOT / "adaptations"

# 颜色
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
NC = "\033[0m"

PERF_PATTERNS = ["*_perf.json", "*_perf.pt", "accuracy_run_perf.py"]


def get_db_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def get_models(status: str | None = None, require_benchmark_completed: bool = False) -> list[dict]:
    """从 board.db 读取模型列表，按 optimization_status 过滤。"""
    conn = get_db_connection()
    cur = conn.cursor()
    if status:
        if require_benchmark_completed:
            cur.execute(
                "SELECT model_id, optimization_status, adaptation_path, benchmark_status FROM models WHERE optimization_status = ? AND benchmark_status = 'completed' ORDER BY model_id",
                (status,),
            )
        else:
            cur.execute(
                "SELECT model_id, optimization_status, adaptation_path, benchmark_status FROM models WHERE optimization_status = ? ORDER BY model_id",
                (status,),
            )
    else:
        cur.execute(
            "SELECT model_id, optimization_status, adaptation_path, benchmark_status FROM models WHERE optimization_status != '' AND optimization_status IS NOT NULL ORDER BY model_id"
        )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def fmt_size(n: int) -> str:
    for unit in ["G", "M", "K"]:
        factor = {"G": 1024**3, "M": 1024**2, "K": 1024}[unit]
        if n >= factor:
            return f"{n / factor:.2f}{unit}"
    return f"{n}B"


def cmd_list(args):
    """列出模型。"""
    status = args.status or "completed"
    models = get_models(status)
    print(f"\n{BOLD}Optimization 模型列表 (optimization_status={status}){NC}\n")
    if not models:
        print(f"{YELLOW}没有找到符合条件的模型{NC}\n")
        return
    for i, m in enumerate(models, 1):
        print(f"  {i}. {CYAN}{m['model_id']}{NC}")
        print(f"     benchmark={m['benchmark_status']} optimization={m['optimization_status']} path={m['adaptation_path']}")
    print()


def cmd_run(args):
    """运行 accuracy_run_perf.py。"""
    conn = get_db_connection()
    cur = conn.cursor()
    if args.model:
        cur.execute(
            "SELECT model_id, adaptation_path FROM models WHERE model_id = ? AND adaptation_status = 'completed' AND benchmark_status = 'completed'",
            (args.model,),
        )
        row = cur.fetchone()
        conn.close()
        if not row:
            print(f"{RED}模型 {args.model} 需 adaptation_status=completed 且 benchmark_status=completed{NC}")
            sys.exit(1)
        models = [dict(row)]
    else:
        cur.execute(
            "SELECT model_id, adaptation_path FROM models WHERE adaptation_status = 'completed' AND benchmark_status = 'completed' ORDER BY model_id"
        )
        models = [dict(r) for r in cur.fetchall()]
        conn.close()
        if not models:
            print(f"{YELLOW}没有找到 adaptation_status=completed 且 benchmark_status=completed 的模型{NC}")
            return

    for m in models:
        adapt_path = m.get("adaptation_path") or m["model_id"].replace("/", "_")
        if adapt_path.startswith("adaptations/"):
            adapt_dir = PROJECT_ROOT / adapt_path
        else:
            adapt_dir = ADAPTATIONS_DIR / adapt_path

        perf_py = adapt_dir / "accuracy_run_perf.py"
        if not perf_py.exists():
            print(f"{YELLOW}跳过 {m['model_id']}: 无 accuracy_run_perf.py{NC}")
            continue

        cmd = ["uv", "run", "python", "accuracy_run_perf.py", "run", "--max-samples", str(args.max_samples)]
        if args.use_pretrained:
            cmd.append("--use-pretrained")

        print(f"[run] {m['model_id']} @ {adapt_dir}")
        subprocess.run(cmd, cwd=str(adapt_dir), env={**os.environ, "PROJECT_ROOT": str(PROJECT_ROOT)})


def cmd_artifacts(args):
    """列出 perf 产出。"""
    model_id = getattr(args, "model_id", None)
    if model_id:
        conn = get_db_connection()
        cur = conn.execute("SELECT adaptation_path FROM models WHERE model_id = ?", (model_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            print(f"{RED}模型不存在{NC}")
            sys.exit(1)
        adapt_path = row["adaptation_path"] or model_id.replace("/", "_")
        dirs = [(model_id, PROJECT_ROOT / adapt_path if adapt_path.startswith("adaptations/") else ADAPTATIONS_DIR / adapt_path)]
    else:
        models = get_models()  # 所有有 optimization_status 的模型
        dirs = []
        for m in models:
            adapt_path = m.get("adaptation_path") or m["model_id"].replace("/", "_")
            adapt_dir = PROJECT_ROOT / adapt_path if adapt_path.startswith("adaptations/") else ADAPTATIONS_DIR / adapt_path
            dirs.append((m["model_id"], adapt_dir))

    print(f"\n{BOLD}Optimization 产出文件{NC}\n")
    for name, adapt_dir in dirs:
        print(f"{CYAN}[{name}]{NC}")
        if not adapt_dir.is_dir():
            print(f"  {YELLOW}目录不存在{NC}\n")
            continue
        for pat in PERF_PATTERNS:
            for f in sorted(adapt_dir.glob(pat)):
                print(f"  {f.name}  {fmt_size(f.stat().st_size)}")
        print()


def cmd_pack(args):
    """打包 perf 产出。"""
    if args.model:
        models = [m for m in get_models() if m["model_id"] == args.model]
        if not models:
            print(f"{RED}模型不存在{NC}")
            sys.exit(1)
    else:
        models = get_models(status="completed")

    output = Path(args.output) if args.output else PROJECT_ROOT / f"optimization_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    files_to_pack = []
    for m in models:
        adapt_path = m.get("adaptation_path") or m["model_id"].replace("/", "_")
        adapt_dir = PROJECT_ROOT / adapt_path if adapt_path.startswith("adaptations/") else ADAPTATIONS_DIR / adapt_path
        if not adapt_dir.is_dir():
            continue
        for pat in ["*_perf.json", "*_perf.pt", "accuracy_run_perf.py"]:
            for f in adapt_dir.glob(pat):
                files_to_pack.append((f, f"{adapt_dir.name}/{f.name}"))

    if not files_to_pack:
        print(f"{YELLOW}无产出可打包{NC}")
        return

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps({"created_at": datetime.now().isoformat(), "files_count": len(files_to_pack)}, indent=2))
        for f, arc in files_to_pack:
            zf.write(f, arc)
    print(f"{GREEN}✓{NC} Wrote {output} ({len(files_to_pack)} files)")


def cmd_unpack(args):
    """解包。"""
    inp = Path(args.input)
    if not inp.exists():
        print(f"{RED}文件不存在: {inp}{NC}")
        sys.exit(1)
    with zipfile.ZipFile(inp, "r") as zf:
        zf.extractall(ADAPTATIONS_DIR)
    print(f"{GREEN}✓{NC} Unpacked to {ADAPTATIONS_DIR}")


def cmd_clean(args):
    """清空 perf 产出。"""
    if args.model:
        models = [m for m in get_models() if m["model_id"] == args.model]
    else:
        models = get_models()

    removed = 0
    for m in models:
        adapt_path = m.get("adaptation_path") or m["model_id"].replace("/", "_")
        adapt_dir = PROJECT_ROOT / adapt_path if adapt_path.startswith("adaptations/") else ADAPTATIONS_DIR / adapt_path
        if not adapt_dir.is_dir():
            continue
        for pat in ["*_perf.json", "*_perf.pt"]:
            for f in adapt_dir.glob(pat):
                f.unlink()
                removed += 1
    print(f"{GREEN}✓{NC} Removed {removed} perf files")


def main():
    parser = argparse.ArgumentParser(description="Optimization Manager")
    sub = parser.add_subparsers(dest="cmd", required=True)

    l = sub.add_parser("list", help="列出模型")
    l.add_argument("--status", help="过滤 optimization_status (completed/pending/in_progress)")
    l.set_defaults(func=cmd_list)

    r = sub.add_parser("run", help="运行 accuracy_run_perf.py")
    r.add_argument("--model", help="指定模型")
    r.add_argument("--use-pretrained", action="store_true")
    r.add_argument("--max-samples", type=int, default=50)
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("artifacts", help="列出产出")
    a.add_argument("model_id", nargs="?", help="指定模型")
    a.set_defaults(func=cmd_artifacts)

    p = sub.add_parser("pack", help="打包产出")
    p.add_argument("--output", help="输出 zip 路径")
    p.add_argument("--model", help="指定模型")
    p.set_defaults(func=cmd_pack)

    u = sub.add_parser("unpack", help="解包")
    u.add_argument("--input", required=True, help="zip 路径")
    u.set_defaults(func=cmd_unpack)

    c = sub.add_parser("clean", help="清空产出")
    c.add_argument("--model", help="指定模型")
    c.set_defaults(func=cmd_clean)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
