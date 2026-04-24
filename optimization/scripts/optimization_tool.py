#!/usr/bin/env python3
"""
Optimization Tool - 聚合 perf 指标、对比 perf 产出。

子命令:
  aggregate  - 聚合 benchmark_metrics_*_perf.json，生成汇总报告
  compare    - 对比 CUDA/NPU 的 outputs_*_perf.pt（委托 benchmark_tool compare）
  list       - 列出 optimization 任务状态

Usage:
  uv run python optimization/scripts/optimization_tool.py aggregate
  uv run python optimization/scripts/optimization_tool.py compare [--adapt ADAPT] [--all]
  uv run python optimization/scripts/optimization_tool.py list [--status STATUS]
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 路径常量
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", _SCRIPT_DIR.parent.parent))
DB_PATH = _PROJECT_ROOT / "board.db"
ADAPTATIONS_DIR = _PROJECT_ROOT / "adaptations"
OPTIMIZATION_DIR = _PROJECT_ROOT / "optimization"
REPORTS_DIR = OPTIMIZATION_DIR / "reports"

# 颜色
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
NC = "\033[0m"


def _parse_iso_datetime(raw_value):
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    try:
        return datetime.fromisoformat(raw_value)
    except ValueError:
        return None


def _extract_wall_clock_s(metric: dict) -> float | None:
    for key in ("wall_clock_s", "total_time_s", "duration_s", "elapsed_s"):
        value = metric.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > 0:
            return float(value)
    start_dt = _parse_iso_datetime(metric.get("start_time"))
    end_dt = _parse_iso_datetime(metric.get("end_time"))
    if start_dt and end_dt:
        duration = (end_dt - start_dt).total_seconds()
        if duration > 0:
            return duration
    return None


def collect_perf_metrics() -> dict[str, list[dict]]:
    """扫描 adaptations/* 收集 benchmark_metrics_*_perf.json。"""
    out: dict[str, list[dict]] = {}
    if not ADAPTATIONS_DIR.exists():
        return out

    for adir in ADAPTATIONS_DIR.iterdir():
        if not adir.is_dir():
            continue
        metrics_files = sorted(adir.glob("benchmark_metrics_*_perf.json"))
        if not metrics_files:
            continue

        out[adir.name] = []
        for mf in metrics_files:
            try:
                data = json.loads(mf.read_text(encoding="utf-8"))
                data["_file"] = mf.name
                out[adir.name].append(data)
            except Exception:
                pass
    return out


def cmd_aggregate(args):
    """聚合 perf 指标。"""
    print(f"{CYAN}{'=' * 60}{NC}")
    print(f"{CYAN}{BOLD}  Aggregate Optimization (perf) Metrics{NC}")
    print(f"{CYAN}{'=' * 60}{NC}\n")

    metrics_by_adapt = collect_perf_metrics()
    reports = []

    for safe, metrics_list in metrics_by_adapt.items():
        rec = {
            "adaptation": safe,
            "adaptation_path": f"adaptations/{safe}",
            "perf_metrics": [],
        }
        for m in metrics_list:
            entry = {
                "latency_s": m.get("latency_s"),
                "peak_memory_mb": m.get("peak_memory_mb"),
                "device": m.get("device"),
                "output_type": m.get("output_type"),
                "file": m.get("_file"),
            }
            # 优化前后对比（若 accuracy_run_perf 已合并 baseline）
            if m.get("baseline_file"):
                entry["baseline_latency_s"] = m.get("baseline_latency_s")
                entry["speedup_ratio"] = m.get("speedup_ratio")
                entry["latency_reduction_pct"] = m.get("latency_reduction_pct")
                entry["baseline_wall_clock_s"] = m.get("baseline_wall_clock_s")
                entry["perf_wall_clock_s"] = _extract_wall_clock_s(m)
                if isinstance(entry["baseline_wall_clock_s"], (int, float)) and isinstance(entry["perf_wall_clock_s"], (int, float)) and entry["baseline_wall_clock_s"] > 0 and entry["perf_wall_clock_s"] > 0:
                    entry["wall_clock_speedup_ratio"] = round(entry["baseline_wall_clock_s"] / entry["perf_wall_clock_s"], 4)
                    entry["wall_clock_reduction_pct"] = round((1 - entry["perf_wall_clock_s"] / entry["baseline_wall_clock_s"]) * 100, 2)
                entry["baseline_peak_memory_mb"] = m.get("baseline_peak_memory_mb")
                entry["memory_reduction_pct"] = m.get("memory_reduction_pct")
                entry["baseline_ttft_ms"] = m.get("baseline_ttft_ms")
                entry["ttft_speedup_ratio"] = m.get("ttft_speedup_ratio")
                entry["baseline_tpot_ms"] = m.get("baseline_tpot_ms")
                entry["tpot_speedup_ratio"] = m.get("tpot_speedup_ratio")
            # 产出对比（baseline vs perf outputs，若 run 时已执行 compare）
            if m.get("output_compare"):
                entry["output_compare"] = m.get("output_compare")
            rec["perf_metrics"].append(entry)
        reports.append(rec)

    out_dir = Path(args.reports_dir or str(REPORTS_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "count": len(reports),
        "reports": reports,
    }

    out_json = out_dir / "optimization_aggregate.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{GREEN}✓{NC} Wrote {out_json} ({len(reports)} adaptations with perf metrics)")
    print()


def cmd_compare(args):
    """对比 CUDA/NPU perf 产出，委托 benchmark_tool compare。"""
    bench_tool = _PROJECT_ROOT / "benchmark" / "scripts" / "benchmark_tool.py"
    if not bench_tool.exists():
        print(f"{RED}Error:{NC} benchmark_tool.py not found at {bench_tool}")
        sys.exit(1)

    cmd = [sys.executable, str(bench_tool), "compare"]
    if args.adapt:
        cmd.extend(["--adapt", args.adapt])
    if args.all:
        cmd.append("--all")

    # benchmark_tool compare 会扫描 outputs_*.pt，自动匹配 *_perf.pt 对
    subprocess.run(cmd, cwd=str(_PROJECT_ROOT))


def cmd_list(args):
    """列出 optimization 任务。"""
    if not DB_PATH.exists():
        print(f"{YELLOW}board.db 不存在{NC}")
        return

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if args.status:
        cur.execute(
            "SELECT model_id, adaptation_status, benchmark_status, optimization_status, optimization_owner, adaptation_path FROM models WHERE optimization_status = ?",
            (args.status,),
        )
    else:
        cur.execute(
            "SELECT model_id, adaptation_status, benchmark_status, optimization_status, optimization_owner, adaptation_path FROM models WHERE optimization_status != '' AND optimization_status IS NOT NULL"
        )

    rows = cur.fetchall()
    conn.close()

    print(f"{CYAN}{'=' * 60}{NC}")
    print(f"{CYAN}{BOLD}  Optimization Tasks{NC}")
    print(f"{CYAN}{'=' * 60}{NC}\n")

    if not rows:
        print(f"{YELLOW}无 optimization 任务{NC}")
        return

    for r in rows:
        d = dict(r)
        print(f"  {BOLD}{d['model_id']}{NC}")
        print(
            f"    adaptation={d['adaptation_status']} benchmark={d['benchmark_status']} "
            f"optimization={d['optimization_status']} owner={d['optimization_owner']}"
        )
        print(f"    path={d['adaptation_path']}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Optimization 测评工具")
    sub = parser.add_subparsers(dest="cmd", required=True)

    agg = sub.add_parser("aggregate", help="聚合 perf 指标")
    agg.add_argument("--reports-dir", default=str(REPORTS_DIR), help="输出目录")
    agg.set_defaults(func=cmd_aggregate)

    cmp = sub.add_parser("compare", help="对比 CUDA/NPU perf 产出")
    cmp.add_argument("--adapt", help="指定 adaptation 目录名")
    cmp.add_argument("--all", action="store_true", help="对比所有")
    cmp.set_defaults(func=cmd_compare)

    lst = sub.add_parser("list", help="列出 optimization 任务")
    lst.add_argument("--status", help="过滤 optimization_status (completed/pending/in_progress)")
    lst.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
