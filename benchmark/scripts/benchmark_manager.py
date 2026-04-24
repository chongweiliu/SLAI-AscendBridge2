#!/usr/bin/env python3
"""
Benchmark Manager - 整合模型 benchmark 运行、产出管理工具。

功能：
  - list:      列出模型（从 board.db）
  - run:       运行 benchmark
  - artifacts: 列出产出文件
  - pack:      打包产出文件为 zip
  - unpack:    解包还原产出文件
  - clean:     一键清空产出文件

Usage:
  uv run python benchmark/scripts/benchmark_manager.py list [--status STATUS]
  uv run python benchmark/scripts/benchmark_manager.py run [--model MODEL_ID] [--hardware cuda|npu] [--pretrained] [--max-samples N] [--resume JSON_PATH]
  uv run python benchmark/scripts/benchmark_manager.py artifacts [MODEL_ID]
  uv run python benchmark/scripts/benchmark_manager.py pack [--output FILE.zip] [--model MODEL_ID]
  uv run python benchmark/scripts/benchmark_manager.py unpack --input FILE.zip
  uv run python benchmark/scripts/benchmark_manager.py clean [--model MODEL_ID]
"""

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "board.db"
ADAPTATIONS_DIR = PROJECT_ROOT / "adaptations"

# 颜色定义
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
NC = "\033[0m"  # No Color


def print_header(title: str, char: str = "=", width: int = 60):
    """打印标题头。"""
    print()
    print(f"{BLUE}{char * width}{NC}")
    print(f"{BLUE}{BOLD}  {title}{NC}")
    print(f"{BLUE}{char * width}{NC}")
    print()


def print_section(title: str, char: str = "-", width: int = 50):
    """打印小节头。"""
    print()
    print(f"{CYAN}{char * width}{NC}")
    print(f"{CYAN}  {title}{NC}")
    print(f"{CYAN}{char * width}{NC}")


def print_success(msg: str):
    """打印成功信息。"""
    print(f"{GREEN}✓ {msg}{NC}")


def print_error(msg: str):
    """打印错误信息。"""
    print(f"{RED}✗ {msg}{NC}")


def print_warning(msg: str):
    """打印警告信息。"""
    print(f"{YELLOW}⚠ {msg}{NC}")


def print_info(msg: str):
    """打印普通信息。"""
    print(f"  {msg}")


def get_db_connection():
    """获取数据库连接。"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def fmt_size(n: int) -> str:
    """将字节数格式化为人类可读大小。"""
    for unit in ["G", "M", "K"]:
        factor = {"G": 1024**3, "M": 1024**2, "K": 1024}[unit]
        if n >= factor:
            return f"{n / factor:.2f}{unit}"
    return f"{n}B"


def fmt_duration(seconds: int) -> str:
    """格式化持续时间。"""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}m {s}s"
    else:
        h, m = divmod(seconds // 60, 60)
        s = seconds % 60
        return f"{h}h {m}m {s}s"


def get_models(
    status: Optional[str] = None,
    *,
    require_both_completed: bool = False,
) -> list[dict]:
    """从 board.db 读取模型列表。

    Args:
        status: 过滤 benchmark_status，None 表示所有有 benchmark_status 的模型
        require_both_completed: 为 True 时仅返回 adaptation_status 与 benchmark_status 均为 completed 的模型

    Returns:
        模型列表，每项包含 model_id, benchmark_status, adaptation_path 等
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    if status:
        if require_both_completed:
            cursor.execute(
                """
                SELECT model_id, benchmark_status, adaptation_path, adaptation_status
                FROM models
                WHERE benchmark_status = ? AND adaptation_status = 'completed'
                ORDER BY model_id
                """,
                (status,),
            )
        else:
            cursor.execute(
                """
                SELECT model_id, benchmark_status, adaptation_path, adaptation_status
                FROM models
                WHERE benchmark_status = ?
                ORDER BY model_id
                """,
                (status,),
            )
    else:
        if require_both_completed:
            cursor.execute(
                """
                SELECT model_id, benchmark_status, adaptation_path, adaptation_status
                FROM models
                WHERE benchmark_status != '' AND benchmark_status IS NOT NULL
                  AND adaptation_status = 'completed'
                ORDER BY model_id
                """
            )
        else:
            cursor.execute(
                """
                SELECT model_id, benchmark_status, adaptation_path, adaptation_status
                FROM models
                WHERE benchmark_status != '' AND benchmark_status IS NOT NULL
                ORDER BY model_id
                """
            )

    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_default_hardware() -> str:
    """获取默认硬件类型 (NPU)。"""
    return "npu"


def cmd_list(args):
    """列出模型。"""
    status = args.status or "completed"  # 默认只显示 completed
    models = get_models(status)

    print_header(f"Benchmark 模型列表 (status={status})")

    if not models:
        print_warning(f"没有找到符合条件的模型 (status={status})")
        return

    print_info(f"共 {BOLD}{len(models)}{NC} 个模型:\n")

    for i, m in enumerate(models, 1):
        adapt_path = m.get("adaptation_path") or ""
        print(f"  {BOLD}{i}.{NC} {CYAN}{m['model_id']}{NC}")
        print(f"      适配状态:     {GREEN}{m['adaptation_status']}{NC}")
        print(f"      Benchmark:   {GREEN}{m['benchmark_status']}{NC}")
        print(f"      适配目录:     {adapt_path}")
        print()


def _model_from_report_entry(m: dict) -> dict:
    """从 JSON 报告条目构建 model 信息。"""
    mid = m.get("model_id")
    adapt_path = m.get("adaptation_path")
    if not adapt_path and m.get("adapt_dir"):
        adapt_dir = Path(m["adapt_dir"])
        try:
            adapt_path = str(adapt_dir.relative_to(PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            adapt_path = adapt_dir.name
            if not adapt_path.startswith("adaptations/"):
                adapt_path = f"adaptations/{adapt_path}"
    if not adapt_path and mid:
        import re

        safe = mid.replace("/", "_").replace("-", "_").lower()
        safe = re.sub(r"[^a-z0-9_]", "_", safe)
        safe = re.sub(r"_+", "_", safe).strip("_")
        adapt_path = f"adaptations/{safe}" if safe else ""
    return {"model_id": mid, "adaptation_path": adapt_path}


def cmd_run(args):
    """运行 benchmark。"""
    # 确定要运行的模型
    if getattr(args, "resume", None):
        # Resume: 从 JSON 恢复，跳过 status=success，重跑 failed 和 running
        if args.model:
            print_error("--resume 与 --model 互斥，请只指定其一")
            sys.exit(1)
        if getattr(args, "from_json", None):
            print_error("--resume 与 --from-json 互斥，请只指定其一")
            sys.exit(1)
        json_path = Path(args.resume)
        if not json_path.is_file():
            print_error(f"JSON 文件不存在: {json_path}")
            sys.exit(1)
        with open(json_path, encoding="utf-8") as f:
            run_report = json.load(f)
        # 从 board.db 获取完整名单（与 run 相同），按原顺序接着跑
        full_list = get_models(status="completed", require_both_completed=True)
        if not full_list:
            print_warning("board.db 中无 completed 模型")
            return
        report_by_id = {m["model_id"]: m for m in run_report.get("models", [])}
        success_ids = {mid for mid, m in report_by_id.items() if m.get("status") == "success"}
        # 按名单顺序：仅跳过 success，重跑 failed、running 与未开始的
        to_rerun = []
        new_models = []
        for idx, model_info in enumerate(full_list, 1):
            mid = model_info["model_id"]
            adapt_path = model_info.get("adaptation_path") or mid.replace("/", "_")
            if adapt_path.startswith("adaptations/"):
                adapt_dir = PROJECT_ROOT / adapt_path
            else:
                adapt_dir = ADAPTATIONS_DIR / adapt_path
            m = {"model_id": mid, "adaptation_path": adapt_path}
            existing = report_by_id.get(mid)
            if mid in success_ids:
                new_models.append(existing)
                continue
            if existing:
                to_rerun.append((existing, m))
                new_models.append(existing)
            else:
                new_entry = {
                    "index": idx,
                    "model_id": mid,
                    "adaptation_path": adapt_path,
                    "adapt_dir": str(adapt_dir),
                    "status": "running",
                    "failure_reason": "",
                    "duration_seconds": 0,
                    "return_code": None,
                    "artifacts": {"outputs": [], "metrics": [], "traces": []},
                }
                to_rerun.append((new_entry, m))
                new_models.append(new_entry)
        run_report["models"] = new_models
        run_report["total_models"] = len(full_list)

        skip_ids = set(args.skip or [])
        if skip_ids:
            to_rerun = [(r, m) for r, m in to_rerun if m["model_id"] not in skip_ids]
            for mid in sorted(skip_ids):
                print_warning(f"跳过: {mid}")

        if not to_rerun:
            print_warning("名单中所有模型均已成功或已跳过，无需 resume")
            return
        models = [info for _, info in to_rerun]
        model_report_refs = [r for r, _ in to_rerun]
        # 使用原 run 的 log/json 路径
        log_file = Path(run_report.get("log_file", ""))
        json_file = Path(run_report.get("json_file", ""))
        if not log_file or not json_file:
            print_error("JSON 中缺少 log_file 或 json_file 路径")
            sys.exit(1)
        # 继承原 run 的参数（命令行可覆盖）
        hardware = args.hardware or run_report.get("hardware") or get_default_hardware()
        args.pretrained = run_report.get("pretrained", False) if not args.pretrained else True
        args.max_samples = run_report.get("max_samples", 250) if args.max_samples == 250 else args.max_samples
        total_start_time = time.time()

        def log_write(msg: str):
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(msg)
            except OSError as e:
                print_warning(f"无法写入日志 (磁盘满?): {e}")

        summary = run_report.get("summary", {})
        n_ok = summary.get("success", 0)
        n_fail = summary.get("failed", 0)
        n_rerun = len(models)
        n_fail_rerun = sum(1 for r in model_report_refs if r.get("status") == "failed")
        n_running_rerun = n_rerun - n_fail_rerun  # 待重跑中非 failed 的为中断/运行中

        print_header("Benchmark Runner (Resume)")
        print_info(f"Resume 自:       {json_path}")
        print_info(f"硬件类型:       {BOLD}{hardware.upper()}{NC}")
        print_info(f"已成功:         {n_ok} | 已失败: {n_fail} | 待重跑: {BOLD}{n_rerun}{NC} (含 {n_fail_rerun} 失败 + {n_running_rerun} 中断)")
        print_info(f"预训练权重:     {BOLD}{'是 (Tier2)' if args.pretrained else '否 (Tier1)'}{NC}")
        print_info(f"最大样本数:     {BOLD}{args.max_samples}{NC}")
        print_info(f"开始时间:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        log_write(f"\n{'=' * 60}\nBenchmark Runner (Resume)\n{'=' * 60}\n")
        log_write(f"Resume 自:       {json_path}\n")
        log_write(f"硬件类型:       {hardware.upper()}\n")
        log_write(f"已成功:         {n_ok} | 已失败: {n_fail} | 待重跑: {n_rerun} (含 {n_fail_rerun} 失败 + {n_running_rerun} 中断)\n")
        if skip_ids:
            log_write(f"跳过模型:       {', '.join(sorted(skip_ids))}\n")
        log_write(f"预训练权重:     {'是 (Tier2)' if args.pretrained else '否 (Tier1)'}\n")
        log_write(f"最大样本数:     {args.max_samples}\n")
        log_write(f"开始时间:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        def write_json_report():
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(run_report, f, ensure_ascii=False, indent=2)

        def _recompute_summary():
            all_m = run_report.get("models", [])
            s = sum(1 for x in all_m if x.get("status") == "success")
            fm = []
            for x in all_m:
                if x.get("status") != "failed":
                    continue
                r = x.get("failure_reason", "")
                if "uv sync 失败" in r:
                    suf = "(uv sync 失败)"
                elif "目录不存在" in r:
                    suf = "(目录不存在)"
                elif "accuracy_run.py 不存在" in r or "脚本不存在" in r:
                    suf = "(脚本不存在)"
                elif "缺少产出" in r:
                    suf = "(缺少产出)"
                else:
                    suf = "(运行错误)"
                fm.append(f"{x['model_id']} {suf}")
            run_report["summary"] = {"success": s, "failed": len(fm), "failed_models": fm}

        all_models = run_report.get("models", [])
        _recompute_summary()

        print_info(f"日志文件:       {log_file}")
        print_info(f"JSON 文件:      {json_file}")
        print_info(f"名单总数:       {len(full_list)}")
        log_write(f"日志文件:       {log_file}\n")
        log_write(f"JSON 文件:      {json_file}\n")
        log_write(f"名单总数:       {len(full_list)}\n")
        print_section("待重跑模型（按原名单顺序）")
        log_write(f"\n--------------------------------------------------\n待重跑模型（按原名单顺序）\n--------------------------------------------------\n")
        for i, (_, info) in enumerate(to_rerun, 1):
            line = f"  {i}. {info['model_id']}\n"
            print(line, end="")
            log_write(line)
        log_write("\n")

        # 执行重跑
        for i, (model_report, m) in enumerate(zip(model_report_refs, models), 1):
            model_id = m["model_id"]
            adapt_path = m.get("adaptation_path") or model_id.replace("/", "_")
            idx = model_report.get("index", i)
            total = run_report.get("total_models", len(all_models))

            if adapt_path.startswith("adaptations/"):
                adapt_dir = PROJECT_ROOT / adapt_path
            else:
                adapt_dir = ADAPTATIONS_DIR / adapt_path

            model_report["status"] = "running"
            model_report["failure_reason"] = ""
            model_report["duration_seconds"] = 0
            model_report["return_code"] = None
            model_report["artifacts"] = {"outputs": [], "metrics": [], "traces": []}
            write_json_report()

            print_header(f"[Resume {i}/{len(models)}] {model_id} (原序号 {idx}/{total})")
            print_info(f"适配目录: {adapt_dir}")

            if not adapt_dir.is_dir():
                print_error(f"适配目录不存在: {adapt_dir}")
                log_write(f"\n[Resume {i}/{len(models)}] {model_id} (原序号 {idx}/{total}) — SKIP: 适配目录不存在 {adapt_dir}\n")
                model_report["status"] = "failed"
                model_report["failure_reason"] = f"适配目录不存在: {adapt_dir}"
                _recompute_summary()
                write_json_report()
                continue

            accuracy_script = adapt_dir / "accuracy_run.py"
            if not accuracy_script.is_file():
                print_error(f"accuracy_run.py 不存在")
                log_write(f"\n[Resume {i}/{len(models)}] {model_id} (原序号 {idx}/{total}) — SKIP: accuracy_run.py 不存在\n")
                model_report["status"] = "failed"
                model_report["failure_reason"] = "accuracy_run.py 不存在"
                _recompute_summary()
                write_json_report()
                continue

            # uv sync
            print_section("同步依赖")
            extra = "cuda" if hardware == "cuda" else "ascend"
            sync_cmd = ["uv", "sync", f"--extra={extra}"]
            print_info(f"执行: uv sync --extra={extra}")
            sync_start = time.time()
            result = subprocess.run(
                sync_cmd,
                cwd=str(adapt_dir),
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "VIRTUAL_ENV": "",
                    "UV_LINK_MODE": "symlink",
                    "UV_PREVIEW_FEATURES": "extra-build-dependencies",
                },
            )
            sync_duration = int(time.time() - sync_start)
            if result.returncode != 0:
                print_error(f"uv sync 失败 (耗时 {sync_duration}s)")
                print(result.stderr)
                log_write(f"\n[Resume {i}/{len(models)}] {model_id} (原序号 {idx}/{total}) — SKIP: uv sync 失败\n{result.stderr}\n")
                model_report["status"] = "failed"
                model_report["duration_seconds"] = sync_duration
                model_report["return_code"] = result.returncode
                model_report["failure_reason"] = f"uv sync 失败: {(result.stderr or '').strip()[-1000:]}"
                _recompute_summary()
                write_json_report()
                continue
            print_success(f"依赖同步完成 (耗时 {sync_duration}s)")

            # benchmark
            bench_args = []
            if args.pretrained:
                bench_args.append("--use-pretrained")
            if args.max_samples:
                bench_args.extend(["--max-samples", str(args.max_samples)])
            print_section("运行 Benchmark")
            cmd_str = f"uv run --extra={extra} python accuracy_run.py {' '.join(bench_args)}".strip()
            print_info(f"执行: {cmd_str}")
            start_time = time.time()
            run_cmd = ["uv", "run", f"--extra={extra}", "python", "accuracy_run.py"] + bench_args
            bench_env = {
                **os.environ,
                "VIRTUAL_ENV": "",
                "PYTHONUNBUFFERED": "1",
                "TRANSFORMERS_OFFLINE": "0",
                "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
                "HF_HUB_DOWNLOAD_TIMEOUT": "120",
                "HF_HUB_ETAG_TIMEOUT": "60",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",  # 确定性算法，修复 DeBERTa 等 CuBLAS 警告
            }
            log_write(f"\n{'=' * 60}\n[Resume {i}/{len(models)}] {model_id} (原序号 {idx}/{total}) — 开始运行\n{'=' * 60}\n")
            proc = subprocess.Popen(
                run_cmd,
                cwd=str(adapt_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=bench_env,
                bufsize=1,
            )
            out_lines = []
            if proc.stdout is not None:
                for line in proc.stdout:
                    line = line if line.endswith("\n") else line + "\n"
                    print(line, end="")
                    out_lines.append(line)
                    log_write(line)  # 实时写入日志，避免中途退出时无记录
            returncode = proc.wait()
            full_stdout = "".join(out_lines)
            duration = int(time.time() - start_time)
            model_report["duration_seconds"] = duration
            model_report["return_code"] = returncode
            log_write(f"\n[Resume {i}/{len(models)}] {model_id} (原序号 {idx}/{total}) — Duration: {duration}s, returncode={returncode}\n")

            if returncode == 0:
                outputs = list(adapt_dir.glob("outputs_*.pt"))
                metrics = list(adapt_dir.glob("benchmark_metrics_*.json"))
                traces = list(adapt_dir.glob("trace_*.json"))
                model_report["artifacts"] = {
                    "outputs": [f.name for f in sorted(outputs, key=lambda p: p.name)],
                    "metrics": [f.name for f in sorted(metrics, key=lambda p: p.name)],
                    "traces": [f.name for f in sorted(traces, key=lambda p: p.name)],
                }
                if outputs and metrics and traces:
                    print_success(f"Benchmark 完成 (耗时 {fmt_duration(duration)})")
                    log_write(f"✓ Benchmark 完成 (耗时 {fmt_duration(duration)})\n")
                    model_report["status"] = "success"
                else:
                    print_error("缺少产出文件")
                    model_report["status"] = "failed"
                    model_report["failure_reason"] = f"缺少产出: outputs={len(outputs)}, metrics={len(metrics)}, traces={len(traces)}"
            else:
                print_error(f"Benchmark 失败 (耗时 {fmt_duration(duration)})")
                log_write(f"✗ Benchmark 失败 (耗时 {fmt_duration(duration)})\n")
                if full_stdout.strip():
                    print(full_stdout[-2000:] if len(full_stdout) > 2000 else full_stdout)
                model_report["status"] = "failed"
                model_report["failure_reason"] = full_stdout.strip()[-2000:] if full_stdout.strip() else "benchmark 运行错误"

            _recompute_summary()
            write_json_report()

        total_duration = int(time.time() - total_start_time)
        _recompute_summary()
        run_report["summary"]["total_duration_seconds"] = total_duration
        run_report["finished_at"] = datetime.now().isoformat()
        write_json_report()

        summary = run_report["summary"]
        success = summary["success"]
        failed = summary["failed"]
        failed_models = summary.get("failed_models", [])

        print_header("Resume 汇总")
        print_info(f"本次耗时:   {fmt_duration(total_duration)}")
        print_info(f"总成功:     {success}")
        print_info(f"总失败:     {failed}")
        if failed_models:
            print_section("失败模型")
            for m in failed_models:
                print_error(m)
        log_write(f"\n{'=' * 60}\nResume 汇总\n本次耗时: {fmt_duration(total_duration)} | 总成功: {success} | 总失败: {failed}\n")
        print_info(f"JSON 文件: {json_file}")
        if failed > 0:
            sys.exit(1)
        return

    if getattr(args, "from_json", None):
        # 从 JSON 报告中只运行 status=failed 的模型
        if args.model:
            print_error("--from-json 与 --model 互斥，请只指定其一")
            sys.exit(1)
        json_path = Path(args.from_json)
        if not json_path.is_file():
            print_error(f"JSON 文件不存在: {json_path}")
            sys.exit(1)
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        failed_entries = [m for m in data.get("models", []) if m.get("status") == "failed"]
        models = []
        for m in failed_entries:
            mid = m.get("model_id")
            adapt_path = m.get("adaptation_path")
            if not adapt_path and m.get("adapt_dir"):
                # 从绝对路径推导 adaptation_path
                adapt_dir = Path(m["adapt_dir"])
                try:
                    adapt_path = str(adapt_dir.relative_to(PROJECT_ROOT)).replace("\\", "/")
                except ValueError:
                    adapt_path = adapt_dir.name
                    if not adapt_path.startswith("adaptations/"):
                        adapt_path = f"adaptations/{adapt_path}"
            if not adapt_path and mid:
                # 使用与 adaptation_utils 一致的规则生成路径
                safe = mid.replace("/", "_").replace("-", "_").lower()
                import re

                safe = re.sub(r"[^a-z0-9_]", "_", safe)
                safe = re.sub(r"_+", "_", safe).strip("_")
                adapt_path = f"adaptations/{safe}" if safe else ""
            models.append({"model_id": mid, "adaptation_path": adapt_path})
        if not models:
            print_warning("JSON 中没有 status=failed 的模型")
            return
        print_info(f"从 JSON 加载 {len(models)} 个失败模型，将仅重跑这些模型")
    elif args.model:
        # 指定单个模型：必须 adaptation_status 与 benchmark_status 均为 completed
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT model_id, adaptation_path, adaptation_status, benchmark_status FROM models WHERE model_id = ?",
            (args.model,),
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            print_error(f"模型 {args.model} 不存在于 board.db")
            sys.exit(1)
        s = (row["adaptation_status"] or "").strip().lower()
        bs = (row["benchmark_status"] or "").strip().lower()
        if s != "completed" or bs != "completed":
            print_error(
                f"模型 {args.model} 需要适配状态(adaptation_status)与 benchmark 状态(benchmark_status)均为 completed 才能运行，"
                f"当前 adaptation_status={row['adaptation_status']!r}, benchmark_status={row['benchmark_status']!r}"
            )
            sys.exit(1)
        models = [{"model_id": row["model_id"], "adaptation_path": row["adaptation_path"]}]
    else:
        # 运行所有 adaptation_status 与 benchmark_status 均为 completed 的模型
        models = get_models(status="completed", require_both_completed=True)
        if not models:
            print_warning("没有找到 adaptation_status 与 benchmark_status 均为 completed 的模型")
            return

    # 确定硬件
    hardware = args.hardware or get_default_hardware()
    total_start_time = time.time()

    print_header("Benchmark Runner")
    print_info(f"硬件类型:       {BOLD}{hardware.upper()}{NC}")
    print_info(f"模型数量:       {BOLD}{len(models)}{NC}")
    print_info(f"预训练权重:     {BOLD}{'是 (Tier2)' if args.pretrained else '否 (Tier1)'}{NC}")
    print_info(f"最大样本数:     {BOLD}{args.max_samples}{NC}")
    print_info(f"开始时间:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 日志文件：run 开始时即创建并写入头部，保存到 logs/ 目录
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"benchmark_runs_{timestamp}.log"
    json_file = log_dir / f"benchmark_runs_{timestamp}.json"

    def log_write(msg: str):
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(msg)
        except OSError as e:
            print_warning(f"无法写入日志 (磁盘满?): {e}")

    run_report = {
        "started_at": datetime.now().isoformat(),
        "hardware": hardware,
        "pretrained": bool(args.pretrained),
        "max_samples": args.max_samples,
        "total_models": len(models),
        "log_file": str(log_file),
        "json_file": str(json_file),
        "models": [],
        "summary": {
            "success": 0,
            "failed": 0,
            "failed_models": [],
        },
    }

    def write_json_report():
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(run_report, f, ensure_ascii=False, indent=2)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Benchmark Run Log — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"硬件: {hardware.upper()} | 预训练: {'是' if args.pretrained else '否'} | max_samples: {args.max_samples}\n")
        f.write(f"模型数: {len(models)}\n")
        f.write("\n".join(f"  {i}. {m['model_id']}" for i, m in enumerate(models, 1)))
        f.write("\n\n")
    write_json_report()

    print_info(f"日志文件:       {log_file}")
    print_info(f"JSON 文件:      {json_file}")

    # 显示待运行模型
    print_section("待运行模型")
    for i, m in enumerate(models, 1):
        print(f"  {i}. {m['model_id']}")

    success = 0
    failed = 0
    failed_models = []

    for i, m in enumerate(models, 1):
        model_id = m["model_id"]
        adapt_path = m.get("adaptation_path") or model_id.replace("/", "_")

        # 构建适配目录路径
        if adapt_path.startswith("adaptations/"):
            adapt_dir = PROJECT_ROOT / adapt_path
        else:
            adapt_dir = ADAPTATIONS_DIR / adapt_path

        model_report = {
            "index": i,
            "model_id": model_id,
            "adaptation_path": adapt_path,
            "adapt_dir": str(adapt_dir),
            "status": "running",
            "failure_reason": "",
            "duration_seconds": 0,
            "return_code": None,
            "artifacts": {
                "outputs": [],
                "metrics": [],
                "traces": [],
            },
        }
        run_report["models"].append(model_report)
        write_json_report()

        print_header(f"[{i}/{len(models)}] {model_id}")
        print_info(f"适配目录: {adapt_dir}")

        # 检查目录和脚本
        if not adapt_dir.is_dir():
            print_error(f"适配目录不存在: {adapt_dir}")
            log_write(f"\n[{i}/{len(models)}] {model_id} — SKIP: 适配目录不存在 {adapt_dir}\n")
            failed += 1
            failed_models.append(f"{model_id} (目录不存在)")
            model_report["status"] = "failed"
            model_report["failure_reason"] = f"适配目录不存在: {adapt_dir}"
            run_report["summary"]["failed"] = failed
            run_report["summary"]["failed_models"] = failed_models
            write_json_report()
            continue

        accuracy_script = adapt_dir / "accuracy_run.py"
        if not accuracy_script.is_file():
            print_error(f"accuracy_run.py 不存在")
            log_write(f"\n[{i}/{len(models)}] {model_id} — SKIP: accuracy_run.py 不存在\n")
            failed += 1
            failed_models.append(f"{model_id} (脚本不存在)")
            model_report["status"] = "failed"
            model_report["failure_reason"] = "accuracy_run.py 不存在"
            run_report["summary"]["failed"] = failed
            run_report["summary"]["failed_models"] = failed_models
            write_json_report()
            continue

        # 同步依赖
        print_section("同步依赖")
        extra = "cuda" if hardware == "cuda" else "ascend"
        sync_cmd = ["uv", "sync", f"--extra={extra}"]
        print_info(f"执行: uv sync --extra={extra}")

        sync_start = time.time()
        result = subprocess.run(
            sync_cmd,
            cwd=str(adapt_dir),
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "VIRTUAL_ENV": "",
                "UV_LINK_MODE": "symlink",
                "UV_PREVIEW_FEATURES": "extra-build-dependencies",
            },
        )
        sync_duration = int(time.time() - sync_start)

        if result.returncode != 0:
            print_error(f"uv sync 失败 (耗时 {sync_duration}s)")
            print(result.stderr)
            log_write(f"\n[{i}/{len(models)}] {model_id} — SKIP: uv sync 失败\n{result.stderr}\n")
            failed += 1
            failed_models.append(f"{model_id} (uv sync 失败)")
            model_report["status"] = "failed"
            model_report["duration_seconds"] = sync_duration
            model_report["return_code"] = result.returncode
            model_report["failure_reason"] = f"uv sync 失败: {(result.stderr or '').strip()[-1000:]}"
            run_report["summary"]["failed"] = failed
            run_report["summary"]["failed_models"] = failed_models
            write_json_report()
            continue
        else:
            print_success(f"依赖同步完成 (耗时 {sync_duration}s)")

        # 构建 benchmark 命令
        bench_args = []
        if args.pretrained:
            bench_args.append("--use-pretrained")
        if args.max_samples:
            bench_args.extend(["--max-samples", str(args.max_samples)])

        # 运行 benchmark（实时输出到终端，同时写入日志）
        print_section("运行 Benchmark")
        cmd_str = f"uv run --extra={extra} python accuracy_run.py {' '.join(bench_args)}".strip()
        print_info(f"执行: {cmd_str}")

        start_time = time.time()
        run_cmd = ["uv", "run", f"--extra={extra}", "python", "accuracy_run.py"] + bench_args
        # accuracy_run 始终需从 HF 拉取 tokenizer/config，不能设 TRANSFORMERS_OFFLINE=1
        env = {
            **os.environ,
            "VIRTUAL_ENV": "",
            "PYTHONUNBUFFERED": "1",
            "TRANSFORMERS_OFFLINE": "0",
            "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),  # 国内环境使用镜像
            "HF_HUB_DOWNLOAD_TIMEOUT": "120",  # hf-mirror 可能较慢，延长超时
            "HF_HUB_ETAG_TIMEOUT": "60",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",  # 确定性算法，修复 DeBERTa 等 CuBLAS 警告
        }
        log_write(f"\n{'=' * 60}\n[{i}/{len(models)}] {model_id} — 开始运行\n{'=' * 60}\n")
        proc = subprocess.Popen(
            run_cmd,
            cwd=str(adapt_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        out_lines: list[str] = []
        if proc.stdout is not None:
            for line in proc.stdout:
                line = line if line.endswith("\n") else line + "\n"
                print(line, end="")
                out_lines.append(line)
                log_write(line)  # 实时写入日志，避免中途退出时无记录
        returncode = proc.wait()
        full_stdout = "".join(out_lines)

        duration = int(time.time() - start_time)
        model_report["duration_seconds"] = duration
        model_report["return_code"] = returncode

        log_write(f"\n[{i}/{len(models)}] {model_id} — Duration: {duration}s, returncode={returncode}\n")

        if returncode == 0:
            # 检查产出文件
            outputs = list(adapt_dir.glob("outputs_*.pt"))
            metrics = list(adapt_dir.glob("benchmark_metrics_*.json"))
            traces = list(adapt_dir.glob("trace_*.json"))
            model_report["artifacts"] = {
                "outputs": [f.name for f in sorted(outputs, key=lambda p: p.name)],
                "metrics": [f.name for f in sorted(metrics, key=lambda p: p.name)],
                "traces": [f.name for f in sorted(traces, key=lambda p: p.name)],
            }

            if outputs and metrics and traces:
                print_success(f"Benchmark 完成 (耗时 {fmt_duration(duration)})")
                log_write(f"✓ Benchmark 完成 (耗时 {fmt_duration(duration)})\n")

                print_section("产出文件")
                print_info(f"outputs ({len(outputs)}):")
                log_write("\n产出文件\n")
                log_write(f"outputs ({len(outputs)}):\n")
                for f in sorted(outputs, key=lambda p: p.name):
                    print_info(f"  {f.name}")
                    log_write(f"  {f.name}\n")
                print_info(f"metrics ({len(metrics)}):")
                log_write(f"metrics ({len(metrics)}):\n")
                for f in sorted(metrics, key=lambda p: p.name):
                    print_info(f"  {f.name}")
                    log_write(f"  {f.name}\n")
                print_info(f"traces ({len(traces)}):")
                log_write(f"traces ({len(traces)}):\n")
                for f in sorted(traces, key=lambda p: p.name):
                    print_info(f"  {f.name}")
                    log_write(f"  {f.name}\n")

                # 显示 metrics 摘要
                metrics_file = metrics[0]
                try:
                    with open(metrics_file) as f:
                        m_data = json.load(f)

                    print_section("性能摘要")
                    print_info(f"设备:      {m_data.get('device', 'N/A')}")
                    print_info(f"延迟:      {m_data.get('latency_s', 0):.4f}s")
                    print_info(f"峰值内存:  {m_data.get('peak_memory_mb', 0):.2f}MB")
                    print_info(f"输出类型:  {m_data.get('output_type', 'N/A')}")
                    log_write(f"\n性能摘要\n设备:      {m_data.get('device', 'N/A')}\n延迟:      {m_data.get('latency_s', 0):.4f}s\n峰值内存:  {m_data.get('peak_memory_mb', 0):.2f}MB\n输出类型:  {m_data.get('output_type', 'N/A')}\n")
                except Exception:
                    pass

                success += 1
                model_report["status"] = "success"
                run_report["summary"]["success"] = success
                run_report["summary"]["failed"] = failed
                run_report["summary"]["failed_models"] = failed_models
                write_json_report()
            else:
                print_error(f"缺少产出文件")
                print_info(f"outputs={len(outputs)}, metrics={len(metrics)}, traces={len(traces)}")
                log_write(f"✗ 缺少产出文件\noutputs={len(outputs)}, metrics={len(metrics)}, traces={len(traces)}\n")
                failed += 1
                failed_models.append(f"{model_id} (缺少产出)")
                model_report["status"] = "failed"
                model_report["failure_reason"] = f"缺少产出文件: outputs={len(outputs)}, metrics={len(metrics)}, traces={len(traces)}"
                run_report["summary"]["success"] = success
                run_report["summary"]["failed"] = failed
                run_report["summary"]["failed_models"] = failed_models
                write_json_report()
        else:
            print_error(f"Benchmark 失败 (耗时 {fmt_duration(duration)})")
            log_write(f"✗ Benchmark 失败 (耗时 {fmt_duration(duration)})\n")
            if full_stdout.strip():
                print()
                print("输出:")
                print(full_stdout[-2000:] if len(full_stdout) > 2000 else full_stdout)
            failed += 1
            failed_models.append(f"{model_id} (运行错误)")
            model_report["status"] = "failed"
            model_report["failure_reason"] = full_stdout.strip()[-2000:] if full_stdout.strip() else "benchmark 运行错误"
            run_report["summary"]["success"] = success
            run_report["summary"]["failed"] = failed
            run_report["summary"]["failed_models"] = failed_models
            write_json_report()

    # 汇总
    total_duration = int(time.time() - total_start_time)

    print_header("运行汇总")
    print_info(f"总耗时:     {fmt_duration(total_duration)}")
    print_info(f"模型总数:   {len(models)}")
    print_success(f"成功:       {success}")
    if failed > 0:
        print_error(f"失败:       {failed}")

    if failed_models:
        print_section("失败模型")
        for m in failed_models:
            print_error(m)

    # 汇总写入日志
    log_write(f"\n{'=' * 60}\n运行汇总\n总耗时: {fmt_duration(total_duration)} | 成功: {success} | 失败: {failed}\n")
    if failed_models:
        log_write("失败模型:\n")
        for m in failed_models:
            log_write(f"  - {m}\n")

    run_report["finished_at"] = datetime.now().isoformat()
    run_report["summary"] = {
        "success": success,
        "failed": failed,
        "failed_models": failed_models,
        "total_duration_seconds": total_duration,
    }
    write_json_report()

    print()
    print_info(f"日志文件: {log_file}")
    print_info(f"JSON 文件: {json_file}")
    print()

    if failed > 0:
        sys.exit(1)


def cmd_artifacts(args):
    """列出产出文件。"""
    model_id = args.model_id

    if model_id:
        # 获取单个模型的适配路径
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT adaptation_path FROM models WHERE model_id = ?",
            (model_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            print_error(f"模型 {model_id} 不存在")
            sys.exit(1)

        adapt_path = row["adaptation_path"] or model_id.replace("/", "_")
        if adapt_path.startswith("adaptations/"):
            adapt_dir = PROJECT_ROOT / adapt_path
        else:
            adapt_dir = ADAPTATIONS_DIR / adapt_path

        dirs = [(model_id, adapt_dir)]
    else:
        # 列出所有 benchmark_status=completed 的模型
        models = get_models(status="completed")
        dirs = []
        for m in models:
            adapt_path = m.get("adaptation_path") or m["model_id"].replace("/", "_")
            if adapt_path.startswith("adaptations/"):
                adapt_dir = PROJECT_ROOT / adapt_path
            else:
                adapt_dir = ADAPTATIONS_DIR / adapt_path
            dirs.append((m["model_id"], adapt_dir))

    print_header("Benchmark 产出文件")

    total_files = 0
    total_size = 0

    for name, adapt_dir in dirs:
        print(f"{BOLD}{CYAN}[{name}]{NC}")
        if not adapt_dir.is_dir():
            print_warning("  目录不存在\n")
            continue

        jsons = sorted(f for f in adapt_dir.glob("*.json") if f.name != ".status.json")
        pts = sorted(adapt_dir.glob("*.pt"))

        dir_size = 0
        dir_files = 0

        if jsons:
            print(f"  {BLUE}.json{NC}:")
            for f in jsons:
                size = f.stat().st_size
                dir_size += size
                dir_files += 1
                print(f"    {f.name}  {fmt_size(size)}")
        if pts:
            print(f"  {BLUE}.pt{NC}:")
            for f in pts:
                size = f.stat().st_size
                dir_size += size
                dir_files += 1
                print(f"    {f.name}  {fmt_size(size)}")
        if not jsons and not pts:
            print_warning("  (无 .json / .pt 文件)")

        total_files += dir_files
        total_size += dir_size

        if dir_files > 0:
            print_info(f"共 {dir_files} 文件, {fmt_size(dir_size)}")
        print()

    if total_files > 0:
        print(f"\n{BOLD}总计: {total_files} 文件, {fmt_size(total_size)}{NC}\n")


def cmd_pack(args):
    """打包产出文件。"""
    # 确定要打包的模型
    if args.model:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT model_id, adaptation_path FROM models WHERE model_id = ?",
            (args.model,),
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            print_error(f"模型 {args.model} 不存在")
            sys.exit(1)
        models = [{"model_id": row["model_id"], "adaptation_path": row["adaptation_path"]}]
    else:
        models = get_models(status="completed")

    if not models:
        print_warning("没有可打包的模型")
        return

    # 输出文件
    output_file = Path(args.output) if args.output else PROJECT_ROOT / f"benchmark_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

    print_header("打包 Benchmark 产出")
    print_info(f"输出文件: {output_file}")
    print_info(f"模型数量: {len(models)}")

    # 收集文件
    files_to_pack = []
    for m in models:
        adapt_path = m.get("adaptation_path") or m["model_id"].replace("/", "_")
        if adapt_path.startswith("adaptations/"):
            adapt_dir = PROJECT_ROOT / adapt_path
        else:
            adapt_dir = ADAPTATIONS_DIR / adapt_path

        if not adapt_dir.is_dir():
            continue

        dir_name = adapt_dir.name
        for pattern in ["outputs_*.pt", "benchmark_metrics_*.json", "trace_*.json"]:
            for f in adapt_dir.glob(pattern):
                files_to_pack.append((f, f"{dir_name}/{f.name}"))

    if not files_to_pack:
        print_warning("没有找到可打包的文件")
        return

    print_section("打包文件")

    # 创建 zip
    with zipfile.ZipFile(output_file, "w", zipfile.ZIP_DEFLATED) as zf:
        # 写入 manifest
        manifest = {
            "created_at": datetime.now().isoformat(),
            "models": [m["model_id"] for m in models],
            "files_count": len(files_to_pack),
        }
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))

        # 写入文件
        for src_path, arcname in files_to_pack:
            zf.write(src_path, arcname)
            print_info(f"{arcname}  ({fmt_size(src_path.stat().st_size)})")

    total_size = sum(f[0].stat().st_size for f in files_to_pack)

    print()
    print_success(f"打包完成")
    print_info(f"输出文件: {output_file}")
    print_info(f"文件数量: {len(files_to_pack)}")
    print_info(f"总大小:   {fmt_size(total_size)}")
    print()


def cmd_unpack(args):
    """解包还原产出文件。"""
    input_file = Path(args.input)
    if not input_file.is_file():
        print_error(f"文件不存在: {input_file}")
        sys.exit(1)

    print_header("解包 Benchmark 产出")

    with zipfile.ZipFile(input_file, "r") as zf:
        # 读取 manifest
        try:
            manifest = json.loads(zf.read("manifest.json"))
            print_info(f"创建时间: {manifest.get('created_at', 'N/A')}")
            print_info(f"模型列表: {manifest.get('models', [])}")
            print_info(f"文件数量: {manifest.get('files_count', 'N/A')}")
        except KeyError:
            print_warning("缺少 manifest.json")
            manifest = {}

        print_section(f"解包到: {ADAPTATIONS_DIR}")

        file_count = 0
        for info in zf.infolist():
            if info.filename == "manifest.json":
                continue

            # 提取到 adaptations 目录
            target_path = ADAPTATIONS_DIR / info.filename
            target_path.parent.mkdir(parents=True, exist_ok=True)

            with zf.open(info) as src, open(target_path, "wb") as dst:
                dst.write(src.read())

            print_success(f"{info.filename}")
            file_count += 1

    print()
    print_success(f"解包完成: {file_count} 文件")
    print()


def cmd_clean(args):
    """一键清空产出文件。"""
    # 确定要清理的模型
    if args.model:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT model_id, adaptation_path FROM models WHERE model_id = ?",
            (args.model,),
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            print_error(f"模型 {args.model} 不存在")
            sys.exit(1)
        models = [{"model_id": row["model_id"], "adaptation_path": row["adaptation_path"]}]
    else:
        models = get_models(status="completed")

    if not models:
        print_warning("没有可清理的模型")
        return

    print_header("清空 Benchmark 产出")
    print_info(f"模型数量: {len(models)}")

    # 要删除的文件模式
    patterns = ["outputs_*.pt", "benchmark_metrics_*.json", "trace_*.json"]

    total_files = 0
    total_size = 0

    print_section("删除文件")

    for m in models:
        adapt_path = m.get("adaptation_path") or m["model_id"].replace("/", "_")
        if adapt_path.startswith("adaptations/"):
            adapt_dir = PROJECT_ROOT / adapt_path
        else:
            adapt_dir = ADAPTATIONS_DIR / adapt_path

        if not adapt_dir.is_dir():
            continue

        dir_name = adapt_dir.name
        dir_files = 0
        dir_size = 0

        for pattern in patterns:
            for f in adapt_dir.glob(pattern):
                file_size = f.stat().st_size
                dir_size += file_size
                dir_files += 1
                f.unlink()
                print_info(f"删除: {dir_name}/{f.name}")

        if dir_files > 0:
            total_files += dir_files
            total_size += dir_size
            print_success(f"[{dir_name}] {dir_files} 文件, {fmt_size(dir_size)}")
            print()

    if total_files == 0:
        print_warning("没有找到可清理的文件")
    else:
        print_success(f"清理完成: {total_files} 文件, {fmt_size(total_size)}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Manager - 模型 benchmark 运行与产出管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 列出 benchmark_status=completed 的模型（默认）
  uv run python benchmark/scripts/benchmark_manager.py list

  # 列出指定状态的模型
  uv run python benchmark/scripts/benchmark_manager.py list --status in_progress

  # 运行所有 completed 模型的 benchmark (使用 CUDA)
  uv run python benchmark/scripts/benchmark_manager.py run --hardware cuda

  # 运行指定模型 (使用 NPU 和预训练权重)
  uv run python benchmark/scripts/benchmark_manager.py run --model Qwen/Qwen2-0.5B --hardware npu --pretrained

  # 从中断的 run 恢复（跳过已成功，重跑 failed/running）
  uv run python benchmark/scripts/benchmark_manager.py run --resume logs/benchmark_runs_20260312_090939.json

  # 列出产出文件
  uv run python benchmark/scripts/benchmark_manager.py artifacts
  uv run python benchmark/scripts/benchmark_manager.py artifacts Qwen/Qwen2-0.5B

  # 打包所有产出文件
  uv run python benchmark/scripts/benchmark_manager.py pack

  # 打包指定模型的产出
  uv run python benchmark/scripts/benchmark_manager.py pack --model Qwen/Qwen2-0.5B

  # 解包还原
  uv run python benchmark/scripts/benchmark_manager.py unpack --input benchmark_outputs_20260227.zip

  # 一键清空所有 completed 模型的产出文件
  uv run python benchmark/scripts/benchmark_manager.py clean

  # 清空指定模型的产出文件
  uv run python benchmark/scripts/benchmark_manager.py clean --model Qwen/Qwen2-0.5B
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list 命令
    list_parser = subparsers.add_parser("list", help="列出模型")
    list_parser.add_argument("--status", help="过滤 benchmark_status (completed/pending/in_progress)")
    list_parser.set_defaults(func=cmd_list)

    # run 命令
    run_parser = subparsers.add_parser("run", help="运行 benchmark")
    run_parser.add_argument("--model", help="指定模型 ID，不指定则运行所有 completed 的模型")
    run_parser.add_argument(
        "--from-json",
        help="从指定 JSON 报告中只运行 status=failed 的模型（与 --model/--resume 互斥）",
    )
    run_parser.add_argument(
        "--resume",
        help="从指定 JSON 恢复：从 board.db 取完整名单，跳过 success/failed，按原顺序接着跑 running 与未开始的，续写同一 log/json",
    )
    run_parser.add_argument(
        "--skip",
        action="append",
        default=[],
        metavar="MODEL_ID",
        help="Resume 时跳过的模型 ID，可多次指定。例: --skip model1 --skip model2",
    )
    run_parser.add_argument("--hardware", choices=["cuda", "npu"], help="硬件类型，默认 npu")
    run_parser.add_argument("--pretrained", action="store_true", help="使用预训练权重 (Tier2)")
    run_parser.add_argument("--max-samples", type=int, default=250, help="最大样本数，默认 250")
    run_parser.set_defaults(func=cmd_run)

    # artifacts 命令
    artifacts_parser = subparsers.add_parser("artifacts", help="列出产出文件")
    artifacts_parser.add_argument("model_id", nargs="?", help="模型 ID，不指定则列出所有")
    artifacts_parser.set_defaults(func=cmd_artifacts)

    # pack 命令
    pack_parser = subparsers.add_parser("pack", help="打包产出文件")
    pack_parser.add_argument("--output", help="输出 zip 文件路径")
    pack_parser.add_argument("--model", help="指定模型 ID，不指定则打包所有 completed 的模型")
    pack_parser.set_defaults(func=cmd_pack)

    # unpack 命令
    unpack_parser = subparsers.add_parser("unpack", help="解包还原产出文件")
    unpack_parser.add_argument("--input", required=True, help="输入 zip 文件路径")
    unpack_parser.set_defaults(func=cmd_unpack)

    # clean 命令
    clean_parser = subparsers.add_parser("clean", help="一键清空产出文件")
    clean_parser.add_argument("--model", help="指定模型 ID，不指定则清空所有 completed 的模型")
    clean_parser.set_defaults(func=cmd_clean)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
