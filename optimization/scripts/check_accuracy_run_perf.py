#!/usr/bin/env python3
from __future__ import annotations

"""
检查 accuracy_run_perf.py 核心结构。board_ops 在 optimization_status=completed 前强制校验，CI 亦会执行。

校验规则（与 model-files-override SKILL、npu-optimizer 产出一致）：
- 必须定义 PERF_SUFFIX = "_perf"
- 必须体现优化承载路径：
  - model_files / adaptation-local patch 加载（MODEL_PATH 或 model_files 路径）
  - 或显式包含 warmup / TASK_QUEUE_ENABLE 等运行时优化逻辑
  - 自定义仓库直接改源码路线通常也应保留 warmup / TASK_QUEUE_ENABLE，避免 perf 测量契约缺失
- 必须支持 run 与 compare 子命令
- 产出路径必须含 _perf 后缀
- 产出路径必须含 mode 后缀，显式区分 pretrained/config
- completed optimization 对应的 baseline/perf benchmark_metrics 工件必须满足 num_samples >= 50（db-only/board_ops 强制）
- 禁止 `--use-pretrained` 加载失败后 silent fallback 到 config 并继续产出结果

用法:
    uv run python optimization/scripts/check_accuracy_run_perf.py              # 检查所有 accuracy_run_perf.py
    uv run python optimization/scripts/check_accuracy_run_perf.py --adapt xxx  # 检查指定 adaptation，并模拟 optimization completed gate
    uv run python optimization/scripts/check_accuracy_run_perf.py --db-only     # 仅检查 db 中 optimization_status=completed 的 adaptation（CI 用）
"""

import importlib.util
import json
import re
import sqlite3
import sys
from pathlib import Path

_MIN_COMPLETED_OPTIMIZATION_SAMPLES = 50


def get_optimization_adapt_names(project_root: Path) -> list[str]:
    """从 board.db 获取 optimization_status=completed 的 adaptation 目录名列表。"""
    db_path = project_root / "board.db"
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        "SELECT adaptation_path FROM models WHERE optimization_status = ? AND adaptation_path IS NOT NULL AND adaptation_path != ''",
        ("completed",),
    )
    paths = [r[0] for r in cur.fetchall()]
    conn.close()
    names = []
    for p in paths:
        p = p.strip().strip("/")
        if p.startswith("adaptations/"):
            names.append(p.replace("adaptations/", ""))
        else:
            names.append(p)
    return sorted(set(names))


def _metric_close(expected: float, actual: float) -> bool:
    abs_tol = max(1e-3, abs(expected) * 0.02)
    return abs(expected - actual) <= abs_tol


def _validate_metric_num_samples(metric: dict, metric_file: Path, label: str) -> str | None:
    num_samples = metric.get("num_samples")
    if not isinstance(num_samples, (int, float)) or isinstance(num_samples, bool):
        return f"{label} 工件 {metric_file.name} 缺少数值型 num_samples；optimization completed 前必须至少测试 {_MIN_COMPLETED_OPTIMIZATION_SAMPLES} 个样本"
    if float(num_samples) < _MIN_COMPLETED_OPTIMIZATION_SAMPLES:
        return f"{label} 工件 {metric_file.name} 的 num_samples={num_samples}，optimization completed 前必须至少测试 {_MIN_COMPLETED_OPTIMIZATION_SAMPLES} 个样本"
    return None


def _load_board_ops_module(project_root: Path):
    board_ops_path = project_root / "scripts" / "board_ops.py"
    spec = importlib.util.spec_from_file_location("board_ops_validation", str(board_ops_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def check_completed_optimization_metric_artifacts(project_root: Path) -> list[str]:
    """db-only/CI 模式下，校验 optimization completed 的 baseline/perf 工件样本数。"""
    board_ops = _load_board_ops_module(project_root)
    db_path = project_root / "board.db"
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        "SELECT model_id, adaptation_path, optimization_notes FROM models WHERE optimization_status = ?",
        ("completed",),
    )
    rows = cur.fetchall()
    conn.close()

    errors: list[str] = []
    for model_id, adaptation_path, notes_str in rows:
        if not adaptation_path or not notes_str:
            continue
        try:
            data = json.loads(notes_str)
        except json.JSONDecodeError as e:
            errors.append(f"{model_id}: optimization_notes 无法解析，跳过样本数校验: {e}")
            continue
        best_result = data.get("best_result")
        if not isinstance(best_result, dict):
            continue
        completed_ok, completed_err = board_ops._validate_completed_optimization_notes(notes_str)
        if not completed_ok:
            errors.append(f"{model_id}: {completed_err}")
            continue
        artifact_ok, artifact_err = board_ops._validate_optimization_metric_artifacts(adaptation_path, notes_str)
        if not artifact_ok:
            errors.append(f"{model_id}: {artifact_err}")
    return errors


def check_adapt_completed_optimization_gate(project_root: Path, adaptation_name: str) -> list[str]:
    """--adapt 模式下模拟 optimization_status=completed 的完整门禁。"""
    board_ops = _load_board_ops_module(project_root)
    # Handle both relative names (e.g. "model_name") and full paths
    # If adaptation_name already contains "adaptations/" prefix, use as-is
    if adaptation_name.startswith("adaptations/"):
        path_val = adaptation_name
    elif "/" in adaptation_name or adaptation_name.startswith(str(project_root).split("/")[-1]):
        # Looks like a full path - try to extract relative part
        try:
            rel_path = Path(adaptation_name).relative_to(project_root)
            path_val = str(rel_path)
        except ValueError:
            path_val = adaptation_name
    else:
        path_val = f"adaptations/{adaptation_name}"
    file_ok, file_notes, file_err = board_ops._read_optimization_notes_file(path_val)
    if not file_ok:
        return [file_err]
    completed_ok, completed_err = board_ops._validate_completed_optimization_notes(file_notes)
    if not completed_ok:
        return [completed_err]
    artifact_ok, artifact_err = board_ops._validate_optimization_metric_artifacts(path_val, file_notes)
    if not artifact_ok:
        return [artifact_err]
    return []


def check_file(path: Path) -> list[str]:
    """检查单个 accuracy_run_perf.py，返回违规列表。"""
    content = path.read_text()
    errors: list[str] = []

    # 1. 必须定义 PERF_SUFFIX
    if "PERF_SUFFIX" not in content or "_perf" not in content:
        errors.append("必须定义 PERF_SUFFIX = '_perf'")

    # 2. 必须体现 patch 或 runtime 优化承载方式
    has_model_files_path = "model_files" in content or "MODEL_PATH" in content
    has_runtime_only_support = "TASK_QUEUE_ENABLE" in content or "warmup" in content.lower()
    if not has_model_files_path and not has_runtime_only_support:
        errors.append("必须体现优化承载方式：model_files/patch 加载，或 warmup / TASK_QUEUE_ENABLE 等运行时优化逻辑")

    # 3. 必须支持 run 子命令
    if "run" not in content or ("subparsers" not in content and "add_parser" not in content):
        errors.append("必须支持 run 子命令")

    # 4. 必须支持 compare 子命令
    if "compare" not in content:
        errors.append("必须支持 compare 子命令（对比 CUDA/NPU 产出）")

    # 5. 产出路径必须含 _perf
    if "_perf" not in content or ("benchmark_metrics" in content and "perf" not in "".join(content.split())):
        # 宽松检查：至少有一处 _perf 相关
        if re.search(r'["\']_perf["\']', content) or "PERF_SUFFIX" in content:
            pass  # 已有 PERF_SUFFIX 则通过
        else:
            errors.append("产出路径必须含 _perf 后缀（如 benchmark_metrics_*_perf.json）")

    path_templates = re.findall(
        r'f["\']([^"\']*(?:outputs_[^"\']*\.pt|benchmark_metrics_[^"\']*\.json))["\']',
        content,
    )
    mode_suffix_ok = re.compile(r"\{mode_str\}|_pretrained|_config")
    for tmpl in path_templates:
        # Skip glob patterns (contain * wildcard) used for file discovery
        if "*" in tmpl:
            continue
        if not mode_suffix_ok.search(tmpl):
            errors.append("产出路径必须含 mode 后缀，如 {mode_str} 或字面 _pretrained/_config，避免 baseline/perf artifact 混淆")
            break

    if re.search(r'(?:HF_HOME|TRANSFORMERS_CACHE|CACHE_DIR|cache_dir)\s*=\s*["\'](?:\.\/)?models["\']', content) or re.search(r'Path\s*\(\s*["\']models["\']\s*\)', content):
        errors.append("缓存目录禁止使用相对路径 models/；必须固定到 adaptation_path/models/")
    if re.search(r'Path\s*\(\s*__file__\s*\)\.resolve\(\)\.parent\.parent\s*/\s*["\']models["\']', content):
        errors.append("缓存目录禁止使用 Path(__file__).resolve().parent.parent / 'models'；这会指向项目根 models/")
    if "models" in content and ("Path.cwd()" in content or "os.getcwd()" in content or "getcwd()" in content):
        errors.append("缓存目录禁止基于 cwd/getcwd() 推导 models/；必须固定到 adaptation_path/models/")

    # 6. 禁止 silent fallback: --use-pretrained 失败后偷偷改为 config 继续跑
    has_pretrained_fallback_text = re.search(r"falling\s+back\s+to\s+config", content, re.I) is not None
    # Only flag actual reassignments INSIDE function bodies, not default parameter values
    mutates_use_pretrained = re.search(r"^\s+use_pretrained\s*=\s*False", content, re.M) is not None
    if has_pretrained_fallback_text or mutates_use_pretrained:
        if "from_pretrained" in content:
            errors.append("禁止 silent fallback：`--use-pretrained` 加载失败时必须直接报错退出，不能回退到 config 继续产出 perf/notes")

    return errors


# ============================================================
# optimization_notes JSON 格式校验（串联 check_optimization_notes.py）
# ============================================================


def check_optimization_notes(project_root: Path) -> list[str]:
    """校验 db 中 optimization_status=completed 记录的 optimization_notes JSON 格式。"""
    db_path = project_root / "board.db"
    if not db_path.exists():
        return []
    check_script = project_root / "optimization" / "scripts" / "check_optimization_notes.py"
    spec = importlib.util.spec_from_file_location("check_optimization_notes", str(check_script))
    if spec is None or spec.loader is None:
        return ["check_optimization_notes.py 加载失败"]
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        "SELECT model_id, optimization_notes FROM models WHERE optimization_status = ?",
        ("completed",),
    )
    rows = cur.fetchall()
    conn.close()

    errors: list[str] = []
    for model_id, notes_str in rows:
        if not notes_str or not notes_str.strip():
            errors.append(f"{model_id}: optimization_notes 为空")
            continue
        model_errors = mod.validate_notes(notes_str, model_id)
        errors.extend(f"{model_id}: {err}" for err in model_errors)
    return errors


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="检查 accuracy_run_perf.py 核心结构")
    parser.add_argument(
        "--adapt",
        default=None,
        help="仅检查指定 adaptation 目录名",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="仅警告不退出码 1",
    )
    parser.add_argument(
        "--db-only",
        action="store_true",
        help="仅检查 board.db 中 optimization_status=completed 的 adaptation",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    adaptations_dir = project_root / "adaptations"

    if args.adapt:
        # Strip "adaptations/" prefix if already present to avoid double-path
        adapt_name = args.adapt.replace("adaptations/", "").replace("adaptations\\", "")
        paths = [adaptations_dir / adapt_name / "accuracy_run_perf.py"]
        if not paths[0].exists():
            print(f"[check] 文件不存在: {paths[0]}")
            return 1
    elif args.db_only:
        completed_names = get_optimization_adapt_names(project_root)
        if not completed_names:
            print("[check] board.db 中无 optimization_status=completed 的 adaptation，跳过检查")
            return 0
        paths = [adaptations_dir / n / "accuracy_run_perf.py" for n in completed_names]
        paths = [p for p in paths if p.exists()]
        if not paths:
            print("[check] 无 optimization completed adaptation 含 accuracy_run_perf.py，跳过检查")
            return 0
        print(f"[check] 仅检查 db 中 optimization_status=completed 的 {len(completed_names)} 个 adaptation（存在 accuracy_run_perf.py 的 {len(paths)} 个）")
    else:
        paths = sorted(adaptations_dir.glob("*/accuracy_run_perf.py"))

    total_errors = 0
    for path in paths:
        rel = path.relative_to(project_root)
        errors = check_file(path)
        if errors:
            total_errors += len(errors)
            print(f"\n[check] ❌ {rel}")
            for e in errors:
                print(f"       - {e}")
        else:
            print(f"[check] ✅ {rel}")

    if args.adapt:
        gate_errors = check_adapt_completed_optimization_gate(project_root, args.adapt)
        if gate_errors:
            total_errors += len(gate_errors)
            print(f"\n[check] ❌ completed gate 模拟校验（{args.adapt}）")
            for e in gate_errors:
                print(f"       - {e}")
        else:
            print(f"[check] ✅ completed gate 模拟校验通过（{args.adapt}）")

    # --db-only 模式下串联 optimization_notes JSON 格式校验
    if args.db_only:
        notes_errors = check_optimization_notes(project_root)
        notes_count = len(completed_names)  # total db records, not just file-checked
        if notes_errors:
            total_errors += len(notes_errors)
            print(f"\n[check] ❌ optimization_notes JSON 格式校验（{len(notes_errors)} 项违规）")
            for e in notes_errors:
                print(f"       - {e}")
        else:
            print(f"[check] ✅ optimization_notes JSON 格式校验通过（{notes_count} 条记录）")

    if args.db_only:
        artifact_errors = check_completed_optimization_metric_artifacts(project_root)
        if artifact_errors:
            total_errors += len(artifact_errors)
            print(f"\n[check] ❌ optimization 工件样本数校验（{len(artifact_errors)} 项违规）")
            for e in artifact_errors:
                print(f"       - {e}")
        else:
            print("[check] ✅ optimization 工件样本数校验通过")

    if total_errors > 0:
        print(f"\n[check] 共 {total_errors} 项违规")
        return 0 if args.warn_only else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
