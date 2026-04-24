#!/usr/bin/env python3
"""
校验 optimization_notes JSON 格式。

标准 schema:
{
  "measurement_contract_version": 3,
  "optimization_kind": "fusion|runtime_only|hybrid",  // 可选；缺省时从 optimization_items 推断
  "optimizations": "描述字符串",
  "results": [
    {
      "dtype": "fp32|fp16|bf16",
      "mode": "pretrained|config",
      "dataset": "数据集名称",
      "output_type": "输出类型",
      "baseline_artifact": "benchmark_metrics_xxx.json",
      "perf_artifact": "benchmark_metrics_xxx_perf.json",
      "num_samples": 50,
      "perf_latency_s": 0.xxx,
      "baseline_latency_s": 0.xxx,
      "perf_wall_clock_s": 0.xxx,
      "baseline_wall_clock_s": 0.xxx,
      "wall_clock_source": "artifact_timestamps|artifact_explicit_field",
      "baseline_warmup_iterations": 3,
      "perf_warmup_iterations": 3,
      "warmup_policy": "symmetric",
      "perf_memory_mb": 0.xxx,
      "cosine_similarity": 0.xxx,
      "optimization_items": ["item1", "item2"],
      "optimization_kind": "fusion|runtime_only|hybrid",  // 可选；缺省时从 optimization_items 推断
      "selected_npus": ["0"],  // runtime_only/hybrid 建议记录
      "device_topology": "single_npu:0",
      "parallel_mode": "single_card",
      "code_modified": false,  // runtime_only 且 speedup_ratio=1.0 时必填
      "code_change_attempts": 2,  // runtime_only 且 speedup_ratio=1.0 时必填（>=2）
      // 必填:
      "speedup_ratio": 1.xxx,
      "latency_reduction_pct": 0.xxx,
      "baseline_memory_mb": 0.xxx,
      "memory_reduction_pct": 0.xxx,
      "text_match_rate": 0.xxx,
      "ppl_avg_rel_diff_pct": 0.xxx
    }
  ],
  "best_result": { /* results[] 中的某一项（浅拷贝） */ }
}

必填字段:
  - measurement_contract_version (number, >= 3)
  - optimizations (string, non-empty)
  - results (array, non-empty, 每项含 dtype/mode/dataset/output_type/baseline_artifact/perf_artifact/num_samples/perf_latency_s/baseline_latency_s/perf_wall_clock_s/baseline_wall_clock_s/wall_clock_source/baseline_warmup_iterations/perf_warmup_iterations/warmup_policy/perf_memory_mb/speedup_ratio/cosine_similarity)
  - best_result (dict, non-null, 必须是 results[] 中的某一项)
  - 若 runtime_only 且 speedup_ratio=1.0：必须声明 code_modified=false、code_change_attempts>=2，并在 validation_note/optimizations 注明模型代码无更改

用法:
    uv run python optimization/scripts/check_optimization_notes.py                              # 校验所有 completed 记录
    uv run python optimization/scripts/check_optimization_notes.py --adapt adaptations/xxx      # 校验指定 adaptation 的磁盘文件
    uv run python optimization/scripts/check_optimization_notes.py --adapt xxx                  # 校验 adaptations/xxx/optimization_notes.json
    uv run python optimization/scripts/check_optimization_notes.py --db-only --adapt xxx        # 仅校验 db 中 matching completed 记录
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

# ============================================================
# Schema Definition
# ============================================================

REQUIRED_TOP_LEVEL = ["measurement_contract_version", "optimizations", "results", "best_result"]

REQUIRED_RESULT_FIELDS = [
    "dtype",
    "mode",
    "dataset",
    "output_type",
    "baseline_artifact",
    "perf_artifact",
    "num_samples",
    "perf_latency_s",
    "baseline_latency_s",
    "perf_wall_clock_s",
    "baseline_wall_clock_s",
    "wall_clock_source",
    "baseline_warmup_iterations",
    "perf_warmup_iterations",
    "warmup_policy",
    "perf_memory_mb",
    "speedup_ratio",
    "cosine_similarity",
]

VALID_DTYPES = {"fp32", "fp16", "bf16"}
VALID_MODES = {"pretrained", "config"}
VALID_COMPARISON_SCOPES = {"cold_start", "steady_state", "mixed"}
VALID_WALL_CLOCK_SOURCES = {"artifact_timestamps", "artifact_explicit_field", "artifact_latency_times_samples"}
VALID_WARMUP_POLICIES = {"symmetric"}
VALID_OPTIMIZATION_KINDS = {"fusion", "runtime_only", "hybrid"}


def _metric_close(expected: float, actual: float) -> bool:
    abs_tol = max(1e-3, abs(expected) * 0.02)
    return abs(expected - actual) <= abs_tol


def _requires_per_sample_wall_clock_alignment(output_type: str) -> bool:
    lowered = (output_type or "").strip().lower()
    if not lowered:
        return False
    return not any(token in lowered for token in ("generated_text", "transcription", "qa_answer", "generated_action", "generated_image"))


def _contains_measurement_red_flag(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    red_flags = (
        "derived from per-sample latencies",
        "derived from latency",
        "artifact timestamps appear to measure partial run",
        "partial run",
        "baseline was cold",
        "cold baseline",
        "no warmup",
        "violates team-lead measurement requirement",
        "not directly comparable",
        "ttft vs batch-avg-latency",
    )
    return any(flag in lowered for flag in red_flags)


def _normalize_optimization_items(raw_items: Any) -> set[str]:
    normalized: set[str] = set()
    if isinstance(raw_items, str):
        pieces = [raw_items]
    elif isinstance(raw_items, list):
        pieces = [item for item in raw_items if isinstance(item, str)]
    else:
        pieces = []
    for piece in pieces:
        lowered = piece.strip().lower()
        if lowered:
            normalized.add(lowered)
    return normalized


def _has_runtime_only_item(items: set[str]) -> bool:
    runtime_tokens = ("warmup", "task_queue_enable", "task queue", "queue_enable")
    return any(any(token in item for token in runtime_tokens) for item in items)


def _has_fusion_item(items: set[str]) -> bool:
    return any(item.startswith("npu_") for item in items)


def _infer_optimization_kind(notes: dict[str, Any], result: dict[str, Any]) -> str:
    explicit = str(result.get("optimization_kind") or notes.get("optimization_kind") or "").strip().lower()
    if explicit in VALID_OPTIMIZATION_KINDS:
        return explicit
    items = _normalize_optimization_items(result.get("optimization_items"))
    has_runtime = _has_runtime_only_item(items)
    has_fusion = _has_fusion_item(items)
    if has_runtime and has_fusion:
        return "hybrid"
    if has_runtime:
        return "runtime_only"
    return "fusion"


def _extract_selected_npus(payload: dict[str, Any]) -> list[str]:
    selected_npus = payload.get("selected_npus")
    if isinstance(selected_npus, list):
        values = [str(item).strip() for item in selected_npus if str(item).strip()]
        if values:
            return values
    selected_npu = payload.get("selected_npu")
    if selected_npu is None:
        return []
    value = str(selected_npu).strip()
    return [value] if value else []


def _contains_no_model_code_change_note(notes: dict[str, Any], result: dict[str, Any]) -> bool:
    texts: list[str] = []
    for raw in (
        result.get("validation_note"),
        result.get("optimizations"),
        notes.get("optimizations"),
    ):
        if isinstance(raw, str):
            stripped = raw.strip().lower()
            if stripped:
                texts.append(stripped)
    if not texts:
        return False
    merged = " | ".join(texts)
    markers = (
        "模型代码无更改",
        "模型代码未改动",
        "未修改模型代码",
        "model code unchanged",
        "no model code change",
        "without model code changes",
    )
    return any(marker in merged for marker in markers)


# ============================================================
# Validation
# ============================================================


def validate_notes(notes_str: str, model_id: str) -> list[str]:
    """Validate optimization_notes JSON, return list of errors."""
    errors: list[str] = []

    # 1. Must be valid JSON
    try:
        notes = json.loads(notes_str)
    except json.JSONDecodeError as e:
        errors.append(f"无效 JSON: {e}")
        return errors

    if not isinstance(notes, dict):
        errors.append("必须是 JSON object")
        return errors

    # 2. Required top-level keys
    for key in REQUIRED_TOP_LEVEL:
        if key not in notes:
            errors.append(f"缺少必填字段: {key}")

    contract_v = notes.get("measurement_contract_version")
    if contract_v is not None:
        if not isinstance(contract_v, (int, float)) or isinstance(contract_v, bool):
            errors.append("measurement_contract_version 必须是数字")
        elif int(contract_v) < 3:
            errors.append(f"measurement_contract_version 必须 >= 3，当前: {contract_v}")

    if "optimizations" in notes:
        if not isinstance(notes["optimizations"], str) or not notes["optimizations"].strip():
            errors.append("optimizations 必须是非空字符串")

    if "results" in notes:
        results = notes["results"]
        if not isinstance(results, list):
            errors.append("results 必须是数组")
        elif len(results) == 0:
            errors.append("results 不能为空")
        else:
            for i, r in enumerate(results):
                prefix = f"results[{i}]"
                if not isinstance(r, dict):
                    errors.append(f"{prefix} 必须是 object")
                    continue
                r = cast(dict[str, Any], r)
                for field in REQUIRED_RESULT_FIELDS:
                    if field not in r:
                        errors.append(f"{prefix} 缺少必填字段: {field}")
                dtype_v = r.get("dtype")
                if dtype_v is not None and dtype_v not in VALID_DTYPES:
                    errors.append(f"{prefix} dtype 无效: {dtype_v} (应为 {VALID_DTYPES})")
                mode_v = r.get("mode")
                if mode_v is not None and mode_v not in VALID_MODES:
                    errors.append(f"{prefix} mode 无效: {mode_v} (应为 {VALID_MODES})")
                explicit_optimization_kind = str(r.get("optimization_kind") or notes.get("optimization_kind") or "").strip().lower()
                optimization_kind = _infer_optimization_kind(notes, r)
                for field in ("dataset", "output_type", "baseline_artifact", "perf_artifact"):
                    field_v = r.get(field)
                    if field_v is not None and (not isinstance(field_v, str) or not field_v.strip()):
                        errors.append(f"{prefix} {field} 必须是非空字符串")
                if "perf_latency_s" in r and not isinstance(r.get("perf_latency_s"), (int, float)):
                    errors.append(f"{prefix} perf_latency_s 必须是数字")
                if "baseline_latency_s" in r and not isinstance(r.get("baseline_latency_s"), (int, float)):
                    errors.append(f"{prefix} baseline_latency_s 必须是数字")
                if "perf_wall_clock_s" in r:
                    perf_wall_clock_v = r.get("perf_wall_clock_s")
                    if not isinstance(perf_wall_clock_v, (int, float)) or perf_wall_clock_v <= 0:
                        errors.append(f"{prefix} perf_wall_clock_s 必须是正数")
                if "baseline_wall_clock_s" in r:
                    baseline_wall_clock_v = r.get("baseline_wall_clock_s")
                    if not isinstance(baseline_wall_clock_v, (int, float)) or baseline_wall_clock_v <= 0:
                        errors.append(f"{prefix} baseline_wall_clock_s 必须是正数")
                if "num_samples" in r:
                    num_samples_v = r.get("num_samples")
                    if not isinstance(num_samples_v, (int, float)) or isinstance(num_samples_v, bool):
                        errors.append(f"{prefix} num_samples 必须是数字")
                    elif float(num_samples_v) < 50:
                        errors.append(f"{prefix} num_samples 必须 >= 50，当前: {num_samples_v}")
                if "speedup_ratio" in r and not isinstance(r.get("speedup_ratio"), (int, float)):
                    errors.append(f"{prefix} speedup_ratio 必须是数字")
                wall_clock_source = r.get("wall_clock_source")
                if wall_clock_source is not None and wall_clock_source not in VALID_WALL_CLOCK_SOURCES:
                    errors.append(f"{prefix} wall_clock_source 无效: {wall_clock_source} (应为 {VALID_WALL_CLOCK_SOURCES})")
                for warmup_field in ("baseline_warmup_iterations", "perf_warmup_iterations"):
                    warmup_value = r.get(warmup_field)
                    if not isinstance(warmup_value, (int, float)) or isinstance(warmup_value, bool) or float(warmup_value) < 0:
                        errors.append(f"{prefix} {warmup_field} 必须是 >= 0 的数字")
                warmup_policy = r.get("warmup_policy")
                if warmup_policy is not None and warmup_policy not in VALID_WARMUP_POLICIES:
                    errors.append(f"{prefix} warmup_policy 无效: {warmup_policy} (应为 {VALID_WARMUP_POLICIES})")
                if all(isinstance(r.get(key), (int, float)) for key in ("baseline_warmup_iterations", "perf_warmup_iterations")):
                    if not _metric_close(float(r["baseline_warmup_iterations"]), float(r["perf_warmup_iterations"])):
                        errors.append(f"{prefix} baseline_warmup_iterations 与 perf_warmup_iterations 必须一致")
                if "cosine_similarity" in r:
                    v = r.get("cosine_similarity")
                    if not isinstance(v, (int, float)) or not (0 <= v <= 1.0 + 1e-6):
                        errors.append(f"{prefix} cosine_similarity 必须在 [0, 1] 范围内，当前: {v}")
                if all(isinstance(r.get(key), (int, float)) for key in ("baseline_wall_clock_s", "perf_wall_clock_s", "speedup_ratio")):
                    perf_wall_clock_v = float(r["perf_wall_clock_s"])
                    baseline_wall_clock_v = float(r["baseline_wall_clock_s"])
                    if perf_wall_clock_v <= 0 or baseline_wall_clock_v <= 0:
                        errors.append(f"{prefix} baseline_wall_clock_s/perf_wall_clock_s 必须为正数")
                    else:
                        expected_speedup = baseline_wall_clock_v / perf_wall_clock_v
                        if not _metric_close(expected_speedup, float(r["speedup_ratio"])):
                            errors.append(f"{prefix} speedup_ratio 必须按 baseline_wall_clock_s / perf_wall_clock_s 计算")
                if _requires_per_sample_wall_clock_alignment(str(r.get("output_type") or "")) and all(
                    isinstance(r.get(key), (int, float)) for key in ("baseline_wall_clock_s", "perf_wall_clock_s", "baseline_latency_s", "perf_latency_s", "num_samples")
                ):
                    num_samples_v = float(r["num_samples"])
                    if num_samples_v > 0:
                        baseline_per_sample_wall_clock = float(r["baseline_wall_clock_s"]) / num_samples_v
                        perf_per_sample_wall_clock = float(r["perf_wall_clock_s"]) / num_samples_v
                        if not _metric_close(baseline_per_sample_wall_clock, float(r["baseline_latency_s"])):
                            errors.append(f"{prefix} 非生成类任务要求 baseline_wall_clock_s / num_samples 与 baseline_latency_s 一致")
                        if not _metric_close(perf_per_sample_wall_clock, float(r["perf_latency_s"])):
                            errors.append(f"{prefix} 非生成类任务要求 perf_wall_clock_s / num_samples 与 perf_latency_s 一致")
                if all(isinstance(r.get(key), (int, float)) for key in ("baseline_latency_s", "perf_latency_s", "forward_latency_speedup_ratio")):
                    perf_latency_v = float(r["perf_latency_s"])
                    baseline_latency_v = float(r["baseline_latency_s"])
                    if perf_latency_v <= 0 or baseline_latency_v <= 0:
                        errors.append(f"{prefix} baseline_latency_s/perf_latency_s 必须为正数")
                    else:
                        expected_forward_speedup = baseline_latency_v / perf_latency_v
                        if not _metric_close(expected_forward_speedup, float(r["forward_latency_speedup_ratio"])):
                            errors.append(f"{prefix} forward_latency_speedup_ratio 必须按 baseline_latency_s / perf_latency_s 计算")
                comparison_scope = r.get("comparison_scope")
                if comparison_scope is not None and comparison_scope not in VALID_COMPARISON_SCOPES:
                    errors.append(f"{prefix} comparison_scope 无效: {comparison_scope} (应为 {VALID_COMPARISON_SCOPES})")
                if "comparison_method" in r and r.get("comparison_method") is not None and not isinstance(r.get("comparison_method"), str):
                    errors.append(f"{prefix} comparison_method 必须是字符串")
                if "precision_method" in r and r.get("precision_method") is not None and not isinstance(r.get("precision_method"), str):
                    errors.append(f"{prefix} precision_method 必须是字符串")
                if "validation_note" in r and r.get("validation_note") is not None and not isinstance(r.get("validation_note"), str):
                    errors.append(f"{prefix} validation_note 必须是字符串")
                if isinstance(r.get("validation_note"), str) and _contains_measurement_red_flag(r["validation_note"]):
                    errors.append(f"{prefix} validation_note 暴露了不可靠测量口径（cold baseline / derived from latency / not directly comparable 等），不得标记 completed")
                if explicit_optimization_kind == "runtime_only":
                    merged_items = _normalize_optimization_items(notes.get("optimization_items")) | _normalize_optimization_items(r.get("optimization_items"))
                    if not _has_runtime_only_item(merged_items):
                        errors.append(f"{prefix} runtime_only 结果必须在 optimization_items 中显式记录 warmup / TASK_QUEUE_ENABLE")
                    if _has_fusion_item(merged_items):
                        errors.append(f"{prefix} runtime_only 结果不得包含 npu_* 融合算子；混合路径请标记为 hybrid")
                    selected_npus = _extract_selected_npus(r) or _extract_selected_npus(notes)
                    if not selected_npus:
                        errors.append(f"{prefix} runtime_only 结果必须记录 selected_npu 或 selected_npus")
                    device_topology = str(r.get("device_topology") or notes.get("device_topology") or "").strip()
                    if not device_topology:
                        errors.append(f"{prefix} runtime_only 结果必须记录 device_topology")
                    parallel_mode = str(r.get("parallel_mode") or notes.get("parallel_mode") or "").strip()
                    if not parallel_mode:
                        errors.append(f"{prefix} runtime_only 结果必须记录 parallel_mode")
                speedup_v = r.get("speedup_ratio")
                if explicit_optimization_kind == "runtime_only" and isinstance(speedup_v, (int, float)) and abs(float(speedup_v) - 1.0) <= 1e-6:
                    code_modified = r.get("code_modified", notes.get("code_modified"))
                    if code_modified is not False:
                        errors.append(f"{prefix} runtime_only 且 speedup_ratio=1.0 时必须声明 code_modified=false（模型代码无更改）")
                    code_change_attempts = r.get("code_change_attempts", notes.get("code_change_attempts"))
                    if not isinstance(code_change_attempts, (int, float)) or isinstance(code_change_attempts, bool) or float(code_change_attempts) < 2:
                        errors.append(f"{prefix} runtime_only 且 speedup_ratio=1.0 时必须记录 code_change_attempts>=2")
                    if not _contains_no_model_code_change_note(notes, r):
                        errors.append(f"{prefix} runtime_only 且 speedup_ratio=1.0 时必须在 validation_note/optimizations 写明模型代码无更改")
                if isinstance(speedup_v, (int, float)) and speedup_v >= 3.0:
                    comparison_method = r.get("comparison_method")
                    precision_method = str(r.get("precision_method") or "")
                    if comparison_method != "independent_baseline_artifact":
                        errors.append(f"{prefix} speedup_ratio >= 3x 时 comparison_method 必须为 independent_baseline_artifact")
                    if "self_baseline" in precision_method:
                        errors.append(f"{prefix} speedup_ratio >= 3x 时 precision_method 禁止使用 self_baseline 系取值")
                    if comparison_scope not in VALID_COMPARISON_SCOPES:
                        errors.append(f"{prefix} speedup_ratio >= 3x 时必须提供有效 comparison_scope")
                    validation_note = r.get("validation_note")
                    if not isinstance(validation_note, str) or not validation_note.strip():
                        errors.append(f"{prefix} speedup_ratio >= 3x 时必须提供非空 validation_note")
                    for steady_key in ("steady_state_baseline_latency_s", "steady_state_perf_latency_s"):
                        steady_v = r.get(steady_key)
                        if not isinstance(steady_v, (int, float)) or steady_v <= 0:
                            errors.append(f"{prefix} speedup_ratio >= 3x 时 {steady_key} 必须为正数")

    # 3. best_result must exist and reference a results entry
    if "best_result" in notes:
        br = notes["best_result"]
        if br is None:
            errors.append("best_result 不能为 null")
        elif isinstance(br, dict):
            # Verify best_result matches a results entry
            if "results" in notes and isinstance(notes["results"], list):
                found = False
                for r in notes["results"]:
                    if isinstance(r, dict) and all(r.get(key) == br.get(key) for key in ("dtype", "mode", "dataset", "output_type", "baseline_artifact", "perf_artifact")):
                        found = True
                        break
                if not found:
                    errors.append(f"best_result 必须与 results[] 中某一项完全一致 (dtype={br.get('dtype')}, mode={br.get('mode')}, dataset={br.get('dataset')}, output_type={br.get('output_type')})")
        else:
            errors.append("best_result 必须是 object 或 null")

    return errors


# ============================================================
# DB Operations
# ============================================================


def get_db_path() -> Path:
    """Get board.db path."""
    # Try from script location
    script_dir = Path(__file__).resolve().parent
    db_path = script_dir.parent.parent / "board.db"
    if db_path.exists():
        return db_path
    # Try CWD
    db_path = Path.cwd() / "board.db"
    if db_path.exists():
        return db_path
    print("Error: board.db not found", file=sys.stderr)
    sys.exit(1)


def get_completed_models(db_path: Path, adapt_filter: Optional[str] = None) -> list[dict]:
    """Get optimization_status=completed models from db."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if adapt_filter:
        cur = conn.execute(
            "SELECT model_id, adaptation_path, optimization_notes FROM models WHERE optimization_status = 'completed' AND adaptation_path LIKE ?",
            (f"%{adapt_filter}%",),
        )
    else:
        cur = conn.execute("SELECT model_id, adaptation_path, optimization_notes FROM models WHERE optimization_status = 'completed'")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def resolve_adaptation_dir(project_root: Path, adapt_arg: str) -> Path | None:
    """Resolve an adaptation directory from sanitized name or path."""
    raw = Path(adapt_arg)
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                Path.cwd() / raw,
                project_root / raw,
                project_root / "adaptations" / adapt_arg,
            ]
        )

    for candidate in candidates:
        if candidate.is_file() and candidate.name == "optimization_notes.json":
            candidate = candidate.parent
        if candidate.is_dir():
            return candidate.resolve()
    return None


# ============================================================
# Main
# ============================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 optimization_notes JSON 格式")
    parser.add_argument("--adapt", default=None, help="仅校验指定 adaptation")
    parser.add_argument("--db-only", action="store_true", help="仅检查 db 中 completed 的记录（不读取 adaptation 本地文件）")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    db_path = get_db_path()

    if args.adapt and not args.db_only:
        adapt_dir = resolve_adaptation_dir(project_root, args.adapt)
        if adapt_dir is None:
            print(f"[check_optimization_notes] adaptation 不存在: {args.adapt}", file=sys.stderr)
            print("[check_optimization_notes] 如需仅检查 board.db 中的 completed 记录，请显式传入 --db-only", file=sys.stderr)
            return 1

        notes_file = adapt_dir / "optimization_notes.json"
        if not notes_file.exists():
            print(f"[check_optimization_notes] 缺少文件: {notes_file}", file=sys.stderr)
            return 1

        notes_str = notes_file.read_text(encoding="utf-8")
        errors = validate_notes(notes_str, adapt_dir.name)
        print(f"[check_optimization_notes] 校验 adaptation 本地文件: {notes_file}")
        if errors:
            print(f"\n❌ {adapt_dir.name}")
            for e in errors:
                print(f"   - {e}")
            print(f"\n[check_optimization_notes] 共 {len(errors)} 项违规")
            return 1

        print(f"✅ {adapt_dir.name}")
        print("\n[check_optimization_notes] 全部通过")
        return 0

    models = get_completed_models(db_path, adapt_filter=args.adapt)

    if not models:
        print("[check_optimization_notes] 无 optimization_status=completed 的记录")
        return 0

    print(f"[check_optimization_notes] 校验 {len(models)} 条记录")

    total_errors = 0
    for model in models:
        model_id = model["model_id"]
        notes_str = model.get("optimization_notes") or ""

        errors = validate_notes(notes_str, model_id)
        if errors:
            total_errors += len(errors)
            print(f"\n❌ {model_id}")
            for e in errors:
                print(f"   - {e}")
        else:
            print(f"✅ {model_id}")

    if total_errors > 0:
        print(f"\n[check_optimization_notes] 共 {total_errors} 项违规")
        return 1

    print(f"\n[check_optimization_notes] 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
