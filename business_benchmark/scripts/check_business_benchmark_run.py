#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dataset_mapping import resolve_multilingual_asr_dataset

MIN_BUSINESS_SAMPLE_LOWER_BOUND = 50
REQUIRED_METRIC_FIELDS = ("latency_s", "num_samples", "mode", "dataset", "dtype", "output_type", "device", "start_time", "end_time")
VALID_COMPARISON_SCOPES = {"real_business", "cold_start", "steady_state", "mixed"}
VALID_LATENCY_MEASUREMENT_SCOPES = {"real_business", "cold_start", "steady_state", "mixed"}
REQUIRED_TOP_LEVEL = (
    "dataset",
    "comparison_scope",
    "num_samples",
    "remote_execution",
    "comparison_evidence",
    "results",
    "best_result",
    "npu_baseline_artifact",
    "npu_perf_artifact",
    "cuda_baseline_artifact",
)
QUALITY_VALUE_KEYS = ("exact_match", "f1", "accuracy", "top1_accuracy", "rougeL", "ndcg_at_10", "mAP", "map50", "match_rate", "text_match_rate", "wer", "mrr", "perplexity", "keypoint_repeatability")
SUSPICIOUS_TINY_LATENCY_S = 1e-4
MIN_WALL_CLOCK_FOR_TINY_LATENCY_S = 1e-2
MODEL_FILES_EVIDENCE_CANDIDATES = ("config.json", "npu_patches.py", "__init__.py")
MIN_SANE_NPU_SPEEDUP_RATIO = 0.9
MIN_SANE_VS_CUDA_LATENCY_RATIO = 0.22
AUTO_MAPPING_RECHECK_VS_CUDA_LATENCY_RATIO = 0.4
MAX_ALLOWED_BOUNDED_QUALITY_DELTA = 0.005
BOUNDED_QUALITY_METRIC_KEYS = {
    "exact_match",
    "f1",
    "accuracy",
    "top1_accuracy",
    "rougel",
    "ndcg_at_10",
    "map",
    "map50",
    "match_rate",
    "text_match_rate",
    "mrr",
    "cosine_similarity",
    "keypoint_repeatability",
}
STRICT_ALIGNMENT_QUALITY_KEYS = {
    "exact_match",
    "accuracy",
    "top1_accuracy",
    "match_rate",
    "text_match_rate",
}


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def _metric_close(expected: float, actual: float) -> bool:
    if not math.isfinite(expected) or not math.isfinite(actual):
        return False
    abs_tol = max(1e-3, abs(expected) * 0.02)
    return abs(expected - actual) <= abs_tol


def _normalized_quality_metric_name(metric_name: object) -> str:
    return str(metric_name or "").strip().lower()


def _lookup_numeric_metric(source: object, metric_name: object) -> float | None:
    metric_key = str(metric_name or "").strip()
    if not metric_key or not isinstance(source, dict):
        return None
    value = source.get(metric_key)
    if _is_number(value):
        return float(value)
    lowered_metric_key = metric_key.lower()
    for key, candidate in source.items():
        if str(key or "").strip().lower() == lowered_metric_key and _is_number(candidate):
            return float(candidate)
    return None


def _is_bounded_quality_metric(metric_name: object) -> bool:
    return _normalized_quality_metric_name(metric_name) in BOUNDED_QUALITY_METRIC_KEYS


def _is_alignment_quality_metric(metric_name: object) -> bool:
    return _normalized_quality_metric_name(metric_name) in STRICT_ALIGNMENT_QUALITY_KEYS


def _validate_bounded_quality_metric(metric_name: object, quality_value: object, field_path: str) -> list[str]:
    if quality_value is None or not _is_bounded_quality_metric(metric_name):
        return []
    if not _is_number(quality_value):
        return [f"{field_path} 必须是数值"]
    if not _is_finite_number(quality_value):
        return [f"{field_path} 不能是 NaN/Inf"]
    value = float(quality_value)
    if value > 1.0:
        return [f"{field_path}={value:.6g} 超出 0~1 合法范围；疑似评分实现或汇总口径异常"]
    return []


def _same_hardware_quality_tolerance(num_samples: float) -> float:
    return max(0.03, 3.0 / max(num_samples, 1.0))


def _cross_device_quality_tolerance(num_samples: float) -> float:
    return max(0.08, 6.0 / max(num_samples, 1.0))


def _bounded_quality_delta_tolerance(sample_counts: list[float]) -> float:
    if not sample_counts:
        return MAX_ALLOWED_BOUNDED_QUALITY_DELTA
    positive_counts = [count for count in sample_counts if count > 0]
    if not positive_counts:
        return MAX_ALLOWED_BOUNDED_QUALITY_DELTA
    # 对 accuracy / exact_match 这类离散指标，允许三样本粒度内的轻微抖动；
    # 速度比仍由独立硬门禁把关，精度只负责拦截明显口径异常。
    return max(MAX_ALLOWED_BOUNDED_QUALITY_DELTA, 0.03, 3.0 / min(positive_counts))


def _detect_throughput(metric: dict) -> tuple[str, float | None]:
    for key in ("throughput_qps", "throughput", "samples_per_second", "qps", "items_per_second"):
        value = metric.get(key)
        if _is_number(value) and float(value) > 0:
            return key, float(value)
    latency_s = metric.get("latency_s")
    if _is_number(latency_s) and float(latency_s) > 0:
        return "derived_from_latency_qps", 1.0 / float(latency_s)
    return "throughput", None


def _detect_quality(metric: dict) -> tuple[str, float | None]:
    compare = metric.get("output_compare")
    primary_metric = str(metric.get("primary_metric") or "").strip()
    if isinstance(compare, dict):
        primary_value = _lookup_numeric_metric(compare, primary_metric)
        if primary_metric and primary_value is not None:
            return primary_metric, primary_value
        for key in ("avg_cosine_similarity", "logits_avg_cosine_similarity", "cosine_similarity"):
            value = compare.get(key)
            if _is_number(value):
                return "cosine_similarity", float(value)
        for key in QUALITY_VALUE_KEYS:
            value = compare.get(key)
            if _is_number(value):
                return key, float(value)
    primary_value = _lookup_numeric_metric(metric, primary_metric)
    if primary_metric and primary_value is not None:
        return primary_metric, primary_value
    for key in ("cosine_similarity", *QUALITY_VALUE_KEYS):
        value = metric.get(key)
        if _is_number(value):
            return key, float(value)
    return "quality_metric", None


def _parse_iso_datetime(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _resolve_metric_wall_clock_s(metric: dict) -> float | None:
    explicit_wall_clock_s = metric.get("wall_clock_s")
    if _is_finite_number(explicit_wall_clock_s) and float(explicit_wall_clock_s) > 0:
        return float(explicit_wall_clock_s)
    total_duration_s = metric.get("total_duration_s")
    if _is_finite_number(total_duration_s) and float(total_duration_s) > 0:
        return float(total_duration_s)
    start_dt = _parse_iso_datetime(metric.get("start_time"))
    end_dt = _parse_iso_datetime(metric.get("end_time"))
    if start_dt is None or end_dt is None or end_dt < start_dt:
        return None
    derived_wall_clock_s = (end_dt - start_dt).total_seconds()
    if derived_wall_clock_s <= 0:
        return None
    return derived_wall_clock_s


def _has_model_files_evidence(adapt_dir: Path) -> bool:
    model_files_dir = adapt_dir / "model_files"
    if not model_files_dir.is_dir():
        return False
    if any((model_files_dir / name).exists() for name in MODEL_FILES_EVIDENCE_CANDIDATES):
        return True
    if any(model_files_dir.glob("modeling_*.py")):
        return True
    if any(model_files_dir.glob("*patch*.py")):
        return True
    return False


def _load_json_object(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _load_business_config(adapt_dir: Path) -> dict | None:
    return _load_json_object(adapt_dir / "business_benchmark_config.json")


def _should_require_model_files_in_perf(adapt_dir: Path) -> bool:
    if not _has_model_files_evidence(adapt_dir):
        return False
    config = _load_business_config(adapt_dir)
    if isinstance(config, dict) and "npu_perf_use_model_files" in config:
        return bool(config.get("npu_perf_use_model_files"))
    return True


def _metric_uses_model_files_patch(metric: dict) -> bool:
    if metric.get("loaded_from_model_files") is True:
        return True
    patch_modules = [str(module_name or "").strip().lower() for module_name in metric.get("patch_modules") or []]
    if any(module_name in {"accuracy_run_perf", "model_files"} or module_name.startswith("model_files.") for module_name in patch_modules):
        return True
    patch_hooks = [str(hook_name or "").strip().lower() for hook_name in metric.get("patch_hooks") or []]
    return any("model_files" in hook_name or "accuracy_run_perf" in hook_name for hook_name in patch_hooks)


def _quality_metric_required(metric: dict) -> bool:
    evaluation_profile = str(metric.get("evaluation_profile") or "").strip().lower()
    primary_metric = str(metric.get("primary_metric") or "").strip().lower()
    if evaluation_profile == "latency_only":
        return False
    return primary_metric not in {"", "latency_s"}


def _uses_auto_npu_mapping(metric: dict | None) -> bool:
    if not isinstance(metric, dict):
        return False
    parallel_mode = str(metric.get("parallel_mode") or "").strip().lower()
    if parallel_mode in {"auto", "device_map_auto"}:
        return True
    device_topology = str(metric.get("device_topology") or "").strip().lower()
    if device_topology == "all_visible_devices":
        return True
    return False


def validate_summary(summary_str: str, model_id: str = "") -> list[str]:
    errors: list[str] = []
    try:
        summary = json.loads(summary_str)
    except json.JSONDecodeError as e:
        return [f"无效 JSON: {e}"]
    if not isinstance(summary, dict):
        return ["business_summary 必须是 JSON object"]

    measurement_contract_version = summary.get("measurement_contract_version")
    if measurement_contract_version is not None:
        if not isinstance(measurement_contract_version, (int, float)) or isinstance(measurement_contract_version, bool) or int(measurement_contract_version) < 1:
            errors.append("measurement_contract_version 必须是 >= 1 的整数")
    contract_version = int(measurement_contract_version) if isinstance(measurement_contract_version, (int, float)) and not isinstance(measurement_contract_version, bool) else 1

    for key in REQUIRED_TOP_LEVEL:
        if key not in summary:
            errors.append(f"缺少必填字段: {key}")

    dataset = str(summary.get("dataset") or "").strip()
    if not dataset:
        errors.append("dataset 必须是非空字符串")
    comparison_scope = str(summary.get("comparison_scope") or "").strip()
    if comparison_scope and comparison_scope not in VALID_COMPARISON_SCOPES:
        errors.append(f"comparison_scope 无效: {comparison_scope}")
    latency_measurement_scope = str(summary.get("latency_measurement_scope") or "").strip()
    if contract_version >= 2 and not latency_measurement_scope:
        errors.append("measurement_contract_version>=2 时 latency_measurement_scope 必须是非空字符串")
    elif latency_measurement_scope and latency_measurement_scope not in VALID_LATENCY_MEASUREMENT_SCOPES:
        errors.append(f"latency_measurement_scope 无效: {latency_measurement_scope}")

    num_samples = summary.get("num_samples")
    if not isinstance(num_samples, (int, float)) or isinstance(num_samples, bool) or float(num_samples) <= MIN_BUSINESS_SAMPLE_LOWER_BOUND:
        errors.append(f"num_samples 必须 > {MIN_BUSINESS_SAMPLE_LOWER_BOUND}")

    remote_execution = summary.get("remote_execution")
    if not isinstance(remote_execution, dict):
        errors.append("remote_execution 必须是 object")
    elif not str(remote_execution.get("mode") or "").strip():
        errors.append("remote_execution.mode 必须是非空字符串")

    comparison_evidence = summary.get("comparison_evidence")
    if not isinstance(comparison_evidence, dict):
        errors.append("comparison_evidence 必须是 object")
    else:
        for field in (
            "npu_baseline_device_model",
            "npu_perf_device_model",
            "cuda_baseline_device_model",
            "npu_baseline_quality_metric_name",
            "npu_perf_quality_metric_name",
            "cuda_baseline_quality_metric_name",
            "npu_baseline_throughput_metric_name",
            "npu_perf_throughput_metric_name",
            "cuda_baseline_throughput_metric_name",
        ):
            if not str(comparison_evidence.get(field) or "").strip():
                errors.append(f"comparison_evidence.{field} 必须是非空字符串")
        for field in (
            "npu_baseline_quality_metric_value",
            "npu_perf_quality_metric_value",
            "cuda_baseline_quality_metric_value",
        ):
            value = comparison_evidence.get(field)
            if value is not None and (not _is_finite_number(value) or float(value) < 0):
                errors.append(f"comparison_evidence.{field} 必须为非负数或 null")
        for name_field, value_field in (
            ("npu_baseline_quality_metric_name", "npu_baseline_quality_metric_value"),
            ("npu_perf_quality_metric_name", "npu_perf_quality_metric_value"),
            ("cuda_baseline_quality_metric_name", "cuda_baseline_quality_metric_value"),
        ):
            errors.extend(
                _validate_bounded_quality_metric(
                    comparison_evidence.get(name_field),
                    comparison_evidence.get(value_field),
                    f"comparison_evidence.{value_field}",
                )
            )
        for field in (
            "npu_baseline_peak_memory_mb",
            "npu_perf_peak_memory_mb",
            "cuda_baseline_peak_memory_mb",
        ):
            value = comparison_evidence.get(field)
            if value is not None and (not _is_finite_number(value) or float(value) < 0):
                errors.append(f"comparison_evidence.{field} 必须为非负数或 null")
        for field in (
            "npu_baseline_throughput_metric_value",
            "npu_perf_throughput_metric_value",
            "cuda_baseline_throughput_metric_value",
        ):
            value = comparison_evidence.get(field)
            if not _is_finite_number(value) or float(value) <= 0:
                errors.append(f"comparison_evidence.{field} 必须为正数")

    results = summary.get("results")
    benchmark_run_id = str(summary.get("benchmark_run_id") or "").strip()
    benchmark_run_started_at = str(summary.get("benchmark_run_started_at") or "").strip()
    legacy_missing_run_id = not benchmark_run_id
    if benchmark_run_id and not benchmark_run_started_at:
        errors.append("benchmark_run_id 非空时，business_summary.benchmark_run_started_at 必须是非空字符串")
    elif benchmark_run_started_at and _parse_iso_datetime(benchmark_run_started_at) is None:
        errors.append("business_summary.benchmark_run_started_at 必须是合法 ISO 时间字符串")
    required_roles = {"npu_baseline", "npu_perf", "cuda_baseline"}
    seen_roles: set[str] = set()
    if not isinstance(results, list) or len(results) < 3:
        errors.append("results 至少包含 3 条结果")
    else:
        result_run_ids = {str(result.get("benchmark_run_id") or "").strip() for result in results if isinstance(result, dict)}
        result_run_ids.discard("")
        if legacy_missing_run_id:
            if result_run_ids:
                errors.append("business_summary 未填写 benchmark_run_id 时，results[*].benchmark_run_id 也必须全部为空")
        elif not benchmark_run_id:
            errors.append("benchmark_run_id 必须是非空字符串")
        for idx, result in enumerate(results):
            if not isinstance(result, dict):
                errors.append(f"results[{idx}] 必须是 object")
                continue
            role = str(result.get("role") or "").strip()
            if role not in required_roles:
                errors.append(f"results[{idx}].role 无效: {role}")
            else:
                seen_roles.add(role)
            for field in ("artifact", "device", "device_model", "mode", "dtype", "dataset", "output_type", "throughput_metric_name", "quality_metric_name"):
                if not str(result.get(field) or "").strip():
                    errors.append(f"results[{idx}].{field} 必须是非空字符串")
            result_run_id = str(result.get("benchmark_run_id") or "").strip()
            if legacy_missing_run_id:
                if result_run_id:
                    errors.append(f"results[{idx}].benchmark_run_id 在 legacy summary 中必须为空")
            elif not result_run_id:
                errors.append(f"results[{idx}].benchmark_run_id 必须是非空字符串")
            latency_s = result.get("latency_s")
            if not _is_finite_number(latency_s) or float(latency_s) <= 0:
                errors.append(f"results[{idx}].latency_s 必须为正数")
            peak_memory_mb = result.get("peak_memory_mb")
            if peak_memory_mb is not None and (not _is_finite_number(peak_memory_mb) or float(peak_memory_mb) < 0):
                errors.append(f"results[{idx}].peak_memory_mb 必须为非负数或 null")
            throughput_metric_value = result.get("throughput_metric_value")
            if not _is_finite_number(throughput_metric_value) or float(throughput_metric_value) <= 0:
                errors.append(f"results[{idx}].throughput_metric_value 必须为正数")
            quality_metric_value = result.get("quality_metric_value")
            if quality_metric_value is not None and (not _is_finite_number(quality_metric_value) or float(quality_metric_value) < 0):
                errors.append(f"results[{idx}].quality_metric_value 必须为非负数或 null")
            errors.extend(
                _validate_bounded_quality_metric(
                    result.get("quality_metric_name"),
                    quality_metric_value,
                    f"results[{idx}].quality_metric_value",
                )
            )
            result_num_samples = result.get("num_samples")
            if not _is_finite_number(result_num_samples) or float(result_num_samples) <= MIN_BUSINESS_SAMPLE_LOWER_BOUND:
                errors.append(f"results[{idx}].num_samples 必须 > {MIN_BUSINESS_SAMPLE_LOWER_BOUND}")
            if contract_version >= 2:
                warmup_iterations = result.get("warmup_iterations")
                if not _is_finite_number(warmup_iterations) or float(warmup_iterations) < 0:
                    errors.append(f"results[{idx}].warmup_iterations 必须为非负整数")
                if not str(result.get("task_queue_enable") or "").strip():
                    errors.append(f"results[{idx}].task_queue_enable 必须是非空字符串")
                latency_scope = str(result.get("latency_measurement_scope") or "").strip()
                if not latency_scope:
                    errors.append(f"results[{idx}].latency_measurement_scope 必须是非空字符串")
                elif latency_scope not in VALID_LATENCY_MEASUREMENT_SCOPES:
                    errors.append(f"results[{idx}].latency_measurement_scope 无效: {latency_scope}")
        if seen_roles != required_roles:
            errors.append("results 必须同时包含 npu_baseline / npu_perf / cuda_baseline")

    best_result = summary.get("best_result")
    if not isinstance(best_result, dict):
        errors.append("best_result 必须是 object")
    else:
        if str(best_result.get("role") or "").strip() != "npu_perf":
            errors.append("best_result.role 必须为 npu_perf")
        for field in (
            "npu_speedup_ratio",
            "vs_cuda_latency_ratio",
            "npu_perf_throughput_metric_value",
            "cuda_baseline_throughput_metric_value",
        ):
            value = best_result.get(field)
            if not _is_finite_number(value) or float(value) <= 0:
                errors.append(f"best_result.{field} 必须为正数")
        for field in ("npu_perf_peak_memory_mb", "cuda_baseline_peak_memory_mb"):
            value = best_result.get(field)
            if value is not None and (not _is_finite_number(value) or float(value) < 0):
                errors.append(f"best_result.{field} 必须为非负数或 null")
        for field in ("output_type", "npu_perf_device_model", "cuda_baseline_device_model", "npu_perf_throughput_metric_name", "cuda_baseline_throughput_metric_name", "quality_metric_name"):
            if not str(best_result.get(field) or "").strip():
                errors.append(f"best_result.{field} 必须是非空字符串")
        quality_metric_value = best_result.get("quality_metric_value")
        if quality_metric_value is not None and (not _is_finite_number(quality_metric_value) or float(quality_metric_value) < 0):
            errors.append("best_result.quality_metric_value 必须为非负数或 null")
        errors.extend(
            _validate_bounded_quality_metric(
                best_result.get("quality_metric_name"),
                quality_metric_value,
                "best_result.quality_metric_value",
            )
        )
        npu_speedup_ratio = best_result.get("npu_speedup_ratio")
        if _is_finite_number(npu_speedup_ratio) and float(npu_speedup_ratio) < MIN_SANE_NPU_SPEEDUP_RATIO:
            errors.append(f"best_result.npu_speedup_ratio={float(npu_speedup_ratio):.6g} < {MIN_SANE_NPU_SPEEDUP_RATIO:.6g}；疑似 NPU perf 退化或 measurement 口径异常")
        vs_cuda_latency_ratio = best_result.get("vs_cuda_latency_ratio")
        if _is_finite_number(vs_cuda_latency_ratio) and float(vs_cuda_latency_ratio) < MIN_SANE_VS_CUDA_LATENCY_RATIO:
            errors.append(f"best_result.vs_cuda_latency_ratio={float(vs_cuda_latency_ratio):.6g} < {MIN_SANE_VS_CUDA_LATENCY_RATIO:.6g}；疑似跨设备对比口径异常")
        if contract_version >= 2:
            warmup_iterations = best_result.get("warmup_iterations")
            if not _is_finite_number(warmup_iterations) or float(warmup_iterations) < 0:
                errors.append("best_result.warmup_iterations 必须为非负整数")
            if not str(best_result.get("task_queue_enable") or "").strip():
                errors.append("best_result.task_queue_enable 必须是非空字符串")
            latency_scope = str(best_result.get("latency_measurement_scope") or "").strip()
            if not latency_scope:
                errors.append("best_result.latency_measurement_scope 必须是非空字符串")
            elif latency_scope not in VALID_LATENCY_MEASUREMENT_SCOPES:
                errors.append(f"best_result.latency_measurement_scope 无效: {latency_scope}")

    return errors


def _load_metric(metric_path: Path) -> tuple[dict | None, str | None]:
    try:
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
    except Exception as e:
        return None, f"读取 {metric_path.name} 失败: {e}"
    if not isinstance(metric, dict):
        return None, f"{metric_path.name} 必须是 JSON object"
    return metric, None


def _validate_metric(metric: dict, metric_path: Path, *, dataset: str, expected_device: str, benchmark_run_id: str, allow_legacy_missing_run_id: bool, adapt_dir: Path | None = None, role: str = "") -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_METRIC_FIELDS:
        if field not in metric:
            errors.append(f"{metric_path.name} 缺少字段 {field}")
    metric_run_id = str(metric.get("benchmark_run_id") or "").strip()
    if allow_legacy_missing_run_id:
        if metric_run_id:
            errors.append(f"{metric_path.name} 在 legacy 工件中不应带 benchmark_run_id；请重新 summarize 统一迁移")
    elif not metric_run_id:
        errors.append(f"{metric_path.name} 缺少非空 benchmark_run_id")
    elif benchmark_run_id and metric_run_id != benchmark_run_id:
        errors.append(f"{metric_path.name} 的 benchmark_run_id 与 business_summary.benchmark_run_id 不一致")
    num_samples = metric.get("num_samples")
    if _is_finite_number(num_samples):
        if float(num_samples) <= MIN_BUSINESS_SAMPLE_LOWER_BOUND:
            errors.append(f"{metric_path.name} 的 num_samples={num_samples}，必须 > {MIN_BUSINESS_SAMPLE_LOWER_BOUND}")
    else:
        errors.append(f"{metric_path.name} 缺少数值型 num_samples")
    latency_s = metric.get("latency_s")
    if not _is_finite_number(latency_s) or float(latency_s) <= 0:
        errors.append(f"{metric_path.name} 的 latency_s 必须为正数")
    else:
        start_dt = _parse_iso_datetime(metric.get("start_time"))
        end_dt = _parse_iso_datetime(metric.get("end_time"))
        num_samples_value = metric.get("num_samples")
        if start_dt is not None and end_dt is not None and _is_finite_number(num_samples_value) and float(num_samples_value) > 0:
            wall_clock_per_sample = (end_dt - start_dt).total_seconds() / float(num_samples_value)
            if float(latency_s) <= SUSPICIOUS_TINY_LATENCY_S and wall_clock_per_sample >= MIN_WALL_CLOCK_FOR_TINY_LATENCY_S:
                errors.append(f"{metric_path.name} 的 latency_s={float(latency_s):.6g}s 过小，但 start/end 推导的 wall-clock/sample={wall_clock_per_sample:.6g}s；疑似未真实执行模型推理或计时口径错误")
    if str(metric.get("dataset") or "").strip() != dataset:
        errors.append(f"{metric_path.name} 的 dataset 与 business_summary.dataset 不一致")
    if expected_device not in str(metric.get("device") or "").lower():
        errors.append(f"{metric_path.name} 的 device={metric.get('device')}，必须包含 {expected_device}")
    if not str(metric.get("device_model") or "").strip():
        errors.append(f"{metric_path.name} 缺少非空 device_model")
    peak_memory_mb = metric.get("peak_memory_mb")
    if peak_memory_mb is not None and (not _is_finite_number(peak_memory_mb) or float(peak_memory_mb) < 0):
        errors.append(f"{metric_path.name} 的 peak_memory_mb 必须为非负数或 null")
    measurement_contract_version = metric.get("measurement_contract_version")
    contract_version = 1
    if measurement_contract_version is not None:
        if not _is_finite_number(measurement_contract_version) or int(measurement_contract_version) < 1:
            errors.append(f"{metric_path.name} 的 measurement_contract_version 必须是 >=1 的整数")
        else:
            contract_version = int(measurement_contract_version)
    if contract_version >= 2:
        latency_measurement_scope = str(metric.get("latency_measurement_scope") or "").strip()
        if not latency_measurement_scope:
            errors.append(f"{metric_path.name} 缺少非空 latency_measurement_scope")
        elif latency_measurement_scope not in VALID_LATENCY_MEASUREMENT_SCOPES:
            errors.append(f"{metric_path.name} 的 latency_measurement_scope 无效: {latency_measurement_scope}")
        warmup_iterations = metric.get("warmup_iterations")
        if not _is_finite_number(warmup_iterations) or float(warmup_iterations) < 0:
            errors.append(f"{metric_path.name} 的 warmup_iterations 必须为非负整数")
        if not str(metric.get("task_queue_enable") or "").strip():
            errors.append(f"{metric_path.name} 缺少非空 task_queue_enable")
        if "loaded_from_model_files" not in metric:
            errors.append(f"{metric_path.name} 缺少 loaded_from_model_files")
        if not str(metric.get("model_source_kind") or "").strip():
            errors.append(f"{metric_path.name} 缺少非空 model_source_kind")
        if not str(metric.get("tokenizer_source_kind") or "").strip():
            errors.append(f"{metric_path.name} 缺少非空 tokenizer_source_kind")
        if role == "npu_perf" and adapt_dir is not None and _has_model_files_evidence(adapt_dir):
            loaded_from_model_files = metric.get("loaded_from_model_files")
            patch_load_status = str(metric.get("patch_load_status") or "").strip().lower()
            uses_model_files_patch = _metric_uses_model_files_patch(metric)
            if _should_require_model_files_in_perf(adapt_dir):
                if not uses_model_files_patch:
                    errors.append(f"{metric_path.name} 检测到 adaptation 存在 model_files，但 npu_perf 未体现有效 patch 继承（loaded_from_model_files={loaded_from_model_files}, patch_load_status={patch_load_status or 'empty'}）")
            elif uses_model_files_patch:
                errors.append(f"{metric_path.name} 对应模型已显式禁用第四阶段 model_files 继承（通常是 runtime_only），但当前工件仍显示 loaded_from_model_files={loaded_from_model_files}, patch_load_status={patch_load_status or 'empty'}")
    quality_name, quality_value = _detect_quality(metric)
    errors.extend(_validate_bounded_quality_metric(quality_name, quality_value, f"{metric_path.name}.{quality_name or 'quality_metric'}"))
    return errors


def _validate_quality_alignment(loaded_metrics: dict[str, dict]) -> list[str]:
    errors: list[str] = []
    role_values: dict[str, float] = {}
    role_metric_names: dict[str, str] = {}
    sample_counts: list[float] = []
    for role in ("npu_baseline", "npu_perf", "cuda_baseline"):
        metric = loaded_metrics.get(role)
        if not isinstance(metric, dict):
            return errors
        quality_name, quality_value = _detect_quality(metric)
        role_metric_names[role] = str(quality_name or "")
        if not _is_finite_number(quality_value):
            return errors
        role_values[role] = float(quality_value)
        num_samples = metric.get("num_samples")
        if _is_finite_number(num_samples) and float(num_samples) > 0:
            sample_counts.append(float(num_samples))
    if len(role_values) != 3 or not sample_counts:
        return errors

    normalized_names = {_normalized_quality_metric_name(name) for name in role_metric_names.values()}
    normalized_names.discard("")
    if len(normalized_names) != 1:
        if normalized_names:
            errors.append("业务测评三路 quality_metric_name 不一致：" + ", ".join(f"{role}={role_metric_names.get(role) or 'empty'}" for role in ("npu_baseline", "npu_perf", "cuda_baseline")))
        return errors
    metric_name = next(iter(normalized_names), "")
    quality_span = max(role_values.values()) - min(role_values.values())
    bounded_delta_tolerance = _bounded_quality_delta_tolerance(sample_counts)
    if _is_bounded_quality_metric(metric_name) and quality_span > bounded_delta_tolerance:
        errors.append(
            f"{metric_name} 三路结果差异过大：Δ={quality_span:.6g} > {bounded_delta_tolerance:.6g} (npu_baseline={role_values['npu_baseline']:.6g}, npu_perf={role_values['npu_perf']:.6g}, cuda_baseline={role_values['cuda_baseline']:.6g})"
        )
    if not _is_alignment_quality_metric(metric_name):
        return errors

    effective_samples = min(sample_counts)
    same_tol = _same_hardware_quality_tolerance(effective_samples)
    cross_tol = _cross_device_quality_tolerance(effective_samples)
    npu_baseline_quality = role_values["npu_baseline"]
    npu_perf_quality = role_values["npu_perf"]
    cuda_quality = role_values["cuda_baseline"]

    if max(role_values.values()) == 0.0:
        errors.append(f"{metric_name} 三路结果全部为 0.0；疑似 evaluator / 标签归一化 / 数据集画像异常，禁止 completed")

    if min(role_values.values()) == 0.0 and max(role_values.values()) >= same_tol:
        low_roles = sorted(role for role, value in role_values.items() if value == 0.0)
        high_roles = sorted(role for role, value in role_values.items() if value >= same_tol)
        errors.append(f"{metric_name} 出现 0 分塌陷：{', '.join(low_roles)}=0.0，但 {', '.join(high_roles)} 明显更高；疑似评分口径异常")

    if abs(npu_baseline_quality - npu_perf_quality) > same_tol:
        errors.append(f"{metric_name} 在 npu_baseline/npu_perf 间漂移过大：|{npu_baseline_quality:.6g}-{npu_perf_quality:.6g}|>{same_tol:.6g}")

    if abs(cuda_quality - npu_baseline_quality) > cross_tol and abs(cuda_quality - npu_perf_quality) > cross_tol:
        errors.append(f"{metric_name} 的 CUDA/NPU 三路结果漂移过大：cuda={cuda_quality:.6g}, npu_baseline={npu_baseline_quality:.6g}, npu_perf={npu_perf_quality:.6g}")

    return errors


def check_adapt_completed_business_gate(project_root: Path, adaptation_name: str) -> list[str]:
    adapt_dir = project_root / "adaptations" / adaptation_name
    summary_path = adapt_dir / "business_summary.json"
    if not summary_path.exists():
        return [f"缺少 business_summary.json: {summary_path}"]
    raw = summary_path.read_text(encoding="utf-8").strip()
    errors = validate_summary(raw)
    if errors:
        return errors
    summary = json.loads(raw)
    summary_contract_version = summary.get("measurement_contract_version")
    contract_version = int(summary_contract_version) if isinstance(summary_contract_version, (int, float)) and not isinstance(summary_contract_version, bool) else 1
    benchmark_run_id = str(summary.get("benchmark_run_id") or "").strip()
    legacy_missing_run_id = not benchmark_run_id
    dataset = str(summary["dataset"])
    config = _load_business_config(adapt_dir)
    expected_dataset = ""
    expected_run_id = ""
    expected_run_started_at = ""
    expected_eval_profile = ""
    expected_primary_metric = ""
    expected_output_type = ""
    if isinstance(config, dict):
        expected_dataset = str(config.get("dataset") or "").strip()
        expected_run_id = str(config.get("benchmark_run_id") or "").strip()
        expected_run_started_at = str(config.get("benchmark_run_started_at") or "").strip()
        expected_eval_profile = str(config.get("evaluation_profile") or "").strip()
        expected_primary_metric = str(config.get("primary_metric") or "").strip()
        expected_output_type = str(config.get("output_type_hint") or "").strip()
        errors.extend(_validate_non_english_asr_dataset_contract(config))
        if expected_dataset and expected_dataset != dataset:
            errors.append(f"business_summary.dataset={dataset} 与 business_benchmark_config.json.dataset={expected_dataset} 不一致")
        if expected_run_id and benchmark_run_id and expected_run_id != benchmark_run_id:
            errors.append("business_summary.benchmark_run_id 与 business_benchmark_config.json.benchmark_run_id 不一致")
        if expected_run_started_at:
            summary_run_started_at = str(summary.get("benchmark_run_started_at") or "").strip()
            if not summary_run_started_at:
                errors.append("business_summary 缺少 benchmark_run_started_at，但 business_benchmark_config.json 已记录当前轮次 start time")
            elif expected_run_started_at != summary_run_started_at:
                errors.append("business_summary.benchmark_run_started_at 与 business_benchmark_config.json.benchmark_run_started_at 不一致")
    results = summary.get("results") if isinstance(summary.get("results"), list) else []
    result_entries = {str(item.get("role") or "").strip(): item for item in results if isinstance(item, dict)}
    summary_output_type = ""
    if result_entries:
        output_types = {str(item.get("output_type") or "").strip() for item in result_entries.values()}
        output_types.discard("")
        if len(output_types) == 1:
            summary_output_type = output_types.pop()
        elif len(output_types) > 1:
            errors.append(f"business_summary.results 的 output_type 不一致: {sorted(output_types)}")

    artifact_specs = (
        ("npu_baseline", "npu_baseline_artifact", "npu"),
        ("npu_perf", "npu_perf_artifact", "npu"),
        ("cuda_baseline", "cuda_baseline_artifact", "cuda"),
    )
    loaded_metrics: dict[str, dict] = {}
    for role, key, expected_device in artifact_specs:
        metric_path = adapt_dir / str(summary[key])
        if not metric_path.exists():
            errors.append(f"缺少业务测评工件: {metric_path}")
            continue
        metric, metric_err = _load_metric(metric_path)
        if metric_err:
            errors.append(metric_err)
            continue
        assert metric is not None
        loaded_metrics[role] = metric
        if expected_eval_profile and str(metric.get("evaluation_profile") or "").strip() != expected_eval_profile:
            errors.append(f"{metric_path.name}.evaluation_profile 与 business_benchmark_config.json 不一致")
        if expected_primary_metric and str(metric.get("primary_metric") or "").strip() != expected_primary_metric:
            errors.append(f"{metric_path.name}.primary_metric 与 business_benchmark_config.json 不一致")
        if expected_output_type and str(metric.get("output_type") or "").strip() != expected_output_type:
            errors.append(f"{metric_path.name}.output_type 与 business_benchmark_config.json 不一致")
        quality_name, quality_value = _detect_quality(metric)
        if _quality_metric_required(metric):
            if quality_name == "quality_metric" or not _is_finite_number(quality_value):
                primary_metric = str(metric.get("primary_metric") or "").strip() or "quality_metric"
                errors.append(f"{metric_path.name} 缺少可用质量指标：primary_metric={primary_metric}, detected={quality_name}, value={quality_value!r}")
        errors.extend(
            _validate_metric(
                metric,
                metric_path,
                dataset=dataset,
                expected_device=expected_device,
                benchmark_run_id=benchmark_run_id,
                allow_legacy_missing_run_id=legacy_missing_run_id,
                adapt_dir=adapt_dir,
                role=role,
            )
        )
        role_entry = result_entries.get(role)
        if isinstance(role_entry, dict):
            role_run_id = str(role_entry.get("benchmark_run_id") or "").strip()
            metric_run_id = str(metric.get("benchmark_run_id") or "").strip()
            if legacy_missing_run_id:
                if role_run_id:
                    errors.append(f"results.{role}.benchmark_run_id 在 legacy summary 中必须为空")
                if metric_run_id:
                    errors.append(f"{metric_path.name} 在 legacy summary 中必须为空 benchmark_run_id")
            else:
                if role_run_id != benchmark_run_id:
                    errors.append(f"results.{role}.benchmark_run_id 与 business_summary.benchmark_run_id 不一致")
            if str(role_entry.get("output_type") or "").strip() != str(metric.get("output_type") or "").strip():
                errors.append(f"results.{role}.output_type 与 {metric_path.name} 不一致")
            if str(role_entry.get("device_model") or "").strip() != str(metric.get("device_model") or "").strip():
                errors.append(f"results.{role}.device_model 与 {metric_path.name} 不一致")
            peak_memory_mb = role_entry.get("peak_memory_mb")
            metric_peak_memory_mb = metric.get("peak_memory_mb")
            if _is_finite_number(peak_memory_mb) and _is_finite_number(metric_peak_memory_mb):
                if not _metric_close(float(metric_peak_memory_mb), float(peak_memory_mb)):
                    errors.append(f"results.{role}.peak_memory_mb 与 {metric_path.name} 不一致")
            throughput_name, throughput_value = _detect_throughput(metric)
            if str(role_entry.get("throughput_metric_name") or "").strip() != throughput_name:
                errors.append(f"results.{role}.throughput_metric_name 与 {metric_path.name} 不一致")
            entry_tp = role_entry.get("throughput_metric_value")
            if _is_finite_number(entry_tp) and throughput_value is not None and math.isfinite(float(throughput_value)):
                if not _metric_close(float(throughput_value), float(entry_tp)):
                    errors.append(f"results.{role}.throughput_metric_value 与 {metric_path.name} 不一致")
            quality_name, quality_value = _detect_quality(metric)
            if str(role_entry.get("quality_metric_name") or "").strip() != quality_name:
                errors.append(f"results.{role}.quality_metric_name 与 {metric_path.name} 不一致")
            entry_quality = role_entry.get("quality_metric_value")
            if quality_value is None:
                if entry_quality is not None:
                    errors.append(f"results.{role}.quality_metric_value 与 {metric_path.name} 不一致")
            elif _is_finite_number(entry_quality):
                if not _metric_close(float(quality_value), float(entry_quality)):
                    errors.append(f"results.{role}.quality_metric_value 与 {metric_path.name} 不一致")
            else:
                errors.append(f"results.{role}.quality_metric_value 与 {metric_path.name} 不一致")
            if contract_version >= 2:
                metric_warmup = metric.get("warmup_iterations")
                entry_warmup = role_entry.get("warmup_iterations")
                if _is_finite_number(metric_warmup) and _is_finite_number(entry_warmup):
                    if int(metric_warmup) != int(entry_warmup):
                        errors.append(f"results.{role}.warmup_iterations 与 {metric_path.name} 不一致")
                elif metric_warmup != entry_warmup:
                    errors.append(f"results.{role}.warmup_iterations 与 {metric_path.name} 不一致")
                if str(role_entry.get("task_queue_enable") or "").strip() != str(metric.get("task_queue_enable") or "").strip():
                    errors.append(f"results.{role}.task_queue_enable 与 {metric_path.name} 不一致")
                if str(role_entry.get("latency_measurement_scope") or "").strip() != str(metric.get("latency_measurement_scope") or "").strip():
                    errors.append(f"results.{role}.latency_measurement_scope 与 {metric_path.name} 不一致")

    if len(loaded_metrics) == 3:
        metric_output_types = {str(metric.get("output_type") or "").strip() for metric in loaded_metrics.values()}
        metric_output_types.discard("")
        if len(metric_output_types) != 1:
            errors.append(f"业务测评工件的 output_type 不一致: {sorted(metric_output_types)}")
        elif summary_output_type and summary_output_type not in metric_output_types:
            errors.append("business_summary.results.output_type 与业务测评工件不一致")
        errors.extend(_validate_quality_alignment(loaded_metrics))

        evidence = summary.get("comparison_evidence")
        if isinstance(evidence, dict):
            for role, device_key, peak_key, tp_name_key, tp_value_key in (
                ("npu_baseline", "npu_baseline_device_model", "npu_baseline_peak_memory_mb", "npu_baseline_throughput_metric_name", "npu_baseline_throughput_metric_value"),
                ("npu_perf", "npu_perf_device_model", "npu_perf_peak_memory_mb", "npu_perf_throughput_metric_name", "npu_perf_throughput_metric_value"),
                ("cuda_baseline", "cuda_baseline_device_model", "cuda_baseline_peak_memory_mb", "cuda_baseline_throughput_metric_name", "cuda_baseline_throughput_metric_value"),
            ):
                metric = loaded_metrics[role]
                if str(evidence.get(device_key) or "").strip() != str(metric.get("device_model") or "").strip():
                    errors.append(f"comparison_evidence.{device_key} 与 {role} 工件不一致")
                evidence_peak = evidence.get(peak_key)
                metric_peak = metric.get("peak_memory_mb")
                if _is_finite_number(evidence_peak) and _is_finite_number(metric_peak):
                    if not _metric_close(float(metric_peak), float(evidence_peak)):
                        errors.append(f"comparison_evidence.{peak_key} 与 {role} 工件不一致")
                throughput_name, throughput_value = _detect_throughput(metric)
                if str(evidence.get(tp_name_key) or "").strip() != throughput_name:
                    errors.append(f"comparison_evidence.{tp_name_key} 与 {role} 工件不一致")
                evidence_tp = evidence.get(tp_value_key)
                if _is_finite_number(evidence_tp) and throughput_value is not None and math.isfinite(float(throughput_value)):
                    if not _metric_close(float(throughput_value), float(evidence_tp)):
                        errors.append(f"comparison_evidence.{tp_value_key} 与 {role} 工件不一致")
            for role, name_key, value_key in (
                ("npu_baseline", "npu_baseline_quality_metric_name", "npu_baseline_quality_metric_value"),
                ("npu_perf", "npu_perf_quality_metric_name", "npu_perf_quality_metric_value"),
                ("cuda_baseline", "cuda_baseline_quality_metric_name", "cuda_baseline_quality_metric_value"),
            ):
                quality_name, quality_value = _detect_quality(loaded_metrics[role])
                if str(evidence.get(name_key) or "").strip() != quality_name:
                    errors.append(f"comparison_evidence.{name_key} 与 {role} 工件不一致")
                evidence_quality = evidence.get(value_key)
                if quality_value is None:
                    if evidence_quality is not None:
                        errors.append(f"comparison_evidence.{value_key} 与 {role} 工件不一致")
                elif _is_finite_number(evidence_quality):
                    if not _metric_close(float(quality_value), float(evidence_quality)):
                        errors.append(f"comparison_evidence.{value_key} 与 {role} 工件不一致")
                else:
                    errors.append(f"comparison_evidence.{value_key} 与 {role} 工件不一致")
            if contract_version >= 2:
                for role, warmup_key, tqe_key in (
                    ("npu_baseline", "npu_baseline_warmup_iterations", "npu_baseline_task_queue_enable"),
                    ("npu_perf", "npu_perf_warmup_iterations", "npu_perf_task_queue_enable"),
                    ("cuda_baseline", "cuda_baseline_warmup_iterations", "cuda_baseline_task_queue_enable"),
                ):
                    metric = loaded_metrics[role]
                    metric_warmup = metric.get("warmup_iterations")
                    evidence_warmup = evidence.get(warmup_key)
                    if _is_finite_number(metric_warmup) and _is_finite_number(evidence_warmup):
                        if int(metric_warmup) != int(evidence_warmup):
                            errors.append(f"comparison_evidence.{warmup_key} 与 {role} 工件不一致")
                    elif metric_warmup != evidence_warmup:
                        errors.append(f"comparison_evidence.{warmup_key} 与 {role} 工件不一致")
                    if str(evidence.get(tqe_key) or "").strip() != str(metric.get("task_queue_enable") or "").strip():
                        errors.append(f"comparison_evidence.{tqe_key} 与 {role} 工件不一致")

        npu_perf_metric = loaded_metrics["npu_perf"]
        cuda_metric = loaded_metrics["cuda_baseline"]
        best_result = summary.get("best_result")
        if isinstance(best_result, dict):
            npu_baseline_metric = loaded_metrics["npu_baseline"]
            expected_vs_cuda = float(cuda_metric["latency_s"]) / float(npu_perf_metric["latency_s"])
            actual_vs_cuda = best_result.get("vs_cuda_latency_ratio")
            if not _is_finite_number(actual_vs_cuda) or not _metric_close(expected_vs_cuda, float(actual_vs_cuda)):
                errors.append("best_result.vs_cuda_latency_ratio 与 CUDA baseline / NPU perf 工件不一致")
            elif float(actual_vs_cuda) < AUTO_MAPPING_RECHECK_VS_CUDA_LATENCY_RATIO:
                auto_mapping_roles = []
                if _uses_auto_npu_mapping(npu_baseline_metric):
                    auto_mapping_roles.append("npu_baseline")
                if _uses_auto_npu_mapping(npu_perf_metric):
                    auto_mapping_roles.append("npu_perf")
                if auto_mapping_roles:
                    errors.append(
                        f"best_result.vs_cuda_latency_ratio={float(actual_vs_cuda):.6g} < {AUTO_MAPPING_RECHECK_VS_CUDA_LATENCY_RATIO:.6g}，且 {', '.join(auto_mapping_roles)} 仍使用 auto/all_visible NPU 映射；请先固定 1-die 或 2-die 的 ASCEND_RT_VISIBLE_DEVICES 重跑后再尝试 completed"
                    )
            npu_baseline_wall_clock_s = _resolve_metric_wall_clock_s(npu_baseline_metric)
            npu_perf_wall_clock_s = _resolve_metric_wall_clock_s(npu_perf_metric)
            if (
                npu_baseline_wall_clock_s is not None
                and npu_baseline_wall_clock_s > 0
                and npu_perf_wall_clock_s is not None
                and npu_perf_wall_clock_s > 0
            ):
                expected_npu_speedup = npu_baseline_wall_clock_s / npu_perf_wall_clock_s
            else:
                expected_npu_speedup = float(npu_baseline_metric["latency_s"]) / float(npu_perf_metric["latency_s"])
            actual_npu_speedup = best_result.get("npu_speedup_ratio")
            if not _is_finite_number(actual_npu_speedup) or not _metric_close(expected_npu_speedup, float(actual_npu_speedup)):
                errors.append("best_result.npu_speedup_ratio 与 NPU baseline / NPU perf 工件不一致")
            if str(best_result.get("output_type") or "").strip() != str(npu_perf_metric.get("output_type") or "").strip():
                errors.append("best_result.output_type 与 npu_perf 工件不一致")
            quality_name, quality_value = _detect_quality(npu_perf_metric)
            if str(best_result.get("quality_metric_name") or "").strip() != quality_name:
                errors.append("best_result.quality_metric_name 与 npu_perf 工件不一致")
            best_quality = best_result.get("quality_metric_value")
            if quality_value is None:
                if best_quality is not None:
                    errors.append("best_result.quality_metric_value 与 npu_perf 工件不一致")
            elif _is_finite_number(best_quality):
                if not _metric_close(float(quality_value), float(best_quality)):
                    errors.append("best_result.quality_metric_value 与 npu_perf 工件不一致")
            else:
                errors.append("best_result.quality_metric_value 与 npu_perf 工件不一致")
            if contract_version >= 2:
                best_warmup = best_result.get("warmup_iterations")
                metric_warmup = npu_perf_metric.get("warmup_iterations")
                if _is_finite_number(best_warmup) and _is_finite_number(metric_warmup):
                    if int(best_warmup) != int(metric_warmup):
                        errors.append("best_result.warmup_iterations 与 npu_perf 工件不一致")
                elif best_warmup != metric_warmup:
                    errors.append("best_result.warmup_iterations 与 npu_perf 工件不一致")
                if str(best_result.get("task_queue_enable") or "").strip() != str(npu_perf_metric.get("task_queue_enable") or "").strip():
                    errors.append("best_result.task_queue_enable 与 npu_perf 工件不一致")
                if str(best_result.get("latency_measurement_scope") or "").strip() != str(npu_perf_metric.get("latency_measurement_scope") or "").strip():
                    errors.append("best_result.latency_measurement_scope 与 npu_perf 工件不一致")
    return errors


def _resolve_wait_cuda_metric(adapt_dir: Path, *, role: str, benchmark_run_id: str) -> tuple[Path | None, dict | None, str | None]:
    if role == "npu_baseline":
        pattern = "business_metrics_npu_*_baseline.json"
    elif role == "npu_perf":
        pattern = "business_metrics_npu_*_perf.json"
    else:
        raise ValueError(f"不支持的 wait_cuda role: {role}")

    matches: list[tuple[Path, dict]] = []
    for metric_path in sorted(adapt_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True):
        metric, metric_err = _load_metric(metric_path)
        if metric is None or metric_err:
            continue
        if str(metric.get("scenario") or "").strip() != role:
            continue
        metric_run_id = str(metric.get("benchmark_run_id") or "").strip()
        if benchmark_run_id:
            if metric_run_id == benchmark_run_id:
                matches.append((metric_path, metric))
        else:
            matches.append((metric_path, metric))

    if matches:
        metric_path, metric = matches[0]
        return metric_path, metric, None
    if benchmark_run_id:
        return None, None, f"缺少当前轮次 {benchmark_run_id} 的 {role} 工件；禁止进入 wait_cuda"
    return None, None, f"缺少 {role} 工件；禁止进入 wait_cuda"


def _validate_wait_cuda_npu_quality_alignment(loaded_metrics: dict[str, dict]) -> list[str]:
    errors: list[str] = []
    role_values: dict[str, float] = {}
    role_metric_names: dict[str, str] = {}
    sample_counts: list[float] = []
    for role in ("npu_baseline", "npu_perf"):
        metric = loaded_metrics.get(role)
        if not isinstance(metric, dict):
            return errors
        quality_name, quality_value = _detect_quality(metric)
        role_metric_names[role] = str(quality_name or "")
        if _quality_metric_required(metric) and not _is_finite_number(quality_value):
            errors.append(f"{role} 缺少可用 quality metric；wait_cuda 前必须先确认本机 NPU 两路输出正常")
            continue
        if not _is_finite_number(quality_value):
            continue
        role_values[role] = float(quality_value)
        num_samples = metric.get("num_samples")
        if _is_finite_number(num_samples) and float(num_samples) > 0:
            sample_counts.append(float(num_samples))
    if errors or len(role_values) != 2 or not sample_counts:
        return errors

    normalized_names = {_normalized_quality_metric_name(name) for name in role_metric_names.values()}
    normalized_names.discard("")
    if len(normalized_names) != 1:
        if normalized_names:
            errors.append(
                "wait_cuda 前本机 NPU 双路 quality_metric_name 不一致："
                + ", ".join(f"{role}={role_metric_names.get(role) or 'empty'}" for role in ("npu_baseline", "npu_perf"))
            )
        return errors
    metric_name = next(iter(normalized_names), "")
    quality_span = max(role_values.values()) - min(role_values.values())
    bounded_delta_tolerance = _bounded_quality_delta_tolerance(sample_counts)
    if _is_bounded_quality_metric(metric_name) and quality_span > bounded_delta_tolerance:
        errors.append(
            f"wait_cuda 前本机 NPU 双路 {metric_name} 差异过大：Δ={quality_span:.6g} > {bounded_delta_tolerance:.6g} "
            f"(npu_baseline={role_values['npu_baseline']:.6g}, npu_perf={role_values['npu_perf']:.6g})"
        )
    if _is_bounded_quality_metric(metric_name):
        max_quality = max(role_values.values())
        min_quality = min(role_values.values())
        if max_quality == 0.0:
            errors.append(f"wait_cuda 前本机 NPU 双路 {metric_name} 全为 0.0；疑似 evaluator / 标签归一化 / 数据集画像异常，禁止进入 wait_cuda")
        elif min_quality == 0.0 and max_quality >= bounded_delta_tolerance:
            low_roles = sorted(role for role, value in role_values.items() if value == 0.0)
            high_roles = sorted(role for role, value in role_values.items() if value >= bounded_delta_tolerance)
            errors.append(f"wait_cuda 前本机 NPU 双路 {metric_name} 出现 0 分塌陷：{', '.join(low_roles)}=0.0，但 {', '.join(high_roles)} 明显更高")
    baseline_latency = loaded_metrics["npu_baseline"].get("latency_s")
    perf_latency = loaded_metrics["npu_perf"].get("latency_s")
    if _is_finite_number(baseline_latency) and _is_finite_number(perf_latency) and float(perf_latency) > 0:
        npu_speedup_ratio = float(baseline_latency) / float(perf_latency)
        if npu_speedup_ratio < MIN_SANE_NPU_SPEEDUP_RATIO:
            errors.append(
                f"wait_cuda 前本机 NPU 双路 npu_speedup_ratio={npu_speedup_ratio:.6g} < {MIN_SANE_NPU_SPEEDUP_RATIO:.6g}；疑似 NPU perf 退化或 measurement 口径异常"
            )
    if not _is_alignment_quality_metric(metric_name):
        return errors

    effective_samples = min(sample_counts)
    same_tol = _same_hardware_quality_tolerance(effective_samples)
    npu_baseline_quality = role_values["npu_baseline"]
    npu_perf_quality = role_values["npu_perf"]
    if max(role_values.values()) == 0.0:
        errors.append(f"wait_cuda 前本机 NPU 双路 {metric_name} 全为 0.0；疑似 evaluator / 标签归一化 / 数据集画像异常，禁止进入 wait_cuda")
    if min(role_values.values()) == 0.0 and max(role_values.values()) >= same_tol:
        low_roles = sorted(role for role, value in role_values.items() if value == 0.0)
        high_roles = sorted(role for role, value in role_values.items() if value >= same_tol)
        errors.append(f"wait_cuda 前本机 NPU 双路 {metric_name} 出现 0 分塌陷：{', '.join(low_roles)}=0.0，但 {', '.join(high_roles)} 明显更高")
    if abs(npu_baseline_quality - npu_perf_quality) > same_tol:
        errors.append(f"wait_cuda 前本机 NPU 双路 {metric_name} 漂移过大：|{npu_baseline_quality:.6g}-{npu_perf_quality:.6g}|>{same_tol:.6g}")
    return errors


def _validate_non_english_asr_dataset_contract(config: dict) -> list[str]:
    model_id = str(config.get("model_id") or "").strip()
    if not model_id:
        return []
    multilingual_asr = resolve_multilingual_asr_dataset(model_id)
    if not multilingual_asr:
        return []
    errors: list[str] = []
    model_type = str(config.get("model_type") or "").strip().lower()
    if model_type != "asr":
        errors.append(f"非英语 ASR 模型 {model_id} 的 business_benchmark_config.json 必须保持 model_type=asr，当前为 {model_type or '<empty>'}")
    expected_dataset = str(multilingual_asr["dataset_key"]).strip().lower()
    actual_dataset = str(config.get("dataset") or "").strip().lower()
    if actual_dataset != expected_dataset:
        errors.append(f"非英语 ASR 模型 {model_id} 应使用 dataset={expected_dataset}，当前为 {actual_dataset or '<empty>'}")
    expected_asr_language = str(multilingual_asr.get("asr_language") or "").strip().lower()
    actual_asr_language = str(config.get("asr_language") or "").strip().lower()
    if actual_asr_language and expected_asr_language and actual_asr_language != expected_asr_language:
        errors.append(f"非英语 ASR 模型 {model_id} 应使用 asr_language={expected_asr_language}，当前为 {actual_asr_language}")
    return errors


def check_adapt_wait_cuda_npu_gate(project_root: Path, adaptation_name: str) -> list[str]:
    adapt_dir = project_root / "adaptations" / adaptation_name
    if not adapt_dir.exists():
        return [f"未找到 adaptation 目录: {adapt_dir}"]

    config = _load_business_config(adapt_dir)
    if not isinstance(config, dict):
        return [f"缺少 business_benchmark_config.json: {adapt_dir / 'business_benchmark_config.json'}"]
    dataset = str(config.get("dataset") or "").strip()
    if not dataset:
        return ["business_benchmark_config.json.dataset 不能为空；wait_cuda 前无法确认本机 NPU 双路口径"]
    benchmark_run_id = str(config.get("benchmark_run_id") or "").strip()
    allow_legacy_missing_run_id = not benchmark_run_id

    errors: list[str] = _validate_non_english_asr_dataset_contract(config)
    loaded_metrics: dict[str, dict] = {}
    for role in ("npu_baseline", "npu_perf"):
        metric_path, metric, metric_err = _resolve_wait_cuda_metric(adapt_dir, role=role, benchmark_run_id=benchmark_run_id)
        if metric_path is None or metric is None:
            errors.append(metric_err or f"缺少 {role} 工件；禁止进入 wait_cuda")
            continue
        loaded_metrics[role] = metric
        errors.extend(
            _validate_metric(
                metric,
                metric_path,
                dataset=dataset,
                expected_device="npu",
                benchmark_run_id=benchmark_run_id,
                allow_legacy_missing_run_id=allow_legacy_missing_run_id,
                adapt_dir=adapt_dir,
                role=role,
            )
        )

    if len(loaded_metrics) == 2:
        errors.extend(_validate_wait_cuda_npu_quality_alignment(loaded_metrics))
    return errors


def get_business_completed_adapt_names(project_root: Path) -> list[str]:
    db_path = project_root / "board.db"
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        "SELECT adaptation_path FROM models WHERE business_benchmark_status = ? AND adaptation_path IS NOT NULL AND adaptation_path != ''",
        ("completed",),
    )
    paths = [r[0] for r in cur.fetchall()]
    conn.close()
    names = []
    for path in paths:
        path = path.strip().strip("/")
        if path.startswith("adaptations/"):
            names.append(path.replace("adaptations/", ""))
        else:
            names.append(path)
    return sorted(set(names))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="检查 business_summary.json 与业务测评工件")
    parser.add_argument("--adapt", default=None, help="仅检查指定 adaptation 目录名")
    parser.add_argument("--warn-only", action="store_true")
    parser.add_argument("--db-only", action="store_true", help="仅检查 board.db 中 business_benchmark_status=completed 的 adaptation")
    parser.add_argument("--wait-cuda-npu-only", action="store_true", help="仅检查进入 wait_cuda 前的本机 NPU baseline/perf sanity gate")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    adaptations_dir = project_root / "adaptations"

    if args.adapt:
        names = [args.adapt]
    elif args.db_only:
        names = get_business_completed_adapt_names(project_root)
        if not names:
            print("[check] board.db 中无 business_benchmark_status=completed 的 adaptation，跳过检查")
            return 0
    else:
        names = sorted(path.name for path in adaptations_dir.iterdir() if path.is_dir() and (path / "business_summary.json").exists())

    all_errors: list[str] = []
    for name in names:
        errors = check_adapt_wait_cuda_npu_gate(project_root, name) if args.wait_cuda_npu_only else check_adapt_completed_business_gate(project_root, name)
        if errors:
            print(f"[check] ❌ adaptations/{name}")
            for err in errors:
                print(f"       - {err}")
            all_errors.extend(f"{name}: {err}" for err in errors)
        else:
            if args.wait_cuda_npu_only:
                print(f"[check] ✅ adaptations/{name} (wait_cuda NPU gate)")
            else:
                print(f"[check] ✅ adaptations/{name}")

    if all_errors:
        print(f"\n[check] 共 {len(all_errors)} 项违规")
        return 0 if args.warn_only else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
