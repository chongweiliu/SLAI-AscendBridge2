#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os
import socket
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ADAPTATIONS_DIR = PROJECT_ROOT / "adaptations"
DEFAULT_REMOTE_PROJECT_ROOT_ENV_KEY = "SLAI_REMOTE_PROJECT_ROOT"
REMOTE_PROJECT_ROOT_PLACEHOLDER = f"${DEFAULT_REMOTE_PROJECT_ROOT_ENV_KEY}"
DEFAULT_REMOTE_PROJECT_BASE = str(os.environ.get("SLAI_REMOTE_PROJECT_BASE") or "/workspace").strip().rstrip("/")
DEFAULT_REMOTE_PROJECT_OWNER = str(os.environ.get("SLAI_REMOTE_PROJECT_OWNER") or "").strip()


def _default_remote_project_root() -> str:
    env_value = str(os.environ.get("SLAI_REMOTE_PROJECT_ROOT") or "").strip().rstrip("/")
    if env_value:
        return env_value
    repo_name = PROJECT_ROOT.name
    if DEFAULT_REMOTE_PROJECT_OWNER:
        return f"{DEFAULT_REMOTE_PROJECT_BASE}/{DEFAULT_REMOTE_PROJECT_OWNER}/{repo_name}"
    return f"{DEFAULT_REMOTE_PROJECT_BASE}/{repo_name}"


DEFAULT_REMOTE_PROJECT_ROOT = _default_remote_project_root()
VALID_COMPARISON_SCOPES = {"real_business", "cold_start", "steady_state", "mixed"}
ARTIFACT_PATTERNS = {
    "npu_baseline": "business_metrics_npu_*_baseline.json",
    "npu_perf": "business_metrics_npu_*_perf.json",
    "cuda_baseline": "business_metrics_cuda_*_baseline.json",
}
QUALITY_VALUE_KEYS = ("exact_match", "f1", "accuracy", "top1_accuracy", "rougeL", "ndcg_at_10", "mAP", "map50", "match_rate", "text_match_rate", "wer", "mrr", "perplexity")
BOUNDED_QUALITY_METRIC_KEYS = {"exact_match", "f1", "accuracy", "top1_accuracy", "rougel", "ndcg_at_10", "map", "map50", "match_rate", "text_match_rate", "mrr", "cosine_similarity"}


def _load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} 必须是 JSON object")
    return data


def _portable_path_text(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    expanded = os.path.expandvars(text).strip()
    candidate = Path(expanded).expanduser()
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        except Exception:
            marker = f"/{PROJECT_ROOT.name}"
            if expanded.startswith(f"{DEFAULT_REMOTE_PROJECT_BASE}/") and marker in expanded:
                _, suffix = expanded.split(marker, 1)
                suffix = suffix.lstrip("/")
                return REMOTE_PROJECT_ROOT_PLACEHOLDER if not suffix else f"{REMOTE_PROJECT_ROOT_PLACEHOLDER}/{suffix}"
    return text


def _resolve_remote_project_root(value: object) -> str:
    text = str(value or "").strip().rstrip("/")
    if not text or text == REMOTE_PROJECT_ROOT_PLACEHOLDER:
        return DEFAULT_REMOTE_PROJECT_ROOT
    expanded = os.path.expandvars(text).strip().rstrip("/")
    if not expanded or expanded.startswith("$"):
        return DEFAULT_REMOTE_PROJECT_ROOT
    return expanded


def _parse_iso_datetime(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalize_quality_value(metric_name: object, value: object) -> float | None:
    if not _is_number(value):
        return None
    numeric = float(value)
    if str(metric_name or "").strip().lower() in BOUNDED_QUALITY_METRIC_KEYS:
        return max(0.0, min(1.0, numeric))
    return numeric


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


def _write_json(path: Path, payload: dict) -> None:
    content = json.dumps(_with_generation_metadata(payload), ensure_ascii=False, indent=2) + "\n"
    if (not path.exists()) or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")


def _best_effort_host_ip() -> str:
    candidate_ips: list[str] = []
    try:
        host_infos = socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET, type=socket.SOCK_DGRAM)
        for info in host_infos:
            host_ip = str(info[4][0]).strip()
            if host_ip and host_ip not in candidate_ips:
                candidate_ips.append(host_ip)
    except OSError:
        pass

    for probe_target in ("10.255.255.255", "8.8.8.8"):
        sock: socket.socket | None = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect((probe_target, 1))
            host_ip = str(sock.getsockname()[0]).strip()
            if host_ip and not host_ip.startswith("127."):
                return host_ip
            if host_ip and host_ip not in candidate_ips:
                candidate_ips.append(host_ip)
        except OSError:
            continue
        finally:
            if sock is not None:
                sock.close()

    for host_ip in candidate_ips:
        if host_ip and not host_ip.startswith("127."):
            return host_ip
    return candidate_ips[0] if candidate_ips else "127.0.0.1"


def _with_generation_metadata(payload: dict) -> dict:
    enriched = dict(payload)
    enriched.update(
        {
            "generated_at": datetime.now().isoformat(),
            "generated_by_tool": "business_benchmark/scripts/business_benchmark_tool.py",
            "generated_by_user": getpass.getuser(),
            "generated_by_hostname": socket.gethostname(),
            "generated_by_host_ip": _best_effort_host_ip(),
            "generated_by_pid": os.getpid(),
        }
    )
    return enriched


def _load_summary_artifact_selection(adapt_dir: Path) -> tuple[dict[str, Path], dict[str, dict]] | None:
    summary_path = adapt_dir / "business_summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = _load_json(summary_path)
    except Exception:
        return None
    artifact_paths: dict[str, Path] = {}
    metrics_by_role: dict[str, dict] = {}
    for role, field in (
        ("npu_baseline", "npu_baseline_artifact"),
        ("npu_perf", "npu_perf_artifact"),
        ("cuda_baseline", "cuda_baseline_artifact"),
    ):
        artifact_name = str(summary.get(field) or "").strip()
        if not artifact_name:
            return None
        artifact_path = adapt_dir / artifact_name
        if not artifact_path.exists():
            return None
        artifact_paths[role] = artifact_path
        metrics_by_role[role] = _load_json(artifact_path)
    return artifact_paths, metrics_by_role


def _select_group_artifacts(group: dict[str, tuple[Path, dict]], *, required_roles: tuple[str, ...]) -> tuple[dict[str, Path], dict[str, dict]]:
    artifact_paths = {role: group[role][0] for role in required_roles}
    metrics_by_role = {role: group[role][1] for role in required_roles}
    if "cuda_baseline" in group:
        artifact_paths["cuda_baseline"] = group["cuda_baseline"][0]
        metrics_by_role["cuda_baseline"] = group["cuda_baseline"][1]
    return artifact_paths, metrics_by_role


def _select_artifacts_for_summary(adapt_dir: Path) -> tuple[dict[str, Path], dict[str, dict]]:
    candidates_by_role: dict[str, list[tuple[Path, dict]]] = {}
    required_roles = ("npu_baseline", "npu_perf")
    missing_roles = [role for role in required_roles if not list(adapt_dir.glob(ARTIFACT_PATTERNS[role]))]
    if missing_roles:
        raise FileNotFoundError(f"缺少业务测评工件: {', '.join(missing_roles)}")

    for role, pattern in ARTIFACT_PATTERNS.items():
        paths = sorted(adapt_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if paths:
            candidates_by_role[role] = [(path, _load_json(path)) for path in paths]

    groups_by_run_id: dict[str, dict[str, tuple[Path, dict]]] = {}
    for role, candidates in candidates_by_role.items():
        for path, metric in candidates:
            run_id = str(metric.get("benchmark_run_id") or "").strip()
            if not run_id:
                continue
            group = groups_by_run_id.setdefault(run_id, {})
            current = group.get(role)
            if current is None or path.stat().st_mtime > current[0].stat().st_mtime:
                group[role] = (path, metric)

    candidate_groups = {run_id: group for run_id, group in groups_by_run_id.items() if all(role in group for role in required_roles)}
    if candidate_groups:
        selected_run_id, selected_group = max(
            candidate_groups.items(),
            key=lambda item: max(candidate[0].stat().st_mtime for candidate in item[1].values()),
        )
        del selected_run_id
        return _select_group_artifacts(selected_group, required_roles=required_roles)

    latest_npu_selection = (
        {role: candidates_by_role[role][0][0] for role in required_roles},
        {role: candidates_by_role[role][0][1] for role in required_roles},
    )

    existing_summary_selection = _load_summary_artifact_selection(adapt_dir)
    if existing_summary_selection is not None:
        existing_paths, _ = existing_summary_selection
        if all(existing_paths.get(role) == latest_npu_selection[0].get(role) for role in required_roles):
            return existing_summary_selection

    return latest_npu_selection


def _derive_legacy_benchmark_run_id(adapt_dir: Path, artifact_paths: dict[str, Path]) -> str:
    latest_mtime = max(path.stat().st_mtime for path in artifact_paths.values())
    timestamp = datetime.fromtimestamp(latest_mtime).strftime("%Y%m%dT%H%M%S")
    return f"legacy-{adapt_dir.name}-{timestamp}"


def _resolve_benchmark_run_id(adapt_dir: Path, artifact_paths: dict[str, Path], metrics_by_role: dict[str, dict]) -> tuple[str, dict[str, dict]]:
    observed_run_ids = {role: str(metric.get("benchmark_run_id") or "").strip() for role, metric in metrics_by_role.items()}
    present_run_ids = {run_id for run_id in observed_run_ids.values() if run_id}
    roles_with_run_id = {role for role, run_id in observed_run_ids.items() if run_id}
    if len(present_run_ids) > 1:
        raise ValueError(f"业务测评工件的benchmark_run_id不一致: {sorted(present_run_ids)}")
    if roles_with_run_id and len(roles_with_run_id) < len(observed_run_ids):
        missing_roles = sorted(role for role, run_id in observed_run_ids.items() if not run_id)
        raise ValueError(f"业务测评工件处于新旧混合状态，以下角色缺少 benchmark_run_id: {missing_roles}；请补齐该轮工件后再 summarize")

    summary_run_id = ""
    summary_path = adapt_dir / "business_summary.json"
    if summary_path.exists():
        try:
            summary = _load_json(summary_path)
        except Exception:
            summary = {}
        summary_run_id = str(summary.get("benchmark_run_id") or "").strip()

    benchmark_run_id = next(iter(present_run_ids), "") or summary_run_id or _derive_legacy_benchmark_run_id(adapt_dir, artifact_paths)
    repaired_metrics: dict[str, dict] = {}
    for role, metric in metrics_by_role.items():
        current_run_id = str(metric.get("benchmark_run_id") or "").strip()
        if current_run_id and current_run_id != benchmark_run_id:
            raise ValueError(f"{artifact_paths[role].name} 的 benchmark_run_id 与当前汇总轮次不一致")
        if current_run_id != benchmark_run_id:
            updated_metric = dict(metric)
            updated_metric["benchmark_run_id"] = benchmark_run_id
            _write_json(artifact_paths[role], updated_metric)
            repaired_metrics[role] = updated_metric
        else:
            repaired_metrics[role] = metric
    return benchmark_run_id, repaired_metrics


def _detect_quality(metric: dict) -> tuple[str, float | None]:
    compare = metric.get("output_compare")
    primary_metric = str(metric.get("primary_metric") or "").strip()
    if isinstance(compare, dict):
        primary_value = _lookup_numeric_metric(compare, primary_metric)
        if primary_metric and primary_value is not None:
            return primary_metric, _normalize_quality_value(primary_metric, primary_value)
        for key in ("avg_cosine_similarity", "logits_avg_cosine_similarity", "cosine_similarity"):
            value = compare.get(key)
            if _is_number(value):
                return "cosine_similarity", _normalize_quality_value("cosine_similarity", value)
        for key in QUALITY_VALUE_KEYS:
            value = compare.get(key)
            if _is_number(value):
                return key, _normalize_quality_value(key, value)
    primary_value = _lookup_numeric_metric(metric, primary_metric)
    if primary_metric and primary_value is not None:
        return primary_metric, _normalize_quality_value(primary_metric, primary_value)
    for key in ("cosine_similarity", *QUALITY_VALUE_KEYS):
        value = metric.get(key)
        if _is_number(value):
            return key, _normalize_quality_value(key, value)
    return "quality_metric", None


def _detect_throughput(metric: dict) -> tuple[str, float | None]:
    for key in ("throughput_qps", "throughput", "samples_per_second", "qps", "items_per_second"):
        value = metric.get(key)
        if _is_number(value) and float(value) > 0:
            return key, float(value)
    latency_s = metric.get("latency_s")
    if _is_number(latency_s) and float(latency_s) > 0:
        return "derived_from_latency_qps", 1.0 / float(latency_s)
    return "throughput", None


def _as_number_or_none(value):
    if _is_number(value):
        return float(value)
    return None


def _as_int_or_none(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None


def _resolve_metric_wall_clock_s(metric: dict) -> tuple[float | None, str]:
    explicit_wall_clock_s = _as_number_or_none(metric.get("wall_clock_s"))
    if explicit_wall_clock_s is not None and explicit_wall_clock_s > 0:
        return explicit_wall_clock_s, "artifact_explicit_field"

    total_duration_s = _as_number_or_none(metric.get("total_duration_s"))
    if total_duration_s is not None and total_duration_s > 0:
        return total_duration_s, "artifact_explicit_field"

    start_dt = _parse_iso_datetime(metric.get("start_time"))
    end_dt = _parse_iso_datetime(metric.get("end_time"))
    if start_dt is None or end_dt is None or end_dt < start_dt:
        return None, ""

    derived_wall_clock_s = (end_dt - start_dt).total_seconds()
    if derived_wall_clock_s <= 0:
        return None, ""
    return derived_wall_clock_s, "artifact_timestamps"


def _ensure_metric_consistency(metrics: dict[str, dict], *, field: str, label: str) -> str:
    values = {str(metric.get(field) or "").strip() for metric in metrics.values()}
    values.discard("")
    if not values:
        raise ValueError(f"无法从业务测评工件中推断 {field}")
    if len(values) != 1:
        raise ValueError(f"业务测评工件的{label}不一致: {sorted(values)}")
    return values.pop()


def _metric_to_result(role: str, artifact_path: Path, metric: dict) -> dict:
    quality_name, quality_value = _detect_quality(metric)
    throughput_name, throughput_value = _detect_throughput(metric)
    wall_clock_s, wall_clock_source = _resolve_metric_wall_clock_s(metric)
    return {
        "role": role,
        "artifact": artifact_path.name,
        "benchmark_run_id": metric.get("benchmark_run_id", ""),
        "device": metric.get("device", ""),
        "device_model": metric.get("device_model", ""),
        "mode": metric.get("mode", ""),
        "dtype": metric.get("dtype", ""),
        "dataset": metric.get("dataset", ""),
        "output_type": metric.get("output_type", ""),
        "latency_s": metric.get("latency_s"),
        "wall_clock_s": wall_clock_s,
        "wall_clock_source": wall_clock_source,
        "num_samples": metric.get("num_samples"),
        "peak_memory_mb": _as_number_or_none(metric.get("peak_memory_mb")),
        "ttft_ms": _as_number_or_none(metric.get("ttft_ms")),
        "tpot_ms": _as_number_or_none(metric.get("tpot_ms")),
        "throughput_metric_name": throughput_name,
        "throughput_metric_value": throughput_value,
        "quality_metric_name": quality_name,
        "quality_metric_value": quality_value,
        "measurement_contract_version": _as_int_or_none(metric.get("measurement_contract_version")),
        "latency_measurement_scope": str(metric.get("latency_measurement_scope") or "").strip(),
        "warmup_iterations": _as_int_or_none(metric.get("warmup_iterations")),
        "warmup_sample_count": _as_int_or_none(metric.get("warmup_sample_count")),
        "task_queue_enable": str(metric.get("task_queue_enable") or "").strip(),
        "loaded_from_model_files": bool(metric.get("loaded_from_model_files", False)),
        "model_source_effective": _portable_path_text(metric.get("model_source_effective")),
        "model_source_kind": str(metric.get("model_source_kind") or ""),
        "tokenizer_source_effective": _portable_path_text(metric.get("tokenizer_source_effective")),
        "tokenizer_source_kind": str(metric.get("tokenizer_source_kind") or ""),
        "patch_load_status": str(metric.get("patch_load_status") or ""),
        "patch_modules": list(metric.get("patch_modules") or []),
        "patch_hooks": list(metric.get("patch_hooks") or []),
    }


def _resolve_benchmark_run_started_at(config: dict, metrics: dict[str, dict]) -> str:
    configured_started_at = str(config.get("benchmark_run_started_at") or "").strip()
    if configured_started_at:
        return configured_started_at
    candidate_pairs: list[tuple[datetime, str]] = []
    for metric in metrics.values():
        started_at = str(metric.get("start_time") or "").strip()
        started_dt = _parse_iso_datetime(started_at)
        if started_dt is not None:
            candidate_pairs.append((started_dt, started_at))
    if not candidate_pairs:
        return ""
    candidate_pairs.sort(key=lambda item: item[0])
    return candidate_pairs[0][1]


def build_summary(
    adapt_dir: Path,
    *,
    comparison_scope: str = "real_business",
    remote_mode: str = "ssh_direct_transfer",
) -> dict:
    if comparison_scope not in VALID_COMPARISON_SCOPES:
        raise ValueError(f"comparison_scope 无效: {comparison_scope}")
    if not adapt_dir.is_dir():
        raise FileNotFoundError(f"adaptation 目录不存在: {adapt_dir}")

    artifact_paths, metrics_by_role = _select_artifacts_for_summary(adapt_dir)
    benchmark_run_id, metrics_by_role = _resolve_benchmark_run_id(adapt_dir, artifact_paths, metrics_by_role)
    dataset = _ensure_metric_consistency(metrics_by_role, field="dataset", label="dataset")
    output_type = _ensure_metric_consistency(metrics_by_role, field="output_type", label="output_type")
    npu_baseline = artifact_paths["npu_baseline"]
    npu_perf = artifact_paths["npu_perf"]
    npu_baseline_metric = metrics_by_role["npu_baseline"]
    npu_perf_metric = metrics_by_role["npu_perf"]
    cuda_baseline = artifact_paths.get("cuda_baseline")
    cuda_baseline_metric = metrics_by_role.get("cuda_baseline")
    has_cuda = cuda_baseline is not None and cuda_baseline_metric is not None
    contract_versions = {_as_int_or_none(metric.get("measurement_contract_version")) for metric in metrics_by_role.values() if _as_int_or_none(metric.get("measurement_contract_version")) is not None}
    if len(contract_versions) > 1:
        raise ValueError(f"业务测评工件的 measurement_contract_version 不一致: {sorted(contract_versions)}")
    measurement_contract_version = next(iter(contract_versions), 1)
    measurement_scopes = {str(metric.get("latency_measurement_scope") or "").strip() for metric in metrics_by_role.values() if str(metric.get("latency_measurement_scope") or "").strip()}
    if len(measurement_scopes) > 1:
        raise ValueError(f"业务测评工件的 latency_measurement_scope 不一致: {sorted(measurement_scopes)}")
    latency_measurement_scope = next(iter(measurement_scopes), "")

    results = [
        _metric_to_result("npu_baseline", npu_baseline, npu_baseline_metric),
        _metric_to_result("npu_perf", npu_perf, npu_perf_metric),
    ]
    if has_cuda and cuda_baseline is not None and cuda_baseline_metric is not None:
        results.append(_metric_to_result("cuda_baseline", cuda_baseline, cuda_baseline_metric))

    npu_baseline_latency = float(npu_baseline_metric["latency_s"])
    npu_perf_latency = float(npu_perf_metric["latency_s"])
    npu_baseline_wall_clock_s, npu_baseline_wall_clock_source = _resolve_metric_wall_clock_s(npu_baseline_metric)
    npu_perf_wall_clock_s, npu_perf_wall_clock_source = _resolve_metric_wall_clock_s(npu_perf_metric)
    npu_speedup_ratio_source = "latency_fallback"
    if (
        npu_baseline_wall_clock_s is not None
        and npu_baseline_wall_clock_s > 0
        and npu_perf_wall_clock_s is not None
        and npu_perf_wall_clock_s > 0
    ):
        npu_speedup_ratio = npu_baseline_wall_clock_s / npu_perf_wall_clock_s
        npu_speedup_ratio_source = npu_baseline_wall_clock_source if npu_baseline_wall_clock_source == npu_perf_wall_clock_source else f"{npu_baseline_wall_clock_source}+{npu_perf_wall_clock_source}"
    else:
        npu_speedup_ratio = npu_baseline_latency / npu_perf_latency
    quality_name, quality_value = _detect_quality(npu_perf_metric)
    npu_baseline_throughput_name, npu_baseline_throughput_value = _detect_throughput(npu_baseline_metric)
    npu_perf_throughput_name, npu_perf_throughput_value = _detect_throughput(npu_perf_metric)
    cuda_baseline_latency = float(cuda_baseline_metric["latency_s"]) if has_cuda and cuda_baseline_metric is not None else None
    cuda_baseline_wall_clock_s, cuda_baseline_wall_clock_source = _resolve_metric_wall_clock_s(cuda_baseline_metric) if has_cuda and cuda_baseline_metric is not None else (None, "")
    if has_cuda and cuda_baseline_metric is not None:
        cuda_baseline_throughput_name, cuda_baseline_throughput_value = _detect_throughput(cuda_baseline_metric)
        cuda_device_model = str(cuda_baseline_metric.get("device_model") or "")
        cuda_peak_memory_mb = _as_number_or_none(cuda_baseline_metric.get("peak_memory_mb"))
        cuda_quality_name = results[-1]["quality_metric_name"]
        cuda_quality_value = results[-1]["quality_metric_value"]
        cuda_warmup_iterations = _as_int_or_none(cuda_baseline_metric.get("warmup_iterations"))
        cuda_task_queue_enable = str(cuda_baseline_metric.get("task_queue_enable") or "").strip()
    else:
        cuda_baseline_throughput_name, cuda_baseline_throughput_value = "throughput_qps", None
        cuda_device_model = ""
        cuda_peak_memory_mb = None
        cuda_quality_name = quality_name
        cuda_quality_value = None
        cuda_warmup_iterations = None
        cuda_task_queue_enable = ""
    result_num_samples = [int(float(result["num_samples"])) for result in results if result.get("num_samples") is not None]
    if not result_num_samples:
        raise ValueError("业务测评工件缺少可用的 num_samples，无法生成 business_summary.json")
    num_samples = min(result_num_samples)
    config = _load_json(adapt_dir / "business_benchmark_config.json") if (adapt_dir / "business_benchmark_config.json").exists() else {}
    model_id = str(config.get("model_id") or adapt_dir.name)
    benchmark_run_started_at = _resolve_benchmark_run_started_at(config, metrics_by_role)
    ssh_host = str(config.get("remote_ssh_host") or "").strip()
    configured_remote_project_root = str(config.get("remote_project_root") or "").strip().rstrip("/") or REMOTE_PROJECT_ROOT_PLACEHOLDER
    remote_project_root = _resolve_remote_project_root(config.get("remote_project_root"))
    expected_cuda_artifact = f"business_metrics_cuda_{str(npu_perf_metric.get('dtype') or 'fp32').strip()}_pretrained_{dataset}_baseline.json"
    remote_execution: dict[str, object]
    if has_cuda:
        remote_execution = {"mode": remote_mode, "status": "completed"}
    else:
        remote_command = (
            f"uv run python business_benchmark/scripts/business_benchmark_manager.py run-remote-cuda "
            f"--model '{model_id}' --ssh-host '{ssh_host or 'cuda-remote'}' "
            f"--remote-project-root '{configured_remote_project_root}' --gpu-id 0"
        )
        remote_execution = {
            "mode": remote_mode,
            "status": "waiting_artifacts",
            "host": ssh_host,
            "command": remote_command,
            "expected_artifact": expected_cuda_artifact,
        }
    note = "" if has_cuda else f"CUDA baseline pending; run `{remote_execution.get('command', '')}` after SSH recovers."
    validation_note = ""
    if npu_baseline_wall_clock_s is not None and npu_perf_wall_clock_s is not None:
        validation_note = (
            f"npu_speedup_ratio uses wall-clock {npu_baseline_wall_clock_s:.6f}s / "
            f"{npu_perf_wall_clock_s:.6f}s ({npu_speedup_ratio_source})."
        )
    else:
        validation_note = (
            f"wall-clock unavailable; npu_speedup_ratio falls back to latency_s "
            f"{npu_baseline_latency:.6f}s / {npu_perf_latency:.6f}s."
        )
    npu_perf_parallel_mode = str(npu_perf_metric.get("parallel_mode") or "").strip().lower()
    npu_perf_device_topology = str(npu_perf_metric.get("device_topology") or "").strip().lower()
    if has_cuda and cuda_baseline_latency and npu_perf_latency:
        vs_cuda_latency_ratio = cuda_baseline_latency / npu_perf_latency
        if vs_cuda_latency_ratio < 0.4 and (
            npu_perf_parallel_mode in {"auto", "device_map_auto"} or npu_perf_device_topology == "all_visible_devices"
        ):
            note = "vs_cuda_latency_ratio < 0.4 且 NPU 使用 auto/all_visible 映射；请先固定 1-die 或 2-die 的 ASCEND_RT_VISIBLE_DEVICES 重跑 NPU 再判断结果。"

    return {
        "model_id": model_id,
        "benchmark_run_id": benchmark_run_id,
        "benchmark_run_started_at": benchmark_run_started_at,
        "measurement_contract_version": measurement_contract_version,
        "dataset": dataset,
        "comparison_scope": comparison_scope,
        "latency_measurement_scope": latency_measurement_scope,
        "num_samples": num_samples,
        "status": "completed" if has_cuda else "pending_remote_cuda",
        "remote_execution": remote_execution,
        "npu_baseline_artifact": npu_baseline.name,
        "npu_perf_artifact": npu_perf.name,
        "cuda_baseline_artifact": cuda_baseline.name if has_cuda and cuda_baseline is not None else expected_cuda_artifact,
        "comparison_evidence": {
            "npu_baseline_device_model": npu_baseline_metric.get("device_model", ""),
            "npu_perf_device_model": npu_perf_metric.get("device_model", ""),
            "cuda_baseline_device_model": cuda_device_model,
            "npu_baseline_peak_memory_mb": _as_number_or_none(npu_baseline_metric.get("peak_memory_mb")),
            "npu_perf_peak_memory_mb": _as_number_or_none(npu_perf_metric.get("peak_memory_mb")),
            "cuda_baseline_peak_memory_mb": cuda_peak_memory_mb,
            "npu_baseline_quality_metric_name": results[0]["quality_metric_name"],
            "npu_baseline_quality_metric_value": results[0]["quality_metric_value"],
            "npu_perf_quality_metric_name": results[1]["quality_metric_name"],
            "npu_perf_quality_metric_value": results[1]["quality_metric_value"],
            "cuda_baseline_quality_metric_name": cuda_quality_name,
            "cuda_baseline_quality_metric_value": cuda_quality_value,
            "npu_baseline_throughput_metric_name": npu_baseline_throughput_name,
            "npu_baseline_throughput_metric_value": npu_baseline_throughput_value,
            "npu_perf_throughput_metric_name": npu_perf_throughput_name,
            "npu_perf_throughput_metric_value": npu_perf_throughput_value,
            "cuda_baseline_throughput_metric_name": cuda_baseline_throughput_name,
            "cuda_baseline_throughput_metric_value": cuda_baseline_throughput_value,
            "npu_baseline_wall_clock_s": npu_baseline_wall_clock_s,
            "npu_perf_wall_clock_s": npu_perf_wall_clock_s,
            "cuda_baseline_wall_clock_s": cuda_baseline_wall_clock_s,
            "npu_speedup_ratio_source": npu_speedup_ratio_source,
            "cuda_wall_clock_source": cuda_baseline_wall_clock_source,
            "npu_baseline_warmup_iterations": _as_int_or_none(npu_baseline_metric.get("warmup_iterations")),
            "npu_perf_warmup_iterations": _as_int_or_none(npu_perf_metric.get("warmup_iterations")),
            "cuda_baseline_warmup_iterations": cuda_warmup_iterations,
            "npu_baseline_task_queue_enable": str(npu_baseline_metric.get("task_queue_enable") or "").strip(),
            "npu_perf_task_queue_enable": str(npu_perf_metric.get("task_queue_enable") or "").strip(),
            "cuda_baseline_task_queue_enable": cuda_task_queue_enable,
        },
        "results": results,
        "best_result": {
            "role": "npu_perf",
            "measurement_contract_version": measurement_contract_version,
            "dataset": dataset,
            "num_samples": num_samples,
            "output_type": output_type,
            "latency_measurement_scope": latency_measurement_scope,
            "comparison_method": "independent_baseline_artifact",
            "comparison_scope": comparison_scope,
            "baseline_artifact": npu_baseline.name,
            "perf_artifact": npu_perf.name,
            "baseline_latency_s": npu_baseline_latency,
            "perf_latency_s": npu_perf_latency,
            "steady_state_baseline_latency_s": npu_baseline_latency,
            "steady_state_perf_latency_s": npu_perf_latency,
            "baseline_wall_clock_s": npu_baseline_wall_clock_s,
            "perf_wall_clock_s": npu_perf_wall_clock_s,
            "wall_clock_source": npu_speedup_ratio_source,
            "npu_speedup_ratio": npu_speedup_ratio,
            "vs_cuda_latency_ratio": (cuda_baseline_latency / npu_perf_latency) if cuda_baseline_latency else None,
            "npu_perf_device_model": npu_perf_metric.get("device_model", ""),
            "cuda_baseline_device_model": cuda_device_model,
            "npu_perf_peak_memory_mb": _as_number_or_none(npu_perf_metric.get("peak_memory_mb")),
            "cuda_baseline_peak_memory_mb": cuda_peak_memory_mb,
            "npu_perf_throughput_metric_name": npu_perf_throughput_name,
            "npu_perf_throughput_metric_value": npu_perf_throughput_value,
            "cuda_baseline_throughput_metric_name": cuda_baseline_throughput_name,
            "cuda_baseline_throughput_metric_value": cuda_baseline_throughput_value,
            "quality_metric_name": quality_name,
            "quality_metric_value": quality_value,
            "warmup_iterations": _as_int_or_none(npu_perf_metric.get("warmup_iterations")),
            "task_queue_enable": str(npu_perf_metric.get("task_queue_enable") or "").strip(),
            "loaded_from_model_files": bool(npu_perf_metric.get("loaded_from_model_files", False)),
            "patch_load_status": str(npu_perf_metric.get("patch_load_status") or ""),
            "patch_hooks": list(npu_perf_metric.get("patch_hooks") or []),
            "validation_note": validation_note,
            "note": note,
        },
    }


def cmd_summarize(args):
    adapt_name = args.adaptation.strip().strip("/")
    adapt_dir = ADAPTATIONS_DIR / adapt_name
    summary = build_summary(adapt_dir, comparison_scope=args.comparison_scope, remote_mode=args.remote_mode)
    output = adapt_dir / (args.output or "business_summary.json")
    _write_json(output, summary)
    print(f"[business] wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Business benchmark tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize_parser = subparsers.add_parser("summarize", help="从业务测评工件生成 business_summary.json")
    summarize_parser.add_argument("--adaptation", required=True, help="adaptation 目录名（不带 adaptations/）")
    summarize_parser.add_argument("--comparison-scope", default="real_business")
    summarize_parser.add_argument("--remote-mode", default="ssh_direct_transfer")
    summarize_parser.add_argument("--output", default="business_summary.json")
    summarize_parser.set_defaults(func=cmd_summarize)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
