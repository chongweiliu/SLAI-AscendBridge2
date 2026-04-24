#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shlex
import shutil
import sqlite3
import socket
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback for local ops
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "board.db"
ADAPTATIONS_DIR = PROJECT_ROOT / "adaptations"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dataset_mapping import get_business_benchmark_profile, resolve_multilingual_asr_dataset
from download_datasets import ensure_dataset, get_dataset_disk_path

try:
    from .business_benchmark_tool import build_summary
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from business_benchmark_tool import build_summary  # type: ignore[import-not-found]
BUSINESS_PATTERNS = [
    "business_model_eval.py",
    "business_eval.py",
    "business_run.py",
    "business_summary.json",
    "business_benchmark_config.json",
    "business_metrics_*.json",
    "business_outputs_*.pt",
]
BUSINESS_EVAL_FILENAME = "business_eval.py"
BUSINESS_MODEL_EVAL_FILENAME = "business_model_eval.py"
BUSINESS_RUN_FILENAME = "business_run.py"
BUSINESS_EVAL_TEMPLATE = PROJECT_ROOT / "business_benchmark" / "templates" / BUSINESS_EVAL_FILENAME
BUSINESS_MODEL_EVAL_TEMPLATE = PROJECT_ROOT / "business_benchmark" / "templates" / BUSINESS_MODEL_EVAL_FILENAME
BUSINESS_LEGACY_OLMO_HELPER_TEMPLATE = PROJECT_ROOT / "business_benchmark" / "templates" / "modeling_olmo_v1.py"
SCENARIO_TO_CONFIG_KEY = {
    "npu_baseline": "local_npu_baseline_command",
    "npu_perf": "local_npu_perf_command",
    "cuda_baseline": "remote_cuda_baseline_command",
}
SCENARIO_TO_ENV_KEY = {
    "npu_baseline": "npu_baseline_env",
    "npu_perf": "npu_perf_env",
    "cuda_baseline": "cuda_baseline_env",
}
SCENARIO_TO_REQUIRED_EXTRA = {
    "npu_baseline": "ascend",
    "npu_perf": "ascend",
    "cuda_baseline": "cuda",
}
DEFAULT_SSH_HOST_ALIAS = "cuda-remote"
DEFAULT_REMOTE_PROJECT_ROOT_ENV_KEY = "SLAI_REMOTE_PROJECT_ROOT"
REMOTE_PROJECT_ROOT_PLACEHOLDER = f"${DEFAULT_REMOTE_PROJECT_ROOT_ENV_KEY}"
DEFAULT_REMOTE_PROJECT_BASE = str(os.environ.get("SLAI_REMOTE_PROJECT_BASE") or "/workspace").strip().rstrip("/")
DEFAULT_REMOTE_PROJECT_OWNER = str(os.environ.get("SLAI_REMOTE_PROJECT_OWNER") or "").strip()


def _derived_remote_project_root() -> str:
    repo_name = PROJECT_ROOT.name
    if DEFAULT_REMOTE_PROJECT_OWNER:
        return f"{DEFAULT_REMOTE_PROJECT_BASE}/{DEFAULT_REMOTE_PROJECT_OWNER}/{repo_name}"
    return f"{DEFAULT_REMOTE_PROJECT_BASE}/{repo_name}"


def _default_remote_project_root() -> str:
    env_value = str(os.environ.get(DEFAULT_REMOTE_PROJECT_ROOT_ENV_KEY) or "").strip().rstrip("/")
    if env_value:
        return env_value
    return _derived_remote_project_root()


DEFAULT_REMOTE_PROJECT_ROOT = _default_remote_project_root()
LEGACY_REMOTE_PROJECT_ROOTS = {
    f"{DEFAULT_REMOTE_PROJECT_ROOT}-adapt",
}
DEFAULT_REMOTE_EXECUTION_MODE = "ssh_direct_transfer"
DEFAULT_MEASUREMENT_CONTRACT_VERSION = 2
DEFAULT_LATENCY_MEASUREMENT_SCOPE = "steady_state"
DEFAULT_BASELINE_WARMUP_ITERATIONS = 1
DEFAULT_PERF_WARMUP_ITERATIONS = 3
DEFAULT_CUDA_BASELINE_WARMUP_ITERATIONS = 1
DEFAULT_EMBEDDING_BATCH_SIZE = 8
DEFAULT_EMBEDDING_STEADY_STATE_REPEATS = 8
DEFAULT_RUNTIME_ONLY_EMBEDDING_STEADY_STATE_REPEATS = 16
DEFAULT_HF_ENDPOINT = str(os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com").strip()
DEFAULT_HF_HUB_DOWNLOAD_TIMEOUT = str(os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT") or "120").strip()
DEFAULT_HF_HUB_ETAG_TIMEOUT = str(os.environ.get("HF_HUB_ETAG_TIMEOUT") or "60").strip()
DEFAULT_REMOTE_UV_HTTP_TIMEOUT = str(os.environ.get("UV_HTTP_TIMEOUT") or "1800").strip()
DEFAULT_REMOTE_UV_DEFAULT_INDEX = str(os.environ.get("UV_DEFAULT_INDEX") or os.environ.get("UV_INDEX_URL") or os.environ.get("PIP_INDEX_URL") or "https://pypi.tuna.tsinghua.edu.cn/simple").strip()
DEFAULT_REMOTE_UV_EXTRA_INDEX = str(os.environ.get("UV_EXTRA_INDEX_URL") or os.environ.get("PIP_EXTRA_INDEX_URL") or "").strip()
DEFAULT_LOCAL_UV_CACHE_FALLBACK = str((PROJECT_ROOT / ".uv-cache").resolve())
DEFAULT_REMOTE_CUDA_BASELINE_TOTAL_RUNS = 2
DEFAULT_NPU_BASELINE_ENV: dict[str, str] = {}
DEFAULT_NPU_PERF_ENV: dict[str, str] = {"TASK_QUEUE_ENABLE": "1"}
DEFAULT_CUDA_BASELINE_ENV: dict[str, str] = {}
PHASE4_PROFILE_OVERRIDE_KEYS = {
    "business_intent": "business_intent_override",
    "business_intent_name": "business_intent_name_override",
    "model_type": "model_type_override",
    "model_type_name": "model_type_name_override",
    "dataset": "dataset_override",
    "evaluation_profile": "evaluation_profile_override",
    "primary_metric": "primary_metric_override",
    "secondary_metrics": "secondary_metrics_override",
    "output_type_hint": "output_type_hint_override",
    "model_backend": "model_backend_override",
    "model_file": "model_file_override",
    "detection_target_labels": "detection_target_labels_override",
    "asr_task": "asr_task_override",
    "asr_language": "asr_language_override",
}
PHASE4_PROFILE_LIST_FIELDS = {"secondary_metrics", "detection_target_labels"}
NPU_VISIBLE_ENV_KEY = "ASCEND_RT_VISIBLE_DEVICES"
SSH_CONNECT_TIMEOUT_SECONDS = 8
SSH_SERVER_ALIVE_INTERVAL_SECONDS = 15
SSH_SERVER_ALIVE_COUNT_MAX = 2
REMOTE_TRANSPORT_RETRY_ATTEMPTS = 5
REMOTE_TRANSPORT_RETRY_SLEEP_SECONDS = 4
SSH_CONTROL_PERSIST_SECONDS = 120
SSH_CONTROL_PATH = "/tmp/slai-bbm-%C"
REMOTE_FETCH_STAGE_DIRNAME = ".remote_fetch_staging"
REMOTE_FETCH_CONFLICT_DIRNAME = "remote_fetch_conflicts"
REMOTE_RUNTIME_CLEANUP_PATTERNS = [
    "business_summary.json",
    "business_metrics_*.json",
    "business_outputs_*.pt",
]
REMOTE_ADAPT_SYNC_EXCLUDES = [
    ".venv*/",
    ".uv_cache_local/",
    ".uv_cache_remote/",
    ".phase4_stale_*/",
    "models/",
    "business_model_cache/",
    "business_dataset_cache/",
    "datasets/",
    "profiling/",
    "export_only_prof_dir/",
    "Atlas-*-ascend_pt/",
    "__pycache__/",
    "*.pyc",
    "benchmark_metrics_*.json",
    "outputs_*.pt",
    "trace_*.json",
    "output.txt",
    "business_metrics_*.json",
    "business_outputs_*.pt",
    "business_summary.json",
]
REMOTE_ROOT_SYNC_FILES = [
    "dataset_mapping.py",
    "download_datasets.py",
]
REMOTE_INPUT_SNAPSHOT_ASSET_FILENAMES = (
    "config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "sentencepiece.bpe.model",
    "sentencepiece.model",
    "spiece.model",
    "tokenizer.model",
    "added_tokens.json",
    "chat_template.jinja",
    "configuration.py",
    "modeling.py",
    "processing.py",
    "feature_extraction.py",
    "image_processing.py",
    "tokenization.py",
    "tokenization_fast.py",
    "configuration_*.py",
    "modeling_*.py",
    "processing_*.py",
    "feature_extraction_*.py",
    "image_processing_*.py",
    "tokenization_*.py",
)
REMOTE_MODEL_WEIGHT_FILE_PATTERNS = (
    "*.safetensors",
    "*.bin",
    "*.ot",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.h5",
    "*.hdf5",
    "*.msgpack",
)
REMOTE_MODEL_WEIGHT_FILE_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".ot",
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
    ".hdf5",
    ".msgpack",
)
REMOTE_SYNC_TRANSIENT_FILE_PATTERNS = (
    ".*",
    "*.tmp",
    "*.tmp.*",
    "*.part",
    "*.partial",
)
REMOTE_SNAPSHOT_EXPORT_EXCLUDE_PATTERNS = (
    "onnx/",
    "openvino/",
    "*.onnx",
    "*.xml",
)
MANAGER_SCRIPT_RELATIVE_PATH = "business_benchmark/scripts/business_benchmark_manager.py"
BUSINESS_RUNTIME_COMMON_DEPENDENCIES = (
    "datasets>=3.0.0",
    "protobuf>=4.25.3",
    "sentencepiece>=0.2.0",
    "transformers>=4.45.0",
)
BUSINESS_RUNTIME_DEPENDENCIES_BY_PROFILE = {
    # asr_wer now has a built-in fallback in business_eval.py, so jiwer stays optional.
    # Audio decoding still needs a local backend for FLEURS / LibriSpeech payloads.
    "asr_wer": ("soundfile>=0.13.1", "librosa>=0.10.2"),
    "summarization_rouge": ("rouge-score>=0.1.2",),
    "token_classification_f1": ("seqeval>=1.2.2",),
    "reranker_ndcg": ("scikit-learn>=1.4.0",),
}
BUSINESS_RUNTIME_DEPENDENCIES_BY_MODEL_TYPE = {
    "image_matting": ("pillow>=10.0.0",),
    "timeseries": ("granite-tsfm",),
    "video": ("pillow>=10.0.0",),
    "vision_classification": ("pillow>=10.0.0",),
    "vision_detection": ("pillow>=10.0.0",),
    "vision_embedding": ("pillow>=10.0.0",),
    "vlm": ("pillow>=10.0.0",),
}
BUSINESS_RUNTIME_SPAN_MARKER_DEPENDENCIES = ("span-marker>=1.5.0",)
BUSINESS_RUNTIME_LEGACY_OLMO_DEPENDENCIES = ("ai2-olmo==0.6.0",)
INTERNAL_SYNTHETIC_BUSINESS_DATASETS = {
    "builtin",
    "builtin_smiles",
    "latency_only",
    "synthetic_3d",
    "synthetic_colour_checker",
    "synthetic_dna",
    "synthetic_ocr",
    "synthetic_timeseries",
    "synthetic_triplets",
}


def _is_writable_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".uv-cache-write-test-{os.getpid()}"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _resolve_local_uv_cache_dir() -> str | None:
    candidates = [DEFAULT_LOCAL_UV_CACHE_FALLBACK, str((Path.home() / ".cache" / "uv").resolve())]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = str(Path(candidate).expanduser())
        if _is_writable_directory(Path(resolved)):
            return resolved
    return None


LOCAL_UV_CACHE_DIR = _resolve_local_uv_cache_dir()


def _repo_relative_path_text(path: str | Path, *, base_dir: Path | None = None) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    expanded = os.path.expandvars(text)
    candidate = Path(expanded).expanduser()
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        rel_to_repo = candidate.resolve().relative_to(PROJECT_ROOT.resolve())
    except Exception:
        return text
    if base_dir is None:
        return rel_to_repo.as_posix()
    return os.path.relpath(PROJECT_ROOT / rel_to_repo, base_dir).replace(os.sep, "/")


def _resolve_config_path(path_value: object, *, adapt_dir: Path) -> Path | None:
    text = str(path_value or "").strip()
    if not text:
        return None
    expanded = os.path.expandvars(text)
    candidate = Path(expanded).expanduser()
    if candidate.is_absolute():
        return candidate
    for resolved in (adapt_dir / candidate, PROJECT_ROOT / candidate):
        if resolved.exists():
            return resolved
    return adapt_dir / candidate


def _should_placeholder_remote_root(remote_project_root: str) -> bool:
    normalized = remote_project_root.strip().rstrip("/")
    if not normalized:
        return True
    if normalized == REMOTE_PROJECT_ROOT_PLACEHOLDER:
        return True
    if normalized in LEGACY_REMOTE_PROJECT_ROOTS or normalized == DEFAULT_REMOTE_PROJECT_ROOT:
        return True
    repo_suffix = f"/{PROJECT_ROOT.name}"
    return normalized.startswith(f"{DEFAULT_REMOTE_PROJECT_BASE}/") and normalized.endswith(repo_suffix)


def _portable_remote_project_root(remote_project_root: object) -> str:
    text = str(remote_project_root or "").strip().rstrip("/")
    expanded = os.path.expandvars(text).strip().rstrip("/") if text else ""
    if _should_placeholder_remote_root(expanded or text):
        return REMOTE_PROJECT_ROOT_PLACEHOLDER
    if text.startswith("$"):
        return text
    return text or REMOTE_PROJECT_ROOT_PLACEHOLDER


def _resolve_remote_project_root(remote_project_root: object) -> str:
    text = str(remote_project_root or "").strip().rstrip("/")
    if not text or text == REMOTE_PROJECT_ROOT_PLACEHOLDER:
        return DEFAULT_REMOTE_PROJECT_ROOT
    expanded = os.path.expandvars(text).strip().rstrip("/")
    if not expanded or expanded.startswith("$"):
        return DEFAULT_REMOTE_PROJECT_ROOT
    return expanded


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


def _generation_metadata(*, tool: str) -> dict:
    return {
        "generated_at": datetime.now().isoformat(),
        "generated_by_tool": tool,
        "generated_by_user": getpass.getuser(),
        "generated_by_hostname": socket.gethostname(),
        "generated_by_host_ip": _best_effort_host_ip(),
        "generated_by_pid": os.getpid(),
    }


def _with_generation_metadata(payload: dict, *, tool: str) -> dict:
    enriched = dict(payload)
    enriched.update(_generation_metadata(tool=tool))
    return enriched


def _render_generated_python_script(script_content: str, *, tool: str) -> str:
    metadata = _generation_metadata(tool=tool)
    metadata_lines = ["# Generator metadata:"]
    metadata_lines.extend(f"# {key}: {value}" for key, value in metadata.items())
    if script_content.startswith("#!"):
        first_line, _, rest = script_content.partition("\n")
        lines = [first_line, *metadata_lines, ""]
        if rest:
            lines.append(rest)
        return "\n".join(lines)
    return "\n".join([*metadata_lines, "", script_content])


def _canonical_business_eval_command(scenario: str) -> str:
    required_extra = SCENARIO_TO_REQUIRED_EXTRA[scenario]
    return f'uv run --no-sync --extra {required_extra} python "{BUSINESS_EVAL_FILENAME}" --scenario {scenario}'


def _canonical_business_run_command(scenario: str) -> str:
    required_extra = SCENARIO_TO_REQUIRED_EXTRA[scenario]
    return f'uv run --no-sync --extra {required_extra} python "{BUSINESS_RUN_FILENAME}" --scenario {scenario}'


def _preferred_uv_project_environment(cwd: Path) -> str | None:
    try:
        resolved_cwd = cwd.resolve()
    except Exception:
        resolved_cwd = cwd
    if resolved_cwd != ADAPTATIONS_DIR and ADAPTATIONS_DIR not in resolved_cwd.parents:
        return None
    # Phase-4 formal evidence only accepts the adaptation's canonical `.venv`.
    # Falling back to `.venv_user` produces an invalid measurement environment.
    return None


def _looks_like_stale_phase4_python_link(python_path: Path) -> bool:
    try:
        raw_target = os.readlink(python_path) if python_path.is_symlink() else str(python_path)
    except OSError:
        raw_target = str(python_path)
    normalized_target = raw_target.replace("\\", "/").lower()
    return any(marker in normalized_target for marker in ("/root/anaconda", "/root/miniconda", "/opt/conda/"))


def _phase4_default_venv_stale_reason(adapt_dir: Path) -> str | None:
    default_env_dir = adapt_dir / ".venv"
    if not default_env_dir.exists():
        return None
    default_python = default_env_dir / "bin" / "python"
    if _looks_like_stale_phase4_python_link(default_python):
        return "stale_conda_python_link"
    try:
        if not default_python.exists():
            return "missing_python"
        if not os.access(default_python, os.X_OK):
            return "python_not_executable"
    except OSError:
        return "python_not_executable"
    return None


def _phase4_lockfile_stale_reason(adapt_dir: Path) -> str | None:
    lock_path = adapt_dir / "uv.lock"
    if not lock_path.exists():
        return None
    try:
        if not os.access(lock_path, os.W_OK):
            return "lock_not_writable"
    except OSError:
        return "lock_not_writable"
    return None


def _quarantine_phase4_local_path(path: Path, label: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    candidate = path.parent / f".phase4_stale_{label}_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = path.parent / f".phase4_stale_{label}_{timestamp}_{suffix}"
        suffix += 1
    path.rename(candidate)
    return candidate


def _quarantine_stale_phase4_local_env(adapt_dir: Path) -> list[Path]:
    moved: list[Path] = []
    env_reason = _phase4_default_venv_stale_reason(adapt_dir)
    if env_reason:
        for name, label in ((".venv", "venv"), (".venv_user", "venv_user"), ("__pycache__", "pycache")):
            path = adapt_dir / name
            if not path.exists():
                continue
            target = _quarantine_phase4_local_path(path, label)
            moved.append(target)
            print(f"[business][npu][quarantine] {path.name} -> {target.name} ({env_reason})")

    lock_reason = _phase4_lockfile_stale_reason(adapt_dir)
    if lock_reason:
        lock_path = adapt_dir / "uv.lock"
        target = _quarantine_phase4_local_path(lock_path, "lock")
        moved.append(target)
        print(f"[business][npu][quarantine] {lock_path.name} -> {target.name} ({lock_reason})")
    return moved


def _parse_shell_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return []


def _extract_uv_run_extras(command: str) -> list[str]:
    tokens = _parse_shell_command(command)
    extras: list[str] = []
    for idx, token in enumerate(tokens):
        if token == "--extra" and idx + 1 < len(tokens):
            extra = str(tokens[idx + 1]).strip()
            if extra:
                extras.append(extra)
    return extras


def _normalize_business_scenario_command(raw_command: object, scenario: str) -> str:
    command = str(raw_command or "").strip()
    canonical = _canonical_business_eval_command(scenario)
    if not command:
        return canonical
    legacy_commands = {
        f'uv run python "{BUSINESS_EVAL_FILENAME}" --scenario {scenario}',
        f"uv run python {BUSINESS_EVAL_FILENAME} --scenario {scenario}",
        f'uv run --extra {SCENARIO_TO_REQUIRED_EXTRA[scenario]} python "{BUSINESS_EVAL_FILENAME}" --scenario {scenario}',
        f'uv run --extra {SCENARIO_TO_REQUIRED_EXTRA[scenario]} python {BUSINESS_EVAL_FILENAME} --scenario {scenario}',
        f'.venv/bin/python "{BUSINESS_EVAL_FILENAME}" --scenario {scenario}',
        f".venv/bin/python {BUSINESS_EVAL_FILENAME} --scenario {scenario}",
        f'python "{BUSINESS_EVAL_FILENAME}" --scenario {scenario}',
        f"python {BUSINESS_EVAL_FILENAME} --scenario {scenario}",
        f'uv run --extra {SCENARIO_TO_REQUIRED_EXTRA[scenario]} python "{BUSINESS_RUN_FILENAME}" --scenario {scenario}',
        f"uv run --extra {SCENARIO_TO_REQUIRED_EXTRA[scenario]} python {BUSINESS_RUN_FILENAME} --scenario {scenario}",
        f'uv run python "{BUSINESS_RUN_FILENAME}" --scenario {scenario}',
        f"uv run python {BUSINESS_RUN_FILENAME} --scenario {scenario}",
        f'.venv/bin/python "{BUSINESS_RUN_FILENAME}" --scenario {scenario}',
        f".venv/bin/python {BUSINESS_RUN_FILENAME} --scenario {scenario}",
        f'python "{BUSINESS_RUN_FILENAME}" --scenario {scenario}',
        f"python {BUSINESS_RUN_FILENAME} --scenario {scenario}",
    }
    if command in legacy_commands:
        return canonical
    return command


def _validate_business_scenario_command(command: str, scenario: str, *, config_key: str) -> None:
    required_extra = SCENARIO_TO_REQUIRED_EXTRA[scenario]
    tokens = _parse_shell_command(command)
    if len(tokens) < 2 or tokens[0] != "uv" or tokens[1] != "run":
        raise ValueError(f"{config_key} 必须显式使用 `uv run --extra {required_extra}`，当前为: {command}")
    if "--no-sync" not in tokens:
        raise ValueError(f"{config_key} 必须显式使用 `uv run --no-sync --extra {required_extra}`，当前为: {command}")
    extras = _extract_uv_run_extras(command)
    if required_extra not in extras:
        raise ValueError(f"{config_key} 必须显式使用 `uv run --extra {required_extra}`，当前为: {command}")
    wrong_extras = sorted({"ascend", "cuda"}.intersection(extras) - {required_extra})
    if wrong_extras:
        raise ValueError(f"{config_key} 不得混用错误 extra={wrong_extras}，当前为: {command}")
    scenario_token = f"--scenario={scenario}"
    if "--scenario" in tokens:
        idx = tokens.index("--scenario")
        if idx + 1 >= len(tokens) or tokens[idx + 1] != scenario:
            raise ValueError(f"{config_key} 的场景参数必须是 {scenario}，当前为: {command}")
    elif scenario_token not in tokens:
        raise ValueError(f"{config_key} 缺少 `--scenario {scenario}`，当前为: {command}")


def _coerce_text(value: object) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    if value is None:
        return ""
    return str(value).strip()


def _resolve_open_llama_easylm_base_model_id(model_id: str) -> str:
    normalized_model_id = str(model_id or "").strip()
    if "/" not in normalized_model_id:
        return ""
    org, name = normalized_model_id.split("/", 1)
    if org.lower() != "openlm-research":
        return ""
    lowered_name = name.lower()
    if not lowered_name.startswith("open_llama_") or not lowered_name.endswith("_easylm"):
        return ""
    base_name = name[: -len("_easylm")].strip()
    if not base_name:
        return ""
    return f"{org}/{base_name}"


def _contains_keyword_signal(text: str, signal: str) -> bool:
    normalized_text = str(text or "").strip().lower()
    normalized_signal = str(signal or "").strip().lower()
    if not normalized_text or not normalized_signal:
        return False
    if normalized_signal.isalnum():
        pattern = rf"(?<![a-z0-9]){re.escape(normalized_signal)}(?![a-z0-9])"
        return re.search(pattern, normalized_text) is not None
    return normalized_signal in normalized_text


def _contains_any_keyword_signal(text: str, signals: tuple[str, ...]) -> bool:
    return any(_contains_keyword_signal(text, signal) for signal in signals)


def _normalize_requirement_name(requirement: str) -> str:
    base = re.split(r"[<>=!~\[\];\s]", str(requirement or "").strip(), maxsplit=1)[0]
    return base.replace("_", "-").lower()


def _dataset_requires_local_path(dataset_key: str | None) -> bool:
    normalized = str(dataset_key or "").strip().lower()
    return bool(normalized) and normalized not in INTERNAL_SYNTHETIC_BUSINESS_DATASETS


def _replace_project_dependencies_block(pyproject_text: str, dependencies: list[str]) -> str:
    lines = pyproject_text.splitlines(keepends=True)
    in_project = False
    dep_start: int | None = None
    dep_end: int | None = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if stripped == "[project]":
                in_project = True
                continue
            if in_project:
                break
        if not in_project:
            continue
        if not stripped.startswith("dependencies"):
            continue
        dep_start = idx
        if "]" in line.split("=", 1)[-1]:
            dep_end = idx
            break
        for end_idx in range(idx + 1, len(lines)):
            if lines[end_idx].strip() == "]":
                dep_end = end_idx
                break
        break
    if dep_start is None or dep_end is None:
        raise ValueError("pyproject.toml 缺少 [project].dependencies 块，无法自动补齐第四阶段依赖")
    newline = "\r\n" if "\r\n" in pyproject_text else "\n"
    block = ["dependencies = [" + newline]
    block.extend(f'    "{dependency}",{newline}' for dependency in dependencies)
    block.append("]" + newline)
    return "".join(lines[:dep_start] + block + lines[dep_end + 1 :])


def _load_first_local_model_config_payload(adapt_dir: Path) -> dict[str, Any]:
    models_dir = adapt_dir / "models"
    if not models_dir.exists():
        return {}
    for config_path in sorted(models_dir.rglob("config.json")):
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _requires_ai2_olmo_runtime(adapt_dir: Path, *, model_id: str, model_type: str) -> bool:
    if str(model_type or "").strip().lower() != "causal_lm":
        return False
    config_payload = _load_first_local_model_config_payload(adapt_dir)
    payload_model_type = str(config_payload.get("model_type") or "").strip().lower()
    payload_architectures = [str(item).strip().lower() for item in list(config_payload.get("architectures") or []) if str(item).strip()]
    detected = False
    if payload_model_type == "olmo" or any("olmoforcausallm" in arch for arch in payload_architectures):
        detected = True
    model_id_text = str(model_id or "").strip().lower()
    if "olmo" in model_id_text:
        detected = True
    if not detected:
        context = _read_adaptation_context(adapt_dir).lower()
        detected = "olmo" in context
    if not detected:
        return False
    if (adapt_dir / "modeling_olmo_v1.py").exists():
        return False
    if (adapt_dir / "model_files" / "modeling_olmo_v1.py").exists():
        return False
    if BUSINESS_LEGACY_OLMO_HELPER_TEMPLATE.exists():
        return False
    return True


def _requires_span_marker_runtime(adapt_dir: Path, *, model_id: str, model_type: str) -> bool:
    canonical_model_type = str(model_type or "").strip().lower()
    if canonical_model_type not in {"token_classification", "biomedical_token_classification"}:
        return False
    config_payload = _load_first_local_model_config_payload(adapt_dir)
    payload_model_type = re.sub(r"[^a-z0-9]+", "", str(config_payload.get("model_type") or "").strip().lower())
    payload_architectures = [re.sub(r"[^a-z0-9]+", "", str(item).strip().lower()) for item in list(config_payload.get("architectures") or []) if str(item).strip()]
    model_id_text = re.sub(r"[^a-z0-9]+", "", str(model_id or "").strip().lower())
    if payload_model_type == "spanmarker" or "spanmarkermodel" in payload_architectures or "spanmarker" in model_id_text:
        return True
    context = _read_adaptation_context(adapt_dir).lower()
    return "span-marker" in context or "spanmarker" in context


def _ensure_business_runtime_dependencies(adapt_dir: Path, evaluation_profile: str, model_type: str = "", model_id: str = "", config: dict[str, Any] | None = None) -> None:
    config = config or {}
    auto_patch_value = config.get("auto_patch_runtime_dependencies")
    if isinstance(auto_patch_value, bool) and not auto_patch_value:
        return

    required_dependencies = list(BUSINESS_RUNTIME_COMMON_DEPENDENCIES)
    required_dependencies.extend(BUSINESS_RUNTIME_DEPENDENCIES_BY_PROFILE.get(str(evaluation_profile or "").strip(), ()))
    required_dependencies.extend(BUSINESS_RUNTIME_DEPENDENCIES_BY_MODEL_TYPE.get(str(model_type or "").strip().lower(), ()))
    if _requires_span_marker_runtime(adapt_dir, model_id=model_id, model_type=model_type):
        required_dependencies.extend(BUSINESS_RUNTIME_SPAN_MARKER_DEPENDENCIES)
    if _requires_ai2_olmo_runtime(adapt_dir, model_id=model_id, model_type=model_type):
        required_dependencies.extend(BUSINESS_RUNTIME_LEGACY_OLMO_DEPENDENCIES)
    pyproject_path = adapt_dir / "pyproject.toml"
    if not pyproject_path.exists():
        raise FileNotFoundError(f"{adapt_dir} 缺少 pyproject.toml，无法执行第四阶段业务测评")
    raw_text = pyproject_path.read_text(encoding="utf-8")
    parsed = tomllib.loads(raw_text)
    project = parsed.get("project")
    if not isinstance(project, dict):
        raise ValueError(f"{pyproject_path} 缺少 [project] 段")
    dependencies = project.get("dependencies")
    if not isinstance(dependencies, list):
        raise ValueError(f"{pyproject_path} 缺少 project.dependencies 列表")

    existing_names = {_normalize_requirement_name(item) for item in dependencies if isinstance(item, str) and str(item).strip()}
    merged_dependencies = [str(item) for item in dependencies if isinstance(item, str) and str(item).strip()]
    missing = [item for item in required_dependencies if _normalize_requirement_name(item) not in existing_names]
    if not missing:
        return

    merged_dependencies.extend(missing)
    updated_text = _replace_project_dependencies_block(raw_text, merged_dependencies)
    pyproject_path.write_text(updated_text, encoding="utf-8")
    print(f"[business][deps] patched {pyproject_path.relative_to(PROJECT_ROOT)} + {', '.join(missing)}")


def _first_text(mapping: dict, *keys: str) -> str:
    for key in keys:
        text = _coerce_text(mapping.get(key))
        if text:
            return text
    return ""


def _first_list(mapping: dict, *keys: str) -> list[str]:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, list):
            items = [str(item).strip() for item in value if str(item).strip()]
            if items:
                return items
    return []


def get_db_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def get_models(status: str | None = None, require_optimization_completed: bool = False) -> list[dict]:
    conn = get_db_connection()
    cur = conn.cursor()
    if status:
        if require_optimization_completed:
            cur.execute(
                """
                SELECT model_id, adaptation_path, optimization_status, business_benchmark_status
                FROM models
                WHERE business_benchmark_status = ? AND optimization_status = 'completed'
                ORDER BY model_id
                """,
                (status,),
            )
        else:
            cur.execute(
                """
                SELECT model_id, adaptation_path, optimization_status, business_benchmark_status
                FROM models
                WHERE business_benchmark_status = ?
                ORDER BY model_id
                """,
                (status,),
            )
    else:
        if require_optimization_completed:
            cur.execute(
                """
                SELECT model_id, adaptation_path, optimization_status, business_benchmark_status
                FROM models
                WHERE optimization_status = 'completed'
                ORDER BY model_id
                """
            )
        else:
            cur.execute(
                """
                SELECT model_id, adaptation_path, optimization_status, business_benchmark_status
                FROM models
                WHERE business_benchmark_status != '' AND business_benchmark_status IS NOT NULL
                ORDER BY model_id
                """
            )
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _resolve_adapt_dir(model: dict) -> Path:
    adapt_path = model.get("adaptation_path") or model["model_id"].replace("/", "_")
    return PROJECT_ROOT / adapt_path if adapt_path.startswith("adaptations/") else ADAPTATIONS_DIR / adapt_path


def _load_config(adapt_dir: Path, *, missing_ok: bool = False) -> dict:
    config_path = adapt_dir / "business_benchmark_config.json"
    if not config_path.exists():
        if missing_ok:
            return {}
        raise FileNotFoundError(f"缺少业务测评配置文件: {config_path}")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("business_benchmark_config.json 必须是 JSON object")
    return data


def _write_config(adapt_dir: Path, payload: dict) -> None:
    config_path = adapt_dir / "business_benchmark_config.json"
    _write_json_text(config_path, payload)


def _normalize_env_mapping(raw_value: object) -> dict[str, str]:
    if not isinstance(raw_value, dict):
        return {}
    env_map: dict[str, str] = {}
    for key, value in raw_value.items():
        key_text = str(key or "").strip()
        if not key_text:
            continue
        env_map[key_text] = str(value if value is not None else "")
    return env_map


def _merge_env_mapping_with_defaults(raw_value: object, defaults: dict[str, str] | None = None) -> dict[str, str]:
    merged: dict[str, str] = {}
    if defaults:
        merged.update({str(key): str(value) for key, value in defaults.items()})
    merged.update(_normalize_env_mapping(raw_value))
    return merged


def _write_json_text(path: Path, payload: dict, *, tool: str = MANAGER_SCRIPT_RELATIVE_PATH) -> None:
    content = json.dumps(_with_generation_metadata(payload, tool=tool), ensure_ascii=False, indent=2) + "\n"
    try:
        path.write_text(content, encoding="utf-8")
    except PermissionError:
        if not path.exists():
            raise
        path.unlink()
        path.write_text(content, encoding="utf-8")


def _rank_idle_npu_ids() -> list[str]:
    result = _run_command(["npu-smi", "info"], PROJECT_ROOT, capture_output=True, check=False)
    if result.returncode != 0 or not result.stdout:
        return []

    device_ids: list[str] = []
    process_counts: dict[str, int] = {}
    in_process_table = False
    for line in result.stdout.splitlines():
        if "Process id" in line and "Process name" in line:
            in_process_table = True
            continue
        if not in_process_table:
            match = re.match(r"^\|\s*(\d+)\s+Ascend", line)
            if match:
                device_id = match.group(1)
                if device_id not in device_ids:
                    device_ids.append(device_id)
            continue
        match = re.match(r"^\|\s*(\d+)\s+\d+\s+\|", line)
        if match:
            device_id = match.group(1)
            process_counts[device_id] = process_counts.get(device_id, 0) + 1

    return sorted(device_ids, key=lambda device_id: (process_counts.get(device_id, 0), int(device_id)))


def _apply_local_npu_device_selection(adapt_dir: Path, config: dict) -> dict:
    updated = dict(config)
    baseline_env = _merge_env_mapping_with_defaults(updated.get("npu_baseline_env"), DEFAULT_NPU_BASELINE_ENV)
    perf_env = _merge_env_mapping_with_defaults(updated.get("npu_perf_env"), DEFAULT_NPU_PERF_ENV)
    configured_selected_npus = _normalize_device_id_list(updated.get("selected_npus"))
    configured_parallel_mode = _coerce_text(updated.get("parallel_mode"))
    configured_device_topology = _coerce_text(updated.get("device_topology"))

    def _should_preserve_multi_card_plan() -> bool:
        if len(configured_selected_npus) <= 1:
            return False
        if configured_parallel_mode in {"default", "manual_visible_devices", "tensor_parallel", "pipeline_parallel"}:
            return True
        if configured_device_topology and configured_device_topology not in {"all_visible_devices"} and not configured_device_topology.startswith("single_npu:"):
            return True
        return False

    def _select_single_visible_device(candidate_ids: list[str] | None = None) -> str:
        ranked_device_ids = _rank_idle_npu_ids()
        if candidate_ids:
            candidate_set = {str(device_id).strip() for device_id in candidate_ids if str(device_id).strip()}
            for device_id in ranked_device_ids:
                if device_id in candidate_set:
                    return device_id
            if candidate_set:
                return sorted(candidate_set, key=lambda device_id: int(device_id))[0]
        if ranked_device_ids:
            return ranked_device_ids[0]
        raise RuntimeError(f"未检测到可用 Ascend 单卡；请先设置 {NPU_VISIBLE_ENV_KEY}，或检查 `npu-smi info` 输出是否正常")

    visible_devices = baseline_env.get(NPU_VISIBLE_ENV_KEY) or perf_env.get(NPU_VISIBLE_ENV_KEY) or str(os.environ.get(NPU_VISIBLE_ENV_KEY) or "").strip()
    if visible_devices:
        visible_devices = str(visible_devices).strip()
        configured_visible_npus = _normalize_device_id_list(visible_devices)
        if len(configured_visible_npus) > 1:
            if _should_preserve_multi_card_plan():
                visible_devices = ",".join(configured_selected_npus or configured_visible_npus)
                print(f"[business][npu] preserving explicit multi-card plan {NPU_VISIBLE_ENV_KEY}={visible_devices}")
            else:
                selected_device = _select_single_visible_device(configured_visible_npus)
                print(f"[business][npu] narrowed {NPU_VISIBLE_ENV_KEY}={visible_devices} to single-card {selected_device}")
                visible_devices = selected_device
        else:
            print(f"[business][npu] using {NPU_VISIBLE_ENV_KEY}={visible_devices}")
    elif configured_parallel_mode in {"auto", "device_map_auto"} and configured_device_topology == "all_visible_devices":
        visible_devices = _select_single_visible_device()
        print(f"[business][npu] narrowed dynamic auto multicard plan to {NPU_VISIBLE_ENV_KEY}={visible_devices}")
    elif configured_selected_npus:
        if len(configured_selected_npus) > 1:
            if _should_preserve_multi_card_plan():
                visible_devices = ",".join(configured_selected_npus)
                print(f"[business][npu] preserving configured multi-card NPU set {configured_selected_npus}")
            else:
                visible_devices = _select_single_visible_device(configured_selected_npus)
                print(f"[business][npu] narrowed configured NPU set {configured_selected_npus} to single-card {visible_devices}")
        else:
            visible_devices = ",".join(configured_selected_npus)
            print(f"[business][npu] inherited {NPU_VISIBLE_ENV_KEY}={visible_devices} from business config")
    else:
        visible_devices = _select_single_visible_device()
        print(f"[business][npu] auto-selected {NPU_VISIBLE_ENV_KEY}={visible_devices} from npu-smi info")

    baseline_env[NPU_VISIBLE_ENV_KEY] = visible_devices
    perf_env[NPU_VISIBLE_ENV_KEY] = visible_devices
    updated["npu_baseline_env"] = baseline_env
    updated["npu_perf_env"] = perf_env

    selected_npus = [item.strip() for item in visible_devices.split(",") if item.strip()]
    updated["selected_npus"] = selected_npus
    if len(selected_npus) > 1:
        updated["parallel_mode"] = configured_parallel_mode if configured_parallel_mode and configured_parallel_mode != "single_card" else "manual_visible_devices"
        updated["device_topology"] = configured_device_topology or f"visible_devices:{','.join(selected_npus)}"
    else:
        updated["parallel_mode"] = "single_card"
        updated["device_topology"] = f"single_npu:{selected_npus[0]}" if selected_npus else "unknown"

    if updated != config:
        _write_config(adapt_dir, updated)
    return updated


def _ensure_business_eval_script(adapt_dir: Path) -> Path:
    if not BUSINESS_EVAL_TEMPLATE.exists():
        raise FileNotFoundError(f"缺少业务测评模板: {BUSINESS_EVAL_TEMPLATE}")
    script_path = adapt_dir / BUSINESS_EVAL_FILENAME
    template_content = _render_generated_python_script(BUSINESS_EVAL_TEMPLATE.read_text(encoding="utf-8"), tool=MANAGER_SCRIPT_RELATIVE_PATH)
    if not script_path.exists():
        script_path.write_text(template_content, encoding="utf-8")
    elif script_path.read_text(encoding="utf-8") != template_content:
        try:
            script_path.write_text(template_content, encoding="utf-8")
        except PermissionError:
            print(f"[business][warn] cannot refresh {script_path}; using existing file")
    return script_path


def _ensure_business_model_eval_script(adapt_dir: Path) -> Path:
    if not BUSINESS_MODEL_EVAL_TEMPLATE.exists():
        raise FileNotFoundError(f"缺少业务测评模板: {BUSINESS_MODEL_EVAL_TEMPLATE}")
    script_path = adapt_dir / BUSINESS_MODEL_EVAL_FILENAME
    template_content = _render_generated_python_script(BUSINESS_MODEL_EVAL_TEMPLATE.read_text(encoding="utf-8"), tool=MANAGER_SCRIPT_RELATIVE_PATH)
    managed_marker = "CURSOR-MANAGED-BUSINESS-MODEL-EVAL"
    # Preserve customized evaluators inside the adaptation directory.
    # Only files explicitly marked as managed templates are auto-refreshed.
    if not script_path.exists():
        script_path.write_text(template_content, encoding="utf-8")
    else:
        existing_content = script_path.read_text(encoding="utf-8")
        if managed_marker in existing_content and existing_content != template_content:
            try:
                script_path.write_text(template_content, encoding="utf-8")
            except PermissionError:
                print(f"[business][warn] cannot refresh {script_path}; using existing file")
    if BUSINESS_LEGACY_OLMO_HELPER_TEMPLATE.exists():
        helper_path = adapt_dir / BUSINESS_LEGACY_OLMO_HELPER_TEMPLATE.name
        helper_content = _render_generated_python_script(BUSINESS_LEGACY_OLMO_HELPER_TEMPLATE.read_text(encoding="utf-8"), tool=MANAGER_SCRIPT_RELATIVE_PATH)
        if not helper_path.exists():
            helper_path.write_text(helper_content, encoding="utf-8")
        elif helper_path.read_text(encoding="utf-8") != helper_content:
            try:
                helper_path.write_text(helper_content, encoding="utf-8")
            except PermissionError:
                print(f"[business][warn] cannot refresh {helper_path}; using existing file")
    return script_path


def _read_adaptation_context(adapt_dir: Path) -> str:
    chunks: list[str] = []
    # Phase-4 business profiling should reflect adaptation/runtime intent rather than
    # stage-3 optimization evidence. optimization_notes.json often contains generic
    # wikitext/logits/cosine-similarity metadata that can incorrectly reclassify
    # causal LM workloads as embedding similarity during business profile inference.
    for candidate_name in ("accuracy_run.py", "accuracy_run_perf.py", "demo.py", "README.md", "pyproject.toml"):
        candidate_path = adapt_dir / candidate_name
        if not candidate_path.exists():
            continue
        try:
            chunks.append(candidate_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
    return "\n".join(chunks)


def _resolve_local_transformers_config_source(model_id: str, adapt_dir: Path, config: dict[str, object] | None = None) -> str | None:
    local_models_dir = adapt_dir / "models"
    if not local_models_dir.is_dir():
        return None

    config_mapping = config if isinstance(config, dict) else _load_config(adapt_dir, missing_ok=True)
    preferred_sources = [
        config_mapping.get("model_source_override"),
        config_mapping.get("model_source"),
        config_mapping.get("tokenizer_source_override"),
        config_mapping.get("input_source_override"),
        config_mapping.get("base_model_id"),
    ]
    for source in preferred_sources:
        source_text = str(source or "").strip()
        if not source_text:
            continue
        resolved_path = _resolve_config_path(source_text, adapt_dir=adapt_dir)
        if resolved_path is not None and resolved_path.is_dir() and (resolved_path / "config.json").is_file():
            return str(resolved_path)
        for snapshot_dir in _resolve_required_snapshot_dirs_for_source(source_text, adapt_dir=adapt_dir, allow_hub_model_id_lookup=False):
            if (snapshot_dir / "config.json").is_file():
                return str(snapshot_dir)

    snapshot_dirs: list[Path] = []
    if model_id and "/" in model_id:
        org, name = model_id.split("/", 1)
        cache_dir = local_models_dir / f"models--{org.replace('/', '--')}--{name.replace('/', '--')}"
        snapshots_dir = cache_dir / "snapshots"
        if snapshots_dir.is_dir():
            snapshot_dirs.extend(sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), reverse=True))

    if not snapshot_dirs:
        for cache_dir in sorted(local_models_dir.glob("models--*")):
            snapshots_dir = cache_dir / "snapshots"
            if not snapshots_dir.is_dir():
                continue
            snapshot_dirs.extend(sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), reverse=True))

    for snapshot_dir in snapshot_dirs:
        if (snapshot_dir / "config.json").exists():
            return str(snapshot_dir)
    return None


def _iter_snapshot_asset_dirs(snapshot_dir: Path, *, max_depth: int = 4):
    if not snapshot_dir.exists():
        return
    yield snapshot_dir
    nested_dirs: list[Path] = []
    try:
        for candidate in snapshot_dir.rglob("*"):
            if not candidate.is_dir():
                continue
            try:
                depth = len(candidate.relative_to(snapshot_dir).parts)
            except ValueError:
                continue
            if depth > max_depth:
                continue
            nested_dirs.append(candidate)
    except Exception:
        return
    for candidate in sorted(nested_dirs, key=lambda path: (len(path.relative_to(snapshot_dir).parts), str(path))):
        yield candidate


def _snapshot_dir_has_model_assets(snapshot_dir: Path) -> bool:
    weight_patterns = (
        "model.safetensors",
        "model-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model-*.bin",
        "last.ckpt",
        "*.ckpt",
        "*.h5",
        "*.hdf5",
        "tf_model.h5",
        "flax_model.msgpack",
    )
    for pattern in weight_patterns:
        for candidate in snapshot_dir.glob(pattern):
            if candidate.exists():
                return True
    return False


def _snapshot_dir_has_input_assets(snapshot_dir: Path, input_kind: str) -> bool:
    if input_kind == "tokenizer":
        candidates = (
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
            "vocab.txt",
            "merges.txt",
            "sentencepiece.bpe.model",
            "spiece.model",
            "special_tokens_map.json",
        )
    elif input_kind == "image_processor":
        candidates = ("preprocessor_config.json", "processor_config.json")
    else:
        candidates = (
            "processor_config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
            "vocab.txt",
            "merges.txt",
            "special_tokens_map.json",
        )
    return any((snapshot_dir / candidate).exists() for candidate in candidates)


def _iter_local_snapshot_asset_dirs(model_id: str, adapt_dir: Path) -> list[Path]:
    local_models_dir = adapt_dir / "models"
    if not local_models_dir.is_dir():
        return []

    snapshot_dirs = _iter_local_snapshot_dirs_for_model_id(model_id, adapt_dir)
    if not snapshot_dirs:
        for cache_dir in sorted(local_models_dir.glob("models--*")):
            snapshots_dir = cache_dir / "snapshots"
            if not snapshots_dir.is_dir():
                continue
            snapshot_dirs.extend(sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), reverse=True))

    asset_dirs: list[Path] = []
    seen: set[str] = set()
    for snapshot_dir in snapshot_dirs:
        for asset_dir in _iter_snapshot_asset_dirs(snapshot_dir):
            asset_dir_str = str(asset_dir)
            if asset_dir_str in seen:
                continue
            seen.add(asset_dir_str)
            asset_dirs.append(asset_dir)
    return asset_dirs


def _iter_local_snapshot_dirs_for_model_id(model_id: str, adapt_dir: Path) -> list[Path]:
    local_models_dir = adapt_dir / "models"
    if not local_models_dir.is_dir():
        return []

    snapshot_dirs: list[Path] = []
    if model_id and "/" in model_id:
        org, name = model_id.split("/", 1)
        cache_dir = local_models_dir / f"models--{org.replace('/', '--')}--{name.replace('/', '--')}"
        snapshots_dir = cache_dir / "snapshots"
        if snapshots_dir.is_dir():
            snapshot_dirs.extend(sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), reverse=True))
    return snapshot_dirs


def _snapshot_dir_has_adapter_assets(snapshot_dir: Path) -> bool:
    return (snapshot_dir / "adapter_config.json").is_file()


def _extract_base_model_id_from_runtime_scripts(adapt_dir: Path) -> str:
    pattern = re.compile(r"(?m)^\s*BASE_MODEL_ID\s*=\s*['\"]([^'\"]+)['\"]")
    for candidate_name in ("accuracy_run_perf.py", "accuracy_run.py", "demo.py"):
        candidate_path = adapt_dir / candidate_name
        if not candidate_path.is_file():
            continue
        try:
            content = candidate_path.read_text(encoding="utf-8")
        except Exception:
            continue
        match = pattern.search(content)
        if match is not None:
            base_model_id = str(match.group(1) or "").strip()
            if base_model_id:
                return base_model_id
    return ""


def _extract_runtime_subfolder_from_scripts(adapt_dir: Path) -> str:
    pattern = re.compile(r"(?m)^\s*SUBFOLDER\s*=\s*['\"]([^'\"]+)['\"]")
    for candidate_name in ("accuracy_run_perf.py", "accuracy_run.py", "demo.py"):
        candidate_path = adapt_dir / candidate_name
        if not candidate_path.is_file():
            continue
        try:
            content = candidate_path.read_text(encoding="utf-8")
        except Exception:
            continue
        match = pattern.search(content)
        if match is not None:
            subfolder = str(match.group(1) or "").strip().strip("/")
            if subfolder:
                return subfolder
    return ""


def _try_normalize_snapshot_root(path: Path, *, local_models_dir: Path) -> Path | None:
    try:
        resolved_path = path.resolve()
        resolved_models_dir = local_models_dir.resolve()
    except Exception:
        resolved_path = path
        resolved_models_dir = local_models_dir
    try:
        relative = resolved_path.relative_to(resolved_models_dir)
    except Exception:
        return None
    parts = relative.parts
    if len(parts) >= 3 and parts[0].startswith("models--") and parts[1] == "snapshots":
        return resolved_models_dir / parts[0] / parts[1] / parts[2]
    if parts and parts[0].startswith("local_snapshot"):
        return resolved_models_dir / parts[0]
    return resolved_path if resolved_path.is_dir() else None


def _resolve_required_snapshot_dirs_for_source(source_value: object, *, adapt_dir: Path, allow_hub_model_id_lookup: bool = True) -> list[Path]:
    source_text = str(source_value or "").strip()
    if not source_text:
        return []
    local_models_dir = adapt_dir / "models"
    if not local_models_dir.is_dir():
        return []

    resolved_path = _resolve_config_path(source_text, adapt_dir=adapt_dir)
    if resolved_path is not None and resolved_path.exists():
        snapshot_root = _try_normalize_snapshot_root(resolved_path, local_models_dir=local_models_dir)
        if snapshot_root is not None and snapshot_root.exists():
            return [snapshot_root]
        return []

    if allow_hub_model_id_lookup and "/" in source_text:
        return _iter_local_snapshot_dirs_for_model_id(source_text, adapt_dir)
    return []


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        path_str = str(path)
        if path_str in seen:
            continue
        seen.add(path_str)
        deduped.append(path)
    return deduped


def _build_required_remote_snapshot_plan(adapt_dir: Path) -> dict[str, list[Path]]:
    local_models_dir = adapt_dir / "models"
    if not local_models_dir.is_dir():
        return {"weight_asset_dirs": [], "support_snapshot_dirs": []}

    config = _load_config(adapt_dir, missing_ok=True)
    model_id = str(config.get("model_id") or "").strip()
    explicit_model_sources = [
        config.get("model_source_override"),
        config.get("model_source"),
    ]
    runtime_base_model_id = _extract_base_model_id_from_runtime_scripts(adapt_dir)
    runtime_subfolder = _extract_runtime_subfolder_from_scripts(adapt_dir)
    base_model_sources = [
        config.get("base_model_id"),
        runtime_base_model_id,
        config.get("encoder_checkpoint_override"),
    ]
    input_sources = [
        config.get("tokenizer_source_override"),
        config.get("input_source_override"),
        config.get("input_source"),
        config.get("base_model_id"),
    ]

    explicit_weight_snapshot_dirs: list[Path] = []
    for source in explicit_model_sources:
        explicit_weight_snapshot_dirs.extend(_resolve_required_snapshot_dirs_for_source(source, adapt_dir=adapt_dir, allow_hub_model_id_lookup=False))
    explicit_weight_snapshot_dirs = _dedupe_paths(explicit_weight_snapshot_dirs)

    weight_snapshot_dirs: list[Path] = list(explicit_weight_snapshot_dirs)
    if not weight_snapshot_dirs:
        inferred_base_snapshot_dirs: list[Path] = []
        for source in base_model_sources:
            inferred_base_snapshot_dirs.extend(_resolve_required_snapshot_dirs_for_source(source, adapt_dir=adapt_dir, allow_hub_model_id_lookup=False))
        inferred_base_snapshot_dirs = _dedupe_paths(inferred_base_snapshot_dirs)
        if inferred_base_snapshot_dirs:
            weight_snapshot_dirs.extend(inferred_base_snapshot_dirs)

    model_snapshot_dirs = _iter_local_snapshot_dirs_for_model_id(model_id, adapt_dir) if model_id else []
    if model_snapshot_dirs and (
        any(_snapshot_dir_has_adapter_assets(snapshot_dir) for snapshot_dir in model_snapshot_dirs) or not weight_snapshot_dirs
    ):
        # Plain HuggingFace snapshots without LoRA/adapters still need to be mirrored
        # for remote phase-4 CUDA baseline. Otherwise business_eval.py can resolve the
        # local snapshot on NPU, while run-remote-cuda skips the remote weights entirely
        # and crashes with FileNotFoundError on the first checkpoint file.
        weight_snapshot_dirs.extend(model_snapshot_dirs)
    weight_snapshot_dirs = _dedupe_paths(weight_snapshot_dirs)

    support_snapshot_dirs = list(weight_snapshot_dirs)
    for source in input_sources:
        support_snapshot_dirs.extend(_resolve_required_snapshot_dirs_for_source(source, adapt_dir=adapt_dir, allow_hub_model_id_lookup=False))
    support_snapshot_dirs = _dedupe_paths(support_snapshot_dirs)

    weight_asset_dirs: list[Path] = []
    for snapshot_dir in weight_snapshot_dirs:
        preferred_subdir = snapshot_dir / runtime_subfolder if runtime_subfolder else None
        if preferred_subdir is not None and preferred_subdir.is_dir() and _snapshot_has_model_weight_assets(preferred_subdir):
            weight_asset_dirs.append(preferred_subdir)
            continue
        if snapshot_dir.name.startswith("local_snapshot"):
            if _snapshot_has_model_weight_assets(snapshot_dir):
                weight_asset_dirs.append(snapshot_dir)
            continue
        weight_asset_dirs.extend(_iter_snapshot_asset_dirs_with_weights(snapshot_dir))
    weight_asset_dirs = _dedupe_paths(weight_asset_dirs)
    return {
        "weight_asset_dirs": weight_asset_dirs,
        "support_snapshot_dirs": support_snapshot_dirs,
    }


def _normalize_local_snapshot_hint(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").strip().lower())


def _load_local_snapshot_config_payload(snapshot_dir: Path) -> dict[str, object]:
    config_path = snapshot_dir / "config.json"
    if not config_path.is_file():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _infer_local_custom_embedding_source_overrides(adapt_dir: Path) -> dict[str, str]:
    local_models_dir = adapt_dir / "models"
    if not local_models_dir.is_dir():
        return {}

    local_snapshot_dirs = sorted(path for path in local_models_dir.glob("local_snapshot*") if path.is_dir())
    if not local_snapshot_dirs:
        return {}

    custom_model_dir: Path | None = None
    custom_model_payload: dict[str, object] = {}
    for candidate_dir in local_snapshot_dirs:
        if not _snapshot_dir_has_model_assets(candidate_dir):
            continue
        payload = _load_local_snapshot_config_payload(candidate_dir)
        auto_map = payload.get("auto_map")
        has_custom_code = any(candidate_dir.glob("modeling_*.py")) or any(candidate_dir.glob("configuration_*.py"))
        if has_custom_code and isinstance(auto_map, dict) and auto_map:
            custom_model_dir = candidate_dir
            custom_model_payload = payload
            break

    if custom_model_dir is None:
        return {}

    overrides = {
        "model_source_override": _repo_relative_path_text(custom_model_dir, base_dir=adapt_dir),
    }
    if _snapshot_dir_has_input_assets(custom_model_dir, "tokenizer"):
        tokenizer_rel = _repo_relative_path_text(custom_model_dir, base_dir=adapt_dir)
        overrides["tokenizer_source_override"] = tokenizer_rel
        overrides["input_source_override"] = tokenizer_rel

    encoder_checkpoint = str(custom_model_payload.get("encoder_checkpoint") or "").strip()
    if not encoder_checkpoint:
        return overrides

    encoder_checkpoint_norm = _normalize_local_snapshot_hint(encoder_checkpoint)
    fallback_candidates: list[Path] = []
    matched_encoder_dir: Path | None = None
    for candidate_dir in local_snapshot_dirs:
        if candidate_dir == custom_model_dir or not _snapshot_dir_has_model_assets(candidate_dir):
            continue
        fallback_candidates.append(candidate_dir)
        payload = _load_local_snapshot_config_payload(candidate_dir)
        candidate_hints = [
            candidate_dir.name.replace("local_snapshot_", "", 1),
            payload.get("_name_or_path"),
            payload.get("model_type"),
        ]
        architectures = payload.get("architectures")
        if isinstance(architectures, (list, tuple, set)):
            candidate_hints.extend(architectures)
        normalized_hints = [hint for hint in (_normalize_local_snapshot_hint(item) for item in candidate_hints) if hint]
        if any(hint in encoder_checkpoint_norm or encoder_checkpoint_norm in hint for hint in normalized_hints):
            matched_encoder_dir = candidate_dir
            break

    if matched_encoder_dir is None and len(fallback_candidates) == 1:
        matched_encoder_dir = fallback_candidates[0]
    if matched_encoder_dir is not None:
        overrides["encoder_checkpoint_override"] = _repo_relative_path_text(matched_encoder_dir, base_dir=adapt_dir)
    return overrides


def _infer_local_custom_transformers_source_overrides(model_id: str, adapt_dir: Path) -> dict[str, str]:
    local_models_dir = adapt_dir / "models"
    if not local_models_dir.is_dir():
        return {}

    candidate_dirs: list[Path] = []
    if model_id:
        candidate_dirs.extend(_iter_local_snapshot_dirs_for_model_id(model_id, adapt_dir))
    candidate_dirs.extend(sorted((path for path in local_models_dir.glob("local_snapshot*") if path.is_dir()), reverse=True))

    seen: set[str] = set()
    for candidate_dir in candidate_dirs:
        candidate_str = str(candidate_dir)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if not (candidate_dir / "config.json").is_file():
            continue
        if not _snapshot_dir_has_model_assets(candidate_dir):
            continue
        has_custom_code = any(candidate_dir.glob("modeling_*.py")) or any(candidate_dir.glob("configuration_*.py"))
        if not has_custom_code:
            continue
        snapshot_rel = _repo_relative_path_text(candidate_dir, base_dir=adapt_dir)
        if not snapshot_rel:
            continue
        overrides = {"model_source_override": snapshot_rel}
        if _snapshot_dir_has_input_assets(candidate_dir, "tokenizer"):
            overrides["tokenizer_source_override"] = snapshot_rel
            overrides["input_source_override"] = snapshot_rel
        return overrides
    return {}


def _infer_component_source_overrides(model_id: str, adapt_dir: Path, profile: dict, context_text: str) -> dict[str, str]:
    open_llama_easylm_base_model_id = _resolve_open_llama_easylm_base_model_id(model_id)
    if open_llama_easylm_base_model_id:
        # EasyLM checkpoints lack standard HF transformers weights/config. Phase-4
        # should execute against the compatible OpenLLaMA PyTorch repo instead.
        return {
            "base_model_id": open_llama_easylm_base_model_id,
            "model_source_override": open_llama_easylm_base_model_id,
            "tokenizer_source_override": open_llama_easylm_base_model_id,
            "input_source_override": open_llama_easylm_base_model_id,
        }

    model_type = _coerce_text(profile.get("model_type")).lower()
    evaluation_profile = _coerce_text(profile.get("evaluation_profile")).lower()
    output_type_hint = _coerce_text(profile.get("output_type_hint")).lower()
    context_lower = context_text.lower()
    generic_custom_overrides = _infer_local_custom_transformers_source_overrides(model_id, adapt_dir)

    if model_type != "embedding":
        return generic_custom_overrides
    if evaluation_profile != "embedding_similarity":
        return generic_custom_overrides

    has_clip_text_encoder_signal = output_type_hint in {"text_embeddings", "sentence_embeddings", "cls_embeddings", "embeddings"} and any(
        signal in context_lower
        for signal in (
            "cliptextmodel",
            "clip text encoder",
            "text encoder component",
            'subfolder="text_encoder"',
            "subfolder='text_encoder'",
            'subfolder="tokenizer"',
            "subfolder='tokenizer'",
        )
    )
    if not has_clip_text_encoder_signal:
        embedding_overrides = _infer_local_custom_embedding_source_overrides(adapt_dir)
        return embedding_overrides or generic_custom_overrides

    text_encoder_dir: Path | None = None
    tokenizer_dir: Path | None = None
    for asset_dir in _iter_local_snapshot_asset_dirs(model_id, adapt_dir):
        if asset_dir.name == "text_encoder" and (asset_dir / "config.json").exists() and _snapshot_dir_has_model_assets(asset_dir):
            text_encoder_dir = asset_dir
        elif asset_dir.name == "tokenizer" and _snapshot_dir_has_input_assets(asset_dir, "tokenizer"):
            tokenizer_dir = asset_dir
        if text_encoder_dir is not None and tokenizer_dir is not None:
            break

    overrides: dict[str, str] = {}
    if text_encoder_dir is not None:
        overrides["model_source_override"] = _repo_relative_path_text(text_encoder_dir, base_dir=adapt_dir)
    if tokenizer_dir is not None:
        tokenizer_rel = _repo_relative_path_text(tokenizer_dir, base_dir=adapt_dir)
        overrides["tokenizer_source_override"] = tokenizer_rel
        overrides["input_source_override"] = tokenizer_rel
    if overrides:
        return overrides
    embedding_overrides = _infer_local_custom_embedding_source_overrides(adapt_dir)
    return embedding_overrides or generic_custom_overrides


def _infer_transformers_config_metadata(model_id: str, adapt_dir: Path, config: dict[str, object] | None = None) -> dict[str, object]:
    try:
        from transformers import AutoConfig
    except Exception:
        return {}

    config_source = _resolve_local_transformers_config_source(model_id, adapt_dir, config)
    if config_source is None:
        context_lower = _read_adaptation_context(adapt_dir).lower()
        if any(
            signal in context_lower
            for signal in (
                "segment_anything",
                "segment anything",
                "sam_model_registry",
                "sam.image_encoder",
            )
        ):
            return {}
    try:
        config = AutoConfig.from_pretrained(
            config_source or model_id,
            cache_dir=str(adapt_dir / "models"),
            trust_remote_code=True,
            local_files_only=bool(config_source),
        )
    except Exception:
        return {}

    architectures = getattr(config, "architectures", None)
    if isinstance(architectures, (list, tuple, set)):
        architectures_value = ", ".join(str(item).strip() for item in architectures if str(item).strip())
    else:
        architectures_value = str(architectures or "").strip()

    problem_type_value = str(getattr(config, "problem_type", "") or "").strip()

    try:
        num_labels = int(getattr(config, "num_labels", 0) or 0)
    except Exception:
        num_labels = 0
    if num_labels <= 0:
        id2label = getattr(config, "id2label", None)
        if isinstance(id2label, dict):
            num_labels = len(id2label)

    model_class_value = ""
    auto_map = getattr(config, "auto_map", None)
    if isinstance(auto_map, dict):
        for preferred_key in (
            "AutoModelForCausalLM",
            "AutoModelForSeq2SeqLM",
            "AutoModelForQuestionAnswering",
            "AutoModelForSequenceClassification",
            "AutoModelForTokenClassification",
            "AutoModelForVision2Seq",
            "AutoModel",
        ):
            class_ref = str(auto_map.get(preferred_key) or "").strip()
            if not class_ref:
                continue
            class_name = class_ref.split(".")[-1].strip()
            if class_name:
                model_class_value = class_name
                break

    return {
        "model_class": model_class_value or None,
        "architectures": architectures_value or None,
        "problem_type": problem_type_value or None,
        "num_labels": num_labels if num_labels > 0 else None,
    }


def _infer_transformers_num_labels(model_id: str, adapt_dir: Path, config: dict[str, object] | None = None) -> int:
    metadata = _infer_transformers_config_metadata(model_id, adapt_dir, config)
    try:
        return int(metadata.get("num_labels") or 0)
    except Exception:
        return 0


def _infer_business_profile_from_adaptation_context(model_id: str, adapt_dir: Path) -> dict:
    context_text = _read_adaptation_context(adapt_dir)
    if not context_text:
        return {}

    if _resolve_open_llama_easylm_base_model_id(model_id):
        # These adaptations intentionally reuse a standard OpenLLaMA PyTorch repo at
        # runtime. Their README/bench/optimization artifacts are prone to mixed
        # signals, so rely on dataset_mapping as the source of truth for phase-4.
        return {}

    context_lower = context_text.lower()
    model_id_lower = model_id.lower()
    has_timeseries_model_id_signal = any(
        signal in model_id_lower
        for signal in (
            "timeseries",
            "tinytimemixer",
            "chronos",
        )
    )
    has_timeseries_context_signal = any(
        signal in context_lower
        for signal in (
            "model type: timeseries",
            "time series forecasting",
            "tinytimemixerforprediction",
            "chronospipeline",
            "prediction_outputs",
            "past_values",
            "random (synthetic time series)",
            "synthetic time series",
        )
    )
    has_timeseries_signal = has_timeseries_model_id_signal or has_timeseries_context_signal
    if has_timeseries_signal:
        backend = "tinytimemixer" if "tinytimemixer" in context_lower or "tinytimemixer" in model_id_lower else "timeseries"
        return {
            "business_intent": "timeseries_forecasting",
            "business_intent_name": "Time Series Forecasting",
            "model_type": "timeseries",
            "model_type_name": "Time Series Forecasting",
            "dataset_key": "synthetic_timeseries",
            "evaluation_profile": "timeseries_forecasting",
            "primary_metric": "mae",
            "secondary_metrics": ["rmse", "latency_s"],
            "output_type_hint": "timeseries_predictions",
            "model_backend": backend,
        }

    has_asr_context_signal = any(
        signal in model_id_lower or signal in context_lower
        for signal in (
            "wav2vec",
            "wav2vec2",
            "wav2vec2forctc",
            "hubert",
            "whisper",
            "speechseq2seq",
            "ctc",
            "transcribe",
            "transcriptions",
            "automatic speech recognition",
            "speech recognition",
            "语音识别",
        )
    )
    has_acestep_diffusion_signal = any(
        signal in model_id_lower or signal in context_lower
        for signal in (
            "ace-step",
            "acestep",
            "acestepconditiongenerationmodel",
            "conditional diffusion",
            "condition generation",
            "latent_hidden_states",
            "simple_mode",
            "acestep-v15-turbo",
        )
    )
    if has_acestep_diffusion_signal:
        return {
            "business_intent": "diffusion",
            "business_intent_name": "Diffusion Latency",
            "model_type": "diffusion",
            "model_type_name": "Diffusion Model",
            "dataset_key": None,
            "evaluation_profile": "latency_only",
            "primary_metric": "latency_s",
            "secondary_metrics": ["throughput_qps"],
            "output_type_hint": "diffusion_latency",
            "model_backend": "accuracy_latency_bridge",
        }
    has_musicgen_tts_signal = not has_asr_context_signal and any(
        signal in model_id_lower or signal in context_lower
        for signal in (
            "musicgen",
            "musicgenforconditionalgeneration",
            "automodelfortexttowaveform",
            "texttowaveform",
            "text-to-audio",
            "text to audio",
            "text-to-music",
            "text to music",
        )
    )
    if has_musicgen_tts_signal:
        output_type_hint = "text_encoder_hidden_states" if "text_encoder_hidden_states" in context_lower or "text_encoder" in context_lower else "audio_waveform"
        return {
            "business_intent": "tts",
            "business_intent_name": "Text-to-Audio Latency",
            "model_type": "tts",
            "model_type_name": "Text-to-Speech",
            "dataset_key": None,
            "evaluation_profile": "latency_only",
            "primary_metric": "latency_s",
            "secondary_metrics": ["throughput_qps"],
            "output_type_hint": output_type_hint,
            "model_backend": "musicgen_text_encoder" if output_type_hint == "text_encoder_hidden_states" else "musicgen_waveform",
        }
    has_seq2seq_signal = any(
        signal in context_lower
        for signal in (
            "automodelforseq2seqlm",
            "t5forconditionalgeneration",
            "bartforconditionalgeneration",
            "pegasusforconditionalgeneration",
            "seq2seq",
            "sequence-to-sequence",
            "encoder-decoder",
        )
    )
    has_seq2seq_qa_signal = has_seq2seq_signal and any(
        signal in context_lower
        for signal in (
            "question answering",
            "question-answering",
            "qa_exact_match",
            "pubmed_qa",
            "rag-style",
            "rag style",
            "rag",
            "问答",
        )
    )
    has_discriminator_signal = any(
        signal in context_lower
        for signal in (
            "electraforpretraining",
            "automodelforpretraining",
            "discriminator",
            "判别器",
            "discriminator_logits",
            "token replacement",
            "replaced token",
            "real/fake",
            "真实 token",
            "被替换",
        )
    ) or ("electra" in model_id_lower and "discriminator" in model_id_lower)
    if has_discriminator_signal:
        return {
            "business_intent": "discriminator",
            "business_intent_name": "Token Discriminator",
            "model_type": "discriminator",
            "model_type_name": "Discriminator",
            "dataset_key": "wikitext",
            "evaluation_profile": "embedding_similarity",
            "primary_metric": "cosine_similarity",
            "secondary_metrics": ["latency_s", "throughput_qps", "match_rate"],
            "output_type_hint": "discriminator_logits",
        }

    has_feature_extraction_signal = any(signal in context_lower for signal in ("feature extraction", "feature-extraction"))
    has_keypoint_signal = any(
        signal in context_lower
        for signal in (
            "keypoint detection",
            "keypoint detector",
            "automodelforkeypointdetection",
            "superpoint",
            "local feature",
            "local-feature",
            '"output_type": "keypoints"',
            "detected keypoints",
        )
    ) or (has_feature_extraction_signal and any(signal in context_lower for signal in ("keypoint", "superpoint", "local feature", "local-feature")))
    if has_keypoint_signal:
        return {
            "business_intent": "vision_keypoint_detection",
            "business_intent_name": "Keypoint Detection",
            "model_type": "vision_keypoint_detection",
            "model_type_name": "Keypoint Detection",
            "dataset_key": "synthetic_keypoints",
            "evaluation_profile": "keypoint_repeatability",
            "primary_metric": "keypoint_repeatability",
            "secondary_metrics": ["latency_s", "throughput_qps", "num_keypoints"],
            "output_type_hint": "keypoints",
        }

    has_forced_alignment_signal = any(
        signal in model_id_lower or signal in context_lower
        for signal in (
            "forcedaligner",
            "forced aligner",
            "forced-aligner",
            "forced_aligner",
            "qwen3forcedaligner",
            "forced alignment",
            "音频-文本对齐",
            "强制对齐",
        )
    )
    if has_forced_alignment_signal:
        return {
            "business_intent": "forced_alignment",
            "business_intent_name": "Forced Alignment",
            "model_type": "asr",
            "model_type_name": "Speech Forced Alignment",
            "dataset_key": "librispeech",
            "evaluation_profile": "asr_wer",
            "primary_metric": "wer",
            "secondary_metrics": ["latency_s", "throughput_qps", "text_match_rate", "alignment_token_coverage"],
            "output_type_hint": "alignment_spans",
            "model_backend": "qwen_forced_aligner",
            "asr_task": "forced_align",
        }

    has_knowledge_graph_embedding_signal = any(
        signal in model_id_lower or signal in context_lower
        for signal in (
            "knowledge graph embedding",
            "knowledge-graph embedding",
            "knowledge_graph_embedding",
            "transe",
            "trans e",
            "transemodel",
            "synthetic_triplets",
            "triplet scores",
            "predict_tail",
            "relation_map.json",
        )
    )
    if has_knowledge_graph_embedding_signal:
        return {
            "business_intent": "knowledge_graph_embedding",
            "business_intent_name": "Knowledge Graph Embedding",
            "model_type": "knowledge_graph_embedding",
            "model_type_name": "Knowledge Graph Embedding",
            "dataset_key": "synthetic_triplets",
            "evaluation_profile": "embedding_similarity",
            "primary_metric": "cosine_similarity",
            "secondary_metrics": ["latency_s", "throughput_qps", "match_rate"],
            "output_type_hint": "triplet_scores",
            "model_backend": "transe",
        }

    has_audio_embedding_signal = not any(
        signal in context_lower
        for signal in (
            "ctc",
            "speechseq2seq",
            "transcribe",
            "transcription",
            "asr_wer",
            "whisperfor",
        )
    ) and any(
        signal in model_id_lower or signal in context_lower
        for signal in (
            "model type: audio_embedding",
            "audio embedding",
            "audio_embeddings",
            "speaker embedding",
            "speaker embeddings",
            "speaker verification",
            "speaker_encoder",
            "wespeaker",
            "voxceleb",
            "pyannote.audio",
        )
    )
    if has_audio_embedding_signal:
        return {
            "business_intent": "audio_embedding",
            "business_intent_name": "Audio Embedding",
            "model_type": "audio_embedding",
            "model_type_name": "Audio Embedding",
            "dataset_key": "librispeech",
            "evaluation_profile": "audio_embedding_similarity",
            "primary_metric": "cosine_similarity",
            "secondary_metrics": ["latency_s", "throughput_qps", "match_rate"],
            "output_type_hint": "audio_embeddings",
        }

    if has_asr_context_signal:
        multilingual_asr = resolve_multilingual_asr_dataset(model_id)
        resolved_dataset_key = str(multilingual_asr.get("dataset_key") or "").strip() if multilingual_asr else ""
        resolved_asr_language = str(multilingual_asr.get("asr_language") or "").strip() if multilingual_asr else ""
        has_nemo_asr_signal = any(
            signal in model_id_lower or signal in context_lower
            for signal in (
                "reazonspeech-nemo",
                "encdecrnntbpemodel",
                "nemo.collections.asr",
                ".nemo",
                "nemo asr",
            )
        )
        return {
            "business_intent": "asr",
            "business_intent_name": "Speech Recognition",
            "model_type": "asr",
            "model_type_name": "Automatic Speech Recognition",
            "dataset_key": resolved_dataset_key or "librispeech",
            "evaluation_profile": "asr_wer",
            "primary_metric": "wer",
            "secondary_metrics": ["latency_s", "throughput_qps", "text_match_rate"],
            "output_type_hint": "transcriptions",
            "model_backend": "nemo_asr" if has_nemo_asr_signal else None,
            "asr_language": resolved_asr_language,
            "asr_task": "transcribe",
        }

    has_sequence_classification_signal = any(
        signal in context_lower
        for signal in (
            "automodelforsequenceclassification",
            "forsequenceclassification",
            "sequence classification",
            "debertaforsequenceclassification",
            "bertforsequenceclassification",
            "robertaforsequenceclassification",
        )
    )
    has_nli_signal = (
        "mnli" in model_id_lower
        or "xnli" in model_id_lower
        or any(
            signal in context_lower
            for signal in (
                "natural language inference",
                "entailment",
                "contradiction",
                "mnli",
                "xnli",
            )
        )
    )
    if has_sequence_classification_signal and has_nli_signal:
        return {
            "business_intent": "natural_language_inference",
            "business_intent_name": "Natural Language Inference",
            "model_type": "classification",
            "model_type_name": "Classification",
            "dataset_key": "glue_mnli",
            "evaluation_profile": "classification_accuracy",
            "primary_metric": "accuracy",
            "secondary_metrics": ["latency_s", "match_rate", "macro_f1"],
            "output_type_hint": "class_labels",
        }

    has_strong_token_classification_signal = (
        any(
            signal in context_lower
            for signal in (
                "automodelfortokenclassification",
                "model type: token_classification",
                "token classification",
                "named entity recognition",
                "scientific ner",
                "scienceie",
                "实体识别",
                "命名实体识别",
            )
        )
        or "scienceie" in model_id_lower
    )
    has_masked_lm_embedding_hint = any(
        signal in context_lower
        for signal in (
            "automodelformaskedlm",
            "masked language model",
            "fill-mask",
            "fill mask",
            '"output_type": "cls_embeddings"',
            "output_type': 'cls_embeddings'",
            "cls_embeddings",
        )
    )
    has_token_classification_signal = has_strong_token_classification_signal and not (
        has_masked_lm_embedding_hint and "automodelfortokenclassification" not in context_lower and "named entity recognition" not in context_lower and "scienceie" not in context_lower
    )
    if has_token_classification_signal:
        is_bionlp_biomedical = "bionlp" in model_id_lower or any(
            signal in context_lower
            for signal in (
                "bionlp2004",
                "bio-entity recognition",
                "bio entity recognition",
                "cell_line",
                "cell line",
                "cell_type",
                "cell type",
            )
        )
        is_science_ie = "scienceie" in model_id_lower or any(
            signal in context_lower
            for signal in (
                "scienceie",
                "science ie",
                "scientific ner",
                "scientific literature",
                "material science",
                "materials science",
                "materials synthesis",
                "process extraction",
            )
        )
        return {
            "business_intent": "biomedical_token_classification" if is_bionlp_biomedical else "scientific_token_classification" if is_science_ie else "token_classification",
            "business_intent_name": "Biomedical Token Classification" if is_bionlp_biomedical else "Scientific Token Classification" if is_science_ie else "Token Classification",
            "model_type": "token_classification",
            "model_type_name": "Token Classification (NER)",
            "dataset_key": "bionlp2004" if is_bionlp_biomedical else "science_ie" if is_science_ie else "conll2003",
            "evaluation_profile": "token_classification_f1",
            "primary_metric": "f1",
            "secondary_metrics": ["latency_s", "precision", "recall", "match_rate"],
            "output_type_hint": "predicted_tokens",
        }

    has_image_matting_signal = (
        any(
            signal in context_lower
            for signal in (
                "birefnet",
                "briarmbg",
                "bria rmbg",
                "rmbg",
                "image matting",
                "alpha matte",
                "background removal",
                "background-removal",
                "alpha mask",
                "alpha masks",
                "trimap",
            )
        )
        or "birefnet" in model_id_lower
        or "rmbg" in model_id_lower
    )
    if has_image_matting_signal:
        return {
            "business_intent": "image_matting",
            "business_intent_name": "Image Matting",
            "model_type": "image_matting",
            "model_type_name": "Image Matting",
            "dataset_key": "synthetic_matting",
            "evaluation_profile": "matting_mae",
            "primary_metric": "mae",
            "secondary_metrics": ["latency_s", "throughput_qps", "cosine_similarity"],
            "output_type_hint": "alpha_masks",
        }

    has_clipseg_signal = "clipseg" in model_id_lower or any(
        signal in context_lower
        for signal in (
            "clipsegforimagesegmentation",
            "clipseg",
            "zero-shot image segmentation",
            "zero shot image segmentation",
            "image segmentation model based on clip",
        )
    )
    if has_clipseg_signal:
        return {
            "business_intent": "semantic_segmentation",
            "business_intent_name": "Prompted Image Segmentation Latency",
            "model_type": "semantic_segmentation",
            "model_type_name": "Semantic Segmentation",
            "dataset_key": "latency_only",
            "evaluation_profile": "latency_only",
            "primary_metric": "latency_s",
            "secondary_metrics": ["throughput_qps"],
            "output_type_hint": "segmentation_logits",
            "model_backend": "clipseg",
        }

    has_semantic_segmentation_signal = "segformer" in model_id_lower or any(
        signal in context_lower
        for signal in (
            "semantic segmentation",
            "semantic-segmentation",
            "segformerforsemanticsegmentation",
            "segmentation logits",
            "pixel-level classification",
            "pixel level classification",
        )
    )
    if has_semantic_segmentation_signal:
        return {
            "business_intent": "semantic_segmentation",
            "business_intent_name": "Semantic Segmentation Latency",
            "model_type": "semantic_segmentation",
            "model_type_name": "Semantic Segmentation",
            "dataset_key": "latency_only",
            "evaluation_profile": "latency_only",
            "primary_metric": "latency_s",
            "secondary_metrics": ["throughput_qps"],
            "output_type_hint": "segmentation_logits",
            "model_backend": "semantic_segmentation",
        }

    has_openvla_signal = any(
        signal in model_id_lower or signal in context_lower
        for signal in (
            "openvla",
            "vision-language-action",
            "vision language action",
            "openvlaforactionprediction",
            "action prediction",
            "predict_action",
            "unnorm_key",
            "bridge_orig",
            "robotics",
        )
    )
    if has_openvla_signal:
        return {
            "business_intent": "vision_language_action",
            "business_intent_name": "Vision-Language-Action Latency",
            "model_type": "vlm",
            "model_type_name": "Vision-Language-Action",
            "dataset_key": "latency_only",
            "evaluation_profile": "latency_only",
            "primary_metric": "latency_s",
            "secondary_metrics": ["throughput_qps"],
            "output_type_hint": "generated_action",
            "model_backend": "openvla_action_prediction",
        }

    has_zero_shot_vision_signal = (
        any(token in model_id_lower for token in ("clip", "siglip"))
        or any(
            token in context_lower
            for token in (
                "clipmodel",
                "clipprocessor",
                "siglipmodel",
                "sigliptokenizer",
                "siglipimageprocessor",
            )
        )
    ) and any(
        signal in context_lower
        for signal in (
            "zero-shot image classification",
            "zero shot image classification",
            "clipmodel",
            "clipprocessor",
            "siglipmodel",
            "sigliptokenizer",
            "siglipimageprocessor",
            "automodelforzeroshotimageclassification",
            "logits_per_image",
            "image-text matching",
            "image-text retrieval",
            "vision encoder",
            "vision config",
            "pixel_values",
            "a photo of ",
            "image embeddings shape",
        )
    )
    has_clip_retrieval_embedding_signal = any(
        signal in context_lower
        for signal in (
            "fashion item retrieval",
            "fashion retrieval",
            "product retrieval",
            "similarity search",
            "image retrieval",
            "instance retrieval",
            "visual retrieval",
            "retrieval benchmark",
        )
    )
    if has_zero_shot_vision_signal and not has_clip_retrieval_embedding_signal:
        inferred_num_labels = _infer_transformers_num_labels(model_id, adapt_dir)
        imagenet_like = "imagenet" in context_lower or "in1k" in context_lower or "num_classes=1000" in context_lower or "num_classes = 1000" in context_lower or "in1k" in model_id_lower or inferred_num_labels >= 1000
        return {
            "business_intent": "vision_classification",
            "business_intent_name": "Image Classification",
            "model_type": "vision_classification",
            "model_type_name": "Image Classification",
            "dataset_key": "imagenet" if imagenet_like else "cifar100",
            "evaluation_profile": "vision_topk_accuracy",
            "primary_metric": "top1_accuracy",
            "secondary_metrics": ["latency_s", "top5_accuracy", "match_rate"],
            "output_type_hint": "class_labels",
        }

    has_text_encoder_embedding_signal = (
        not has_asr_context_signal
        and not has_audio_embedding_signal
        and not any(
            signal in context_lower
            for signal in (
                "automodelformaskedlm",
                "masked language model",
                "fill-mask",
                "fill mask",
            )
        )
        and any(
            signal in context_lower
            for signal in (
                "cliptextmodel",
                "clip text encoder",
                "text encoder component",
                "this benchmark tests the clip text encoder component",
                'output_type": "text_embeddings"',
                "output_type': 'text_embeddings'",
                'output_type": "sentence_embeddings"',
                "output_type': 'sentence_embeddings'",
                '"output_type": "text_embeddings"',
                '"output_type": "sentence_embeddings"',
                '"output_type": "embeddings"',
                '"output_type": "cls_embeddings"',
                '"dataset": "wikitext"',
                '"evaluation_profile": "embedding_similarity"',
                '"primary_metric": "cosine_similarity"',
                '"model_type": "embedding"',
            )
        )
        and any(
            signal in context_lower
            for signal in (
                "wikitext",
                "embedding_similarity",
                "cosine_similarity",
                "text embedding",
                "text embeddings",
                "sentence embedding",
                "sentence embeddings",
                "last_hidden_state",
                "pooler_output",
                "mean pooling",
                "model type: bert",
                'model_type": "embedding"',
                "model_type': 'embedding'",
            )
        )
        and not any(
            signal in context_lower
            for signal in (
                "clipprocessor",
                "siglipimageprocessor",
                "pixel_values",
                "logits_per_image",
                "image-text matching",
                "image-text retrieval",
                "generated_images",
                "diffusers",
                "unet2dconditionmodel",
                "unet3dconditionmodel",
                "stable diffusion",
                "stablediffusionpipeline",
                "text-to-image",
                "text to image",
                '"output_type": "latents"',
                "output_type': 'latents'",
                "noise_pred",
                "load_lora_weights",
            )
        )
    )
    if has_text_encoder_embedding_signal:
        output_type_hint = "sentence_embeddings"
        if "text_embeddings" in context_lower:
            output_type_hint = "text_embeddings"
        elif "cls_embeddings" in context_lower:
            output_type_hint = "cls_embeddings"
        elif '"output_type": "embeddings"' in context_lower or "output_type': 'embeddings'" in context_lower:
            output_type_hint = "embeddings"
        biomedical_like = any(
            signal in model_id_lower or signal in context_lower
            for signal in (
                "pubmed",
                "biomed",
                "biomedical",
                "medical",
                "clinical",
                "radiology",
                "chest x-ray",
                "cxr",
            )
        )
        return {
            "business_intent": "embedding",
            "business_intent_name": "Text Embedding",
            "model_type": "embedding",
            "model_type_name": "Embedding",
            "dataset_key": "pubmed_qa" if biomedical_like else "wikitext",
            "evaluation_profile": "embedding_similarity",
            "primary_metric": "cosine_similarity",
            "secondary_metrics": ["latency_s", "throughput_qps", "match_rate"],
            "output_type_hint": output_type_hint,
        }

    has_ltx_diffusion_signal = "ltx-" in model_id_lower and any(
        signal in context_lower
        for signal in (
            "text-to-video",
            "text to video",
            "image-to-video",
            "image to video",
            "audio-video generation",
            "audio video generation",
            "audio-video joint generation",
            "audio video joint generation",
            "joint audio-video generation",
            "diffusion transformer",
            "denoising diffusion",
            "distilledpipeline",
            "x0model",
            "ltx_diffusers_pipeline",
        )
    )
    if has_ltx_diffusion_signal:
        return {
            "business_intent": "diffusion",
            "business_intent_name": "Diffusion Latency",
            "model_type": "diffusion",
            "model_type_name": "Diffusion Model",
            "dataset_key": None,
            "evaluation_profile": "latency_only",
            "primary_metric": "latency_s",
            "secondary_metrics": ["throughput_qps"],
            "output_type_hint": "diffusion_latency",
            "model_backend": "ltx_audio_video_diffusion",
        }

    has_video_generation_latency_signal = model_id_lower == "bytedance-research/phantom" or any(
        signal in context_lower
        for signal in (
            "video generation",
            "multi-modal video generation",
            "multimodal video generation",
            "video generation model",
            "video_generation_validation",
            "text-to-video",
            "text to video",
            "image-to-video",
            "image to video",
            "subject-to-video",
            "subject to video",
            "text-audio-to-video",
            "text audio to video",
            "text-image-audio-to-video",
            "text image audio to video",
            "audio-video generation",
            "audio video generation",
            "dit-based diffusion",
            "dit based diffusion",
        )
    )
    if has_video_generation_latency_signal:
        return {
            "business_intent": "video",
            "business_intent_name": "Video Generation Latency",
            "model_type": "video",
            "model_type_name": "Video Generation",
            "dataset_key": None,
            "evaluation_profile": "latency_only",
            "primary_metric": "latency_s",
            "secondary_metrics": ["throughput_qps"],
            "output_type_hint": "video_latency",
        }

    has_image_generation_latency_signal = any(
        signal in model_id_lower or signal in context_lower
        for signal in (
            "stablediffusionpipeline",
            "stable diffusion pipeline",
            "stable-diffusion",
            "stable diffusion",
            "sdxl",
            "text-to-image",
            "text to image",
            "image generation",
            "generated_images",
            "diffusers",
            "load_lora_weights",
            "unet2dconditionmodel",
            "unet3dconditionmodel",
            '"output_type": "latents"',
            "output_type': 'latents'",
            "noise_pred",
            "from_single_file",
        )
    )
    if has_image_generation_latency_signal:
        output_type_hint = "generated_images"
        if any(
            signal in context_lower
            for signal in (
                "unet2dconditionmodel",
                "unet3dconditionmodel",
                '"output_type": "latents"',
                "output_type': 'latents'",
                "noise_pred",
            )
        ):
            output_type_hint = "diffusion_latency"
        return {
            "business_intent": "diffusion",
            "business_intent_name": "Diffusion Latency",
            "model_type": "diffusion",
            "model_type_name": "Diffusion Model",
            "dataset_key": None,
            "evaluation_profile": "latency_only",
            "primary_metric": "latency_s",
            "secondary_metrics": ["throughput_qps"],
            "output_type_hint": output_type_hint,
        }

    has_vlm_captioning_signal = any(
        signal in model_id_lower or signal in context_lower
        for signal in (
            "blip",
            "image captioning",
            "image-captioning",
            "image caption",
            "图像描述",
            "blipforconditionalgeneration",
        )
    )
    has_vision_text_ocr_signal = any(
        signal in model_id_lower or signal in context_lower
        for signal in (
            "trocr",
            "visionencoderdecodermodel",
            "vision encoder decoder",
            "vision-encoder-decoder",
            "handwritten text recognition",
            "text recognition",
            "ocr",
        )
    ) and not any(signal in model_id_lower or signal in context_lower for signal in ("glm-ocr", "glm_ocr", "deepseek-ocr", "deepseekocr", "internvl", "llava", "qwen2vl", "qwen2_5_vl", "qwen3vl"))
    if has_vision_text_ocr_signal:
        return {
            "business_intent": "vision_text_ocr",
            "business_intent_name": "Vision Text OCR",
            "model_type": "vision_text_ocr",
            "model_type_name": "Vision Text OCR",
            "dataset_key": "synthetic_ocr",
            "evaluation_profile": "generation_exact_match",
            "primary_metric": "exact_match",
            "secondary_metrics": ["latency_s", "match_rate", "text_match_rate"],
            "output_type_hint": "generated_text",
        }
    if has_vlm_captioning_signal:
        return {
            "business_intent": "vlm",
            "business_intent_name": "Vision-Language Understanding",
            "model_type": "vlm",
            "model_type_name": "Vision-Language Model",
            "dataset_key": "scienceqa",
            "evaluation_profile": "vlm_accuracy",
            "primary_metric": "accuracy",
            "secondary_metrics": ["latency_s", "match_rate", "text_match_rate"],
            "output_type_hint": "generated_text",
        }

    has_explicit_vlm_signal = any(
        signal in model_id_lower or signal in context_lower
        for signal in (
            "glm-ocr",
            "glm_ocr",
            "glmocrforconditionalgeneration",
            "internvl",
            "internvlchatmodel",
            "image-text-to-text",
            "qwen2vlforconditionalgeneration",
            "qwen2_5_vlforconditionalgeneration",
            "qwen3vlforconditionalgeneration",
            "llavaforconditionalgeneration",
        )
    )
    if has_explicit_vlm_signal:
        return {
            "business_intent": "vlm",
            "business_intent_name": "Vision-Language Understanding",
            "model_type": "vlm",
            "model_type_name": "Vision-Language Model",
            "dataset_key": "scienceqa",
            "evaluation_profile": "vlm_accuracy",
            "primary_metric": "accuracy",
            "secondary_metrics": ["latency_s", "match_rate", "text_match_rate"],
            "output_type_hint": "generated_text",
        }

    has_causal_lm_signal = any(
        signal in context_lower
        for signal in (
            "model type: causal_lm",
            "automodelforcausallm",
            "causal language model",
            "task: text generation",
            "text generation",
            "文本生成",
        )
    )
    if has_causal_lm_signal:
        has_dna_sequence_signal = any(
            signal in model_id_lower or signal in context_lower
            for signal in (
                "biofm",
                "dna tokenizer",
                "dna sequence",
                "dna sequences",
                "dna 序列",
                "genome",
                "genomic",
                "genomics",
                "nucleotide",
                "nucleotides",
                "char_ords",
                "custom dna tokenizer",
                "字符级 tokenizer",
            )
        )
        has_biomedical_qa_signal = any(
            signal in context_lower
            for signal in (
                "pubmed_qa",
                "pubmed qa",
                "qa_exact_match",
                "medical question answering",
                "medical qa",
                "医疗问答",
                "生物医学",
                "问答",
            )
        ) or (not has_dna_sequence_signal and any(signal in model_id_lower for signal in ("bio", "pubmed", "clinical", "biomed", "medical", "medic")))
        base_signals = (
            "-base",
            "_base",
            " base",
            "base model",
            "基础模型",
        )
        instruct_signals = (
            "instruct",
            "chat",
            "assistant",
            "alpaca",
            "orca",
            "openorca",
            "dpo",
            "sft",
            "hermes",
            "dolphin",
            "tulu",
            "chatglm",
            "guard-gen",
            "guard gen",
            "qwen3guard",
            "instruction tuning",
            "instruction-tuned",
            "instruction tuned",
            "指令微调",
            "instruction-following",
            "instruction following",
            "role-playing",
            "role playing",
            "安全对话",
            "安全防护",
        )
        guard_generation_signals = (
            "guard-gen",
            "guard gen",
            "qwen3guard",
            "safety moderation",
            "moderation model",
            "内容审核",
            "安全审核",
            "prompt moderation",
            "response moderation",
            "安全防护",
        )
        has_explicit_base_signal = _contains_any_keyword_signal(model_id_lower, base_signals) or _contains_any_keyword_signal(context_lower, base_signals)
        has_strong_instruct_signal = _contains_any_keyword_signal(model_id_lower, instruct_signals) or _contains_any_keyword_signal(context_lower, instruct_signals)
        has_guard_generation_signal = _contains_any_keyword_signal(model_id_lower, guard_generation_signals) or _contains_any_keyword_signal(context_lower, guard_generation_signals)
        has_weak_conversation_signal = _contains_any_keyword_signal(context_lower, ("conversation", "conversational"))
        is_instruct_like = has_strong_instruct_signal or (has_weak_conversation_signal and not has_explicit_base_signal)
        prefer_mmlu_for_instruct = is_instruct_like and any(
            signal in model_id_lower or signal in context_lower
            for signal in (
                "llm-jp",
                "japanese",
                "日本語",
                "日语",
            )
        )
        if has_dna_sequence_signal:
            return {
                "business_intent": "dna_next_token",
                "business_intent_name": "DNA Next Token Prediction",
                "model_type": "causal_lm",
                "model_type_name": "Causal Language Model",
                "dataset_key": "synthetic_dna",
                "evaluation_profile": "classification_accuracy",
                "primary_metric": "accuracy",
                "secondary_metrics": ["latency_s", "top1_accuracy", "match_rate"],
                "output_type_hint": "class_labels",
            }
        if has_biomedical_qa_signal:
            return {
                "business_intent": "biomedical_qa",
                "business_intent_name": "Biomedical QA",
                "model_type": "causal_lm",
                "model_type_name": "Causal Language Model",
                "dataset_key": "pubmed_qa",
                "evaluation_profile": "qa_exact_match",
                "primary_metric": "exact_match",
                "secondary_metrics": ["latency_s", "f1", "match_rate"],
                "output_type_hint": "qa_answers",
            }
        if has_guard_generation_signal:
            return {
                "business_intent": "safety_guard_generation",
                "business_intent_name": "Safety Prompt Moderation",
                "model_type": "causal_lm",
                "model_type_name": "Causal Language Model",
                "dataset_key": "tweet_eval_offensive",
                "evaluation_profile": "classification_accuracy",
                "primary_metric": "accuracy",
                "secondary_metrics": ["latency_s", "top1_accuracy", "match_rate"],
                "output_type_hint": "class_labels",
            }
        resolved_dataset_key = "mmlu" if (prefer_mmlu_for_instruct or not is_instruct_like) else "gsm8k"
        return {
            "business_intent": "causal_lm_instruct" if is_instruct_like else "causal_lm_base",
            "business_intent_name": "Instruction Following" if is_instruct_like else "Base Model Reasoning",
            "model_type": "causal_lm",
            "model_type_name": "Causal Language Model",
            "dataset_key": resolved_dataset_key,
            "evaluation_profile": "mmlu" if resolved_dataset_key == "mmlu" else "generation_exact_match",
            "primary_metric": "accuracy" if resolved_dataset_key == "mmlu" else "exact_match",
            "secondary_metrics": ["latency_s", "throughput_qps", "match_rate"],
            "output_type_hint": "class_labels" if resolved_dataset_key == "mmlu" else "generated_text",
        }

    if has_seq2seq_qa_signal:
        return {
            "business_intent": "seq2seq_qa",
            "business_intent_name": "Seq2Seq Question Answering",
            "model_type": "seq2seq",
            "model_type_name": "Sequence-to-Sequence",
            "dataset_key": "pubmed_qa",
            "evaluation_profile": "qa_exact_match",
            "primary_metric": "exact_match",
            "secondary_metrics": ["latency_s", "f1", "match_rate"],
            "output_type_hint": "qa_answers",
        }

    has_reranker_signal = any(
        signal in model_id_lower or signal in context_lower
        for signal in (
            "cross-encoder",
            "cross encoder",
            "reranker",
            "rerank",
            "re-rank",
            "ms-marco",
            "ms marco",
            "colbert",
            "late interaction",
        )
    )
    if has_reranker_signal:
        return {
            "business_intent": "reranker",
            "business_intent_name": "Passage Reranking",
            "model_type": "reranker",
            "model_type_name": "Reranker",
            "dataset_key": "ms_marco",
            "evaluation_profile": "reranker_ndcg",
            "primary_metric": "ndcg_at_10",
            "secondary_metrics": ["latency_s", "mrr", "match_rate"],
            "output_type_hint": "relevance_scores",
        }

    has_protein_embedding_signal = any(
        signal in context_lower
        for signal in (
            "automodelformaskedlm",
            "masked language model",
            "fill-mask",
            "fill mask",
            "protein sequence",
            "protein sequences",
            "synthetic_protein",
            "amino acid",
            "esmfor",
            "esm2",
        )
    ) and any(signal in model_id_lower or signal in context_lower for signal in ("esm", "protein", "ur50"))
    has_protein_t5_embedding_signal = any(
        signal in model_id_lower or signal in context_lower
        for signal in (
            "prot_t5",
            "prott5",
            "uniref50",
            "protein",
            "amino acid",
        )
    ) and any(
        signal in context_lower
        for signal in (
            "protein embedding",
            "protein embeddings",
            "embedding generation",
            "embeddings",
            "protein sequence",
            "protein sequences",
            "t5 (encoder-decoder)",
            "encoder-decoder",
            "seq2seq",
        )
    )
    if has_protein_embedding_signal or has_protein_t5_embedding_signal:
        return {
            "business_intent": "protein_embedding",
            "business_intent_name": "Protein Embedding",
            "model_type": "embedding",
            "model_type_name": "Embedding",
            "dataset_key": "synthetic_protein",
            "evaluation_profile": "embedding_similarity",
            "primary_metric": "cosine_similarity",
            "secondary_metrics": ["latency_s", "throughput_qps", "match_rate"],
            "output_type_hint": "embeddings",
        }

    masked_lm_biomedical_like = any(
        signal in model_id_lower or signal in context_lower
        for signal in (
            "pubmed",
            "biomed",
            "biomedical",
            "medical",
            "clinical",
            "disease",
            "chemprot",
        )
    )
    prefer_bert_family_embedding_profile = has_masked_lm_embedding_hint and any(
        signal in model_id_lower or signal in context_lower
        for signal in (
            "deberta",
            "roberta",
            "modernbert",
            "longformer",
            "distilbert",
            "albert",
            "bert",
        )
    ) and not any(
        signal in model_id_lower or signal in context_lower
        for signal in (
            "esm",
            "protein",
            "ur50",
            "amino acid",
            "electraforpretraining",
            "discriminator",
            "real/fake",
        )
    )
    if prefer_bert_family_embedding_profile:
        output_type_hint = "cls_embeddings" if "cls_embeddings" in context_lower else "embeddings"
        return {
            "business_intent": "embedding",
            "business_intent_name": "Text Embedding",
            "model_type": "embedding",
            "model_type_name": "Embedding",
            "dataset_key": "pubmed_qa" if masked_lm_biomedical_like else "wikitext",
            "evaluation_profile": "embedding_similarity",
            "primary_metric": "cosine_similarity",
            "secondary_metrics": ["latency_s", "throughput_qps", "match_rate"],
            "output_type_hint": output_type_hint,
        }

    has_generic_masked_lm_business_signal = has_masked_lm_embedding_hint and not any(
        signal in model_id_lower or signal in context_lower
        for signal in (
            "esm",
            "protein",
            "ur50",
            "amino acid",
            "automodelfortokenclassification",
            "named entity recognition",
            "scientific ner",
            "scienceie",
            "automodelforquestionanswering",
            "question answering",
            "automodelforsequenceclassification",
            "sequence classification",
            "cross-encoder",
            "cross encoder",
            "reranker",
            "rerank",
            "re-rank",
            "ms-marco",
            "ms marco",
            "colbert",
            "electraforpretraining",
            "discriminator",
            "real/fake",
        )
    )
    if has_generic_masked_lm_business_signal:
        return {
            "business_intent": "masked_language_modeling",
            "business_intent_name": "Masked Token Prediction",
            "model_type": "masked_lm",
            "model_type_name": "Masked Language Model",
            "dataset_key": "pubmed_qa" if masked_lm_biomedical_like else "wikitext",
            "evaluation_profile": "classification_accuracy",
            "primary_metric": "accuracy",
            "secondary_metrics": ["latency_s", "match_rate"],
            "output_type_hint": "predicted_tokens",
            "model_backend": "masked_lm",
        }

    has_generic_masked_lm_embedding_signal = has_masked_lm_embedding_hint and not any(
        signal in context_lower
        for signal in (
            "automodelfortokenclassification",
            "named entity recognition",
            "scientific ner",
            "scienceie",
            "automodelforquestionanswering",
            "question answering",
            "automodelforsequenceclassification",
            "sequence classification",
            "cross-encoder",
            "cross encoder",
            "reranker",
            "rerank",
            "re-rank",
            "ms-marco",
            "ms marco",
            "colbert",
            "electraforpretraining",
            "discriminator",
            "real/fake",
        )
    )
    if has_generic_masked_lm_embedding_signal:
        output_type_hint = "cls_embeddings" if "cls_embeddings" in context_lower else "embeddings"
        return {
            "business_intent": "embedding",
            "business_intent_name": "Text Embedding",
            "model_type": "embedding",
            "model_type_name": "Embedding",
            "dataset_key": "wikitext",
            "evaluation_profile": "embedding_similarity",
            "primary_metric": "cosine_similarity",
            "secondary_metrics": ["latency_s", "throughput_qps", "match_rate"],
            "output_type_hint": output_type_hint,
        }

    has_multimodal_embedding_signal = not any(
        signal in context_lower
        for signal in (
            "clipmodel",
            "open_clip",
            "openclip",
            "zero-shot-image-classification",
            "zeroshotimageclassification",
            "logits_per_image",
            "vision classification",
        )
    ) and any(
        signal in model_id_lower or signal in context_lower
        for signal in (
            "qwen3-vl-embedding",
            "qwen3 vl embedding",
            "vision-language embedding",
            "visual-language embedding",
            "visual-text embedding",
            "vision-text embedding",
            "multimodal embedding",
            "multi-modal embedding",
            "vlm embedding",
            "joint embedding",
        )
    )
    if has_multimodal_embedding_signal:
        return {
            "business_intent": "embedding",
            "business_intent_name": "Text Embedding",
            "model_type": "embedding",
            "model_type_name": "Embedding",
            "dataset_key": "wikitext",
            "evaluation_profile": "embedding_similarity",
            "primary_metric": "cosine_similarity",
            "secondary_metrics": ["latency_s", "throughput_qps", "match_rate"],
            "output_type_hint": "embeddings",
        }

    has_vlm_context_signal = any(
        signal in model_id_lower or signal in context_lower
        for signal in (
            "glm-ocr",
            "glm_ocr",
            "glmocrforconditionalgeneration",
            "glm46vprocessor",
            "internvl",
            "internvlchatmodel",
            "vision-language model",
            "vision-language chat model",
            "visual language model",
            "visual language chat model",
            "image-text-to-text",
            "vlm_accuracy",
            "scienceqa",
            "qwen2vlforconditionalgeneration",
            "qwen2_5_vlforconditionalgeneration",
            "qwen3vlforconditionalgeneration",
            "llavaforconditionalgeneration",
        )
    )
    has_face_recognition_signal = any(
        signal in model_id_lower or signal in context_lower
        for signal in (
            "face recognition",
            "face-recognition",
            "face_recognition",
            "face embeddings",
            "face_embeddings",
            "lfw",
        )
    )
    if has_face_recognition_signal:
        return {
            "business_intent": "vision_embedding",
            "business_intent_name": "Image Embedding",
            "model_type": "vision_embedding",
            "model_type_name": "Image Embedding",
            "dataset_key": "cifar100",
            "evaluation_profile": "embedding_similarity",
            "primary_metric": "cosine_similarity",
            "secondary_metrics": ["latency_s", "throughput_qps", "match_rate"],
            "output_type_hint": "image_embeddings",
        }
    if has_vlm_context_signal:
        return {
            "business_intent": "vlm",
            "business_intent_name": "Vision-Language Understanding",
            "model_type": "vlm",
            "model_type_name": "Vision-Language Model",
            "dataset_key": "scienceqa",
            "evaluation_profile": "vlm_accuracy",
            "primary_metric": "accuracy",
            "secondary_metrics": ["latency_s", "match_rate", "text_match_rate"],
            "output_type_hint": "generated_text",
        }
    has_embedding_signal = (
        not has_vlm_context_signal
        and not has_asr_context_signal
        and not has_audio_embedding_signal
        and not any(
            signal in context_lower
            for signal in (
                "image encoder",
                "image embeddings",
                "image_embeddings",
                "segment_anything",
                "segment anything",
                "sam_model_registry",
                "sam.image_encoder",
            )
        )
        and any(
            signal in context_lower
            for signal in (
                "model type: embedding",
                "text embedding",
                "text embeddings",
                "sentence embedding",
                "sentence embeddings",
                "semantic similarity",
                '"output_type": "text_embeddings"',
                "output_type': 'text_embeddings'",
                'output_type": "sentence_embeddings"',
                "output_type': 'sentence_embeddings'",
                "embedding shape",
                "mean pooling",
                "mean pooling for sentence embeddings",
            )
        )
    )
    if has_embedding_signal:
        biomedical_like = any(
            signal in model_id_lower or signal in context_lower
            for signal in (
                "pubmed",
                "biomed",
                "biomedical",
                "medical",
                "clinical",
                "radiology",
                "chest x-ray",
                "cxr",
            )
        )
        return {
            "business_intent": "embedding",
            "business_intent_name": "Text Embedding",
            "model_type": "embedding",
            "model_type_name": "Embedding",
            "dataset_key": "pubmed_qa" if biomedical_like else "wikitext",
            "evaluation_profile": "embedding_similarity",
            "primary_metric": "cosine_similarity",
            "secondary_metrics": ["latency_s", "throughput_qps", "match_rate"],
            "output_type_hint": "sentence_embeddings",
        }

    has_segment_anything_signal = any(
        signal in context_lower
        for signal in (
            "segment_anything",
            "segment anything",
            "sam_model_registry",
            "sam.image_encoder",
            "sampredictor",
        )
    )
    has_dino_vision_embedding_signal = any(signal in model_id_lower or signal in context_lower for signal in ("dinov2", "dinov2model", "dino-vit", "dinovit")) and any(
        signal in context_lower
        for signal in (
            "image feature extraction",
            "feature extraction",
            "last_hidden_state",
            "autoimageprocessor",
            "pixel_values",
        )
    )
    has_vision_embedding_signal = (
        has_segment_anything_signal
        or has_dino_vision_embedding_signal
        or any(
            signal in context_lower
            for signal in (
                "image encoder",
                "image embeddings",
                "image_embeddings",
                "face recognition",
                "face-recognition",
                "face_recognition",
                "face embeddings",
                "face_embeddings",
                "lfw",
                '"output_type": "image_embeddings"',
                "output_type': 'image_embeddings'",
            )
        )
    )
    if has_vision_embedding_signal or ("sam-vit" in model_id_lower and "classification" not in context_lower):
        overrides = {
            "business_intent": "vision_embedding",
            "business_intent_name": "Image Embedding",
            "model_type": "vision_embedding",
            "model_type_name": "Image Embedding",
            "dataset_key": "cifar100",
            "evaluation_profile": "embedding_similarity",
            "primary_metric": "cosine_similarity",
            "secondary_metrics": ["latency_s", "throughput_qps", "match_rate"],
            "output_type_hint": "image_embeddings",
        }
        if has_segment_anything_signal or "sam-vit" in model_id_lower:
            overrides["model_backend"] = "segment_anything_image_encoder"
        return overrides

    has_matting_signal = any(
        signal in context_lower
        for signal in (
            "image matting",
            "vitmatte",
            "trimap",
            "alpha mask",
            "alpha_masks",
        )
    )
    if has_matting_signal:
        return {
            "business_intent": "image_matting",
            "business_intent_name": "Image Matting",
            "model_type": "image_matting",
            "model_type_name": "Image Matting",
            "dataset_key": "synthetic_matting",
            "evaluation_profile": "matting_mae",
            "primary_metric": "mae",
            "secondary_metrics": ["latency_s", "throughput_qps", "cosine_similarity"],
            "output_type_hint": "alpha_masks",
        }

    has_image_to_image_latency_signal = any(
        signal in context_lower
        for signal in (
            "image-to-image",
            "image to image",
            "rrdbnet",
            "esrgan",
            "material map",
            "pbr material",
            "normal map generator",
            "franken map generator",
            "normal_map_generator",
            "franken_map_generator",
        )
    )
    if has_image_to_image_latency_signal:
        return {
            "business_intent": "diffusion",
            "business_intent_name": "Image-to-Image Latency",
            "model_type": "diffusion",
            "model_type_name": "Image-to-Image Generation",
            "dataset_key": None,
            "evaluation_profile": "latency_only",
            "primary_metric": "latency_s",
            "secondary_metrics": ["throughput_qps"],
            "output_type_hint": "image_tensors",
        }

    has_detection_signal = any(
        signal in context_lower
        for signal in (
            "vision_detection",
            "object detection",
            "ultralytics",
            "yolo",
            "automodelforobjectdetection",
        )
    )
    has_colour_checker_signal = any(
        signal in model_id_lower or signal in context_lower
        for signal in (
            "colour-checker",
            "color-checker",
            "colour checker",
            "color checker",
            "colorcheckerclassic24",
            "colour rendition charts detection",
        )
    )
    has_fairface_age_signal = "fairface" in context_lower and any(
        signal in context_lower
        for signal in (
            "age detection",
            "age_detection",
            "image classification",
            "automodelforimageclassification",
            "vitforimageclassification",
        )
    )
    if has_colour_checker_signal and has_detection_signal:
        return {
            "business_intent": "colour_checker_detection",
            "business_intent_name": "Colour Checker Detection",
            "model_type": "vision_detection",
            "model_type_name": "Object Detection",
            "dataset_key": "synthetic_colour_checker",
            "evaluation_profile": "detection_map",
            "primary_metric": "mAP",
            "secondary_metrics": ["latency_s", "map50", "match_rate"],
            "output_type_hint": "detection_boxes",
            "model_backend": "ultralytics_yolo",
            "detection_target_labels": ["ColorCheckerClassic24"],
            "detection_fixed_label_id": 0,
        }
    if has_fairface_age_signal:
        return {
            "business_intent": "face_age_classification",
            "business_intent_name": "Face Age Classification",
            "model_type": "vision_classification",
            "model_type_name": "Image Classification",
            "dataset_key": "fairface",
            "evaluation_profile": "vision_topk_accuracy",
            "primary_metric": "top1_accuracy",
            "secondary_metrics": ["latency_s", "top5_accuracy", "match_rate"],
            "output_type_hint": "class_labels",
        }
    has_table_transformer_detection_signal = "table-transformer-detection" in model_id_lower or (
        "tabletransformerforobjectdetection" in context_lower
        and "structure-recognition" not in model_id_lower
        and "table column" not in context_lower
        and "table row" not in context_lower
    )
    if has_table_transformer_detection_signal:
        return {
            "business_intent": "table_detection",
            "business_intent_name": "Table Detection",
            "model_type": "vision_detection",
            "model_type_name": "Object Detection",
            "dataset_key": "pubtables_detection_1500",
            "evaluation_profile": "detection_map",
            "primary_metric": "mAP",
            "secondary_metrics": ["latency_s", "map50", "match_rate"],
            "output_type_hint": "detection_boxes",
            "detection_target_labels": ["table"],
            "detection_fixed_label_id": 0,
        }
    if has_detection_signal:
        overrides: dict[str, object] = {
            "business_intent": "vision_detection",
            "business_intent_name": "Object Detection",
            "model_type": "vision_detection",
            "model_type_name": "Object Detection",
            "dataset_key": "coco",
            "evaluation_profile": "detection_map",
            "primary_metric": "mAP",
            "secondary_metrics": ["latency_s", "map50", "match_rate"],
            "output_type_hint": "detection_boxes",
        }
        if any(signal in context_lower for signal in ("ultralytics", "yolo")):
            overrides["model_backend"] = "ultralytics_yolo"

        # Prefer the person checkpoint for COCO because it aligns with an available
        # detection label in the business dataset, unlike the face-only default.
        if model_id == "Bingsu/adetailer" or "person_yolov8n-seg.pt" in context_text:
            overrides["model_file"] = "person_yolov8n-seg.pt"
            overrides["detection_target_labels"] = ["person"]
        return overrides

    has_timm_signal = any(
        signal in context_lower
        for signal in (
            "import timm",
            "timm.create_model",
            "resolve_model_data_config",
            "create_transform",
        )
    )
    has_vision_classification_signal = has_timm_signal or any(
        signal in context_lower
        for signal in (
            "vision_classification",
            "image classification",
            "automodelforimageclassification",
        )
    )
    if has_vision_classification_signal:
        inferred_num_labels = 0 if has_timm_signal else _infer_transformers_num_labels(model_id, adapt_dir, config)
        imagenet_like = "imagenet" in context_lower or "in1k" in context_lower or "num_classes=1000" in context_lower or "num_classes = 1000" in context_lower or "in1k" in model_id.lower() or inferred_num_labels >= 1000
        overrides = {
            "business_intent": "vision_classification",
            "business_intent_name": "Image Classification",
            "model_type": "vision_classification",
            "model_type_name": "Image Classification",
            "dataset_key": "imagenet" if imagenet_like else "cifar100",
            "evaluation_profile": "vision_topk_accuracy",
            "primary_metric": "top1_accuracy",
            "secondary_metrics": ["latency_s", "top5_accuracy", "match_rate"],
            "output_type_hint": "class_labels",
        }
        if has_timm_signal:
            overrides["model_backend"] = "timm_image_classification"
        return overrides

    return {}


def _resolve_business_profile(model_id: str, config: dict, adapt_dir: Path) -> dict:
    transformers_metadata = _infer_transformers_config_metadata(model_id, adapt_dir, config)
    optimization_notes = _load_optimization_notes(adapt_dir)
    model_class = _first_text(config, "model_class") or _coerce_text(transformers_metadata.get("model_class"))
    architectures = _first_text(config, "architectures") or _coerce_text(transformers_metadata.get("architectures"))
    problem_type = _first_text(config, "problem_type") or _coerce_text(transformers_metadata.get("problem_type"))
    num_labels = config.get("num_labels")
    try:
        parsed_num_labels = int(num_labels)
    except Exception:
        parsed_num_labels = 0
    if parsed_num_labels <= 0:
        num_labels = transformers_metadata.get("num_labels")
    inferred = get_business_benchmark_profile(
        model_id,
        model_class,
        architectures=architectures,
        problem_type=problem_type,
        num_labels=num_labels,
    )
    context_overrides = _infer_business_profile_from_adaptation_context(model_id, adapt_dir)
    datasets_dir = PROJECT_ROOT / "datasets"
    # business_benchmark_config.json 中的 dataset/model_type/... 属于自动字段；
    # 重新画像时只允许 *_override 锁定人工选择，避免旧自动字段把新规则永久锁死。
    context_eval_profile = _coerce_text(context_overrides.get("evaluation_profile"))
    explicit_dataset_override = _first_text(config, "dataset_override")
    context_dataset_key = _coerce_text(context_overrides.get("dataset_key"))
    if context_eval_profile == "latency_only" and not explicit_dataset_override and not context_dataset_key:
        dataset_key = "latency_only"
    else:
        dataset_key = explicit_dataset_override or context_dataset_key or _coerce_text(inferred.get("dataset_key")) or None
    secondary_metrics = _first_list(config, "secondary_metrics_override")
    if not secondary_metrics:
        secondary_metrics = [str(item).strip() for item in list(context_overrides.get("secondary_metrics") or inferred.get("secondary_metrics") or []) if str(item).strip()]
    profile = {
        **inferred,
        "model_id": model_id,
        "business_intent": _first_text(config, "business_intent_override") or _coerce_text(context_overrides.get("business_intent")) or _coerce_text(inferred.get("business_intent")),
        "business_intent_name": _first_text(config, "business_intent_name_override") or _coerce_text(context_overrides.get("business_intent_name")) or _coerce_text(inferred.get("business_intent_name")),
        "model_type": _first_text(config, "model_type_override") or _coerce_text(context_overrides.get("model_type")) or _coerce_text(inferred.get("model_type")),
        "model_type_name": _first_text(config, "model_type_name_override") or _coerce_text(context_overrides.get("model_type_name")) or _coerce_text(inferred.get("model_type_name")),
        "dataset_key": dataset_key,
        "dataset_required": _dataset_requires_local_path(dataset_key),
        "evaluation_profile": _first_text(config, "evaluation_profile_override") or _coerce_text(context_overrides.get("evaluation_profile")) or _coerce_text(inferred.get("evaluation_profile")),
        "primary_metric": _first_text(config, "primary_metric_override") or _coerce_text(context_overrides.get("primary_metric")) or _coerce_text(inferred.get("primary_metric")),
        "secondary_metrics": secondary_metrics,
        "output_type_hint": _first_text(config, "output_type_hint_override") or _coerce_text(context_overrides.get("output_type_hint")) or _coerce_text(inferred.get("output_type_hint")),
        "model_backend": _first_text(config, "model_backend_override") or _coerce_text(context_overrides.get("model_backend")),
        "model_file": _first_text(config, "model_file_override") or _coerce_text(context_overrides.get("model_file")),
        "detection_target_labels": list(config.get("detection_target_labels_override") or context_overrides.get("detection_target_labels") or config.get("detection_target_labels") or []),
        "asr_task": _first_text(config, "asr_task_override") or _coerce_text(context_overrides.get("asr_task")) or _coerce_text(inferred.get("asr_task")),
        "asr_language": _first_text(config, "asr_language_override") or _coerce_text(context_overrides.get("asr_language")) or _coerce_text(inferred.get("asr_language")),
    }
    has_explicit_model_type_override = bool(_first_text(config, "model_type_override"))
    inferred_model_type = _coerce_text(inferred.get("model_type"))
    inferred_is_explicit_non_causal = inferred_model_type and inferred_model_type not in {"", "causal_lm", "biomedical_nlp"}
    context_forced_causal = _coerce_text(context_overrides.get("model_type")) == "causal_lm"
    context_has_explicit_generation_contract = context_forced_causal and _coerce_text(context_overrides.get("evaluation_profile")) not in {"", "embedding_similarity"}
    if not has_explicit_model_type_override and inferred_is_explicit_non_causal and profile["model_type"] == "causal_lm" and context_forced_causal and not context_has_explicit_generation_contract:
        for key in (
            "business_intent",
            "business_intent_name",
            "model_type",
            "model_type_name",
            "dataset_key",
            "dataset_required",
            "evaluation_profile",
            "primary_metric",
            "secondary_metrics",
            "output_type_hint",
        ):
            if key in inferred:
                profile[key] = inferred[key]
    if profile["model_type"] == "biomedical_nlp" and not context_has_explicit_generation_contract:
        if profile["evaluation_profile"] == "embedding_similarity":
            profile["model_type"] = "embedding"
            profile["model_type_name"] = "Text Embedding"
            normalized_dataset_key = _coerce_text(profile.get("dataset_key"))
            # Biomedical encoders often carry legacy wikitext hints in accuracy_run.py.
            # For phase-4 business eval, prefer biomedical text when no explicit override
            # was requested, otherwise we can waste a full CUDA round on the wrong workload.
            if not explicit_dataset_override and normalized_dataset_key in {"", "wikitext"}:
                profile["dataset_key"] = "pubmed_qa"
                profile["dataset_required"] = _dataset_requires_local_path("pubmed_qa")
        else:
            profile["model_type"] = "causal_lm"
            profile["model_type_name"] = "Causal Language Model"
    inherited_stage3_profile = _extract_stage3_phase4_profile_contract(adapt_dir, optimization_notes)
    inherited_stage3_dataset = _coerce_text(inherited_stage3_profile.get("dataset_key"))
    inherited_stage3_output_type = _coerce_text(inherited_stage3_profile.get("output_type_hint"))
    current_dataset_key = _coerce_text(profile.get("dataset_key"))
    current_model_type = _coerce_text(profile.get("model_type"))
    should_apply_stage3_profile = current_model_type in {"causal_lm", "biomedical_nlp"}
    if (
        inherited_stage3_dataset == "synthetic_3d"
        and inherited_stage3_output_type == "latent_embeddings"
        and current_dataset_key in {"", "wikitext"}
        and current_model_type in {"", "causal_lm", "biomedical_nlp", "discriminator"}
    ):
        should_apply_stage3_profile = True
    if inherited_stage3_profile and not has_explicit_model_type_override and not context_overrides and should_apply_stage3_profile:
        for key in (
            "business_intent",
            "business_intent_name",
            "model_type",
            "model_type_name",
            "dataset_key",
            "dataset_required",
            "evaluation_profile",
            "primary_metric",
            "secondary_metrics",
            "output_type_hint",
        ):
            if key in inherited_stage3_profile:
                profile[key] = inherited_stage3_profile[key]
    if profile["evaluation_profile"] == "latency_only":
        profile["dataset_key"] = _coerce_text(profile.get("dataset_key")) or "latency_only"
        profile["dataset_required"] = False
        if not profile["primary_metric"]:
            profile["primary_metric"] = "latency_s"
        if not profile["secondary_metrics"]:
            profile["secondary_metrics"] = ["throughput_qps"]
    resolved_dataset_key = _coerce_text(profile.get("dataset_key"))
    # mmlu dataset 必须使用 mmlu 评测画像和 accuracy 指标
    if resolved_dataset_key == "mmlu":
        profile["evaluation_profile"] = "mmlu"
        profile["primary_metric"] = "accuracy"
        profile["secondary_metrics"] = ["latency_s", "throughput_qps", "match_rate"]
        profile["output_type_hint"] = "class_labels"
    dataset_path = None
    if profile["dataset_required"] and resolved_dataset_key:
        local_dataset_path = str(config.get("dataset_local_path") or "").strip()
        resolved_local_dataset_path = _resolve_config_path(local_dataset_path, adapt_dir=adapt_dir) if local_dataset_path else None
        if resolved_local_dataset_path is not None and resolved_local_dataset_path.exists():
            dataset_path = resolved_local_dataset_path
        else:
            try:
                dataset_path = get_dataset_disk_path(resolved_dataset_key, cache_dir=datasets_dir)
            except Exception:
                dataset_path = resolved_local_dataset_path
    profile["dataset_path"] = str(dataset_path) if dataset_path else ""
    return profile


def _new_benchmark_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"business-{timestamp}-{uuid4().hex[:8]}"


def _should_refresh_benchmark_run_id(existing_run_id: str, *, refresh_run_id: bool) -> bool:
    if refresh_run_id:
        return True
    if not existing_run_id:
        return True
    return existing_run_id.startswith("legacy-")


def _load_optimization_notes(adapt_dir: Path) -> dict:
    notes_path = adapt_dir / "optimization_notes.json"
    if not notes_path.exists():
        return {}
    try:
        payload = json.loads(notes_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize_phase4_profile_field_value(field_name: str, value: object) -> object:
    if field_name in PHASE4_PROFILE_LIST_FIELDS:
        if not isinstance(value, (list, tuple, set)):
            return []
        return [str(item).strip() for item in value if str(item).strip()]
    return _coerce_text(value)


def _has_phase4_profile_override(config: dict, override_key: str) -> bool:
    field_name = override_key.removesuffix("_override")
    value = config.get(override_key)
    if field_name in PHASE4_PROFILE_LIST_FIELDS:
        return bool(_normalize_phase4_profile_field_value(field_name, value))
    return bool(_coerce_text(value))


def _promote_manual_phase4_profile_overrides(config: dict) -> bool:
    snapshot = config.get("phase4_auto_profile_snapshot")
    if not isinstance(snapshot, dict):
        return False

    changed = False
    for field_name, override_key in PHASE4_PROFILE_OVERRIDE_KEYS.items():
        if _has_phase4_profile_override(config, override_key):
            continue
        previous_auto_value = _normalize_phase4_profile_field_value(field_name, snapshot.get(field_name))
        current_value = _normalize_phase4_profile_field_value(field_name, config.get(field_name))
        if not previous_auto_value or not current_value or current_value == previous_auto_value:
            continue
        if field_name in PHASE4_PROFILE_LIST_FIELDS:
            config[override_key] = list(current_value)
        else:
            config[override_key] = current_value
        changed = True

    if changed:
        config["phase4_profile_override_source"] = "auto_promoted_from_manual_canonical_edits"
    return changed


def _build_phase4_auto_profile_snapshot(config: dict) -> dict[str, object]:
    snapshot: dict[str, object] = {}
    for field_name in PHASE4_PROFILE_OVERRIDE_KEYS:
        normalized_value = _normalize_phase4_profile_field_value(field_name, config.get(field_name))
        if field_name in PHASE4_PROFILE_LIST_FIELDS:
            if normalized_value:
                snapshot[field_name] = list(normalized_value)
            continue
        if normalized_value:
            snapshot[field_name] = normalized_value
    return snapshot


def _load_metric_artifact_from_adaptation(adapt_dir: Path, artifact_name: object) -> dict:
    artifact_text = str(artifact_name or "").strip()
    if not artifact_text:
        return {}
    artifact_path = adapt_dir / artifact_text
    if not artifact_path.exists():
        return {}
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _coerce_optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _normalize_device_id_list(value: object) -> list[str]:
    if isinstance(value, str):
        candidates = re.split(r"[\s,]+", value.strip())
    elif isinstance(value, int) and not isinstance(value, bool):
        candidates = [value]
    elif isinstance(value, (list, tuple, set)):
        candidates = list(value)
    else:
        return []
    normalized: list[str] = []
    for candidate in candidates:
        device_id = "" if candidate is None else str(candidate).strip()
        if not device_id:
            continue
        prefix_match = re.match(r"^(?:npu|cuda):(\d+)$", device_id, flags=re.IGNORECASE)
        if prefix_match:
            device_id = prefix_match.group(1)
        annotated_match = re.match(r"^(\d+)(?:\D.*)?$", device_id)
        if annotated_match:
            device_id = annotated_match.group(1)
        if not device_id.isdigit() or device_id in normalized:
            continue
        normalized.append(device_id)
    return normalized


def _extract_single_npu_hint(*values: object) -> list[str]:
    for value in values:
        text = _coerce_text(value)
        if not text:
            continue
        matches = re.findall(r"\bnpu:(\d+)\b", text, flags=re.IGNORECASE)
        if matches:
            return _normalize_device_id_list(matches)
    return []


def _extract_optimization_device_plan(notes: dict) -> tuple[list[str], str, str, bool]:
    candidate_rows: list[dict] = []
    if isinstance(notes, dict):
        candidate_rows.append(notes)
    best_result = notes.get("best_result")
    if isinstance(best_result, dict):
        candidate_rows.append(best_result)
    for result in notes.get("results") or []:
        if isinstance(result, dict):
            candidate_rows.append(result)

    fallback_parallel_mode = ""
    fallback_device_topology = ""
    explicit_device_plan = False
    for row in candidate_rows:
        selected_npus = _normalize_device_id_list(row.get("selected_npus") or row.get("selected_npu") or row.get("npu_ids") or row.get("visible_devices") or row.get("ascend_rt_visible_devices"))
        parallel_mode = _coerce_text(row.get("parallel_mode"))
        device_topology = _coerce_text(row.get("device_topology"))
        if selected_npus:
            return selected_npus, parallel_mode, device_topology, True
        single_npu_hint = _extract_single_npu_hint(
            row.get("device"),
            row.get("validation_note"),
            row.get("note"),
        )
        if single_npu_hint:
            topology = device_topology or f"visible_devices:{','.join(single_npu_hint)}"
            return single_npu_hint, parallel_mode or "single_card", topology, True
        if not fallback_parallel_mode and parallel_mode:
            fallback_parallel_mode = parallel_mode
        if not fallback_device_topology and device_topology:
            fallback_device_topology = device_topology
        explicit_device_plan = explicit_device_plan or bool(single_npu_hint)
    return [], fallback_parallel_mode, fallback_device_topology, explicit_device_plan


def _extract_multicard_device_plan_from_business_config_backups(adapt_dir: Path) -> tuple[list[str], str, str, str]:
    candidate_paths = [adapt_dir / "business_benchmark_config.json"]
    candidate_paths.extend(sorted(adapt_dir.glob("business_benchmark_config.json__prev_rule_refresh_*"), reverse=True))
    for candidate_path in candidate_paths:
        if not candidate_path.exists():
            continue
        try:
            payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        selected_npus = _normalize_device_id_list(payload.get("selected_npus") or payload.get("selected_npu") or payload.get("npu_ids") or payload.get("visible_devices") or payload.get("ascend_rt_visible_devices"))
        if not selected_npus:
            baseline_env = _normalize_env_mapping(payload.get("npu_baseline_env"))
            perf_env = _normalize_env_mapping(payload.get("npu_perf_env"))
            selected_npus = _normalize_device_id_list(baseline_env.get(NPU_VISIBLE_ENV_KEY) or perf_env.get(NPU_VISIBLE_ENV_KEY))
        if len(selected_npus) <= 1:
            continue
        parallel_mode = _coerce_text(payload.get("parallel_mode")) or "manual_visible_devices"
        device_topology = _coerce_text(payload.get("device_topology")) or f"visible_devices:{','.join(selected_npus)}"
        return selected_npus, parallel_mode, device_topology, f"{candidate_path.name}:business_config_backup"
    return [], "", "", ""


def _infer_multicard_npu_plan_from_accuracy_perf(adapt_dir: Path) -> tuple[list[str], str, str, str]:
    candidate_paths = [
        adapt_dir / "accuracy_run_perf.py",
        adapt_dir / "accuracy_run.py",
        adapt_dir / "demo.py",
    ]
    auto_dispatch_detected = False
    for candidate_path in candidate_paths:
        if not candidate_path.exists():
            continue
        try:
            source = candidate_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if re.search(r"device_map\s*=\s*[\"']auto[\"']", source) or ("infer_auto_device_map" in source and "dispatch_model" in source):
            auto_dispatch_detected = True

        max_memory_match = re.search(r"max_memory\s*=\s*\{\s*i\s*:\s*['\"][^'\"]+['\"]\s+for\s+i\s+in\s+range\((\d+)\)\s*\}", source)
        if max_memory_match:
            count = int(max_memory_match.group(1))
            if count > 1:
                selected_npus = [str(index) for index in range(count)]
                topology = f"visible_devices:{','.join(selected_npus)}"
                return selected_npus, "manual_visible_devices", topology, f"{candidate_path.name}:max_memory_range"

        dynamic_auto_dispatch_match = re.search(
            r"max_memory\s*=\s*\{\s*i\s*:\s*['\"][^'\"]+['\"]\s+for\s+i\s+in\s+range\(\s*(n_npu|num_npu|npu_count|torch\.npu\.device_count\(\))\s*\)\s*\}",
            source,
        )
        if dynamic_auto_dispatch_match and auto_dispatch_detected:
            return [], "auto", "all_visible_devices", f"{candidate_path.name}:dynamic_auto_dispatch"

        visible_devices_match = re.search(r"ASCEND_RT_VISIBLE_DEVICES[\"']?\s*\]?\s*=\s*[\"']([0-9,\s]+)[\"']", source)
        if visible_devices_match:
            selected_npus = _normalize_device_id_list(visible_devices_match.group(1))
            if len(selected_npus) > 1:
                topology = f"visible_devices:{','.join(selected_npus)}"
                return selected_npus, "manual_visible_devices", topology, f"{candidate_path.name}:visible_devices_literal"

        multicard_comment_match = re.search(r"Multi-card:\s*(\d+)x\s*NPU", source, flags=re.IGNORECASE)
        if multicard_comment_match:
            count = int(multicard_comment_match.group(1))
            if count > 1:
                selected_npus = [str(index) for index in range(count)]
                topology = f"visible_devices:{','.join(selected_npus)}"
                return selected_npus, "manual_visible_devices", topology, f"{candidate_path.name}:multicard_comment"

        required_hbm_match = re.search(r"requires?\s*~?\s*(\d+)\s*GB", source, flags=re.IGNORECASE)
        single_npu_capacity_match = re.search(r"single\s+NPU\s+capacity\s*\(\s*~?\s*(\d+)\s*GB", source, flags=re.IGNORECASE)
        if "exceeding single NPU capacity" in source.lower() and required_hbm_match:
            required_hbm_gb = int(required_hbm_match.group(1))
            single_npu_capacity_gb = int(single_npu_capacity_match.group(1)) if single_npu_capacity_match else 0
            inferred_count = math.ceil(required_hbm_gb / single_npu_capacity_gb) if single_npu_capacity_gb > 0 else 2
            inferred_count = max(2, inferred_count)
            selected_npus = [str(index) for index in range(inferred_count)]
            topology = f"visible_devices:{','.join(selected_npus)}"
            return selected_npus, "manual_visible_devices", topology, f"{candidate_path.name}:single_npu_capacity_exceeded"

    if auto_dispatch_detected:
        readme_path = adapt_dir / "README.md"
        readme_text = ""
        if readme_path.exists():
            try:
                readme_text = readme_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                readme_text = ""
        normalized_readme = readme_text.lower()
        chinese_multicard_signals = ("需要多卡", "多卡时自动", "多卡")
        english_multicard_signals = ("requires multiple cards", "multiple cards", "need multiple cards", "all visible devices")
        readme_mentions_multicard = any(signal in normalized_readme for signal in english_multicard_signals) or any(signal in readme_text for signal in chinese_multicard_signals)
        readme_mentions_auto_dispatch = 'device_map="auto"' in readme_text or "device_map='auto'" in readme_text or "auto dispatch" in normalized_readme
        if readme_mentions_multicard or (readme_mentions_auto_dispatch and ("多卡" in readme_text or "multiple cards" in normalized_readme)):
            selected_npus, parallel_mode, device_topology, reason = _extract_multicard_device_plan_from_business_config_backups(adapt_dir)
            if len(selected_npus) > 1:
                return selected_npus, parallel_mode, device_topology, reason
            return [], "auto", "all_visible_devices", f"{readme_path.name}:dynamic_auto_dispatch"

    return [], "", "", ""


def _infer_optimization_kind(notes: dict) -> str:
    candidates = [
        notes.get("optimization_kind"),
        notes.get("best_result", {}).get("optimization_kind") if isinstance(notes.get("best_result"), dict) else None,
    ]
    for candidate in candidates:
        text = str(candidate or "").strip().lower()
        if text:
            return text
    result_rows: list[dict] = []
    best_result = notes.get("best_result")
    if isinstance(best_result, dict):
        result_rows.append(best_result)
    for result in notes.get("results") or []:
        if isinstance(result, dict):
            result_rows.append(result)
    for result in result_rows:
        optimization_items = result.get("optimization_items")
        if not isinstance(optimization_items, list):
            continue
        normalized_items = [str(item or "").strip().lower() for item in optimization_items if str(item or "").strip()]
        if normalized_items and all(item.startswith("task_queue_enable") or item.startswith("warmup") for item in normalized_items):
            return "runtime_only"
    legacy_text_candidates = [
        notes.get("optimizations"),
        notes.get("summary"),
        notes.get("notes"),
    ]
    if isinstance(best_result, dict):
        legacy_text_candidates.extend(
            [
                best_result.get("validation_note"),
                best_result.get("note"),
            ]
        )
    for candidate in legacy_text_candidates:
        text = str(candidate or "").strip().lower()
        if "runtime_only" in text:
            return "runtime_only"
        if "task_queue_enable" in text and "warmup" in text and ("does not support fusion" in text or "does not support fusion operators" in text):
            return "runtime_only"
        if "task_queue_enable" in text and "warmup" in text:
            if any(phrase in text for phrase in ("fusion ops removed", "fusion operators removed", "fusion ops disabled", "fusion operators disabled")):
                return "runtime_only"
            if ("regressed" in text or "regression" in text) and ("fusion op" in text or "fusion operator" in text):
                return "runtime_only"
    return ""


def _should_use_model_files_for_perf(adapt_dir: Path, optimization_notes: dict) -> bool:
    if not (adapt_dir / "model_files").exists():
        return False
    optimization_kind = _infer_optimization_kind(optimization_notes)
    if optimization_kind == "runtime_only":
        return False
    return True


def _resolve_business_optimization_kind(config: dict, optimization_notes: dict, explicit_npu_perf_use_model_files: bool | None) -> str:
    explicit = str(config.get("optimization_kind") or "").strip().lower()
    # Phase-4 may need to fall back to runtime-only even when optimization stage
    # remains fusion_ops-valid on its own benchmark workload.
    if explicit == "runtime_only" and explicit_npu_perf_use_model_files is False:
        return "runtime_only"
    return _infer_optimization_kind(optimization_notes)


def _infer_optimization_dtype(notes: dict) -> str:
    candidates: list[object] = []
    best_result = notes.get("best_result")
    if isinstance(best_result, dict):
        candidates.append(best_result.get("dtype"))
    for result in notes.get("results") or []:
        if isinstance(result, dict):
            candidates.append(result.get("dtype"))
    for candidate in candidates:
        dtype = str(candidate or "").strip().lower()
        if dtype in {"fp32", "fp16", "bf16"}:
            return dtype
    return ""


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = int(text)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _extract_optimization_warmup_contract(optimization_notes: dict, *, optimization_kind: str) -> tuple[int | None, int | None, str]:
    normalized_kind = str(optimization_kind or "").strip().lower()
    candidate_rows: list[dict] = []
    best_result = optimization_notes.get("best_result")
    if isinstance(best_result, dict):
        candidate_rows.append(best_result)
    for result in optimization_notes.get("results") or []:
        if isinstance(result, dict):
            candidate_rows.append(result)
    for row in candidate_rows:
        row_kind = str(row.get("optimization_kind") or "").strip().lower()
        if normalized_kind and row_kind and row_kind != normalized_kind:
            continue
        baseline_warmup = _coerce_positive_int(row.get("baseline_warmup_iterations"))
        perf_warmup = _coerce_positive_int(row.get("perf_warmup_iterations"))
        warmup_policy = str(row.get("warmup_policy") or "").strip().lower()
        if baseline_warmup is not None or perf_warmup is not None or warmup_policy:
            return baseline_warmup, perf_warmup, warmup_policy
    return (
        _coerce_positive_int(optimization_notes.get("baseline_warmup_iterations")),
        _coerce_positive_int(optimization_notes.get("perf_warmup_iterations")),
        str(optimization_notes.get("warmup_policy") or "").strip().lower(),
    )


def _resolve_phase4_warmup_iterations(config: dict, optimization_notes: dict, *, optimization_kind: str) -> tuple[int | None, int | None, str]:
    baseline_warmup = _coerce_positive_int(config.get("baseline_warmup_iterations"))
    perf_warmup = _coerce_positive_int(config.get("perf_warmup_iterations"))
    notes_baseline_warmup, notes_perf_warmup, warmup_policy = _extract_optimization_warmup_contract(
        optimization_notes,
        optimization_kind=optimization_kind,
    )
    normalized_kind = str(optimization_kind or "").strip().lower()
    if normalized_kind == "runtime_only":
        if warmup_policy == "symmetric":
            symmetric_warmup = max(notes_baseline_warmup or 0, notes_perf_warmup or 0)
            if symmetric_warmup > 0:
                return symmetric_warmup, symmetric_warmup, "optimization_notes:symmetric_runtime_only"
        if notes_baseline_warmup is not None or notes_perf_warmup is not None:
            return notes_baseline_warmup or baseline_warmup, notes_perf_warmup or perf_warmup, "optimization_notes:runtime_only"
    if baseline_warmup is None:
        baseline_warmup = notes_baseline_warmup
    if perf_warmup is None:
        perf_warmup = notes_perf_warmup
    return baseline_warmup, perf_warmup, ""


def _extract_embedding_phase4_contract(adapt_dir: Path, optimization_notes: dict) -> dict[str, int | str]:
    candidate_rows: list[dict] = []
    best_result = optimization_notes.get("best_result")
    if isinstance(best_result, dict):
        candidate_rows.append(best_result)
    for result in optimization_notes.get("results") or []:
        if isinstance(result, dict):
            candidate_rows.append(result)
    for row in candidate_rows:
        row_output_type = str(row.get("output_type") or "").strip().lower()
        baseline_metric = _load_metric_artifact_from_adaptation(adapt_dir, row.get("baseline_artifact"))
        perf_metric = _load_metric_artifact_from_adaptation(adapt_dir, row.get("perf_artifact"))
        metric_output_type = str(perf_metric.get("output_type") or baseline_metric.get("output_type") or "").strip().lower()
        resolved_output_type = row_output_type or metric_output_type
        if resolved_output_type not in {"cls_embeddings", "embeddings", "sentence_embeddings", "text_embeddings"}:
            continue
        contract: dict[str, int | str] = {"output_type_hint": resolved_output_type}
        baseline_batch_size = _coerce_positive_int(baseline_metric.get("batch_size"))
        if baseline_batch_size is None and baseline_metric:
            baseline_batch_size = 1
        perf_batch_size = _coerce_positive_int(row.get("perf_batch_size")) or _coerce_positive_int(perf_metric.get("batch_size"))
        baseline_repeats = _coerce_positive_int(baseline_metric.get("steady_state_repeat_iterations"))
        perf_repeats = _coerce_positive_int(perf_metric.get("steady_state_repeat_iterations"))
        # Cross-device latency compares CUDA baseline against the selected NPU perf
        # path, so default CUDA workload should follow perf-side batching/repeats.
        cuda_baseline_batch_size = perf_batch_size or baseline_batch_size
        cuda_baseline_repeats = _coerce_positive_int(row.get("steady_state_baseline_repeats")) or perf_repeats or baseline_repeats
        if baseline_batch_size is not None:
            contract["embedding_baseline_batch_size"] = baseline_batch_size
        if perf_batch_size is not None:
            contract["embedding_perf_batch_size"] = perf_batch_size
        if cuda_baseline_batch_size is not None:
            contract["embedding_cuda_baseline_batch_size"] = cuda_baseline_batch_size
        if baseline_repeats is not None:
            contract["embedding_baseline_steady_state_repeats"] = baseline_repeats
        if perf_repeats is not None:
            contract["embedding_perf_steady_state_repeats"] = perf_repeats
        if cuda_baseline_repeats is not None:
            contract["embedding_cuda_baseline_steady_state_repeats"] = cuda_baseline_repeats
        return contract
    return {}


def _extract_stage3_phase4_profile_contract(adapt_dir: Path, optimization_notes: dict) -> dict[str, object]:
    candidate_rows: list[dict] = []
    best_result = optimization_notes.get("best_result")
    if isinstance(best_result, dict):
        candidate_rows.append(best_result)
    for result in optimization_notes.get("results") or []:
        if isinstance(result, dict):
            candidate_rows.append(result)

    for row in candidate_rows:
        baseline_metric = _load_metric_artifact_from_adaptation(adapt_dir, row.get("baseline_artifact"))
        perf_metric = _load_metric_artifact_from_adaptation(adapt_dir, row.get("perf_artifact"))
        resolved_dataset = str(
            row.get("dataset")
            or perf_metric.get("dataset")
            or baseline_metric.get("dataset")
            or ""
        ).strip().lower()
        resolved_output_type = str(
            row.get("output_type")
            or perf_metric.get("output_type")
            or baseline_metric.get("output_type")
            or ""
        ).strip().lower()

        if resolved_output_type == "class_labels" and resolved_dataset in {"cifar100", "imagenet", "fairface"}:
            primary_metric = "top1_accuracy"
            secondary_metrics = ["latency_s", "top5_accuracy", "match_rate"]
            if resolved_dataset == "fairface":
                business_intent = "face_age_classification"
                business_intent_name = "Face Age Classification"
            else:
                business_intent = "vision_classification"
                business_intent_name = "Image Classification"
            return {
                "business_intent": business_intent,
                "business_intent_name": business_intent_name,
                "model_type": "vision_classification",
                "model_type_name": "Image Classification",
                "dataset_key": resolved_dataset,
                "dataset_required": True,
                "evaluation_profile": "vision_topk_accuracy",
                "primary_metric": primary_metric,
                "secondary_metrics": secondary_metrics,
                "output_type_hint": "class_labels",
            }

        if resolved_output_type == "image_embeddings" and resolved_dataset in {"cifar100", "imagenet", "lfw"}:
            return {
                "business_intent": "vision_embedding",
                "business_intent_name": "Image Embedding",
                "model_type": "vision_embedding",
                "model_type_name": "Image Embedding",
                "dataset_key": resolved_dataset,
                "dataset_required": True,
                "evaluation_profile": "embedding_similarity",
                "primary_metric": "cosine_similarity",
                "secondary_metrics": ["latency_s", "throughput_qps", "match_rate"],
                "output_type_hint": "image_embeddings",
            }

        if resolved_output_type == "latent_embeddings" and resolved_dataset == "synthetic_3d":
            return {
                "business_intent": "volumetric_embedding",
                "business_intent_name": "3D Latent Embedding",
                "model_type": "vqgan_3d",
                "model_type_name": "3D Latent Model",
                "dataset_key": "synthetic_3d",
                "dataset_required": False,
                "evaluation_profile": "embedding_similarity",
                "primary_metric": "cosine_similarity",
                "secondary_metrics": ["latency_s", "throughput_qps", "match_rate"],
                "output_type_hint": "latent_embeddings",
            }

        if resolved_output_type == "molecular_embeddings" and resolved_dataset == "builtin_smiles":
            return {
                "business_intent": "molecular_embedding",
                "business_intent_name": "Molecular Embedding",
                "model_type": "molecular_embedding",
                "model_type_name": "Molecular Embedding",
                "dataset_key": "builtin_smiles",
                "dataset_required": False,
                "evaluation_profile": "embedding_similarity",
                "primary_metric": "cosine_similarity",
                "secondary_metrics": ["latency_s", "throughput_qps", "match_rate"],
                "output_type_hint": "molecular_embeddings",
            }

        if resolved_output_type in {"generated_text", "qa_answers"}:
            dataset_key = resolved_dataset
            if dataset_key in {"", "builtin", "latency_only"}:
                dataset_key = "pubmed_qa" if resolved_output_type == "qa_answers" else "mmlu"

            if dataset_key == "mmlu":
                return {
                    "business_intent": "causal_lm_base",
                    "business_intent_name": "Base Model Reasoning",
                    "model_type": "causal_lm",
                    "model_type_name": "Causal Language Model",
                    "dataset_key": "mmlu",
                    "dataset_required": True,
                    "evaluation_profile": "mmlu",
                    "primary_metric": "accuracy",
                    "secondary_metrics": ["latency_s", "throughput_qps", "match_rate"],
                    "output_type_hint": "class_labels",
                }

            if dataset_key == "pubmed_qa" or resolved_output_type == "qa_answers":
                return {
                    "business_intent": "biomedical_qa",
                    "business_intent_name": "Biomedical QA",
                    "model_type": "causal_lm",
                    "model_type_name": "Causal Language Model",
                    "dataset_key": "pubmed_qa",
                    "dataset_required": True,
                    "evaluation_profile": "qa_exact_match",
                    "primary_metric": "exact_match",
                    "secondary_metrics": ["latency_s", "f1", "match_rate"],
                    "output_type_hint": "qa_answers",
                }

            return {
                "business_intent": "causal_lm_base",
                "business_intent_name": "Base Model Reasoning",
                "model_type": "causal_lm",
                "model_type_name": "Causal Language Model",
                "dataset_key": dataset_key or "gsm8k",
                "dataset_required": _dataset_requires_local_path(dataset_key or "gsm8k"),
                "evaluation_profile": "generation_exact_match",
                "primary_metric": "exact_match",
                "secondary_metrics": ["latency_s", "throughput_qps", "match_rate"],
                "output_type_hint": "generated_text",
            }
    return {}


def _infer_trust_remote_code(adapt_dir: Path, config: dict) -> bool:
    explicit = config.get("trust_remote_code")
    if isinstance(explicit, bool):
        if explicit:
            return True
    explicit_text = str(explicit or "").strip().lower()
    if explicit_text in {"1", "true", "yes", "on"}:
        return True

    for candidate_name in ("demo.py", "accuracy_run.py", "accuracy_run_perf.py"):
        candidate_path = adapt_dir / candidate_name
        if not candidate_path.exists():
            continue
        try:
            source = candidate_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "trust_remote_code=True" in source or "trust_remote_code = True" in source:
            return True

    model_files_dir = adapt_dir / "model_files"
    if any(model_files_dir.glob("modeling_*.py")) or any(model_files_dir.glob("configuration_*.py")):
        return True

    local_models_dir = adapt_dir / "models"
    if local_models_dir.is_dir():
        model_id = str(config.get("model_id") or "").strip()
        candidate_dirs: list[Path] = []
        if model_id:
            candidate_dirs.extend(_iter_local_snapshot_asset_dirs(model_id, adapt_dir))
        candidate_dirs.extend(sorted(path for path in local_models_dir.glob("local_snapshot*") if path.is_dir()))
        seen: set[str] = set()
        for candidate_dir in candidate_dirs:
            candidate_dir_str = str(candidate_dir)
            if candidate_dir_str in seen:
                continue
            seen.add(candidate_dir_str)
            if any(candidate_dir.glob("modeling_*.py")) or any(candidate_dir.glob("configuration_*.py")):
                return True
            config_path = candidate_dir / "config.json"
            if not config_path.exists():
                continue
            try:
                payload = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            auto_map = payload.get("auto_map")
            if isinstance(auto_map, dict) and auto_map:
                return True
    return False


def _prepare_business_config(model: dict, adapt_dir: Path, *, ensure_local_dataset: bool, refresh_run_id: bool = False) -> dict:
    config_path = adapt_dir / "business_benchmark_config.json"
    original_config = _load_config(adapt_dir, missing_ok=True)
    config = dict(original_config)
    # Canonical phase-4 fields are regenerated every run. If an operator edited
    # them directly after the previous run, lift those edits into *_override so
    # they survive subsequent run-npu refreshes.
    _promote_manual_phase4_profile_overrides(config)
    optimization_notes = _load_optimization_notes(adapt_dir)
    transformers_metadata = _infer_transformers_config_metadata(model["model_id"], adapt_dir, config)
    profile = _resolve_business_profile(model["model_id"], config, adapt_dir)
    adaptation_context_text = _read_adaptation_context(adapt_dir)
    _ensure_business_runtime_dependencies(
        adapt_dir,
        str(profile.get("evaluation_profile") or ""),
        str(profile.get("model_type") or ""),
        str(model.get("model_id") or ""),
        config,
    )
    datasets_dir = PROJECT_ROOT / "datasets"
    if DEFAULT_HF_ENDPOINT:
        os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
    if DEFAULT_HF_HUB_DOWNLOAD_TIMEOUT:
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", DEFAULT_HF_HUB_DOWNLOAD_TIMEOUT)
    if DEFAULT_HF_HUB_ETAG_TIMEOUT:
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", DEFAULT_HF_HUB_ETAG_TIMEOUT)

    if ensure_local_dataset and profile["dataset_required"] and profile["dataset_key"]:
        dataset_path = ensure_dataset(str(profile["dataset_key"]), cache_dir=datasets_dir)
        profile["dataset_path"] = str(dataset_path)

    updated = dict(config)
    updated["model_id"] = model["model_id"]
    updated["model_class"] = _first_text(config, "model_class") or _coerce_text(transformers_metadata.get("model_class")) or None
    updated["architectures"] = _first_text(config, "architectures") or _coerce_text(transformers_metadata.get("architectures")) or None
    updated["problem_type"] = _first_text(config, "problem_type") or _coerce_text(transformers_metadata.get("problem_type")) or None
    updated["num_labels"] = config.get("num_labels") or transformers_metadata.get("num_labels") or None
    updated["model_type"] = profile["model_type"]
    updated["dataset"] = profile["dataset_key"]
    updated["dataset_required"] = profile["dataset_required"]
    updated["dataset_local_path"] = _repo_relative_path_text(profile["dataset_path"], base_dir=adapt_dir) or None
    updated["evaluation_profile"] = profile["evaluation_profile"]
    updated["primary_metric"] = profile["primary_metric"]
    updated["secondary_metrics"] = profile["secondary_metrics"]
    updated["output_type_hint"] = profile["output_type_hint"]
    updated["model_backend"] = profile.get("model_backend") or None
    updated["model_file"] = profile.get("model_file") or None
    updated["detection_target_labels"] = list(profile.get("detection_target_labels") or []) or None
    component_source_overrides = _infer_component_source_overrides(model["model_id"], adapt_dir, profile, adaptation_context_text)
    for key, value in component_source_overrides.items():
        if value:
            updated[key] = value
    explicit_npu_perf_use_model_files = _coerce_optional_bool(config.get("npu_perf_use_model_files"))
    updated["optimization_kind"] = _resolve_business_optimization_kind(config, optimization_notes, explicit_npu_perf_use_model_files) or None
    if explicit_npu_perf_use_model_files is None:
        updated["npu_perf_use_model_files"] = _should_use_model_files_for_perf(adapt_dir, optimization_notes)
    else:
        updated["npu_perf_use_model_files"] = explicit_npu_perf_use_model_files
    configured_baseline_env = _normalize_env_mapping(config.get("npu_baseline_env"))
    configured_perf_env = _normalize_env_mapping(config.get("npu_perf_env"))
    configured_selected_npus = _normalize_device_id_list(config.get("selected_npus") or configured_baseline_env.get(NPU_VISIBLE_ENV_KEY) or configured_perf_env.get(NPU_VISIBLE_ENV_KEY))
    configured_parallel_mode = _coerce_text(config.get("parallel_mode"))
    configured_device_topology = _coerce_text(config.get("device_topology"))
    configured_has_explicit_device_plan = bool(configured_selected_npus) or (
        configured_parallel_mode in {"auto", "device_map_auto"} and configured_device_topology == "all_visible_devices"
    )
    optimization_selected_npus, optimization_parallel_mode, optimization_device_topology, optimization_has_explicit_device_plan = _extract_optimization_device_plan(optimization_notes)
    if configured_has_explicit_device_plan:
        inherited_selected_npus = list(configured_selected_npus)
        inherited_parallel_mode = configured_parallel_mode
        inherited_device_topology = configured_device_topology
        inherited_device_plan_source = "business_benchmark_config"
    else:
        inherited_selected_npus = list(optimization_selected_npus)
        inherited_parallel_mode = optimization_parallel_mode
        inherited_device_topology = optimization_device_topology
        inherited_device_plan_source = "optimization_notes"
    if not configured_has_explicit_device_plan and not optimization_has_explicit_device_plan and len(inherited_selected_npus) <= 1:
        inferred_selected_npus, inferred_parallel_mode, inferred_device_topology, inferred_reason = _infer_multicard_npu_plan_from_accuracy_perf(adapt_dir)
        inferred_dynamic_auto_multicard = inferred_parallel_mode in {"auto", "device_map_auto"} and inferred_device_topology == "all_visible_devices"
        if len(inferred_selected_npus) > 1 or inferred_dynamic_auto_multicard:
            inherited_selected_npus = inferred_selected_npus
            inherited_parallel_mode = inferred_parallel_mode
            inherited_device_topology = inferred_device_topology
            inherited_device_plan_source = inferred_reason
    if inherited_selected_npus:
        inherited_visible_devices = ",".join(inherited_selected_npus)
        baseline_env = _normalize_env_mapping(updated.get("npu_baseline_env"))
        perf_env = _normalize_env_mapping(updated.get("npu_perf_env"))
        baseline_env[NPU_VISIBLE_ENV_KEY] = inherited_visible_devices
        perf_env[NPU_VISIBLE_ENV_KEY] = inherited_visible_devices
        updated["npu_baseline_env"] = baseline_env
        updated["npu_perf_env"] = perf_env
        updated["selected_npus"] = inherited_selected_npus
        if len(inherited_selected_npus) > 1:
            updated["parallel_mode"] = inherited_parallel_mode or "manual_visible_devices"
            updated["device_topology"] = inherited_device_topology or f"visible_devices:{','.join(inherited_selected_npus)}"
        else:
            updated["parallel_mode"] = inherited_parallel_mode or "single_card"
            updated["device_topology"] = inherited_device_topology or f"visible_devices:{','.join(inherited_selected_npus)}"
        if inherited_device_plan_source != "optimization_notes":
            updated["phase4_npu_device_plan_source"] = inherited_device_plan_source
    elif inherited_parallel_mode in {"auto", "device_map_auto"} and inherited_device_topology == "all_visible_devices":
        baseline_env = _normalize_env_mapping(updated.get("npu_baseline_env"))
        perf_env = _normalize_env_mapping(updated.get("npu_perf_env"))
        configured_visible_npus = _normalize_device_id_list(baseline_env.get(NPU_VISIBLE_ENV_KEY) or perf_env.get(NPU_VISIBLE_ENV_KEY) or updated.get("selected_npus"))
        if configured_visible_npus:
            visible_devices = ",".join(configured_visible_npus)
            baseline_env[NPU_VISIBLE_ENV_KEY] = visible_devices
            perf_env[NPU_VISIBLE_ENV_KEY] = visible_devices
            updated["npu_baseline_env"] = baseline_env
            updated["npu_perf_env"] = perf_env
            updated["selected_npus"] = configured_visible_npus
            updated["parallel_mode"] = "manual_visible_devices" if len(configured_visible_npus) > 1 else "single_card"
            updated["device_topology"] = f"visible_devices:{visible_devices}"
            if inherited_device_plan_source != "optimization_notes":
                updated["phase4_npu_device_plan_source"] = f"{inherited_device_plan_source}+config_visible_devices"
        else:
            baseline_env.pop(NPU_VISIBLE_ENV_KEY, None)
            perf_env.pop(NPU_VISIBLE_ENV_KEY, None)
            updated["npu_baseline_env"] = baseline_env
            updated["npu_perf_env"] = perf_env
            updated.pop("selected_npus", None)
            updated["parallel_mode"] = inherited_parallel_mode
            updated["device_topology"] = inherited_device_topology
            if inherited_device_plan_source != "optimization_notes":
                updated["phase4_npu_device_plan_source"] = inherited_device_plan_source
    updated["trust_remote_code"] = _infer_trust_remote_code(adapt_dir, updated)
    explicit_business_dtype = _first_text(config, "torch_dtype_override", "dtype_override")
    optimization_dtype = _infer_optimization_dtype(optimization_notes)
    resolved_business_dtype = explicit_business_dtype or optimization_dtype
    if resolved_business_dtype:
        updated["dtype"] = resolved_business_dtype
        updated["torch_dtype"] = resolved_business_dtype
    else:
        updated.pop("torch_dtype", None)
    existing_run_id = str(updated.get("benchmark_run_id") or "").strip()
    if _should_refresh_benchmark_run_id(existing_run_id, refresh_run_id=refresh_run_id):
        updated["benchmark_run_id"] = _new_benchmark_run_id()
        updated["benchmark_run_started_at"] = datetime.now().isoformat()
    if str(updated.get("model_type") or "").strip().lower() == "asr":
        multilingual_asr = resolve_multilingual_asr_dataset(str(updated.get("model_id") or ""))
        resolved_dataset = str(updated.get("dataset") or "").strip().lower()
        resolved_asr_language = str(profile.get("asr_language") or updated.get("asr_language") or "").strip()
        if not resolved_asr_language and multilingual_asr:
            resolved_asr_language = str(multilingual_asr.get("asr_language") or "").strip()
        if not resolved_asr_language and resolved_dataset == "librispeech":
            resolved_asr_language = "en"
        updated["asr_language"] = resolved_asr_language
        resolved_asr_task = str(profile.get("asr_task") or updated.get("asr_task") or "").strip()
        if not resolved_asr_task:
            resolved_asr_task = "forced_align" if str(updated.get("model_backend") or "").strip().lower() == "qwen_forced_aligner" else "transcribe"
        updated["asr_task"] = resolved_asr_task
    if str(updated.get("model_type") or "").strip().lower() == "embedding":
        embedding_phase4_contract = _extract_embedding_phase4_contract(adapt_dir, optimization_notes)
        inherited_output_type_hint = str(embedding_phase4_contract.get("output_type_hint") or "").strip().lower()
        if inherited_output_type_hint in {"cls_embeddings", "embeddings", "sentence_embeddings", "text_embeddings"}:
            updated["output_type_hint"] = inherited_output_type_hint
        configured_embedding_batch_size = _coerce_positive_int(updated.get("embedding_batch_size"))
        configured_baseline_batch_size = _coerce_positive_int(updated.get("embedding_baseline_batch_size"))
        configured_perf_batch_size = _coerce_positive_int(updated.get("embedding_perf_batch_size"))
        configured_cuda_batch_size = _coerce_positive_int(updated.get("embedding_cuda_baseline_batch_size"))
        inherited_baseline_batch_size = _coerce_positive_int(embedding_phase4_contract.get("embedding_baseline_batch_size"))
        inherited_perf_batch_size = _coerce_positive_int(embedding_phase4_contract.get("embedding_perf_batch_size"))
        inherited_cuda_batch_size = _coerce_positive_int(embedding_phase4_contract.get("embedding_cuda_baseline_batch_size"))
        updated["embedding_baseline_batch_size"] = configured_baseline_batch_size or inherited_baseline_batch_size or 1
        updated["embedding_perf_batch_size"] = configured_perf_batch_size or inherited_perf_batch_size or configured_embedding_batch_size or DEFAULT_EMBEDDING_BATCH_SIZE
        resolved_cuda_batch_size = configured_cuda_batch_size or inherited_cuda_batch_size or updated["embedding_baseline_batch_size"]
        # Migrate older phase-4 configs that defaulted CUDA baseline to the
        # baseline workload. Cross-device comparison is defined against the NPU
        # perf path, so if the persisted CUDA batch size simply mirrors baseline
        # while perf uses a larger inherited batch, promote CUDA to the perf
        # workload automatically.
        if (
            configured_cuda_batch_size is not None
            and configured_baseline_batch_size is not None
            and configured_cuda_batch_size == configured_baseline_batch_size
            and inherited_cuda_batch_size is not None
            and inherited_perf_batch_size is not None
            and inherited_cuda_batch_size == inherited_perf_batch_size
            and inherited_cuda_batch_size > configured_cuda_batch_size
        ):
            resolved_cuda_batch_size = inherited_cuda_batch_size
        updated["embedding_cuda_baseline_batch_size"] = resolved_cuda_batch_size
        updated["embedding_batch_size"] = configured_embedding_batch_size or updated["embedding_perf_batch_size"] or DEFAULT_EMBEDDING_BATCH_SIZE
        configured_baseline_repeats = _coerce_positive_int(updated.get("embedding_baseline_steady_state_repeats"))
        configured_perf_repeats = _coerce_positive_int(updated.get("embedding_perf_steady_state_repeats"))
        configured_cuda_repeats = _coerce_positive_int(updated.get("embedding_cuda_baseline_steady_state_repeats"))
        inherited_baseline_repeats = _coerce_positive_int(embedding_phase4_contract.get("embedding_baseline_steady_state_repeats"))
        inherited_perf_repeats = _coerce_positive_int(embedding_phase4_contract.get("embedding_perf_steady_state_repeats"))
        inherited_cuda_repeats = _coerce_positive_int(embedding_phase4_contract.get("embedding_cuda_baseline_steady_state_repeats"))
        configured_embedding_repeats = _coerce_positive_int(updated.get("embedding_steady_state_repeats"))
        if configured_embedding_repeats:
            updated["embedding_steady_state_repeats"] = configured_embedding_repeats
        elif str(updated.get("optimization_kind") or "").strip().lower() == "runtime_only":
            updated["embedding_steady_state_repeats"] = DEFAULT_RUNTIME_ONLY_EMBEDDING_STEADY_STATE_REPEATS
        else:
            updated["embedding_steady_state_repeats"] = DEFAULT_EMBEDDING_STEADY_STATE_REPEATS
        updated["embedding_baseline_steady_state_repeats"] = configured_baseline_repeats or inherited_baseline_repeats or updated["embedding_steady_state_repeats"]
        updated["embedding_perf_steady_state_repeats"] = configured_perf_repeats or inherited_perf_repeats or updated["embedding_steady_state_repeats"]
        resolved_cuda_repeats = configured_cuda_repeats or inherited_cuda_repeats or updated["embedding_baseline_steady_state_repeats"]
        if (
            configured_cuda_repeats is not None
            and configured_baseline_repeats is not None
            and configured_cuda_repeats == configured_baseline_repeats
            and inherited_cuda_repeats is not None
            and inherited_perf_repeats is not None
            and inherited_cuda_repeats == inherited_perf_repeats
            and inherited_cuda_repeats > configured_cuda_repeats
        ):
            resolved_cuda_repeats = inherited_cuda_repeats
        updated["embedding_cuda_baseline_steady_state_repeats"] = resolved_cuda_repeats
    updated.setdefault("custom_evaluator", "business_model_eval.py")
    updated.setdefault("remote_ssh_host", DEFAULT_SSH_HOST_ALIAS)
    updated["remote_project_root"] = _portable_remote_project_root(updated.get("remote_project_root"))
    updated.setdefault("measurement_contract_version", DEFAULT_MEASUREMENT_CONTRACT_VERSION)
    updated.setdefault("latency_measurement_scope", DEFAULT_LATENCY_MEASUREMENT_SCOPE)
    baseline_warmup, perf_warmup, phase4_warmup_source = _resolve_phase4_warmup_iterations(
        updated,
        optimization_notes,
        optimization_kind=str(updated.get("optimization_kind") or ""),
    )
    updated["baseline_warmup_iterations"] = baseline_warmup or DEFAULT_BASELINE_WARMUP_ITERATIONS
    updated["perf_warmup_iterations"] = perf_warmup or DEFAULT_PERF_WARMUP_ITERATIONS
    if phase4_warmup_source:
        updated["phase4_warmup_source"] = phase4_warmup_source
    else:
        updated.pop("phase4_warmup_source", None)
    configured_cuda_baseline_warmup = _coerce_positive_int(updated.get("cuda_baseline_warmup_iterations"))
    if str(updated.get("optimization_kind") or "").strip().lower() == "runtime_only" and phase4_warmup_source == "optimization_notes:symmetric_runtime_only":
        symmetric_cuda_warmup = max(updated["baseline_warmup_iterations"] or 0, updated["perf_warmup_iterations"] or 0)
        updated["cuda_baseline_warmup_iterations"] = max(configured_cuda_baseline_warmup or 0, symmetric_cuda_warmup) or DEFAULT_CUDA_BASELINE_WARMUP_ITERATIONS
        updated["phase4_cuda_warmup_source"] = "optimization_notes:symmetric_runtime_only"
    else:
        updated["cuda_baseline_warmup_iterations"] = configured_cuda_baseline_warmup or DEFAULT_CUDA_BASELINE_WARMUP_ITERATIONS
        updated.pop("phase4_cuda_warmup_source", None)
    updated["npu_baseline_env"] = _merge_env_mapping_with_defaults(updated.get("npu_baseline_env"), DEFAULT_NPU_BASELINE_ENV)
    updated["npu_perf_env"] = _merge_env_mapping_with_defaults(updated.get("npu_perf_env"), DEFAULT_NPU_PERF_ENV)
    updated["cuda_baseline_env"] = _merge_env_mapping_with_defaults(updated.get("cuda_baseline_env"), DEFAULT_CUDA_BASELINE_ENV)
    for scenario, config_key in SCENARIO_TO_CONFIG_KEY.items():
        updated[config_key] = _normalize_business_scenario_command(updated.get(config_key), scenario)
        _validate_business_scenario_command(updated[config_key], scenario, config_key=config_key)
    updated["phase4_auto_profile_snapshot"] = _build_phase4_auto_profile_snapshot(updated)

    if updated != original_config:
        _write_config(adapt_dir, updated)
    return updated


def _refresh_business_stage_start(model: dict, adapt_dir: Path, config: dict) -> None:
    notes = {
        "reason": "phase4_rerun_started",
        "action": "refresh_start_time_before_local_npu_rerun",
        "adaptation_path": str(adapt_dir.relative_to(PROJECT_ROOT)),
        "benchmark_run_id": str(config.get("benchmark_run_id") or "").strip(),
        "benchmark_run_started_at": str(config.get("benchmark_run_started_at") or "").strip(),
    }
    command = [
        "uv",
        "run",
        "python",
        "scripts/board_ops.py",
        "update_business_benchmark_status",
        "--model_id",
        str(model["model_id"]),
        "--business_benchmark_status",
        "in_progress",
        "--notes",
        json.dumps(notes, ensure_ascii=False),
    ]
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), env=_subprocess_env_for_cwd(PROJECT_ROOT), text=True, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or "unknown error"
        raise RuntimeError(f"刷新第四阶段 start time 失败: {detail}")


def _business_run_script_content() -> str:
    return """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ADAPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ADAPT_DIR / "business_benchmark_config.json"
PROJECT_ROOT = ADAPT_DIR.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dataset_mapping import get_business_benchmark_profile
from download_datasets import ensure_dataset, get_dataset_disk_path

SCENARIO_TO_CONFIG_KEY = {
    "npu_baseline": "local_npu_baseline_command",
    "npu_perf": "local_npu_perf_command",
    "cuda_baseline": "remote_cuda_baseline_command",
}
SCENARIO_TO_ENV_KEY = {
    "npu_baseline": "npu_baseline_env",
    "npu_perf": "npu_perf_env",
    "cuda_baseline": "cuda_baseline_env",
}
SCENARIO_TO_REQUIRED_EXTRA = {
    "npu_baseline": "ascend",
    "npu_perf": "ascend",
    "cuda_baseline": "cuda",
}
INTERNAL_SYNTHETIC_BUSINESS_DATASETS = {
    "builtin",
    "latency_only",
    "synthetic_3d",
    "synthetic_colour_checker",
    "synthetic_dna",
    "synthetic_ocr",
    "synthetic_timeseries",
    "synthetic_triplets",
}


def _resolve_config_path(path_value: object, *, adapt_dir: Path) -> Path | None:
    text = str(path_value or "").strip()
    if not text:
        return None
    expanded = os.path.expandvars(text)
    candidate = Path(expanded).expanduser()
    if candidate.is_absolute():
        return candidate
    for resolved in (adapt_dir / candidate, PROJECT_ROOT / candidate):
        if resolved.exists():
            return resolved
    return adapt_dir / candidate


def load_config() -> dict:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("business_benchmark_config.json 必须是 JSON object")
    return data


def _stringify_env_mapping(raw_value) -> dict[str, str]:
    if not isinstance(raw_value, dict):
        return {}
    env_map: dict[str, str] = {}
    for key, value in raw_value.items():
        key_text = str(key or "").strip()
        if not key_text:
            continue
        env_map[key_text] = str(value if value is not None else "")
    return env_map


def _extract_uv_run_extras(command: str) -> list[str]:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return []
    extras = []
    for idx, token in enumerate(tokens):
        if token == "--extra" and idx + 1 < len(tokens):
            extra = str(tokens[idx + 1]).strip()
            if extra:
                extras.append(extra)
    return extras


def _validate_scenario_command(command: str, scenario: str, command_key: str) -> None:
    required_extra = SCENARIO_TO_REQUIRED_EXTRA[scenario]
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise ValueError(f"{command_key} 不是合法命令: {command}") from exc
    if len(tokens) < 2 or tokens[0] != "uv" or tokens[1] != "run":
        raise ValueError(f"{command_key} 必须显式使用 `uv run --extra {required_extra}`，当前为: {command}")
    if "--no-sync" not in tokens:
        raise ValueError(f"{command_key} 必须显式使用 `uv run --no-sync --extra {required_extra}`，当前为: {command}")
    extras = _extract_uv_run_extras(command)
    if required_extra not in extras:
        raise ValueError(f"{command_key} 必须显式使用 `uv run --extra {required_extra}`，当前为: {command}")
    wrong_extras = sorted({"ascend", "cuda"}.intersection(extras) - {required_extra})
    if wrong_extras:
        raise ValueError(f"{command_key} 不得混用错误 extra={wrong_extras}，当前为: {command}")
    scenario_token = f"--scenario={scenario}"
    if "--scenario" in tokens:
        idx = tokens.index("--scenario")
        if idx + 1 >= len(tokens) or tokens[idx + 1] != scenario:
            raise ValueError(f"{command_key} 的场景参数必须是 {scenario}，当前为: {command}")
    elif scenario_token not in tokens:
        raise ValueError(f"{command_key} 缺少 `--scenario {scenario}`，当前为: {command}")


def resolve_profile(config: dict) -> dict:
    model_id = str(config.get("model_id") or ADAPT_DIR.name).strip()
    def _coerce_text(value):
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(item).strip() for item in value if str(item).strip())
        if value is None:
            return ""
        return str(value).strip()

    def _first_text(mapping: dict, *keys: str) -> str:
        for key in keys:
            text = _coerce_text(mapping.get(key))
            if text:
                return text
        return ""

    def _first_list(mapping: dict, *keys: str) -> list[str]:
        for key in keys:
            value = mapping.get(key)
            if isinstance(value, list):
                items = [str(item).strip() for item in value if str(item).strip()]
                if items:
                    return items
        return []

    model_class = _first_text(config, "model_class")
    inferred = get_business_benchmark_profile(
        model_id,
        model_class,
        architectures=_first_text(config, "architectures"),
        problem_type=_first_text(config, "problem_type"),
        num_labels=config.get("num_labels"),
    )
    dataset_key = _first_text(config, "dataset_override", "dataset") or _coerce_text(inferred.get("dataset_key")) or None
    secondary_metrics = _first_list(config, "secondary_metrics_override", "secondary_metrics")
    if not secondary_metrics:
        secondary_metrics = [str(item).strip() for item in list(inferred.get("secondary_metrics") or []) if str(item).strip()]
    evaluation_profile = _first_text(config, "evaluation_profile_override", "evaluation_profile") or _coerce_text(inferred.get("evaluation_profile"))
    primary_metric = _first_text(config, "primary_metric_override", "primary_metric") or _coerce_text(inferred.get("primary_metric"))
    output_type_hint = _first_text(config, "output_type_hint_override", "output_type_hint") or _coerce_text(inferred.get("output_type_hint"))
    dataset_required = dataset_key is not None and str(dataset_key).strip().lower() not in INTERNAL_SYNTHETIC_BUSINESS_DATASETS
    if evaluation_profile == "latency_only":
        dataset_key = dataset_key or "latency_only"
        dataset_required = False

    dataset_path = ""
    if dataset_required and dataset_key:
        local_path = str(config.get("dataset_local_path") or "").strip()
        candidate = _resolve_config_path(local_path, adapt_dir=ADAPT_DIR) if local_path else None
        if candidate is None:
            try:
                candidate = get_dataset_disk_path(dataset_key)
            except Exception:
                candidate = None
        if candidate is not None and not candidate.exists():
            try:
                candidate = ensure_dataset(dataset_key)
            except Exception:
                pass
        dataset_path = str(candidate) if candidate is not None else ""

    return {
        **inferred,
        "model_id": model_id,
        "benchmark_run_id": str(config.get("benchmark_run_id") or "").strip(),
        "model_type": _first_text(config, "model_type_override", "model_type") or _coerce_text(inferred.get("model_type")),
        "dataset_key": dataset_key,
        "dataset_required": dataset_required,
        "dataset_path": dataset_path,
        "evaluation_profile": evaluation_profile,
        "primary_metric": primary_metric,
        "secondary_metrics": secondary_metrics,
        "output_type_hint": output_type_hint,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one business benchmark scenario")
    parser.add_argument("--scenario", choices=sorted(SCENARIO_TO_CONFIG_KEY), required=True)
    args = parser.parse_args()

    config = load_config()
    profile = resolve_profile(config)
    run_id = profile["benchmark_run_id"]
    if not run_id or run_id.startswith("legacy-"):
        raise ValueError("business_benchmark_config.json 缺少可执行的 benchmark_run_id，请先运行 business_benchmark_manager.py run-npu；若只需为已有 NPU 工件打印远端 CUDA 命令，可执行 print-remote-command（默认复用现有 run_id）")
    command_key = SCENARIO_TO_CONFIG_KEY[args.scenario]
    command = str(config.get(command_key) or "").strip()
    if not command:
        raise ValueError(f"business_benchmark_config.json 缺少 {command_key}")
    _validate_scenario_command(command, args.scenario, command_key)
    scenario_env_key = SCENARIO_TO_ENV_KEY[args.scenario]
    scenario_env = _stringify_env_mapping(config.get(scenario_env_key))

    env = {
        **os.environ,
        **scenario_env,
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "BUSINESS_BENCHMARK_SCENARIO": args.scenario,
        "BUSINESS_BENCHMARK_SCENARIO_COMMAND": command,
        "BUSINESS_BENCHMARK_RUN_ID": run_id,
        "BUSINESS_BENCHMARK_MODEL_ID": profile["model_id"],
        "BUSINESS_BENCHMARK_MODEL_TYPE": profile["model_type"],
        "BUSINESS_BENCHMARK_DATASET": str(profile["dataset_key"] or ""),
        "BUSINESS_BENCHMARK_DATASET_REQUIRED": "1" if profile["dataset_required"] else "0",
        "BUSINESS_BENCHMARK_DATASET_PATH": profile["dataset_path"],
        "BUSINESS_BENCHMARK_EVAL_PROFILE": profile["evaluation_profile"],
        "BUSINESS_BENCHMARK_PRIMARY_METRIC": profile["primary_metric"],
        "BUSINESS_BENCHMARK_SECONDARY_METRICS": json.dumps(profile["secondary_metrics"], ensure_ascii=False),
        "BUSINESS_BENCHMARK_OUTPUT_TYPE_HINT": profile["output_type_hint"],
        "BUSINESS_BENCHMARK_MEASUREMENT_CONTRACT_VERSION": str(config.get("measurement_contract_version") or 1),
        "BUSINESS_BENCHMARK_LATENCY_MEASUREMENT_SCOPE": str(config.get("latency_measurement_scope") or ""),
        "BUSINESS_BENCHMARK_BASELINE_WARMUP_ITERATIONS": str(config.get("baseline_warmup_iterations") or 0),
        "BUSINESS_BENCHMARK_PERF_WARMUP_ITERATIONS": str(config.get("perf_warmup_iterations") or 0),
        "BUSINESS_BENCHMARK_CUDA_BASELINE_WARMUP_ITERATIONS": str(config.get("cuda_baseline_warmup_iterations") or 0),
    }
    print(
        f"[business-run] scenario={args.scenario} "
        f"run_id={profile['benchmark_run_id'] or 'none'} "
        f"dataset={profile['dataset_key'] or 'None'} "
        f"eval_profile={profile['evaluation_profile']} "
        f"primary_metric={profile['primary_metric']} "
        f"scenario_env={json.dumps(scenario_env, ensure_ascii=False, sort_keys=True)}"
    )
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    env.pop("VIRTUAL_ENV", None)
    result = subprocess.run(command, cwd=str(ADAPT_DIR), shell=True, env=env)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
"""


def _ensure_business_run_script(adapt_dir: Path, config: dict) -> Path:
    script_path = adapt_dir / BUSINESS_RUN_FILENAME
    script_content = _render_generated_python_script(_business_run_script_content(), tool=MANAGER_SCRIPT_RELATIVE_PATH)
    if not script_path.exists():
        script_path.write_text(script_content, encoding="utf-8")
    elif script_path.read_text(encoding="utf-8") != script_content:
        try:
            script_path.write_text(script_content, encoding="utf-8")
        except PermissionError:
            print(f"[business][warn] cannot refresh {script_path}; using existing file")
    for scenario, config_key in SCENARIO_TO_CONFIG_KEY.items():
        command = str(config.get(config_key) or "").strip()
        if not command:
            raise ValueError(f"{adapt_dir}/business_benchmark_config.json 缺少 {config_key}")
        _validate_business_scenario_command(command, scenario, config_key=config_key)
    return script_path


def _subprocess_env_for_cwd(cwd: Path) -> dict[str, str]:
    env = {**os.environ, "PROJECT_ROOT": str(PROJECT_ROOT)}
    if LOCAL_UV_CACHE_DIR:
        env["UV_CACHE_DIR"] = LOCAL_UV_CACHE_DIR
    try:
        resolved_cwd = cwd.resolve()
    except Exception:
        resolved_cwd = cwd
    if resolved_cwd == ADAPTATIONS_DIR or ADAPTATIONS_DIR in resolved_cwd.parents:
        env.pop("VIRTUAL_ENV", None)
        preferred_uv_env = _preferred_uv_project_environment(resolved_cwd)
        if preferred_uv_env is None:
            env.pop("UV_PROJECT_ENVIRONMENT", None)
        else:
            env["UV_PROJECT_ENVIRONMENT"] = preferred_uv_env
    return env


def _build_business_scenario_env(adapt_dir: Path, config: dict, scenario: str) -> dict[str, str]:
    model_id = str(config.get("model_id") or adapt_dir.name).strip() or adapt_dir.name
    profile = _resolve_business_profile(model_id, config, adapt_dir)
    run_id = str(config.get("benchmark_run_id") or "").strip()
    if not run_id or run_id.startswith("legacy-"):
        raise ValueError("business_benchmark_config.json 缺少可执行的 benchmark_run_id，请先运行 run-npu 刷新本机 NPU 工件")
    command_key = SCENARIO_TO_CONFIG_KEY[scenario]
    command = str(config.get(command_key) or "").strip()
    if not command:
        raise ValueError(f"business_benchmark_config.json 缺少 {command_key}")
    _validate_business_scenario_command(command, scenario, config_key=command_key)
    scenario_env = _normalize_env_mapping(config.get(SCENARIO_TO_ENV_KEY[scenario]))
    return {
        **scenario_env,
        "BUSINESS_BENCHMARK_SCENARIO": scenario,
        "BUSINESS_BENCHMARK_SCENARIO_COMMAND": command,
        "BUSINESS_BENCHMARK_RUN_ID": run_id,
        "BUSINESS_BENCHMARK_MODEL_ID": model_id,
        "BUSINESS_BENCHMARK_MODEL_TYPE": str(profile.get("model_type") or ""),
        "BUSINESS_BENCHMARK_DATASET": str(profile.get("dataset_key") or ""),
        "BUSINESS_BENCHMARK_DATASET_REQUIRED": "1" if bool(profile.get("dataset_required")) else "0",
        "BUSINESS_BENCHMARK_DATASET_PATH": str(profile.get("dataset_path") or ""),
        "BUSINESS_BENCHMARK_EVAL_PROFILE": str(profile.get("evaluation_profile") or ""),
        "BUSINESS_BENCHMARK_PRIMARY_METRIC": str(profile.get("primary_metric") or ""),
        "BUSINESS_BENCHMARK_SECONDARY_METRICS": json.dumps(list(profile.get("secondary_metrics") or []), ensure_ascii=False),
        "BUSINESS_BENCHMARK_OUTPUT_TYPE_HINT": str(profile.get("output_type_hint") or ""),
        "BUSINESS_BENCHMARK_MEASUREMENT_CONTRACT_VERSION": str(config.get("measurement_contract_version") or 1),
        "BUSINESS_BENCHMARK_LATENCY_MEASUREMENT_SCOPE": str(config.get("latency_measurement_scope") or ""),
        "BUSINESS_BENCHMARK_BASELINE_WARMUP_ITERATIONS": str(config.get("baseline_warmup_iterations") or 0),
        "BUSINESS_BENCHMARK_PERF_WARMUP_ITERATIONS": str(config.get("perf_warmup_iterations") or 0),
        "BUSINESS_BENCHMARK_CUDA_BASELINE_WARMUP_ITERATIONS": str(config.get("cuda_baseline_warmup_iterations") or 0),
    }


def _run_shell_command(command: str, cwd: Path, *, capture_output: bool = False, extra_env: dict[str, str] | None = None):
    env = _subprocess_env_for_cwd(cwd)
    if extra_env:
        env.update({str(key): str(value) for key, value in extra_env.items()})
    result = subprocess.run(command, cwd=str(cwd), shell=True, env=env, text=True, capture_output=capture_output)
    if result.returncode != 0:
        detail = f"\nstdout:\n{result.stdout}" if capture_output and result.stdout else ""
        detail += f"\nstderr:\n{result.stderr}" if capture_output and result.stderr else ""
        raise RuntimeError(f"命令执行失败: {command}{detail}")
    return result


def _run_command(args: list[str], cwd: Path, *, capture_output: bool = False, check: bool = True, extra_env: dict[str, str] | None = None):
    env = _subprocess_env_for_cwd(cwd)
    if extra_env:
        env.update({str(key): str(value) for key, value in extra_env.items()})
    result = subprocess.run(args, cwd=str(cwd), env=env, text=True, capture_output=capture_output)
    if check and result.returncode != 0:
        cmd = " ".join(shlex.quote(part) for part in args)
        detail = f"\nstdout:\n{result.stdout}" if capture_output and result.stdout else ""
        detail += f"\nstderr:\n{result.stderr}" if capture_output and result.stderr else ""
        raise RuntimeError(f"命令执行失败: {cmd}{detail}")
    return result


def _ensure_local_business_runtime_installed(adapt_dir: Path) -> None:
    dependency_specs = _load_project_dependency_specs(adapt_dir)
    if not dependency_specs:
        return
    python_path = adapt_dir / ".venv" / "bin" / "python"
    if not python_path.exists():
        raise FileNotFoundError(f"{adapt_dir} 缺少 .venv/bin/python，无法补齐第四阶段本机依赖")
    extra_env = {
        "UV_LINK_MODE": "copy",
        "UV_PREVIEW_FEATURES": "extra-build-dependencies",
    }
    _run_command(
        ["uv", "pip", "install", "--python", str(python_path), *dependency_specs],
        adapt_dir,
        extra_env=extra_env,
    )


def _run_local_business_eval(adapt_dir: Path, config: dict, scenario: str):
    extra_env = _build_business_scenario_env(adapt_dir, config, scenario)
    _run_shell_command(_canonical_business_eval_command(scenario), adapt_dir, extra_env=extra_env)


def _is_transient_remote_transport_error(message: str) -> bool:
    lowered = str(message or "").lower()
    needles = (
        "connection closed by remote host",
        "connection reset by peer",
        "kex_exchange_identification",
        "broken pipe",
        "rsync error: error in rsync protocol data stream",
        "connection unexpectedly closed",
        "connection timed out",
    )
    return any(needle in lowered for needle in needles)


def _run_retryable_remote_command(args: list[str], cwd: Path, *, capture_output: bool = False, check: bool = True):
    if not args:
        raise ValueError("args 不能为空")
    env = {**os.environ, "PROJECT_ROOT": str(PROJECT_ROOT)}
    last_result: subprocess.CompletedProcess[str] | None = None
    last_error: RuntimeError | None = None
    for attempt in range(1, REMOTE_TRANSPORT_RETRY_ATTEMPTS + 1):
        result = subprocess.run(args, cwd=str(cwd), env=env, text=True, capture_output=True)
        last_result = result
        combined_output = f"{result.stdout or ''}\n{result.stderr or ''}".strip()
        if result.returncode == 0 or not check:
            if not capture_output:
                if result.stdout:
                    print(result.stdout, end="")
                if result.stderr:
                    print(result.stderr, end="", file=sys.stderr)
            return result
        cmd = " ".join(shlex.quote(part) for part in args)
        detail = f"\nstdout:\n{result.stdout}" if result.stdout else ""
        detail += f"\nstderr:\n{result.stderr}" if result.stderr else ""
        last_error = RuntimeError(f"命令执行失败: {cmd}{detail}")
        if attempt >= REMOTE_TRANSPORT_RETRY_ATTEMPTS or not _is_transient_remote_transport_error(combined_output):
            raise last_error
        print(
            f"[business][remote][retry] transient transport error on attempt {attempt}/{REMOTE_TRANSPORT_RETRY_ATTEMPTS}: {cmd}",
            file=sys.stderr,
        )
        time.sleep(REMOTE_TRANSPORT_RETRY_SLEEP_SECONDS)
    if last_error is not None:
        raise last_error
    if last_result is None:
        raise RuntimeError("远端命令未执行")
    return last_result


def _shell_join(args: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in args)


def _remote_runtime_env_mapping(*, remote_adapt_dir: str | None = None, allow_online_hub: bool = False) -> dict[str, str]:
    env: dict[str, str] = {}
    # Remote shared-storage setups have repeatedly deadlocked during `uv sync`
    # with symlink mode on large CUDA dependency sets. Prefer copy mode for
    # deterministic remote closure over marginal disk savings.
    env["UV_LINK_MODE"] = "copy"
    env["UV_PREVIEW_FEATURES"] = "extra-build-dependencies"
    if DEFAULT_HF_ENDPOINT:
        env["HF_ENDPOINT"] = DEFAULT_HF_ENDPOINT
    if DEFAULT_HF_HUB_DOWNLOAD_TIMEOUT:
        env["HF_HUB_DOWNLOAD_TIMEOUT"] = DEFAULT_HF_HUB_DOWNLOAD_TIMEOUT
    if DEFAULT_HF_HUB_ETAG_TIMEOUT:
        env["HF_HUB_ETAG_TIMEOUT"] = DEFAULT_HF_HUB_ETAG_TIMEOUT
    if DEFAULT_REMOTE_UV_DEFAULT_INDEX:
        env["UV_DEFAULT_INDEX"] = DEFAULT_REMOTE_UV_DEFAULT_INDEX
        env["PIP_INDEX_URL"] = DEFAULT_REMOTE_UV_DEFAULT_INDEX
    if DEFAULT_REMOTE_UV_EXTRA_INDEX:
        env["UV_EXTRA_INDEX_URL"] = DEFAULT_REMOTE_UV_EXTRA_INDEX
        env["PIP_EXTRA_INDEX_URL"] = DEFAULT_REMOTE_UV_EXTRA_INDEX
    if DEFAULT_REMOTE_UV_HTTP_TIMEOUT:
        env["UV_HTTP_TIMEOUT"] = DEFAULT_REMOTE_UV_HTTP_TIMEOUT
    if remote_adapt_dir:
        # Remote phase4 closure must not inherit the caller's local UV cache.
        # Otherwise `run-remote-cuda` falls back to the remote host's global uv
        # cache and repeatedly deadlocks on a shared `.cache/uv/.lock`.
        env["UV_CACHE_DIR"] = f"{remote_adapt_dir}/.uv_cache_remote"
        env["HF_HOME"] = f"{remote_adapt_dir}/models"
        env["HF_HUB_CACHE"] = f"{remote_adapt_dir}/models"
        env["TRANSFORMERS_CACHE"] = f"{remote_adapt_dir}/models"
        if not allow_online_hub:
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def _remote_runtime_env_prefix(*, remote_adapt_dir: str | None = None, allow_online_hub: bool = False) -> str:
    env = _remote_runtime_env_mapping(remote_adapt_dir=remote_adapt_dir, allow_online_hub=allow_online_hub)
    if not env:
        return ""
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items() if str(value).strip()) + " "


def _load_project_dependency_specs(adapt_dir: Path) -> list[str]:
    pyproject_path = adapt_dir / "pyproject.toml"
    if not pyproject_path.exists():
        return []
    try:
        parsed = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    project = parsed.get("project")
    if not isinstance(project, dict):
        return []
    dependencies = project.get("dependencies")
    if not isinstance(dependencies, list):
        return []
    collected: list[str] = []
    seen_names: set[str] = set()
    for item in dependencies:
        if not isinstance(item, str) or not item.strip():
            continue
        normalized_name = _normalize_requirement_name(item)
        if normalized_name in {"torch", "torchaudio", "torch-npu"} or normalized_name in seen_names:
            continue
        seen_names.add(normalized_name)
        collected.append(item.strip())
    return collected


def _remote_cuda_bootstrap_command(adapt_dir: Path, remote_adapt_dir: str, *, allow_online_hub: bool) -> str:
    env_prefix = _remote_runtime_env_prefix(remote_adapt_dir=remote_adapt_dir, allow_online_hub=allow_online_hub)
    dependency_specs = _load_project_dependency_specs(adapt_dir)
    dependency_install = ""
    if dependency_specs:
        dependency_install = (
            f"{env_prefix}uv pip install --python .venv/bin/python "
            + " ".join(shlex.quote(spec) for spec in dependency_specs)
        )
    probe_modules = [
        "torch",
        "torchaudio",
        "transformers",
        "datasets",
        "google.protobuf",
        "sentencepiece",
        "pyarrow",
        "numpy",
        "safetensors",
    ]
    if any(_normalize_requirement_name(spec) == "ai2-olmo" for spec in dependency_specs):
        probe_modules.append("olmo")
    probe_command = (
        f"{env_prefix}.venv/bin/python -c "
        + shlex.quote("import importlib; " + "; ".join(f"importlib.import_module({module_name!r})" for module_name in probe_modules))
    )
    python_candidates = [
        '"$HOME/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12"',
        '"$HOME/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/bin/python3.13"',
        "python3.12",
        "python3.11",
        "python3.10",
        "python3",
    ]
    resolve_python = (
        "UV_PY=''; "
        + "for candidate in "
        + " ".join(python_candidates)
        + "; do "
        + 'if [ -x "$candidate" ]; then UV_PY="$candidate"; break; fi; '
        + 'if command -v "$candidate" >/dev/null 2>&1; then UV_PY="$(command -v "$candidate")"; break; fi; '
        + "done; "
        + 'if [ -z "$UV_PY" ]; then echo "[business][remote] no usable Python >=3.10 found for CUDA bootstrap" >&2; exit 1; fi'
    )
    install_steps = [
        f"cd {shlex.quote(remote_adapt_dir)}",
        "mkdir -p .uv_cache_remote",
        f"if [ -x .venv/bin/python ] && {probe_command}; then echo '[business][remote] reusing existing CUDA runtime'; exit 0; fi",
        resolve_python,
        f"{env_prefix}uv venv --clear --python \"$UV_PY\" .venv",
        f"{env_prefix}uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu124 'torch>=2.6.0' torchaudio",
    ]
    if dependency_install:
        install_steps.append(dependency_install)
    install_steps.append(probe_command)
    return " && ".join(install_steps)


def _ssh_option_args() -> list[str]:
    return [
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={SSH_CONNECT_TIMEOUT_SECONDS}",
        "-o",
        f"ServerAliveInterval={SSH_SERVER_ALIVE_INTERVAL_SECONDS}",
        "-o",
        f"ServerAliveCountMax={SSH_SERVER_ALIVE_COUNT_MAX}",
        "-o",
        "ControlMaster=auto",
        "-o",
        f"ControlPersist={SSH_CONTROL_PERSIST_SECONDS}",
        "-o",
        f"ControlPath={SSH_CONTROL_PATH}",
    ]


def _ssh_base_args(ssh_host: str) -> list[str]:
    return ["ssh", *_ssh_option_args(), ssh_host]


def _scp_base_args() -> list[str]:
    return ["scp", *_ssh_option_args()]


def _rsync_remote_shell() -> str:
    return _shell_join(["ssh", *_ssh_option_args()])


def _load_json_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} 必须是 JSON object")
    return payload


def _resolve_model(model_id: str) -> dict:
    models = [m for m in get_models(require_optimization_completed=True) if m["model_id"] == model_id]
    if not models:
        raise ValueError(f"未找到模型: {model_id}")
    return models[0]


def _remote_runtime(config: dict, *, ssh_host: str, remote_project_root: str, adapt_dir: Path) -> dict:
    resolved_ssh_host = str(ssh_host or config.get("remote_ssh_host") or DEFAULT_SSH_HOST_ALIAS).strip() or DEFAULT_SSH_HOST_ALIAS
    resolved_remote_project_root = _resolve_remote_project_root(remote_project_root or config.get("remote_project_root"))
    remote_adapt_dir = f"{resolved_remote_project_root}/adaptations/{adapt_dir.name}"
    remote_scripts_dir = f"{resolved_remote_project_root}/scripts"
    remote_models_dir = f"{remote_adapt_dir}/models"
    return {
        "ssh_host": resolved_ssh_host,
        "remote_project_root": resolved_remote_project_root,
        "remote_adapt_dir": remote_adapt_dir,
        "remote_scripts_dir": remote_scripts_dir,
        "remote_models_dir": remote_models_dir,
    }


def _ssh(ssh_host: str, remote_command: str, *, capture_output: bool = False, check: bool = True):
    return _run_command([*_ssh_base_args(ssh_host), remote_command], PROJECT_ROOT, capture_output=capture_output, check=check)


def _ssh_retryable(ssh_host: str, remote_command: str, *, capture_output: bool = False, check: bool = True):
    return _run_retryable_remote_command([*_ssh_base_args(ssh_host), remote_command], PROJECT_ROOT, capture_output=capture_output, check=check)


def _manager_cli_command(subcommand: str, *, model_id: str, ssh_host: str, remote_project_root: str) -> str:
    return _shell_join(
        [
            "uv",
            "run",
            "python",
            MANAGER_SCRIPT_RELATIVE_PATH,
            subcommand,
            "--model",
            model_id,
            "--ssh-host",
            ssh_host,
            "--remote-project-root",
            remote_project_root,
        ]
    )


def _remote_cuda_exec_command(*, ssh_host: str, remote_adapt_dir: str, gpu_id: str) -> str:
    remote_command = f"cd {shlex.quote(remote_adapt_dir)} && {_remote_runtime_env_prefix(remote_adapt_dir=remote_adapt_dir)}CUDA_VISIBLE_DEVICES={shlex.quote(gpu_id)} {_canonical_business_run_command('cuda_baseline')}"
    return _shell_join([*_ssh_base_args(ssh_host), remote_command])


def _find_metric_for_run_id(adapt_dir: Path, role: str, run_id: str) -> Path | None:
    if role == "npu_baseline":
        pattern = "business_metrics_npu_*_baseline.json"
    elif role == "npu_perf":
        pattern = "business_metrics_npu_*_perf.json"
    elif role == "cuda_baseline":
        pattern = "business_metrics_cuda_*_baseline.json"
    else:
        raise ValueError(f"未知 role: {role}")
    candidates = sorted(adapt_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("benchmark_run_id") or "").strip() == run_id:
            return candidate
    return None


def _ensure_local_npu_artifacts(adapt_dir: Path, run_id: str):
    missing_roles = [role for role in ("npu_baseline", "npu_perf") if _find_metric_for_run_id(adapt_dir, role, run_id) is None]
    if missing_roles:
        raise ValueError(f"{adapt_dir.name} 缺少当前轮次 {run_id} 的本机 NPU 工件: {missing_roles}；请先执行 run-npu")


def _remote_models_missing(ssh_host: str, remote_models_dir: str) -> bool:
    result = _ssh_retryable(ssh_host, f"test -d {shlex.quote(remote_models_dir)}", check=False)
    return result.returncode != 0


def _remote_cleanup_pattern_command(remote_adapt_dir: str, patterns: list[str]) -> str:
    pattern_terms = " -o ".join(f"-name {shlex.quote(pattern)}" for pattern in patterns)
    return f"if [ -d {shlex.quote(remote_adapt_dir)} ]; then find {shlex.quote(remote_adapt_dir)} -maxdepth 1 -type f \\( {pattern_terms} \\) -delete; fi"


def _remote_cleanup_business_artifacts_command(remote_adapt_dir: str) -> str:
    return _remote_cleanup_pattern_command(remote_adapt_dir, BUSINESS_PATTERNS)


def _remote_cleanup_runtime_artifacts_command(remote_adapt_dir: str) -> str:
    return _remote_cleanup_pattern_command(remote_adapt_dir, REMOTE_RUNTIME_CLEANUP_PATTERNS)


def _cleanup_remote_business_artifacts(ssh_host: str, remote_adapt_dir: str):
    remote_command = _remote_cleanup_business_artifacts_command(remote_adapt_dir)
    _ssh_retryable(ssh_host, remote_command)


def _cleanup_remote_runtime_artifacts(ssh_host: str, remote_adapt_dir: str):
    remote_command = _remote_cleanup_runtime_artifacts_command(remote_adapt_dir)
    _ssh_retryable(ssh_host, remote_command)


def _cleanup_remote_dynamic_module_cache(ssh_host: str, remote_models_dir: str):
    remote_modules_dir = Path(remote_models_dir) / "modules"
    _ssh_retryable(ssh_host, f"rm -rf {shlex.quote(str(remote_modules_dir))}")
    print(f"[business][remote] cleared dynamic module cache: {remote_modules_dir}")


def _sync_remote_dynamic_module_support_cache(local_models_dir: Path, remote: dict, *, snapshot_dirs: list[Path] | None = None) -> None:
    if not local_models_dir.is_dir():
        return

    ssh_host = remote["ssh_host"]
    remote_models_dir = Path(remote["remote_models_dir"])
    remote_modules_root = remote_models_dir / "modules"
    remote_transformers_modules_root = remote_modules_root / "transformers_modules"
    synced_snapshots = 0
    synced_files = 0

    planned_snapshot_dirs = _iter_local_snapshot_dirs(local_models_dir) if snapshot_dirs is None else snapshot_dirs
    for snapshot_dir in planned_snapshot_dirs:
        remote_module_dir = remote_transformers_modules_root / f"_{snapshot_dir.name}"
        _ssh_retryable(
            ssh_host,
            " && ".join(
                [
                    f"mkdir -p {shlex.quote(str(remote_module_dir))}",
                    f"touch {shlex.quote(str(remote_modules_root / '__init__.py'))}",
                    f"touch {shlex.quote(str(remote_transformers_modules_root / '__init__.py'))}",
                    f"touch {shlex.quote(str(remote_module_dir / '__init__.py'))}",
                ]
            ),
        )
        _run_retryable_remote_command(
            [
                "rsync",
                "-avL",
                "--delete",
                "--prune-empty-dirs",
                "--no-o",
                "--no-g",
                *[f"--exclude={pattern}" for pattern in REMOTE_MODEL_WEIGHT_FILE_PATTERNS],
                *[f"--exclude={pattern}" for pattern in REMOTE_SYNC_TRANSIENT_FILE_PATTERNS],
                *[f"--exclude={pattern}" for pattern in REMOTE_SNAPSHOT_EXPORT_EXCLUDE_PATTERNS],
                "-e",
                _rsync_remote_shell(),
                f"{snapshot_dir}/",
                f"{ssh_host}:{remote_module_dir}/",
            ],
            PROJECT_ROOT,
        )
        synced_snapshots += 1
        synced_files += _count_local_non_weight_support_files(snapshot_dir)

    if synced_snapshots:
        print(f"[business][remote] mirrored {synced_files} dynamic-module support asset(s) across {synced_snapshots} snapshot cache dir(s)")


def _iter_local_input_snapshot_assets(local_models_dir: Path) -> list[tuple[Path, list[Path]]]:
    snapshot_assets: list[tuple[Path, list[Path]]] = []
    for cache_dir in sorted(local_models_dir.glob("models--*")):
        snapshots_dir = cache_dir / "snapshots"
        if not snapshots_dir.is_dir():
            continue
        for snapshot_dir in sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), reverse=True):
            config_path = snapshot_dir / "config.json"
            asset_paths: list[Path] = []
            seen_assets: set[Path] = set()
            if config_path.exists():
                asset_paths.append(config_path)
                seen_assets.add(config_path)
            for name in REMOTE_INPUT_SNAPSHOT_ASSET_FILENAMES:
                if name == "config.json":
                    continue
                if any(token in name for token in ("*", "?", "[")):
                    for candidate in sorted(snapshot_dir.glob(name)):
                        if candidate.exists() and candidate not in seen_assets:
                            asset_paths.append(candidate)
                            seen_assets.add(candidate)
                    continue
                candidate = snapshot_dir / name
                if candidate.exists() and candidate not in seen_assets:
                    asset_paths.append(candidate)
                    seen_assets.add(candidate)
            has_non_config_input_asset = any(path.name != "config.json" for path in asset_paths)
            if not has_non_config_input_asset:
                continue
            snapshot_assets.append((snapshot_dir, asset_paths))
    return snapshot_assets


def _iter_local_support_snapshot_dirs(local_models_dir: Path) -> list[Path]:
    support_dirs: list[Path] = []
    for cache_dir in sorted(local_models_dir.glob("models--*")):
        for base_name in ("snapshots", ".no_exist"):
            base_dir = cache_dir / base_name
            if not base_dir.is_dir():
                continue
            support_dirs.extend(sorted((path for path in base_dir.iterdir() if path.is_dir()), reverse=True))
    support_dirs.extend(_iter_local_exported_snapshot_dirs(local_models_dir))
    return support_dirs


def _count_local_non_weight_support_files(root_dir: Path) -> int:
    count = 0
    for candidate in root_dir.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.name.endswith(REMOTE_MODEL_WEIGHT_FILE_SUFFIXES):
            continue
        if any(part in {"onnx", "openvino"} for part in candidate.relative_to(root_dir).parts[:-1]):
            continue
        if candidate.suffix in {".onnx", ".xml"}:
            continue
        count += 1
    return count


def _rsync_remote_weight_protect_args() -> list[str]:
    args: list[str] = []
    for pattern in REMOTE_MODEL_WEIGHT_FILE_PATTERNS:
        args.extend(["--filter", f"P {pattern}"])
    return args


def _iter_local_snapshot_symlink_target_files(local_models_dir: Path, snapshot_dir: Path) -> list[Path]:
    target_files: list[Path] = []
    seen: set[str] = set()
    try:
        resolved_models_dir = local_models_dir.resolve()
    except Exception:
        resolved_models_dir = local_models_dir
    for candidate in snapshot_dir.rglob("*"):
        try:
            if not candidate.is_symlink():
                continue
            resolved_target = candidate.resolve()
        except OSError:
            continue
        if not resolved_target.is_file():
            continue
        try:
            resolved_target.relative_to(resolved_models_dir)
        except ValueError:
            continue
        target_key = str(resolved_target)
        if target_key in seen:
            continue
        seen.add(target_key)
        target_files.append(resolved_target)
    return target_files


def _iter_local_special_snapshot_dirs(local_models_dir: Path) -> list[Path]:
    special_snapshot_dirs: list[Path] = []
    for snapshot_dir in _iter_local_snapshot_dirs(local_models_dir):
        if (snapshot_dir / "meta.json").exists() and (snapshot_dir / "config.cfg").exists():
            special_snapshot_dirs.append(snapshot_dir)
    return special_snapshot_dirs


def _iter_local_exported_snapshot_dirs(local_models_dir: Path) -> list[Path]:
    return sorted((path for path in local_models_dir.glob("local_snapshot*") if path.is_dir()), reverse=True)


def _remote_special_snapshot_is_ready(ssh_host: str, remote_snapshot_dir: Path) -> bool:
    remote_transformer_model_dir = remote_snapshot_dir / "transformer" / "model"
    patterns = (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "vocab.txt",
        "*.bin",
        "*.safetensors",
    )
    pattern_terms = " -o ".join(f"-name {shlex.quote(pattern)}" for pattern in patterns)
    remote_command = (
        f"if [ -f {shlex.quote(str(remote_snapshot_dir / 'meta.json'))} ] "
        f"&& [ -f {shlex.quote(str(remote_snapshot_dir / 'config.cfg'))} ] "
        f"&& [ -d {shlex.quote(str(remote_transformer_model_dir))} ] "
        f"&& find {shlex.quote(str(remote_transformer_model_dir))} -maxdepth 1 \\( -type f -o -xtype f \\) \\( {pattern_terms} \\) | head -n 1 | grep -q .; "
        f"then exit 0; else exit 1; fi"
    )
    result = _ssh_retryable(ssh_host, remote_command, check=False)
    return result.returncode == 0


def _iter_local_snapshot_dirs(local_models_dir: Path) -> list[Path]:
    snapshot_dirs: list[Path] = []
    for cache_dir in sorted(local_models_dir.glob("models--*")):
        snapshots_dir = cache_dir / "snapshots"
        if not snapshots_dir.is_dir():
            continue
        snapshot_dirs.extend(sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), reverse=True))
    return snapshot_dirs


def _iter_local_model_snapshot_asset_dirs(local_models_dir: Path) -> list[Path]:
    asset_dirs: list[Path] = []
    seen: set[str] = set()
    for snapshot_dir in _iter_local_snapshot_dirs(local_models_dir):
        for asset_dir in _iter_snapshot_asset_dirs_with_weights(snapshot_dir):
            asset_dir_str = str(asset_dir)
            if asset_dir_str in seen:
                continue
            seen.add(asset_dir_str)
            asset_dirs.append(asset_dir)
    for snapshot_dir in _iter_local_exported_snapshot_dirs(local_models_dir):
        if not _snapshot_has_model_weight_assets(snapshot_dir):
            continue
        snapshot_dir_str = str(snapshot_dir)
        if snapshot_dir_str in seen:
            continue
        seen.add(snapshot_dir_str)
        asset_dirs.append(snapshot_dir)
    return asset_dirs


def _iter_snapshot_asset_dirs_with_weights(snapshot_dir: Path) -> list[Path]:
    asset_dirs: list[Path] = []
    seen: set[str] = set()
    for asset_dir in _iter_snapshot_asset_dirs(snapshot_dir):
        asset_dir_str = str(asset_dir)
        if asset_dir_str in seen:
            continue
        seen.add(asset_dir_str)
        if _snapshot_has_model_weight_assets(asset_dir):
            asset_dirs.append(asset_dir)
    return asset_dirs


def _snapshot_has_model_weight_assets(snapshot_dir: Path) -> bool:
    index_names = (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    if any((snapshot_dir / name).exists() for name in index_names):
        return True
    return any(any(snapshot_dir.glob(pattern)) for pattern in REMOTE_MODEL_WEIGHT_FILE_PATTERNS)


def _iter_local_snapshot_required_weight_entries(snapshot_dir: Path) -> list[str]:
    required_names: list[str] = []
    seen: set[str] = set()
    for fixed_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        candidate = snapshot_dir / fixed_name
        if candidate.exists() and fixed_name not in seen:
            seen.add(fixed_name)
            required_names.append(fixed_name)
    for candidate in sorted(snapshot_dir.iterdir()):
        if not candidate.is_file() and not candidate.is_symlink():
            continue
        if not candidate.name.endswith(REMOTE_MODEL_WEIGHT_FILE_SUFFIXES):
            continue
        if candidate.name in seen:
            continue
        seen.add(candidate.name)
        required_names.append(candidate.name)
    return required_names


def _remote_snapshot_has_model_weight_assets(ssh_host: str, local_snapshot_dir: Path, remote_snapshot_dir: Path) -> bool:
    required_names = _iter_local_snapshot_required_weight_entries(local_snapshot_dir)
    if not required_names:
        return False
    checks = " && ".join(f"test -e {shlex.quote(str(remote_snapshot_dir / name))}" for name in required_names)
    remote_command = f"test -d {shlex.quote(str(remote_snapshot_dir))} && {checks}"
    result = _ssh_retryable(ssh_host, remote_command, check=False)
    return result.returncode == 0


def _remote_missing_model_snapshot_assets(local_models_dir: Path, remote: dict, *, asset_dirs: list[Path] | None = None) -> bool:
    ssh_host = remote["ssh_host"]
    remote_models_dir = Path(remote["remote_models_dir"])
    planned_asset_dirs = _iter_local_model_snapshot_asset_dirs(local_models_dir) if asset_dirs is None else asset_dirs
    for asset_dir in planned_asset_dirs:
        relative_asset_dir = asset_dir.relative_to(local_models_dir)
        remote_asset_dir = remote_models_dir / relative_asset_dir
        if not _remote_snapshot_has_model_weight_assets(ssh_host, asset_dir, remote_asset_dir):
            return True
    return False


def _sync_remote_model_snapshot_assets(local_models_dir: Path, remote: dict, *, asset_dirs: list[Path] | None = None) -> None:
    ssh_host = remote["ssh_host"]
    remote_models_dir = Path(remote["remote_models_dir"])
    synced_asset_dirs = 0
    synced_blob_files = 0
    planned_asset_dirs = _iter_local_model_snapshot_asset_dirs(local_models_dir) if asset_dirs is None else asset_dirs
    for asset_dir in planned_asset_dirs:
        relative_asset_dir = asset_dir.relative_to(local_models_dir)
        remote_asset_dir = remote_models_dir / relative_asset_dir
        if _remote_snapshot_has_model_weight_assets(ssh_host, asset_dir, remote_asset_dir):
            continue
        referenced_blob_files = _iter_local_snapshot_symlink_target_files(local_models_dir, asset_dir)
        for blob_file in referenced_blob_files:
            relative_blob_file = blob_file.relative_to(local_models_dir)
            remote_blob_file = remote_models_dir / relative_blob_file
            remote_blob_check = _ssh_retryable(ssh_host, f"test -f {shlex.quote(str(remote_blob_file))}", check=False)
            if remote_blob_check.returncode == 0:
                continue
            _ssh_retryable(
                ssh_host,
                f"mkdir -p {shlex.quote(str(remote_blob_file.parent))}",
            )
            _run_retryable_remote_command(
                [
                    "rsync",
                    "-av",
                    "--no-o",
                    "--no-g",
                    "-e",
                    _rsync_remote_shell(),
                    str(blob_file),
                    f"{ssh_host}:{remote_blob_file}",
                ],
                PROJECT_ROOT,
            )
            synced_blob_files += 1
        _ssh_retryable(
            ssh_host,
            f"rm -rf {shlex.quote(str(remote_asset_dir))} && mkdir -p {shlex.quote(str(remote_asset_dir))}",
        )
        _run_retryable_remote_command(
            [
                "rsync",
                "-avL",
                "--no-o",
                "--no-g",
                "-e",
                _rsync_remote_shell(),
                f"{asset_dir}/",
                f"{ssh_host}:{remote_asset_dir}/",
            ],
            PROJECT_ROOT,
        )
        synced_asset_dirs += 1
    if synced_asset_dirs:
        print(f"[business][remote] refreshed {synced_asset_dirs} model snapshot asset dir(s)")
    if synced_blob_files:
        print(f"[business][remote] refreshed {synced_blob_files} snapshot blob file(s)")


def _sync_remote_input_snapshot_assets(local_models_dir: Path, remote: dict, *, snapshot_dirs: list[Path] | None = None) -> None:
    support_snapshot_dirs = _iter_local_support_snapshot_dirs(local_models_dir) if snapshot_dirs is None else snapshot_dirs
    special_snapshot_dirs = [snapshot_dir for snapshot_dir in support_snapshot_dirs if (snapshot_dir / "meta.json").exists() and (snapshot_dir / "config.cfg").exists()]
    if not support_snapshot_dirs and not special_snapshot_dirs:
        return

    ssh_host = remote["ssh_host"]
    remote_models_dir = Path(remote["remote_models_dir"])
    synced_special_snapshots = 0
    for snapshot_dir in special_snapshot_dirs:
        relative_snapshot_dir = snapshot_dir.relative_to(local_models_dir)
        remote_snapshot_dir = remote_models_dir / relative_snapshot_dir
        if _remote_special_snapshot_is_ready(ssh_host, remote_snapshot_dir):
            continue
        _ssh_retryable(
            ssh_host,
            f"rm -rf {shlex.quote(str(remote_snapshot_dir))} && mkdir -p {shlex.quote(str(remote_snapshot_dir))}",
        )
        _run_retryable_remote_command(
            [
                "rsync",
                "-avL",
                "--delete",
                "--no-o",
                "--no-g",
                "-e",
                _rsync_remote_shell(),
                f"{snapshot_dir}/",
                f"{ssh_host}:{remote_snapshot_dir}/",
            ],
            PROJECT_ROOT,
        )
        synced_special_snapshots += 1

    synced_snapshots = 0
    synced_files = 0
    for snapshot_dir in support_snapshot_dirs:
        if snapshot_dir in special_snapshot_dirs:
            continue
        relative_snapshot_dir = snapshot_dir.relative_to(local_models_dir)
        remote_snapshot_dir = remote_models_dir / relative_snapshot_dir
        _ssh_retryable(
            ssh_host,
            f"mkdir -p {shlex.quote(str(remote_snapshot_dir))}",
        )
        _run_retryable_remote_command(
            [
                "rsync",
                "-avL",
                "--delete",
                "--prune-empty-dirs",
                "--no-o",
                "--no-g",
                *_rsync_remote_weight_protect_args(),
                *[f"--exclude={pattern}" for pattern in REMOTE_MODEL_WEIGHT_FILE_PATTERNS],
                *[f"--exclude={pattern}" for pattern in REMOTE_SYNC_TRANSIENT_FILE_PATTERNS],
                *[f"--exclude={pattern}" for pattern in REMOTE_SNAPSHOT_EXPORT_EXCLUDE_PATTERNS],
                "-e",
                _rsync_remote_shell(),
                f"{snapshot_dir}/",
                f"{ssh_host}:{remote_snapshot_dir}/",
            ],
            PROJECT_ROOT,
        )
        synced_snapshots += 1
        synced_files += _count_local_non_weight_support_files(snapshot_dir)

    if synced_special_snapshots:
        print(f"[business][remote] refreshed {synced_special_snapshots} special snapshot dir(s)")
    print(f"[business][remote] refreshed {synced_files} local snapshot support asset(s) across {synced_snapshots} dir(s)")


def _resolve_local_business_dataset_cache(adapt_dir: Path) -> tuple[str, Path] | None:
    config = _load_config(adapt_dir, missing_ok=True)
    model_id = str(config.get("model_id") or adapt_dir.name).strip() or adapt_dir.name
    profile = _resolve_business_profile(model_id, config, adapt_dir)
    dataset_key = str(profile.get("dataset_key") or "").strip()
    if not dataset_key:
        return None

    try:
        canonical_path = get_dataset_disk_path(dataset_key, cache_dir=PROJECT_ROOT / "datasets")
    except Exception:
        canonical_path = None

    candidate_paths: list[Path] = []
    dataset_local_path = str(config.get("dataset_local_path") or "").strip()
    resolved_local_path = _resolve_config_path(dataset_local_path, adapt_dir=adapt_dir)
    if resolved_local_path is not None:
        candidate_paths.append(resolved_local_path)
    if canonical_path is not None and canonical_path not in candidate_paths:
        candidate_paths.append(canonical_path)

    for candidate in candidate_paths:
        if candidate.is_dir():
            return dataset_key, candidate
    return None


def _remote_business_dataset_is_ready(ssh_host: str, remote_dataset_dir: str) -> bool:
    readiness_command = f"test -d {shlex.quote(remote_dataset_dir)} && test -f {shlex.quote(remote_dataset_dir + '/state.json')} && find {shlex.quote(remote_dataset_dir)} -maxdepth 1 -type f -name '*.arrow' -print -quit | grep -q ."
    result = _ssh(ssh_host, readiness_command, check=False)
    return result.returncode == 0


def _sync_remote_business_dataset_cache(adapt_dir: Path, remote: dict) -> None:
    resolved = _resolve_local_business_dataset_cache(adapt_dir)
    if not resolved:
        return

    dataset_key, local_dataset_dir = resolved
    ssh_host = remote["ssh_host"]
    remote_datasets_root = f"{remote['remote_project_root']}/datasets"
    remote_dataset_dir = f"{remote_datasets_root}/{local_dataset_dir.name}"
    if _remote_business_dataset_is_ready(ssh_host, remote_dataset_dir):
        return

    _ssh_retryable(
        ssh_host,
        f"mkdir -p {shlex.quote(remote_datasets_root)} && rm -rf {shlex.quote(remote_dataset_dir)}",
    )
    _run_retryable_remote_command(
        [
            "rsync",
            "-avL",
            "--delete",
            "--no-o",
            "--no-g",
            "-e",
            _rsync_remote_shell(),
            f"{local_dataset_dir}/",
            f"{ssh_host}:{remote_dataset_dir}/",
        ],
        PROJECT_ROOT,
    )
    print(f"[business][remote] synced local dataset cache {local_dataset_dir.name} -> {remote_dataset_dir}")


def _sync_remote_workspace(adapt_dir: Path, remote: dict):
    ssh_host = remote["ssh_host"]
    remote_project_root = remote["remote_project_root"]
    remote_adapt_dir = remote["remote_adapt_dir"]
    remote_scripts_dir = remote["remote_scripts_dir"]
    remote_models_dir = remote["remote_models_dir"]

    _ssh_retryable(
        ssh_host,
        f"test -d {shlex.quote(remote_project_root)} && mkdir -p {shlex.quote(remote_adapt_dir)} {shlex.quote(remote_scripts_dir)}",
    )
    _cleanup_remote_business_artifacts(ssh_host, remote_adapt_dir)

    rsync_adapt = ["rsync", "-av", "--no-o", "--no-g", "-e", _rsync_remote_shell()]
    rsync_adapt.extend(f"--exclude={pattern}" for pattern in REMOTE_ADAPT_SYNC_EXCLUDES)
    rsync_adapt.extend([f"{adapt_dir}/", f"{ssh_host}:{remote_adapt_dir}/"])
    _run_retryable_remote_command(rsync_adapt, PROJECT_ROOT)

    root_files = [str(SCRIPTS_DIR / name) for name in REMOTE_ROOT_SYNC_FILES if (SCRIPTS_DIR / name).exists()]
    if root_files:
        _run_retryable_remote_command(["rsync", "-av", "--no-o", "--no-g", "-e", _rsync_remote_shell(), *root_files, f"{ssh_host}:{remote_scripts_dir}/"], PROJECT_ROOT)

    _sync_remote_business_dataset_cache(adapt_dir, remote)

    local_models_dir = adapt_dir / "models"
    if local_models_dir.is_dir():
        remote_snapshot_plan = _build_required_remote_snapshot_plan(adapt_dir)
        planned_weight_asset_dirs = remote_snapshot_plan["weight_asset_dirs"]
        planned_support_snapshot_dirs = remote_snapshot_plan["support_snapshot_dirs"]
        if planned_weight_asset_dirs:
            planned_weight_rel_paths = [str(path.relative_to(local_models_dir)) for path in planned_weight_asset_dirs]
            print(f"[business][remote] planned weight snapshot assets: {json.dumps(planned_weight_rel_paths, ensure_ascii=False)}")
        remote_missing_models_dir = _remote_models_missing(ssh_host, remote_models_dir)
        if remote_missing_models_dir:
            _ssh_retryable(ssh_host, f"mkdir -p {shlex.quote(remote_models_dir)}")
            print("[business][remote] remote models dir missing; creating it and syncing targeted snapshot assets only")
        elif _remote_missing_model_snapshot_assets(local_models_dir, remote, asset_dirs=planned_weight_asset_dirs):
            print("[business][remote] remote model snapshot weights missing; syncing targeted snapshot assets only")
        _sync_remote_model_snapshot_assets(local_models_dir, remote, asset_dirs=planned_weight_asset_dirs)
        _sync_remote_input_snapshot_assets(local_models_dir, remote, snapshot_dirs=planned_support_snapshot_dirs)


def _list_remote_artifacts(remote: dict) -> list[str]:
    remote_command = f"find {shlex.quote(remote['remote_adapt_dir'])} -maxdepth 1 -type f -name 'business_metrics_cuda_*_baseline.json' -print | sort"
    result = _ssh_retryable(remote["ssh_host"], remote_command, capture_output=True)
    return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]


def _validate_remote_cuda_artifact(metric: dict, *, remote_name: str) -> tuple[bool, str]:
    device = str(metric.get("device") or "").strip().lower()
    if not device:
        return False, f"{remote_name} 缺少 device，无法确认这是纯 CUDA 工件"
    if "cuda" not in device:
        return False, f"{remote_name} 的 device={metric.get('device')}，不是 CUDA 工件"
    scenario = str(metric.get("scenario") or "").strip()
    if scenario and scenario != "cuda_baseline":
        return False, f"{remote_name} 的 scenario={scenario}，不是 cuda_baseline"
    suspicious_keys = sorted(str(key) for key in metric if str(key).startswith("npu_"))
    if suspicious_keys:
        return False, f"{remote_name} 包含可疑 NPU 字段: {', '.join(suspicious_keys[:5])}"
    return True, ""


def _next_conflict_path(conflict_dir: Path, original_name: str) -> Path:
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix or ".json"
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    candidate = conflict_dir / f"{stem}__remote_conflict_{timestamp}{suffix}"
    index = 1
    while candidate.exists():
        index += 1
        candidate = conflict_dir / f"{stem}__remote_conflict_{timestamp}_{index}{suffix}"
    return candidate


def _next_prev_backup_path(path: Path, *, tag: str = "rule_refresh") -> Path:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    candidate = path.with_name(f"{path.name}__prev_{tag}_{timestamp}")
    index = 1
    while candidate.exists():
        index += 1
        candidate = path.with_name(f"{path.name}__prev_{tag}_{timestamp}_{index}")
    return candidate


def _quarantine_remote_artifact(staged_path: Path, adapt_dir: Path, *, reason: str) -> Path:
    conflict_dir = adapt_dir / REMOTE_FETCH_CONFLICT_DIRNAME
    conflict_dir.mkdir(parents=True, exist_ok=True)
    conflict_path = _next_conflict_path(conflict_dir, staged_path.name)
    shutil.move(str(staged_path), str(conflict_path))
    print(f"[business][fetch][quarantine] {conflict_path.name}: {reason}")
    return conflict_path


def _ingest_remote_cuda_artifact(staged_path: Path, adapt_dir: Path) -> Path | None:
    remote_name = staged_path.name
    try:
        metric = _load_json_object(staged_path)
    except Exception as exc:
        _quarantine_remote_artifact(staged_path, adapt_dir, reason=f"非法 JSON 工件: {exc}")
        return None

    valid, reason = _validate_remote_cuda_artifact(metric, remote_name=remote_name)
    if not valid:
        _quarantine_remote_artifact(staged_path, adapt_dir, reason=reason)
        return None

    destination = adapt_dir / remote_name
    if destination.exists():
        if destination.read_bytes() == staged_path.read_bytes():
            staged_path.unlink()
            print(f"[business][fetch][skip] 保留本地同名工件: {destination.name}（内容一致）")
            return None
        local_run_id = ""
        try:
            local_metric = _load_json_object(destination)
            local_run_id = str(local_metric.get("benchmark_run_id") or "").strip()
        except Exception:
            local_run_id = ""
        remote_run_id = str(metric.get("benchmark_run_id") or "").strip()
        backup_path = _next_prev_backup_path(destination)
        shutil.move(str(destination), str(backup_path))
        reason = "run_id 变化，自动备份旧正式工件" if remote_run_id and remote_run_id != local_run_id else "内容变化，自动备份旧正式工件"
        print(f"[business][fetch][backup] {backup_path.name}: {reason}")

    shutil.move(str(staged_path), str(destination))
    return destination


def _fetch_remote_artifacts(adapt_dir: Path, remote: dict) -> list[Path]:
    remote_paths = _list_remote_artifacts(remote)
    if not remote_paths:
        raise FileNotFoundError(f"远端未找到 CUDA baseline 工件: {remote['remote_adapt_dir']}")
    fetched_paths: list[Path] = []
    stage_dir = adapt_dir / REMOTE_FETCH_STAGE_DIRNAME
    stage_dir.mkdir(parents=True, exist_ok=True)
    for remote_path in remote_paths:
        staged_path = stage_dir / Path(remote_path).name
        if staged_path.exists():
            staged_path.unlink()
        _run_retryable_remote_command([*_scp_base_args(), f"{remote['ssh_host']}:{remote_path}", str(staged_path)], PROJECT_ROOT)
        accepted_path = _ingest_remote_cuda_artifact(staged_path, adapt_dir)
        if accepted_path is not None:
            fetched_paths.append(accepted_path)
    if stage_dir.exists() and not any(stage_dir.iterdir()):
        stage_dir.rmdir()
    return fetched_paths


def _run_remote_cuda_and_fetch(adapt_dir: Path, remote: dict, *, gpu_id: str):
    ssh_host = remote["ssh_host"]
    remote_project_root = remote["remote_project_root"]
    remote_adapt_dir = remote["remote_adapt_dir"]
    remote_models_dir = remote["remote_models_dir"]
    local_models_dir = adapt_dir / "models"
    remote_snapshot_plan = _build_required_remote_snapshot_plan(adapt_dir)
    allow_online_hub = not remote_snapshot_plan["weight_asset_dirs"] and not remote_snapshot_plan["support_snapshot_dirs"]
    remote_env = _remote_runtime_env_mapping(remote_adapt_dir=remote_adapt_dir, allow_online_hub=allow_online_hub)

    _ssh_retryable(ssh_host, f"cd {shlex.quote(remote_project_root)} && pwd && test -d {shlex.quote(remote_adapt_dir)}")
    _sync_remote_workspace(adapt_dir, remote)
    _cleanup_remote_dynamic_module_cache(ssh_host, remote_models_dir)
    _sync_remote_dynamic_module_support_cache(local_models_dir, remote, snapshot_dirs=remote_snapshot_plan["support_snapshot_dirs"])
    if remote_env:
        print(f"[business][remote] runtime env: {json.dumps(remote_env, ensure_ascii=False, sort_keys=True)}")
    _ssh_retryable(
        ssh_host,
        _remote_cuda_bootstrap_command(adapt_dir, remote_adapt_dir, allow_online_hub=allow_online_hub),
    )
    for run_index in range(1, DEFAULT_REMOTE_CUDA_BASELINE_TOTAL_RUNS + 1):
        phase = "warmup" if run_index < DEFAULT_REMOTE_CUDA_BASELINE_TOTAL_RUNS else "formal"
        print(f"[business][remote][cuda] {phase} run {run_index}/{DEFAULT_REMOTE_CUDA_BASELINE_TOTAL_RUNS}")
        _ssh(
            ssh_host,
            f"cd {shlex.quote(remote_adapt_dir)} && {_remote_runtime_env_prefix(remote_adapt_dir=remote_adapt_dir, allow_online_hub=allow_online_hub)}CUDA_VISIBLE_DEVICES={shlex.quote(gpu_id)} {_canonical_business_run_command('cuda_baseline')}",
        )
        if run_index < DEFAULT_REMOTE_CUDA_BASELINE_TOTAL_RUNS:
            print("[business][remote][cuda] discarding warmup artifacts before formal run")
            _cleanup_remote_runtime_artifacts(ssh_host, remote_adapt_dir)
    fetched = _fetch_remote_artifacts(adapt_dir, remote)
    print(f"[business][remote] fetched {len(fetched)} artifact(s)")


def _write_summary(adapt_dir: Path, *, comparison_scope: str, remote_mode: str):
    summary = build_summary(adapt_dir, comparison_scope=comparison_scope, remote_mode=remote_mode)
    run_id = str(summary.get("benchmark_run_id") or "").strip()
    started_at = str(summary.get("benchmark_run_started_at") or "").strip()
    if run_id and not started_at:
        config_path = adapt_dir / "business_benchmark_config.json"
        if config_path.exists():
            config = _load_json(config_path)
            configured_started_at = str(config.get("benchmark_run_started_at") or "").strip()
            if configured_started_at:
                summary["benchmark_run_started_at"] = configured_started_at
    output = adapt_dir / "business_summary.json"
    _write_json_text(output, summary)
    print(f"[business] wrote {output}")
    return summary


def cmd_list(args):
    status = args.status or "completed"
    models = get_models(status)
    print(f"\nBusiness benchmark 模型列表 (business_benchmark_status={status})\n")
    if not models:
        print("没有找到符合条件的模型\n")
        return
    for index, model in enumerate(models, 1):
        print(f"{index}. {model['model_id']}")
        print(f"   optimization={model['optimization_status']} business={model['business_benchmark_status']} path={model['adaptation_path']}")


def cmd_run_npu(args):
    if args.model:
        models = [_resolve_model(args.model)]
    else:
        models = get_models(status="pending", require_optimization_completed=True)
    if not models:
        print("没有可运行的 business benchmark 模型")
        return
    for model in models:
        adapt_dir = _resolve_adapt_dir(model)
        _quarantine_stale_phase4_local_env(adapt_dir)
        config = _prepare_business_config(model, adapt_dir, ensure_local_dataset=True, refresh_run_id=True)
        _refresh_business_stage_start(model, adapt_dir, config)
        config = _apply_local_npu_device_selection(adapt_dir, config)
        _ensure_business_eval_script(adapt_dir)
        _ensure_business_model_eval_script(adapt_dir)
        _ensure_local_business_runtime_installed(adapt_dir)
        print(f"[business][npu] {model['model_id']} @ {adapt_dir}")
        _run_local_business_eval(adapt_dir, config, "npu_baseline")
        _run_local_business_eval(adapt_dir, config, "npu_perf")


def cmd_print_remote_command(args):
    if not args.model:
        raise ValueError("--model 必填")
    model = _resolve_model(args.model)
    adapt_dir = _resolve_adapt_dir(model)
    if args.reuse_run_id and args.fresh_run_id:
        raise ValueError("--reuse-run-id 与 --fresh-run-id 不能同时使用")
    config = _prepare_business_config(model, adapt_dir, ensure_local_dataset=False, refresh_run_id=bool(args.fresh_run_id))
    _ensure_business_eval_script(adapt_dir)
    _ensure_business_model_eval_script(adapt_dir)
    _ensure_business_run_script(adapt_dir, config)
    profile = _resolve_business_profile(model["model_id"], config, adapt_dir)
    dataset_label = str(profile.get("dataset_key") or "None")
    eval_profile = str(profile.get("evaluation_profile") or "")
    remote = _remote_runtime(
        config,
        ssh_host=args.ssh_host,
        remote_project_root=args.remote_project_root,
        adapt_dir=adapt_dir,
    )
    remote_snapshot_plan = _build_required_remote_snapshot_plan(adapt_dir)
    allow_online_hub = not remote_snapshot_plan["weight_asset_dirs"] and not remote_snapshot_plan["support_snapshot_dirs"]
    sync_command = _manager_cli_command(
        "sync-remote-workspace",
        model_id=args.model,
        ssh_host=remote["ssh_host"],
        remote_project_root=remote["remote_project_root"],
    )
    fetch_command = _manager_cli_command(
        "fetch-remote-artifacts",
        model_id=args.model,
        ssh_host=remote["ssh_host"],
        remote_project_root=remote["remote_project_root"],
    )
    remote_run_command = _remote_cuda_exec_command(
        ssh_host=remote["ssh_host"],
        remote_adapt_dir=remote["remote_adapt_dir"],
        gpu_id=args.gpu_id,
    )
    summarize_command = _shell_join(
        [
            "uv",
            "run",
            "python",
            MANAGER_SCRIPT_RELATIVE_PATH,
            "summarize",
            "--model",
            args.model,
            "--remote-mode",
            DEFAULT_REMOTE_EXECUTION_MODE,
        ]
    )
    print("# 建议远端执行步骤（SSH 直传，无需 git pull）")
    print(f"# ssh_host={remote['ssh_host']}")
    print(f"# remote_project_root={remote['remote_project_root']}")
    print(f"# dataset={dataset_label} eval_profile={eval_profile}")
    print(sync_command)
    print(
        _shell_join(
            [
                *_ssh_base_args(remote["ssh_host"]),
                _remote_cuda_bootstrap_command(adapt_dir, remote["remote_adapt_dir"], allow_online_hub=allow_online_hub),
            ]
        )
    )
    print("# 第 1 次 CUDA baseline 仅用于热身，结果丢弃")
    print(remote_run_command)
    print(_shell_join([*_ssh_base_args(remote["ssh_host"]), _remote_cleanup_runtime_artifacts_command(remote["remote_adapt_dir"])]))
    print("# 第 2 次 CUDA baseline 才是正式结果")
    print(remote_run_command)
    print("\n# 远端完成后本地回收并汇总")
    print(fetch_command)
    print(summarize_command)
    print(f"# 若 SSH 已打通且希望一键自动执行，可直接运行:")
    print(
        _shell_join(
            [
                "uv",
                "run",
                "python",
                MANAGER_SCRIPT_RELATIVE_PATH,
                "run-remote-cuda",
                "--model",
                args.model,
                "--ssh-host",
                remote["ssh_host"],
                "--remote-project-root",
                remote["remote_project_root"],
                "--gpu-id",
                args.gpu_id,
            ]
        )
    )


def cmd_sync_remote_workspace(args):
    if not args.model:
        raise ValueError("--model 必填")
    model = _resolve_model(args.model)
    adapt_dir = _resolve_adapt_dir(model)
    config = _prepare_business_config(model, adapt_dir, ensure_local_dataset=False)
    _ensure_business_eval_script(adapt_dir)
    _ensure_business_model_eval_script(adapt_dir)
    _ensure_business_run_script(adapt_dir, config)
    remote = _remote_runtime(
        config,
        ssh_host=args.ssh_host,
        remote_project_root=args.remote_project_root,
        adapt_dir=adapt_dir,
    )
    print(f"[business][sync] {model['model_id']} -> {remote['ssh_host']}:{remote['remote_adapt_dir']}")
    _sync_remote_workspace(adapt_dir, remote)


def cmd_fetch_remote_artifacts(args):
    if not args.model:
        raise ValueError("--model 必填")
    model = _resolve_model(args.model)
    adapt_dir = _resolve_adapt_dir(model)
    config = _prepare_business_config(model, adapt_dir, ensure_local_dataset=False)
    remote = _remote_runtime(
        config,
        ssh_host=args.ssh_host,
        remote_project_root=args.remote_project_root,
        adapt_dir=adapt_dir,
    )
    fetched = _fetch_remote_artifacts(adapt_dir, remote)
    for path in fetched:
        print(f"[business][fetch] {path}")


def cmd_run_remote_cuda(args):
    if not args.model:
        raise ValueError("--model 必填")
    if args.reuse_run_id and args.fresh_run_id:
        raise ValueError("--reuse-run-id 与 --fresh-run-id 不能同时使用")
    model = _resolve_model(args.model)
    adapt_dir = _resolve_adapt_dir(model)
    config = _prepare_business_config(model, adapt_dir, ensure_local_dataset=False, refresh_run_id=bool(args.fresh_run_id))
    run_id = str(config.get("benchmark_run_id") or "").strip()
    if not run_id or run_id.startswith("legacy-"):
        raise ValueError("当前 adaptation 缺少可执行的 benchmark_run_id，请先 run-npu")
    _ensure_local_npu_artifacts(adapt_dir, run_id)
    _run_shell_command(
        f'python "business_benchmark/scripts/check_business_benchmark_run.py" --adapt {shlex.quote(adapt_dir.name)} --wait-cuda-npu-only',
        PROJECT_ROOT,
    )
    print(f"[check] ✅ {adapt_dir} (wait_cuda NPU gate)")
    _ensure_business_eval_script(adapt_dir)
    _ensure_business_model_eval_script(adapt_dir)
    _ensure_business_run_script(adapt_dir, config)
    remote = _remote_runtime(
        config,
        ssh_host=args.ssh_host,
        remote_project_root=args.remote_project_root,
        adapt_dir=adapt_dir,
    )
    print(f"[business][remote] {model['model_id']} @ {adapt_dir}")
    print(f"[business][remote] ssh_host={remote['ssh_host']} remote_project_root={remote['remote_project_root']} run_id={run_id}")
    _run_remote_cuda_and_fetch(adapt_dir, remote, gpu_id=args.gpu_id)
    if args.summarize:
        _write_summary(adapt_dir, comparison_scope=args.comparison_scope, remote_mode=args.remote_mode)
    if args.check:
        _run_shell_command(f'python "business_benchmark/scripts/check_business_benchmark_run.py" --adapt {shlex.quote(adapt_dir.name)}', PROJECT_ROOT)
        print(f"[check] ✅ {adapt_dir}")


def cmd_generate_script(args):
    if args.model:
        models = [m for m in get_models(require_optimization_completed=True) if m["model_id"] == args.model]
    else:
        models = get_models(require_optimization_completed=True)
    if not models:
        print("没有可生成脚本的模型")
        return
    for model in models:
        adapt_dir = _resolve_adapt_dir(model)
        config = _prepare_business_config(model, adapt_dir, ensure_local_dataset=False)
        eval_script_path = _ensure_business_eval_script(adapt_dir)
        model_eval_path = _ensure_business_model_eval_script(adapt_dir)
        script_path = _ensure_business_run_script(adapt_dir, config)
        print(f"[business] wrote {eval_script_path}")
        print(f"[business] wrote {model_eval_path}")
        print(f"[business] wrote {script_path}")


def cmd_summarize(args):
    if args.model and args.adaptation:
        raise ValueError("--model 与 --adaptation 不能同时使用")
    if args.adaptation:
        adapt_name = str(args.adaptation).strip().strip("/")
        if adapt_name.startswith("adaptations/"):
            adapt_name = adapt_name.replace("adaptations/", "", 1)
        adapt_dir = ADAPTATIONS_DIR / adapt_name
        if not adapt_dir.is_dir():
            raise FileNotFoundError(f"未找到 adaptation 目录: {adapt_dir}")
        _write_summary(adapt_dir, comparison_scope=args.comparison_scope, remote_mode=args.remote_mode)
        return
    if args.model:
        models = [m for m in get_models(require_optimization_completed=True) if m["model_id"] == args.model]
    else:
        models = get_models(require_optimization_completed=True)
    if not models:
        print("没有可汇总的模型")
        return
    for model in models:
        adapt_dir = _resolve_adapt_dir(model)
        _write_summary(adapt_dir, comparison_scope=args.comparison_scope, remote_mode=args.remote_mode)


def cmd_artifacts(args):
    if args.model_id:
        models = [m for m in get_models(require_optimization_completed=True) if m["model_id"] == args.model_id]
    else:
        models = get_models()
    for model in models:
        adapt_dir = _resolve_adapt_dir(model)
        print(f"\n[{model['model_id']}]")
        if not adapt_dir.is_dir():
            print("  目录不存在")
            continue
        found = False
        for pattern in BUSINESS_PATTERNS:
            for file in sorted(adapt_dir.glob(pattern)):
                found = True
                print(f"  {file.name}")
        if not found:
            print("  无业务测评产物")


def cmd_pack(args):
    if args.model:
        models = [m for m in get_models() if m["model_id"] == args.model]
    else:
        models = get_models(status="completed")
    output = Path(args.output) if args.output else PROJECT_ROOT / f"business_benchmark_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    files_to_pack: list[tuple[Path, str]] = []
    for model in models:
        adapt_dir = _resolve_adapt_dir(model)
        if not adapt_dir.is_dir():
            continue
        for pattern in BUSINESS_PATTERNS:
            for file in adapt_dir.glob(pattern):
                files_to_pack.append((file, f"{adapt_dir.name}/{file.name}"))
    if not files_to_pack:
        print("无业务测评产出可打包")
        return
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps({"created_at": datetime.now().isoformat(), "files_count": len(files_to_pack)}, indent=2))
        for file, arcname in files_to_pack:
            zf.write(file, arcname)
    print(f"[business] wrote {output}")


def cmd_unpack(args):
    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f"文件不存在: {inp}")
    with zipfile.ZipFile(inp, "r") as zf:
        zf.extractall(ADAPTATIONS_DIR)
    print(f"[business] unpacked to {ADAPTATIONS_DIR}")


def cmd_clean(args):
    if args.model:
        models = [m for m in get_models() if m["model_id"] == args.model]
    else:
        models = get_models()
    removed = 0
    for model in models:
        adapt_dir = _resolve_adapt_dir(model)
        if not adapt_dir.is_dir():
            continue
        for pattern in BUSINESS_PATTERNS:
            for file in adapt_dir.glob(pattern):
                if file.name in {"business_benchmark_config.json", BUSINESS_RUN_FILENAME, BUSINESS_MODEL_EVAL_FILENAME}:
                    continue
                if file.name == BUSINESS_EVAL_FILENAME:
                    continue
                file.unlink()
                removed += 1
    print(f"[business] removed {removed} files")


def main() -> int:
    parser = argparse.ArgumentParser(description="Business Benchmark Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="列出模型")
    list_parser.add_argument("--status", help="过滤 business_benchmark_status")
    list_parser.set_defaults(func=cmd_list)

    run_npu_parser = subparsers.add_parser("run-npu", help="读取配置并执行本机 NPU baseline/perf 业务测评")
    run_npu_parser.add_argument("--model", help="指定模型")
    run_npu_parser.set_defaults(func=cmd_run_npu)

    remote_parser = subparsers.add_parser("print-remote-command", help="打印远端 CUDA baseline 执行命令模板")
    remote_parser.add_argument("--model", required=True, help="指定模型")
    remote_parser.add_argument("--ssh-host", default=DEFAULT_SSH_HOST_ALIAS, help=f"SSH host alias，默认 {DEFAULT_SSH_HOST_ALIAS}")
    remote_parser.add_argument("--remote-project-root", default=DEFAULT_REMOTE_PROJECT_ROOT, help=f"远端仓库根目录，默认 {DEFAULT_REMOTE_PROJECT_ROOT}")
    remote_parser.add_argument("--gpu-id", default="0", help="远端 CUDA_VISIBLE_DEVICES，默认 0")
    remote_parser.add_argument("--reuse-run-id", action="store_true", help="兼容旧用法：显式声明复用已有 benchmark_run_id；当前已是默认行为")
    remote_parser.add_argument("--fresh-run-id", action="store_true", help="强制生成新的 benchmark_run_id；仅在你明确要开启全新一轮业务测评时使用")
    remote_parser.set_defaults(func=cmd_print_remote_command)

    sync_parser = subparsers.add_parser("sync-remote-workspace", help="通过 SSH 直传同步第四阶段远端工作目录")
    sync_parser.add_argument("--model", required=True, help="指定模型")
    sync_parser.add_argument("--ssh-host", default=DEFAULT_SSH_HOST_ALIAS, help=f"SSH host alias，默认 {DEFAULT_SSH_HOST_ALIAS}")
    sync_parser.add_argument("--remote-project-root", default=DEFAULT_REMOTE_PROJECT_ROOT, help=f"远端仓库根目录，默认 {DEFAULT_REMOTE_PROJECT_ROOT}")
    sync_parser.set_defaults(func=cmd_sync_remote_workspace)

    fetch_parser = subparsers.add_parser("fetch-remote-artifacts", help="通过 SSH 拉回远端 CUDA baseline 工件")
    fetch_parser.add_argument("--model", required=True, help="指定模型")
    fetch_parser.add_argument("--ssh-host", default=DEFAULT_SSH_HOST_ALIAS, help=f"SSH host alias，默认 {DEFAULT_SSH_HOST_ALIAS}")
    fetch_parser.add_argument("--remote-project-root", default=DEFAULT_REMOTE_PROJECT_ROOT, help=f"远端仓库根目录，默认 {DEFAULT_REMOTE_PROJECT_ROOT}")
    fetch_parser.set_defaults(func=cmd_fetch_remote_artifacts)

    remote_run_parser = subparsers.add_parser("run-remote-cuda", help="通过 SSH 自动执行远端 CUDA baseline、回收工件，并可选自动汇总/校验")
    remote_run_parser.add_argument("--model", required=True, help="指定模型")
    remote_run_parser.add_argument("--ssh-host", default=DEFAULT_SSH_HOST_ALIAS, help=f"SSH host alias，默认 {DEFAULT_SSH_HOST_ALIAS}")
    remote_run_parser.add_argument("--remote-project-root", default=DEFAULT_REMOTE_PROJECT_ROOT, help=f"远端仓库根目录，默认 {DEFAULT_REMOTE_PROJECT_ROOT}")
    remote_run_parser.add_argument("--gpu-id", default="0", help="远端 CUDA_VISIBLE_DEVICES，默认 0")
    remote_run_parser.add_argument("--reuse-run-id", action="store_true", help="兼容旧用法：显式声明复用已有 benchmark_run_id；当前已是默认行为")
    remote_run_parser.add_argument("--fresh-run-id", action="store_true", help="强制生成新的 benchmark_run_id；仅在你明确要开启全新一轮业务测评时使用")
    remote_run_parser.add_argument("--comparison-scope", default="real_business")
    remote_run_parser.add_argument("--remote-mode", default=DEFAULT_REMOTE_EXECUTION_MODE)
    remote_run_parser.add_argument("--summarize", action=argparse.BooleanOptionalAction, default=True, help="远端工件回收后是否自动生成 business_summary.json（默认开启）")
    remote_run_parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True, help="自动 summarize 后是否继续跑 completed gate（默认开启）")
    remote_run_parser.set_defaults(func=cmd_run_remote_cuda)

    generate_parser = subparsers.add_parser("generate-script", help="生成业务测评统一入口 business_run.py")
    generate_parser.add_argument("--model", help="指定模型")
    generate_parser.set_defaults(func=cmd_generate_script)

    artifacts_parser = subparsers.add_parser("artifacts", help="列出业务测评产出")
    artifacts_parser.add_argument("model_id", nargs="?")
    artifacts_parser.set_defaults(func=cmd_artifacts)

    summarize_parser = subparsers.add_parser("summarize", help="从现有业务测评工件生成 business_summary.json")
    summarize_parser.add_argument("--model", help="指定模型")
    summarize_parser.add_argument("--adaptation", help="直接指定 adaptation 目录名；不依赖 board.db")
    summarize_parser.add_argument("--comparison-scope", default="real_business")
    summarize_parser.add_argument("--remote-mode", default=DEFAULT_REMOTE_EXECUTION_MODE)
    summarize_parser.set_defaults(func=cmd_summarize)

    pack_parser = subparsers.add_parser("pack", help="打包业务测评产出")
    pack_parser.add_argument("--output", help="输出 zip 路径")
    pack_parser.add_argument("--model", help="指定模型")
    pack_parser.set_defaults(func=cmd_pack)

    unpack_parser = subparsers.add_parser("unpack", help="解包业务测评产出")
    unpack_parser.add_argument("--input", required=True)
    unpack_parser.set_defaults(func=cmd_unpack)

    clean_parser = subparsers.add_parser("clean", help="清理业务测评产出")
    clean_parser.add_argument("--model", help="指定模型")
    clean_parser.set_defaults(func=cmd_clean)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
