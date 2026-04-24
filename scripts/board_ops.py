from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypedDict, cast

# 相对导入便于 IDE/类型检查器解析；直接运行脚本时无父包，回退到 path + 绝对导入
_scripts_dir = Path(__file__).resolve().parent
try:
    from .adaptation_utils import model_id_to_adaptation_path  # type: ignore[import-not-found]
except ImportError:
    if str(_scripts_dir) not in sys.path:
        sys.path.insert(0, str(_scripts_dir))
    import importlib

    model_id_to_adaptation_path = importlib.import_module("adaptation_utils").model_id_to_adaptation_path

# 解析 board.db 路径：优先 PROJECT_ROOT，否则用脚本所在目录的上级（项目根）
_PROJECT_ROOT: str = os.environ.get("PROJECT_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_PROJECT_ROOT, "board.db")
_MIN_COMPLETED_METRIC_SAMPLES = 50
_MIN_COMPLETED_BUSINESS_SAMPLE_LOWER_BOUND = 50
_REQUIRED_COMPLETED_BENCHMARK_FIELDS = ("latency_s", "num_samples", "mode", "dataset", "dtype", "output_type", "device", "start_time", "end_time")
_REQUIRED_COMPLETED_OPTIMIZATION_FIELDS = ("latency_s", "num_samples", "mode", "dataset", "dtype", "output_type", "device", "start_time", "end_time")
_VALID_COMPARISON_SCOPES = {"cold_start", "steady_state", "mixed"}
_VALID_BUSINESS_COMPARISON_SCOPES = {"real_business", "cold_start", "steady_state", "mixed"}
_VALID_OPTIMIZATION_COMPLETION_KINDS = {"fusion", "runtime_only", "hybrid"}
_MIN_SANE_BUSINESS_NPU_SPEEDUP_RATIO = 0.9
_MIN_SANE_BUSINESS_VS_CUDA_LATENCY_RATIO = 0.22
_MAX_COMPLETED_BOUNDED_BUSINESS_QUALITY_DELTA = 0.005
_BUSINESS_BOUNDED_QUALITY_METRIC_KEYS = {
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
}
_BUSINESS_STRICT_ALIGNMENT_QUALITY_KEYS = {
    "exact_match",
    "accuracy",
    "top1_accuracy",
    "match_rate",
    "text_match_rate",
}
_OPTIMIZATION_REASON_REQUIRED_FIELDS = ("reason_code", "retryable", "recommended_action", "evidence", "next_step")
_OPTIMIZATION_PENDING_REASON_CODES = {
    "sample_insufficient",
    "dataset_reselection_required",
    "version_incompatible",
    "dependency_missing",
    "oom_retryable",
    "multi_card_required",
    "timeout_retryable",
    "artifact_inconsistent",
    "artifact_missing",
    "evidence_chain_incomplete",
    "repeated_assignment_no_output",
    "empty_reason",
    "benchmark_blocked",
    "measurement_inconsistent",
    "measurement_bug",
    "precision_regression_requires_runtime_only",
    "runtime_only_not_attempted",
    "custom_retryable",
}
_OPTIMIZATION_SKIPPED_REASON_CODES = {
    "true_no_gain_after_runtime_only",
    "fusion_regression_runtime_only_no_gain",
    "precision_regression_runtime_only_no_gain",
}
_OPTIMIZATION_NOT_APPLICABLE_REASON_CODES = {
    "architecture_not_applicable_after_runtime_only",
    "runtime_unsupported_irrecoverable",
    "model_format_irrecoverable",
}
_TEAM_LEAD_AGENT_ID = "team-lead"
_TEAM_LEAD_HOST_IP_ENV_KEYS = ("TEAM_LEAD_HOST_IP", "LOCAL_IP", "HOST_IP")


class RollbackStageConfig(TypedDict):
    status_col: str
    owner_col: str
    updated_col: str
    notes_col: str
    valid_targets: set[str]


class StatusUpdateStageConfig(TypedDict):
    label: str
    status_col: str
    owner_col: str
    started_col: str
    updated_col: str
    notes_col: str
    valid_statuses: list[str]
    release_statuses: set[str]
    commit_message: str


class AssignStageConfig(TypedDict):
    status_col: str
    owner_col: str
    started_col: str
    updated_col: str
    task_label: str
    task_prefix: str
    select_where_sql: str
    order_by_col: str


_MODELS_SCHEMA_COLUMNS: list[tuple[str, str]] = [
    ("model_id", "TEXT PRIMARY KEY"),
    ("source", "TEXT"),
    ("priority", "TEXT"),
    ("url", "TEXT"),
    ("description", "TEXT"),
    ("adaptation_status", "TEXT"),
    ("adaptation_started_at", "TEXT"),
    ("adaptation_last_updated", "TEXT"),
    ("adaptation_failure_reason", "TEXT"),
    ("adaptation_notes", "TEXT"),
    ("adaptation_path", "TEXT"),
    ("adaptation_owner", "TEXT"),
    ("benchmark_status", "TEXT DEFAULT ''"),
    ("benchmark_started_at", "TEXT DEFAULT ''"),
    ("benchmark_last_updated", "TEXT DEFAULT ''"),
    ("benchmark_owner", "TEXT DEFAULT ''"),
    ("benchmark_notes", "TEXT DEFAULT ''"),
    ("optimization_status", "TEXT DEFAULT ''"),
    ("optimization_started_at", "TEXT DEFAULT ''"),
    ("optimization_last_updated", "TEXT DEFAULT ''"),
    ("optimization_owner", "TEXT DEFAULT ''"),
    ("optimization_notes", "TEXT DEFAULT ''"),
    ("business_benchmark_status", "TEXT DEFAULT ''"),
    ("business_benchmark_started_at", "TEXT DEFAULT ''"),
    ("business_benchmark_last_updated", "TEXT DEFAULT ''"),
    ("business_benchmark_owner", "TEXT DEFAULT ''"),
    ("business_benchmark_notes", "TEXT DEFAULT ''"),
    ("human_review_status", "TEXT DEFAULT ''"),
]


def _create_models_table(cursor):
    models_cols_sql = ",\n        ".join(f"{name} {type_sql}" for name, type_sql in _MODELS_SCHEMA_COLUMNS)
    cursor.execute(
        f"""
    CREATE TABLE IF NOT EXISTS models (
        {models_cols_sql}
    );
    """
    )


def _get_model_columns(cursor) -> list[str]:
    cursor.execute("PRAGMA table_info(models)")
    return [row[1] for row in cursor.fetchall()]


def _ensure_models_table_schema(cursor):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='models'")
    if not cursor.fetchone():
        _create_models_table(cursor)
        return

    existing_columns = _get_model_columns(cursor)
    desired_columns = [name for name, _ in _MODELS_SCHEMA_COLUMNS]
    if existing_columns == desired_columns:
        _backfill_human_review_pending(cursor)
        return

    existing_set = set(existing_columns)

    def _coalesce_from_candidates(*candidates: str) -> str:
        available = [candidate for candidate in candidates if candidate in existing_set]
        if not available:
            return "''"
        if len(available) == 1:
            return available[0]
        return f"COALESCE({', '.join(available)}, '')"

    business_status_expr = _coalesce_from_candidates("business_benchmark_status")
    human_review_expr = _coalesce_from_candidates("human_review_status")
    if human_review_expr == "''":
        human_review_expr = f"CASE WHEN LOWER(TRIM({business_status_expr})) = 'completed' THEN 'pending' ELSE '' END"

    select_expr_by_new_col = {
        "model_id": _coalesce_from_candidates("model_id"),
        "source": _coalesce_from_candidates("source"),
        "priority": _coalesce_from_candidates("priority"),
        "adaptation_status": _coalesce_from_candidates("adaptation_status", "status"),
        "adaptation_started_at": _coalesce_from_candidates("adaptation_started_at", "started_at"),
        "adaptation_last_updated": _coalesce_from_candidates("adaptation_last_updated", "last_updated"),
        "adaptation_failure_reason": _coalesce_from_candidates("adaptation_failure_reason", "failure_reason"),
        "url": _coalesce_from_candidates("url"),
        "description": _coalesce_from_candidates("description"),
        "adaptation_notes": _coalesce_from_candidates("adaptation_notes", "notes"),
        "adaptation_path": _coalesce_from_candidates("adaptation_path"),
        "adaptation_owner": _coalesce_from_candidates("adaptation_owner", "owner"),
        "benchmark_status": _coalesce_from_candidates("benchmark_status"),
        "benchmark_started_at": _coalesce_from_candidates("benchmark_started_at"),
        "benchmark_last_updated": _coalesce_from_candidates("benchmark_last_updated"),
        "benchmark_owner": _coalesce_from_candidates("benchmark_owner"),
        "benchmark_notes": _coalesce_from_candidates("benchmark_notes"),
        "optimization_status": _coalesce_from_candidates("optimization_status"),
        "optimization_started_at": _coalesce_from_candidates("optimization_started_at"),
        "optimization_last_updated": _coalesce_from_candidates("optimization_last_updated"),
        "optimization_owner": _coalesce_from_candidates("optimization_owner"),
        "optimization_notes": _coalesce_from_candidates("optimization_notes"),
        "business_benchmark_status": business_status_expr,
        "business_benchmark_started_at": _coalesce_from_candidates("business_benchmark_started_at"),
        "business_benchmark_last_updated": _coalesce_from_candidates("business_benchmark_last_updated"),
        "business_benchmark_owner": _coalesce_from_candidates("business_benchmark_owner"),
        "business_benchmark_notes": _coalesce_from_candidates("business_benchmark_notes"),
        "human_review_status": human_review_expr,
    }

    temp_table = "models_legacy_schema_migration"
    cursor.execute(f"ALTER TABLE models RENAME TO {temp_table}")
    _create_models_table(cursor)
    insert_cols = ", ".join(desired_columns)
    select_cols = ", ".join(f"{select_expr_by_new_col[col]} AS {col}" for col in desired_columns)
    cursor.execute(f"INSERT INTO models ({insert_cols}) SELECT {select_cols} FROM {temp_table}")
    cursor.execute(f"DROP TABLE {temp_table}")
    _backfill_human_review_pending(cursor)


def _backfill_human_review_pending(cursor):
    cursor.execute(
        """
        UPDATE models
        SET human_review_status = 'pending'
        WHERE LOWER(TRIM(COALESCE(business_benchmark_status, ''))) = 'completed'
          AND TRIM(COALESCE(human_review_status, '')) = ''
        """
    )


def _clear_human_review_status(cursor, model_id: str):
    cursor.execute("UPDATE models SET human_review_status = '' WHERE model_id = ?", (model_id,))


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    _ensure_models_table_schema(cursor)
    _ensure_agents_table(cursor)
    conn.commit()
    return conn


def _priority_rank_sql(column: str = "priority") -> str:
    """Return a stable SQL expression for mixed numeric/text priorities."""
    col = column.strip()
    return f"""
    CASE
        WHEN TRIM({col}) GLOB '[0-9]*' THEN CAST(TRIM({col}) AS INTEGER)
        WHEN LOWER(TRIM({col})) = 'critical' THEN 300
        WHEN LOWER(TRIM({col})) = 'high' THEN 200
        WHEN LOWER(TRIM({col})) = 'medium' THEN 100
        WHEN LOWER(TRIM({col})) = 'low' THEN 0
        ELSE 0
    END
    """


def _append_status_note_sql(column: str) -> str:
    """Append a status note once while preserving existing content."""
    return f"""
    CASE
        WHEN COALESCE({column}, '') = '' THEN ?
        WHEN instr({column}, ?) > 0 THEN {column}
        ELSE {column} || ' | ' || ?
    END
    """


def git_commit_and_push(message, max_retries=3, retry_delay=5, paths: list[str] | None = None):
    """
    执行 git add, commit 和 push，带重试机制以应对网络问题。

    Args:
        message: commit 消息
        max_retries: 最大重试次数（默认 3 次）
        retry_delay: 重试延迟（秒），使用指数退避
    """
    # git add 和 commit 通常不需要重试（本地操作）
    # 必须在项目根目录下执行 git 命令
    try:
        add_targets = paths or ["."]
        subprocess.run(["git", "add", *add_targets], check=True, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    except subprocess.CalledProcessError as e:
        print(f"Git add failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

    # 检查是否有变更需要提交
    result = subprocess.run(["git", "diff", "--cached", "--quiet"], capture_output=True, cwd=_PROJECT_ROOT)
    if result.returncode == 0:
        print("No changes to commit.")
        return True

    try:
        subprocess.run(["git", "commit", "-m", message], check=True, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    except subprocess.CalledProcessError as e:
        print(f"Git commit failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

    # git push 需要重试机制
    # for attempt in range(max_retries):
    #     try:
    #         result = subprocess.run(
    #             ["git", "push", "origin", "main"],
    #             check=True,
    #             capture_output=True,
    #             text=True,
    #             timeout=10,  # 10 秒超时
    #             cwd=_PROJECT_ROOT,
    #         )
    #         print(f"Git commit and push successful: {message}")
    #         return True
    #     except subprocess.TimeoutExpired:
    #         print(f"Git push timeout (attempt {attempt + 1}/{max_retries})")
    #         if attempt < max_retries - 1:
    #             delay = retry_delay * (2**attempt)  # 指数退避
    #             print(f"Retrying in {delay} seconds...")
    #             time.sleep(delay)
    #     except subprocess.CalledProcessError as e:
    #         print(f"Git push failed (attempt {attempt + 1}/{max_retries}): {e}")
    #         if e.stdout:
    #             print(f"stdout: {e.stdout}")
    #         if e.stderr:
    #             print(f"stderr: {e.stderr}")

    #         if attempt < max_retries - 1:
    #             delay = retry_delay * (2**attempt)  # 指数退避：5s, 10s, 20s
    #             print(f"Retrying in {delay} seconds...")
    #             time.sleep(delay)
    #         else:
    #             print(f"Git push failed after {max_retries} attempts. Please retry manually.")
    #             return False

    return False


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Enable WAL mode
    cursor.execute("PRAGMA journal_mode=WAL;")

    _ensure_models_table_schema(cursor)
    _ensure_agents_table(cursor)

    # 非空 url 唯一：允许多个空 url，但非空 url 不可重复
    cursor.execute("""
    CREATE UNIQUE INDEX IF NOT EXISTS idx_models_url_unique ON models(url) WHERE url != ''
    """)

    conn.commit()
    conn.close()
    print("Database initialized.")


def _ensure_agents_table(cursor):
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS agents (
        id TEXT PRIMARY KEY,
        last_heartbeat TEXT,
        status TEXT,
        current_task TEXT
    );
    """)


def reset_board():
    conn = get_db_connection()
    cursor = conn.cursor()
    # reset 常被直接用于旧库，先自愈 schema 再做批量更新，避免新增阶段列缺失时报错
    _ensure_models_table_schema(cursor)
    _ensure_agents_table(cursor)
    cursor.execute("DELETE FROM agents")
    # 全表清空 adaptation_owner（与历史 shell reset 对齐）；再仅对 in_progress 行回退 adaptation_status
    cursor.execute("UPDATE models SET adaptation_owner = ''")
    cursor.execute("UPDATE models SET adaptation_status = 'pending' WHERE adaptation_status = 'in_progress'")
    cursor.execute("UPDATE models SET benchmark_owner = '', benchmark_status = 'pending' WHERE benchmark_status = 'in_progress'")
    cursor.execute("UPDATE models SET optimization_owner = '', optimization_status = 'pending' WHERE optimization_status = 'in_progress'")
    cursor.execute("UPDATE models SET business_benchmark_owner = '', business_benchmark_status = 'pending' WHERE business_benchmark_status IN ('in_progress', 'wait_cuda')")
    conn.commit()
    conn.close()
    print("Board reset: Agents cleared; models.adaptation_owner cleared for all rows; adaptation/benchmark/optimization in_progress reset to pending; business_benchmark in_progress/wait_cuda reset to pending.")


def clear_agents():
    conn = get_db_connection()
    cursor = conn.cursor()
    _ensure_agents_table(cursor)
    cursor.execute("SELECT COUNT(*) AS count FROM agents")
    row = cursor.fetchone()
    existing_count = int(row["count"]) if row and row["count"] is not None else 0
    cursor.execute("DELETE FROM agents")
    conn.commit()
    conn.close()
    print(f"Cleared agents table: removed {existing_count} rows.")


_ROLLBACK_STAGE_CONFIG: dict[str, RollbackStageConfig] = {
    "adaptation": {
        "status_col": "adaptation_status",
        "owner_col": "adaptation_owner",
        "updated_col": "adaptation_last_updated",
        "notes_col": "adaptation_notes",
        "valid_targets": {"pending", "skipped", "not_applicable", "needs_authorization"},
    },
    "benchmark": {
        "status_col": "benchmark_status",
        "owner_col": "benchmark_owner",
        "updated_col": "benchmark_last_updated",
        "notes_col": "benchmark_notes",
        "valid_targets": {"pending", "skipped", "not_applicable"},
    },
    "optimization": {
        "status_col": "optimization_status",
        "owner_col": "optimization_owner",
        "updated_col": "optimization_last_updated",
        "notes_col": "optimization_notes",
        "valid_targets": {"pending", "skipped", "not_applicable"},
    },
    "business_benchmark": {
        "status_col": "business_benchmark_status",
        "owner_col": "business_benchmark_owner",
        "updated_col": "business_benchmark_last_updated",
        "notes_col": "business_benchmark_notes",
        "valid_targets": {"pending", "skipped", "not_applicable"},
    },
}
_ROLLBACK_STAGE_ORDER = ("adaptation", "benchmark", "optimization", "business_benchmark")
_ROLLBACK_TARGET_CHOICES = tuple(f"{stage}:{status}" for stage in _ROLLBACK_STAGE_ORDER for status in sorted(_ROLLBACK_STAGE_CONFIG[stage]["valid_targets"]))
_DOWNSTREAM_RESETTABLE_STATUSES = {"completed", "in_progress", "wait_cuda"}
_STATUS_UPDATE_STAGE_CONFIG: dict[str, StatusUpdateStageConfig] = {
    "benchmark": {
        "label": "benchmark_status",
        "status_col": "benchmark_status",
        "owner_col": "benchmark_owner",
        "started_col": "benchmark_started_at",
        "updated_col": "benchmark_last_updated",
        "notes_col": "benchmark_notes",
        "valid_statuses": ["completed", "skipped", "not_applicable", "in_progress", "pending"],
        "release_statuses": {"completed", "skipped", "not_applicable", "pending"},
        "commit_message": "feat: complete benchmark for {model_id}",
    },
    "optimization": {
        "label": "optimization_status",
        "status_col": "optimization_status",
        "owner_col": "optimization_owner",
        "started_col": "optimization_started_at",
        "updated_col": "optimization_last_updated",
        "notes_col": "optimization_notes",
        "valid_statuses": ["completed", "skipped", "not_applicable", "in_progress", "pending"],
        "release_statuses": {"completed", "skipped", "not_applicable", "pending"},
        "commit_message": "feat: complete NPU optimization for {model_id}",
    },
    "business_benchmark": {
        "label": "business_benchmark_status",
        "status_col": "business_benchmark_status",
        "owner_col": "business_benchmark_owner",
        "started_col": "business_benchmark_started_at",
        "updated_col": "business_benchmark_last_updated",
        "notes_col": "business_benchmark_notes",
        "valid_statuses": ["completed", "skipped", "not_applicable", "in_progress", "wait_cuda", "pending"],
        "release_statuses": {"completed", "skipped", "not_applicable", "wait_cuda", "pending"},
        "commit_message": "feat: complete business benchmark for {model_id}",
    },
}
_ASSIGN_STAGE_CONFIG: dict[str, AssignStageConfig] = {
    "adaptation": {
        "status_col": "adaptation_status",
        "owner_col": "adaptation_owner",
        "started_col": "adaptation_started_at",
        "updated_col": "adaptation_last_updated",
        "task_label": "",
        "task_prefix": "",
        "select_where_sql": "adaptation_status = 'pending'",
        "order_by_col": "adaptation_started_at",
    },
    "benchmark": {
        "status_col": "benchmark_status",
        "owner_col": "benchmark_owner",
        "started_col": "benchmark_started_at",
        "updated_col": "benchmark_last_updated",
        "task_label": "benchmark",
        "task_prefix": "benchmark",
        "select_where_sql": "adaptation_status = 'completed' AND (benchmark_status = 'pending' OR benchmark_status = '')",
        "order_by_col": "adaptation_last_updated",
    },
    "optimization": {
        "status_col": "optimization_status",
        "owner_col": "optimization_owner",
        "started_col": "optimization_started_at",
        "updated_col": "optimization_last_updated",
        "task_label": "optimization",
        "task_prefix": "optimization",
        "select_where_sql": "adaptation_status = 'completed' AND benchmark_status = 'completed' AND (optimization_status = 'pending' OR optimization_status = '')",
        "order_by_col": "benchmark_last_updated",
    },
    "business_benchmark": {
        "status_col": "business_benchmark_status",
        "owner_col": "business_benchmark_owner",
        "started_col": "business_benchmark_started_at",
        "updated_col": "business_benchmark_last_updated",
        "task_label": "business benchmark",
        "task_prefix": "business-benchmark",
        "select_where_sql": "adaptation_status = 'completed' AND benchmark_status = 'completed' AND optimization_status = 'completed' AND (business_benchmark_status = 'pending' OR business_benchmark_status = '')",
        "order_by_col": "optimization_last_updated",
    },
}


def _ensure_valid_stage_status(stage_cfg: StatusUpdateStageConfig, new_status: str):
    if new_status not in stage_cfg["valid_statuses"]:
        print(f"Error: Invalid {stage_cfg['label']} '{new_status}'.")
        sys.exit(1)


def _get_model_stage_status(model_id: str, status_col: str) -> str:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT {status_col} FROM models WHERE model_id = ?", (model_id,))
    row = cursor.fetchone()
    conn.close()
    return (row[status_col] or "").strip().lower() if row else ""


def _ensure_stage_not_completed(stage_cfg: StatusUpdateStageConfig, current_status: str, *, allow_completed_rewrite: bool = False):
    if current_status == "completed" and not allow_completed_rewrite:
        print(f"Error: {stage_cfg['label']} is already completed and cannot be changed.")
        sys.exit(1)


def _load_owner_and_prev_status(cursor, model_id: str, stage_cfg: StatusUpdateStageConfig, new_status: str) -> tuple[str | None, str | None]:
    owner_to_idle = None
    prev_status = None
    if new_status in stage_cfg["release_statuses"]:
        cursor.execute(f"SELECT {stage_cfg['owner_col']}, {stage_cfg['status_col']} FROM models WHERE model_id = ?", (model_id,))
        row = cursor.fetchone()
        if row:
            prev_status = row[stage_cfg["status_col"]] or ""
            if row[stage_cfg["owner_col"]]:
                owner_to_idle = row[stage_cfg["owner_col"]]
    return owner_to_idle, prev_status


def _update_stage_status_row(cursor, model_id: str, stage_cfg: StatusUpdateStageConfig, new_status: str, notes: str, now: str):
    release_owner = new_status in stage_cfg["release_statuses"]
    query = f"UPDATE models SET {stage_cfg['status_col']} = ?, {stage_cfg['updated_col']} = ?, {stage_cfg['notes_col']} = ?"
    params = [new_status, now, notes]
    if new_status == "in_progress":
        query += f", {stage_cfg['started_col']} = ?"
        params.append(now)
    if release_owner:
        query += f", {stage_cfg['owner_col']} = ?"
        params.append("")
    query += " WHERE model_id = ?"
    params.append(model_id)
    cursor.execute(query, params)


def _set_agent_idle(cursor, owner_to_idle: str | None):
    if owner_to_idle:
        cursor.execute(
            "UPDATE agents SET status = 'idle', current_task = '' WHERE id = ?",
            (owner_to_idle,),
        )
        print(f"Set agent {owner_to_idle} status to idle, current_task cleared.")


def _initialize_downstream_stage(cursor, model_id: str, status_col: str):
    cursor.execute(f"UPDATE models SET {status_col} = 'pending' WHERE model_id = ? AND ({status_col} = '' OR {status_col} IS NULL)", (model_id,))


def _load_python_module_from_path(module_name: str, module_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec for {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _detect_local_ip() -> str:
    for env_key in _TEAM_LEAD_HOST_IP_ENV_KEYS:
        env_val = (os.environ.get(env_key) or "").strip()
        if env_val:
            return env_val

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0].strip()
            if ip and not ip.startswith("127."):
                return ip
    except OSError:
        pass

    try:
        hostname = socket.gethostname()
        for addr_info in socket.getaddrinfo(hostname, None, family=socket.AF_INET):
            ip = str(addr_info[4][0]).strip()
            if ip and not ip.startswith("127."):
                return ip
    except OSError:
        pass

    return ""


def _format_agent_current_task(agent_id: str, current_task: str) -> str:
    task_text = (current_task or "").strip()
    if agent_id != _TEAM_LEAD_AGENT_ID:
        return task_text

    local_ip = _detect_local_ip() or "unknown"
    pid = os.getpid()
    task_text = re.sub(r"(?:\s+\|\s+(?:host_ip|pid)=[^|]+)+$", "", task_text).strip()
    if task_text:
        return f"{task_text} | host_ip={local_ip} | pid={pid}"
    return f"host_ip={local_ip} | pid={pid}"


def _assign_stage_task(
    agent_id: str,
    stage_cfg: AssignStageConfig,
    *,
    select_cols_sql: str = "model_id, adaptation_path",
    extra_assignments_factory: Callable[[str], list[tuple[str, Any]]] | None = None,
):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        f"SELECT model_id FROM models WHERE {stage_cfg['owner_col']} = ? AND {stage_cfg['status_col']} = 'in_progress'",
        (agent_id,),
    )
    existing_task = cursor.fetchone()
    if existing_task:
        if stage_cfg["task_label"]:
            print(f"Agent {agent_id} is already assigned to {stage_cfg['task_label']} {existing_task['model_id']}")
        else:
            print(f"Agent {agent_id} is already assigned to {existing_task['model_id']}")
        conn.close()
        return

    cursor.execute(
        f"""
        SELECT {select_cols_sql} FROM models
        WHERE {stage_cfg["select_where_sql"]}
        ORDER BY {_priority_rank_sql("priority")} DESC, {stage_cfg["order_by_col"]} ASC LIMIT 1
        """
    )
    task = cursor.fetchone()

    if task:
        model_id = task["model_id"]
        extra_assignments = extra_assignments_factory(model_id) if extra_assignments_factory else []
        adaptation_path = ""
        if "adaptation_path" in task.keys():
            adaptation_path = str(task["adaptation_path"] or "")
        if not adaptation_path:
            for col, value in extra_assignments:
                if col == "adaptation_path":
                    adaptation_path = str(value or "")
                    break
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        assignments = [
            (stage_cfg["status_col"], "in_progress"),
            (stage_cfg["owner_col"], agent_id),
            (stage_cfg["started_col"], now),
            (stage_cfg["updated_col"], now),
            *extra_assignments,
        ]
        set_clause = ",\n                ".join(f"{col} = ?" for col, _ in assignments)
        params = [value for _, value in assignments]
        params.append(model_id)
        cursor.execute(
            f"""
            UPDATE models SET
                {set_clause}
            WHERE model_id = ?
            """,
            params,
        )
        cursor.execute(
            """
            INSERT OR REPLACE INTO agents (id, last_heartbeat, status, current_task)
            VALUES (?, ?, 'active', ?)
            """,
            (
                agent_id,
                now,
                _format_agent_current_task(agent_id, f"{stage_cfg['task_prefix']}: {model_id}" if stage_cfg["task_prefix"] else model_id),
            ),
        )
        conn.commit()
        if stage_cfg["task_label"]:
            print(f"Assigned {stage_cfg['task_label']} {model_id} to {agent_id}")
        else:
            print(f"Assigned {model_id} to {agent_id}")
        print(f"adaptation_path={adaptation_path}")
    else:
        if stage_cfg["task_label"]:
            print(f"No pending {stage_cfg['task_label']} tasks found.")
        else:
            print("No pending tasks found.")

    conn.close()


def _parse_model_ids_arg(model_ids_raw: str) -> list[str]:
    tokens: list[str] = []
    for sep in [",", "\n", "\t", ";", "，", "；"]:
        model_ids_raw = model_ids_raw.replace(sep, " ")
    for token in model_ids_raw.split(" "):
        model_id = token.strip()
        if model_id:
            tokens.append(model_id)
    deduped: list[str] = []
    seen: set[str] = set()
    for model_id in tokens:
        if model_id not in seen:
            deduped.append(model_id)
            seen.add(model_id)
    return deduped


def _parse_rollback_target(rollback_to: str) -> tuple[str, str]:
    target = (rollback_to or "").strip().lower()
    if ":" not in target:
        print("Error: rollback_to must be '<stage>:<status>' (e.g. optimization:pending).")
        sys.exit(1)
    stage, status = target.split(":", 1)
    cfg = _ROLLBACK_STAGE_CONFIG.get(stage)
    if not cfg:
        print(f"Error: Unsupported rollback stage '{stage}'.")
        sys.exit(1)
    if status not in cfg["valid_targets"]:
        print(f"Error: Unsupported rollback target '{target}'.")
        sys.exit(1)
    return stage, status


def _reset_later_stage_statuses_for_model(target_stage: str, model_row: sqlite3.Row, note_text: str, *, cursor, now: str) -> set[str]:
    owners_to_idle: set[str] = set()
    start_idx = _ROLLBACK_STAGE_ORDER.index(target_stage) + 1
    model_id = model_row["model_id"]
    for stage in _ROLLBACK_STAGE_ORDER[start_idx:]:
        cfg = _ROLLBACK_STAGE_CONFIG[stage]
        current_status = (model_row[cfg["status_col"]] or "").strip()
        current_owner = (model_row[cfg["owner_col"]] or "").strip()
        # Preserve terminal/non-active statuses such as skipped/not_applicable/needs_authorization.
        if current_status not in _DOWNSTREAM_RESETTABLE_STATUSES:
            continue
        if current_owner:
            owners_to_idle.add(current_owner)
        cursor.execute(
            f"""
            UPDATE models
            SET {cfg["status_col"]} = 'pending',
                {cfg["owner_col"]} = '',
                {cfg["updated_col"]} = ?,
                {cfg["notes_col"]} = ?
            WHERE model_id = ?
            """,
            (now, note_text, model_id),
        )
    current_human_review = (model_row["human_review_status"] or "").strip().lower() if "human_review_status" in model_row.keys() else ""
    if current_human_review in {"pending", "completed"}:
        cursor.execute("UPDATE models SET human_review_status = '' WHERE model_id = ?", (model_id,))
    return owners_to_idle


def rollback_models(model_ids_raw: str, rollback_to: str, notes: str = ""):
    """按模型列表统一回退到指定 stage/status。"""
    model_ids = _parse_model_ids_arg(model_ids_raw or "")
    if not model_ids:
        print("Error: at least one model_id is required.")
        sys.exit(1)

    stage, status = _parse_rollback_target(rollback_to)
    stage_cfg = _ROLLBACK_STAGE_CONFIG[stage]
    default_note = f"手工回退：{stage} -> {status}"
    note_text = (notes or "").strip() or default_note
    downstream_note = f"{note_text} | 上游 {stage} 已回退为 {status}，downstream 需重新执行"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_db_connection()
    cursor = conn.cursor()

    placeholders = ", ".join("?" for _ in model_ids)
    cursor.execute(f"SELECT * FROM models WHERE model_id IN ({placeholders})", model_ids)
    rows = cursor.fetchall()
    row_by_model_id = {row["model_id"]: row for row in rows}

    missing_model_ids = [model_id for model_id in model_ids if model_id not in row_by_model_id]
    if missing_model_ids:
        conn.close()
        print("Error: some model_id values were not found:")
        for model_id in missing_model_ids:
            print(model_id)
        sys.exit(1)

    owners_to_idle: set[str] = set()
    updated_models: list[str] = []

    for model_id in model_ids:
        row = row_by_model_id[model_id]
        owner = (row[stage_cfg["owner_col"]] or "").strip()
        if owner:
            owners_to_idle.add(owner)

        assignments = [
            (stage_cfg["status_col"], status),
            (stage_cfg["owner_col"], ""),
            (stage_cfg["updated_col"], now),
            (stage_cfg["notes_col"], note_text),
        ]
        if stage == "adaptation":
            assignments.append(("adaptation_failure_reason", ""))

        set_clause = ", ".join(f"{col} = ?" for col, _ in assignments)
        params = [value for _, value in assignments]
        params.append(model_id)
        cursor.execute(f"UPDATE models SET {set_clause} WHERE model_id = ?", params)

        owners_to_idle.update(
            _reset_later_stage_statuses_for_model(
                stage,
                row,
                downstream_note,
                cursor=cursor,
                now=now,
            )
        )
        updated_models.append(model_id)

    for owner in sorted(owners_to_idle):
        cursor.execute("UPDATE agents SET status = 'idle', current_task = '' WHERE id = ?", (owner,))
        print(f"Set agent {owner} status to idle, current_task cleared.")

    conn.commit()
    conn.close()
    print(f"Rolled back {len(updated_models)} models to {stage}:{status}.")
    for model_id in updated_models:
        print(model_id)


def heartbeat(agent_id, status, current_task):
    if status not in ["active", "idle", "offline"]:
        print(f"Error: Invalid status '{status}'. Must be active, idle, or offline.")
        sys.exit(1)

    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """
        INSERT OR REPLACE INTO agents (id, last_heartbeat, status, current_task)
        VALUES (?, ?, ?, ?)
    """,
        (agent_id, now, status, _format_agent_current_task(agent_id, current_task)),
    )
    conn.commit()
    conn.close()
    print(f"Heartbeat updated for {agent_id}.")


def register_model(model_id, source="huggingface", priority="medium", url="", description="", adaptation_notes="", adaptation_status="pending", adaptation_failure_reason=""):
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check if model exists to avoid overwriting existing status
    cursor.execute("SELECT model_id FROM models WHERE model_id = ?", (model_id,))
    if cursor.fetchone():
        print(f"Model {model_id} already exists. Skipping.")
        conn.close()
        return

    # url 必须存在且非空
    url_val = (url or "").strip()
    if not url_val:
        print("Error: url is required and must be non-empty.")
        conn.close()
        sys.exit(1)

    # url 必须唯一
    cursor.execute("SELECT model_id FROM models WHERE url = ?", (url_val,))
    existing = cursor.fetchone()
    if existing:
        print(f"URL already registered for model {existing['model_id']}. Skipping {model_id}.")
        conn.close()
        return

    cursor.execute(
        """
        INSERT INTO models (
            model_id, source, priority, adaptation_status, adaptation_started_at, adaptation_last_updated,
            url, description, adaptation_notes, adaptation_failure_reason, adaptation_path, adaptation_owner
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '')
    """,
        (model_id, source, priority, adaptation_status, now, now, url_val, description, adaptation_notes, adaptation_failure_reason, model_id_to_adaptation_path(model_id)),
    )

    conn.commit()
    conn.close()
    print(f"Model {model_id} registered.")


def assign_adaptation_task(agent_id):
    _assign_stage_task(
        agent_id,
        _ASSIGN_STAGE_CONFIG["adaptation"],
        select_cols_sql="model_id",
        extra_assignments_factory=lambda model_id: [("adaptation_path", model_id_to_adaptation_path(model_id))],
    )


def update_adaptation_status(model_id, adaptation_status, adaptation_notes="", adaptation_failure_reason="", adaptation_path=""):
    valid_statuses = ["completed", "needs_authorization", "not_applicable", "skipped", "in_progress", "pending"]
    if adaptation_status not in valid_statuses:
        print(f"Error: Invalid adaptation_status '{adaptation_status}'.")
        sys.exit(1)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT adaptation_status, adaptation_owner, adaptation_path, adaptation_failure_reason FROM models WHERE model_id = ?",
        (model_id,),
    )
    row_full = cursor.fetchone()
    if not row_full:
        conn.close()
        print(f"Error: model_id {model_id} not found.")
        sys.exit(1)

    old_status = (row_full["adaptation_status"] or "").strip().lower()
    if old_status == "completed":
        conn.close()
        print("Error: adaptation_status is already completed and cannot be changed.")
        sys.exit(1)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_status = (adaptation_status or "").strip().lower()

    # 若任务进入终态或回退为 pending，需要先查出 owner，以便把该 adapter 状态设回 idle
    owner_to_idle = None
    row = row_full
    if adaptation_status in ["completed", "skipped", "not_applicable", "needs_authorization", "pending"]:
        if row and row["adaptation_owner"]:
            owner_to_idle = row["adaptation_owner"]

    # 拦截与完成时路径：adaptation_status=completed 前必须存在 demo.py 且通过 check_adaptation.py
    path_val, adapt_name = "", ""
    if adaptation_status == "completed":
        path_val, adapt_name = _resolve_adapt_path(model_id, adaptation_path or "", row)
        if not (path_val or "").strip():
            conn.close()
            print("Intercepted: cannot set adaptation_status=completed: adaptation path is empty")
            print(f"INTERCEPTED: model_id={model_id} owner={owner_to_idle or ''} notes=无 adaptation_path")
            sys.exit(1)
        path_ok, path_err = _validate_adaptation_path_boundary(path_val)
        if not path_ok:
            conn.close()
            print(f"Intercepted: cannot set adaptation_status=completed: {path_err}")
            print(f"INTERCEPTED: model_id={model_id} owner={owner_to_idle or ''} notes={path_err}")
            sys.exit(1)
        demo_py = os.path.join(_PROJECT_ROOT, path_val, "demo.py")
        if not os.path.isfile(demo_py):
            conn.close()
            print(f"Intercepted: cannot set adaptation_status=completed: demo.py not found at {demo_py}")
            print(f"INTERCEPTED: model_id={model_id} owner={owner_to_idle or ''} notes=缺少 demo.py")
            sys.exit(1)
        check_ok, check_err = _run_check_script(Path(_PROJECT_ROOT) / "adaptation" / "scripts" / "check_adaptation.py", adapt_name, "check_adaptation.py")
        if not check_ok:
            conn.close()
            adaptation_notes = f"check_adaptation.py 未通过，需修复 adaptation 后重新完成。{check_err}"
            print(f"Intercepted: cannot set adaptation_status=completed until check passes for {adapt_name}")
            print(f"INTERCEPTED: model_id={model_id} owner={owner_to_idle or ''} notes={adaptation_notes[:200]}")
            sys.exit(1)

    owner_val = "" if adaptation_status in ["completed", "skipped", "not_applicable", "needs_authorization", "pending"] else None  # Release owner if done

    query = "UPDATE models SET adaptation_status = ?, adaptation_last_updated = ?, adaptation_notes = ?"
    params = [adaptation_status, now, adaptation_notes]

    # 完成时写入适配目录路径，并设 benchmark_status='pending' 以便评测任务分配
    if adaptation_status == "completed":
        query += ", benchmark_status = ?"
        params.append("pending")
        abs_path = os.path.join(str(_PROJECT_ROOT), path_val) if path_val else ""
        if path_val and os.path.isdir(abs_path):
            query += ", adaptation_path = ?"
            params.append(path_val)
        elif path_val:
            print(f"Warning: adaptation_path not set (directory not found: {path_val})")

    if adaptation_failure_reason:
        query += ", adaptation_failure_reason = ?"
        params.append(adaptation_failure_reason)
    elif old_status != new_status:
        # 状态变更且本次未显式写 failure_reason 时清空，避免历史失败原因污染新状态
        query += ", adaptation_failure_reason = ?"
        params.append("")

    if owner_val is not None:
        query += ", adaptation_owner = ?"
        params.append(owner_val)

    query += " WHERE model_id = ?"
    params.append(model_id)

    cursor.execute(query, params)

    # 任务终态时：将对应 owner 在 agents 表中设为 idle，并清空 current_task，便于再次分配
    if owner_to_idle:
        cursor.execute(
            "UPDATE agents SET status = 'idle', current_task = '' WHERE id = ?",
            (owner_to_idle,),
        )
        print(f"Set agent {owner_to_idle} status to idle, current_task cleared.")

    conn.commit()
    conn.close()
    print(f"Updated {model_id} adaptation_status to {adaptation_status}.")

    if adaptation_status == "completed":
        git_commit_and_push(
            f"feat: complete adaptation for {model_id}",
            paths=_build_model_commit_paths(model_id, adaptation_path=path_val, row=row),
        )


def assign_benchmark_task(agent_id):
    """分配 benchmark 任务给 agent（仅限 adaptation_status=completed 且 benchmark_status=pending 的模型）"""
    _assign_stage_task(agent_id, _ASSIGN_STAGE_CONFIG["benchmark"])


def _resolve_adapt_path(
    model_id: str,
    adaptation_path: str = "",
    row: sqlite3.Row | None = None,
) -> tuple[str, str]:
    """从 adaptation_path/row 解析出 path_val 与 adapt_name；不再隐式回退到 model_id。"""
    path_val = (adaptation_path or "").strip()
    if not path_val and row is not None:
        path_val = row["adaptation_path"] or ""
    if not path_val:
        return "", ""
    if path_val and not path_val.startswith("adaptations/"):
        path_val = f"adaptations/{path_val.lstrip('/')}"
    adapt_name = path_val.replace("adaptations/", "").strip("/") if path_val else ""
    return path_val or "", adapt_name


def _validate_adaptation_path_boundary(path_val: str) -> tuple[bool, str]:
    """校验 adaptation_path 必须位于项目 adaptations/ 目录内。"""
    normalized = (path_val or "").strip().replace("\\", "/").strip("/")
    if not normalized:
        return False, "adaptation_path 为空"
    if not normalized.startswith("adaptations/"):
        return False, "adaptation_path 必须位于 adaptations/ 下"

    project_root = Path(_PROJECT_ROOT).resolve()
    adaptations_root = (project_root / "adaptations").resolve()
    resolved = (project_root / normalized).resolve()

    try:
        resolved.relative_to(adaptations_root)
    except ValueError:
        return False, "adaptation_path 越界，未位于 adaptations/ 目录内"

    if resolved == project_root or resolved == adaptations_root:
        return False, "adaptation_path 不能指向项目根或 adaptations 根目录"

    return True, ""


def _build_model_commit_paths(
    model_id: str,
    adaptation_path: str = "",
    row: sqlite3.Row | None = None,
) -> list[str]:
    """仅收集当前模型负责的看板与 adaptation 目录，避免误提交无关文件。"""
    commit_paths = [os.path.relpath(DB_PATH, _PROJECT_ROOT)]
    path_val, _ = _resolve_adapt_path(model_id, adaptation_path, row)
    if not path_val:
        return commit_paths

    path_ok, path_err = _validate_adaptation_path_boundary(path_val)
    if not path_ok:
        print(f"Warning: skipping adaptation_path for git add on {model_id}: {path_err}")
        return commit_paths

    abs_path = Path(_PROJECT_ROOT) / path_val
    if not abs_path.exists():
        print(f"Warning: skipping missing adaptation_path for git add on {model_id}: {path_val}")
        return commit_paths

    commit_paths.append(path_val)
    return commit_paths


def _run_check_script(check_script: Path, adapt_name: str, script_name: str, extra_args: list[str] | None = None) -> tuple[bool, str]:
    """运行检查脚本，通过返回 (True, '')，否则 (False, error_detail)。"""
    if not check_script.is_file():
        print(f"Warning: {script_name} not found at {check_script}, skip interception.")
        return True, ""
    command = [sys.executable, str(check_script), "--adapt", adapt_name]
    if extra_args:
        command.extend(extra_args)
    result = subprocess.run(
        command,
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        err_detail = (result.stderr or "") + (result.stdout or "")
        err_detail = err_detail.strip()[:500] if err_detail else f"违规项见 {script_name} 输出"
        print(f"Error: {script_name} failed for {adapt_name}:")
        if result.stderr:
            print(result.stderr)
        if result.stdout:
            print(result.stdout)
        return False, err_detail
    return True, ""


def _run_check_optimization_notes(notes_str: str) -> tuple[bool, str]:
    """校验 optimization_notes JSON 格式，通过返回 (True, '')，否则 (False, error_detail)。"""
    check_script = Path(_PROJECT_ROOT) / "optimization" / "scripts" / "check_optimization_notes.py"
    if not check_script.is_file():
        # check script 不存在时做内联校验（fallback）
        try:
            import json

            data = json.loads(notes_str)
            if not isinstance(data, dict):
                return False, "notes 必须是 JSON object"
            for key in ["measurement_contract_version", "optimizations", "results", "best_result"]:
                if key not in data:
                    return False, f"缺少必填字段: {key}"
            contract_v = data.get("measurement_contract_version")
            if not isinstance(contract_v, (int, float)) or isinstance(contract_v, bool) or int(contract_v) < 3:
                return False, "measurement_contract_version 必须 >= 3"
            if not isinstance(data.get("results"), list) or len(data["results"]) == 0:
                return False, "results 必须是非空数组"
            for i, r in enumerate(data["results"]):
                for field in [
                    "dtype",
                    "mode",
                    "dataset",
                    "output_type",
                    "baseline_artifact",
                    "perf_artifact",
                    "perf_latency_s",
                    "baseline_latency_s",
                    "baseline_wall_clock_s",
                    "perf_wall_clock_s",
                    "wall_clock_source",
                    "baseline_warmup_iterations",
                    "perf_warmup_iterations",
                    "warmup_policy",
                    "perf_memory_mb",
                    "speedup_ratio",
                ]:
                    if field not in r:
                        return False, f"results[{i}] 缺少必填字段: {field}"
                if r.get("wall_clock_source") not in {"artifact_timestamps", "artifact_explicit_field"}:
                    return False, f"results[{i}].wall_clock_source 必须为 artifact_timestamps 或 artifact_explicit_field"
                if r.get("warmup_policy") != "symmetric":
                    return False, f"results[{i}].warmup_policy 必须为 symmetric"
                if _contains_measurement_red_flag(str(r.get("validation_note") or "")):
                    return False, f"results[{i}].validation_note 包含测量红旗"
            br = data.get("best_result")
            if br is None:
                return False, "best_result 不能为 null"
            elif isinstance(br, dict):
                found = any(isinstance(r, dict) and r.get("dtype") == br.get("dtype") and r.get("mode") == br.get("mode") for r in data["results"])
                if not found:
                    return False, f"best_result (dtype={br.get('dtype')}, mode={br.get('mode')}) 不在 results[] 中"
                for field in (
                    "baseline_wall_clock_s",
                    "perf_wall_clock_s",
                    "wall_clock_source",
                    "baseline_warmup_iterations",
                    "perf_warmup_iterations",
                    "warmup_policy",
                    "speedup_ratio",
                ):
                    if field not in br:
                        return False, f"best_result 缺少必填字段: {field}"
                if br.get("wall_clock_source") not in {"artifact_timestamps", "artifact_explicit_field"}:
                    return False, "best_result.wall_clock_source 必须为 artifact_timestamps 或 artifact_explicit_field"
                if br.get("warmup_policy") != "symmetric":
                    return False, "best_result.warmup_policy 必须为 symmetric"
                if _contains_measurement_red_flag(str(br.get("validation_note") or "")):
                    return False, "best_result.validation_note 包含测量红旗"
            return True, ""
        except json.JSONDecodeError as e:
            return False, f"无效 JSON: {e}"
        except Exception as e:
            return False, str(e)
    # 通过 import check script 的 validate_notes 进行完整校验
    try:
        mod = _load_python_module_from_path("check_optimization_notes", check_script)
        errors = mod.validate_notes(notes_str, model_id="")
        if errors:
            return False, "; ".join(errors)
        return True, ""
    except Exception as e:
        return False, f"校验脚本执行异常: {e}"


def _read_optimization_notes_file(path_val: str) -> tuple[bool, str, str]:
    """读取 adaptation 目录中的 optimization_notes.json，并做基础合法性校验。"""
    notes_path = Path(_PROJECT_ROOT) / path_val / "optimization_notes.json"
    if not notes_path.is_file():
        return False, "", f"缺少 optimization_notes.json: {notes_path}"
    try:
        raw = notes_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        return False, "", f"读取 optimization_notes.json 失败: {e}"
    if not raw:
        return False, "", "optimization_notes.json 为空"
    check_ok, check_err = _run_check_optimization_notes(raw)
    if not check_ok:
        return False, "", f"optimization_notes.json 非法: {check_err}"
    return True, raw, ""


def _verify_db_optimization_notes(raw: str, expected_raw: str) -> tuple[bool, str]:
    """验证写入 DB 的 optimization_notes 非空、合法且与期望一致。"""
    value = (raw or "").strip()
    expected = (expected_raw or "").strip()
    if not value:
        return False, "board.db.optimization_notes 为空"
    if value != expected:
        return False, "board.db.optimization_notes 与传入 notes 不一致"
    try:
        obj = json.loads(value)
    except json.JSONDecodeError as e:
        return False, f"board.db.optimization_notes 非法 JSON: {e}"
    if not isinstance(obj, dict):
        return False, "board.db.optimization_notes 必须是 JSON object"
    for key in ("optimizations", "results", "best_result"):
        if key not in obj:
            return False, f"board.db.optimization_notes 缺少字段: {key}"
    return True, ""


def _run_check_business_summary(summary_str: str) -> tuple[bool, str]:
    """校验 business_summary JSON 格式，通过返回 (True, '')，否则 (False, error_detail)。"""
    check_script = Path(_PROJECT_ROOT) / "business_benchmark" / "scripts" / "check_business_benchmark_run.py"
    if not check_script.is_file():
        try:
            data = json.loads(summary_str)
        except json.JSONDecodeError as e:
            return False, f"business_summary 非法 JSON: {e}"
        if not isinstance(data, dict):
            return False, "business_summary 必须是 JSON object"
        for key in (
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
        ):
            if key not in data:
                return False, f"business_summary 缺少字段: {key}"
        return True, ""

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("check_business_benchmark_run", str(check_script))
        if spec is None or spec.loader is None:
            return False, "check_business_benchmark_run.py 加载失败"
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        errors = mod.validate_summary(summary_str, model_id="")
        if errors:
            return False, "; ".join(errors)
        return True, ""
    except Exception as e:
        return False, f"校验脚本执行异常: {e}"


def _read_business_summary_file(path_val: str) -> tuple[bool, str, str]:
    """读取 adaptation 目录中的 business_summary.json，并做基础合法性校验。"""
    summary_path = Path(_PROJECT_ROOT) / path_val / "business_summary.json"
    if not summary_path.is_file():
        return False, "", f"缺少 business_summary.json: {summary_path}"
    try:
        raw = summary_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        return False, "", f"读取 business_summary.json 失败: {e}"
    if not raw:
        return False, "", "business_summary.json 为空"
    check_ok, check_err = _run_check_business_summary(raw)
    if not check_ok:
        return False, "", f"business_summary.json 非法: {check_err}"
    return True, raw, ""


def _verify_db_business_summary(raw: str, expected_raw: str) -> tuple[bool, str]:
    """验证写入 DB 的 business_benchmark_notes 非空、合法且与期望一致。"""
    value = (raw or "").strip()
    expected = (expected_raw or "").strip()
    if not value:
        return False, "board.db.business_benchmark_notes 为空"
    if value != expected:
        return False, "board.db.business_benchmark_notes 与传入 notes 不一致"
    try:
        obj = json.loads(value)
    except json.JSONDecodeError as e:
        return False, f"board.db.business_benchmark_notes 非法 JSON: {e}"
    if not isinstance(obj, dict):
        return False, "board.db.business_benchmark_notes 必须是 JSON object"
    for key in ("dataset", "comparison_evidence", "results", "best_result", "npu_baseline_artifact", "npu_perf_artifact", "cuda_baseline_artifact"):
        if key not in obj:
            return False, f"board.db.business_benchmark_notes 缺少字段: {key}"
    return True, ""


def _validate_completed_business_summary(notes_str: str) -> tuple[bool, str]:
    """completed 状态下，business_summary 必须包含完整三路真实业务结果。"""
    try:
        data = json.loads(notes_str)
    except json.JSONDecodeError as e:
        return False, f"business_summary 非法 JSON: {e}"
    if not isinstance(data, dict):
        return False, "business_summary 必须是 JSON object"

    for key in ("dataset", "comparison_scope", "num_samples", "remote_execution", "comparison_evidence", "results", "best_result", "npu_baseline_artifact", "npu_perf_artifact", "cuda_baseline_artifact"):
        if key not in data:
            return False, f"business_summary 缺少字段: {key}"

    dataset = str(data.get("dataset") or "").strip()
    if not dataset:
        return False, "business_summary.dataset 不能为空"
    comparison_scope = str(data.get("comparison_scope") or "").strip()
    if comparison_scope not in _VALID_BUSINESS_COMPARISON_SCOPES:
        return False, f"business_summary.comparison_scope 无效: {comparison_scope}"
    num_samples = data.get("num_samples")
    if not isinstance(num_samples, (int, float)) or isinstance(num_samples, bool) or float(num_samples) <= _MIN_COMPLETED_BUSINESS_SAMPLE_LOWER_BOUND:
        return False, f"business_summary.num_samples 必须 > {_MIN_COMPLETED_BUSINESS_SAMPLE_LOWER_BOUND}"
    remote_execution = data.get("remote_execution")
    if not isinstance(remote_execution, dict):
        return False, "business_summary.remote_execution 必须是 object"
    remote_execution = cast(dict[str, Any], remote_execution)
    remote_mode = str(remote_execution.get("mode") or "").strip()
    if not remote_mode:
        return False, "business_summary.remote_execution.mode 不能为空"
    comparison_evidence = data.get("comparison_evidence")
    if not isinstance(comparison_evidence, dict):
        return False, "business_summary.comparison_evidence 必须是 object"
    comparison_evidence = cast(dict[str, Any], comparison_evidence)
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
            return False, f"business_summary.comparison_evidence.{field} 不能为空"
    for field in (
        "npu_baseline_peak_memory_mb",
        "npu_perf_peak_memory_mb",
        "cuda_baseline_peak_memory_mb",
        "npu_baseline_throughput_metric_value",
        "npu_perf_throughput_metric_value",
        "cuda_baseline_throughput_metric_value",
    ):
        value = comparison_evidence.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0:
            return False, f"business_summary.comparison_evidence.{field} 必须为正数"
    for field in ("npu_baseline_quality_metric_value", "npu_perf_quality_metric_value", "cuda_baseline_quality_metric_value"):
        value = comparison_evidence.get(field)
        if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) < 0):
            return False, f"business_summary.comparison_evidence.{field} 必须为非负数或 null"
    for name_field, value_field in (
        ("npu_baseline_quality_metric_name", "npu_baseline_quality_metric_value"),
        ("npu_perf_quality_metric_name", "npu_perf_quality_metric_value"),
        ("cuda_baseline_quality_metric_name", "cuda_baseline_quality_metric_value"),
    ):
        if not _business_quality_within_unit_range(comparison_evidence.get(name_field), comparison_evidence.get(value_field)):
            return False, f"business_summary.comparison_evidence.{value_field} 超出 0~1 合法范围；疑似评分实现或汇总口径异常"

    results = data.get("results")
    if not isinstance(results, list) or len(results) < 3:
        return False, "business_summary.results 至少包含 3 条结果（npu_baseline / npu_perf / cuda_baseline）"
    best_result = data.get("best_result")
    if not isinstance(best_result, dict):
        return False, "business_summary.best_result 必须是 object"

    required_roles = {"npu_baseline", "npu_perf", "cuda_baseline"}
    seen_roles: set[str] = set()
    for idx, result in enumerate(results):
        if not isinstance(result, dict):
            return False, f"business_summary.results[{idx}] 必须是 object"
        result = cast(dict[str, Any], result)
        role = str(result.get("role") or "").strip()
        if role not in required_roles:
            return False, f"business_summary.results[{idx}].role 无效: {role}"
        seen_roles.add(role)
        artifact = str(result.get("artifact") or "").strip()
        if not artifact:
            return False, f"business_summary.results[{idx}].artifact 不能为空"
        for field in ("device", "device_model", "mode", "dtype", "dataset", "output_type", "throughput_metric_name", "quality_metric_name"):
            if not str(result.get(field) or "").strip():
                return False, f"business_summary.results[{idx}].{field} 不能为空"
        result_num_samples = result.get("num_samples")
        if not isinstance(result_num_samples, (int, float)) or isinstance(result_num_samples, bool) or float(result_num_samples) <= _MIN_COMPLETED_BUSINESS_SAMPLE_LOWER_BOUND:
            return False, f"business_summary.results[{idx}].num_samples 必须 > {_MIN_COMPLETED_BUSINESS_SAMPLE_LOWER_BOUND}"
        latency_s = result.get("latency_s")
        if not isinstance(latency_s, (int, float)) or float(latency_s) <= 0:
            return False, f"business_summary.results[{idx}].latency_s 必须为正数"
        peak_memory_mb = result.get("peak_memory_mb")
        if not isinstance(peak_memory_mb, (int, float)) or isinstance(peak_memory_mb, bool) or float(peak_memory_mb) <= 0:
            return False, f"business_summary.results[{idx}].peak_memory_mb 必须为正数"
        throughput_metric_value = result.get("throughput_metric_value")
        if not isinstance(throughput_metric_value, (int, float)) or isinstance(throughput_metric_value, bool) or float(throughput_metric_value) <= 0:
            return False, f"business_summary.results[{idx}].throughput_metric_value 必须为正数"
        quality_metric_value = result.get("quality_metric_value")
        if quality_metric_value is not None and (not isinstance(quality_metric_value, (int, float)) or isinstance(quality_metric_value, bool) or float(quality_metric_value) < 0):
            return False, f"business_summary.results[{idx}].quality_metric_value 必须为非负数或 null"
        if not _business_quality_within_unit_range(result.get("quality_metric_name"), quality_metric_value):
            return False, f"business_summary.results[{idx}].quality_metric_value 超出 0~1 合法范围；疑似评分实现或汇总口径异常"

    if seen_roles != required_roles:
        return False, "business_summary.results 必须同时包含 npu_baseline / npu_perf / cuda_baseline"

    best_role = str(best_result.get("role") or "").strip()
    if best_role != "npu_perf":
        return False, "business_summary.best_result.role 必须为 npu_perf"
    for field in ("npu_speedup_ratio", "vs_cuda_latency_ratio", "npu_perf_peak_memory_mb", "cuda_baseline_peak_memory_mb", "npu_perf_throughput_metric_value", "cuda_baseline_throughput_metric_value"):
        value = best_result.get(field)
        if not isinstance(value, (int, float)) or float(value) <= 0:
            return False, f"business_summary.best_result.{field} 必须为正数"
    for field in ("output_type", "npu_perf_device_model", "cuda_baseline_device_model", "npu_perf_throughput_metric_name", "cuda_baseline_throughput_metric_name", "quality_metric_name"):
        if not str(best_result.get(field) or "").strip():
            return False, f"business_summary.best_result.{field} 不能为空"
    quality_metric_value = best_result.get("quality_metric_value")
    if quality_metric_value is not None and (not isinstance(quality_metric_value, (int, float)) or isinstance(quality_metric_value, bool) or float(quality_metric_value) < 0):
        return False, "business_summary.best_result.quality_metric_value 必须为非负数或 null"
    if not _business_quality_within_unit_range(best_result.get("quality_metric_name"), quality_metric_value):
        return False, "business_summary.best_result.quality_metric_value 超出 0~1 合法范围；疑似评分实现或汇总口径异常"
    npu_speedup_ratio = best_result.get("npu_speedup_ratio")
    if isinstance(npu_speedup_ratio, (int, float)) and not isinstance(npu_speedup_ratio, bool) and float(npu_speedup_ratio) < _MIN_SANE_BUSINESS_NPU_SPEEDUP_RATIO:
        return False, (f"business_summary.best_result.npu_speedup_ratio={float(npu_speedup_ratio):.6g} < {_MIN_SANE_BUSINESS_NPU_SPEEDUP_RATIO:.6g}；疑似 NPU perf 退化或 measurement 口径异常")
    vs_cuda_latency_ratio = best_result.get("vs_cuda_latency_ratio")
    if isinstance(vs_cuda_latency_ratio, (int, float)) and not isinstance(vs_cuda_latency_ratio, bool):
        vs_cuda_ratio_value = float(vs_cuda_latency_ratio)
        if vs_cuda_ratio_value < _MIN_SANE_BUSINESS_VS_CUDA_LATENCY_RATIO:
            return False, (f"business_summary.best_result.vs_cuda_latency_ratio={vs_cuda_ratio_value:.6g} < {_MIN_SANE_BUSINESS_VS_CUDA_LATENCY_RATIO:.6g}；疑似跨设备对比口径异常")
    return True, ""


def _validate_business_metric_artifacts(path_val: str, notes_str: str) -> tuple[bool, str]:
    """校验 business_summary 引用的业务测评工件存在且元数据健康。"""
    try:
        data = json.loads(notes_str)
    except json.JSONDecodeError as e:
        return False, f"business_summary 非法 JSON: {e}"
    if not isinstance(data, dict):
        return False, "business_summary 必须是 JSON object"

    adapt_dir = Path(_PROJECT_ROOT) / path_val
    if not adapt_dir.is_dir():
        return False, f"adaptation 目录不存在: {adapt_dir}"

    dataset = str(data.get("dataset") or "").strip()
    result_entries = {}
    metric_entries = {}
    for result in data.get("results", []):
        if isinstance(result, dict):
            result_entries[str(result.get("role") or "").strip()] = result

    for role, artifact_key, expected_device in (
        ("npu_baseline", "npu_baseline_artifact", "npu"),
        ("npu_perf", "npu_perf_artifact", "npu"),
        ("cuda_baseline", "cuda_baseline_artifact", "cuda"),
    ):
        artifact_name = str(data.get(artifact_key) or "").strip()
        if not artifact_name:
            return False, f"business_summary.{artifact_key} 不能为空"
        artifact_path = adapt_dir / artifact_name
        if not artifact_path.is_file():
            return False, f"缺少业务测评工件: {artifact_path}"
        try:
            metric = json.loads(artifact_path.read_text(encoding="utf-8"))
        except Exception as e:
            return False, f"读取业务测评工件失败 {artifact_name}: {e}"
        if not isinstance(metric, dict):
            return False, f"业务测评工件 {artifact_name} 必须是 JSON object"
        ok, err = _validate_required_metric_fields(metric, artifact_path, f"业务测评工件 {artifact_name}", _REQUIRED_COMPLETED_BENCHMARK_FIELDS)
        if not ok:
            return False, err
        meta_ok, meta_err = _validate_metric_metadata_health(metric, artifact_path, f"业务测评工件 {artifact_name}")
        if not meta_ok:
            return False, meta_err
        if expected_device not in str(metric.get("device") or "").lower():
            return False, f"业务测评工件 {artifact_name} 的 device={metric.get('device')}，必须包含 {expected_device}"
        if not str(metric.get("device_model") or "").strip():
            return False, f"业务测评工件 {artifact_name} 缺少非空 device_model"
        peak_memory_mb = metric.get("peak_memory_mb")
        if not isinstance(peak_memory_mb, (int, float)) or isinstance(peak_memory_mb, bool) or float(peak_memory_mb) <= 0:
            return False, f"业务测评工件 {artifact_name} 缺少正数 peak_memory_mb"
        if str(metric.get("dataset") or "").strip() != dataset:
            return False, f"业务测评工件 {artifact_name} 的 dataset 与 business_summary.dataset 不一致"
        metric_entries[role] = metric
        role_entry = result_entries.get(role)
        if not isinstance(role_entry, dict):
            return False, f"business_summary.results 缺少 {role} 记录"
        if str(role_entry.get("artifact") or "").strip() != artifact_name:
            return False, f"business_summary.results 中 {role} 的 artifact 与顶层 {artifact_key} 不一致"
        if not _metric_close(float(metric["latency_s"]), float(role_entry.get("latency_s", -1))):
            return False, f"business_summary.results 中 {role}.latency_s 与工件 {artifact_name} 不一致"
        if str(role_entry.get("output_type") or "").strip() != str(metric.get("output_type") or "").strip():
            return False, f"business_summary.results 中 {role}.output_type 与工件 {artifact_name} 不一致"
        if str(role_entry.get("device_model") or "").strip() != str(metric.get("device_model") or "").strip():
            return False, f"business_summary.results 中 {role}.device_model 与工件 {artifact_name} 不一致"
        role_peak_memory_mb = role_entry.get("peak_memory_mb")
        if not isinstance(role_peak_memory_mb, (int, float)) or isinstance(role_peak_memory_mb, bool):
            return False, f"business_summary.results 中 {role}.peak_memory_mb 必须为正数"
        if not _metric_close(float(metric["peak_memory_mb"]), float(role_peak_memory_mb)):
            return False, f"business_summary.results 中 {role}.peak_memory_mb 与工件 {artifact_name} 不一致"
        throughput_name, throughput_value = _detect_business_throughput(metric)
        if str(role_entry.get("throughput_metric_name") or "").strip() != throughput_name:
            return False, f"business_summary.results 中 {role}.throughput_metric_name 与工件 {artifact_name} 不一致"
        role_tp_value = role_entry.get("throughput_metric_value")
        if not isinstance(role_tp_value, (int, float)) or isinstance(role_tp_value, bool):
            return False, f"business_summary.results 中 {role}.throughput_metric_value 必须为正数"
        if throughput_value is None or not _metric_close(float(throughput_value), float(role_tp_value)):
            return False, f"business_summary.results 中 {role}.throughput_metric_value 与工件 {artifact_name} 不一致"
        quality_name, quality_value = _detect_business_quality(metric)
        if not _business_quality_within_unit_range(quality_name, quality_value):
            return False, f"业务测评工件 {artifact_name} 的 {quality_name or 'quality_metric'} 超出 0~1 合法范围；疑似评分实现或汇总口径异常"
        if str(role_entry.get("quality_metric_name") or "").strip() != quality_name:
            return False, f"business_summary.results 中 {role}.quality_metric_name 与工件 {artifact_name} 不一致"
        role_quality_value = role_entry.get("quality_metric_value")
        if quality_value is None:
            if role_quality_value is not None:
                return False, f"business_summary.results 中 {role}.quality_metric_value 与工件 {artifact_name} 不一致"
        else:
            if not isinstance(role_quality_value, (int, float)) or isinstance(role_quality_value, bool) or not _metric_close(float(quality_value), float(role_quality_value)):
                return False, f"business_summary.results 中 {role}.quality_metric_value 与工件 {artifact_name} 不一致"

    output_types = {str(metric.get("output_type") or "").strip() for metric in metric_entries.values()}
    output_types.discard("")
    if len(output_types) > 1:
        return False, f"业务测评工件的 output_type 不一致: {sorted(output_types)}"
    quality_alignment_ok, quality_alignment_err = _validate_business_quality_alignment(metric_entries)
    if not quality_alignment_ok:
        return False, quality_alignment_err

    best = data.get("best_result") or {}
    if isinstance(best, dict):
        npu_baseline = result_entries.get("npu_baseline") or {}
        npu_perf = result_entries.get("npu_perf") or {}
        cuda_baseline = result_entries.get("cuda_baseline") or {}
        if isinstance(npu_baseline, dict) and isinstance(npu_perf, dict):
            baseline_artifact_name = str(data.get("npu_baseline_artifact") or "").strip()
            perf_artifact_name = str(data.get("npu_perf_artifact") or "").strip()
            baseline_path = adapt_dir / baseline_artifact_name if baseline_artifact_name else adapt_dir / "business_metrics_npu_baseline.json"
            perf_path = adapt_dir / perf_artifact_name if perf_artifact_name else adapt_dir / "business_metrics_npu_perf.json"
            baseline_wall_clock, baseline_wall_clock_err = _extract_metric_wall_clock_s(metric_entries.get("npu_baseline", {}), baseline_path, "business NPU baseline")
            perf_wall_clock, perf_wall_clock_err = _extract_metric_wall_clock_s(metric_entries.get("npu_perf", {}), perf_path, "business NPU perf")
            if not baseline_wall_clock_err and not perf_wall_clock_err and baseline_wall_clock and perf_wall_clock:
                expected_speedup = baseline_wall_clock / perf_wall_clock
            else:
                baseline_latency = float(npu_baseline.get("latency_s", 0) or 0)
                perf_latency = float(npu_perf.get("latency_s", 0) or 0)
                expected_speedup = baseline_latency / perf_latency if baseline_latency > 0 and perf_latency > 0 else None
            if expected_speedup is not None and not _metric_close(expected_speedup, float(best.get("npu_speedup_ratio", -1) or -1)):
                return False, "business_summary.best_result.npu_speedup_ratio 与 NPU baseline/perf 工件不一致"
        if isinstance(cuda_baseline, dict) and isinstance(npu_perf, dict):
            cuda_latency = float(cuda_baseline.get("latency_s", 0) or 0)
            perf_latency = float(npu_perf.get("latency_s", 0) or 0)
            if cuda_latency > 0 and perf_latency > 0:
                expected_vs_cuda = cuda_latency / perf_latency
                if not _metric_close(expected_vs_cuda, float(best.get("vs_cuda_latency_ratio", -1) or -1)):
                    return False, "business_summary.best_result.vs_cuda_latency_ratio 与 CUDA baseline / NPU perf 工件不一致"

        if str(best.get("output_type") or "").strip() != str(npu_perf.get("output_type") or "").strip():
            return False, "business_summary.best_result.output_type 与 npu_perf 工件不一致"
        npu_perf_quality_name, npu_perf_quality_value = _detect_business_quality(metric_entries.get("npu_perf", {}))
        if str(best.get("quality_metric_name") or "").strip() != npu_perf_quality_name:
            return False, "business_summary.best_result.quality_metric_name 与 npu_perf 工件不一致"
        best_quality_value = best.get("quality_metric_value")
        if npu_perf_quality_value is None:
            if best_quality_value is not None:
                return False, "business_summary.best_result.quality_metric_value 与 npu_perf 工件不一致"
        elif not isinstance(best_quality_value, (int, float)) or isinstance(best_quality_value, bool) or not _metric_close(float(npu_perf_quality_value), float(best_quality_value)):
            return False, "business_summary.best_result.quality_metric_value 与 npu_perf 工件不一致"
        if str(best.get("npu_perf_device_model") or "").strip() != str(npu_perf.get("device_model") or "").strip():
            return False, "business_summary.best_result.npu_perf_device_model 与 npu_perf 工件不一致"
        if str(best.get("cuda_baseline_device_model") or "").strip() != str(cuda_baseline.get("device_model") or "").strip():
            return False, "business_summary.best_result.cuda_baseline_device_model 与 cuda_baseline 工件不一致"

        npu_perf_peak_memory_mb = best.get("npu_perf_peak_memory_mb")
        cuda_peak_memory_mb = best.get("cuda_baseline_peak_memory_mb")
        if not isinstance(npu_perf_peak_memory_mb, (int, float)) or isinstance(npu_perf_peak_memory_mb, bool) or not _metric_close(float(npu_perf.get("peak_memory_mb", -1) or -1), float(npu_perf_peak_memory_mb)):
            return False, "business_summary.best_result.npu_perf_peak_memory_mb 与 npu_perf 工件不一致"
        if not isinstance(cuda_peak_memory_mb, (int, float)) or isinstance(cuda_peak_memory_mb, bool) or not _metric_close(float(cuda_baseline.get("peak_memory_mb", -1) or -1), float(cuda_peak_memory_mb)):
            return False, "business_summary.best_result.cuda_baseline_peak_memory_mb 与 cuda_baseline 工件不一致"

        npu_perf_tp_name, npu_perf_tp_value = _detect_business_throughput(metric_entries.get("npu_perf", {}))
        cuda_tp_name, cuda_tp_value = _detect_business_throughput(metric_entries.get("cuda_baseline", {}))
        if str(best.get("npu_perf_throughput_metric_name") or "").strip() != npu_perf_tp_name:
            return False, "business_summary.best_result.npu_perf_throughput_metric_name 与 npu_perf 工件不一致"
        if str(best.get("cuda_baseline_throughput_metric_name") or "").strip() != cuda_tp_name:
            return False, "business_summary.best_result.cuda_baseline_throughput_metric_name 与 cuda_baseline 工件不一致"
        best_npu_perf_tp = best.get("npu_perf_throughput_metric_value")
        best_cuda_tp = best.get("cuda_baseline_throughput_metric_value")
        if not isinstance(best_npu_perf_tp, (int, float)) or isinstance(best_npu_perf_tp, bool) or npu_perf_tp_value is None or not _metric_close(float(npu_perf_tp_value), float(best_npu_perf_tp)):
            return False, "business_summary.best_result.npu_perf_throughput_metric_value 与 npu_perf 工件不一致"
        if not isinstance(best_cuda_tp, (int, float)) or isinstance(best_cuda_tp, bool) or cuda_tp_value is None or not _metric_close(float(cuda_tp_value), float(best_cuda_tp)):
            return False, "business_summary.best_result.cuda_baseline_throughput_metric_value 与 cuda_baseline 工件不一致"

        evidence = data.get("comparison_evidence") or {}
        if isinstance(evidence, dict):
            if str(evidence.get("npu_baseline_device_model") or "").strip() != str(npu_baseline.get("device_model") or "").strip():
                return False, "business_summary.comparison_evidence.npu_baseline_device_model 与 npu_baseline 工件不一致"
            if str(evidence.get("npu_perf_device_model") or "").strip() != str(npu_perf.get("device_model") or "").strip():
                return False, "business_summary.comparison_evidence.npu_perf_device_model 与 npu_perf 工件不一致"
            if str(evidence.get("cuda_baseline_device_model") or "").strip() != str(cuda_baseline.get("device_model") or "").strip():
                return False, "business_summary.comparison_evidence.cuda_baseline_device_model 与 cuda_baseline 工件不一致"

            for evidence_key, role_name in (
                ("npu_baseline_peak_memory_mb", "npu_baseline"),
                ("npu_perf_peak_memory_mb", "npu_perf"),
                ("cuda_baseline_peak_memory_mb", "cuda_baseline"),
            ):
                evidence_value = evidence.get(evidence_key)
                metric_value = metric_entries.get(role_name, {}).get("peak_memory_mb")
                if not isinstance(evidence_value, (int, float)) or isinstance(evidence_value, bool) or not isinstance(metric_value, (int, float)) or isinstance(metric_value, bool) or not _metric_close(float(metric_value), float(evidence_value)):
                    return False, f"business_summary.comparison_evidence.{evidence_key} 与业务测评工件不一致"

            for name_key, value_key, role_name in (
                ("npu_baseline_throughput_metric_name", "npu_baseline_throughput_metric_value", "npu_baseline"),
                ("npu_perf_throughput_metric_name", "npu_perf_throughput_metric_value", "npu_perf"),
                ("cuda_baseline_throughput_metric_name", "cuda_baseline_throughput_metric_value", "cuda_baseline"),
            ):
                detected_name, detected_value = _detect_business_throughput(metric_entries.get(role_name, {}))
                if str(evidence.get(name_key) or "").strip() != detected_name:
                    return False, f"business_summary.comparison_evidence.{name_key} 与业务测评工件不一致"
                evidence_value = evidence.get(value_key)
                if not isinstance(evidence_value, (int, float)) or isinstance(evidence_value, bool) or detected_value is None or not _metric_close(float(detected_value), float(evidence_value)):
                    return False, f"business_summary.comparison_evidence.{value_key} 与业务测评工件不一致"
            for name_key, value_key, role_name in (
                ("npu_baseline_quality_metric_name", "npu_baseline_quality_metric_value", "npu_baseline"),
                ("npu_perf_quality_metric_name", "npu_perf_quality_metric_value", "npu_perf"),
                ("cuda_baseline_quality_metric_name", "cuda_baseline_quality_metric_value", "cuda_baseline"),
            ):
                detected_name, detected_value = _detect_business_quality(metric_entries.get(role_name, {}))
                if str(evidence.get(name_key) or "").strip() != detected_name:
                    return False, f"business_summary.comparison_evidence.{name_key} 与业务测评工件不一致"
                evidence_value = evidence.get(value_key)
                if detected_value is None:
                    if evidence_value is not None:
                        return False, f"business_summary.comparison_evidence.{value_key} 与业务测评工件不一致"
                elif not isinstance(evidence_value, (int, float)) or isinstance(evidence_value, bool) or not _metric_close(float(detected_value), float(evidence_value)):
                    return False, f"business_summary.comparison_evidence.{value_key} 与业务测评工件不一致"
    return True, ""


def _validate_completed_optimization_notes(notes_str: str) -> tuple[bool, str]:
    """completed 状态下，optimization_notes 必须包含真实 pretrained 结果。"""
    try:
        data = json.loads(notes_str)
    except json.JSONDecodeError as e:
        return False, f"optimization_notes 非法 JSON: {e}"
    if not isinstance(data, dict):
        return False, "optimization_notes 必须是 JSON object"

    results = data.get("results")
    best_result = data.get("best_result")
    contract_v = data.get("measurement_contract_version")
    if not isinstance(results, list) or not results:
        return False, "optimization_notes.results 不能为空"
    if not isinstance(best_result, dict):
        return False, "optimization_notes.best_result 必须是 object"
    if not isinstance(contract_v, (int, float)) or isinstance(contract_v, bool) or int(contract_v) < 3:
        return False, "optimization completed 的 measurement_contract_version 必须 >= 3"

    pretrained_results = [r for r in results if isinstance(r, dict) and (r.get("mode") or "").strip().lower() == "pretrained"]
    if not pretrained_results:
        return False, "optimization completed 必须至少包含一条 pretrained 结果，config-only 结果不得标记 completed"
    if (best_result.get("mode") or "").strip().lower() != "pretrained":
        return False, "optimization completed 的 best_result.mode 必须为 pretrained"
    explicit_completion_kind = str(best_result.get("optimization_kind") or data.get("optimization_kind") or "").strip().lower()
    completion_kind = _infer_optimization_completion_kind(data, best_result)
    for field_name in ("dataset", "dtype", "output_type", "baseline_artifact", "perf_artifact"):
        field_value = best_result.get(field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            return False, f"optimization completed 的 best_result.{field_name} 必须是非空字符串"

    baseline_latency = best_result.get("baseline_latency_s")
    perf_latency = best_result.get("perf_latency_s")
    baseline_wall_clock = best_result.get("baseline_wall_clock_s")
    perf_wall_clock = best_result.get("perf_wall_clock_s")
    wall_clock_source = (best_result.get("wall_clock_source") or "").strip()
    warmup_policy = (best_result.get("warmup_policy") or "").strip()
    baseline_warmup_iterations = best_result.get("baseline_warmup_iterations")
    perf_warmup_iterations = best_result.get("perf_warmup_iterations")
    speedup_ratio = best_result.get("speedup_ratio")
    num_samples = best_result.get("num_samples")
    if not isinstance(baseline_latency, (int, float)):
        return False, "optimization completed 的 best_result.baseline_latency_s 必须为数值"
    if not isinstance(perf_latency, (int, float)):
        return False, "optimization completed 的 best_result.perf_latency_s 必须为数值"
    if not isinstance(baseline_wall_clock, (int, float)) or float(baseline_wall_clock) <= 0:
        return False, "optimization completed 的 best_result.baseline_wall_clock_s 必须为正数"
    if not isinstance(perf_wall_clock, (int, float)) or float(perf_wall_clock) <= 0:
        return False, "optimization completed 的 best_result.perf_wall_clock_s 必须为正数"
    if not isinstance(speedup_ratio, (int, float)):
        return False, "optimization completed 的 best_result.speedup_ratio 必须为数值"
    if not isinstance(num_samples, (int, float)) or isinstance(num_samples, bool):
        return False, "optimization completed 的 best_result.num_samples 必须为数值"
    if float(num_samples) < _MIN_COMPLETED_METRIC_SAMPLES:
        return False, f"optimization completed 的 best_result.num_samples 必须 >= {_MIN_COMPLETED_METRIC_SAMPLES}"
    if wall_clock_source not in {"artifact_timestamps", "artifact_explicit_field"}:
        return False, "optimization completed 的 best_result.wall_clock_source 必须为 artifact_timestamps 或 artifact_explicit_field"
    if warmup_policy != "symmetric":
        return False, "optimization completed 的 best_result.warmup_policy 必须为 symmetric"
    if not isinstance(baseline_warmup_iterations, (int, float)) or isinstance(baseline_warmup_iterations, bool) or float(baseline_warmup_iterations) < 0:
        return False, "optimization completed 的 best_result.baseline_warmup_iterations 必须为 >=0 的数值"
    if not isinstance(perf_warmup_iterations, (int, float)) or isinstance(perf_warmup_iterations, bool) or float(perf_warmup_iterations) < 0:
        return False, "optimization completed 的 best_result.perf_warmup_iterations 必须为 >=0 的数值"
    if not _metric_close(float(baseline_warmup_iterations), float(perf_warmup_iterations)):
        return False, "optimization completed 的 baseline/perf warmup_iterations 必须一致"
    expected_speedup = float(baseline_wall_clock) / float(perf_wall_clock)
    if not _metric_close(expected_speedup, float(speedup_ratio)):
        return False, "optimization completed 的 best_result.speedup_ratio 必须按 baseline_wall_clock_s / perf_wall_clock_s 计算"
    speedup_value = float(speedup_ratio)
    is_speedup_equal_one = abs(speedup_value - 1.0) <= 1e-6
    if speedup_value < 1.0 - 1e-6:
        return False, "optimization completed 的 best_result.speedup_ratio 必须 >= 1；<1 的记录必须回退为 pending"
    if is_speedup_equal_one:
        if completion_kind != "runtime_only":
            return False, "仅 runtime_only completed 允许 speedup_ratio=1.0；fusion/hybrid 必须大于 1"
        if explicit_completion_kind != "runtime_only":
            return False, "speedup_ratio=1.0 的 completed 必须显式声明 optimization_kind=runtime_only"
    if isinstance(speedup_ratio, (int, float)) and speedup_ratio >= 3.0:
        comparison_method = (best_result.get("comparison_method") or "").strip()
        precision_method = (best_result.get("precision_method") or "").strip()
        validation_note = (best_result.get("validation_note") or "").strip()
        comparison_scope = (best_result.get("comparison_scope") or "").strip()
        steady_state_baseline = best_result.get("steady_state_baseline_latency_s")
        steady_state_perf = best_result.get("steady_state_perf_latency_s")

        if "self_baseline" in precision_method:
            return False, "speedup_ratio >= 3x 时禁止使用 self_baseline 系 precision_method 作为 completed 依据"
        if comparison_method != "independent_baseline_artifact":
            return False, "speedup_ratio >= 3x 时必须声明 comparison_method=independent_baseline_artifact"
        if comparison_scope not in _VALID_COMPARISON_SCOPES:
            return False, "speedup_ratio >= 3x 时必须声明 comparison_scope=cold_start|steady_state|mixed"
        if not validation_note:
            return False, "speedup_ratio >= 3x 时 best_result.validation_note 不能为空，必须说明高倍提速核查结论"
        if not isinstance(steady_state_baseline, (int, float)) or float(steady_state_baseline) <= 0:
            return False, "speedup_ratio >= 3x 时 best_result.steady_state_baseline_latency_s 必须为正数"
        if not isinstance(steady_state_perf, (int, float)) or float(steady_state_perf) <= 0:
            return False, "speedup_ratio >= 3x 时 best_result.steady_state_perf_latency_s 必须为正数"
    validation_note = str(best_result.get("validation_note") or "").strip()
    if _contains_measurement_red_flag(validation_note):
        return False, "optimization completed 的 validation_note 暴露了不可靠测量口径，必须回退为 pending"
    if explicit_completion_kind == "runtime_only":
        runtime_ok, runtime_err = _validate_runtime_only_completed_metadata(
            data,
            best_result,
            require_no_code_change=is_speedup_equal_one,
        )
        if not runtime_ok:
            return False, runtime_err
    return True, ""


def _metric_close(expected: float, actual: float) -> bool:
    abs_tol = max(1e-3, abs(expected) * 0.02)
    return abs(expected - actual) <= abs_tol


def _requires_per_sample_wall_clock_alignment(output_type: str) -> bool:
    lowered = (output_type or "").strip().lower()
    if not lowered:
        return False
    return not any(token in lowered for token in ("generated_text", "transcription", "qa_answer", "generated_action"))


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


def _normalize_optimization_items(raw_items) -> set[str]:
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


def _infer_optimization_completion_kind(data: dict, result: dict) -> str:
    explicit = str(result.get("optimization_kind") or data.get("optimization_kind") or "").strip().lower()
    if explicit in _VALID_OPTIMIZATION_COMPLETION_KINDS:
        return explicit
    items = _normalize_optimization_items(result.get("optimization_items"))
    has_runtime = _has_runtime_only_item(items)
    has_fusion = _has_fusion_item(items)
    if has_runtime and has_fusion:
        return "hybrid"
    if has_runtime:
        return "runtime_only"
    return "fusion"


def _extract_selected_npus(payload: dict) -> list[str]:
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


def _contains_no_model_code_change_note(data: dict, best_result: dict) -> bool:
    texts: list[str] = []
    for raw in (
        best_result.get("validation_note"),
        best_result.get("optimizations"),
        data.get("optimizations"),
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


def _validate_runtime_only_completed_metadata(data: dict, best_result: dict, *, require_no_code_change: bool = False) -> tuple[bool, str]:
    merged_items = _normalize_optimization_items(data.get("optimization_items")) | _normalize_optimization_items(best_result.get("optimization_items"))
    if not _has_runtime_only_item(merged_items):
        return False, "runtime_only completed 必须在 optimization_items 中显式记录 warmup / TASK_QUEUE_ENABLE 之类的 runtime 优化项"
    if _has_fusion_item(merged_items):
        return False, "runtime_only completed 的 optimization_items 不得包含 npu_* 融合算子；混合路径请标记为 hybrid"
    selected_npus = _extract_selected_npus(best_result) or _extract_selected_npus(data)
    if not selected_npus:
        return False, "runtime_only completed 必须记录 selected_npu 或 selected_npus，确保 baseline/perf 设备拓扑可追溯"
    device_topology = str(best_result.get("device_topology") or data.get("device_topology") or "").strip()
    if not device_topology:
        return False, "runtime_only completed 必须记录 device_topology"
    parallel_mode = str(best_result.get("parallel_mode") or data.get("parallel_mode") or "").strip()
    if not parallel_mode:
        return False, "runtime_only completed 必须记录 parallel_mode"
    if require_no_code_change:
        code_modified = best_result.get("code_modified", data.get("code_modified"))
        if code_modified is not False:
            return False, "runtime_only 且 speedup_ratio=1.0 的 completed 必须声明 code_modified=false（模型代码无更改）"
        code_change_attempts = best_result.get("code_change_attempts", data.get("code_change_attempts"))
        if not isinstance(code_change_attempts, (int, float)) or isinstance(code_change_attempts, bool) or float(code_change_attempts) < 2:
            return False, "runtime_only 且 speedup_ratio=1.0 的 completed 必须记录 code_change_attempts>=2（模型代码多次尝试无果）"
        if not _contains_no_model_code_change_note(data, best_result):
            return False, "runtime_only 且 speedup_ratio=1.0 的 completed 必须在 validation_note/optimizations 写明模型代码无更改"
    return True, ""


def _parse_json_object_notes(notes: str, *, label: str) -> tuple[bool, dict[str, Any] | None, str]:
    notes_raw = (notes or "").strip()
    if not notes_raw:
        return False, None, f"{label} 必须是非空 JSON object"
    try:
        payload = json.loads(notes_raw)
    except json.JSONDecodeError as e:
        return False, None, f"{label} 必须是合法 JSON: {e}"
    if not isinstance(payload, dict):
        return False, None, f"{label} 必须是 JSON object"
    return True, cast(dict[str, Any], payload), ""


def _validate_runtime_only_failure_evidence(payload: dict[str, Any], *, status: str) -> tuple[bool, str]:
    if payload.get("runtime_only_attempted") is not True:
        return False, f"optimization_status={status} 时必须明确记录 runtime_only_attempted=true"
    speedup = payload.get("runtime_only_speedup_ratio")
    if not isinstance(speedup, (int, float)) or isinstance(speedup, bool):
        return False, f"optimization_status={status} 时必须记录数值型 runtime_only_speedup_ratio"
    if float(speedup) > 1.0 + 1e-6:
        return False, f"runtime_only_speedup_ratio={speedup} 表明已有真实提速，应改记 completed 而不是 {status}"
    selected_npus = _extract_selected_npus(payload)
    if not selected_npus:
        return False, f"optimization_status={status} 时必须记录 runtime_only 使用的 selected_npu 或 selected_npus"
    return True, ""


def _validate_optimization_non_completed_status(new_status: str, notes: str) -> tuple[bool, str]:
    if new_status == "in_progress":
        return True, ""
    notes_raw = (notes or "").strip()
    if new_status == "pending" and not notes_raw:
        return True, ""
    notes_ok, payload, notes_err = _parse_json_object_notes(notes, label=f"optimization_status={new_status} 的 notes")
    if not notes_ok or payload is None:
        return False, notes_err
    for field_name in _OPTIMIZATION_REASON_REQUIRED_FIELDS:
        if field_name not in payload:
            return False, f"optimization_status={new_status} 的 notes 缺少字段 {field_name}"
    reason_code = str(payload.get("reason_code") or "").strip()
    recommended_action = str(payload.get("recommended_action") or "").strip()
    evidence = str(payload.get("evidence") or "").strip()
    next_step = str(payload.get("next_step") or "").strip()
    retryable = payload.get("retryable")
    if not reason_code:
        return False, f"optimization_status={new_status} 的 reason_code 不能为空"
    if not isinstance(retryable, bool):
        return False, f"optimization_status={new_status} 的 retryable 必须是布尔值"
    if not recommended_action or not evidence or not next_step:
        return False, f"optimization_status={new_status} 的 recommended_action/evidence/next_step 必须为非空字符串"

    if new_status == "pending":
        if reason_code in _OPTIMIZATION_SKIPPED_REASON_CODES or reason_code in _OPTIMIZATION_NOT_APPLICABLE_REASON_CODES:
            return False, f"reason_code={reason_code} 已是非重试终态，应使用 skipped/not_applicable 而不是 pending"
        if reason_code not in _OPTIMIZATION_PENDING_REASON_CODES:
            return False, f"pending 只允许可重试 reason_code，当前: {reason_code}"
        if retryable is not True:
            return False, "pending 只允许 retryable=true 的原因"
        return True, ""

    if new_status == "skipped":
        if reason_code not in _OPTIMIZATION_SKIPPED_REASON_CODES:
            return False, f"reason_code={reason_code} 属于可重试或不适用问题，应改记 pending/not_applicable"
        if retryable is not False:
            return False, "skipped 只允许 retryable=false 的原因"
        runtime_ok, runtime_err = _validate_runtime_only_failure_evidence(payload, status=new_status)
        if not runtime_ok:
            return False, runtime_err
        return True, ""

    if new_status == "not_applicable":
        if reason_code not in _OPTIMIZATION_NOT_APPLICABLE_REASON_CODES:
            return False, f"reason_code={reason_code} 不属于 not_applicable；请改记 pending 或 skipped"
        if retryable is not False:
            return False, "not_applicable 只允许 retryable=false 的原因"
        runtime_ok, runtime_err = _validate_runtime_only_failure_evidence(payload, status=new_status)
        if not runtime_ok:
            return False, runtime_err
        return True, ""

    return True, ""


def _detect_business_throughput(metric: dict) -> tuple[str, float | None]:
    for key in ("throughput_qps", "throughput", "samples_per_second", "qps", "items_per_second"):
        value = metric.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > 0:
            return key, float(value)
    latency_s = metric.get("latency_s")
    if isinstance(latency_s, (int, float)) and not isinstance(latency_s, bool) and float(latency_s) > 0:
        return "derived_from_latency_qps", 1.0 / float(latency_s)
    return "throughput", None


def _lookup_business_numeric_metric(source: object, metric_name: object) -> float | None:
    metric_key = str(metric_name or "").strip()
    if not metric_key or not isinstance(source, dict):
        return None
    value = source.get(metric_key)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    lowered_metric_key = metric_key.lower()
    for key, candidate in source.items():
        if str(key or "").strip().lower() == lowered_metric_key and isinstance(candidate, (int, float)) and not isinstance(candidate, bool) and math.isfinite(float(candidate)):
            return float(candidate)
    return None


def _normalize_business_quality_metric_name(metric_name: object) -> str:
    return str(metric_name or "").strip().lower()


def _is_bounded_business_quality_metric(metric_name: object) -> bool:
    return _normalize_business_quality_metric_name(metric_name) in _BUSINESS_BOUNDED_QUALITY_METRIC_KEYS


def _is_alignment_business_quality_metric(metric_name: object) -> bool:
    return _normalize_business_quality_metric_name(metric_name) in _BUSINESS_STRICT_ALIGNMENT_QUALITY_KEYS


def _business_quality_within_unit_range(metric_name: object, quality_value: object) -> bool:
    if quality_value is None or not _is_bounded_business_quality_metric(metric_name):
        return True
    return isinstance(quality_value, (int, float)) and not isinstance(quality_value, bool) and math.isfinite(float(quality_value)) and float(quality_value) <= 1.0


def _same_business_hardware_quality_tolerance(num_samples: float) -> float:
    return max(0.03, 3.0 / max(num_samples, 1.0))


def _cross_business_device_quality_tolerance(num_samples: float) -> float:
    return max(0.08, 6.0 / max(num_samples, 1.0))


def _bounded_business_quality_delta_tolerance(sample_counts: list[float]) -> float:
    if not sample_counts:
        return _MAX_COMPLETED_BOUNDED_BUSINESS_QUALITY_DELTA
    positive_counts = [count for count in sample_counts if count > 0]
    if not positive_counts:
        return _MAX_COMPLETED_BOUNDED_BUSINESS_QUALITY_DELTA
    # 对 accuracy / exact_match 这类离散指标，允许三样本粒度内的轻微抖动；
    # 速度比仍由独立硬门禁把关，精度只负责拦截明显口径异常。
    return max(_MAX_COMPLETED_BOUNDED_BUSINESS_QUALITY_DELTA, 0.03, 3.0 / min(positive_counts))


def _detect_business_quality(metric: dict) -> tuple[str, float | None]:
    compare = metric.get("output_compare")
    primary_metric = str(metric.get("primary_metric") or "").strip()
    if isinstance(compare, dict):
        primary_value = _lookup_business_numeric_metric(compare, primary_metric)
        if primary_metric and primary_value is not None:
            return primary_metric, primary_value
        for key in ("avg_cosine_similarity", "logits_avg_cosine_similarity", "cosine_similarity"):
            value = compare.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return "cosine_similarity", float(value)
        for key in ("exact_match", "f1", "accuracy", "top1_accuracy", "rougeL", "ndcg_at_10", "mAP", "map50", "match_rate", "text_match_rate", "wer", "mrr", "perplexity"):
            value = compare.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return key, float(value)
    primary_value = _lookup_business_numeric_metric(metric, primary_metric)
    if primary_metric and primary_value is not None:
        return primary_metric, primary_value
    for key in ("cosine_similarity", "exact_match", "f1", "accuracy", "top1_accuracy", "rougeL", "ndcg_at_10", "mAP", "map50", "match_rate", "text_match_rate", "wer", "mrr", "perplexity"):
        value = metric.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return key, float(value)
    return "quality_metric", None


def _validate_business_quality_alignment(metric_entries: dict[str, dict]) -> tuple[bool, str]:
    role_values: dict[str, float] = {}
    role_metric_names: dict[str, str] = {}
    sample_counts: list[float] = []
    for role in ("npu_baseline", "npu_perf", "cuda_baseline"):
        metric = metric_entries.get(role)
        if not isinstance(metric, dict):
            return True, ""
        quality_name, quality_value = _detect_business_quality(metric)
        role_metric_names[role] = str(quality_name or "")
        if not isinstance(quality_value, (int, float)) or isinstance(quality_value, bool) or not math.isfinite(float(quality_value)):
            return True, ""
        role_values[role] = float(quality_value)
        num_samples = metric.get("num_samples")
        if isinstance(num_samples, (int, float)) and not isinstance(num_samples, bool) and math.isfinite(float(num_samples)) and float(num_samples) > 0:
            sample_counts.append(float(num_samples))
    if len(role_values) != 3 or not sample_counts:
        return True, ""

    normalized_names = {_normalize_business_quality_metric_name(name) for name in role_metric_names.values()}
    normalized_names.discard("")
    if len(normalized_names) != 1:
        if normalized_names:
            return False, ("业务测评三路 quality_metric_name 不一致：" + ", ".join(f"{role}={role_metric_names.get(role) or 'empty'}" for role in ("npu_baseline", "npu_perf", "cuda_baseline")))
        return True, ""
    metric_name = next(iter(normalized_names), "")
    quality_span = max(role_values.values()) - min(role_values.values())
    bounded_delta_tolerance = _bounded_business_quality_delta_tolerance(sample_counts)
    if _is_bounded_business_quality_metric(metric_name) and quality_span > bounded_delta_tolerance:
        return False, (f"{metric_name} 三路结果差异过大：Δ={quality_span:.6g} > {bounded_delta_tolerance:.6g} (npu_baseline={role_values['npu_baseline']:.6g}, npu_perf={role_values['npu_perf']:.6g}, cuda_baseline={role_values['cuda_baseline']:.6g})")
    if not _is_alignment_business_quality_metric(metric_name):
        return True, ""

    same_tol = _same_business_hardware_quality_tolerance(min(sample_counts))
    cross_tol = _cross_business_device_quality_tolerance(min(sample_counts))
    npu_baseline_quality = role_values["npu_baseline"]
    npu_perf_quality = role_values["npu_perf"]
    cuda_quality = role_values["cuda_baseline"]
    if max(role_values.values()) == 0.0:
        return False, f"{metric_name} 三路结果全部为 0.0；疑似 evaluator / 标签归一化 / 数据集画像异常"
    if min(role_values.values()) == 0.0 and max(role_values.values()) >= same_tol:
        low_roles = sorted(role for role, value in role_values.items() if value == 0.0)
        high_roles = sorted(role for role, value in role_values.items() if value >= same_tol)
        return False, f"{metric_name} 出现 0 分塌陷：{', '.join(low_roles)}=0.0，但 {', '.join(high_roles)} 明显更高"
    if abs(npu_baseline_quality - npu_perf_quality) > same_tol:
        return False, f"{metric_name} 在 npu_baseline/npu_perf 间漂移过大：|{npu_baseline_quality:.6g}-{npu_perf_quality:.6g}|>{same_tol:.6g}"
    if abs(cuda_quality - npu_baseline_quality) > cross_tol and abs(cuda_quality - npu_perf_quality) > cross_tol:
        return False, f"{metric_name} 的 CUDA/NPU 三路结果漂移过大：cuda={cuda_quality:.6g}, npu_baseline={npu_baseline_quality:.6g}, npu_perf={npu_perf_quality:.6g}"
    return True, ""


def _normalize_rate(value) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    rate = float(value)
    if rate > 1.0 and rate <= 100.0:
        rate = rate / 100.0
    return rate


def _normalize_pct(value) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    pct = float(value)
    if 0.0 <= pct <= 1.0:
        pct = pct * 100.0
    return pct


def _validate_metric_num_samples(metric: dict, metric_file: Path, label: str) -> tuple[bool, str]:
    """校验工件中的 num_samples，completed 记录至少需要 50 个样本。"""
    num_samples = metric.get("num_samples")
    if not isinstance(num_samples, (int, float)) or isinstance(num_samples, bool):
        return False, f"{label} 工件 {metric_file.name} 缺少数值型 num_samples；completed 前必须至少测试 {_MIN_COMPLETED_METRIC_SAMPLES} 个样本"
    if float(num_samples) < _MIN_COMPLETED_METRIC_SAMPLES:
        return False, f"{label} 工件 {metric_file.name} 的 num_samples={num_samples}，completed 前必须至少测试 {_MIN_COMPLETED_METRIC_SAMPLES} 个样本"
    return True, ""


def _validate_required_metric_fields(metric: dict, metric_file: Path, label: str, required_fields: tuple[str, ...]) -> tuple[bool, str]:
    """校验 benchmark_metrics 工件必要字段完整。"""
    string_fields = {"mode", "dataset", "dtype", "output_type", "device", "start_time", "end_time"}
    numeric_fields = {"latency_s", "num_samples"}
    for field in required_fields:
        if field not in metric:
            return False, f"{label} 工件 {metric_file.name} 缺少字段 {field}"
        value = metric.get(field)
        if field in string_fields and (not isinstance(value, str) or not value.strip()):
            return False, f"{label} 工件 {metric_file.name} 的 {field} 必须是非空字符串"
        if field in numeric_fields and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            return False, f"{label} 工件 {metric_file.name} 的 {field} 必须是数值"
    if metric.get("mode") not in {"pretrained", "config"}:
        return False, f"{label} 工件 {metric_file.name} 的 mode 必须为 pretrained/config"
    if metric.get("dtype") not in {"fp32", "fp16", "bf16"}:
        return False, f"{label} 工件 {metric_file.name} 的 dtype 必须为 fp32/fp16/bf16"
    return True, ""


def _parse_metric_timestamp(raw_value, metric_file: Path, label: str, field_name: str) -> tuple[datetime | None, str]:
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None, f"{label} 工件 {metric_file.name} 的 {field_name} 必须是非空 ISO 时间字符串"
    try:
        return datetime.fromisoformat(raw_value), ""
    except ValueError:
        return None, f"{label} 工件 {metric_file.name} 的 {field_name} 不是合法 ISO 时间: {raw_value}"


def _extract_metric_wall_clock_s(metric: dict, metric_file: Path, label: str) -> tuple[float | None, str]:
    """优先读显式 wall-clock 字段，否则用 start_time/end_time 推导整轮耗时。"""
    for field_name in ("wall_clock_s", "total_time_s", "duration_s", "elapsed_s"):
        value = metric.get(field_name)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None, f"{label} 工件 {metric_file.name} 的 {field_name} 必须是数值"
        if float(value) <= 0:
            return None, f"{label} 工件 {metric_file.name} 的 {field_name} 必须为正数"
        return float(value), ""

    start_dt, start_err = _parse_metric_timestamp(metric.get("start_time"), metric_file, label, "start_time")
    if start_err:
        return None, start_err
    end_dt, end_err = _parse_metric_timestamp(metric.get("end_time"), metric_file, label, "end_time")
    if end_err:
        return None, end_err
    if start_dt is None or end_dt is None:
        return None, f"{label} 工件 {metric_file.name} 缺少可推导 wall-clock 的时间戳"
    duration = (end_dt - start_dt).total_seconds()
    if duration <= 0:
        return None, f"{label} 工件 {metric_file.name} 的 wall-clock 必须为正数"
    return duration, ""


def _detect_metric_wall_clock_source(metric: dict, metric_file: Path, label: str) -> tuple[str | None, str]:
    for field_name in ("wall_clock_s", "total_time_s", "duration_s", "elapsed_s"):
        value = metric.get(field_name)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0:
            return None, f"{label} 工件 {metric_file.name} 的 {field_name} 必须为正数"
        return "artifact_explicit_field", ""

    start_dt, start_err = _parse_metric_timestamp(metric.get("start_time"), metric_file, label, "start_time")
    if start_err:
        return None, start_err
    end_dt, end_err = _parse_metric_timestamp(metric.get("end_time"), metric_file, label, "end_time")
    if end_err:
        return None, end_err
    if start_dt is None or end_dt is None or (end_dt - start_dt).total_seconds() <= 0:
        return None, f"{label} 工件 {metric_file.name} 缺少可用 wall-clock 来源"
    return "artifact_timestamps", ""


def _validate_metric_metadata_health(metric: dict, metric_file: Path, label: str) -> tuple[bool, str]:
    """校验工件元数据是否健康。"""
    latency = metric.get("latency_s")
    if not isinstance(latency, (int, float)) or isinstance(latency, bool) or float(latency) <= 0:
        return False, f"{label} 工件 {metric_file.name} 的 latency_s 必须为正数"

    start_dt, start_err = _parse_metric_timestamp(metric.get("start_time"), metric_file, label, "start_time")
    if start_err:
        return False, start_err
    end_dt, end_err = _parse_metric_timestamp(metric.get("end_time"), metric_file, label, "end_time")
    if end_err:
        return False, end_err
    if start_dt and end_dt and end_dt < start_dt:
        return False, f"{label} 工件 {metric_file.name} 的 end_time 早于 start_time"

    for field_name in ("ttft_ms", "tpot_ms"):
        raw = metric.get(field_name)
        if raw is not None and (not isinstance(raw, (int, float)) or isinstance(raw, bool)):
            return False, f"{label} 工件 {metric_file.name} 的 {field_name} 必须是数值或 null"
    ttft_ms = metric.get("ttft_ms")
    if isinstance(ttft_ms, (int, float)) and float(ttft_ms) > float(latency) * 1000 + 1e-3:
        return False, f"{label} 工件 {metric_file.name} 的 latency_s 小于 ttft_ms，对应元数据不可信"
    tpot_ms = metric.get("tpot_ms")
    if isinstance(tpot_ms, (int, float)) and float(tpot_ms) < 0:
        return False, f"{label} 工件 {metric_file.name} 的 tpot_ms 不能为负数"
    return True, ""


def _read_metric_artifact(adapt_dir: Path, artifact_name: str, label: str) -> tuple[bool, Path | None, dict | None, str]:
    """读取并解析 adaptation 下的 benchmark_metrics 工件。"""
    if not isinstance(artifact_name, str) or not artifact_name.strip():
        return False, None, None, f"{label} artifact 不能为空"
    artifact_path = (adapt_dir / artifact_name).resolve()
    adapt_root = adapt_dir.resolve()
    if adapt_root not in artifact_path.parents:
        return False, None, None, f"{label} artifact 路径越界: {artifact_name}"
    if not artifact_path.is_file():
        return False, None, None, f"{label} artifact 不存在: {artifact_name}"
    try:
        metric = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, None, None, f"读取 {label} artifact 失败: {artifact_name}: {e}"
    if not isinstance(metric, dict):
        return False, None, None, f"{label} artifact {artifact_name} 必须是 JSON object"
    return True, artifact_path, metric, ""


def _validate_metric_identity(
    metric: dict,
    metric_file: Path,
    label: str,
    *,
    expected_mode: str,
    expected_dataset: str,
    expected_dtype: str,
    expected_output_type: str,
) -> tuple[bool, str]:
    """校验 baseline/perf 工件的 mode/dataset/dtype/output_type 与 notes 保持一致。"""
    expected_map = {
        "mode": expected_mode,
        "dataset": expected_dataset,
        "dtype": expected_dtype,
        "output_type": expected_output_type,
    }
    for field_name, expected_value in expected_map.items():
        if not isinstance(expected_value, str) or not expected_value.strip():
            return False, f"best_result 缺少有效 {field_name}，无法校验 {label} 工件"
        actual = metric.get(field_name)
        if not isinstance(actual, str) or not actual.strip():
            return False, f"{label} 工件 {metric_file.name} 缺少非空 {field_name}"
        if actual.strip() != expected_value.strip():
            return False, f"{label} 工件 {metric_file.name} 的 {field_name}={actual!r} 与 best_result.{field_name}={expected_value!r} 不一致"
    return True, ""


def _first_numeric(sources: list[dict], keys: tuple[str, ...]) -> float | None:
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            value = source.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


def _validate_precision_evidence(best_result: dict, baseline_metric: dict, perf_metric: dict, perf_file: Path) -> tuple[bool, str]:
    """校验 optimization completed 的精度证据与 compare 样本数。"""
    output_compare = perf_metric.get("output_compare")
    if output_compare is None:
        output_compare = {}
    if not isinstance(output_compare, dict):
        return False, f"optimization perf 工件 {perf_file.name} 的 output_compare 必须是 object"
    if output_compare.get("error"):
        return False, f"optimization perf 工件 {perf_file.name} 的 output_compare 报错: {output_compare.get('error')}"

    baseline_compare_samples = _first_numeric([output_compare], ("baseline_samples", "reference_samples", "compare_baseline_samples"))
    perf_compare_samples = _first_numeric([output_compare], ("ascend_samples", "perf_samples", "compare_perf_samples", "npu_samples"))
    if baseline_compare_samples is None:
        baseline_compare_samples = _first_numeric([baseline_metric], ("num_samples",))
    if perf_compare_samples is None:
        perf_compare_samples = _first_numeric([perf_metric], ("num_samples",))
    if baseline_compare_samples is None or perf_compare_samples is None:
        return False, f"optimization perf 工件 {perf_file.name} 缺少 compare 样本数证据"
    if baseline_compare_samples < _MIN_COMPLETED_METRIC_SAMPLES or perf_compare_samples < _MIN_COMPLETED_METRIC_SAMPLES:
        return False, (f"optimization perf 工件 {perf_file.name} 的 compare 样本数不足：baseline={baseline_compare_samples}, perf={perf_compare_samples}；completed 前必须至少测试 {_MIN_COMPLETED_METRIC_SAMPLES} 个可比样本")

    sources = [output_compare, perf_metric, best_result]
    exact_rate = _normalize_rate(_first_numeric(sources, ("text_match_rate", "match_rate")))
    cosine = _first_numeric(sources, ("cosine_similarity", "avg_cosine_similarity", "logits_avg_cosine_similarity"))
    ppl_diff_pct = _normalize_pct(_first_numeric(sources, ("ppl_avg_rel_diff_pct", "ppl_avg_rel_diff")))
    max_abs_error = _first_numeric(sources, ("max_abs_error", "logits_max_abs_error"))
    output_type = str(best_result.get("output_type") or "").strip().lower()

    exact_family = any(token in output_type for token in ("generated_text", "transcription", "class_label", "class_prediction", "qa_answer", "predicted_token", "generated_action"))
    vector_family = any(token in output_type for token in ("embedding", "logit", "hidden_state", "score", "similarity", "latent"))
    generated_image_family = any(token in output_type for token in ("generated_image",))

    exact_ok = exact_rate is not None and exact_rate >= 1.0 - 1e-6
    cosine_ok = cosine is not None and cosine >= 0.999
    ppl_ok = ppl_diff_pct is not None and ppl_diff_pct < 5.0
    max_abs_ok = max_abs_error is None or abs(max_abs_error) < 1e-3

    if exact_family:
        if not (exact_ok or cosine_ok):
            return False, f"optimization perf 工件 {perf_file.name} 缺少可靠精度证据：{output_type} 需 text/match_rate=100% 或 cosine>=0.999"
    elif vector_family:
        if not cosine_ok:
            return False, f"optimization perf 工件 {perf_file.name} 缺少可靠精度证据：{output_type} 需 cosine>=0.999"
        if not max_abs_ok:
            return False, f"optimization perf 工件 {perf_file.name} 的 max_abs_error 过大，需 < 0.001"
    elif generated_image_family:
        image_cosine_ok = cosine is not None and cosine >= 0.7
        if not image_cosine_ok:
            return False, f"optimization perf 工件 {perf_file.name} 缺少可靠精度证据：{output_type} 需 cosine>=0.7（扩散生成存在随机性，0.7 为可接受阈值）"
    else:
        if not (exact_ok or cosine_ok or ppl_ok):
            return False, f"optimization perf 工件 {perf_file.name} 缺少可接受的精度证据（match_rate / cosine_similarity / ppl_avg_rel_diff）"
    return True, ""


def _validate_benchmark_metric_artifacts(path_val: str) -> tuple[bool, str]:
    """benchmark completed 前，至少存在一份 NPU baseline benchmark_metrics，且样本数 >= 50。"""
    adapt_dir = Path(_PROJECT_ROOT) / path_val
    metric_files = sorted(adapt_dir.glob("benchmark_metrics*.json"))
    if not metric_files:
        return False, "缺少 benchmark_metrics*.json，无法校验 benchmark 样本数"

    matched_count = 0
    for metric_file in metric_files:
        name_lower = metric_file.name.lower()
        if "cuda" in name_lower or "perf" in name_lower:
            continue
        try:
            metric = json.loads(metric_file.read_text(encoding="utf-8"))
        except Exception as e:
            return False, f"读取 benchmark 工件失败: {metric_file.name}: {e}"
        if not isinstance(metric, dict):
            return False, f"benchmark 工件 {metric_file.name} 必须是 JSON object"
        metric_device = str(metric.get("device") or "").strip().lower()
        if metric_device and "npu" not in metric_device:
            continue
        matched_count += 1
        required_ok, required_err = _validate_required_metric_fields(metric, metric_file, "benchmark baseline", _REQUIRED_COMPLETED_BENCHMARK_FIELDS)
        if not required_ok:
            return False, required_err
        samples_ok, samples_err = _validate_metric_num_samples(metric, metric_file, "benchmark baseline")
        if not samples_ok:
            return False, samples_err
        health_ok, health_err = _validate_metric_metadata_health(metric, metric_file, "benchmark baseline")
        if not health_ok:
            return False, health_err

    if matched_count == 0:
        return False, "缺少 NPU baseline benchmark_metrics*.json，无法校验 benchmark 样本数"
    return True, ""


def _validate_optimization_metric_artifacts(path_val: str, notes_str: str) -> tuple[bool, str]:
    """optimization completed 前，artifact 必须显式回溯且 baseline/perf 工件元数据一致。"""
    try:
        data = json.loads(notes_str)
    except json.JSONDecodeError as e:
        return False, f"optimization_notes 非法 JSON: {e}"
    if not isinstance(data, dict):
        return False, "optimization_notes 必须是 JSON object"

    best_result = data.get("best_result")
    if not isinstance(best_result, dict):
        return False, "optimization_notes.best_result 必须是 object"

    baseline_latency = best_result.get("baseline_latency_s")
    perf_latency = best_result.get("perf_latency_s")
    baseline_wall_clock = best_result.get("baseline_wall_clock_s")
    perf_wall_clock = best_result.get("perf_wall_clock_s")
    speedup_ratio = best_result.get("speedup_ratio")
    if not isinstance(baseline_latency, (int, float)) and not isinstance(speedup_ratio, (int, float)):
        return True, ""
    if not isinstance(perf_latency, (int, float)) or not isinstance(baseline_latency, (int, float)):
        return False, "声明 speedup_ratio/baseline_latency_s 时，best_result 必须同时包含数值型 perf_latency_s 和 baseline_latency_s"
    if not isinstance(baseline_wall_clock, (int, float)) or not isinstance(perf_wall_clock, (int, float)):
        return False, "声明 speedup_ratio 时，best_result 必须同时包含数值型 baseline_wall_clock_s 和 perf_wall_clock_s"

    adapt_dir = Path(_PROJECT_ROOT) / path_val
    baseline_artifact = best_result.get("baseline_artifact")
    perf_artifact = best_result.get("perf_artifact")
    baseline_read_ok, baseline_path, baseline_metric, baseline_read_err = _read_metric_artifact(adapt_dir, baseline_artifact, "optimization baseline")
    if not baseline_read_ok or baseline_path is None or baseline_metric is None:
        return False, baseline_read_err
    perf_read_ok, perf_path, perf_metric, perf_read_err = _read_metric_artifact(adapt_dir, perf_artifact, "optimization perf")
    if not perf_read_ok or perf_path is None or perf_metric is None:
        return False, perf_read_err

    for metric, metric_path, label in (
        (baseline_metric, baseline_path, "optimization baseline"),
        (perf_metric, perf_path, "optimization perf"),
    ):
        required_ok, required_err = _validate_required_metric_fields(metric, metric_path, label, _REQUIRED_COMPLETED_OPTIMIZATION_FIELDS)
        if not required_ok:
            return False, required_err
        samples_ok, samples_err = _validate_metric_num_samples(metric, metric_path, label)
        if not samples_ok:
            return False, samples_err
        health_ok, health_err = _validate_metric_metadata_health(metric, metric_path, label)
        if not health_ok:
            return False, health_err
        identity_ok, identity_err = _validate_metric_identity(
            metric,
            metric_path,
            label,
            expected_mode=str(best_result.get("mode") or "").strip(),
            expected_dataset=str(best_result.get("dataset") or "").strip(),
            expected_dtype=str(best_result.get("dtype") or "").strip(),
            expected_output_type=str(best_result.get("output_type") or "").strip(),
        )
        if not identity_ok:
            return False, identity_err

    baseline_metric_latency = baseline_metric.get("latency_s")
    perf_metric_latency = perf_metric.get("latency_s")
    if not _metric_close(float(baseline_latency), float(baseline_metric_latency)):
        return False, "optimization_notes.best_result.baseline_latency_s 与 baseline_artifact.latency_s 不一致"
    if not _metric_close(float(perf_latency), float(perf_metric_latency)):
        return False, "optimization_notes.best_result.perf_latency_s 与 perf_artifact.latency_s 不一致"
    baseline_wall_clock_actual, baseline_wall_clock_err = _extract_metric_wall_clock_s(baseline_metric, baseline_path, "optimization baseline")
    if baseline_wall_clock_err:
        return False, baseline_wall_clock_err
    perf_wall_clock_actual, perf_wall_clock_err = _extract_metric_wall_clock_s(perf_metric, perf_path, "optimization perf")
    if perf_wall_clock_err:
        return False, perf_wall_clock_err
    if baseline_wall_clock_actual is None or perf_wall_clock_actual is None:
        return False, "无法从 optimization baseline/perf 工件推导 wall-clock"
    baseline_wall_clock_source, baseline_source_err = _detect_metric_wall_clock_source(baseline_metric, baseline_path, "optimization baseline")
    if baseline_source_err:
        return False, baseline_source_err
    perf_wall_clock_source, perf_source_err = _detect_metric_wall_clock_source(perf_metric, perf_path, "optimization perf")
    if perf_source_err:
        return False, perf_source_err
    if baseline_wall_clock_source != perf_wall_clock_source:
        return False, "baseline_artifact 与 perf_artifact 的 wall-clock 来源不一致"
    best_wall_clock_source = str(best_result.get("wall_clock_source") or "").strip()
    if best_wall_clock_source != str(baseline_wall_clock_source or ""):
        return False, "optimization_notes.best_result.wall_clock_source 与 benchmark 工件实际来源不一致"
    if not _metric_close(float(baseline_wall_clock), baseline_wall_clock_actual):
        return False, "optimization_notes.best_result.baseline_wall_clock_s 与 baseline_artifact wall-clock 不一致"
    if not _metric_close(float(perf_wall_clock), perf_wall_clock_actual):
        return False, "optimization_notes.best_result.perf_wall_clock_s 与 perf_artifact wall-clock 不一致"
    expected_speedup = baseline_wall_clock_actual / perf_wall_clock_actual
    if not _metric_close(float(speedup_ratio), expected_speedup):
        return False, "optimization_notes.best_result.speedup_ratio 与 baseline/perf 工件推导出的 wall-clock 提速不一致"
    output_type = str(best_result.get("output_type") or "")
    num_samples = best_result.get("num_samples")
    if _requires_per_sample_wall_clock_alignment(output_type) and isinstance(num_samples, (int, float)) and not isinstance(num_samples, bool) and float(num_samples) > 0:
        baseline_per_sample_wall_clock = baseline_wall_clock_actual / float(num_samples)
        perf_per_sample_wall_clock = perf_wall_clock_actual / float(num_samples)
        if not _metric_close(baseline_per_sample_wall_clock, float(baseline_latency)):
            return False, "非生成类任务要求 baseline_artifact 的 wall-clock/num_samples 与 baseline_latency_s 一致"
        if not _metric_close(perf_per_sample_wall_clock, float(perf_latency)):
            return False, "非生成类任务要求 perf_artifact 的 wall-clock/num_samples 与 perf_latency_s 一致"

    if baseline_metric.get("output_type") != perf_metric.get("output_type"):
        return False, "baseline_artifact 与 perf_artifact 的 output_type 不一致，禁止用不同工作负载计算 speedup"
    if baseline_metric.get("mode") != perf_metric.get("mode"):
        return False, "baseline_artifact 与 perf_artifact 的 mode 不一致"
    if baseline_metric.get("dtype") != perf_metric.get("dtype"):
        return False, "baseline_artifact 与 perf_artifact 的 dtype 不一致"
    if baseline_metric.get("dataset") != perf_metric.get("dataset"):
        return False, "baseline_artifact 与 perf_artifact 的 dataset 不一致"

    if perf_metric.get("baseline_file") and perf_metric.get("baseline_file") != baseline_path.name:
        return False, "perf_artifact.baseline_file 与 best_result.baseline_artifact 不一致"
    perf_metric_wall_clock_speedup = perf_metric.get("wall_clock_speedup_ratio")
    if isinstance(perf_metric_wall_clock_speedup, (int, float)) and not _metric_close(float(speedup_ratio), float(perf_metric_wall_clock_speedup)):
        return False, "perf_artifact.wall_clock_speedup_ratio 与 optimization_notes.best_result.speedup_ratio 不一致"
    if perf_metric.get("comparison_method") and perf_metric.get("comparison_method") != best_result.get("comparison_method"):
        return False, "perf_artifact.comparison_method 与 optimization_notes.best_result.comparison_method 不一致"
    if perf_metric.get("precision_method") and perf_metric.get("precision_method") != best_result.get("precision_method"):
        return False, "perf_artifact.precision_method 与 optimization_notes.best_result.precision_method 不一致"
    if perf_metric.get("baseline_type") == "self-baseline" and best_result.get("comparison_method") == "independent_baseline_artifact":
        return False, "perf_artifact 标记为 self-baseline，但 optimization_notes 声明为 independent_baseline_artifact"

    precision_ok, precision_err = _validate_precision_evidence(best_result, baseline_metric, perf_metric, perf_path)
    if not precision_ok:
        return False, precision_err
    return True, ""


def _validate_benchmark_completion(model_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT adaptation_path, benchmark_owner, adaptation_status FROM models WHERE model_id = ?", (model_id,))
    row = cursor.fetchone()
    path_val, adapt_name = _resolve_adapt_path(model_id, "", row)
    benchmark_owner = str(row["benchmark_owner"] or "") if row else ""
    status_val = str(row["adaptation_status"] or "").strip().lower() if row else ""
    if status_val != "completed":
        conn.close()
        print(f"Intercepted: cannot set benchmark_status=completed: adaptation_status must be completed (current: {status_val or 'empty'})")
        print(f"INTERCEPTED: model_id={model_id} benchmark_owner={benchmark_owner} notes=链式依赖：需先完成 adaptation")
        sys.exit(1)
    if not (path_val or "").strip():
        conn.close()
        print("Intercepted: cannot set benchmark_status=completed: adaptation path is empty")
        print(f"INTERCEPTED: model_id={model_id} benchmark_owner={benchmark_owner} notes=无 adaptation_path")
        sys.exit(1)
    path_ok, path_err = _validate_adaptation_path_boundary(path_val)
    if not path_ok:
        conn.close()
        print(f"Intercepted: cannot set benchmark_status=completed: {path_err}")
        print(f"INTERCEPTED: model_id={model_id} benchmark_owner={benchmark_owner} notes={path_err}")
        sys.exit(1)
    acc_run = os.path.join(_PROJECT_ROOT, path_val, "accuracy_run.py")
    if not os.path.isfile(acc_run):
        conn.close()
        print(f"Intercepted: cannot set benchmark_status=completed: accuracy_run.py not found at {acc_run}")
        print(f"INTERCEPTED: model_id={model_id} benchmark_owner={benchmark_owner} notes=缺少 accuracy_run.py")
        sys.exit(1)
    check_ok, check_err = _run_check_script(
        Path(_PROJECT_ROOT) / "benchmark" / "scripts" / "check_accuracy_run.py",
        adapt_name,
        "check_accuracy_run.py",
    )
    if not check_ok:
        conn.close()
        error_notes = f"check_accuracy_run.py 未通过，需修复 accuracy_run.py 后重新评测。{check_err}"
        print(f"Intercepted: cannot set benchmark_status=completed until check passes for {adapt_name}")
        print(f"INTERCEPTED: model_id={model_id} benchmark_owner={benchmark_owner} notes={error_notes[:200]}")
        sys.exit(1)
    metric_ok, metric_err = _validate_benchmark_metric_artifacts(path_val)
    if not metric_ok:
        conn.close()
        print(f"Intercepted: cannot set benchmark_status=completed: {metric_err}")
        print(f"INTERCEPTED: model_id={model_id} benchmark_owner={benchmark_owner} notes={metric_err}")
        sys.exit(1)
    conn.close()


def _validate_optimization_completion(model_id: str, notes: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT adaptation_path, optimization_owner, adaptation_status, benchmark_status FROM models WHERE model_id = ?",
        (model_id,),
    )
    row = cursor.fetchone()
    if not row:
        conn.close()
        print(f"Intercepted: model_id={model_id} not found")
        sys.exit(1)
    status_val = (row["adaptation_status"] or "").strip().lower()
    bench_val = (row["benchmark_status"] or "").strip().lower()
    if status_val != "completed":
        conn.close()
        print(f"Intercepted: cannot set optimization_status=completed: adaptation_status must be completed (current: {status_val or 'empty'})")
        print(f"INTERCEPTED: model_id={model_id} notes=链式依赖：需先完成 adaptation")
        sys.exit(1)
    if bench_val != "completed":
        conn.close()
        print(f"Intercepted: cannot set optimization_status=completed: benchmark_status must be completed (current: {bench_val or 'empty'})")
        print(f"INTERCEPTED: model_id={model_id} notes=链式依赖：需先完成 benchmark")
        sys.exit(1)
    path_val, adapt_name = _resolve_adapt_path(model_id, "", row)
    optimization_owner = str(row["optimization_owner"] or "")
    if not (path_val or "").strip():
        conn.close()
        print("Intercepted: cannot set optimization_status=completed: adaptation path is empty")
        print(f"INTERCEPTED: model_id={model_id} optimization_owner={optimization_owner} notes=无 adaptation_path")
        sys.exit(1)
    path_ok, path_err = _validate_adaptation_path_boundary(path_val)
    if not path_ok:
        conn.close()
        print(f"Intercepted: cannot set optimization_status=completed: {path_err}")
        print(f"INTERCEPTED: model_id={model_id} optimization_owner={optimization_owner} notes={path_err}")
        sys.exit(1)
    acc_run_perf = os.path.join(_PROJECT_ROOT, path_val, "accuracy_run_perf.py")
    if not os.path.isfile(acc_run_perf):
        conn.close()
        print(f"Intercepted: cannot set optimization_status=completed: accuracy_run_perf.py not found at {acc_run_perf}")
        print(f"INTERCEPTED: model_id={model_id} optimization_owner={optimization_owner} notes=缺少 accuracy_run_perf.py")
        sys.exit(1)
    check_ok, check_err = _run_check_script(
        Path(_PROJECT_ROOT) / "optimization" / "scripts" / "check_accuracy_run_perf.py",
        adapt_name,
        "check_accuracy_run_perf.py",
    )
    if not check_ok:
        conn.close()
        error_notes = f"check_accuracy_run_perf.py 未通过，需修复 accuracy_run_perf.py 后重新完成。{check_err}"
        print(f"Intercepted: cannot set optimization_status=completed until check passes for {adapt_name}")
        print(f"INTERCEPTED: model_id={model_id} optimization_owner={optimization_owner} notes={error_notes[:200]}")
        sys.exit(1)
    file_ok, file_notes, file_err = _read_optimization_notes_file(path_val)
    if not file_ok:
        conn.close()
        print(f"Intercepted: cannot set optimization_status=completed: {file_err}")
        print(f"INTERCEPTED: model_id={model_id} optimization_owner={optimization_owner} notes=optimization_notes.json 缺失或不合法")
        sys.exit(1)
    notes_raw = notes.strip()
    if not notes_raw:
        conn.close()
        print("Intercepted: optimization_notes notes is empty for completed status")
        print(f"INTERCEPTED: model_id={model_id} optimization_owner={optimization_owner} notes=completed 时必须传入 optimization_notes.json 原文")
        sys.exit(1)
    check_notes_ok, check_notes_err = _run_check_optimization_notes(notes_raw)
    if not check_notes_ok:
        conn.close()
        print(f"Intercepted: optimization_notes JSON 格式校验失败: {check_notes_err}")
        print(f"INTERCEPTED: model_id={model_id} optimization_owner={optimization_owner} notes=optimization_notes JSON 不合法")
        sys.exit(1)
    if notes_raw != file_notes:
        conn.close()
        print("Intercepted: optimization_notes notes does not match optimization_notes.json")
        print(f"INTERCEPTED: model_id={model_id} optimization_owner={optimization_owner} notes=completed 时必须传入 optimization_notes.json 原文")
        sys.exit(1)
    completed_notes_ok, completed_notes_err = _validate_completed_optimization_notes(notes_raw)
    if not completed_notes_ok:
        conn.close()
        print(f"Intercepted: {completed_notes_err}")
        print(f"INTERCEPTED: model_id={model_id} optimization_owner={optimization_owner} notes=completed 时必须使用真实 pretrained 对比结果")
        sys.exit(1)
    artifact_ok, artifact_err = _validate_optimization_metric_artifacts(path_val, notes_raw)
    if not artifact_ok:
        conn.close()
        print(f"Intercepted: {artifact_err}")
        print(f"INTERCEPTED: model_id={model_id} optimization_owner={optimization_owner} notes=benchmark_metrics 工件与 optimization_notes 不一致")
        sys.exit(1)
    conn.close()


def _validate_business_benchmark_completion(model_id: str, notes: str, *, allow_completed_rewrite: bool):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT adaptation_path, business_benchmark_owner,
               adaptation_status, benchmark_status, optimization_status
        FROM models WHERE model_id = ?
        """,
        (model_id,),
    )
    row = cursor.fetchone()
    if not row:
        conn.close()
        print(f"Intercepted: model_id={model_id} not found")
        sys.exit(1)
    status_val = (row["adaptation_status"] or "").strip().lower()
    bench_val = (row["benchmark_status"] or "").strip().lower()
    opt_val = (row["optimization_status"] or "").strip().lower()
    if not allow_completed_rewrite and status_val != "completed":
        conn.close()
        print(f"Intercepted: cannot set business_benchmark_status=completed: adaptation_status must be completed (current: {status_val or 'empty'})")
        print(f"INTERCEPTED: model_id={model_id} notes=链式依赖：需先完成 adaptation")
        sys.exit(1)
    if not allow_completed_rewrite and bench_val != "completed":
        conn.close()
        print(f"Intercepted: cannot set business_benchmark_status=completed: benchmark_status must be completed (current: {bench_val or 'empty'})")
        print(f"INTERCEPTED: model_id={model_id} notes=链式依赖：需先完成 benchmark")
        sys.exit(1)
    if not allow_completed_rewrite and opt_val != "completed":
        conn.close()
        print(f"Intercepted: cannot set business_benchmark_status=completed: optimization_status must be completed (current: {opt_val or 'empty'})")
        print(f"INTERCEPTED: model_id={model_id} notes=链式依赖：需先完成 optimization")
        sys.exit(1)

    path_val, adapt_name = _resolve_adapt_path(model_id, "", row)
    business_owner = str(row["business_benchmark_owner"] or "")
    if not (path_val or "").strip():
        conn.close()
        print("Intercepted: cannot set business_benchmark_status=completed: adaptation path is empty")
        print(f"INTERCEPTED: model_id={model_id} business_benchmark_owner={business_owner} notes=无 adaptation_path")
        sys.exit(1)
    path_ok, path_err = _validate_adaptation_path_boundary(path_val)
    if not path_ok:
        conn.close()
        print(f"Intercepted: cannot set business_benchmark_status=completed: {path_err}")
        print(f"INTERCEPTED: model_id={model_id} business_benchmark_owner={business_owner} notes={path_err}")
        sys.exit(1)

    check_ok, check_err = _run_check_script(
        Path(_PROJECT_ROOT) / "business_benchmark" / "scripts" / "check_business_benchmark_run.py",
        adapt_name,
        "check_business_benchmark_run.py",
    )
    if not check_ok:
        conn.close()
        error_notes = f"check_business_benchmark_run.py 未通过，需修复 business_summary.json / 业务测评工件后重新完成。{check_err}"
        print(f"Intercepted: cannot set business_benchmark_status=completed until check passes for {adapt_name}")
        print(f"INTERCEPTED: model_id={model_id} business_benchmark_owner={business_owner} notes={error_notes[:200]}")
        sys.exit(1)

    file_ok, file_notes, file_err = _read_business_summary_file(path_val)
    if not file_ok:
        conn.close()
        print(f"Intercepted: cannot set business_benchmark_status=completed: {file_err}")
        print(f"INTERCEPTED: model_id={model_id} business_benchmark_owner={business_owner} notes=business_summary.json 缺失或不合法")
        sys.exit(1)
    notes_raw = notes.strip()
    if not notes_raw:
        conn.close()
        print("Intercepted: business_benchmark notes is empty for completed status")
        print(f"INTERCEPTED: model_id={model_id} business_benchmark_owner={business_owner} notes=completed 时必须传入 business_summary.json 原文")
        sys.exit(1)
    check_notes_ok, check_notes_err = _run_check_business_summary(notes_raw)
    if not check_notes_ok:
        conn.close()
        print(f"Intercepted: business_summary JSON 格式校验失败: {check_notes_err}")
        print(f"INTERCEPTED: model_id={model_id} business_benchmark_owner={business_owner} notes=business_summary JSON 不合法")
        sys.exit(1)
    if notes_raw != file_notes:
        conn.close()
        print("Intercepted: business summary notes does not match business_summary.json")
        print(f"INTERCEPTED: model_id={model_id} business_benchmark_owner={business_owner} notes=completed 时必须传入 business_summary.json 原文")
        sys.exit(1)
    completed_ok, completed_err = _validate_completed_business_summary(notes_raw)
    if not completed_ok:
        conn.close()
        print(f"Intercepted: {completed_err}")
        print(f"INTERCEPTED: model_id={model_id} business_benchmark_owner={business_owner} notes=business completed 需包含 NPU/CUDA 真实业务结果")
        sys.exit(1)
    artifact_ok, artifact_err = _validate_business_metric_artifacts(path_val, notes_raw)
    if not artifact_ok:
        conn.close()
        print(f"Intercepted: {artifact_err}")
        print(f"INTERCEPTED: model_id={model_id} business_benchmark_owner={business_owner} notes=business_metrics 工件与 business_summary 不一致")
        sys.exit(1)
    conn.close()


def _validate_business_benchmark_wait_cuda(model_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT adaptation_path, business_benchmark_owner
        FROM models WHERE model_id = ?
        """,
        (model_id,),
    )
    row = cursor.fetchone()
    if not row:
        conn.close()
        print(f"Intercepted: model_id={model_id} not found")
        sys.exit(1)

    path_val, adapt_name = _resolve_adapt_path(model_id, "", row)
    business_owner = str(row["business_benchmark_owner"] or "")
    if not (path_val or "").strip():
        conn.close()
        print("Intercepted: cannot set business_benchmark_status=wait_cuda: adaptation path is empty")
        print(f"INTERCEPTED: model_id={model_id} business_benchmark_owner={business_owner} notes=wait_cuda 前缺少 adaptation_path")
        sys.exit(1)
    path_ok, path_err = _validate_adaptation_path_boundary(path_val)
    if not path_ok:
        conn.close()
        print(f"Intercepted: cannot set business_benchmark_status=wait_cuda: {path_err}")
        print(f"INTERCEPTED: model_id={model_id} business_benchmark_owner={business_owner} notes={path_err}")
        sys.exit(1)

    check_ok, check_err = _run_check_script(
        Path(_PROJECT_ROOT) / "business_benchmark" / "scripts" / "check_business_benchmark_run.py",
        adapt_name,
        "check_business_benchmark_run.py",
        extra_args=["--wait-cuda-npu-only"],
    )
    if not check_ok:
        conn.close()
        error_notes = f"wait_cuda 前置 NPU gate 未通过，需先修复本机 NPU baseline/perf 工件后再等待或执行远端 CUDA。{check_err}"
        print(f"Intercepted: cannot set business_benchmark_status=wait_cuda until local NPU sanity gate passes for {adapt_name}")
        print(f"INTERCEPTED: model_id={model_id} business_benchmark_owner={business_owner} notes={error_notes[:200]}")
        sys.exit(1)

    conn.close()


def update_benchmark_status(model_id, benchmark_status, notes=""):
    """更新 benchmark 状态"""
    stage_cfg = _STATUS_UPDATE_STAGE_CONFIG["benchmark"]
    _ensure_valid_stage_status(stage_cfg, benchmark_status)
    current_status = _get_model_stage_status(model_id, stage_cfg["status_col"])
    _ensure_stage_not_completed(stage_cfg, current_status)

    if benchmark_status == "completed":
        _validate_benchmark_completion(model_id)

    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 若任务进入终态，需要先查出 benchmark_owner，以便把该 agent 状态设回 idle
    # 同时检查当前状态，避免重复 commit（benchmark-runner 与 team-lead 都会调用，仅首次完成时 commit）
    owner_to_idle, prev_status = _load_owner_and_prev_status(cursor, model_id, stage_cfg, benchmark_status)
    _update_stage_status_row(cursor, model_id, stage_cfg, benchmark_status, notes, now)

    # 任务终态时：将对应 owner 在 agents 表中设为 idle，并清空 current_task
    _set_agent_idle(cursor, owner_to_idle)

    # benchmark 完成时设 optimization_status='pending' 以便 NPU 优化任务分配
    if benchmark_status == "completed":
        _initialize_downstream_stage(cursor, model_id, "optimization_status")
        cursor.execute("SELECT adaptation_path FROM models WHERE model_id = ?", (model_id,))
        path_row = cursor.fetchone()
        adaptation_path = str(path_row["adaptation_path"] or "").strip() if path_row else ""
    else:
        adaptation_path = ""

    conn.commit()
    conn.close()
    print(f"Updated {model_id} benchmark_status to {benchmark_status}.")

    # 仅当状态从非 completed 变为 completed 时 commit，避免 benchmark-runner 与 team-lead 重复调用导致重复 commit
    if benchmark_status == "completed" and prev_status != "completed":
        git_commit_and_push(
            stage_cfg["commit_message"].format(model_id=model_id),
            paths=_build_model_commit_paths(model_id, adaptation_path=adaptation_path),
        )


def assign_optimization_task(agent_id):
    """分配 NPU 优化任务给 agent（仅限 benchmark_status=completed 且 optimization_status=pending 的模型）"""
    _assign_stage_task(agent_id, _ASSIGN_STAGE_CONFIG["optimization"])


def update_optimization_status(model_id, optimization_status, notes=""):
    """更新 NPU 优化状态"""
    stage_cfg = _STATUS_UPDATE_STAGE_CONFIG["optimization"]
    _ensure_valid_stage_status(stage_cfg, optimization_status)
    current_status = _get_model_stage_status(model_id, stage_cfg["status_col"])
    _ensure_stage_not_completed(stage_cfg, current_status)

    if optimization_status == "completed":
        _validate_optimization_completion(model_id, notes)
    else:
        notes_ok, notes_err = _validate_optimization_non_completed_status(optimization_status, notes)
        if not notes_ok:
            print(f"Intercepted: cannot set optimization_status={optimization_status}: {notes_err}")
            sys.exit(1)

    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    owner_to_idle, prev_status = _load_owner_and_prev_status(cursor, model_id, stage_cfg, optimization_status)
    _update_stage_status_row(cursor, model_id, stage_cfg, optimization_status, notes, now)

    if optimization_status == "completed":
        cursor.execute("SELECT optimization_notes FROM models WHERE model_id = ?", (model_id,))
        verify_row = cursor.fetchone()
        verify_raw = (verify_row["optimization_notes"] if verify_row else "") if verify_row else ""
        verify_ok, verify_err = _verify_db_optimization_notes(verify_raw, notes)
        if not verify_ok:
            conn.rollback()
            conn.close()
            print(f"Error: post-write verification failed for {model_id}: {verify_err}")
            sys.exit(1)

    _set_agent_idle(cursor, owner_to_idle)

    if optimization_status == "completed":
        _initialize_downstream_stage(cursor, model_id, "business_benchmark_status")
        cursor.execute("SELECT adaptation_path FROM models WHERE model_id = ?", (model_id,))
        path_row = cursor.fetchone()
        adaptation_path = str(path_row["adaptation_path"] or "").strip() if path_row else ""
    else:
        adaptation_path = ""

    conn.commit()
    conn.close()
    print(f"Updated {model_id} optimization_status to {optimization_status}.")

    if optimization_status == "completed" and prev_status != "completed":
        git_commit_and_push(
            stage_cfg["commit_message"].format(model_id=model_id),
            paths=_build_model_commit_paths(model_id, adaptation_path=adaptation_path),
        )


def list_optimization_tasks(status=None):
    """列出 NPU 优化任务"""
    conn = get_db_connection()
    cursor = conn.cursor()
    if status:
        cursor.execute(
            "SELECT model_id, adaptation_status, benchmark_status, optimization_status, optimization_owner, adaptation_path FROM models WHERE optimization_status = ?",
            (status,),
        )
    else:
        cursor.execute("SELECT model_id, adaptation_status, benchmark_status, optimization_status, optimization_owner, adaptation_path FROM models WHERE optimization_status != ''")

    rows = cursor.fetchall()
    for row in rows:
        print(dict(row))
    conn.close()


def assign_business_benchmark_task(agent_id):
    """分配第四阶段业务测评任务给 agent。"""
    _assign_stage_task(agent_id, _ASSIGN_STAGE_CONFIG["business_benchmark"])


def update_business_benchmark_status(model_id, business_benchmark_status, notes=""):
    """更新第四阶段业务测评状态。"""
    stage_cfg = _STATUS_UPDATE_STAGE_CONFIG["business_benchmark"]
    _ensure_valid_stage_status(stage_cfg, business_benchmark_status)
    current_stage = _get_model_stage_status(model_id, stage_cfg["status_col"])
    allow_completed_rewrite = current_stage == "completed" and business_benchmark_status == "completed"
    _ensure_stage_not_completed(stage_cfg, current_stage, allow_completed_rewrite=allow_completed_rewrite)

    if business_benchmark_status == "completed":
        _validate_business_benchmark_completion(
            model_id,
            notes,
            allow_completed_rewrite=allow_completed_rewrite,
        )
    elif business_benchmark_status == "wait_cuda":
        _validate_business_benchmark_wait_cuda(model_id)

    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    owner_to_idle, prev_status = _load_owner_and_prev_status(cursor, model_id, stage_cfg, business_benchmark_status)
    _update_stage_status_row(cursor, model_id, stage_cfg, business_benchmark_status, notes, now)

    if business_benchmark_status == "completed":
        cursor.execute("SELECT business_benchmark_notes FROM models WHERE model_id = ?", (model_id,))
        verify_row = cursor.fetchone()
        verify_raw = (verify_row["business_benchmark_notes"] if verify_row else "") if verify_row else ""
        verify_ok, verify_err = _verify_db_business_summary(verify_raw, notes)
        if not verify_ok:
            conn.rollback()
            conn.close()
            print(f"Error: post-write verification failed for {model_id}: {verify_err}")
            sys.exit(1)

    _set_agent_idle(cursor, owner_to_idle)

    if business_benchmark_status == "completed":
        _initialize_downstream_stage(cursor, model_id, "human_review_status")
    else:
        _clear_human_review_status(cursor, model_id)

    cursor.execute("SELECT adaptation_path FROM models WHERE model_id = ?", (model_id,))
    path_row = cursor.fetchone()
    adaptation_path = str(path_row["adaptation_path"] or "").strip() if path_row else ""

    conn.commit()
    conn.close()
    print(f"Updated {model_id} business_benchmark_status to {business_benchmark_status}.")

    if business_benchmark_status == "completed" and prev_status != "completed":
        git_commit_and_push(
            stage_cfg["commit_message"].format(model_id=model_id),
            paths=_build_model_commit_paths(model_id, adaptation_path=adaptation_path),
        )


def update_human_review_status(model_id, human_review_status):
    """更新第五阶段人工核验状态。"""
    status = (human_review_status or "").strip().lower()
    if status not in {"pending", "completed"}:
        print(f"Error: Invalid human_review_status '{human_review_status}'.")
        sys.exit(1)

    conn = get_db_connection()
    cursor = conn.cursor()
    _ensure_models_table_schema(cursor)
    cursor.execute(
        "SELECT business_benchmark_status, human_review_status, adaptation_path FROM models WHERE model_id = ?",
        (model_id,),
    )
    row = cursor.fetchone()
    if not row:
        conn.close()
        print(f"Error: model_id '{model_id}' not found.")
        sys.exit(1)

    business_status = (row["business_benchmark_status"] or "").strip().lower()
    if business_status != "completed":
        conn.close()
        print(f"Intercepted: cannot set human_review_status={status}: business_benchmark_status must be completed (current: {business_status or 'empty'})")
        sys.exit(1)

    prev_status = (row["human_review_status"] or "").strip().lower()
    if prev_status == status:
        conn.close()
        print(f"human_review_status for {model_id} is already {status}.")
        return

    cursor.execute("UPDATE models SET human_review_status = ? WHERE model_id = ?", (status, model_id))
    conn.commit()
    conn.close()
    print(f"Updated {model_id} human_review_status to {status}.")

    commit_message = "feat: complete human review for {model_id}" if status == "completed" else "chore: reset human review status for {model_id}"
    git_commit_and_push(
        commit_message.format(model_id=model_id),
        paths=_build_model_commit_paths(model_id, adaptation_path=str(row["adaptation_path"] or "").strip()),
    )


def list_business_benchmark_tasks(status=None):
    """列出第四阶段业务测评任务。"""
    conn = get_db_connection()
    cursor = conn.cursor()
    if status:
        cursor.execute(
            """
            SELECT model_id, adaptation_status, benchmark_status, optimization_status,
                   business_benchmark_status, business_benchmark_owner, adaptation_path
            FROM models WHERE business_benchmark_status = ?
            """,
            (status,),
        )
    else:
        cursor.execute(
            """
            SELECT model_id, adaptation_status, benchmark_status, optimization_status,
                   business_benchmark_status, business_benchmark_owner, adaptation_path
            FROM models WHERE business_benchmark_status != ''
            """
        )

    rows = cursor.fetchall()
    for row in rows:
        print(dict(row))
    conn.close()


def list_benchmark_tasks(status=None):
    """列出 benchmark 任务"""
    conn = get_db_connection()
    cursor = conn.cursor()
    if status:
        cursor.execute("SELECT model_id, adaptation_status, benchmark_status, benchmark_owner, adaptation_path FROM models WHERE benchmark_status = ?", (status,))
    else:
        cursor.execute("SELECT model_id, adaptation_status, benchmark_status, benchmark_owner, adaptation_path FROM models WHERE benchmark_status != ''")

    rows = cursor.fetchall()
    for row in rows:
        print(dict(row))
    conn.close()


def list_stale_agents(minutes=10):
    """列出心跳超过指定时间的 agent，不删除也不回收任务。"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cutoff_time = (datetime.now() - timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute(
        "SELECT id, last_heartbeat, status, current_task FROM agents WHERE last_heartbeat < ?",
        (cutoff_time,),
    )
    stale_agents = cursor.fetchall()
    conn.close()

    if stale_agents:
        print(f"Stale agents (heartbeat > {minutes} min):")
        for row in stale_agents:
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "last_heartbeat": row["last_heartbeat"],
                        "status": row["status"],
                        "current_task": row["current_task"],
                    }
                )
            )
    else:
        print(f"No stale agents found (heartbeat > {minutes} min).")

    return stale_agents


def cleanup_stale_agents(minutes=15):
    conn = get_db_connection()
    cursor = conn.cursor()
    _ensure_models_table_schema(cursor)
    _ensure_agents_table(cursor)
    cutoff_time = (datetime.now() - timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stale_note = f"超时回收（{minutes} 分钟无心跳），待重新分配"

    # Find stale agents
    cursor.execute("SELECT id FROM agents WHERE last_heartbeat < ?", (cutoff_time,))
    stale_agents = [row["id"] for row in cursor.fetchall()]

    if stale_agents:
        print(f"Found stale agents: {stale_agents}")
        # Delete stale agents
        cursor.execute("DELETE FROM agents WHERE last_heartbeat < ?", (cutoff_time,))

        # Release adaptation tasks (owner + status only)
        placeholders = ",".join(["?"] * len(stale_agents))
        cursor.execute(
            f"""
            UPDATE models SET
                adaptation_owner = '',
                adaptation_status = 'pending',
                adaptation_last_updated = ?,
                adaptation_notes = {_append_status_note_sql("adaptation_notes")}
            WHERE adaptation_owner IN ({placeholders}) AND adaptation_status = 'in_progress'
            """,
            [now, stale_note, stale_note, stale_note, *stale_agents],
        )
        adapt_count = cursor.rowcount
        # Release benchmark tasks (benchmark_owner + benchmark_status only)
        cursor.execute(
            f"""
            UPDATE models SET
                benchmark_owner = '',
                benchmark_status = 'pending',
                benchmark_last_updated = ?,
                benchmark_notes = {_append_status_note_sql("benchmark_notes")}
            WHERE benchmark_owner IN ({placeholders}) AND benchmark_status = 'in_progress'
            """,
            [now, stale_note, stale_note, stale_note, *stale_agents],
        )
        bench_count = cursor.rowcount
        # Release optimization tasks (optimization_owner + optimization_status only)
        cursor.execute(
            f"""
            UPDATE models SET
                optimization_owner = '',
                optimization_status = 'pending',
                optimization_last_updated = ?,
                optimization_notes = {_append_status_note_sql("optimization_notes")}
            WHERE optimization_owner IN ({placeholders}) AND optimization_status = 'in_progress'
            """,
            [now, stale_note, stale_note, stale_note, *stale_agents],
        )
        opt_count = cursor.rowcount
        cursor.execute(
            f"""
            UPDATE models SET
                business_benchmark_owner = '',
                business_benchmark_status = 'pending',
                business_benchmark_last_updated = ?,
                business_benchmark_notes = {_append_status_note_sql("business_benchmark_notes")}
            WHERE business_benchmark_owner IN ({placeholders}) AND business_benchmark_status = 'in_progress'
            """,
            [now, stale_note, stale_note, stale_note, *stale_agents],
        )
        business_count = cursor.rowcount
        conn.commit()
        print(f"Cleaned up {len(stale_agents)} stale agents. Reset adaptation: {adapt_count}, benchmark: {bench_count}, optimization: {opt_count}, business_benchmark: {business_count}.")
    else:
        print("No stale agents found.")

    conn.close()


def normalize_empty_strings():
    """将 board.db 中所有 NULL 替换为 ''，与项目统一空值约定一致。"""
    conn = get_db_connection()
    cursor = conn.cursor()
    _ensure_models_table_schema(cursor)
    _ensure_agents_table(cursor)
    models_cols = [
        "source",
        "priority",
        "adaptation_status",
        "adaptation_started_at",
        "adaptation_last_updated",
        "adaptation_failure_reason",
        "url",
        "description",
        "adaptation_notes",
        "adaptation_path",
        "adaptation_owner",
        "benchmark_status",
        "benchmark_started_at",
        "benchmark_last_updated",
        "benchmark_owner",
        "benchmark_notes",
        "optimization_status",
        "optimization_started_at",
        "optimization_last_updated",
        "optimization_owner",
        "optimization_notes",
        "business_benchmark_status",
        "business_benchmark_started_at",
        "business_benchmark_last_updated",
        "business_benchmark_owner",
        "business_benchmark_notes",
        "human_review_status",
    ]
    for col in models_cols:
        cursor.execute(f"UPDATE models SET {col} = '' WHERE {col} IS NULL")
    for col in ["last_heartbeat", "status", "current_task"]:
        cursor.execute(f"UPDATE agents SET {col} = '' WHERE {col} IS NULL")
    conn.commit()
    conn.close()
    print("Normalized all NULL to '' in models and agents.")


def list_agents(status=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    if status:
        cursor.execute("SELECT * FROM agents WHERE status = ?", (status,))
    else:
        cursor.execute("SELECT * FROM agents")
    rows = cursor.fetchall()
    for row in rows:
        print(dict(row))
    conn.close()


def list_adaptation_tasks(status=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    if status:
        cursor.execute("SELECT * FROM models WHERE adaptation_status = ?", (status,))
    else:
        cursor.execute("SELECT * FROM models")

    rows = cursor.fetchall()
    for row in rows:
        print(dict(row))
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Board Operations Script")
    subparsers = parser.add_subparsers(dest="action", required=True)

    # Init
    subparsers.add_parser("init", help="Initialize database")

    # Reset
    subparsers.add_parser("reset", help="Reset board (clear agents, reset in_progress tasks; business_benchmark wait_cuda -> pending)")
    subparsers.add_parser("clear_agents", help="Clear agents table only")

    rollback_models_parser = subparsers.add_parser("rollback_models", help="Rollback model list to a chosen stage/status")
    rollback_models_parser.add_argument("--model_ids", "--models", dest="model_ids", required=True, help="Comma/newline/space separated model_id list")
    rollback_models_parser.add_argument("--rollback_to", "--to", dest="rollback_to", required=True, choices=_ROLLBACK_TARGET_CHOICES, help="Target rollback state, e.g. optimization:pending")
    rollback_models_parser.add_argument("--notes", default="")

    # Normalize NULL -> ''
    subparsers.add_parser("normalize_empty", help="Replace all NULL in models/agents with ''")

    # Heartbeat
    hb_parser = subparsers.add_parser("heartbeat", help="Update agent heartbeat")
    hb_parser.add_argument("--id", required=True, help="Agent ID")
    hb_parser.add_argument("--status", required=True, choices=["active", "idle", "offline"])
    hb_parser.add_argument("--task", required=True, help="Current task description")

    # Register Model
    reg_parser = subparsers.add_parser("register_model", help="Register a new model")
    reg_parser.add_argument("--model_id", required=True)
    reg_parser.add_argument("--source", default="huggingface")
    reg_parser.add_argument("--url", default="", help="Required. Model page URL (must be unique).")
    reg_parser.add_argument("--description", default="")
    reg_parser.add_argument("--adaptation_status", default="pending")
    reg_parser.add_argument("--adaptation_failure_reason", default="")

    # Assign adaptation task
    assign_parser = subparsers.add_parser("assign_adaptation_task", help="Assign a pending adaptation task to an agent")
    assign_parser.add_argument("--agent_id", required=True)

    # Update adaptation task
    update_parser = subparsers.add_parser("update_adaptation_status", help="Update adaptation status")
    update_parser.add_argument("--model_id", required=True)
    update_parser.add_argument("--adaptation_status", required=True)
    update_parser.add_argument("--adaptation_notes", default="")
    update_parser.add_argument("--adaptation_failure_reason", default="")
    update_parser.add_argument("--adaptation_path", default="", help="When adaptation_status=completed, set adaptation dir (e.g. adaptations/org_name); optional, else derived from model_id")

    # Cleanup
    cleanup_parser = subparsers.add_parser("cleanup", help="Cleanup stale agents")
    cleanup_parser.add_argument("--minutes", type=int, default=15, help="Minutes threshold (default 15)")
    # List agents
    list_agents_parser = subparsers.add_parser("list_agents", help="List agents (optionally filter by status)")
    list_agents_parser.add_argument("--status", choices=["active", "idle", "offline"], help="Filter by agent status")

    # List stale agents
    list_stale_parser = subparsers.add_parser("list_stale_agents", help="List agents with stale heartbeat")
    list_stale_parser.add_argument("--minutes", type=int, default=10, help="Minutes threshold (default 10)")

    # List
    list_parser = subparsers.add_parser("list_adaptation_tasks", help="List adaptation tasks")
    list_parser.add_argument("--status", help="Filter by adaptation_status")

    # Assign Benchmark Task
    assign_bench_parser = subparsers.add_parser("assign_benchmark_task", help="Assign a pending benchmark task to an agent")
    assign_bench_parser.add_argument("--agent_id", required=True)

    # Update Benchmark Status
    update_bench_parser = subparsers.add_parser("update_benchmark_status", help="Update benchmark status")
    update_bench_parser.add_argument("--model_id", required=True)
    update_bench_parser.add_argument("--benchmark_status", required=True, help="New benchmark status")
    update_bench_parser.add_argument("--notes", default="")

    # List Benchmark Tasks
    list_bench_parser = subparsers.add_parser("list_benchmark_tasks", help="List benchmark tasks")
    list_bench_parser.add_argument("--status", help="Filter by benchmark status")

    # Assign Optimization Task
    assign_opt_parser = subparsers.add_parser("assign_optimization_task", help="Assign a pending NPU optimization task to an agent")
    assign_opt_parser.add_argument("--agent_id", required=True)

    # Update Optimization Status
    update_opt_parser = subparsers.add_parser("update_optimization_status", help="Update NPU optimization status")
    update_opt_parser.add_argument("--model_id", required=True)
    update_opt_parser.add_argument("--optimization_status", required=True, help="New optimization status")
    update_opt_parser.add_argument("--notes", default="")

    # List Optimization Tasks
    list_opt_parser = subparsers.add_parser("list_optimization_tasks", help="List NPU optimization tasks")
    list_opt_parser.add_argument("--status", help="Filter by optimization status")

    # Assign Business Benchmark Task
    assign_business_parser = subparsers.add_parser("assign_business_benchmark_task", help="Assign a pending business benchmark task to an agent")
    assign_business_parser.add_argument("--agent_id", required=True)

    # Update Business Benchmark Status
    update_business_parser = subparsers.add_parser("update_business_benchmark_status", help="Update business benchmark status")
    update_business_parser.add_argument("--model_id", required=True)
    update_business_parser.add_argument("--business_benchmark_status", required=True, help="New business benchmark status")
    update_business_parser.add_argument("--notes", default="")

    update_human_review_parser = subparsers.add_parser("update_human_review_status", help="Update human review status")
    update_human_review_parser.add_argument("--model_id", required=True)
    update_human_review_parser.add_argument("--human_review_status", required=True, choices=["pending", "completed"], help="New human review status")

    # List Business Benchmark Tasks
    list_business_parser = subparsers.add_parser("list_business_benchmark_tasks", help="List business benchmark tasks")
    list_business_parser.add_argument("--status", help="Filter by business_benchmark status")

    args = parser.parse_args()

    action_handlers: dict[str, Callable[[], None]] = {
        "init": init_db,
        "reset": reset_board,
        "clear_agents": clear_agents,
        "rollback_models": lambda: rollback_models(args.model_ids, args.rollback_to, args.notes),
        "normalize_empty": normalize_empty_strings,
        "heartbeat": lambda: heartbeat(args.id, args.status, args.task),
        "register_model": lambda: register_model(
            args.model_id,
            args.source,
            url=args.url,
            description=args.description,
            adaptation_status=args.adaptation_status,
            adaptation_failure_reason=args.adaptation_failure_reason,
        ),
        "assign_adaptation_task": lambda: assign_adaptation_task(args.agent_id),
        "update_adaptation_status": lambda: update_adaptation_status(
            args.model_id,
            args.adaptation_status,
            args.adaptation_notes,
            args.adaptation_failure_reason,
            args.adaptation_path,
        ),
        "cleanup": lambda: cleanup_stale_agents(args.minutes),
        "list_agents": lambda: list_agents(args.status),
        "list_stale_agents": lambda: list_stale_agents(args.minutes),
        "list_adaptation_tasks": lambda: list_adaptation_tasks(args.status),
        "assign_benchmark_task": lambda: assign_benchmark_task(args.agent_id),
        "update_benchmark_status": lambda: update_benchmark_status(args.model_id, args.benchmark_status, args.notes),
        "list_benchmark_tasks": lambda: list_benchmark_tasks(args.status),
        "assign_optimization_task": lambda: assign_optimization_task(args.agent_id),
        "update_optimization_status": lambda: update_optimization_status(args.model_id, args.optimization_status, args.notes),
        "list_optimization_tasks": lambda: list_optimization_tasks(args.status),
        "assign_business_benchmark_task": lambda: assign_business_benchmark_task(args.agent_id),
        "update_business_benchmark_status": lambda: update_business_benchmark_status(args.model_id, args.business_benchmark_status, args.notes),
        "update_human_review_status": lambda: update_human_review_status(args.model_id, args.human_review_status),
        "list_business_benchmark_tasks": lambda: list_business_benchmark_tasks(args.status),
    }
    if args.action not in action_handlers:
        parser.error(f"Unsupported action: {args.action}")
    handler = action_handlers[args.action]
    handler()
