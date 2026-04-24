#!/usr/bin/env python3
"""
Team-lead monitoring loop for NPU optimization tasks.
Continuously checks agent heartbeats, inbox messages, assigns tasks, and handles completions.
"""

import json
import os
import subprocess
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parent.parent)
BOARD_OPS = f"{PROJECT_ROOT / '.venv' / 'bin' / 'python'} {PROJECT_ROOT / 'scripts' / 'board_ops.py'}"
INBOX_PATH = os.path.expanduser("~/.claude/teams/optimization-team-v7/inboxes/team-lead.json")


def run_board_ops(args):
    result = subprocess.run(f"{BOARD_OPS} {args}", shell=True, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def get_agents():
    out, _, _ = run_board_ops("list_agents")
    agents = {}
    for line in out.split("\n"):
        if not line.strip() or line.startswith("Agent"):
            continue
        parts = line.split(", ")
        if len(parts) >= 4:
            agent_id = parts[0].strip()
            status = parts[2].split("=")[1].strip() if "=" in parts[2] else ""
            task = parts[3].split("=")[1].strip() if "=" in parts[3] else ""
            agents[agent_id] = {"status": status, "task": task}
    return agents


def get_optimization_status(model_id):
    conn = sqlite3.connect(str(PROJECT_ROOT / "board.db"))
    cur = conn.cursor()
    cur.execute("SELECT optimization_status, optimization_owner FROM models WHERE model_id=?", (model_id,))
    row = cur.fetchone()
    conn.close()
    return row


def update_optimization_status(model_id, status, notes):
    notes_json = json.dumps(notes) if isinstance(notes, dict) else notes
    # Escape for shell
    notes_escaped = notes_json.replace("'", "'\\''")
    out, err, code = run_board_ops(f'update_optimization_status --model_id "{model_id}" --optimization_status "{status}" --notes \'{notes_escaped}\'')
    return out, err, code


def assign_optimization_task(agent_id):
    out, err, code = run_board_ops(f"assign_optimization_task --agent_id {agent_id}")
    return out, err, code


def read_inbox():
    if not os.path.exists(INBOX_PATH):
        return []
    with open(INBOX_PATH) as f:
        return json.load(f)


def write_inbox(msgs):
    with open(INBOX_PATH, "w") as f:
        json.dump(msgs, f, indent=2)


def mark_all_read(msgs):
    for m in msgs:
        m["read"] = True
    write_inbox(msgs)


def process_completed(model_id, adaptation_path, notes_str, owner):
    """Process a completed optimization message."""
    # Verify notes match disk
    disk_path = PROJECT_ROOT / adaptation_path / "optimization_notes.json"
    if not disk_path.exists():
        print(f"  [WARN] Disk file not found: {disk_path}")
        return False

    try:
        with disk_path.open() as f:
            disk_notes = json.load(f)
    except Exception as e:
        print(f"  [ERROR] Failed to parse disk notes: {e}")
        return False

    # Try to update board
    out, err, code = update_optimization_status(model_id, "completed", disk_notes)
    if code == 0:
        print(f"  [OK] Updated board: {model_id} -> completed (speedup: {disk_notes.get('best_result', {}).get('speedup_ratio', 'N/A')}x)")
        return True
    elif "already completed" in err.lower():
        print(f"  [OK] Already completed: {model_id}")
        return True
    else:
        print(f"  [ERROR] Failed to update: {err}")
        return False


def process_failed(model_id, failure_reason, owner):
    """Process a failed optimization message."""
    # Classify failure
    retry_keywords = ["transformers", "meta tensor", "from_tf", "trust_remote_code", "OOM", "NPU", "超时", "queue", "heartbeat", "network", "403", "404", "gated", "LFS", "checkpoint", "missing", "unavailable", "version"]
    skip_keywords = ["架构", "architecture", "not applicable", "不适用"]

    failure_lower = failure_reason.lower()
    if any(kw in failure_lower for kw in retry_keywords):
        status = "pending"
        print(f"  [RETRY] {model_id}: failure_reason contains retry keywords, setting to pending")
    else:
        status = "skipped"
        print(f"  [SKIP] {model_id}: setting to skipped")

    out, err, code = run_board_ops(f'update_optimization_status --model_id "{model_id}" --optimization_status "{status}" --notes "{failure_reason[:500]}"')
    print(f"  [DB] {out}")
    return True


def assign_task_to_optimizer(agent_id):
    """Assign a new optimization task to an optimizer."""
    out, err, code = assign_optimization_task(agent_id)
    if code != 0 or not out.strip():
        print(f"  [SKIP] {agent_id}: no tasks available or error: {err}")
        return False

    # Parse output: "Assigned optimization MODEL_ID to AGENT_ID\nadaptation_path=..."
    lines = out.strip().split("\n")
    if not lines:
        return False

    first_line = lines[0]
    if "Assigned optimization" not in first_line:
        print(f"  [SKIP] {agent_id}: {first_line}")
        return False

    parts = first_line.split()
    model_id = parts[2]
    adaptation_path = lines[1].split("=")[1].strip() if len(lines) > 1 else ""

    # Send message
    msg = f"""action=optimize
model_id={model_id}
adaptation_path={adaptation_path}
requirements=参考 .claude/skills/torch-npu-optimization/SKILL.md；若为 diffusers pipeline，再参考 .claude/skills/ascend-diffusers-optimization/SKILL.md
boundary=所有有副作用操作仅限 adaptation_path；模型缓存仅限 adaptation_path/models；严禁项目根 models

【必填产出清单】：
1. accuracy_run_perf.py
2. benchmark_metrics_*_perf.json
3. optimization_notes.json（必须创建且格式合法；measurement_contract_version>=3；results[*] 至少需包含 dtype/mode/dataset/output_type/baseline_artifact/perf_artifact/perf_latency_s/perf_memory_mb/baseline_latency_s/baseline_wall_clock_s/perf_wall_clock_s/wall_clock_source/baseline_warmup_iterations/perf_warmup_iterations/warmup_policy/speedup_ratio/cosine_similarity；其中 speedup_ratio 统一按 baseline_wall_clock_s / perf_wall_clock_s 计算，wall-clock 禁止由 latency 反推，baseline/perf warmup 必须对称，validation_note 不得包含 cold baseline / partial run / derived from latency / not directly comparable 等红旗描述；完成前运行 check_optimization_notes.py）
4. 若走代码 patch 路线：额外提供 model_files/（可含 modeling_*.py、npu_patches.py 或其他 patch 模块）或 adaptation 内已修改的克隆源码文件；若走 runtime_only，不得伪造 model_files 充数

完成后发送 result=completed 给 team-lead。
严格要求：result=completed 中的 notes 必须是 adaptation_path/optimization_notes.json 的完整原文 JSON，严禁发送摘要。"""

    # Write to optimizer inbox
    opt_inbox_path = os.path.expanduser(f"~/.claude/teams/optimization-team-v7/inboxes/{agent_id}.json")
    if os.path.exists(opt_inbox_path):
        with open(opt_inbox_path) as f:
            opt_msgs = json.load(f)
        opt_msgs.append({"from": "team-lead", "text": msg, "summary": f"Assign {model_id} optimization task", "timestamp": datetime.now(timezone.utc).isoformat(), "color": "blue", "read": False})
        with open(opt_inbox_path, "w") as f:
            json.dump(opt_msgs, f, indent=2)
        print(f"  [ASSIGNED] {model_id} -> {agent_id} ({adaptation_path})")
        return True
    else:
        print(f"  [WARN] {agent_id} inbox not found: {opt_inbox_path}")
        return False


def main():
    print("=" * 60)
    print("NPU Optimization Monitoring Loop Started")
    print(f"Timestamp: {datetime.now()}")
    print("=" * 60)

    completed_count = 0
    iteration = 0

    while True:
        iteration += 1
        now = datetime.now()

        # 1. Check and update heartbeat
        run_board_ops('heartbeat --id "team-lead" --status "active" --task "monitoring"')

        # 2. Read inbox for new messages
        msgs = read_inbox()
        unread = [m for m in msgs if not m.get("read", False)]

        # 3. Process unread messages
        for m in unread:
            sender = m.get("from", "")
            text = m.get("text", "")
            timestamp = m.get("timestamp", "")

            if "result=completed" in text or '"type":"completed"' in text:
                # Parse completed message
                lines = text.split("\n")
                model_id = adaptation_path = notes = None
                for line in lines:
                    if line.startswith("model_id="):
                        model_id = line.split("=", 1)[1].strip()
                    elif line.startswith("adaptation_path="):
                        adaptation_path = line.split("=", 1)[1].strip()
                    elif line.startswith("notes="):
                        notes = line.split("=", 1)[1].strip()
                    elif line.startswith('"model_id"'):
                        parts = text.split('"model_id"')
                        if len(parts) > 1:
                            model_id = parts[1].split('"')[1] if '"' in parts[1] else parts[1].split(",")[0].strip()

                if model_id:
                    opt_status, owner = get_optimization_status(model_id)
                    if opt_status == "completed":
                        print(f"[{iteration}] Already completed: {model_id}")
                    elif opt_status in ("pending", "in_progress"):
                        success = process_completed(model_id, adaptation_path, notes, sender)
                        if success:
                            completed_count += 1
                            m["read"] = True
                    else:
                        print(f"[{iteration}] Unknown status {opt_status} for {model_id}")

            elif "result=failed" in text or '"type":"failed"' in text:
                lines = text.split("\n")
                model_id = failure_reason = None
                for line in lines:
                    if line.startswith("model_id="):
                        model_id = line.split("=", 1)[1].strip()
                    elif line.startswith("failure_reason="):
                        failure_reason = line.split("=", 1)[1].strip()
                if model_id:
                    process_failed(model_id, failure_reason or "unknown", sender)
                    m["read"] = True

            elif "status=idle" in text or '"type":"idle' in text or sender.startswith("npu-optimizer"):
                if "idle" in text.lower() or sender.startswith("npu-optimizer"):
                    # Try to assign new task
                    if sender.startswith("npu-optimizer"):
                        assign_task_to_optimizer(sender)
                    m["read"] = True

        # 4. Write back read status
        write_inbox(msgs)

        # 5. Check agents and assign idle optimizers
        agents = get_agents()
        for agent_id, info in agents.items():
            if agent_id.startswith("npu-optimizer") and info["status"] == "idle":
                assign_task_to_optimizer(agent_id)

        # 6. Count remaining
        conn = sqlite3.connect(str(PROJECT_ROOT / "board.db"))
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM models WHERE benchmark_status='completed' AND optimization_status='pending'")
        remaining = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM models WHERE optimization_status='in_progress'")
        in_progress = cur.fetchone()[0]
        conn.close()

        # 7. Print status
        if iteration % 10 == 0 or iteration <= 3:
            print(f"[{iteration}] {now.strftime('%H:%M:%S')} | Completed: {completed_count} | In-progress: {in_progress} | Pending: {remaining}")

        # 8. Check if done
        if remaining == 0 and in_progress == 0:
            print(f"\n[COMPLETE] All optimization tasks done! Total completed: {completed_count}")
            break

        # Sleep between iterations
        import time

        time.sleep(30)


if __name__ == "__main__":
    main()
