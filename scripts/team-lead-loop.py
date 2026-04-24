#!/usr/bin/env python3
"""
Team-lead monitoring loop - run in background.
Checks inbox, processes completions, assigns tasks.
"""

import json
import os
import subprocess
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parent.parent)
INBOX_PATH = os.path.expanduser("~/.claude/teams/optimization-team-v7/inboxes/team-lead.json")
LOG_FILE = "/tmp/team-lead-loop.log"


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run(args, cwd=None):
    if cwd is None:
        cwd = PROJECT_ROOT
    python = PROJECT_ROOT / ".venv" / "bin" / "python"
    board_ops = PROJECT_ROOT / "scripts" / "board_ops.py"
    r = subprocess.run(
        f"{python} {board_ops} {args}",
        shell=True,
        capture_output=True,
        text=True,
        cwd=str(cwd),
    )
    return r.stdout.strip(), r.stderr.strip(), r.returncode


def get_agents():
    out, _, _ = run("list_agents")
    agents = {}
    for line in out.split("\n"):
        if line.startswith("Agent ") or not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            id_part = parts[0]
            agent_id = id_part.split("=")[1] if "=" in id_part else id_part
            agents[agent_id] = {}
            for p in parts[1:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    agents[agent_id][k.strip()] = v.strip()
    return agents


def get_optimization_status(model_id):
    conn = sqlite3.connect(str(PROJECT_ROOT / "board.db"))
    cur = conn.cursor()
    cur.execute("SELECT optimization_status, optimization_owner FROM models WHERE model_id=?", (model_id,))
    row = cur.fetchone()
    conn.close()
    return row or (None, None)


def parse_message_notes(text):
    """Parse notes from a message text."""
    for line in text.split("\n"):
        if line.startswith("notes="):
            return line[6:].strip()
    return None


def parse_message_model_id(text):
    """Parse model_id from a message text."""
    for line in text.split("\n"):
        if line.startswith("model_id="):
            return line[9:].strip()
    return None


def parse_message_adaptation_path(text):
    """Parse adaptation_path from a message text."""
    for line in text.split("\n"):
        if line.startswith("adaptation_path="):
            return line[16:].strip()
    return None


def process_completed_message(model_id, adaptation_path, owner):
    """Process a completed optimization message."""
    # Read disk notes
    disk_path = PROJECT_ROOT / adaptation_path / "optimization_notes.json"
    if not disk_path.exists():
        log(f"  [WARN] Disk file not found: {disk_path}")
        return False
    try:
        with disk_path.open() as f:
            disk_notes = json.load(f)
    except Exception as e:
        log(f"  [ERROR] Failed to parse disk notes: {e}")
        return False

    # Try to update board
    notes_json = json.dumps(disk_notes)
    notes_escaped = notes_json.replace("'", "'\\''")
    out, err, code = run(f'update_optimization_status --model_id "{model_id}" --optimization_status "completed" --notes \'{notes_escaped}\'')

    if code == 0:
        log(f"  [OK] Updated board: {model_id} -> completed (speedup: {disk_notes.get('best_result', {}).get('speedup_ratio', 'N/A')}x)")
        return True
    elif "already completed" in err.lower():
        log(f"  [OK] Already completed: {model_id}")
        return True
    else:
        log(f"  [ERROR] Failed to update: {err[:200]}")
        return False


def process_failed_message(model_id, failure_reason, owner):
    """Process a failed optimization message."""
    # Classify failure
    retry_keywords = [
        "transformers",
        "meta tensor",
        "from_tf",
        "trust_remote_code",
        "OOM",
        "NPU",
        "超时",
        "queue",
        "heartbeat",
        "network",
        "403",
        "404",
        "gated",
        "LFS",
        "checkpoint",
        "missing",
        "unavailable",
        "version",
        "module",
        "not found",
        "dependency",
        "install",
        "import",
    ]
    failure_lower = (failure_reason or "").lower()

    if any(kw in failure_lower for kw in retry_keywords):
        status = "pending"
        log(f"  [RETRY] {model_id}: setting to pending (retryable)")
    else:
        status = "skipped"
        log(f"  [SKIP] {model_id}: setting to skipped")

    out, err, code = run(f'update_optimization_status --model_id "{model_id}" --optimization_status "{status}" --notes "{str(failure_reason or "")[:500]}"')
    log(f"  [DB] {out[:100]}")
    return True


def assign_task_to_optimizer(agent_id):
    """Assign a new optimization task to an optimizer."""
    out, err, code = run(f"assign_optimization_task --agent_id {agent_id}")
    if code != 0 or not out.strip():
        return False
    lines = out.strip().split("\n")
    if not lines or "Assigned optimization" not in lines[0]:
        return False

    parts = lines[0].split()
    model_id = parts[2]
    adaptation_path = lines[1].split("=", 1)[1].strip() if len(lines) > 1 else ""

    # Send message to optimizer
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
        log(f"  [ASSIGNED] {model_id} -> {agent_id}")
        return True
    return False


def read_inbox():
    if not os.path.exists(INBOX_PATH):
        return []
    with open(INBOX_PATH) as f:
        return json.load(f)


def write_inbox(msgs):
    with open(INBOX_PATH, "w") as f:
        json.dump(msgs, f, indent=2)


def main():
    log("=" * 60)
    log("NPU Optimization Monitoring Loop Started")
    log("=" * 60)

    completed_count = 0
    iteration = 0

    while True:
        iteration += 1
        now = datetime.now()

        # 1. Update heartbeat
        run('heartbeat --id "team-lead" --status "active" --task "monitoring"')

        # 2. Read inbox
        msgs = read_inbox()
        unread = [m for m in msgs if not m.get("read", False)]

        # 3. Process unread messages
        for m in unread:
            sender = m.get("from", "")
            text = str(m.get("text", m.get("content", "")))

            if sender.startswith("npu-optimizer"):
                if "result=completed" in text:
                    model_id = parse_message_model_id(text)
                    adaptation_path = parse_message_adaptation_path(text)
                    if model_id:
                        opt_status, owner = get_optimization_status(model_id)
                        if opt_status == "completed":
                            log(f"[{iteration}] Already completed: {model_id}")
                        else:
                            ok = process_completed_message(model_id, adaptation_path, sender)
                            if ok:
                                completed_count += 1
                        m["read"] = True

                elif "result=failed" in text:
                    model_id = parse_message_model_id(text)
                    failure_reason = None
                    for line in text.split("\n"):
                        if line.startswith("failure_reason="):
                            failure_reason = line.split("=", 1)[1].strip()
                    if model_id:
                        process_failed_message(model_id, failure_reason, sender)
                    m["read"] = True

                elif "status=idle" in text or '"type":"idle' in text or "ping" in text.lower():
                    m["read"] = True

        # 4. Write back read status
        write_inbox(msgs)

        # 5. Check agents and assign idle optimizers
        agents = get_agents()
        for agent_id, info in agents.items():
            if agent_id.startswith("npu-optimizer") and info.get("status") == "idle":
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
        if iteration % 5 == 0:
            log(f"[{iteration}] {now.strftime('%H:%M:%S')} | Completed: {completed_count} | In-progress: {in_progress} | Pending: {remaining}")

        # 8. Check if done
        if remaining == 0 and in_progress == 0:
            log(f"\n[COMPLETE] All optimization tasks done! Total completed: {completed_count}")
            break

        time.sleep(30)


if __name__ == "__main__":
    main()
