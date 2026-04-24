#!/bin/bash
# monitor_optimization_loop.sh - Team-Lead monitoring loop for optimization tasks
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Ignore inherited PROJECT_ROOT from the parent shell so copied worktrees resolve to themselves.
export PROJECT_ROOT="${PROJECT_ROOT_OVERRIDE:-$DEFAULT_PROJECT_ROOT}"
export TEAM_NAME="${TEAM_NAME:-optimization-team-v10-1774451398}"
VENV="$PROJECT_ROOT/.venv/bin/python"
BOARD_OPS="$VENV $PROJECT_ROOT/scripts/board_ops.py"
READ_INBOX="$VENV $PROJECT_ROOT/scripts/read_inbox.py"
HEARTBEAT_INTERVAL=180  # 3 minutes
CHECK_INTERVAL=30       # 30 seconds between checks

last_heartbeat=0
last_inbox_read=0

echo "[$(date)] Monitor loop started. Team: $TEAM_NAME"

while true; do
    now=$(date +%s)

    # ---- 1. Read inbox (primary) ----
    # Get all unhandled messages from last 10 minutes
    if [ $((now - last_inbox_read)) -ge 30 ]; then
        $READ_INBOX --team "$TEAM_NAME" --agent "team-lead" --since 10 > /tmp/inbox_check_$$.txt 2>&1 || true
        last_inbox_read=$now
    fi

    # ---- 2. Check agent heartbeats ----
    AGENTS=$($BOARD_OPS list_agents 2>&1 | grep -v "^$")
    echo "[$(date)] Agent status:" >&2
    echo "$AGENTS" | while read -r line; do
        if echo "$line" | grep -q "last_heartbeat"; then
            echo "  $line" >&2
        fi
    done

    # ---- 3. Check in_progress optimization tasks ----
    echo "[$(date)] In-progress optimization tasks:" >&2
    $BOARD_OPS list_optimization_tasks --status "in_progress" 2>&1 | head -20 >&2

    # ---- 4. Check pending count ----
    pending=$($BOARD_OPS list_optimization_tasks --status "pending" 2>&1 | grep -c "model_id" || echo 0)
    echo "[$(date)] Pending optimization tasks: $pending" >&2

    # ---- 5. Update heartbeat every HEARTBEAT_INTERVAL seconds ----
    if [ $((now - last_heartbeat)) -ge $HEARTBEAT_INTERVAL ]; then
        $BOARD_OPS heartbeat --id "team-lead" --status "active" \
            --task "监控中: pending=$pending" 2>&1 || true
        last_heartbeat=$now
        echo "[$(date)] Heartbeat updated" >&2
    fi

    # ---- 6. Sleep ----
    echo "[$(date)] Sleeping $CHECK_INTERVAL seconds..."
    sleep $CHECK_INTERVAL
done
