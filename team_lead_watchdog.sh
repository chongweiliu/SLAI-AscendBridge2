#!/bin/bash
set -uo pipefail
# 定时 kill + 重启编排，不依赖 claude 自动退出。
# 每轮重启会通过 run_auto_team_lead.sh -> board_ops reset
# 将第四阶段 business_benchmark 的 in_progress / wait_cuda 一并回退为 pending。
# 默认（适配模式）：nohup ./team_lead_watchdog.sh > /dev/null 2>&1 &
# 仅 benchmark：PROMPT_MODE=benchmark nohup ./team_lead_watchdog.sh > /dev/null 2>&1 &
# 仅 optimization：PROMPT_MODE=optimization nohup ./team_lead_watchdog.sh > /dev/null 2>&1 &
# 仅 business benchmark：PROMPT_MODE=business nohup ./team_lead_watchdog.sh > /dev/null 2>&1 &
# 仅 business benchmark（别名）：PROMPT_MODE=business_benchmark nohup ./team_lead_watchdog.sh > /dev/null 2>&1 &
# 注：不用 set -e，避免 pkill/lsof 无命中时导致脚本退出
# 停止：pkill -f team_lead_watchdog.sh 
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PROJECT_ROOT="${PROJECT_ROOT_OVERRIDE:-$SCRIPT_DIR}"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/team_lead_$(date +%Y%m%d).log"
LOCK_FILE="$LOG_DIR/.team_lead.lock"
INTERVAL_SECONDS=7200    # 2 小时

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT" || exit 1
export PATH=$PATH:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.npm-global/bin:$HOME/.local/bin

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Watchdog] $*" >> "$LOG_FILE"; }

while true; do
    log "定时重启：结束已有编排进程..."
    pkill -f "run_auto_team_lead.sh" 2>/dev/null || true
    # 结束可能继承锁的 claude 子进程
    if [ -f "$LOCK_FILE" ]; then
        PIDS=$(lsof "$LOCK_FILE" 2>/dev/null | awk 'NR>1 {print $2}' | tr '\n' ' ')
        for pid in $PIDS; do
            [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
        done
    fi
    pkill -9 -f "claude.*dangerously-skip-permissions" 2>/dev/null || true
    sleep 2
    rm -f "$LOCK_FILE"

    log "启动新一轮编排 (PID 将出现在下方日志)"
    # 子进程继承 PROMPT_MODE/PROMPT_FILES，用法：PROMPT_MODE=benchmark ./team_lead_watchdog.sh
    nohup "$SCRIPT_DIR/run_auto_team_lead.sh" >> "$LOG_FILE" 2>&1 &
    log "等待 ${INTERVAL_SECONDS} 秒后进行下一轮定时重启..."
    sleep $INTERVAL_SECONDS
done
