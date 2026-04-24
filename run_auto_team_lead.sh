#!/bin/bash
set -euo pipefail
# 单次执行，由 team_lead_watchdog.sh 定时 kill+重启。
#
# 用法：
#   ./run_auto_team_lead.sh
#     - 默认模式：inbox + task_adaptation
#
#   PROMPT_MODE=benchmark ./run_auto_team_lead.sh
#     - 仅评测阶段：inbox + task_benchmark
#
#   PROMPT_MODE=optimization ./run_auto_team_lead.sh
#     - 仅优化阶段：inbox + task_optimization
#
#   PROMPT_MODE=business ./run_auto_team_lead.sh
#   PROMPT_MODE=business_benchmark ./run_auto_team_lead.sh
#     - 仅第四阶段业务测评：inbox + task_business_benchmark
#
#   PROMPT_FILES="prompts/inbox.txt prompts/task_business_benchmark.txt" ./run_auto_team_lead.sh
#     - 自定义按顺序拼接 prompt 文件
#
# 推荐长期运行：
#   nohup ./team_lead_watchdog.sh > /dev/null 2>&1 &
#
# 获取脚本所在目录及项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PROJECT_ROOT="${PROJECT_ROOT_OVERRIDE:-$SCRIPT_DIR}"

# 设置日志文件路径
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/team_lead_$(date +%Y%m%d).log"

# 确保日志目录存在
mkdir -p "$LOG_DIR"

# 防重入：同一时间只允许一个编排实例
LOCK_FILE="$LOG_DIR/.team_lead.lock"
exec 200>"$LOCK_FILE"

# 切换到项目根目录，确保 claude 执行时的上下文正确
cd "$PROJECT_ROOT" || { echo "无法切换到项目目录: $PROJECT_ROOT" >> "$LOG_FILE"; exit 1; }

# 加载 .env（HF_TOKEN、HF_DATASETS_CACHE 等）
[ -f "$PROJECT_ROOT/.env" ] && set -a && source "$PROJECT_ROOT/.env" && set +a
# 确保 HF_DATASETS_CACHE 为绝对路径（子进程可能从 adaptation 目录运行）
[[ "${HF_DATASETS_CACHE:-}" != /* ]] && [ -n "${HF_DATASETS_CACHE:-}" ] && export HF_DATASETS_CACHE="$PROJECT_ROOT/$HF_DATASETS_CACHE"

# 设置 PATH，确保能找到 claude 命令 (根据实际情况调整)
export PATH=$PATH:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.npm-global/bin:$HOME/.local/bin

TEAMS_DIR="$HOME/.claude/teams"

# 单次执行一轮（间隔由看门狗控制）
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
echo "[$TIMESTAMP] === 开始执行自动化编排任务 ===" >> "$LOG_FILE"

# 日志轮转：按日已有（team_lead_YYYYMMDD.log）；压缩 7 天前的 .log，删除 30 天前的 .gz
find "$LOG_DIR" -maxdepth 1 -name "team_lead_*.log" -mtime +7 -type f -print0 2>/dev/null | xargs -0 -r gzip -f
find "$LOG_DIR" -maxdepth 1 -name "team_lead_*.log.gz" -mtime +30 -type f -delete 2>/dev/null

# 获取锁，防止与其它实例或 cron 重叠执行
if ! flock -n 200; then
    echo "[$TIMESTAMP] 已有实例在运行，本次退出。" >> "$LOG_FILE"
    exit 1
fi

# 启用 Agent Teams 模式（必须，否则 team-lead 无法使用 SendMessage/收件箱）
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1

# 检查 claude 命令是否存在
if ! command -v claude &> /dev/null; then
    echo "错误: 找不到 'claude' 命令。请检查 PATH 环境变量。" >> "$LOG_FILE"
        echo "当前 PATH: $PATH" >> "$LOG_FILE"
    exit 1
fi

# 第一步：清空 teams 收件箱与看板状态，避免继承上轮残留
{
    echo "[$TIMESTAMP] 清空 teams 目录内容: $TEAMS_DIR"
    mkdir -p "$TEAMS_DIR"
    find "$TEAMS_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
    echo "[$TIMESTAMP] teams 目录清空完成。"
    echo "[$TIMESTAMP] 清空看板: agents 表 + models.owner + in_progress→pending；business benchmark 的 wait_cuda 也会回退为 pending..."
    "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/board_ops.py" reset
    echo "[$TIMESTAMP] 看板清空完成。"
} >> "$LOG_FILE" 2>&1

echo "[$TIMESTAMP] 启动 team-lead..." >> "$LOG_FILE"

# 提示词加载：支持 PROMPT_MODE（adaptation|benchmark|optimization|business|business_benchmark）或 PROMPT_FILES（空格分隔路径）
# 用法：PROMPT_MODE=benchmark ./run_auto_team_lead.sh
# 或：  PROMPT_MODE=optimization ./run_auto_team_lead.sh
# 或：  PROMPT_MODE=business ./run_auto_team_lead.sh
# 或：  PROMPT_FILES="prompts/inbox.txt prompts/task_benchmark.txt" ./run_auto_team_lead.sh
PROMPTS_DIR="$PROJECT_ROOT/prompts"
if [ -n "${PROMPT_FILES:-}" ]; then
    APPEND_PROMPT=""
    for f in $PROMPT_FILES; do
        case "$f" in /*) path="$f" ;; *) path="$PROJECT_ROOT/$f" ;; esac
        [ -f "$path" ] && APPEND_PROMPT="$APPEND_PROMPT$(cat "$path")"
    done
elif [ -n "${PROMPT_MODE:-}" ]; then
    case "$PROMPT_MODE" in
        benchmark)
            [ -f "$PROMPTS_DIR/inbox.txt" ] && APPEND_PROMPT=$(cat "$PROMPTS_DIR/inbox.txt")
            [ -f "$PROMPTS_DIR/task_benchmark.txt" ] && APPEND_PROMPT="${APPEND_PROMPT:-}$(cat "$PROMPTS_DIR/task_benchmark.txt")"
            ;;
        optimization)
            [ -f "$PROMPTS_DIR/inbox.txt" ] && APPEND_PROMPT=$(cat "$PROMPTS_DIR/inbox.txt")
            [ -f "$PROMPTS_DIR/task_optimization.txt" ] && APPEND_PROMPT="${APPEND_PROMPT:-}$(cat "$PROMPTS_DIR/task_optimization.txt")"
            ;;
        business|business_benchmark)
            [ -f "$PROMPTS_DIR/inbox.txt" ] && APPEND_PROMPT=$(cat "$PROMPTS_DIR/inbox.txt")
            [ -f "$PROMPTS_DIR/task_business_benchmark.txt" ] && APPEND_PROMPT="${APPEND_PROMPT:-}$(cat "$PROMPTS_DIR/task_business_benchmark.txt")"
            ;;
        adaptation|*)
            [ -f "$PROMPTS_DIR/inbox.txt" ] && APPEND_PROMPT=$(cat "$PROMPTS_DIR/inbox.txt")
            [ -f "$PROMPTS_DIR/task_adaptation.txt" ] && APPEND_PROMPT="${APPEND_PROMPT:-}$(cat "$PROMPTS_DIR/task_adaptation.txt")"
            ;;
    esac
fi
if [ -z "${APPEND_PROMPT:-}" ]; then
    echo "[$TIMESTAMP] 错误: 未加载到提示词。请设置 PROMPT_MODE=adaptation|benchmark|optimization|business|business_benchmark 或 PROMPT_FILES=..." >> "$LOG_FILE"
    exit 1
fi

export NODE_OPTIONS="--max-old-space-size=131072"
claude -p \
  --agent team-lead \
  --dangerously-skip-permissions \
  --teammate-mode in-process \
  --append-system-prompt "$APPEND_PROMPT" \
  "开始执行" >> "$LOG_FILE" 2>&1

EXIT_CODE=$?
END_TIME=$(date "+%Y-%m-%d %H:%M:%S")

# 确保新起一行，避免与 claude 最后一行无换行的输出拼在一起
echo "" >> "$LOG_FILE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "[$END_TIME] 任务执行成功完成。" >> "$LOG_FILE"
else
    echo "[$END_TIME] 任务执行失败或异常退出 (Exit Code: $EXIT_CODE)。" >> "$LOG_FILE"
fi

flock -u 200
exit $EXIT_CODE
