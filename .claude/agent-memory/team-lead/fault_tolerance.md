# Team-Lead 故障容错经验

## 2026-03-16: 团队目录丢失导致 Agent 超时

### 问题现象
- 4 个 npu-optimizer 启动后正常工作，创建了 `accuracy_run_perf.py` 和 `model_files/`
- 约 40 分钟后心跳停止，收件箱文件为空
- 团队目录 `~/.claude/teams/optimization-team/` 被删除

### 根本原因
- Claude Code 会话重置或清理时删除了团队目录
- Inbox 系统依赖团队目录，目录丢失后通信中断
- Agent 无法发送完成消息，最终超时

### 修复措施
1. **回收任务前检查部分产出**：
   - 检查 `accuracy_run_perf.py` 是否存在
   - 检查 `model_files/` 目录
   - 检查 `benchmark_metrics_*_perf.json`

2. **使用版本化团队名**：
   - 推荐 `optimization-team-v2` 而非 `optimization-team`
   - 避免与旧会话冲突

3. **备用状态同步**：
   - 当 inbox 不可用时，通过 board.db 和适配目录直接检查状态
   - `find adaptations/ -name "accuracy_run_perf.py" -newer board.db`

### 规则更新
已添加到 team-lead.md 第七章"故障容错规则"
