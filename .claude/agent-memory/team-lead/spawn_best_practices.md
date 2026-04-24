# Teammate Spawning 最佳实践

## 关键发现（2026-03-22）

### 并行 spawn 竞态问题

**现象**：一次性 `Task()` 并行启动 4 个 npu-optimizer，只有 1 个能注册到 team config，其余 3 个因竞态丢失。

**原因**：Claude Code 的 team 机制在并发 spawn 时，第一个 `Task()` 会创建团队并触发 join 逻辑，
后续并行的 `Task()` 同时尝试 join，但 team config 写入是串行的，导致只有最后一个写入成功。

**实测结果**：
- 并行 spawn 4 个 optimizer → 只有 npu-optimizer-4 进入 config
- 补召第 1 个 → npu-optimizer-1 进入
- 补召第 2 个 → npu-optimizer-2 进入
- 补召第 3 个 → npu-optimizer-3 进入

### 正确的 Spawn 流程（已实测成功）

```bash
# 1. 清理旧团队
rm -rf ~/.claude/teams/

# 2. 创建团队
TeamCreate(team_name="optimization-team-v12", ...)

# 3. 等待 30 秒让 team config 稳定
sleep 30

# 4. 逐个 spawn（不要并行），每 spawn 一个等待 30 秒
Task(name="npu-optimizer-1", ...)
sleep 30
Task(name="npu-optimizer-2", ...)
sleep 30
Task(name="npu-optimizer-3", ...)
sleep 30
Task(name="npu-optimizer-4", ...)

# 5. 等待 90 秒让所有 agent 完成首次心跳
sleep 90

# 6. 验证 config.json 中所有 4 个 member 都存在
cat ~/.claude/teams/{team_name}/config.json | python3 -c \
  "import json,sys; cfg=json.load(sys.stdin); [print(m['name']) for m in cfg['members']]"

# 7. 验证所有 4 个 agent 心跳已注册
.venv/bin/python scripts/board_ops.py list_agents
```

**注意**：30 秒 spawn 间隔是安全的，90 秒等待也足够了。全部完成后 list_agents 应该看到 4 个 idle optimizer。

### 验证清单

Spawn 完成后必须验证：

1. **config.json members 数量正确**：4 个 optimizer 都应该在 members 数组里
2. **board_ops list_agents 显示 4 个心跳**：所有 agent 都能发心跳到 board.db
3. **inbox 文件存在**（可选，不影响功能）：`~/.claude/teams/{team}/inboxes/npu-optimizer-N.json`
   - 注：即使 inbox 文件不存在，只要 config 有成员且 board_ops 心跳正常，SendMessage 仍能送达
   - inbox 文件是"收件箱读取"的备份来源，不是通信的必要条件
4. **idle 状态正确**：所有 optimizer 都应显示 `status=idle, current_task=等待分配任务`

### SendMessage 送达性验证

只要 `SendMessage` 返回 `success=true`，消息就一定能送达。无需验证 inbox 文件存在。

**唯一需要 inbox 文件的场景**：当 teammate-message 通道没有收到消息、但怀疑 optimizer 可能发过消息时，
用 `read_inbox.py --since 30` 读取以捕获漏掉的消息。

### 补召策略

当发现 config 成员数量不对时：

```bash
# 检查 config 中缺失的成员
cat config.json | python3 -c "import json,sys; cfg=json.load(sys.stdin); print([m['name'] for m in cfg['members']])"

# 只 spawn 缺失的那些，不要全部重新 spawn
Task(name="npu-optimizer-1", ...)  # 如果缺失
# 等待 30 秒后再补下一个
```

### team_name 命名规范

- **禁止** `default`（系统默认，容易冲突）
- 必须带版本号：`optimization-team-v12`、`optimization-team-v13`...
- 新 session 每次递增版本号
- 版本号使用 v10 开始规避旧 session 的遗留问题

### 心跳注册延迟

Agent 启动后需要 **30-90 秒**才能首次发心跳到 board.db。
- 不要 spawn 后立即 `list_agents`，否则可能看不到所有 agent
- 建议：spawn 完成后 sleep 90 秒，再验证心跳

### team-lead 自身心跳注册（必须！！）

**新 session 开始时，team-lead 必须立即注册自己的心跳**，否则 `board_ops list_agents` 看不到自己：

```bash
# 在任何 spawn 之前或之后第一时间执行
.venv/bin/python scripts/board_ops.py heartbeat --id "team-lead" --status "active" --task "初始化完成"
```

这条心跳是 team-lead 的"身份证"，后续 `list_agents` 才能看到完整（包括自己）的 agent 列表。
该命令在任何环节都可以随时重复执行，用于刷新自己的状态。
