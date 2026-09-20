---
name: team-orchestration-pitfalls
description: 团队编排历史踩坑合集（spawn 竞态/僵尸心跳/团队隔离/TeamDelete 卡死/session compaction 无限提醒/board_ops notes 传参），详细流程以 team-lead.md 系统提示词为准
metadata:
  type: feedback
---

# 团队编排踩坑合集（2026-03~04 实战沉淀）

以下规则大多已固化进 `.claude/agents/team-lead.md` 系统提示词（以那边为准），此处仅存根与补充细节。

## Spawn 与团队管理
- **严禁并行 spawn**：一次性 `Task()` 启动多个 agent 会竞态写 team config，只有最后一个成功。必须逐个 spawn + 30s 间隔 + 完成后 90s 等首次心跳。详见 `spawn_best_practices.md`
- **team_name 禁止 default**：必须带版本号（如 optimization-team-v10）；新 session 先 `rm -rf ~/.claude/teams/` 清旧团队
- **团队隔离 bug**（2026-03-22）：optimizer 进程可能注册进另一个团队的 inbox，SendMessage 永远不达。排查：`find ~/.claude/teams/ -name config.json -exec grep -l "{agent}" {} \;`

## 僵尸 agent 判定（最高优先规则）
- **board_ops 心跳 active ≠ agent 存活**。分配任务前必须验证 `~/.claude/teams/{team}/inboxes/{agent}.json` 存在；inbox 缺失 = 僵尸，不分配
- 同名 agent 跨 session 会共享 board.db 心跳行互相覆盖；agent 必须从 team config.json 读自己的真实 name，禁止硬编码

## TeamDelete 卡死
- shutdown_response ≠ 从 config.json 移除 member。卡死时直接编辑 config.json 把 members 清到只剩 team-lead 再 TeamDelete
- 大模型任务（>1GB）的 agent 被 shutdown 时极易卡死；强制关闭前先等任务完成或超时回收
- **session compaction 后 "Shut down your team" 无限循环**是 orchestrator 侧 bug（2026-03 实测超 200 轮无法自救）：每轮回复"继续监控，任务未完成"，不执行 TeamDelete、不退出

## board_ops 操作细节
- **传 notes 含 JSON 时不要走 shell**：`--notes "$(cat file.json)"` 会被 shell 引号/空格破坏。用 Python 直接调 `board_ops.update_optimization_status(model_id=..., notes=raw_text)`
- adaptation_path 的 `/` 一律换 `_`（sanitized name），验证：`ls adaptations/ | grep xxx`
- 验收陷阱：best_result.baseline_latency_s 必须精确匹配 non-perf 工件（≤1e-3 或 2%）；pretrained 的 baseline 只能配 mode=pretrained 工件；speedup≥3x 须 independent_baseline_artifact + comparison_scope + validation_note + steady_state 双值；warmup 工件的假 speedup 禁用
- 详见 `optimization_accuracy.md`、`fault_tolerance.md`、`git_ci_ops.md`、`batch_records.md`
