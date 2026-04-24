# Team-lead 提示词片段

通过 `PROMPT_MODE` 选择任务类型，或直接指定 `PROMPT_FILES` 组合。

## 用法

```bash
# 默认：全量适配（inbox + task_adaptation）
./run_auto_team_lead.sh

# 仅 benchmark
PROMPT_MODE=benchmark ./run_auto_team_lead.sh

# 仅 NPU 优化
PROMPT_MODE=optimization ./run_auto_team_lead.sh

# 仅第四阶段业务测评
PROMPT_MODE=business ./run_auto_team_lead.sh

# 同上，也支持显式写 business_benchmark
PROMPT_MODE=business_benchmark ./run_auto_team_lead.sh

# 自定义：指定多个文件（按顺序拼接）
PROMPT_FILES="prompts/inbox.txt prompts/task_benchmark.txt" ./run_auto_team_lead.sh
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `inbox.txt` | 收件箱通用说明（所有模式共用） |
| `task_adaptation.txt` | 全量适配任务（抓取+适配） |
| `task_benchmark.txt` | 仅 benchmark 评测任务 |
| `task_optimization.txt` | 仅 NPU 优化任务 |
| `task_business_benchmark.txt` | 仅第四阶段业务测评任务 |

## Prompt 约束

- `prompts/task_*.txt` 中写明的 agent 数量、并发数、可执行阶段范围都是**硬约束**。
- 对 team-lead 来说，**写几个就是几个**；不得为了“更快”或“机器空闲”擅自增加、删减或替换人数。
- `.claude/agents/team-lead.md` 里的示例人数只是示例，不是默认值；真实执行始终以本轮实际加载的 `prompts/task_*.txt` 原文为准。

## SSH 约定

涉及远端 CUDA / 远端执行时，推荐统一只向 agent 提供：

- `remote_ssh_host=<ssh alias>`
- `remote_project_root=<remote repo root>`
- 第四阶段命令显式使用 `uv run --extra ascend ...` / `uv run --extra cuda ...`

例如：

```text
remote_ssh_host=cuda-remote
remote_project_root=$SLAI_REMOTE_PROJECT_ROOT
```

真实 `HostName / Port / User / IdentityFile` 应保存在执行机自己的 `~/.ssh/config`，不要写进仓库 prompt、`.env`、README、已提交 JSON 或 notes。

## 新增模式

1. 在 `prompts/` 下新建 `task_xxx.txt`
2. 在 `run_auto_team_lead.sh` 的 case 中增加对应分支，或使用 `PROMPT_FILES` 直接指定
