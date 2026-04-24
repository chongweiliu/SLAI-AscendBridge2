---
name: database-ops
description: 使用 $PROJECT_ROOT/scripts/board_ops.py 管理 board.db。处理任务分配、心跳、状态更新。
---

# 看板数据库操作 Skill

所有与 `board.db` 的交互均通过 `$PROJECT_ROOT/scripts/board_ops.py` 完成。**禁止直接写 SQL。**

## 常用命令

### 1. 心跳（每 5–10 分钟执行一次）

Subagent（adapter、model-crawler、benchmark-runner、npu-optimizer）首次出现在 agents 表依赖首次 heartbeat，应在启动后 **2 分钟内**执行一次。

```bash
$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/board_ops.py heartbeat --id "your-agent-id" --status "active" --task "当前任务描述"
```

*若 `--id "team-lead"`，`board_ops.py` 会自动把本机 IP 和当前进程号追加到 `agents.current_task`，格式为 `... | host_ip=<local-ip> | pid=<process-id>`；不要手工重复拼 IP/PID。*

### 2. 注册新模型（Model Crawler）

```bash
$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/board_ops.py register_model --model_id "org/name" --source "huggingface" --adaptation_status "pending" --url "https://huggingface.co/org/name"
```

*`--url` 必填且非空；且必须唯一，若该 url 已存在则跳过注册。*

### 3. 分配任务（team-lead）

```bash
$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/board_ops.py assign_adaptation_task --agent_id "adapter-N"
```

*`assign_adaptation_task` 会在分配时将该任务的 `adaptation_path` 按 `model_id` 统一推导并写入 `models` 表。*

### 4. 更新任务状态（仅 team-lead）

Adapter 不写看板，只通过 SendMessage 上报；由 team-lead 根据报告调用：

```bash
$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/board_ops.py update_adaptation_status --model_id "org/name" --adaptation_status "completed" --adaptation_notes "验证通过" --adaptation_path "adaptations/org_name"
```

*`adaptation_path` 在分配时已由 `assign_adaptation_task` 写入；`update_adaptation_status` 仅在 `adaptation_status=completed` 时可根据传入的 `--adaptation_path` 或目录存在性补写/覆盖。team-lead 从 Adapter 消息中取 model_id、adaptation_path、adaptation_notes 或 adaptation_failure_reason 后执行。`adaptation_status=completed` 会触发 git commit/push，终态会将 `adaptation_owner` 设回 `idle`。**若 `check_adaptation.py` 未通过**，board_ops 不修改 DB、不 commit，exit 1，team-lead 解析 INTERCEPTED 并通知 adapter 修复。`adaptation_status=pending` 用于超时回收：清空 `adaptation_owner`、将任务回退为待分配，以便重新分配。*

失败时（技术错误用 `skipped`，授权问题用 `needs_authorization`）：

```bash
$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/board_ops.py update_adaptation_status --model_id "org/name" --adaptation_status "skipped" --adaptation_notes "OOM error"
```

### 5. 列出 Agent（team-lead）

```bash
$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/board_ops.py list_agents
$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/board_ops.py list_agents --status "idle"
```

*Assign adaptation task 成功后会：将该 agent 标为 `active`；更新该任务的 `adaptation_started_at` 为分配时间；`update_adaptation_status` 完成/失败后会将其标回 `idle`。*

### 6. 列出任务

```bash
$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/board_ops.py list_adaptation_tasks --status "pending"
$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/board_ops.py list_adaptation_tasks --status "in_progress"
```

*分配前用 `--status "in_progress"` 查看已分配任务及 `adaptation_owner`，避免重复分配给其他 adapter。*

### 6.1 统一回退模型状态（team-lead）

只需要输入模型列表和目标状态；模型列表支持空格 / 逗号 / 换行分隔。

```bash
$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/board_ops.py rollback_models --models "org/a org/b" --to optimization:pending
```

*`--to` 格式为 `<stage>:<status>`，支持：`adaptation:{pending|skipped|not_applicable|needs_authorization}`、`benchmark:{pending|skipped|not_applicable}`、`optimization:{pending|skipped|not_applicable}`、`business_benchmark:{pending|skipped|not_applicable}`。回退上游阶段时，会自动把 downstream 阶段级联重置为 `pending`。*

### 7. Benchmark 相关命令（team-lead）

**分配评测任务**（仅限 adaptation_status=completed 且 benchmark_status=pending 的模型）：

```bash
$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/board_ops.py assign_benchmark_task --agent_id "benchmark-runner-N"
```

**列出评测任务**：

```bash
$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/board_ops.py list_benchmark_tasks --status "pending"
$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/board_ops.py list_benchmark_tasks --status "in_progress"
```

*分配前用 `--status "in_progress"` 查看已分配评测任务及 benchmark_owner，避免重复分配。*

**更新评测状态**（**仅 team-lead** 根据 Benchmark-Runner 的 SendMessage 调用；**Benchmark-Runner 禁止调用**，避免重复 commit）：

```bash
$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/board_ops.py update_benchmark_status --model_id "org/name" --benchmark_status "completed" --notes "评测完成"
```

*有效 benchmark_status*：`completed`、`skipped`、`not_applicable`、`pending`、`in_progress`。适配任务标记为 completed 时，看板会自动写入 `benchmark_status='pending'`，无需额外操作。**若 check_accuracy_run.py 未通过**，board_ops 不修改 DB、不 commit，exit 1，team-lead 解析 INTERCEPTED 并通知 benchmark-runner 修复。

### 8. Optimization 相关命令（team-lead）

**状态链式依赖**：`adaptation_status=completed` → `benchmark_status=completed` → `optimization_status=completed` → `business_benchmark_status=completed`。只有前序状态为 completed 时，才能将后序状态设为 completed。

**分配 NPU 优化任务**（仅限 benchmark_status=completed 且 optimization_status=pending 的模型）：

```bash
$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/board_ops.py assign_optimization_task --agent_id "npu-optimizer-N"
```

**列出 NPU 优化任务**：

```bash
$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/board_ops.py list_optimization_tasks --status "pending"
$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/board_ops.py list_optimization_tasks --status "in_progress"
```

**更新 NPU 优化状态**（**仅 team-lead** 根据 npu-optimizer 的 SendMessage 调用）：

```bash
$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/board_ops.py update_optimization_status --model_id "org/name" --optimization_status "completed" --notes "优化完成"
```

*有效 optimization_status*：`completed`、`skipped`、`not_applicable`、`pending`、`in_progress`。benchmark 完成时看板会自动写入 `optimization_status='pending'`。**拦截条件**：`optimization_status=completed` 前必须 (1) `adaptation_status=completed` 且 `benchmark_status=completed`（链式依赖）；(2) 存在 `accuracy_run_perf.py`；(3) **`check_accuracy_run_perf.py` 通过**。任一不满足则 board_ops 不修改 DB、exit 1，team-lead 解析 INTERCEPTED 并通知 npu-optimizer 修复。

**新的状态选择规则（强制）**：
- `completed`：允许两类正式优化结果
  - 融合算子路径（`fusion` / `hybrid`）
  - 纯运行时路径（`runtime_only`），即**不替换算子**，仅通过 `warmup + TASK_QUEUE_ENABLE` 等手段获得真实提速；但仍必须是 `pretrained`、`num_samples >= 50`、baseline/perf 工件同模式/同数据集/同口径，并在 `optimization_notes` 中显式记录 `optimization_kind=runtime_only`、`selected_npu(s)`、`device_topology`、`parallel_mode`
- `pending`：所有**可重试**问题都必须回 `pending`，并使用结构化 JSON notes，至少包含 `reason_code`、`retryable`、`recommended_action`、`evidence`、`next_step`
  - 包括：版本兼容、依赖缺失、OOM/资源不足/超时、多卡才能跑、样本不足 `<50`、数据集不合适、baseline/perf 工件缺失、工件口径不一致、measurement bug、重复分配但无新产物、空备注、以及“融合算子精度回归但尚未完成 runtime-only 尝试”
- `skipped`：仅允许在**已经完成 runtime-only 尝试**后，确认没有真实提速时使用；notes 必须是结构化 JSON，且 `reason_code` 只能是 `true_no_gain_after_runtime_only` / `fusion_regression_runtime_only_no_gain` / `precision_regression_runtime_only_no_gain`
- `not_applicable`：仅允许在**已经完成 runtime-only 尝试**后，确认模型架构或格式对当前优化路径确实不适用时使用；notes 必须是结构化 JSON，且 `reason_code` 只能是 `architecture_not_applicable_after_runtime_only` / `runtime_unsupported_irrecoverable` / `model_format_irrecoverable`

**样本不足补救规则（新增）**：
- 若 `num_samples < 50`，禁止直接写 `skipped`
- 必须先用 `python scripts/dataset_mapping.py --model_id "{model_id}" --candidates --json` 查看候选数据集
- 再用 `python scripts/download_datasets.py --model-id "{model_id}" --candidate-datasets` 下载候选数据集并重测

**工件脏数据补救规则（新增）**：
- baseline/perf artifact 与 `optimization_notes` 不一致、旧工件污染、缺少 baseline/perf artifact、或 mode/dataset/dtype/output_type 混淆时，必须删除冲突旧工件并重跑
- 此类问题统一回 `pending`，不得写成 `skipped`

**执行顺序要求（新增）**：
- npu-optimizer 接到优化任务后，第一步必须运行 `python benchmark/scripts/check_accuracy_run.py --adapt {sanitized_name}`
- 若 `accuracy_run.py` 未通过，必须先修复 `accuracy_run.py` 并重跑检查
- 只有 `accuracy_run.py` 已通过后，才允许继续创建/修改 `model_files/`、`accuracy_run_perf.py` 和 `_perf` 产物

**重点处理样例**：
- `lblueee/t5-academic-title-generator-model`：`from_tf` / `transformers 5.x` 属于版本/代码可修，默认回 `pending`；**首选方案**是在 adaptation 独立环境中逐步降级到兼容的 `transformers 4.x` 并 pin
- `ibm-research/biomed.rna.bert.110m.wced.v1`：缺 `bmfm_targets` / Lightning checkpoint 属于补依赖或转权重可修，默认回 `pending`

**严格同步规则（新增，必须执行）**：
- `--notes` **必须**传入 `adaptation_path/optimization_notes.json` 的完整原文 JSON；严禁传自然语言摘要
- 在调用 `update_optimization_status --optimization_status completed` 前，必须先运行：
  - `python benchmark/scripts/check_accuracy_run.py --adapt {sanitized_name}`
- `python optimization/scripts/check_accuracy_run_perf.py --adapt {sanitized_name}`
  - `python optimization/scripts/check_optimization_notes.py --adapt adaptations/{sanitized_name}`
  - 若 baseline 证据链依赖 `accuracy_run.py`，则必须先修复 `accuracy_run.py` 并重生成 pretrained baseline artifact，再跑 perf
  - 对比消息中的 `notes` 与磁盘 `optimization_notes.json`，必须逐字一致
- `optimization_status=completed` 时，`optimization_notes` 还必须满足：
  - `results[]` 至少包含一条真实 `pretrained` 结果
  - `best_result.mode` 必须为 `pretrained`
  - config-only 结果不得标记 `completed`
  - 若走纯运行时提速路径，必须显式写 `optimization_kind=runtime_only`
  - `results[]` / `best_result` 中必须包含数值型 `num_samples` / `baseline_latency_s` / `speedup_ratio`
  - `best_result.num_samples` 必须 >= `50`
  - `best_result.speedup_ratio` 默认必须大于 `1.0`；仅当 `optimization_kind=runtime_only` 且在 notes 中满足 `code_modified=false`、`code_change_attempts>=2`、并明确注明模型代码无更改时，允许 `best_result.speedup_ratio = 1.0`（`<1.0` 仍必须回退 `pending`）
- `results[]` 至少应包含 `dtype/mode/dataset/output_type/baseline_artifact/perf_artifact/num_samples/perf_latency_s/perf_memory_mb/baseline_latency_s/speedup_ratio/cosine_similarity`
  - baseline artifact 文件名必须能明确区分 `pretrained/config`，不得让不同 mode 的工件互相覆盖或无法匹配
  - 若 `best_result.speedup_ratio >= 3.0`：
    - `comparison_method` 必须为 `independent_baseline_artifact`
- `precision_method` 不得包含任何 `self_baseline` 系取值
- `comparison_scope` 必须为 `cold_start` / `steady_state` / `mixed`
- `validation_note` 必须写明高倍提速核查结论
- `steady_state_baseline_latency_s` 与 `steady_state_perf_latency_s` 必须为正数
  - 若 `best_result` 写入了 `baseline_latency_s` / `speedup_ratio`，则 adaptation 目录中的 baseline/perf `benchmark_metrics*.json` 必须数值一致；冲突旧工件必须先重生成或删除
  - `speedup_ratio` 统一按整轮 wall-clock 计算，即 `baseline_wall_clock_s / perf_wall_clock_s`；`baseline_latency_s / perf_latency_s` 仅保留为前向延迟证据，不再作为 completed 判定的官方 speedup
- 调用成功后，必须立即查询 `board.db` 核验：
  - `optimization_notes` 非空
  - `optimization_notes` 是合法 JSON object
  - 至少包含 `optimizations` / `results` / `best_result`
- 若写后核验失败，不得视为完成，必须重新用磁盘上的 `optimization_notes.json` 原文回写 DB

### 9. Business Benchmark 相关命令（team-lead）

**状态链式依赖**：只有 `optimization_status=completed` 的模型，才允许进入第四阶段业务测评。

**分配业务测评任务**（仅限 optimization_status=completed 且 business_benchmark_status=pending 的模型）：

```bash
$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/board_ops.py assign_business_benchmark_task --agent_id "business-benchmark-N"
```

**列出业务测评任务**：

```bash
$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/board_ops.py list_business_benchmark_tasks --status "pending"
$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/board_ops.py list_business_benchmark_tasks --status "in_progress"
```

**更新业务测评状态**（仅 team-lead 根据人工/远端回传结果调用）：

```bash
$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/board_ops.py update_business_benchmark_status --model_id "org/name" --business_benchmark_status "completed" --notes "$(cat adaptations/{sanitized_name}/business_summary.json)"
```

*有效 business_benchmark_status*：`completed`、`skipped`、`not_applicable`、`pending`、`in_progress`、`wait_cuda`。`optimization_status=completed` 时，看板会自动写入 `business_benchmark_status='pending'`。其中 `wait_cuda` 表示“本机 NPU 已完成，但远端 CUDA 因 SSH 不通/自动回收失败，只能等待人工回传或后续补跑”；写入 `wait_cuda` 前必须先通过 `python business_benchmark/scripts/check_business_benchmark_run.py --adapt {sanitized_name} --wait-cuda-npu-only`，确认本机 NPU baseline/perf 双路没有全 0、单路塌 0、质量漂移异常、`npu_speedup_ratio < 0.9`、字段缺失或 `num_samples <= 50`；否则应回 `pending` 修本机链路，禁止进入 `wait_cuda`。写入 `wait_cuda` 时会释放 owner，但不会触发 completed gate。**拦截条件**：`business_benchmark_status=completed` 前必须 (1) `adaptation_status=completed`、`benchmark_status=completed`、`optimization_status=completed`；(2) `business_summary.json` 存在且与 `--notes` 逐字一致；(3) `check_business_benchmark_run.py --adapt {sanitized_name}` 通过；(4) 同时存在 `business_metrics_npu_*_baseline.json`、`business_metrics_npu_*_perf.json`、`business_metrics_cuda_*_baseline.json` 三类工件；(5) `business_summary.json` 与三类业务工件中的 `num_samples` 都必须 **大于 50**。任一不满足则 board_ops 不修改 DB、exit 1。*
