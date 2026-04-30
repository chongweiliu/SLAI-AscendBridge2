# SLAI-AscendBridge2

`SLAI-AscendBridge2` 是一款面向华为昇腾 NPU 的自动化智能体编排与单模型适配框架，用于将 PyTorch 模型迁移到 Ascend。它支持从模型发现、环境治理、代码适配、精度评测到 NPU 性能优化的闭环流程；如果需要，也可以继续扩展到第四阶段 `business_benchmark`。

这个仓库是**框架仓**，负责脚本、检查器、调度骨架、dashboard、`.claude` 下的 agents / skills / agent-memory，以及 prompt 模板，不默认携带公开 adaptation 集合。模型级 adaptation 建议放在独立仓库 `SLAI-AscendBridge2-Adaptations`，或按你的内部目录结构单独维护。

为兼容不同 agent / IDE 的入口约定，仓库根目录额外保留了两个软链接：

- `.agents -> .claude`
- `AGENTS.md -> CLAUDE.md`

## 仓库定位

- `SLAI-AscendBridge2`
  - 框架、脚本、检查器、dashboard、`.claude` agents / skills / agent-memory
  - 适合批量调度、多阶段闭环、统一状态管理
- `SLAI-AscendBridge2-Adaptations`
  - 模型级 `adaptations/{name}/` 产物集合
  - 适合单模型复用、逐步公开、按需筛选发布

## 仓库文件架构

```text
SLAI-AscendBridge2/
├── .claude/
│   ├── agent-memory/                  # 智能体阶段记忆与规则沉淀
│   ├── agents/                        # 智能体定义
│   └── skills/                        # 技能库
├── .agents -> .claude                 # 兼容部分 agent 工具的目录入口
├── AGENTS.md -> CLAUDE.md             # 兼容 AGENTS.md 约定
├── adaptation/
│   └── scripts/
│       ├── adaptation_manager.py      # 第一阶段运行与产物管理
│       ├── check_adaptation.py        # demo.py / adaptation 完成度检查
│       ├── package_adaptations.py     # adaptation 打包
│       └── run_completed_adaptations.py
├── benchmark/
│   ├── scripts/
│   │   ├── benchmark_tool.py          # 聚合、对比、trace/profiling 分析
│   │   ├── benchmark_manager.py       # benchmark 运行与产物管理
│   │   └── check_accuracy_run.py      # accuracy_run.py 规范检查
│   ├── figures/
│   └── reports/
├── optimization/
│   ├── scripts/
│   │   ├── optimization_tool.py       # 优化数据聚合与对比
│   │   ├── optimization_manager.py    # 优化运行管理
│   │   ├── check_accuracy_run_perf.py # accuracy_run_perf.py 结构检查
│   │   └── check_optimization_notes.py
│   └── reports/
├── business_benchmark/
│   ├── scripts/
│   └── templates/
├── dashboard/                         # Web 看板
├── prompts/                           # team-lead prompt 模板
├── scripts/
│   ├── board_ops.py                   # 看板 CRUD、心跳、任务分配
│   ├── get_model_info.py              # 模型元数据提取
│   ├── download_datasets.py           # 数据集下载
│   ├── dataset_mapping.py             # 模型 -> 数据集映射
│   ├── ensure_agent_symlinks.sh       # 恢复 .agents / AGENTS.md 兼容软链接
│   └── sanitize_repo_paths.py         # 路径清洗工具
├── tests/
├── adaptations/                       # 默认空目录，可挂载或拷贝单模型 adaptation
├── CLAUDE.md                          # 项目上下文说明
├── pyproject.toml
├── uv.lock
└── README.md
```

`board.db` 由本地初始化生成，默认不纳入版本控制；首次使用时通过 `scripts/board_ops.py init` 创建空数据库。

`.claude/agent-memory` 已随框架仓一起提供，用于沉淀 `adapter`、`benchmark-runner`、`npu-optimizer`、`team-lead` 等角色的阶段记忆和经验规则；如果你要复用 Claude 的多智能体工作流，建议完整保留整个 `.claude/` 目录。

通过 `git clone` 获取仓库时，上述两个软链接会一并保留。  
如果你是通过压缩包、文件管理器拖拽、或不保留符号链接的同步方式获取目录，建议执行：

```bash
bash scripts/ensure_agent_symlinks.sh
```

## 协作流程

```text
Discovery -> Planning -> Adaptation -> Healing -> Benchmark -> Optimization -> [Optional] BusinessBenchmark -> Sync
```

- `Discovery`
  - 搜索模型、提取元数据、去重、确定目标模型
- `Planning`
  - `team-lead` 分发任务，约束阶段范围和优先级
- `Adaptation`
  - 生成 `adaptations/{sanitized_model_id}/demo.py`、`pyproject.toml`、README 等基础产物
- `Healing`
  - 循环执行 `uv run`，根据 Traceback 自愈修复
- `Benchmark`
  - 生成 `accuracy_run.py`，输出 `outputs_*.pt`、`benchmark_metrics_*.json`、`trace_*.json`
- `Optimization`
  - 生成 `accuracy_run_perf.py`、`model_files/`、`optimization_notes.json`，验证精度与性能
- `[Optional] BusinessBenchmark`
  - 在真实数据和真实权重下补齐 NPU/CUDA 侧业务测评
- `Sync`
  - 验收产物、更新状态、提交代码

## 交付标准

### Adaptation

- 完整的 `pyproject.toml` + `uv.lock`
- 代码包含 `torch_npu` 逻辑
- 支持 DRY RUN 模式
- `uv run python demo.py` 成功运行
- 无明显不适合公开的内容

### Benchmark

- `accuracy_run.py` 通过 `benchmark/scripts/check_accuracy_run.py`
- 产出 `outputs_*.pt`、`benchmark_metrics_*.json`、`trace_*.json`

### Optimization

- `accuracy_run_perf.py` 通过 `optimization/scripts/check_accuracy_run_perf.py`
- `optimization_notes.json` 通过格式校验
- 在精度近似前提下实现性能提升，或清楚记录 runtime-only 路径原因

## 环境准备

### 1. 安装 `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 准备工作目录

如果你已经有本地副本，直接进入仓库目录即可：

```bash
cd /path/to/SLAI-AscendBridge2
```

如果你还需要公开 adaptation 仓中的单模型产物，可按需把某个目录同步到本仓库：

```bash
mkdir -p adaptations
rsync -av /path/to/SLAI-AscendBridge2-Adaptations/adaptations/<name>/ ./adaptations/<name>/
```

### 3. 初始化框架环境

```bash
uv sync
uv run python scripts/board_ops.py init
```

### 4. 推荐环境变量

```bash
export HF_ENDPOINT=https://hf-mirror.com
export TASK_QUEUE_ENABLE=1
```

### 5. 权重准备建议

建议提前把模型权重准备到本地路径，避免 agent 在适配过程中临时下载导致等待时间过长或受网络影响。  
如果不提前准备，agent 也可以自行下载，但整体链路通常更慢。

---

## 使用指南

下面分三种典型使用方式：

1. `Claude` 自动批量编排
2. `Claude` 单模型连贯适配
3. `Codex / Cursor` 单模型连贯适配

### 1. Claude 自动批量编排

适用场景：

- 一次要处理多个模型
- 希望通过 `.claude/agents`、`.claude/skills`、`.claude/agent-memory` 配合 `team-lead` 自动推进多个阶段
- 需要 `board.db` + dashboard 的统一状态管理

#### 步骤 1：初始化看板并注册模型

先初始化空数据库：

```bash
uv run python scripts/board_ops.py init
```

可选：先查看模型元信息。

```bash
uv run python scripts/get_model_info.py "Qwen/Qwen2.5-0.5B-Instruct"
```

把模型注册进 `board.db`：

```bash
uv run python scripts/board_ops.py register_model \
  --model_id "Qwen/Qwen2.5-0.5B-Instruct" \
  --source huggingface \
  --url "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct" \
  --description "batch adaptation example"
```

如果是多个模型，可以循环执行 `register_model`。

#### 步骤 2：启动自动批量调度

推荐方式是直接使用仓库内封装好的脚本：

```bash
./run_auto_team_lead.sh
```

只跑某个阶段时，可切换 `PROMPT_MODE`：

```bash
PROMPT_MODE=benchmark ./run_auto_team_lead.sh
PROMPT_MODE=optimization ./run_auto_team_lead.sh
PROMPT_MODE=business ./run_auto_team_lead.sh
```

长期运行时，可交给看门狗脚本：

```bash
nohup ./team_lead_watchdog.sh > /tmp/slai_team_lead.log 2>&1 &
```

#### 步骤 3：如需手动进入 Claude team-lead

如果你更希望自己在交互式终端里驱动 `team-lead`，可以在仓库根目录启动：

```bash
IS_SANDBOX=1 claude --dangerously-skip-permissions --agent team-lead
```

示例提示词：

```text
请检查 board.db 中的 pending 模型，优先推进 adaptation -> benchmark -> optimization。
只处理当前看板中的任务，不要新建无关目录。
每完成一个模型后给出阶段结论与风险摘要。
```

#### 步骤 4：查看执行状态

```bash
uv run python scripts/board_ops.py list_adaptation_tasks --status pending
uv run python scripts/board_ops.py list_benchmark_tasks --status pending
uv run python scripts/board_ops.py list_optimization_tasks --status pending
```

如果本地有 `board.db`，也可以直接打开 `dashboard/index.html` 查看看板。

### 2. Claude 单模型连贯适配

适用场景：

- 只想处理一个模型
- 希望一个对话连续完成 `Adaptation -> Benchmark -> Optimization`
- 需要 agent 在同一上下文里连续修复问题，而不是切到批量队列

#### 推荐做法

1. 进入仓库根目录  
2. 启动一个单独的 Claude 会话  
3. 在同一个会话中要求它只处理一个模型、只改一个 adaptation 目录  
4. 在该会话中依次完成三个阶段，不要中途切换到批量调度

启动示例：

```bash
cd /path/to/SLAI-AscendBridge2
uv sync
uv run python scripts/board_ops.py init
IS_SANDBOX=1 claude --dangerously-skip-permissions
```

如果你希望 Claude 仍然带 `team-lead` 规则，但只处理一个模型，也可以：

```bash
IS_SANDBOX=1 claude --dangerously-skip-permissions --agent team-lead
```

#### 推荐提示词

```text
你当前在 SLAI-AscendBridge2 仓库根目录。

只处理单个模型：Qwen/Qwen2.5-0.5B-Instruct
模型权重路径：/mnt/model/qwen/Qwen2.5-0.5B-Instruct

请按以下顺序连续完成任务：
1. Adaptation：生成并修复 demo.py、pyproject.toml、README 等基础产物
2. Benchmark：生成 accuracy_run.py，并通过 benchmark/scripts/check_accuracy_run.py
3. Optimization：生成 accuracy_run_perf.py、optimization_notes.json，并通过 optimization/scripts/check_accuracy_run_perf.py

约束：
- 只允许修改与该模型相关的 adaptation 目录，以及确有必要的共享脚本
- 不要处理其他模型
- 保持 CUDA / Ascend 双栈兼容
- 每完成一个阶段就先自检，再继续下一阶段

完成标准：
- uv run python demo.py 成功
- adaptation/scripts/check_adaptation.py --adapt <name> 通过
- benchmark/scripts/check_accuracy_run.py --adapt <name> 通过
- optimization/scripts/check_accuracy_run_perf.py --adapt <name> 通过
- 输出最终报告，必须包含 benchmark 与 optimization 前后对比
```

#### 使用建议

- 单模型连贯适配时，尽量保持**一个模型一个会话**
- 让 Claude 在每个阶段结束后先跑 checker，再进入下一阶段
- 如果你已经准备好了权重路径和数据路径，直接写进 prompt，效果会更稳定

### 3. Codex / Cursor 单模型连贯适配

适用场景：

- 你不想走 `Claude team-lead` 的批量队列
- 你更习惯在终端或 IDE 内用一个 agent 连续完成单模型适配
- 你希望 agent 在本地仓库里直接改代码、跑命令、迭代修复

#### 3.1 Codex CLI

在仓库根目录启动 Codex CLI：

```bash
npm i -g @openai/codex
cd /path/to/SLAI-AscendBridge2
codex
```

推荐做法：

- 第一次运行先完成登录
- 单模型任务尽量保持在一个 Codex 会话里完成
- 在 prompt 中显式点名关键目录，例如 `scripts/`、`adaptation/`、`benchmark/`、`optimization/`、`adaptations/<name>/`
- 如果要切换模型或推理强度，可在会话中使用 Codex CLI 的 `/model`

Codex 里建议使用“目标 / 上下文 / 约束 / 完成标准”结构来下达任务：

```text
Goal:
完成 Qwen/Qwen2.5-0.5B-Instruct 的单模型连贯适配，覆盖 Adaptation -> Benchmark -> Optimization。

Context:
- 仓库根目录是当前工作目录
- 权重路径：/mnt/model/qwen/Qwen2.5-0.5B-Instruct
- 重点目录：scripts/, adaptation/, benchmark/, optimization/

Constraints:
- 只处理一个模型
- 不修改其他 adaptation
- benchmark / optimization 产物生成后必须跑对应 checker

Done when:
- demo.py 可运行
- accuracy_run.py / accuracy_run_perf.py 均通过 checker
- 给出最终性能对比摘要
```

#### 3.2 Cursor

在 Cursor 中，推荐使用**单个 agent / 单个 chat**来处理单模型任务，而不是一开始就把多个模型混在一个工作流里。

推荐流程：

1. 用 Cursor 打开 `SLAI-AscendBridge2` 仓库根目录
2. 进入 `Agent mode` 或 `Agents Window`
3. 明确告诉 agent：只处理一个模型、只改一个 adaptation 目录
4. 先让 agent 制定计划，再让它落地修改和运行命令
5. 每个阶段完成后先 review diff，再继续下一阶段

推荐提示词可直接复用上面的 Codex / Claude 单模型模板。  
如果你在 Cursor 中使用并行 agent，也建议把并行任务限制为：

- 一个主 agent 负责主链路
- 一个侧边 agent 负责只读排查或 checker 结果分析

不要让多个 agent 同时改同一个 adaptation 目录。

---

## 单模型连贯适配的最小闭环

不管你用 `Claude`、`Codex` 还是 `Cursor`，建议都按这个最小闭环执行：

```text
1. 明确模型与权重路径
2. 生成 adaptations/<name>/demo.py
3. 跑 demo.py 并修复
4. 生成 accuracy_run.py
5. 跑 benchmark checker
6. 生成 accuracy_run_perf.py + optimization_notes.json
7. 跑 optimization checker
8. 输出最终阶段报告
```

## 常用检查命令

### 根目录检查

```bash
uv run python adaptation/scripts/check_adaptation.py --adapt <name>
uv run python benchmark/scripts/check_accuracy_run.py --adapt <name>
uv run python optimization/scripts/check_accuracy_run_perf.py --adapt <name>
uv run python optimization/scripts/check_optimization_notes.py --adapt adaptations/<name>
uv run python business_benchmark/scripts/check_business_benchmark_run.py --adapt <name>
```

### adaptation 目录内运行

```bash
cd adaptations/<name>
uv run python demo.py
uv run python accuracy_run.py --use-pretrained
uv run python accuracy_run_perf.py --use-pretrained
```

如果当前模型只做到前三个阶段，可以不启用 `business_benchmark`。

## 与 adaptation 仓配合

如果你已经在 `SLAI-AscendBridge2-Adaptations` 中维护了单模型目录，可按需同步到本框架仓：

```bash
rsync -av /path/to/SLAI-AscendBridge2-Adaptations/adaptations/<name>/ ./adaptations/<name>/
```

然后继续在本框架仓中跑 checker、benchmark、optimization 和 dashboard。

## 开源说明

- 本仓库已移除内部运行数据、缓存、历史产物和私有环境文件
- `board.db` 默认不随仓库分发，仅在本地初始化为空 schema
- `adaptations/` 目录默认保持为空，供你按需拷贝单模型目录

## License

Apache-2.0
