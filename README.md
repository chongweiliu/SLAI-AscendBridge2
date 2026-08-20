# SLAI-AscendBridge2.2

`SLAI-AscendBridge2` 是一款面向华为昇腾 NPU 的自动化智能体编排、单模型适配与推理部署框架，用于将 PyTorch 模型迁移到 Ascend。它支持从模型发现、环境治理、代码适配、精度评测到 NPU 性能优化的闭环流程；如果需要，也可以继续扩展到第四阶段 `business_benchmark`。

当前版本为 **v2.2**。它延续 v2.1 的 **vLLM-Ascend 自动部署**能力，并新增 **CANNBot 按需协同适配**：只有在标准 PyTorch 与 `torch_npu` 专用接口均无法解决算子缺口，或性能分析确认需要自定义算子时，才会把最新版 CANNBot 下载到项目内缓存并调用其 Ascend C 专家流程。

这个仓库是**框架仓**，负责脚本、检查器、调度骨架、dashboard、`.claude` 下的 agents / skills / agent-memory，以及 prompt 模板，不默认携带公开 adaptation 集合。模型级 adaptation 建议放在独立仓库 `SLAI-AscendBridge2-Adaptations`，或按你的内部目录结构单独维护。

为兼容不同 agent / IDE 的入口约定，仓库根目录额外保留了两个软链接：

- `.agents -> .claude`
- `AGENTS.md -> CLAUDE.md`

## v2.2 新增功能

### CANNBot 按需协同适配

v2.2 新增 `cannbot-adapter` Agent、CANNBot 协同工作流和项目内同步脚本，用于处理普通框架适配无法覆盖的算子兼容与性能问题。

- 采用四级决策顺序：优先使用可在 Ascend 上直接执行的标准 PyTorch 算子；标准实现存在功能缺口或经 profiling 确认性能不达标时，再尝试 `torch_npu` 提供的昇腾专用接口或优化实现；随后复用 GitCode 上 CANN 与昇腾社区的已有方案；只有这些路径均不可行时，才调用 CANNBot 生成 Ascend C 自定义算子
- 仅在真正进入第四级方案时执行 `scripts/sync_cannbot.sh --print-path`；脚本每次都会检查 `https://gitcode.com/cann/cannbot-skills.git` 的 `master` 最新版本
- CANNBot 只下载到当前项目的 `.cache/cannbot/cannbot-skills/`，该目录已被 Git 忽略；仓库不内置其源码，也不会在项目外或 Claude 全局目录安装插件
- 同步过程不会修改 `~/.claude/settings.json`，运行时也不会加载 CANNBot 根目录的 `AGENTS.md`、`CLAUDE.md` 或 `SessionStart` Hook，因此普通 Claude 会话仍将自己识别为 **SLAI-AscendBridge2**，不会自称 CANNBot
- 需要生成自定义算子时，按 Architect → Design Reviewer → Developer → Reviewer 四个角色推进，再通过 `cannbot_ops.py` 接回模型适配主链路
- 验收同时覆盖真实权重、至少 50 组输入的 fuzz 对比、精度与性能回归；必要时对非连续输入显式调用 `.contiguous()`，避免 CANN 混合调用造成结果异常

详细流程见 [`docs/cannbot-collaborative-adaptation-guide.md`](docs/cannbot-collaborative-adaptation-guide.md)。

### vLLM-Ascend 自动部署（v2.1 延续）

`vllm-ascend-deployer` Agent 和 `vllm-ascend-auto-deploy` Skill 可把模型与部署需求转换为经过预检和真实推理验收的 vLLM-Ascend 服务。

- 支持本机和 SSH 单机部署，以及 SSH 多机部署
- 支持 Kubernetes、CCE、ACK 等调度平台
- 支持普通多机推理和 PD（Prefill/Decode）分离部署
- 优先复用 `adaptations/` 中已有模型制品；未找到时可选择外部权重路径或自动下载准备
- 根据模型 `config.json`、可用 NPU 和官方模型配置自动规划 TP、DP、EP 等并行参数
- 自动检查 Ascend 硬件、模型权重、镜像/运行环境、端口、网络和资源约束
- 自动生成本机、SSH 或 Kubernetes 部署脚本与 YAML，并执行语法检查和 dry-run
- 服务启动后通过 OpenAI 兼容 API 发起真实推理请求；只有健康检查和语义验收均通过才报告部署成功
- 实际启动或提交前先展示完整配置摘要，并等待用户明确确认

示例见下方“vLLM-Ascend 自动部署”章节。

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
│   ├── agents/                        # 智能体定义，包含 cannbot-adapter、vllm-ascend-deployer
│   ├── workflows/
│   │   └── ascend-cannbot-pipeline.js # CANNBot 四角色协同编排
│   └── skills/
│       └── vllm-ascend-auto-deploy/   # vLLM-Ascend 自动部署技能、知识库、脚本和模板
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
│   ├── sync_cannbot.sh                # 按需同步最新版 CANNBot 到项目内缓存
│   └── sanitize_repo_paths.py         # 路径清洗工具
├── tests/
├── adaptations/                       # 默认空目录，可挂载或拷贝单模型 adaptation
├── .cache/cannbot/                     # 按需生成的 CANNBot 项目缓存，不纳入 Git
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
                                  \-> [OperatorGap] CANNBot -> Validation -/
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
- `[OperatorGap] CANNBot`
  - 当适配阶段遇到功能缺口，或优化阶段确认 CPU 回退、同步开销等算子瓶颈时按需触发；完成四角色设计、开发和复核后，再回到项目主流程做真实权重验证
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

下面分五种典型使用方式：

1. `Claude` 自动批量编排
2. `Claude` 单模型连贯适配
3. `Codex / Cursor` 单模型连贯适配
4. `CANNBot` 按需协同适配
5. `vLLM-Ascend` 自动部署

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

### 4. CANNBot 按需协同适配

适用场景：

- 模型在 Ascend 上存在标准 PyTorch 算子和 `torch_npu` 专用接口都无法补齐的算子缺口
- 已经通过 profiling 确认 CPU 回退、频繁同步或低效算子是主要性能瓶颈
- 希望在保留 SLAI-AscendBridge2 主体身份和工作流的前提下，临时借用 CANNBot 的 Ascend C 专家能力

推荐通过 `team-lead` 触发，不需要提前安装 CANNBot：

```bash
IS_SANDBOX=1 claude --dangerously-skip-permissions --agent team-lead
```

示例提示词：

```text
请继续完成当前模型的 Ascend 适配。如果发现算子缺口，先尝试可在 Ascend 上直接执行的标准
PyTorch 算子；标准实现存在功能缺口或经 profiling 确认性能不达标时，再尝试 torch_npu
提供的昇腾专用接口或优化实现。随后搜索 GitCode 上已有的 CANN/昇腾社区方案；只有这些
路径均不可行时，才调用 cannbot-adapter。
请用真实权重完成精度和性能验收，并在最终报告中说明触发原因、采用方案和 CANNBot 版本。
```

进入 CANNBot 阶段时，项目会自动执行：

```bash
scripts/sync_cannbot.sh --print-path
```

该命令会检查并同步上游 `master` 最新提交，只在标准输出中返回项目缓存路径，供协同工作流读取。可用以下命令查看当前已缓存版本；尚未下载时会显示 `not installed`：

```bash
scripts/sync_cannbot.sh --status
```

注意：不要直接在 `.cache/cannbot/cannbot-skills/` 中启动 Claude。CANNBot 是按需读取的外部能力来源，项目主体、任务编排和最终交付仍由 SLAI-AscendBridge2 负责。

---

### 5. vLLM-Ascend 自动部署

适用场景：

- 已有模型，希望快速启动 vLLM-Ascend 推理服务
- 需要部署到本机、SSH 服务器或 Kubernetes/CCE/ACK 调度平台
- 需要自动规划单机、多机或 PD（Prefill/Decode）分离拓扑
- 希望自动完成部署前检查、部署文件生成和真实推理验收

#### 部署前准备镜像

请根据目标机器的 Ascend 代际选择对应的 vLLM-Ascend 镜像变体，并确保镜像版本支持待部署模型。项目提供 `images/fetch-image.sh`，用于把远端镜像保存成可复用的离线 tar：

```bash
# 用法：bash images/fetch-image.sh <version> <variant>
bash images/fetch-image.sh v0.23.0rc1 a3
```

脚本优先使用 `crane` 下载，失败时回退到 `docker pull` + `docker save`，生成以下文件：

```text
images/vllm-ascend-v0.23.0rc1-a3.tar
images/vllm-ascend-v0.23.0rc1-a3.tar.sha256
```

本机或 SSH 部署时，Agent 会优先使用并加载匹配的本地 tar；如果 `images/` 中没有匹配文件，则使用预检确认的远端 registry 镜像。对于 Kubernetes/CCE/ACK，需要提前保证平台节点能够拉取该镜像；如果平台无法访问公共 registry，请先把镜像推送到平台私有镜像仓库。

更完整的镜像命名、下载、校验和手动加载方式见 [`images/README.md`](images/README.md)。

#### 通过 team-lead 启动部署

在仓库根目录沿用项目已有的 `team-lead` 启动命令：

```bash
cd /path/to/SLAI-AscendBridge2
IS_SANDBOX=1 claude --dangerously-skip-permissions --agent team-lead
```

然后明确要求 `team-lead` 调用 `vllm-ascend-deployer`，并用自然语言描述模型和部署目标。部署 Agent 会按需询问无法安全推断的信息，自动完成权重解析、环境预检和并行拓扑规划，并在实际部署前展示完整配置供确认。

#### 本机单机部署示例

```text
请调用 vllm-ascend-deployer，使用 vLLM-Ascend 在本机部署 Qwen/Qwen3-30B-A3B，权重位于adaptations。采用单机部署。
优先使用 adaptations/ 中已有的模型权重。
```

#### SSH 多机部署示例

```text
请调用 vllm-ascend-deployer，使用 vLLM-Ascend 部署 Qwen/Qwen3-235B-A22B。
采用多机 SSH 部署，不启用 PD 分离。
```

Agent 会继续收集目标主机、用户名和认证方式。密码、Token 和私钥不会写入部署 JSON、脚本、日志或 Git。

#### Kubernetes/CCE/ACK 部署示例

```text
请调用 vllm-ascend-deployer，在 Kubernetes 调度平台部署 DeepSeek-R1，采用多机部署并启用 PD 分离，配置为 2P2D。
模型权重已挂载到 /mnt/models/DeepSeek-R1。
```

对于调度平台，Agent 会按需确认 context、namespace、节点与 NPU 资源、镜像和模型挂载等必要信息，并生成标准 Kubernetes YAML 和一键部署脚本。

#### 自动部署流程

```text
描述模型和部署需求
-> 选择单机/多机及部署目标
-> 解析或准备模型权重
-> 检查 Ascend 环境与部署约束
-> 自动规划 TP/DP/EP 或 PD 拓扑
-> 展示完整配置摘要
-> 用户确认执行
-> 生成部署文件并启动服务
-> 通过 OpenAI 兼容 API 完成真实推理验收
```

部署成功后，Agent 会返回：

- 服务 endpoint
- 本地进程 PID、远程进程信息或调度任务 ID
- 日志位置
- 停止和清理命令
- 本次生成的一键部署脚本或 YAML 路径

部署失败时只清理本次创建的进程或调度资源，并保留必要的诊断信息。

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

## 论文

该项目的技术报告：

- Liu, Chognwei, et al. [SLAI-AscendBridge: Execution-Grounded Evaluation for Agent-Built PyTorch-to-Ascend Migration](https://zenodo.org/records/21585210). Zenodo, 2026. DOI: [10.5281/zenodo.21585210](https://doi.org/10.5281/zenodo.21585210)

## 开源说明

- 本仓库已移除内部运行数据、缓存、历史产物和私有环境文件
- `board.db` 默认不随仓库分发，仅在本地初始化为空 schema
- `adaptations/` 目录默认保持为空，供你按需拷贝单模型目录

## License

Apache-2.0
