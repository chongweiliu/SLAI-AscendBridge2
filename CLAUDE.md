# SLAI-AscendBridge2 - Project Context

## 项目概述

自动化智能体编排系统，将 PyTorch 模型适配到华为昇腾 (Ascend) NPU。

这是拆分后的**框架仓**。它负责脚本、检查器、调度骨架、dashboard、`.claude` 下的 agents / skills / agent-memory，以及 prompt 模板；模型级 adaptation 建议放在独立仓库 `SLAI-AscendBridge2-Adaptations` 中按需维护和公开。

为兼容不同 agent / IDE 的入口约定，仓库根目录额外保留：

- `.agents -> .claude`
- `AGENTS.md -> CLAUDE.md`

## 快速开始

```bash
# 同步依赖
uv sync

# (推荐) 配置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
```

## 仓库结构

```
SLAI-AscendBridge2/
.claude/
├── agent-memory/     # 智能体阶段记忆与规则沉淀
├── agents/           # 智能体定义
│   ├── team-lead.md        # PM 角色：任务排期、进度监控、Git 操作
│   ├── adapter.md          # 开发角色：NPU 算子适配、代码生成
│   ├── model-crawler.md    # 搜索角色：模型发现、元数据分析
│   ├── benchmark-runner.md # 评测角色：按适配生成 accuracy_run.py，精度/性能/trace
│   ├── npu-optimizer.md    # NPU 性能优化：torch_npu 融合算子替换
│   └── business-benchmark.md # 第四阶段业务测评：真实权重/数据集与 NPU/CUDA 对比
└── skills/           # 知识库
    ├── ascend-adaptation/  # NPU 适配经验
    ├── ascend-diffusers-adaptation/  # diffusers 图像/视频 pipeline 适配
    ├── ascend-diffusers-benchmark/  # diffusers 图像/视频 pipeline benchmark
    ├── uv-env-setup/       # 环境配置规则
    ├── model-discovery/    # 模型发现流程
    ├── database-ops/       # board.db 操作
    ├── dataset-mapping/    # 模型类型到评测数据集的映射
    ├── benchmark-script/   # 模型评测脚本生成与执行
    ├── benchmark-manager/  # benchmark 运行与产出管理 (list/run/artifacts/pack/unpack/clean)
    ├── benchmark-analysis/ # benchmark 数据聚合、对比、trace 分析
    ├── model-files-override/  # model_files 本地覆盖，性能优化测试（不改 models 缓存）
    ├── torch-npu-optimization/  # torch_npu 推理优化：融合算子替换
    ├── ascend-diffusers-optimization/  # diffusers 图像/视频 pipeline 推理优化
    └── ascend-profiling/   # NPU 性能分析：torch_npu.profiler L0/L1/L2 + MindStudio Insight
.agents -> .claude
AGENTS.md -> CLAUDE.md
adaptation/
benchmark/
optimization/
business_benchmark/
dashboard/
prompts/
scripts/
tests/
adaptations/          # 默认空目录，按需挂载或拷贝单模型 adaptation
CLAUDE.md
README.md
```

## 关键文件

| 文件 | 用途 |
|------|------|
| `board.db` | 任务看板数据库 (唯一事实来源) |
| `scripts/board_ops.py` | 看板 CRUD 操作、心跳更新 |
| `scripts/get_model_info.py` | 模型元数据提取 |
| `adaptation/scripts/check_adaptation.py` | 适配验证脚本（demo.py 等） |
| `benchmark/scripts/check_accuracy_run.py` | accuracy_run.py 规范强制检查 |
| `optimization/scripts/check_accuracy_run_perf.py` | accuracy_run_perf.py 核心结构检查（强制，board_ops 等会校验） |
| `optimization/scripts/check_optimization_notes.py` | optimization_notes JSON 格式校验（仅校验，不做自动修复；支持 `--adapt adaptations/{name}` 校验本地文件） |
| `scripts/dataset_mapping.py` | 模型类型到评测数据集的映射 |
| `benchmark/scripts/benchmark_tool.py` | 聚合/对比/trace 分析（含 NPU Fallback） |
| `benchmark/scripts/benchmark_manager.py` | benchmark 运行与产出管理 (list/run/artifacts/pack/unpack/clean) |
| `optimization/scripts/optimization_tool.py` | NPU 优化数据聚合与对比 |
| `optimization/scripts/optimization_manager.py` | optimization 运行管理 |
| `business_benchmark/scripts/check_business_benchmark_run.py` | business_summary.json 与第四阶段工件 completed gate 校验 |
| `business_benchmark/scripts/business_benchmark_tool.py` | 第四阶段业务测评结果汇总与对比 |
| `business_benchmark/scripts/business_benchmark_manager.py` | 第四阶段运行管理 (list/run-npu/print-remote-command/artifacts/pack/unpack/summarize/clean) |
| `scripts/download_datasets.py` | 评测数据集下载工具 |
| `prompts/` | 智能体 prompt 模板 (task_adaptation/task_benchmark/task_optimization) |
| `dashboard/` | Web 看板 (index.html，浏览器内查询 board.db) |
| `adaptations/` | 适配产出目录 (每个模型一个子目录) |
| `benchmark/` | 评测脚本与聚合 (compare_logits、aggregate_reports 等) |
| `optimization/` | NPU 优化测评脚本 (optimization_tool、optimization_manager) |
| `.claude/agent-memory/` | 多智能体阶段记忆与规则沉淀 |
| `CLAUDE.md` / `AGENTS.md` | 项目上下文入口（`AGENTS.md` 为软链接） |

`board.db` 由本地初始化生成，默认不纳入版本控制。

如果你是通过 `git clone` 获取仓库，`.agents` 和 `AGENTS.md` 软链接会自动保留。  
如果你是通过压缩包、拖拽、或不保留符号链接的同步方式获取目录，请执行：

```bash
bash scripts/ensure_agent_symlinks.sh
```

## Adaptations 目录结构

```
adaptations/{sanitized_model_id}/
├── demo.py           # 主推理脚本 (支持 --dry-run)
├── accuracy_run.py   # [可选] 精度评测脚本
├── accuracy_run_perf.py  # [可选] NPU 优化版评测脚本
├── business_benchmark_config.json # [可选] 第四阶段业务测评命令配置
├── business_run.py # [可选] 第四阶段统一入口脚本（npu_baseline / npu_perf / cuda_baseline）
├── business_summary.json # [可选] 第四阶段业务测评汇总
├── pyproject.toml    # 依赖配置 (含 cuda/ascend extra)
├── README.md         # 适配说明
├── models/           # 模型缓存目录
├── model_files/      # [可选] NPU 优化版模型覆盖文件
├── outputs_*.pt      # [可选] 评测产出 (精度对比用)
├── benchmark_metrics_*.json  # [可选] benchmark 指标
├── trace_*.json      # [可选] trace 分析产出
├── profiling/        # [可选] NPU profiling 数据 (--profile-level L1 产出)
├── {custom_repo}/    # [可选] 克隆的自定义模型库源码 (如 LTX-2/, mivolo_src/)
├── optimization_notes.json   # [可选] 优化记录
└── .status.json      # 适配状态记录
```

## 常用命令

```bash
# 查看待处理任务
sqlite3 board.db "SELECT model_id, adaptation_status FROM models WHERE adaptation_status='pending' LIMIT 10"

# 运行适配验证 (在 adaptations/{id} 目录下)
uv run python demo.py

# 运行精度评测 (可选，在 adaptations/{id} 目录下)
uv run python accuracy_run.py --use-pretrained  # 加载预训练权重
uv run python accuracy_run.py --cpu              # 强制 CPU 推理

# 强制检查 accuracy_run.py 规范 (benchmark 规则)
uv run python benchmark/scripts/check_accuracy_run.py

# 强制检查 accuracy_run_perf.py 规范 (optimization 规则)
uv run python optimization/scripts/check_accuracy_run_perf.py

# 在单个 adaptation 上预检 completed gate（推荐在 adaptations/{id} 目录外执行）
uv run python benchmark/scripts/check_accuracy_run.py --adapt adaptations/{name}
uv run python optimization/scripts/check_accuracy_run_perf.py --adapt adaptations/{name}
uv run python optimization/scripts/check_optimization_notes.py --adapt adaptations/{name}
uv run python business_benchmark/scripts/check_business_benchmark_run.py --adapt {name}

# 模型信息提取
uv run python scripts/get_model_info.py --model_id <model_id>

# 下载评测数据集
uv run python scripts/download_datasets.py <dataset_name>
# 为样本不足模型下载候选数据集
uv run python scripts/dataset_mapping.py --model_id <model_id> --candidates --json
uv run python scripts/download_datasets.py --model-id <model_id> --candidate-datasets

# 回退模型状态（模型列表 + 目标状态）
python scripts/board_ops.py rollback_models --models "org/a org/b" --to optimization:pending
# 回退上游阶段时，只级联重置 downstream 中 completed/in_progress 的阶段；skipped/not_applicable/needs_authorization 保留

# 第五阶段人工核验（仅人工更新，不分配 agent）
python scripts/board_ops.py update_human_review_status --model_id "org/name" --human_review_status pending
python scripts/board_ops.py update_human_review_status --model_id "org/name" --human_review_status completed

# 校验 optimization_notes.json 格式（校验全部 completed 记录或指定 adaptation 本地文件）
uv run python optimization/scripts/check_optimization_notes.py  # 报错模式

# Profiling 分析 (解析 torch_npu.profiler 产出)
uv run python benchmark/scripts/benchmark_tool.py profiling --adaptation {name} -j
uv run python benchmark/scripts/benchmark_tool.py profiling --adaptation {name} --deep -j  # SQLite 深度分析

# Trace 分析 (NPU fallback 检测)
uv run python benchmark/scripts/benchmark_tool.py trace --adaptation {name} -v

# 优化数据聚合与对比
uv run python optimization/scripts/optimization_tool.py compare --adaptation {name}
```

## Code Style

```bash
# 格式化代码
ruff format .

# 整理 import
ruff check --fix --select I .
```

配置见 `pyproject.toml`：`line-length = 250`，lint 启用 `I` (isort)

## 工作流程

1. **Discovery**: Model Crawler 爬取模型元数据，过滤 NSFW，入库 board.db
2. **Planning**: team-lead 分发任务，设定优先级
3. **Adaptation**: Adapter 创建 `adaptations/{id}/`，编写 demo.py (DRY RUN 模式)
4. **Healing**: 循环 `uv run` → 捕捉 Traceback → 自主修复
5. **Benchmark**（可选）: Benchmark-Runner 对已完成适配的模型生成并运行 `accuracy_run.py`，产出 `outputs_*.pt`、`benchmark_metrics_*.json`、`trace_*.json`，并更新看板 `benchmark_status`；**规则要求**：生成或修改 `accuracy_run.py` 后必须执行 `check_accuracy_run.py` 并确保通过；`benchmark_status=completed` 对应的 baseline `benchmark_metrics` 工件必须满足 `num_samples >= 50`
6. **Optimization**（可选）: NPU Optimizer 对已完成 benchmark 的模型做 torch 级别 API 替换（torch_npu 亲和算子）或纯运行时优化（`warmup + TASK_QUEUE_ENABLE`），产出 `accuracy_run_perf.py`、`benchmark_metrics_*_perf.json`、`optimization_notes.json`；若需要 patch 模型实现，额外产出 `model_files/`，且 **`model_files/` 必须且仅能由 npu-optimizer 创建**。**规则要求**：完成前必须通过 `check_accuracy_run_perf.py`，并用 `check_optimization_notes.py --adapt adaptations/{name}` 校验本地 `optimization_notes.json`；`optimization_status=completed` 对应的 baseline/perf `benchmark_metrics` 工件必须满足 `num_samples >= 50`，且 `optimization_notes.results[]` / `best_result` 也必须包含 `num_samples >= 50`，同时 `optimization_notes.best_result` 必须包含 `output_type`、`baseline_artifact`、`perf_artifact`、`baseline_wall_clock_s`、`perf_wall_clock_s` 等可追溯字段；**精度对比**必须使用 pretrained 权重。`best_result.speedup_ratio` 默认必须 `> 1.0`；仅当 `runtime_only` 路径在“模型代码多次改动尝试无果”后退回纯运行时优化，且在 notes 中同时满足 `code_modified=false`、`code_change_attempts>=2`、并明确注明“模型代码无更改”时，允许 `best_result.speedup_ratio = 1.0`（`<1.0` 仍不允许 completed）。`runtime_only` 路径允许作为正式 completed，但必须显式记录 `optimization_kind=runtime_only`、`selected_npu(s)`、`device_topology`、`parallel_mode`；`speedup_ratio` **统一按整轮 wall-clock 计算**，即 `baseline_wall_clock_s / perf_wall_clock_s`，其中 wall-clock 优先取工件显式字段，否则由 `start_time/end_time` 推导；`baseline_latency_s / perf_latency_s` 继续保留为前向延迟证据，但不再作为 completed 判定的官方 speedup；若 `speedup_ratio >= 3x`，还必须提供 `comparison_method=independent_baseline_artifact`、有效 `comparison_scope`、非空 `validation_note`、以及正数 `steady_state_baseline_latency_s` / `steady_state_perf_latency_s`。**状态选择**：样本不足、版本兼容/依赖问题、OOM/多卡重试、工件缺失/口径不一致、重复分配但无新产物等一律回 `pending`；仅当完成 `runtime_only` 尝试后确认无真实提速时才允许 `skipped`，确认架构/格式确实不适用时才允许 `not_applicable`
7. **Business Benchmark**（可选）: 在 `optimization_status=completed` 后执行第四阶段业务测评，先本机产出真实权重、真实数据集下的 NPU baseline/perf，再优先通过 `business_benchmark_manager.py run-remote-cuda` 走 SSH 自动闭环补齐远端 CUDA baseline；只有 SSH 不通、远端环境异常或自动回收失败时才降级到 `print-remote-command` / `wait_cuda`，最终生成 `business_summary.json`；**规则要求**：进入远端 CUDA 前，以及任何模型准备写成 `wait_cuda` 前，必须先通过 `check_business_benchmark_run.py --adapt {name} --wait-cuda-npu-only`，确认本机 NPU baseline/perf 两路 `num_samples`、字段完整性、quality metric、输出质量、以及 `npu_speedup_ratio >= 0.9` 都正常；若本机 NPU 双路出现 `exact_match/accuracy/top1_accuracy/match_rate` 全 0、单路塌 0、双路漂移异常、或 `npu_perf` 相比 baseline 明显退化，必须直接回 `pending` 修 evaluator / 标签归一化 / 数据集画像 / 计时口径 / NPU perf 继承链，禁止继续烧 CUDA 或写成 `wait_cuda`。完成前必须通过 `check_business_benchmark_run.py --adapt {name}`；`business_benchmark_status=completed` 必须同时具备 `business_metrics_npu_*_baseline.json`、`business_metrics_npu_*_perf.json`、`business_metrics_cuda_*_baseline.json` 与 `business_summary.json`，且 `business_summary.best_result` 中的 `npu_speedup_ratio` / `vs_cuda_latency_ratio` 必须为正数；第四阶段必须显式记录测量契约：`measurement_contract_version`、`latency_measurement_scope`、每路工件的 `warmup_iterations` / `task_queue_enable` / `loaded_from_model_files`，并保留运行时证据：`python_executable`、`python_version`、`package_versions`、`scenario_command`；**第四阶段运行环境必须与前三阶段一致，始终使用 adaptation 目录自己的 uv 环境，并显式指定 extra：本机 NPU 只允许 `uv run --extra ascend ...`，远端 CUDA 只允许 `uv run --no-sync --extra cuda ...`；缺少 `--extra` 的 `uv run python ...`、缺少 `--no-sync` 的远端正式 CUDA 命令、与直接 `.venv/bin/python ...` 均视为违规口径；默认远端根目录优先读取执行机环境变量 `SLAI_REMOTE_PROJECT_ROOT`**；远端 CUDA 一律显式设置 adaptation 私有 `UV_CACHE_DIR`（推荐 `$PWD/.uv_cache_remote`），并显式设置国内镜像 `UV_DEFAULT_INDEX/PIP_INDEX_URL`；若远端 `uv sync --extra cuda --no-install-project --frozen` 长时间无新 stdout，禁止无限等待，必须立即检查是否卡在共享 cache 锁、PyPI 超时、或 Python 版本不符；一旦确认自动安装无实质推进，应切换为“手工预装 `.venv` + 正式运行 `uv run --no-sync --extra cuda ...`”路径，而不是继续盲等。`npu_perf` 默认应继承优化环境（至少 `TASK_QUEUE_ENABLE=1`），并优先自动发现并加载 `model_files/` 下的 patch hook 或自定义 modeling 文件；若 `model_files/` 只包含补丁模块而无 `config.json`，也不能静默退化为未打 patch 的 baseline 路径。若 `business_summary.best_result.vs_cuda_latency_ratio < 0.4`，必须额外检查第四阶段 NPU 是否仍在使用 `parallel_mode=auto|device_map_auto`、`device_topology=all_visible_devices` 等自动映射；若是，先改成固定 1-die 或 2-die 的 `ASCEND_RT_VISIBLE_DEVICES` 重跑 NPU，再决定是否接受该轮结果
8. **Human Review**（人工）: 在 `business_benchmark_status=completed` 后进入第五阶段人工核验；该阶段不分配 agent，仅由人工通过 `update_human_review_status` 在 `pending/completed` 之间更新。历史上已完成第四阶段但尚未人工核验的记录默认进入 `human_review_status=pending`
9. **Sync**: team-lead 验收，更新状态，git commit

## 交付标准 (DoD)

- [ ] 完整的 `pyproject.toml` + `uv.lock`
- [ ] 代码包含 `torch_npu` 逻辑
- [ ] 支持 DRY RUN 模式 (无真实推理)
- [ ] `uv run python demo.py` 成功运行
- [ ] 无 NSFW 内容

## 注意事项

- **双栈兼容**: 适配代码需同时支持 NVIDIA CUDA 和 Huawei Ascend
- **环境隔离**: 每个模型使用独立的 uv 虚拟环境
- **Python 版本基线**: 默认统一使用 Python `>=3.12`；只有在明确确认某些依赖/包与 3.12+ 不兼容时，才允许降到 `<3.12`，并必须在对应 adaptation 的 README/notes 中写清原因
- **第四阶段 extra 规则**: Business Benchmark 必须显式写 `uv run --extra ascend ...` / `uv run --extra cuda ...`；不能省略 extra，也不能直接写 `.venv/bin/python ...`
- **第四阶段远端 CUDA uv 规则**: 远端正式 CUDA 执行统一使用 `uv run --no-sync --extra cuda ...`；远端安装阶段必须显式设置 adaptation 私有 `UV_CACHE_DIR`（推荐 `$PWD/.uv_cache_remote`）与镜像 `UV_DEFAULT_INDEX/PIP_INDEX_URL`。若 `uv sync --extra cuda --no-install-project --frozen` 长时间无新 stdout，必须立刻排查共享 cache 锁 / PyPI 超时 / Python 版本，不得无限等待；确认自动安装无推进后应切换到“手工预装 `.venv` + `uv run --no-sync --extra cuda ...`”路径
- **第四阶段运行时证据**: Business Benchmark 工件必须保留 `python_executable`、`python_version`、`package_versions`、`scenario_command`
- **第四阶段历史重跑**: 对已经存在正式 `business_metrics_*` / `business_summary.json` 的模型按新规则补跑时，先把旧正式工件改名备份（推荐后缀 `__prev_rule_refresh_<timestamp>`）；否则 `run-remote-cuda` 回收 CUDA 工件时遇到本地同名但内容不同的文件会拒绝覆盖，只会隔离远端副本，导致本轮结果无法直接成为 canonical 工件
- **第四阶段批量刷新**: 历史 `completed` 记录补跑时按模型逐个闭环执行：`run-npu -> run-remote-cuda -> summarize/check -> scripts/board_ops.py update_business_benchmark_status`；写库只能通过 `board_ops.py`，不要直接写 SQL，也不要等整批结束后再统一回填
- **第四阶段 wait_cuda 前置 gate**: 任何模型写 `wait_cuda` 前，必须先运行 `uv run python business_benchmark/scripts/check_business_benchmark_run.py --adapt {name} --wait-cuda-npu-only`；若本机 NPU baseline/perf 双路存在全 0、单路塌 0、双路质量漂移异常、`npu_speedup_ratio < 0.9`、字段缺失、或 `num_samples <= 50`，一律回 `pending` 修本机链路，禁止进入 `wait_cuda`
- **第四阶段长跑判活**: 远端 CUDA 若长时间无新 stdout，先看远端 `ps` 是否仍有 `business_eval.py`、`nvidia-smi` 是否仍持有进程/显存，以及本地 `business_metrics_cuda_*` / `business_summary.json` 时间戳是否前进，再决定是继续等待还是转 `wait_cuda`
- **Team-Lead 写库标识**: `team-lead` 每次通过 `scripts/board_ops.py` 写 `agents.current_task`（典型是 `heartbeat --id "team-lead"`）时，必须带本机 IP 和当前进程号，用于区分共享存储上来自不同机器、以及同机不同进程的同名 team-lead；`board_ops.py` 会自动追加 `| host_ip=<local-ip> | pid=<process-id>`，调用方只需传业务描述
- **DRY RUN**: 默认使用 `DRY_RUN=1` 避免实际模型加载
- **HuggingFace 镜像**: 国内环境建议设置 `HF_ENDPOINT=https://hf-mirror.com`
- **NPU 优化环境变量**: `TASK_QUEUE_ENABLE=1`（异步算子下发，+5~15%推理加速）；多卡环境必须限制单卡，但**不要默认写死 0 号卡**，应先用 `npu-smi info` 查看各卡占用，再将 `ASCEND_RT_VISIBLE_DEVICES` 设置为当前空闲或低占用的单卡
- **样本不足补救**: 若 optimization/benchmark 产物 `num_samples < 50`，不得直接记 `skipped`；必须先用 `scripts/dataset_mapping.py --model_id <model_id> --candidates` 查候选数据集，再用 `scripts/download_datasets.py --model-id <model_id> --candidate-datasets` 下载并重测
- **结构化状态原因**: `update_optimization_status` 写入 `pending` / `skipped` / `not_applicable` 时，`notes` 必须是 JSON object，至少含 `reason_code`、`retryable`、`recommended_action`、`evidence`、`next_step`
- **Completed Gate**: 新生成或修改 benchmark / optimization / business benchmark 产物后，优先用 `check_accuracy_run.py --adapt adaptations/{name}`、`check_accuracy_run_perf.py --adapt adaptations/{name}`、`check_optimization_notes.py --adapt adaptations/{name}`、`check_business_benchmark_run.py --adapt {name}` 做本地预检；不要等到 team-lead 写库时才发现被 gate 拦截
- **人工核验口径**: `human_review_status` 仅允许 `pending/completed`；只有 `business_benchmark_status=completed` 的模型才允许进入第五阶段。前端星标与“全链路完成”统计均以 `human_review_status=completed` 为准
- **Profiling 采集**: `accuracy_run.py --profile-level L1` 产出 `profiling/` 目录，用 `benchmark_tool.py profiling --deep` 解析
- **自定义模型库源码**: 部分 adaptation 含克隆的第三方仓库源码（如 LTX-2/、mivolo_src/），这些源码需纳入 git 版本控制（含 NPU 修改）；克隆时必须删除嵌套 `.git/` 目录以避免 git submodule `modified content` 警告；`.gitignore` 用 `**/{dir_name}/.git` 仅排除嵌套 `.git`（详见 `.gitignore` 注释）
