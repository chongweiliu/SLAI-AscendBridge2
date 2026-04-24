# Business Benchmark

第四阶段业务测评模块，位于 `optimization` 之后。

目标：

- 本机 NPU 跑真实权重、真实业务数据集的未优化 baseline
- 本机 NPU 跑真实权重、真实业务数据集的优化后 perf
- 远端 CUDA 机器通过 SSH 直传同步第四阶段工作目录后跑真实权重、真实业务数据集的未优化 baseline
- 汇总三路结果为 `business_summary.json`

## 单模型目录约定

业务测评产物仍放在 `adaptations/{name}/` 下：

- `business_benchmark_config.json`
- `business_eval.py`
- `business_model_eval.py`
- `business_run.py`
- `business_metrics_npu_*_baseline.json`
- `business_metrics_npu_*_perf.json`
- `business_metrics_cuda_*_baseline.json`
- `business_outputs_*.pt`
- `business_summary.json`

## 配置文件

每个 adaptation 可放一个 `business_benchmark_config.json`；如果配置文件缺失，`generate-script` / `run-npu` / `print-remote-command` 会按模型画像自动生成一份默认配置：

```json
{
  "dataset": "real_business_dataset",
  "remote_ssh_host": "cuda-remote",
  "remote_project_root": "$SLAI_REMOTE_PROJECT_ROOT",
  "measurement_contract_version": 2,
  "latency_measurement_scope": "steady_state",
  "baseline_warmup_iterations": 1,
  "perf_warmup_iterations": 3,
  "cuda_baseline_warmup_iterations": 1,
  "npu_baseline_env": {},
  "npu_perf_env": {
    "TASK_QUEUE_ENABLE": "1"
  },
  "cuda_baseline_env": {},
  "local_npu_baseline_command": "uv run --extra ascend python \"business_eval.py\" --scenario npu_baseline",
  "local_npu_perf_command": "uv run --extra ascend python \"business_eval.py\" --scenario npu_perf",
  "remote_cuda_baseline_command": "uv run --extra cuda python \"business_eval.py\" --scenario cuda_baseline"
}
```

manager 会基于这个配置自动生成 `business_run.py`，统一支持三个场景：

- `--scenario npu_baseline`
- `--scenario npu_perf`
- `--scenario cuda_baseline`

同时也会生成 `business_eval.py` 作为第四阶段统一测评 harness：

- 负责加载业务数据集
- 根据 `evaluation_profile` 选择对应指标计算，优先使用标准评测库
- 优先调用 adaptation 内的 `custom_evaluator`（默认约定文件名 `business_model_eval.py`）
- 默认把 scenario 级 env 注入到子进程；其中 `npu_perf_env` 默认携带 `TASK_QUEUE_ENABLE=1`
- 默认把 warmup 策略写入工件元数据：`baseline_warmup_iterations=1`、`perf_warmup_iterations=3`、`cuda_baseline_warmup_iterations=1`

同时 manager 也会生成一个通用 `business_model_eval.py` 模板，在文件缺失时自动补齐；若 adaptation 内已经有定制实现，则保留原文件，不再默认覆盖：

- `causal_lm`
- `seq2seq`
- `classification`
- `question_answering`
- `token_classification`
- `reranker`
- `vision_classification`
- `embedding`
- `audio_embedding`
- `vision_detection`（常规 object detection 头 + zero-shot detection，前提是数据样本可提供类别 query）

对于更复杂的模型类型，agent 需要在 adaptation 目录继续定制自己的 `business_model_eval.py`。如果 zero-shot detection 模型依赖额外 prompt engineering、开放词表映射或非 COCO 类别体系，仍建议在 adaptation 目录定制实现。

第四阶段通用模板现在还会自动做两件之前经常漏掉的事情：

- `npu_perf` 场景会优先自动发现 `model_files/` 下的 patch hook；不再只依赖 `model_files.__init__` 是否手工 re-export
- 如果 `model_files/` 里有 `config.json` / tokenizer / processor 资产，会自动切到本地 `model_files/` 作为有效加载源；否则至少仍会尝试导入 `npu_patches.py`、`modeling_*.py` 并执行 hook

当前已优先接入的标准评测实现包括：

- `qa_exact_match` -> `exact_match + token-level F1`
- `summarization_rouge` -> `rouge-score`
- `token_classification_f1` -> `seqeval`
- `asr_wer` -> `jiwer`
- `vision_topk_accuracy` -> 基于 score matrix 的真实 top-k 计算
- `reranker_ndcg` -> `sklearn.metrics.ndcg_score` + `MRR`
- `detection_map` -> IoU 阈值 0.50:0.95 的 AP 聚合与 `map50`

同时 manager 会在生成脚本前自动补齐以下字段：

- `model_id`
- `model_type`
- `dataset`
- `dataset_required`
- `dataset_local_path`
- `benchmark_run_id`
- `evaluation_profile`
- `primary_metric`
- `secondary_metrics`
- `output_type_hint`
- `remote_ssh_host`（默认 `cuda-remote`）
- `remote_project_root`（默认优先读取执行机环境变量 `SLAI_REMOTE_PROJECT_ROOT`）

若 `dataset` 未显式配置，会按模型类型自动选择第四阶段业务数据集；若该数据集本地不存在，`business_run.py` 会在执行时自动调用改造后的下载工具补齐。当前默认推荐在 SSH 可用时直接使用 `run-remote-cuda` 做全自动远端闭环；`remote_cuda_baseline_command` 仍作为远端执行的底层命令来源，但必须显式使用 adaptation 自己的 uv 环境和 `uv run --extra cuda ...`。`generate-script` / `run-remote-cuda` / `print-remote-command` 会优先读取 `remote_ssh_host / remote_project_root`；若配置文件中缺失，则默认补成 `cuda-remote` 与执行机环境里的 `SLAI_REMOTE_PROJECT_ROOT`（未设置时再回退到 manager 内部默认值）。

此外，第四阶段现在会在配置和每个业务工件里写入同一个 `benchmark_run_id`，用于把一次完整的 `npu_baseline / npu_perf / cuda_baseline` 绑定成同一轮测评。`business_summary.json` 只允许汇总同一个 `benchmark_run_id` 下的三类工件，避免把不同时间、不同代码版本或不同实验条件下的产物拼在一起。

从当前测量契约版本开始，业务工件与汇总还会显式记录以下字段，用于保证后续比较可复现、可解释：

- `measurement_contract_version`
- `latency_measurement_scope`
- `warmup_iterations`
- `task_queue_enable`
- `loaded_from_model_files`
- `model_source_kind` / `tokenizer_source_kind`
- `patch_load_status` / `patch_hooks`
- `python_executable`
- `python_version`
- `package_versions`
- `scenario_command`

如果需要长期保留人工覆盖，而不是让 manager 按最新画像重算自动字段，请使用以下 override 字段：

- `model_type_override`
- `dataset_override`
- `evaluation_profile_override`
- `primary_metric_override`
- `secondary_metrics_override`
- `output_type_hint_override`

普通的 `model_type / dataset / evaluation_profile / primary_metric / secondary_metrics / output_type_hint` 视为 manager 自动生成字段；重新执行 `generate-script` / `run-npu` / `print-remote-command` 时会按最新规则重算并覆盖。若你手工改了这些普通字段但没有写进 `*_override`，下次 manager 仍会把它们刷新回自动画像结果。

## 业务画像分层

当前第四阶段画像不再只看顶层 `model_type`，而是按“顶层模型类型 + 业务意图子层”共同决定 `dataset / evaluation_profile`。最重要的一条是：`SequenceClassification` 只是模型头，不是业务语义。

当前已经与通用 harness 对齐、可直接闭环的典型路由包括：

- `reranker -> ms_marco -> reranker_ndcg`
- `extractive_qa -> squad_v2 -> qa_exact_match`
- `sentiment_binary -> imdb -> classification_accuracy`
- `sentiment_multiclass -> tweet_eval_sentiment -> classification_accuracy`
- `emotion_multiclass -> tweet_eval_emotion -> classification_accuracy`
- `offensive_binary -> tweet_eval_offensive -> classification_accuracy`
- `hate_binary -> tweet_eval_hate -> classification_accuracy`
- `topic_classification -> ag_news -> classification_accuracy`
- `natural_language_inference -> glue_mnli -> classification_accuracy`
- `question_pair_classification -> glue_qnli -> classification_accuracy`
- 只有在无法稳定细分时，才回退 `generic_classification -> sst2`

其中 `glue_mnli`、`glue_qnli` 这类文本对分类已经要求模板显式保留第二段文本，并在通用 `business_model_eval.py` 中按 `tokenizer(text, text_pair, ...)` 编码；不能再把它们当成单句分类静默跑掉。

## 常用命令

```bash
# 列出第四阶段模型（默认 `business_benchmark_status=completed`）
uv run python business_benchmark/scripts/business_benchmark_manager.py list

# 本机执行 NPU baseline/perf
uv run python business_benchmark/scripts/business_benchmark_manager.py run-npu --model "org/name"

# 仅生成统一入口脚本与业务配置（会自动补齐 remote_ssh_host / remote_project_root）
uv run python business_benchmark/scripts/business_benchmark_manager.py generate-script --model "org/name"

# 按模型自动决定第四阶段业务数据集并确保已下载
uv run python scripts/download_datasets.py --business-profile --model-id "org/name"

# 通过 SSH 自动执行远端 CUDA baseline、自动回收工件，并默认继续 summarize + check
uv run python business_benchmark/scripts/business_benchmark_manager.py run-remote-cuda --model "org/name"

# 只想自动跑远端并回收工件，暂时不 summarize/check
uv run python business_benchmark/scripts/business_benchmark_manager.py run-remote-cuda --model "org/name" --no-summarize --no-check

# 若你想把自动流程拆成分步执行，也优先走 Python 化的 SSH 直传入口
uv run python business_benchmark/scripts/business_benchmark_manager.py sync-remote-workspace --model "org/name"
ssh cuda-remote 'cd "$SLAI_REMOTE_PROJECT_ROOT/adaptations/sanitized_name" && UV_CACHE_DIR=$PWD/.uv_cache_remote UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple uv sync --extra cuda --no-install-project --frozen && CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=$PWD/.uv_cache_remote uv run --no-sync --extra cuda python "business_run.py" --scenario cuda_baseline'
uv run python business_benchmark/scripts/business_benchmark_manager.py fetch-remote-artifacts --model "org/name"

# 打印远端 CUDA 执行命令模板（仅在 SSH 不通或想人工分步执行时使用）
# 当前输出的是 Python 化的本地 sync/fetch 命令 + 一个远端执行命令，不再依赖 git pull
# 默认复用当前配置中的 `benchmark_run_id`，用于给刚完成的 NPU baseline/perf 补同一轮 CUDA baseline
uv run python business_benchmark/scripts/business_benchmark_manager.py print-remote-command --model "org/name"

# 若你明确要开启一轮全新的业务测评，再显式生成新的 `benchmark_run_id`
uv run python business_benchmark/scripts/business_benchmark_manager.py print-remote-command --model "org/name" --fresh-run-id

# 显式指定 SSH host alias / 远端仓库目录
uv run python business_benchmark/scripts/business_benchmark_manager.py print-remote-command --model "org/name" --ssh-host cuda-remote --remote-project-root "$SLAI_REMOTE_PROJECT_ROOT"

# 根据现有业务测评工件生成 business_summary.json
uv run python business_benchmark/scripts/business_benchmark_manager.py summarize --model "org/name"

# 不依赖 board.db，直接按 adaptation 目录汇总
uv run python business_benchmark/scripts/business_benchmark_manager.py summarize --adaptation "sanitized_name"

# 同样也可以直接使用底层汇总工具
uv run python business_benchmark/scripts/business_benchmark_tool.py summarize --adaptation "sanitized_name"

# 校验单个 adaptation 的 completed gate
uv run python business_benchmark/scripts/check_business_benchmark_run.py --adapt sanitized_name

# 校验进入 wait_cuda 前的本机 NPU baseline/perf sanity gate
uv run python business_benchmark/scripts/check_business_benchmark_run.py --adapt sanitized_name --wait-cuda-npu-only

# 仅校验 board.db 中已 completed 的第四阶段 adaptation
uv run python business_benchmark/scripts/check_business_benchmark_run.py --db-only
```

若本机 NPU baseline/perf 已完成，但远端 CUDA 因 SSH 不通、远端环境异常或自动回收失败暂时无法补齐，team-lead 可将 `business_benchmark_status` 写成 `wait_cuda`。写入前必须先通过 `check_business_benchmark_run.py --adapt sanitized_name --wait-cuda-npu-only`，确认本机 NPU baseline/perf 双路的 `num_samples`、字段完整性、质量指标以及 `npu_speedup_ratio >= 0.9` 都正常；若本机 NPU 双路已出现 `exact_match/accuracy/top1_accuracy/match_rate` 全 0、单路塌 0、明显漂移，或 `npu_perf` 相比 baseline 已明显退化，必须直接回 `pending` 修 evaluator / 标签归一化 / 数据集画像 / 计时口径 / NPU perf 继承链，禁止继续等待 CUDA。该状态会在平时保留为“等待 CUDA”的显式 backlog；但若 watchdog / `run_auto_team_lead.sh` 重启，会被 reset 回 `pending` 重新排队。

## Completed Gate

`business_benchmark_status=completed` 前必须满足：

- `optimization_status=completed`
- `business_summary.json` 存在，且写入 DB 的 notes 与文件原文逐字一致
- 三类 artifact 齐全：
  - NPU baseline
  - NPU perf
  - CUDA baseline
- 每类业务测评工件都包含 `latency_s / num_samples / mode / dataset / dtype / output_type / device / start_time / end_time`
- 新测量契约下（`measurement_contract_version >= 2`）还必须包含 `latency_measurement_scope / warmup_iterations / task_queue_enable / loaded_from_model_files / model_source_kind / tokenizer_source_kind`
- 第四阶段正式工件还必须保留运行时证据：`python_executable / python_version / package_versions / scenario_command`
- `num_samples > 50`
- 小样本 smoke run（如 `--max-samples 8`）不会再覆盖正式 `business_metrics_*.json`，而会自动写成带 `smoke{N}` 标签的工件；这些工件只用于链路验证，不能直接参与 completed 验收

## Remote CUDA Notes

- 推荐把真实 SSH 连接信息放进本机 `~/.ssh/config`，并统一使用 host alias `cuda-remote`；仓库内可参考非敏感模板 `business_benchmark/ssh_config.example`
- 默认优先使用 `business_benchmark_manager.py run-remote-cuda` 做自动 SSH 闭环；只有当 SSH 不通、远端环境异常、或你明确要人工分步执行时，才退回 `print-remote-command`
- `business_benchmark_manager.py print-remote-command` 已支持 `--ssh-host`，后续自动化或人工执行都优先传 host alias，不要把 `HostName / Port / User` 直接写进 repo
- `business_benchmark_manager.py run-remote-cuda` 与 `sync-remote-workspace` 会自动通过 SSH 直传同步 adaptation 目录代码与根下的 `scripts/dataset_mapping.py`、`scripts/download_datasets.py`；若远端缺失 `models/` 且本地存在，会自动补一次带断点续传的 `rsync`
- `fetch-remote-artifacts` 会通过 SSH 枚举远端 `business_metrics_cuda_*_baseline.json` 后逐个拉回本地；不再要求先在远端仓库里 `git pull`
- 拉回时会先校验工件是否仍然是纯 `cuda_baseline` 结果；若远端文件混入 `npu_*` 字段、`device` 不是 CUDA、或本地已存在同名文件，则不会覆盖本地，而是把远端副本隔离到 adaptation 下的 `remote_fetch_conflicts/`
- 远端通过 `ssh '...'` 的非交互命令不会读取 `~/.zshrc`；需要给 agent / 脚本继承的环境变量应放进 `~/.zshenv`
- 业务 CUDA 场景必须显式使用 adaptation 自己的 uv 环境和 `uv run --no-sync --extra cuda ...` 运行 `business_run.py` / `business_eval.py`；不要省略 extra，不要省略 `--no-sync`，也不要直接写 `.venv/bin/python ...`
- 远端 CUDA 安装阶段必须显式设置 adaptation 私有 `UV_CACHE_DIR`（推荐 `$PWD/.uv_cache_remote`），避免落回共享 `~/.cache/uv` 锁；同时建议显式设置 `UV_DEFAULT_INDEX/PIP_INDEX_URL` 到国内镜像，降低 `pypi.org` 超时
- 若远端 `uv sync --extra cuda --no-install-project --frozen` 长时间无新 stdout，不要无限等待；先检查共享 cache 锁、`pypi.org` 超时、以及远端 Python 版本是否满足 adaptation 的 `requires-python`。确认自动安装无推进后，应切换到“手工预装 `.venv` + `uv run --no-sync --extra cuda ...` 正式运行”路径
- `print-remote-command` 默认复用已有 `benchmark_run_id`，这是标准行为；只有你明确要开启全新一轮 NPU/CUDA 对比时，才使用 `--fresh-run-id`
- 若 `torch 2.6.0+cu124` 导入时报缺少 `libnvshmem_host.so.3`，需在该 adaptation 的虚拟环境内补装 `nvidia-nvshmem-cu12`
- `business_benchmark_config.json` 中的 `dataset_local_path` 必须使用远端机器上的真实绝对路径，不能直接复用本地 `/mnt/...` 路径
- 问答类模型若没有 `model_class` 元数据，仍可能靠 `model_id` 关键词推断；对这类历史配置，建议在规则升级后重新执行一次 `generate-script`，让自动字段按新画像刷新
- `business_eval.py` 是由仓库模板同步下发的托管 harness；如果你需要改第四阶段通用加载/打点/写盘逻辑，应修改 `business_benchmark/templates/business_eval.py`，不要直接在 adaptation 目录手改同名文件

## 历史结果刷新技巧

- 对已经存在正式 `business_metrics_npu_*`、`business_metrics_cuda_*`、`business_summary.json` 的模型按新规则重跑时，先把旧正式工件改名备份，推荐后缀 `__prev_rule_refresh_<timestamp>`；否则 `run-remote-cuda` 在回收新 CUDA 工件时遇到本地同名但内容不同的文件，会把远端新工件隔离到 `remote_fetch_conflicts/`，新结果无法直接接管 canonical 文件名
- 历史 `completed` 批量刷新应按模型逐个闭环：备份旧正式工件 -> `run-npu` -> `run-remote-cuda` -> `summarize/check` -> 用根目录 `scripts/board_ops.py update_business_benchmark_status --notes "$(cat business_summary.json)"` 立刻写库；不要等整批跑完后再统一回填，也不要直接写 SQL
- 远端 CUDA 长时间没有新 stdout 时，不要立刻判定失败；先同时看远端 `ps` 里是否还有 `business_eval.py`、`nvidia-smi` 是否仍有显存/进程、本地 `business_metrics_cuda_*` / `business_summary.json` 修改时间是否前进。大模型或 seq2seq 模型跑几分钟仍然健康是正常情况
- 批量刷新建议单独留一份日志目录，例如 `/tmp/slai_phase4_rule_refresh_<timestamp>/`，每个 adaptation 一份日志；这样中断后可以继续从单模型日志定位是卡在本机 NPU、远端 CUDA 还是写库阶段

## Tell Agent SSH

推荐先在本机 `~/.ssh/config` 里配置好 alias，例如 `cuda-remote`。之后每次需要 agent 跑远端 CUDA，只要告诉两项即可：

```text
remote_ssh_host=cuda-remote
remote_project_root=$SLAI_REMOTE_PROJECT_ROOT
```

也可以直接用自然语言：

```text
用 cuda-remote，仓库在 $SLAI_REMOTE_PROJECT_ROOT
```

注意：

- 文档、`.env`、仓库内 JSON 只保留 alias 和远端仓库路径，不提交真实 `HostName / Port / User / IdentityFile`
- 如果临时没有 alias，也可以一次性提供 `ssh user@host -p port` 与 `remote_project_root`，但这类真实连接信息不应写回仓库

## 后续扩展

当前已支持基于 SSH 直传的自动远端闭环；后续若需要进一步扩展为更复杂的远端调度器，也仍可复用现有配置、artifact 结构与 completed gate，而不需要修改 `board.db` 字段或 `business_summary.json` 结构。
