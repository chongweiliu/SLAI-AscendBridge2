# Benchmark-Runner: accuracy_run.py 模式与反模式

本文档记录 accuracy_run.py 编写时的反模式（禁止）与正确示例，供 benchmark-runner 在生成或手动改写脚本时参考。
细节见同目录主题文件（本索引保持精简）：

- `gate_and_seq2seq.md` — completed gate 字段要求（board_ops 强制）+ seq2seq/T5 模板改写详解
- `special_models.md` — MoE 大模型 / ESPnet ASR / TTS (Qwen3-TTS) 特殊处理详解
- `correct_examples.md` — 正确代码示例（CACHE_DIR / setup_model / 数据集 / 命名 / 检查命令）

## completed gate 要点（详见 gate_and_seq2seq.md）

- `check_accuracy_run.py --adapt` 会模拟 completed gate：NPU baseline metrics 必须 `num_samples >= 50`
- 即使 team-lead 说"10 样本即可"，要让 `--adapt` 通过也必须跑 >=50（小模型直接跑 60）
- 模板 metrics 字典**缺** `dataset`、`dtype` 字段，必须手动补（`"dataset": dataset_name`、`"dtype": dtype_str`）
- gate 字段：`latency_s`(>0)/`num_samples`/`mode`(pretrained|config)/`dataset`/`dtype`(fp32|fp16|bf16)/`output_type`/`device`(含npu)/`start_time`/`end_time`；`ttft_ms`/`tpot_ms` 数值或 null 且 `ttft_ms <= latency_s*1000`
- `num_samples` 写实际统计数 `min(len(texts), max_samples)`

## seq2seq (T5) 要点（详见 gate_and_seq2seq.md）

- 模板 `model(**inputs)` 对 encoder-decoder 报 `ValueError: ... decoder_input_ids ...`
- Step 1 用 `generate()` 做 trace 负载；Step 2 非流式 greedy 拿 ids → `decoder_input_ids=ids[:, :-1]` forward 提取 logits + 按生成序列算 PPL；TTFT/TPOT 用第二次流式生成测量
- T5 输入必须带任务前缀（"translate English to German: ..." / "summarize: ..."）；T5 `torch_dtype="auto"` 为 fp32

## MLM (RoBERTa/BERT) 画像要点

- 用 `AutoModelForMaskedLM`，单 `<mask>` 句：forward → mask 位置 argmax 填空 + 该位置 logits
- 伪困惑度 = exp(逐位 mask 的真实 token NLL 均值)（上限 32 位置控开销）；非生成式，`ttft_ms`/`tpot_ms` 写 null（gate 允许）
- outputs 字典沿用 `generated_text`(mask 填空字符串)/`logits`/`perplexity` 三键，`output_type=generated_text`，兼容 compare
- RoBERTa 的 mask 字面量为 `<mask>`，vocab 50265；内置 66 条单 mask 常识句足够跑 60

## MoE / 大模型 / 特殊模型要点（详见 special_models.md）

- MoE 无 flash-linear-attention 时极慢；用 `generate(return_dict_in_generate=True, output_scores=True)` 合并、`local_files_only=True`
- ESPnet ASR 用 `Speech2Text` 非标准模板，Python 3.12 不兼容（llvmlite），建议 `--cpu`
- TTS (Qwen3-TTS) NPU 上 pretrained 崩溃（MultinomialWithReplacement AICPU），只能 config mode；dataset_name="synthetic"

## 反模式清单（严禁）

| 反模式 | 问题 | 正确做法 |
|--------|------|----------|
| `cache_dir = "./models"` | 相对路径依赖 cwd | `CACHE_DIR = (Path(__file__).resolve().parent / "models").as_posix()` |
| 定义 `--use-pretrained` 但不分支 | Tier1/Tier2 无法区分 | `if use_pretrained: from_pretrained(...) else: from_config(...)` |
| `load_dataset(..., trust_remote_code=True)` | HF 2.16+ 自定义脚本数据集已弃用 | `load_from_disk(DATASET_DIR / "xxx")` |
| 输出文件无 dataset 后缀 | 不符合命名规范 | `trace_npu_0_fp32_config_wikitext.json` |
| dataset 后缀与实际不符 | 误导聚合 | 使用实际加载的 dataset_name |
| 硬编码 `_config_` 或 `_fp32_` | mode/dtype 不符实际 | 使用 `mode_str`、`dtype_str` 动态 |
| `dtype_str` 按设备推断（`device.startswith("npu")` 等） | 与实际模型 dtype 可能不符 | `dtype_str = get_dtype_str(next(model.parameters()).dtype)` |
| `max_samples` 默认 10 | 与 R1 规则冲突 | `default=250` |
| `def load_dataset(...)` | 与 datasets 库冲突 | `def load_benchmark_texts()` |
| 未检查 `len(texts)==0` | 空数据集 IndexError | `text = texts[0] if texts else "fallback"` |
| **shrink 函数**（如 `shrink_config_for_dry_run`） | **严禁** | 直接 `from_config(config)`，不修改 config |
| **config 分支中 `model = model.cpu()`** | 模型必须在 device（NPU/CUDA）上推理 | `model = model.to(device)` |
| **init_empty_weights 创建后丢弃再 from_config** | 前者无效、后者全量加载 | 直接 `from_config` + `model.to(device)` |
| pretrained 失败后 silent fallback 到 config | 伪 benchmark 结果 | 直接抛错退出并上报失败 |

## 强制检查（规则强制，禁止跳过）

- 生成或修改 `accuracy_run.py` 后**必须**执行
  `uv run python benchmark/scripts/check_accuracy_run.py --adapt {name}`（项目根，参数只传目录名），
  违规 exit 1 必须修复后重跑直至通过（含 completed gate 模拟）
- **Benchmark-Runner 禁止**调用 `update_benchmark_status`；仅通过 SendMessage 报告结果，
  由 team-lead 统一更新看板并执行 git commit（避免重复 commit）
- CI 已移除自动检查；每轮启动前跑"电子狗"，输出写入 `logs/team_lead_*.log`

## 参考

- benchmark-runner.md 2.10 禁用手册 / 模板强制检查
- benchmark-script/SKILL.md 9.4 手动编写规范、9.5 常见错误
- dataset-mapping/SKILL.md 4.1 数据集加载方式
- scripts/board_ops.py `_validate_benchmark_metric_artifacts`

## ⚠️ nopua skill — 遇到困境必须调用

**nopua 不会自动触发**，需要主动 `Skill("nopua")`。

**触发条件**：同一 action 失败 2+ 次 / 陷入等待循环 / 被动等待而不改变策略。

**正确用法**：1. 停止当前循环 2. 查询 board.db 获取真实状态 3. 根据状态决定下一步 4. 写教训到 MEMORY。

**反面教训**：benchmark-runner 若 check_accuracy_run.py 反复失败，应立即读取脚本源码查根因，而非重试 5+ 次。

## 主题文件索引

- [completed-gate-quirks.md](completed-gate-quirks.md) — `--adapt` 检查隐含 completed gate：metrics 需含 `dataset`/`dtype` 字段且 `num_samples>=50`；本机改模板需换 idle 选卡、去 `device_map="auto"`
