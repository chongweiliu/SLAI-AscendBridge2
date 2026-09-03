---
name: benchmark-analysis
description: Benchmark 数据聚合、CUDA/NPU 输出对比、Raw Profiling 与 Trace 分析。支持 aggregate、compare、profiling、trace 命令。触发词："对比输出"、"聚合报告"、"trace 分析"、"profiling 解析"、"fallback 分析"、"NPU fallback"、"CUDA NPU 对比"。
---

# Benchmark Analysis Skill

聚合报告、对比 CUDA/NPU 输出、分析 raw profiling 与 Trace。`profiling` 子命令直接解析 `torch_npu.profiler` 原始目录，`trace` 子命令包含完整 NPU Fallback 分析（D2H/H2D 数据搬运、算子分类、fallback_ops、优化建议）。

## 与其他 benchmark skill 的区别

| Skill | 功能 |
|-------|------|
| `benchmark-script` | 生成并执行 `accuracy_run.py` |
| `benchmark-manager` | 管理运行、打包/解包产出文件 |
| `benchmark-analysis` | 聚合报告、对比输出、分析 raw profiling / trace |

---

## 命令

### aggregate - 聚合数据

扫描所有 adaptations 目录，聚合 benchmark 数据，生成汇总报告。

```bash
# 聚合所有 benchmark 数据
uv run python benchmark/scripts/benchmark_tool.py aggregate

# 按设备/模式过滤
uv run python benchmark/scripts/benchmark_tool.py aggregate --device npu --mode pretrained

# 指定输出格式
uv run python benchmark/scripts/benchmark_tool.py aggregate --format json,csv,figures
```

**参数**:

- `--device`: 过滤设备 (cuda/npu)

- `--mode`: 过滤模式 (config/pretrained)
- `--format`: 输出格式 (json,csv,figures)，默认 json,csv
- `--reports-dir`: 输出目录，默认 `benchmark/reports/`

**产出**:

- `benchmark/reports/aggregate.json`: 汇总报告

- `benchmark/reports/aggregate.csv`: CSV 格式
- `benchmark/figures/time_cost_hist.png`: 处理时间分布图

---

### compare - 对比输出

对比 CUDA 和 NPU 的输出结果，计算匹配率、余弦相似度等指标。

```bash
# 对比单个文件对
uv run python benchmark/scripts/benchmark_tool.py compare cuda.pt npu.pt

# 对比指定适配目录
uv run python benchmark/scripts/benchmark_tool.py compare --adaptation apple_mobilevit_small

# 对比所有适配目录
uv run python benchmark/scripts/benchmark_tool.py compare --all --format table

# 输出 JSON 报告
uv run python benchmark/scripts/benchmark_tool.py compare --all --output compare_report.json
```

**参数**:

- `cuda_pt`: CUDA .pt 文件路径

- `ascend_pt`: Ascend .pt 文件路径
- `--adaptation`: 指定适配目录名称
- `--all`: 对比所有适配目录
- `--format`: 输出格式 (json/table)，默认 table
- `--output`, `-o`: 输出 JSON 文件路径

---

### trace - 分析 Trace（含 NPU Fallback）

分析 trace 文件：统计 NPU/CPU ops，并基于 correlation id、显式 fallback 标记和关联的数据搬运区分“已确认 fallback”与“疑似 fallback”。单独出现的 `aten::` CPU frontend 事件不是 fallback 证据。

```bash
# 分析单个 trace 文件
uv run python benchmark/scripts/benchmark_tool.py trace trace_npu_fp32_config_wikitext.json

# 详细 Fallback 报告与建议（单文件）
uv run python benchmark/scripts/benchmark_tool.py trace trace_npu_bf16_config_wikitext.json -v

# 仅输出 JSON（单文件）
uv run python benchmark/scripts/benchmark_tool.py trace trace_npu_bf16_config_wikitext.json -j

# 分析指定适配目录的 trace
uv run python benchmark/scripts/benchmark_tool.py trace --adaptation apple_mobilevit_small

# 分析所有 trace 文件
uv run python benchmark/scripts/benchmark_tool.py trace --all --top-ops

# 输出 JSON 报告
uv run python benchmark/scripts/benchmark_tool.py trace --all --output trace_report.json
```

**参数**:

- `trace`: trace .json 文件路径
- `--adaptation`: 指定适配目录名称
- `--all`: 分析所有 trace 文件
- `--top-ops`: 显示 top CPU ops
- `-v`, `--verbose`: 打印详细 Fallback 报告与优化建议
- `-j`, `--json`: 仅输出 JSON（不打印报告）
- `--output`, `-o`: 输出 JSON 文件路径

**输出字段**:

- `npu_ops`: NPU 算子总次数
- `cpu_ops`: CPU 算子总次数
- `fallback_ratio`: 已确认 fallback 调用占可分类计算调用的比例
- `top_cpu_ops`: 最多使用的 CPU 算子
- `total_events`: 总事件数
- `d2h_count`: D2H 数据搬运次数
- `h2d_count`: H2D 数据搬运次数
- `fallback_ops`: 有显式标记或关联数据搬运证据的 fallback
- `suspected_fallback_ops`: 未关联到 NPU activity、但证据不足以确认的计算算子
- `fallback_evidence`: 每个已确认算子的证据类型
- `fallback_confidence`: `confirmed` / `suspected` / `none`
- `compute_on_npu`: 在 NPU 上执行的计算算子
- `dispatch_on_cpu`: 调度入口算子（CPU 调度、NPU 计算）
- `has_fallback`: 是否存在关键算子 fallback
- `has_data_transfer`: 是否存在 D2H/H2D 搬运

---

### profiling - 分析 raw profiling 目录

直接解析 `torch_npu.profiler.tensorboard_trace_handler()` 生成的目录，不依赖 MindStudio。支持 CSV 快速扫描和 SQLite `cluster.db` 深度分析（`--deep`）。

**CANN 版本兼容**：自动适配 CANN 7.x（`NpuOperator` 表）和 CANN 8.x（`PYTORCH_API` + `STRING_IDS` 表）。自动搜索 `cluster.db`、`summary.db`、`analysis.db`、`ascend_pytorch_profiler.db`。

```bash
# 分析单个 profiling 目录
uv run python benchmark/scripts/benchmark_tool.py profiling Profiling_L1

# 仅输出 JSON
uv run python benchmark/scripts/benchmark_tool.py profiling Profiling_L1 -j

# SQLite 深度分析（解析 cluster.db 中的 NPU 算子耗时）
uv run python benchmark/scripts/benchmark_tool.py profiling Profiling_L1 --deep

# 扫描指定 adaptation
uv run python benchmark/scripts/benchmark_tool.py profiling --adaptation lightricks_ltx_2_3

# 扫描指定 adaptation 并启用深度分析
uv run python benchmark/scripts/benchmark_tool.py profiling --adaptation lightricks_ltx_2_3 --deep -j

# 扫描所有 adaptation 并导出报告
uv run python benchmark/scripts/benchmark_tool.py profiling --all --output benchmark/reports/profiling_report.json
```

**参数**:

- `profiling_dir`: raw profiling 目录路径
- `--adaptation`: 扫描指定适配目录中的 profiling 目录
- `--all`: 扫描所有适配目录中的 profiling 目录
- `--deep`: 启用 SQLite cluster.db 深度分析（解析 NPU 算子耗时，自动适配 CANN 7.x/8.x schema）
- `--top-k`: 显示 Top K API / op / kernel，默认 10
- `-j`, `--json`: 仅输出 JSON（不打印报告）
- `--output`, `-o`: 输出 JSON 文件路径

**输出字段**:

- `source_dir`: profiling 目录路径
- `detected_files`: 识别到的 profiling 文件
- `missing_inputs`: 缺失的核心输入文件
- `step_summary`: step 总数、总耗时、平均耗时、计算/通信耗时
- `api_summary`: Top API 耗时、总调用次数
- `op_summary`: Top operator 耗时、总调用次数
- `kernel_summary`: Top kernel 耗时、shape/step 摘要
- `cluster_db`: （仅 `--deep`）NPU 算子耗时统计（`npu_op_summary`）和 CANN API 统计（`cann_api_summary`）
- `warnings`: 缺文件、空目录、字段不兼容等提示

---

## 输出类型

`compare` 命令自动检测输出类型：

| 类型 | 说明 | 对比指标 |
|------|------|---------|
| `generated_text` | CausalLM/Seq2Seq 解码文本 | match_rate, avg_text_similarity |
| `class_labels` | Vision 分类标签 | match_rate |
| `cls_embeddings` | BERT [CLS] 向量 | avg_cosine_similarity, max_abs_error |
| `mixed` | 字典格式：generated_text + logits + perplexity | text_match_rate, logits_avg_cosine_similarity, ppl_avg_rel_diff |

### mixed 格式（新）

```python
# outputs_*.pt 内容
{
  "generated_text": ["文本1", "文本2", ...],
  "logits": [tensor([vocab_size]), ...],
  "perplexity": [15.23, 12.45, ...]
}
```

**对比结果字段**:

- `text_match_rate`: 文本完全匹配率

- `text_exact_matches`: 完全匹配数量
- `logits_avg_cosine_similarity`: logits 平均余弦相似度
- `logits_max_abs_error`: logits 最大绝对误差
- `ppl_avg_rel_diff`: perplexity 平均相对差异
- `ppl_cuda_avg`: CUDA perplexity 平均值
- `ppl_ascend_avg`: NPU perplexity 平均值

---

## 对比结果表格示例

```
Adaptation                         Type       Text Match Logits Cos  PPL Diff   Max Error
------------------------------------------------------------------------------------------------------------------------
qwen_qwen2_0_5b                    mixed      100.0%     0.999998   0.03%      0.000015
apple_mobilevit_small              class_lbl  100.0%     -          -          -
```

---

## 常用工作流

### 1. 完整验证流程

```bash
# 1. 运行 CUDA benchmark
uv run python benchmark/scripts/benchmark_manager.py run --hardware cuda --pretrained

# 2. 运行 NPU benchmark
uv run python benchmark/scripts/benchmark_manager.py run --hardware npu --pretrained

# 3. 对比结果
uv run python benchmark/scripts/benchmark_tool.py compare --all --format table

# 4. 分析 raw profiling / trace
uv run python benchmark/scripts/benchmark_tool.py profiling --adaptation qwen_qwen2_0_5b
uv run python benchmark/scripts/benchmark_tool.py trace --all --top-ops

# 5. 聚合报告
uv run python benchmark/scripts/benchmark_tool.py aggregate --format json,csv,figures
```

### 2. 单模型深度分析

```bash
# 对比单模型
uv run python benchmark/scripts/benchmark_tool.py compare --adaptation qwen_qwen2_0_5b

# 分析 raw profiling 查看热点，再分析 trace 查看 fallback 情况
uv run python benchmark/scripts/benchmark_tool.py profiling --adaptation qwen_qwen2_0_5b
uv run python benchmark/scripts/benchmark_tool.py trace --adaptation qwen_qwen2_0_5b --top-ops
```

### 3. 生成报告

```bash
# 生成完整报告
uv run python benchmark/scripts/benchmark_tool.py aggregate
uv run python benchmark/scripts/benchmark_tool.py profiling --all --output benchmark/reports/profiling.json
uv run python benchmark/scripts/benchmark_tool.py compare --all --output benchmark/reports/compare.json
uv run python benchmark/scripts/benchmark_tool.py trace --all --output benchmark/reports/trace.json
```

---

## 验收标准

### compare 结果

| 输出类型 | 指标 | 要求 |
|---------|------|------|
| generated_text / class_labels | match_rate | 100% |
| cls_embeddings / logits | cosine_similarity | ≥ 0.999 |
| cls_embeddings / logits | max_abs_error | < 0.001 |
| mixed (perplexity) | ppl_avg_rel_diff | < 5% |

### trace 分析

- `fallback_confidence=confirmed`：可以进入算子替代方案调查，但仍需结合耗时确认它是端到端瓶颈
- `fallback_confidence=suspected`：必须重新采集带 correlation id 的 L1 trace 或结合 runtime fallback 日志，不能直接触发自定义算子开发
- `fallback_confidence=none`：仅表示当前 trace 没有可确认的 fallback 证据，不代表所有 shape/dtype 路径均受支持
