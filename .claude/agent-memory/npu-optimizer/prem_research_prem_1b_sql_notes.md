---
name: prem_research_prem_1b_sql_notes
description: prem-research/prem-1B-SQL completed：历史 5.33x 不可直接复用，重建 pretrained teacher-forcing 合同并补齐 perf metrics device 字段后，以 1.03x 通过 gate
type: reference
---

## prem-research/prem-1B-SQL 优化结果

### 结论：completed - fusion

- 历史 memory 里的 `5.33x` 不是这轮正式写库通过的可追溯工件，不能继续当当前结论。
- 当前可交付链路是重新构建 baseline/perf，同一卡串行跑完 baseline、perf、compare、三道 gate 后再写库。

### 正式修法

- `accuracy_run.py`
  - adaptation 私有 snapshot + `local_files_only=True`
  - pretrained teacher-forcing `last_token_logits + perplexity`
  - 显式写 `wall_clock_s`、`device`、`selected_npu(s)`、`device_topology`
- `accuracy_run_perf.py`
  - 保留 `npu_rms_norm + npu_swiglu + npu_rotary_mul`
  - perf 用 `warmup(3x) + TASK_QUEUE_ENABLE=1 + batch_size=1`
  - compare 从 perf 工件继承 `selected_npu(s)` / `device_topology`
  - 关键机械坑：perf metrics 缺少 `device` 字段时，`check_accuracy_run_perf.py` 会直接拦 completed；必须补齐 `device/device_model/runtime metadata/packages`

### 最终通过结果

- 卡：`ASCEND_RT_VISIBLE_DEVICES=12`
- `optimization_kind = fusion`
- `speedup_ratio = 1.030532`
- `baseline_wall_clock_s = 1.796303`
- `perf_wall_clock_s = 1.743084`
- `perf_latency_s = 0.034862`
- `cosine_similarity = 1.0`
- `min_cosine_similarity = 0.999999`
- `max_abs_error = 0.00018692`
- `ppl_avg_rel_diff_pct = 0.0004`

### 教训

1. 历史高倍提速如果对不上当前 baseline/perf/output/notes 四件套，只能当参考，不能当 completed 证据。
2. decoder LM 的 stage3 完成前，三道 gate 必须全跑；compare 正确不代表 perf metrics 字段齐全。
3. 对这类 1B LLaMA 族模型，融合 patch 仍然可能成立，但正式 speedup 应以当前同一卡、同合同、同轮工件为准，而不是旧实验峰值。
