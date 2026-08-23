---
name: sentence_transformers_all_minilm_l6_v2 optimization
description: all-MiniLM-L6-v2 fp32 runtime-only batched inference 优化完成, speedup=5.52x
type: project
---

# sentence-transformers/all-MiniLM-L6-v2 优化笔记

## 任务概述
- model_id: sentence-transformers/all-MiniLM-L6-v2
- optimization_kind: runtime_only
- batched_inference(bs=8) + warmup(3x) + TASK_QUEUE_ENABLE=1
- speedup: 5.517002x (0.372s → 0.067s)

## 关键决策

### 1. runtime-only batched inference（无需融合算子）
- all-MiniLM-L6-v2 是 BERT 编码器模型，baseline 逐样本编码（bs=1）
- 直接用 batched inference（bs=8）将 60 个样本从 60 次 forward 降到 8 次
- 不需要 npu_rms_norm 或 npu_add_layer_norm，纯 runtime 优化即可获得 5.5x
- 精度完美：cosine=1.0, max_abs_error=1.19e-07

### 2. wikitext DatasetDict 修复
- 项目根 `datasets/wikitext___wikitext-2-raw-v1` 已存在
- `load_from_disk` 返回 DatasetDict（含 train/test/validation split）
- 需要显式选择 train split：`if hasattr(ds, "keys"): ds = ds["train"]`
- 不修复会报 `AttributeError: 'str' object has no attribute 'get'`
- baseline 和 perf 都需要修复

### 3. runtime_only notes 必填字段
- `parallel_mode: "single_card"` 必须显式记录
- `measurement_contract_version: 3` 必须存在
- `wall_clock_source: "artifact_explicit_field"`（当 metrics 有显式 wall_clock_s）
- 参考 [[google_t5_t5_small_notes]] 的字段修复经验

## 产物清单
- accuracy_run.py（修改：加 warmup(3x) + wall_clock_s + wikitext fix）
- accuracy_run_perf.py（新建：batched(bs=8) + warmup(3x) + TQE + compare）
- benchmark_metrics_npu_fp32_pretrained_wikitext.json（baseline, 60 samples）
- benchmark_metrics_npu_1_fp32_pretrained_wikitext_perf.json（perf, 60 samples）
- outputs_npu_fp32_pretrained_wikitext.pt, outputs_npu_1_fp32_pretrained_wikitext_perf.pt
- optimization_notes.json

## 最终结果
- baseline_wall_clock_s: 0.371708, perf_wall_clock_s: 0.067375
- speedup_ratio: 5.517002, latency_reduction_pct: 81.87%
- cosine_similarity: 1.0, max_abs_error: 1.192e-07
- NPU: npu:1 (Ascend910_9362), fp32, single-die, single_card
- dataset: wikitext (60 samples), output_type: cls_embeddings
