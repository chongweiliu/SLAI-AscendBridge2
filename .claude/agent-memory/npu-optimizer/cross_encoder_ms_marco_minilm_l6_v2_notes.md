---
name: cross_encoder_ms_marco_minilm_l6_v2 optimization
description: cross-encoder/ms-marco-MiniLM-L6-v2 fp32 runtime-only batched rerank_scores 优化完成, speedup=5.12x
type: project
---

# cross-encoder/ms-marco-MiniLM-L6-v2 优化笔记

## 任务概述
- model_id: cross-encoder/ms-marco-MiniLM-L6-v2
- optimization_kind: runtime_only
- batched_inference(bs=16, 8 queries × 2 pairs) + warmup(3x) + TASK_QUEUE_ENABLE=1
- speedup: 5.12024x (0.268s → 0.052s)

## 关键决策

### 1. Cross-encoder batched 推理模式
- baseline 逐 query 处理（每 forward 2 pairs: relevant + irrelevant）
- perf 批量 8 queries（每 forward 16 pairs），forward 次数从 59 降到 8
- 需要 flatten queries/passages 再 score_batch，然后 split 回 per-query scores

### 2. 从 config → pretrained
- 旧 baseline 是 config 模式（随机权重），ranking_accuracy=0.5763
- pretrained 下 ranking_accuracy=1.0（59/59 全部正确）
- 添加 warmup(3x) + wall_clock_s 到 accuracy_run.py

### 3. rerank_scores compare
- 对比 baseline/perf 的 scores（flatten 成 1D tensor 后计算 cosine）
- cosine=1.0, max_abs_error=2.86e-06, ranking match=True

## 产物清单
- accuracy_run.py（加 warmup(3x) + wall_clock_s）
- accuracy_run_perf.py（batched(bs=16) + warmup(3x) + TQE + compare）
- benchmark_metrics_npu_fp32_pretrained_builtin.json（baseline, 118 samples）
- benchmark_metrics_npu_fp32_pretrained_builtin_perf.json（perf, 118 samples）
- outputs_npu_fp32_pretrained_builtin.pt, outputs_npu_fp32_pretrained_builtin_perf.pt
- optimization_notes.json

## 最终结果
- baseline_wall_clock_s: 0.26849, perf_wall_clock_s: 0.052437
- speedup_ratio: 5.12024, latency_reduction_pct: 80.47%
- cosine_similarity: 1.0, max_abs_error: 2.861e-06
- ranking: 59/59 correct for both baseline and perf (match=True)
- NPU: npu:1 (Ascend910), fp32, single-die, single_card
- dataset: builtin (59 query pairs = 118 samples), output_type: rerank_scores
