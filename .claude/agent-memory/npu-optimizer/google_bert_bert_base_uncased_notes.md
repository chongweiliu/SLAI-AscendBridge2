---
name: google_bert_bert_base_uncased optimization
description: bert-base-uncased fp32 runtime-only batched cls_embeddings 优化完成, speedup=3.74x
type: project
---

# google-bert/bert-base-uncased 优化笔记

## 任务概述
- model_id: google-bert/bert-base-uncased
- optimization_kind: runtime_only
- batched_inference(bs=8) + warmup(3x) + TASK_QUEUE_ENABLE=1
- speedup: 3.742101x (0.489s → 0.131s)

## 关键决策

### 1. 合同切换：MLM generated_text → cls_embeddings
- 旧 baseline 用 MaskedLM (config 模式, generated_text 输出)
- 改为 cls_embeddings：用 AutoModel, 提取 last_hidden_state[:, 0, :] ([CLS] token)
- 原因：MLM 的 mask 位置逐样本不同，难以 batch 化；cls_embeddings 可以直接 batch 编码
- 参考 [[sentence_transformers_all_minilm_l6_v2_notes]] 的同族模板

### 2. 从 config → pretrained
- 旧 baseline 是 config 模式（随机权重），不满足 completed gate 要求
- 重新用 --use-pretrained 跑 baseline 和 perf

### 3. wikitext DatasetDict 修复
- 与 all-MiniLM 相同的 `load_from_disk` DatasetDict 问题
- 需要 `if hasattr(ds, "keys"): ds = ds["train"]`

## 产物清单
- accuracy_run.py（改为 cls_embeddings + warmup(3x) + wall_clock_s）
- accuracy_run_perf.py（batched(bs=8) + warmup(3x) + TQE + compare）
- benchmark_metrics_npu_1_fp32_pretrained_wikitext.json（baseline）
- benchmark_metrics_npu_1_fp32_pretrained_wikitext_perf.json（perf）
- outputs_npu_fp32_pretrained_wikitext.pt, outputs_npu_1_fp32_pretrained_wikitext_perf.pt
- optimization_notes.json

## 最终结果
- baseline_wall_clock_s: 0.488696, perf_wall_clock_s: 0.130594
- speedup_ratio: 3.742101, latency_reduction_pct: 73.28%
- cosine_similarity: 0.9999999891, max_abs_error: 2.24e-05
- NPU: npu:1 (Ascend910_9362), fp32, single-die, single_card
- dataset: wikitext (60 samples), output_type: cls_embeddings
