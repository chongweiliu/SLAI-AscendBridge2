---
name: facebookai_roberta_base_notes
description: FacebookAI/roberta-base (125M MLM) runtime_only batched mask logits 3.77x completed
metadata:
  type: project
---

FacebookAI/roberta-base (RoBERTa 125M, fp32, MaskedLM) NPU 优化完成（2026-08-22）。

**优化路径**：runtime_only
- warmup(3x) + TASK_QUEUE_ENABLE=1 + batched_mask_logits(bs=8)
- 废弃旧 fill-mask + pseudo-PPL 合同（每样本最多33次前向），改为单步前向提取 mask 位置 logits
- baseline/perf 均从 adaptation 私有 snapshot 加载（HF_HUB_OFFLINE=1 + local cache_dir）
- 同卡（npu:1）串行，对称 warmup(3x)

**结果**：
- baseline_wall_clock_s: 0.388319 → perf_wall_clock_s: 0.102878
- speedup_ratio: 3.774558（wall-clock 口径，>= 3.0 已补齐 comparison_method/comparison_scope/validation_note/steady_state_*）
- cosine_similarity: 0.99999992
- max_abs_error: 2.956390380859375e-05

**关键修复点**：
1. RoBERTa batched 推理中 `input_ids[j]` 是 1D [seq_len]，`nonzero(as_tuple=True)` 返回 tuple 长度为 1，需用 `mask_positions[0][0]` 而非 `mask_positions[1][0]`
2. 与 [[openai_community_gpt2_notes]] 同模板：废弃重计算合同改 logits-only，batched + warmup + TQE
3. RoBERTa tokenizer 有 pad_token，不需要像 GPT-2 那样手动设置

与 [[google_bert_bert_base_uncased]] 同族（BERT/RoBERTa 12层 encoder），确认 runtime_only batched logits 是稳定可过 gate 的路径。
