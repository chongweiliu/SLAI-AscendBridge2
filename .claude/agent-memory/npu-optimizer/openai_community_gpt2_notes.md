---
name: openai_community_gpt2_notes
description: openai-community/gpt2 (124M) runtime_only teacher-forcing batched inference 2.81x completed
metadata:
  type: project
---

openai-community/gpt2 (GPT-2 124M, fp32) NPU 优化完成（2026-08-22）。

**优化路径**：runtime_only
- warmup(3x) + TASK_QUEUE_ENABLE=1 + batched_teacher_forcing(bs=4)
- 放弃旧 `generated_text + TextIteratorStreamer` 合同，改为 teacher-forcing `last_token_logits + perplexity`
- baseline/perf 均从 adaptation 私有 snapshot 加载（HF_HUB_OFFLINE=1 + local cache_dir）
- 同卡（npu:1）串行，对称 warmup(3x)

**结果**：
- baseline_wall_clock_s: 0.52702 → perf_wall_clock_s: 0.187826
- speedup_ratio: 2.805895（wall-clock 口径）
- cosine_similarity: 0.99999981
- max_abs_error: 0.0001220703125
- ppl_avg_rel_diff: 0.0002%

**关键修复点**：
1. GPT-2 tokenizer 无 pad_token，批量 padding 前必须 `tokenizer.pad_token = tokenizer.eos_token`
2. `batched_teacher_forcing` 中 `device` 是 `torch.device` 对象，需 `str(device).startswith("npu")` 而非 `device.startswith("npu")`
3. per-sample latency 在 batch 模式下需 `batch_latency / len(batch_texts)` 展开到 per-sample
4. `optimization_notes.json` 必须含 `measurement_contract_version>=3`、`wall_clock_source`、`warmup_policy="symmetric"`
5. 非生成类任务 `perf_latency_s` 必须等于 `perf_wall_clock_s / num_samples`（gate 强制校验）
6. compare 函数需同时更新 baseline/perf metrics 文件中的 `latency_s`，否则 `best_result.perf_latency_s` 与 `perf_artifact.latency_s` 不一致

与 [[IRIIS-RESEARCH/GPT2_Nepali_124M]] 同族模板，确认 GPT-2 124M 级小模型走 teacher-forcing + batched inference 是稳定可过 gate 的路径。
