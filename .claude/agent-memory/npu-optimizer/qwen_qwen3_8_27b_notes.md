# Qwen/Qwen3.8-27B

Date: 2026-08-23
Card: baseline `npu:0`, perf `npu:1` (Ascend910_9362, selected via mem_get_info)
Contract: adaptation cache + pretrained teacher-forcing `last_token_logits + perplexity` on `wikitext` (test split, 50 samples)

## Outcome

- Final completed path: `runtime_only`
- Final optimization items: `warmup(3x) + TASK_QUEUE_ENABLE=1 + batched_inference(bs=2, right-padding)`
- Final metrics:
  - `baseline_wall_clock_s=25.816701`
  - `perf_wall_clock_s=13.681974`
  - `speedup_ratio=1.886913`
  - `baseline_latency_s=0.516334` (wall_clock/num_samples)
  - `perf_latency_s=0.273639` (wall_clock/num_samples)
  - `cosine_similarity=0.99998847`
  - `min_cosine_similarity=0.99998432`
  - `ppl_avg_rel_diff_pct=0.084712`
  - max_abs_error=0.0546875 (kept in output_compare_perf.json only)

## Key Details

1. **Model architecture**: qwen3_5, loaded via `AutoModelForImageTextToText` (not `AutoModelForCausalLM`). Text-only forward works fine with `input_ids` + `attention_mask`.
2. **Model size**: 27B params, fp16 ~52GB on single 64GB Ascend910 card. No device_map=auto.
3. **Vocab size**: 248320 (large — logits tensor is ~248K per sample)
4. **Baseline conversion**: Original accuracy_run.py used `generate()` (max_new_tokens=16, output_type=generated_text). Converted to teacher-forcing forward (logits+PPL) for both baseline and perf, matching [[qwen_qwen2_5_7b_instruct_notes]] pattern.
5. **Dataset**: wikitext test split (2891 samples available, used 50). `datasets` package was missing from pyproject.toml — added `datasets>=2.14` and `numpy>=1.24`.
6. **Per-sample baseline latency**: ~0.516s (bs=1 teacher-forcing, 27B model)
7. **transformers 5.x**: `torch_dtype` deprecated, use `dtype` instead (but `torch_dtype=torch.float16` still works with a warning)
8. **Qwen3.5 modeling warning**: `Cannot create tensor with internal format while allow_internel_format=False` — harmless warning from `torch_npu` TensorFactories.

## What Was Tried

- Runtime-only only (no fusion ops attempted). Based on [[qwen_qwen2_5_7b_instruct_notes]] experience, Qwen family fusion ops (RMSNorm/SwiGLU/RoPE/GQA) typically fail max_abs_error gate on NPU. Runtime-only batched inference is the proven path.
- bs=2 right-padding + TQE gives 1.887x speedup (47% latency reduction). Similar to Qwen2.5-7B-Instruct's 1.61x but higher due to larger model (more compute per forward, better batching gains).
