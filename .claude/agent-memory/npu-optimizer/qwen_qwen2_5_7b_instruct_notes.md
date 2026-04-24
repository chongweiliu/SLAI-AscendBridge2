# Qwen/Qwen2.5-7B-Instruct

Date: 2026-04-23
Card: `ASCEND_RT_VISIBLE_DEVICES=12`
Contract: local snapshot + `local_files_only=True` + pretrained teacher-forcing `last_token_logits + perplexity` on `wikitext`

## Outcome

- Final completed path: `runtime_only`
- Final optimization items: `warmup(3x) + TASK_QUEUE_ENABLE + batched_forward(bs=1)`
- Final metrics:
  - `baseline_wall_clock_s=2.315980`
  - `perf_wall_clock_s=2.212834`
  - `speedup_ratio=1.046613`
  - `baseline_latency_s=0.046320`
  - `perf_latency_s=0.044257`
  - `cosine_similarity=0.999995`
  - `min_cosine_similarity=0.999980`
  - `max_abs_error=0.0`
  - `ppl_avg_rel_diff_pct=0.0`

## What Failed First

- `rms_norm_swiglu_rotary`: `speedup_ratio=1.149541`, but `max_abs_error=0.640625`
- `rms_norm_swiglu`: still failed precision gate, `max_abs_error=0.625`
- `rms_norm`: `speedup_ratio=1.132571`, but `max_abs_error=0.50390625`

All three fusion paths looked faster on wall-clock, but none were eligible for completed because `check_accuracy_run_perf.py` enforces `max_abs_error < 0.001`.

## Key Lesson

For Qwen2.5-7B teacher-forcing logits workloads, do not treat higher wall-clock speedup from fusion as sufficient evidence. The real completed decision is dominated by the precision gate. If `npu_rms_norm` is already introducing `0.5+` max-abs error, narrowing from `rms_norm_swiglu_rotary` to `rms_norm` is enough to confirm the instability source, and the safe fallback is runtime-only.
