# Qwen/Qwen3-8B

Date: 2026-04-23
Card: `ASCEND_RT_VISIBLE_DEVICES=12`
Contract: local snapshot + `local_files_only=True` + pretrained teacher-forcing `last_token_logits + perplexity` on `wikitext`

## Outcome

- Final completed path: `runtime_only`
- Final optimization items: `warmup(3x) + TASK_QUEUE_ENABLE + batched_forward(bs=1)`
- Final metrics:
  - `baseline_wall_clock_s=2.933299`
  - `perf_wall_clock_s=2.655814`
  - `speedup_ratio=1.104482`
  - `baseline_latency_s=0.058666`
  - `perf_latency_s=0.053116`
  - `cosine_similarity=0.999989`
  - `min_cosine_similarity=0.999972`
  - `max_abs_error=0.0`
  - `ppl_avg_rel_diff_pct=0.0`

## What Failed First

- `rms_norm_swiglu_rotary`: wall-clock better than runtime-only, but `max_abs_error=0.84375`
- `rms_norm_swiglu`: still failed precision gate, `max_abs_error=0.6875`
- `rms_norm`: same precision failure plateau, `max_abs_error=0.6875`

All three fusion paths were ineligible for completed because `check_accuracy_run_perf.py` enforces `max_abs_error < 0.001`.

## Key Lesson

For Qwen3-8B teacher-forcing logits workloads, treat fusion as exploratory only. Once narrowing shows the residual `npu_rms_norm` path already produces `0.68+` max-abs error, stop spending cycles on deeper fusion variants and switch to runtime-only on the same card.
