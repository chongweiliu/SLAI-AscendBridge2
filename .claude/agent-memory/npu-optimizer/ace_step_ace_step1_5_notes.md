# ACE-Step/Ace-Step1.5 optimization notes

- Date: 2026-04-21
- Adaptation: `adaptations/ace_step_ace_step1_5`
- Status: completed

## Result

- Final passing path: `fp32 + npu_rms_norm + warmup(3x) + TASK_QUEUE_ENABLE=1`
- Baseline artifact: `benchmark_metrics_npu_0_fp32_pretrained_builtin_simple_mode.json`
- Perf artifact: `benchmark_metrics_npu_0_fp32_pretrained_builtin_simple_mode_perf.json`
- Wall-clock speedup: `1.111063x`
- Cosine similarity: `1.0`
- Max abs error: `1.449e-05`

## Key lesson

- This model's stage-3 workload uses `latent_hidden_states`, so completed gate requires `cosine >= 0.999`.
- In `bf16`, several patch combinations achieved real speedup but failed precision gate:
  - `npu_rms_norm + npu_swiglu + npu_rotary_mul`: `1.094x`, cosine `0.997963`
  - `npu_rms_norm + npu_swiglu`: `1.093x`, cosine `0.998393`
  - `npu_rms_norm`: `1.207x`, cosine `0.998521`
- `npu_swiglu` alone improved cosine (`0.998966`) but regressed wall-clock (`0.8738x`).
- The workable production path was switching both baseline and perf to `fp32`, then using only `npu_rms_norm`.

## Implementation notes

- Added `ACESTEP_MODEL_DTYPE` support in `accuracy_run.py` so baseline/perf can share the same dtype contract.
- Added `model_files/npu_patches.py` and wired `accuracy_run_perf.py` to load model-local patches only inside the adaptation directory.
- `npu_rotary_mul` needed per-tensor expand; reusing q-head expanded cos/sin for k-head tensors breaks when q/k head counts differ.
