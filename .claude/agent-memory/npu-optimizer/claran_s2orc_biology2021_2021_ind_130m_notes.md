# claran/s2orc-biology2021-2021-ind-130m

- Date: 2026-04-21
- Result: `optimization_status=completed` candidate via `runtime_only`
- Final path: `warmup(3x) + TASK_QUEUE_ENABLE`

## Key findings

- The pending state came from a legacy self-baseline / fusion-oriented script that concluded `speedup_ratio < 1`. That path was not aligned with the current completed gate.
- Replacing the benchmark and perf scripts with the standard OLMo baseline/perf/compare contract fixed the measurement path and removed legacy artifact pollution.
- This model also requires adaptation-local snapshot loading with `local_files_only=True` to avoid depending on repo-id online behavior.
- Compare passed with cosine `1.0`, min cosine `0.999999`, text match `50/50`, and PPL relative diff `0.0%`.
- Final runtime-only wall-clock speedup was small but positive: `1.000549x`. Forward latency speedup was `1.574320x`.

## Artifacts

- Baseline metrics: `benchmark_metrics_npu_fp32_pretrained_wikitext.json`
- Perf metrics: `benchmark_metrics_npu_fp32_pretrained_wikitext_perf.json`
- Baseline outputs: `outputs_npu_fp32_pretrained_wikitext.pt`
- Perf outputs: `outputs_npu_fp32_pretrained_wikitext_perf.pt`
- Notes: `optimization_notes.json`
