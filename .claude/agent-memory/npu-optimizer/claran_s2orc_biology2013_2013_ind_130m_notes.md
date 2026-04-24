# claran/s2orc-biology2013-2013-ind-130m

- Date: 2026-04-21
- Result: `optimization_status=completed` candidate via `runtime_only`
- Final path: `warmup(3x) + TASK_QUEUE_ENABLE`

## Key findings

- The previous pending state came from a legacy runtime-only script that measured cold-vs-warm self-baseline artifacts and produced `speedup_ratio=0.981`. That path was not aligned with the current optimization completed gate.
- Replacing both `accuracy_run.py` and `accuracy_run_perf.py` with the standard OLMo baseline/perf/compare contract fixed two issues at once:
  1. baseline artifacts now include the required `dataset` field;
  2. speedup is computed from canonical baseline/perf artifacts instead of the old self-baseline script.
- Like the 2007 variant, this model must load tokenizer/config/model from the adaptation-local HF snapshot with `local_files_only=True`; otherwise repo-id loading can fall back to online access behavior.
- Compare passed with cosine `1.0`, min cosine `0.999999`, text match `50/50`, and PPL relative diff `0.0%`.
- Final runtime-only wall-clock speedup was `1.006968x`; forward latency speedup was `1.418074x`.

## Artifacts

- Baseline metrics: `benchmark_metrics_npu_fp32_pretrained_wikitext.json`
- Perf metrics: `benchmark_metrics_npu_fp32_pretrained_wikitext_perf.json`
- Baseline outputs: `outputs_npu_fp32_pretrained_wikitext.pt`
- Perf outputs: `outputs_npu_fp32_pretrained_wikitext_perf.pt`
- Notes: `optimization_notes.json`
