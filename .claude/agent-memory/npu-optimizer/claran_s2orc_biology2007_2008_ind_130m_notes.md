# claran/s2orc-biology2007-2008-ind-130m

- Date: 2026-04-21
- Result: `optimization_status=completed` candidate via `runtime_only`
- Final path: `warmup(3x) + TASK_QUEUE_ENABLE`

## Key findings

- Legacy 2007 optimization artifacts were rule-incompatible and had to be archived before rerun.
- Reusing the 2009-2010 OLMo family benchmark/optimization scripts was valid, but the 2007 adaptation could not rely on `AutoTokenizer.from_pretrained(MODEL_ID)` because Transformers tried to query HuggingFace and hit `httpx.ConnectTimeout`.
- Fix: resolve the adaptation-local HF snapshot from `models/models--claran--s2orc-biology2007-2008-ind-130m/refs/main` and load tokenizer/config/model from that snapshot with `local_files_only=True`.
- With local snapshot loading, baseline and perf both ran successfully on pretrained weights for 50 wikitext samples on a single NPU.
- This checkpoint prints an OLMo load report with many `UNEXPECTED`/`MISSING` keys, but baseline/perf are numerically aligned with each other. Compare passed with cosine `1.0`, min cosine `0.999999`, text match `50/50`, and PPL relative diff `0.0%`.
- Runtime-only wall-clock speedup was small but positive: `1.008556x`. Forward latency speedup was `1.248724x`.

## Artifacts

- Baseline metrics: `benchmark_metrics_npu_fp32_pretrained_wikitext.json`
- Perf metrics: `benchmark_metrics_npu_fp32_pretrained_wikitext_perf.json`
- Baseline outputs: `outputs_npu_fp32_pretrained_wikitext.pt`
- Perf outputs: `outputs_npu_fp32_pretrained_wikitext_perf.pt`
- Notes: `optimization_notes.json`
