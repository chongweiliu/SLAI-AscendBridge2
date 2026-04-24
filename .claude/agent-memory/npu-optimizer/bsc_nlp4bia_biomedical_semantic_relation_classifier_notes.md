# BSC-NLP4BIA/biomedical-semantic-relation-classifier optimization notes

- Date: 2026-04-21
- Adaptation: `adaptations/bsc_nlp4bia_biomedical_semantic_relation_classifier`
- Status: completed

## Result

- Final passing path: `runtime_only + batched_inference(bs=8) + warmup(3x) + TASK_QUEUE_ENABLE=1`
- Baseline artifact: `benchmark_metrics_npu_0_fp32_pretrained_qiaojin___PubMedQA___pqa_labeled.json`
- Perf artifact: `benchmark_metrics_npu_0_fp32_pretrained_qiaojin___PubMedQA___pqa_labeled_perf.json`
- Wall-clock speedup: `3.810477x`
- Cosine similarity: `1.0`
- Max abs error: `1.2e-07`

## Key lesson

- This RoBERTa CLS-embedding workload is too small for `npu_add_layer_norm` / `npu_gelu(erf)` patching to pay off under the formal wall-clock contract.
- Tried fusion-like paths all with correct precision but negative speedup:
  - `npu_add_layer_norm`: `0.881887x`
  - `npu_gelu(erf)`: `0.937681x`
  - `npu_add_layer_norm + npu_gelu(erf)`: `0.840771x`
- The real bottleneck was Python-side per-sample tokenization / forward / `.cpu()` overhead in the perf script, not model math.
- Rewriting `accuracy_run_perf.py` to run the same pretrained dataset in batched inference mode (`bs=8`) preserved exact sample alignment and turned the runtime-only path into a valid completed result.

## Implementation notes

- Replaced stale non-compliant baseline/perf scripts and archived legacy artifacts before rerunning.
- Kept baseline artifact unchanged, but changed perf execution to batch multiple texts per forward while still writing one embedding per sample to `outputs_*_perf.pt`.
- Removed unnecessary model/tokenizer reload from `compare`, so the compare step became pure local artifact validation and no longer stalled on network/cache resolution.
