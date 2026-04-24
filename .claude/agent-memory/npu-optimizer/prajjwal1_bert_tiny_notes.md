---
name: prajjwal1_bert_tiny_notes
description: prajjwal1/bert-tiny completed：旧脏工件重建后，npu_add_layer_norm + TQE + batched inference 在 CLS embeddings 合同下 9.04x 过 gate
type: reference
---

## prajjwal1/bert-tiny 优化结果

### 结论：completed - fusion

- DB 里这条被人工从 `skipped/not_applicable` 重置成 `pending` 后，不能信旧 notes
- 真正阻塞不是“模型太小没收益”，而是旧 baseline/perf 工件合同全脏：
  - `DATASET_DIR` 指到仓库外层，wikitext 经常误判缺失
  - step1/step2 只留下 `num_samples=1`
  - baseline/perf 缺 `wall_clock_s` / `selected_npu(s)` / `dataset`
  - 旧 compare/notes 把 `<1.0` 的历史结果写成当前结论

### 正式修法

- `accuracy_run.py`
  - 改成 adaptation 私有 snapshot + `local_files_only=True`
  - `DATASET_DIR = ADAPT_DIR.parent.parent / "datasets"`
  - baseline 直接写 50-sample `cls_embeddings` + 显式 `wall_clock_s`
- `accuracy_run_perf.py`
  - 保留 `npu_add_layer_norm` patch
  - perf 用 `warmup(3x) + TASK_QUEUE_ENABLE=1 + batched_inference(bs=8)`
  - compare 显式补齐 `baseline_samples/perf_samples/cuda_samples/ascend_samples`

### 最终通过结果

- 卡：`ASCEND_RT_VISIBLE_DEVICES=12`
- `optimization_kind = fusion`
- `optimization_items = npu_add_layer_norm + warmup(3x) + TASK_QUEUE_ENABLE + batched_inference`
- `speedup_ratio = 9.044886`
- `baseline_wall_clock_s = 0.493896`
- `perf_wall_clock_s = 0.054605`
- `perf_latency_s = 0.0010921`
- `perf_batch_size = 8`
- `cosine_similarity = 1.0`
- `min_cosine_similarity = 1.0`
- `max_abs_error = 1.28e-06`

### 教训

1. 小 BERT 不能被旧 `<1.0` runtime-only 记录直接判死；先看是不是工件合同错了
2. 对 `cls_embeddings` 模型，stage3 完整合同一旦拉正，`npu_add_layer_norm` 这种 post-LN fusion 依然可能给出很高收益
3. benchmark checker 的静态规则还会要求 `--use-pretrained` 对应显式 `from_pretrained/from_config` 分支，修完动态链路后别忘了补这个机械条件
