# google-bert/bert-base-uncased

日期：2026-04-22

## 最终结果

- stage1 `adaptation_status=completed`
- stage2 `benchmark_status=completed`
- stage3 `optimization_status=completed`
- 路线：`runtime_only`
- 设备：物理卡 `13`
- 数据集：`wikitext`
- 样本数：`50`
- wall-clock：`0.417479s -> 0.096083s`
- `speedup_ratio=4.344983`
- `output_type=cls_embeddings`
- `cosine_similarity=0.99999994`
- `max_abs_error=1.09672546e-05`

## 关键修复

1. 重写 `accuracy_run.py`
- 改为只从 adaptation 内本地 snapshot 读取 pretrained 模型
- baseline metrics 补齐 `dataset/dtype/start_time/end_time/wall_clock_s/output_type`
- baseline 输出改成正式 50-sample `cls_embeddings` 工件，直接满足 stage2 completed gate

2. 重写 `accuracy_run_perf.py`
- 放弃旧 `self_baseline_same_model` 口径，改成独立 baseline/perf 工件
- runtime-only 优化项记录为 `batched_inference(bs=8) + warmup(3x) + TASK_QUEUE_ENABLE`
- compare 改成生成 `output_compare_perf.json`，并把 compare 样本数显式写成 50
- notes 使用 `independent_baseline_artifact`，继承 `selected_npu/selected_npus/device_topology/parallel_mode`

## 踩坑记录

1. 历史 stage2 工件会先拦住 stage3
- 旧 `benchmark_metrics_npu_fp32_config_wikitext.json` 只有 step1 单样本信息
- `check_accuracy_run.py --adapt` 会优先按 board gate 检 stage2 baseline 工件
- 所以 stage3 pending 里遇到这种模型，第一步仍然要先修 benchmark baseline，不然优化阶段永远写不回 completed

2. 旧融合失败记录不代表现在不能 completed
- 历史 notes 里是 `npu_add_layer_norm + TASK_QUEUE_ENABLE + warmup(3x)` 退化到 `0.9622x`
- 但把 perf 路线切到 `bs=8` 的 runtime-only 后，同样是 pretrained 同数据集同单卡，可以稳定到 `4.344983x`
- 这说明对小型 `cls_embeddings` 模型，批处理策略往往比算子 patch 更关键

## 复用建议

- 对 `cls_embeddings` / encoder-only 模型，优先考虑 runtime-only `batched_inference + warmup + TASK_QUEUE_ENABLE`
- 如果 board 里 stage3 是 pending，但 adaptation 目录只剩老的 step1 baseline metrics，先补 stage2 口径，不要急着直接修 perf
- baseline/perf 的 `latency_s` 必须严格等于 `wall_clock_s / num_samples`，否则这类非生成任务会被 board gate 拒绝
