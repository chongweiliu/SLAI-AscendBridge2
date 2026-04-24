---
name: openai_clip_vit_base_patch16_notes
description: openai/clip-vit-base-patch16 completed：重建为本地 snapshot + CIFAR100 image_embeddings，最终 bs=1 + TASK_QUEUE_ENABLE=1 过 gate
type: reference
---

## openai/clip-vit-base-patch16 优化结果

### 结论：completed - runtime_only

- 不要再信历史 `not_applicable` 记录；旧 stage3 工件和旧 memory 都是过期结论
- 正式合同改成 `image_embeddings`，baseline/perf 都走 adaptation 私有 snapshot + `local_files_only=True`
- 数据集固定 `cifar100`，基线工件：
  - `outputs_npu_0_bf16_pretrained_cifar100.pt`
  - `benchmark_metrics_npu_0_bf16_pretrained_cifar100.json`
- perf 工件：
  - `outputs_npu_0_bf16_pretrained_cifar100_perf.pt`
  - `benchmark_metrics_npu_0_bf16_pretrained_cifar100_perf.json`

### 关键发现

- `batch_size=16/8/4/2` 都有明显提速，但 `max_abs_error` 过不了 completed gate
- 最终只有 `batch_size=1 + TASK_QUEUE_ENABLE=1` 同时满足：
  - `speedup_ratio > 1.0`
  - `cosine_similarity = 1.0`
  - `max_abs_error = 0.0`

### 最终通过结果

- 卡：`ASCEND_RT_VISIBLE_DEVICES=12`
- `optimization_kind = runtime_only`
- `speedup_ratio = 1.713962`
- `perf_wall_clock_s = 0.663685`
- `perf_latency_s = 0.013274`
- `cosine_similarity = 1.0`
- `min_cosine_similarity = 1.0`
- `max_abs_error = 0.0`

### 教训

1. CLIP image-embedding workload 的 bf16 NPU 数值对 batch 很敏感，批量一放大就容易出现离散绝对误差台阶
2. 这种模型不能只看 cosine；`max_abs_error` 往往才是 completed gate 的真正拦截点
3. 旧 step1/step2、旧文件命名、旧 `device_map=\"auto\"` 工件污染必须整轮清掉重跑，不能沿用旧 notes 修补
