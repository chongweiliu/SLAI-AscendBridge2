---
name: openai_clip_vit_large_patch14_notes
description: openai/clip-vit-large-patch14 completed：重建为本地 snapshot + CIFAR100 image_embeddings，最终 bs=1 + TASK_QUEUE_ENABLE=1 过 gate
type: reference
---

## openai/clip-vit-large-patch14 优化结果

### 结论：completed - runtime_only

- 旧 stage3 是过期 step1/step2 合同，带旧 `_perf` 命名、旧 `device_map=\"auto\"` 和脏工件
- 重新写成和 CLIP-B 同口径的 `image_embeddings` 合同：
  - adaptation 私有 snapshot
  - `local_files_only=True`
  - `cifar100`
  - baseline/perf 成对工件按时间戳配对 compare

### 关键发现

- `batch_size=8/4/2` 虽然 wall-clock 很好看，但都停在同一个误差平台：
  - `max_abs_error = 0.00390625`
- 最终 `batch_size=1 + TASK_QUEUE_ENABLE=1` 才真正过 gate

### 最终通过结果

- 卡：`ASCEND_RT_VISIBLE_DEVICES=12`
- `optimization_kind = runtime_only`
- `speedup_ratio = 1.459697`
- `baseline_wall_clock_s = 1.26784`
- `perf_wall_clock_s = 0.868564`
- `perf_latency_s = 0.017371`
- `cosine_similarity = 1.0`
- `min_cosine_similarity = 1.0`
- `max_abs_error = 0.0`

### 教训

1. CLIP-L 和 CLIP-B 在 bf16 image-embedding 合同下都呈现“batched 误差平台”，不要被高倍 wall-clock 先迷惑
2. compare 不能再按文件 mtime 选工件；会被 compare 回写 metrics 的副作用污染，必须按工件内事件时间配对
3. 这类视觉 embedding 模型优先尝试 `bs=1 + TQE` 作为最后的 gate-safe runtime-only 路径
