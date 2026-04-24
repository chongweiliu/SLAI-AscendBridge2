# minishlab/potion-science-32M

## 结论

- 路径：`runtime_only`
- 设备：`ASCEND_RT_VISIBLE_DEVICES=13`
- 数据集：`wikitext`
- speedup：`0.432990s -> 0.007390s`，`58.59134x`
- 精度：`cosine_similarity=1.0`，`max_abs_error=0.0`

## 有效做法

- 只从 adaptation 私有 snapshot 加载：
  - `models/models--minishlab--potion-science-32M/refs/main`
  - `models/models--minishlab--potion-science-32M/snapshots/<ref>`
- `model2vec.StaticModel.from_pretrained(str(snapshot_dir))` 可直接吃本地 snapshot 路径。
- baseline 不要沿用旧的“单样本 step1 + 全量 step2 但不回写 metrics”。
- 稳定 completed 方案：
  - baseline：逐条 `encode([text])` 循环 50 样本
  - perf：`warmup(3x) + TASK_QUEUE_ENABLE=1 + batched_inference(batch_size=64)`

## 踩坑

- 旧 baseline 工件只有单样本 profile 口径，旧 perf 工件缺少：
  - `mode`
  - `num_samples`
  - `wall_clock_s`
  - `output_compare`
- 旧 `optimization_notes.json` 是 `speedup_ratio=0.61` 的脏结果，不能复用。
- `ttft_ms` 仍要避免两位小数进位，统一保留到 3 位。

## 当前可复用模式

- 对同族 `Model2Vec` 静态 embedding 模型：
  - 本地 snapshot 加载
  - baseline 单条循环
  - perf 大 batch
  - compare 时显式写：
    - `baseline_samples`
    - `perf_samples`
    - `cuda_samples`
    - `ascend_samples`
  - notes 里显式写：
    - `comparison_method=independent_baseline_artifact`
    - `comparison_scope=steady_state`
    - `selected_npus`
    - `device_topology`
    - `parallel_mode`
