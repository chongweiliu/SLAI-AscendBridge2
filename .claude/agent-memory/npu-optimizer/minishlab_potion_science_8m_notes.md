# minishlab/potion-science-8M

## 结论

- 路径：`runtime_only`
- 设备：`ASCEND_RT_VISIBLE_DEVICES=13`
- 数据集：`wikitext`
- speedup：`0.436104s -> 0.007537s`，`57.861749x`
- 精度：`cosine_similarity=1.0`，`max_abs_error=0.0`

## 复用模式

- 和 `potion-science-32M` 同族，直接复用：
  - 本地 snapshot `StaticModel.from_pretrained(str(snapshot_dir))`
  - baseline 单条循环 50 样本
  - perf `warmup(3x) + TASK_QUEUE_ENABLE=1 + batched_inference(batch_size=64)`
  - compare/notes 全量补齐 completed gate 必要字段

## 旧工件问题

- 旧 perf 工件文件名是 `benchmark_metrics_npu_fp32_pretrained_wikitext_perf.json`，缺少设备编号。
- 旧 perf 工件常见缺陷：
  - `num_samples=1`
  - 没有 `wall_clock_s`
  - 没有可靠 `output_compare`
  - notes 里仍是旧的负收益

## 处理原则

- 这类同族静态 embedding 模型不要做“修旧 notes”。
- 直接整轮重跑 baseline/perf/compare/gate，成本低、结果稳定。
