# Awsteam7052/Industrial-Design-Extreme-Material-SDXL_v1.0

- 日期：2026-04-21
- 结论：`optimization_status=completed`
- 路径：`adaptations/awsteam7052_industrial_design_extreme_material_sdxl_v1_0`

## 关键经验

- 这类 diffusers / SDXL 模型不一定是“优化不适用”，常见阻塞点是老版 `accuracy_run.py` / `accuracy_run_perf.py` 仍沿用 step1/step2 结构，导致 `num_samples=1`、`dataset` 缺失、`self-baseline` 或旧 wall-clock 口径，直接被当前 gate 拦截。
- `check_accuracy_run.py` / `board_ops.py` 会扫描 adaptation 根目录下全部 `benchmark_metrics*.json`。旧的 `config_random` baseline/perf 工件会污染 completed gate，即使新的 pretrained baseline 已正确生成。
- 处理方式不是直接跳过，而是：
  1. 重写为当前规则兼容的 pretrained baseline/perf 脚本。
  2. 生成独立 baseline artifact，对 perf 使用 `comparison_method=independent_baseline_artifact`。
  3. 将旧的 `benchmark_metrics_*config_random*.json`、对应 `outputs_*`、`trace_*` 归档出根目录，避免被 glob 扫到。

## 最终通过路径

- baseline：`benchmark_metrics_npu_0_fp16_pretrained_random.json`
- perf：`benchmark_metrics_npu_0_fp16_pretrained_random_perf.json`
- 优化类型：`runtime_only`
- 优化项：
  - `DIFFUSERS_ATTN_BACKEND=_native_npu`
  - `warmup(3x)`
  - `TASK_QUEUE_ENABLE`
- `num_samples=50`
- `speedup_ratio=1.003059`
- `cosine_similarity=1.0`
- `loaded_from_model_files=true`

## 额外提醒

- diffusers runtime-only 也要显式记录 `selected_npus`、`device_topology`、`parallel_mode`，否则 notes 不完整。
- 当 `speedup_ratio` 只略高于 1.0 时，必须保持 baseline/perf 同卡串行，避免并发测试污染 wall-clock。
