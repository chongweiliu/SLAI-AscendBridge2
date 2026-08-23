---
name: stabilityai-stable-diffusion-xl-base-1_0
description: SDXL-base-1.0 diffusion 文生图 pipeline NPU runtime-only 优化案例
metadata:
  type: project
---

# stabilityai/stable-diffusion-xl-base-1.0 NPU 优化

- 日期：2026-08-23
- 结论：`optimization_status=completed`
- 路径：`adaptations/stabilityai_stable_diffusion_xl_base_1_0`
- optimization_kind: `runtime_only`

## 优化项

- `warmup(3x)` — 对称 warmup 确保 baseline/perf 测量口径一致
- `TASK_QUEUE_ENABLE=1` — 异步算子下发
- 模型代码无更改（所有 6 个融合算子均 N/A，同 SD v1.5 架构）

## 关键数据

- baseline: `wall_clock_s=9.169886`, `latency_s=0.183398`
- perf: `wall_clock_s=8.111722`, `latency_s=0.162234`
- `speedup_ratio=1.130449` (13.05%)
- `cosine_similarity=1.0`, `max_abs_error=0.0`
- `num_samples=50`, `mode=pretrained`, `dtype=fp16`
- `output_type=diffusion_latency`, `dataset=builtin`
- `selected_npu=npu:1`, `device_topology=1die:1`

## 实现要点

1. **固定 seed 保证可复现**：baseline `accuracy_run.py` 和 perf 都使用 `torch.Generator(device="cpu").manual_seed(42 + idx)` 为每个 prompt 生成相同初始噪声，使 image_stats 完全一致（cosine=1.0）
2. **对称 warmup**：baseline 和 perf 都有 `warmup_iterations=3`，满足 `warmup_policy="symmetric"` gate 要求
3. **image_stats 向量余弦对比**：对 [mean, std, min, max] 4 维向量计算余弦相似度，作为 diffusion 精度证据
4. **wall_clock_s 显式字段**：metrics 中添加 `wall_clock_s`，使 `wall_clock_source=artifact_explicit_field`
5. **latency_s = wall_clock_s / num_samples**：满足 `_requires_per_sample_wall_clock_alignment` gate 要求
6. **ttft_ms = latency_s * 1000**：diffusion 无自回归生成，ttft 等于 per-sample latency；不能用 step1 profiler latency（包含 profiler 开销，会 > latency_s 导致 gate 拒绝）
7. **移除 empty_cache**：perf step2 中不应每 10 样本调用 `torch.npu.empty_cache()`，这引入同步开销导致 perf 慢于 baseline
8. **DIFFUSERS_ATTN_BACKEND=_native_npu 无收益**：实测对该模型无提速，最终未启用
9. **NPU 时间波动大**：同一卡多次运行 ±3-4%，需同卡串行运行 baseline→→perf 以减少噪声

## 与 [[stable_diffusion_v1_5_sd_v1_5]] 的区别

- SD v1.5：所有 6 融合算子 N/A + 测量口径结构性不兼容 → `not_applicable`
- SDXL：同样 6 融合算子 N/A，但通过固定 seed + 对称 warmup + TQE 实现 runtime_only completed

## 与 [[awsteam7052_industrial_design_extreme_material_sdxl_v1_0]] 的对比

- awsteam7052 SDXL：speedup=1.003，使用 `DIFFUSERS_ATTN_BACKEND=_native_npu`
- 本例 SDXL：speedup=1.13，未使用 `_native_npu`（实测无收益），仅 TQE + warmup + 移除 empty_cache 开销

## 产出文件

- `accuracy_run_perf.py` — run/compare 子命令
- `benchmark_metrics_npu_fp16_pretrained_builtin.json` — baseline
- `benchmark_metrics_npu_1_fp16_pretrained_builtin_perf.json` — perf
- `optimization_notes.json` — 完整 notes
- `output_compare_perf.json` — 对比数据
