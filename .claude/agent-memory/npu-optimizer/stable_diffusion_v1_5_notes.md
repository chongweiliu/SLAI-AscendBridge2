---
name: stable_diffusion_v1_5_sd_v1_5
description: SD v1.5 diffusion 模型：所有 6 个融合算子均 N/A + 测量口径结构性不兼容 completion gate
type: project
---

# Stable Diffusion v1.5 优化结论：not_applicable

## 根因

1. **架构不兼容所有融合算子**：UNet2DConditionModel 使用 pre-norm GroupNorm + SiLU + processor-based 2D attention
   - `npu_add_layer_norm`: N/A (pre-norm vs post-norm 架构不匹配)
   - `npu_gelu`: N/A (SD 使用 SiLU，不是 GELU)
   - `npu_rms_norm`: N/A (SD 使用 GroupNorm/LayerNorm，不是 RMSNorm)
   - `npu_rotary_mul`: N/A (SD 无 RoPE)
   - `npu_fusion_attention`: N/A (SD 用 2D attention processor，布局是 BLC 不是 BHS)
   - `npu_swiglu`: N/A (SD 用 GEGLU，不是 SwiGLU)

2. **测量口径结构性不兼容 completion gate**：
   - `accuracy_run.py` step1 只测 1 个冷样本（无 warmup）
   - `accuracy_run_perf.py` step2 测 50 个暖样本（有 3x warmup）
   - speedup_ratio = baseline_wall_clock / perf_wall_clock = (1 × cold_latency) / (50 × warm_latency)
   - 结果永远 < 1.0（即使 warmup 有 1.63x 提速，也无法覆盖 N=50 的分母）
   - completion gate 要求 speedup_ratio >= 1.0，结构性无法满足

## 关键数据

- 冷启动延迟：1.227s
- 暖启动延迟（per-sample）：0.752s
- 50 样本总耗时：37.58s
- speedup_ratio = 6.86 / 37.58 = 0.1825 < 1.0
- 每样本 warmup 提速：1.63x（但 wall-clock 分母是 50，无法满足）

## 结论

- `result=not_applicable`
- `failure_reason: measurement_methodology_incompatible`
- `retryable: false`
- `recommended_action: none_possible`
- 所有 6 个融合算子均 N/A，且测量口径结构性不兼容 diffusion 模型的 completion gate

## 教训

- Diffusion 模型（SD、FLUX、Wan 等）的 pipeline 测量口径与 LLM 不同
- `speedup_ratio = baseline_wall_clock / perf_wall_clock` 在 diffusion 模型上会因为分母是 N×warm_latency 而永远 < 1.0
- 对于 diffusion 模型，需要重新定义 completion gate 逻辑或使用 per-sample latency speedup 而非 wall-clock speedup
