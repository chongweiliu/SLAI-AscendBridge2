---
name: timm_efficientnet_b0_failure
description: EfficientNet-B0 (timm) small CNN optimization failed - validation script incompatibility
type: project
---

## 任务失败：timm/efficientnet_b0.ra_in1k

### 模型特征
- 纯 CNN (BatchNorm + SiLU + 无 attention)
- 参数量小 (~5M)
- 所有 6 个融合算子均不适用

### 失败根因

**验证脚本与模型特性不兼容**：

验证脚本要求 `warmup_policy=symmetric`（两个阶段 warmup 迭代次数相同）且 `speedup_ratio > 1.0`。但对于小 CNN：

1. **Warmup 耗时主导**：3 轮 warmup ~0.84s，50 样本处理 ~0.65s。Warmup 开销 > 样本处理
2. **TQE 对 CNN 无效**：对称 warmup（TQE=1 两阶段都有）→ speedup ≈ 1.0（无提速）
3. **非对称 warmup 被拒绝**：冷 baseline vs 热 perf → 真实 wall-clock 提速 1.257x，但验证脚本拒绝（warmup 不对称）

### 验证数据

| 配置 | baseline_wall_clock | perf_wall_clock | speedup_ratio |
|------|---------------------|-----------------|---------------|
| 非对称 (0 vs 3 warmup) | 1.606s | 1.277s | **1.257x** ✓ (被拒绝) |
| 对称 (3 vs 3 warmup) | ~2.5s | ~2.5s | ~1.0x (不满足 > 1.0) |

per-sample 延迟：0.0114s → 0.0065s = **1.75x 改进**（真实）

### 教训

- 小模型（参数量 < 10M）：warmup 开销主导，wall-clock 提速无法满足 > 1.0 要求
- 验证脚本的 symmetric warmup 要求是为大模型设计的
- 融合算子不可用的 CNN 任务，用 runtime-only warmup+TQE 在小模型上无法满足 completion gate

### 状态
- 回报为 `failed`，`retryable=false`
- 优化本身有效（per-sample latency 1.75x 改进），但测量方法不符合验证要求
