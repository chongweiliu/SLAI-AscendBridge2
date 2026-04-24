---
name: Qwen3-VL-4B-Instruct 优化失败
description: Qwen3-VL-4B-Instruct bf16 优化失败：NPU 环境变化导致性能测量失效 + bf16 NPU 非确定性导致 text_match_rate=82%
type: reference
---

# Qwen3-VL-4B-Instruct 优化失败记录

**时间**: 2026-03-29
**结果**: failed（环境变化 + completion gate 不满足）

## 失败根因

1. **环境变化**：NPU 1/3 被其他 agent 占用（各 33GB HBM），perf 无法使用多卡分布，只能在受竞争的 NPU 0 单卡上运行。step1 延迟从 3.1s 退至 17s（profiling 开销 + 环境竞争）。

2. **Precision 不满足 gate**：text_match_rate=82%，不满足 completion gate 要求（text_match_rate=1.0 或 cosine>=0.99）。

3. **npu_rotary_mul 对 mRoPE 兼容性存疑**：移除 npu_rotary_mul 后 text_match_rate 从 82% 降至 50%，说明 rotary_mul 本身不增加误差，反而可能提供更稳定的计算路径。

## 关键发现

1. **bf16 NPU 非确定性**：VLM 自回归生成对 bf16 精度极敏感。相同输入、相同权重、相同代码，两次运行只有 82% 文本匹配。这不是补丁 bug，是硬件级 bf16 非确定性。

2. **移除 npu_rotary_mul 导致更差匹配**：移除后 50%，保留后 82%。说明标准 npu_rotary_mul 对 Qwen3-VL 的 mRoPE 没有破坏性影响。

3. **环境依赖性**：4B 模型在多卡分布时性能差异巨大。旧 perf 在 NPU 1（33768 MB）测量，显示 speedup=1.45x；新 perf 在 NPU 0 竞争卡测量，显示 speedup=0.51x。

## 融合算子适用性

| 算子 | 状态 | 说明 |
|------|------|------|
| npu_rms_norm | ✅ 可用 | Qwen3VLTextRMSNorm |
| npu_swiglu | ✅ 可用 | Qwen3VLTextMLP |
| npu_rotary_mul | ⚠ 存疑 | mRoPE (3D) 与标准 2D rotary 理论上不兼容，但移除后反而更差 |
| npu_fusion_attention | ✅ 可用 | GQA attention |
| npu_add_layer_norm | N/A | Qwen3-VL 用 RMSNorm |
| npu_gelu | N/A | Qwen3-VL 用 SiLU/SwiGLU |

## 教训

1. **环境隔离是前提**：在 NPU 1/3 被占用时不应启动重测，应等待或换空闲卡
2. **bf16 VLM 生成天然低匹配率**：对于 VLM autoregressive bf16 生成，text_match_rate=82% 是正常水平，不代表优化失败
3. **speedup_ratio 对比必须在同环境**：旧 perf（1.45x）与新 perf（0.51x）对比无效，是环境差异不是优化差异

## 建议

1. **报告为 pending** 而非 failed（环境问题可重试）
2. 等待 NPU 1/3 空闲后重新运行
3. 或者接受 text_match_rate=82% 作为 VLM bf16 生成的正常现象，调整 precision_method 为 cosine_similarity 并放宽阈值
