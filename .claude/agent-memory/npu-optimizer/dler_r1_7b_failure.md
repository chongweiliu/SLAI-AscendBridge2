---
name: dler_r1_7b_failure
description: nvidia/DLER-R1-7B-Research bf16 28层模型所有优化均失败
type: reference
---

# nvidia/DLER-R1-7B-Research 优化失败

## 模型规格
- Qwen2 架构，28 层，28 heads，4 KV heads，hidden=3584，intermediate=18944
- bf16 精度， wikitext 数据集

## 失败根因

### bf16 精度灾难
28 层 bf16 模型的融合算子替换导致精度灾难：
- `npu_rms_norm`: speedup_ratio=0.5413x 但 cosine=0.6386（PPL 回归 94%）
- `npu_swiglu`: speedup_ratio=0.4714x 但 text_match=5/50（灾难性精度损失）
- `npu_rotary_mul`: API 签名错误（`npu_rotary_mul(input, r1, r2, rotary_mode="half")` 需要 3 个 tensor + 1 个 kwarg，不是 4 个 positional tensor）
- `npu_fusion_attention`: 内部 rotary 调用同样 API 签名错误
- TASK_QUEUE_ENABLE=1: 26% wall-clock 回归 + cosine=0.6386 精度损失

### 经验总结
1. bf16 对 28 层深度模型的融合算子精度极其敏感
2. `npu_rotary_mul` API 签名：`torch_npu.npu_rotary_mul(input, r1, r2, rotary_mode="half")` — 3 tensors + 1 string kwarg，不是 4 positional tensors
3. Qwen2 的 attention 通过 `attention_interface` dispatcher 分发，直接 patch `Qwen2Attention.forward` 对 transformer 4.x+ 架构无效

## optimization_notes.json 最终状态
- `speedup_ratio`: 0.7374x（回归，26% 更慢）
- `cosine_similarity`: 0.6386（远低于 0.99 阈值）
- `reason_code`: fusion_and_runtime_regression_bf16
- `skipped_optimizations`: npu_rms_norm, npu_swiglu, npu_rotary_mul, npu_fusion_attention, TASK_QUEUE_ENABLE
