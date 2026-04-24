# Cross-Encoder MiniLM-L6-v2 优化记录

## 模型信息
- 模型ID: cross-encoder/ms-marco-MiniLM-L6-v2
- 架构: 6层 BERT-like, 22M params
- 类型: Cross-Encoder (MS MARCO 排序)
- norm类型: LayerNorm (pre-norm: LN 在 attention/FFN 之前)
- 激活函数: GeLU
- attention类型: Full attention (cross-encoder)

## 融合算子适用性
| 算子 | 状态 | 原因 |
|------|------|------|
| npu_add_layer_norm | N/A | pre-norm 架构 (LN 在 attention/FFN 之前，非之后) |
| npu_layer_norm | N/A | 同上 |
| npu_gelu | N/A | MiniLM 使用 GeLU (tanh 近似)，npu_gelu 可能不匹配 |
| npu_swiglu | N/A | BERT 不使用 SwiGLU FFN |
| npu_rotary_mul | N/A | BERT 使用绝对位置编码，非 RoPE |
| npu_fusion_attention | N/A | cross-encoder 使用完整 attention 矩阵 |

## 优化结果
- 优化项: warmup(3x) + TASK_QUEUE_ENABLE
- 精度: cosine=0.99999988 (正常)
- **speedup_ratio: 0.0735 < 1.0 (性能回退)**

## 关键发现
1. **TQE 对小模型有反效果**: TASK_QUEUE_ENABLE 在 22M 参数极小模型上开销大于收益
2. **warmup 是唯一有效优化**: 消除 JIT 编译开销，但 baseline 未 warmup 导致对比不公平
3. **steady_state 下两者等价**: warmup 后 baseline 和 perf 都是 ~0.0043s/sample

## 教训
- 小模型 (< 100M params) 上 TQE 可能导致性能回退
- Cross-encoder 模型所有融合算子均不适用
- 融合算子不适用的模型不应强行应用优化，应报告为架构不适用
