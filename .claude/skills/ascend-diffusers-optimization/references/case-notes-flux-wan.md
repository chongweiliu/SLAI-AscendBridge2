# FLUX And Wan Case Notes

## FLUX 类模型

常见特征：

- attention 是 non-causal
- 某些 `SwiGLU` 路径可直接映射到 `npu_swiglu`
- 单卡时小算子融合更容易体现收益
- 多卡时通信开销会明显吞噬端到端收益

经验：

- `RMSNorm`、`GELU`、`SwiGLU` 值得优先检查
- `_native_npu` attention backend 通常比手搓替换更稳
- 若模型需要 dispatch，多卡通信常是上限

## Wan 类模型

常见特征：

- diffusers pipeline，但主干是 3D transformer
- `GELU` 和 `RMSNorm` 常比 `SwiGLU` 更有代表性
- 自定义 3D RoPE 较复杂，通常不适合直接套通用 `npu_rotary_mul`

经验：

- 优先确认哪些融合算子根本不适用
- 不要把 2D 图像模型的 attention / rotary 经验机械照搬到视频模型

## 结论

FLUX / Wan 这类大 diffusers 模型里，性能分析要同时看三层：

- 算子层是否有可替换热点
- backend 与环境变量是否已开到位
- 端到端瓶颈是否其实在 offload 或跨卡通信
