# dice-research/lola_v1 优化记录

## 模型信息
- **类型**: causal_lm (GPT-2 MoE, 24L, 16 experts)
- **精度**: fp32
- **数据集**: wikitext
- **架构**: Pre-LN + nn.LayerNorm + gelu_fast + learned positional embeddings

## transformers 版本问题
- 原始 error: `transformers 5.x` 中 `transformers.utils.model_parallel_utils` 已移除
- 解决方案: 在 adaptation .venv 中 `uv pip install transformers==4.47.1`
- shim modules 在 `model_files/` 中创建但降级后不再需要

## 优化结果
| 指标 | 值 |
|------|-----|
| perf cold latency | 0.401519s |
| perf warm latency | 0.098361s |
| **steady-state speedup** | **4.08x** |
| baseline step1 latency (profiler) | 1.335602s |
| cosine | 1.0 |
| PPL diff | 0% |
| text match | 100% (50/50) |
| wall-clock speedup | **0.983x (< 1.0)** |

## completion gate 失败原因
- speedup_ratio 必须 = baseline_wall_clock_s / perf_wall_clock_s = 164.74 / 167.52 = **0.983** (< 1.0)
- warmup_policy 必须为 "symmetric"
- baseline_warmup_iterations 必须等于 perf_warmup_iterations
- **核心矛盾**: perf 的 3 个额外 warmup 迭代在 step1 中增加 ~2.8s 开销，使 perf wall-clock 比 baseline 更慢
- 即使 symmetric warmup (两边都用 warmup=3)，perf 仍然更慢（perf 多了 3 个 warmup 迭代的开销）

## 关键教训
1. **warmup 开销计入 wall-clock**: 对于多样本测试，warmup 迭代的开销被 step2 的 50 样本均摊，但 step1 本身的 warmup 迭代是纯开销
2. **speedup_ratio 定义**: 必须 = wall_clock_baseline / wall_clock_perf，不能用 latency speedup 替代
3. **Pre-LN + nn.LayerNorm 架构**: 所有 6 大融合算子均不适用（见 skipped_optimizations）
4. **transformers 4.x vs 5.x**: LOLA 这类 trust_remote_code=True 的自定义模型，降级到 4.47.1 是有效的

## skipped_optimizations
- npu_rms_norm: N/A - nn.LayerNorm 不是 RMSNorm
- npu_swiglu: N/A - gelu_fast 不是 SiLU/gated
- npu_rotary_mul: N/A - learned positional embeddings 不是 RoPE
- npu_fusion_attention: N/A - 标准 attention + learned pos embed
- npu_add_layer_norm: N/A - Pre-LN，residual add 与 LN 分离
- npu_gelu: N/A - gelu_fast 已优化，npu_gelu(erf) 会导致精度不匹配