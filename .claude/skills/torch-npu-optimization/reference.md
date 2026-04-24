# torch_npu 推理优化 API 速查

## 核心融合算子（已验证）

| API | 用途 | 关键要点 |
|-----|------|----------|
| `torch_npu.npu_rms_norm(x, weight, epsilon=eps)` | 替代手写 RMSNorm | 返回 tuple，取 `[0]` |
| `torch_npu.npu_swiglu(x, dim=-1)` | SwiGLU: SiLU(first_half) * second_half | 注意 concat 顺序 |
| `torch_npu.npu_rotary_mul(x, cos, sin)` | 旋转位置编码 | cos/sin 需 expand_as(x) |
| `torch_npu.npu_fusion_attention(q, k, v, num_heads, ...)` | 融合 attention | 必须显式 causal mask |

## npu_fusion_attention 参数详解

```python
npu_out = torch_npu.npu_fusion_attention(
    query,              # (B, S, N, D) when input_layout="BSND"
    key,                # (B, S, N, D)
    value,              # (B, S, N, D)
    num_heads,          # int
    input_layout="BSND",  # B=batch, S=seq, N=num_heads, D=head_dim
    pse=None,           # position encoding bias (optional)
    atten_mask=mask,    # bool tensor, True=mask out, False=attend
    scale=1/sqrt(D),    # attention scale factor
    pre_tockens=65536,  # sliding window: attend to N previous tokens
    next_tockens=0,     # sliding window: attend to 0 future tokens (causal)
    keep_prob=1.0,      # 1.0 - dropout_p (inference: 1.0)
)[0]                    # returns tuple, take [0]
```

### causal mask 构建模式

```python
# 基础 causal mask: 上三角为 True (屏蔽未来 token)
causal_mask = torch.triu(
    torch.ones(seq_q, seq_k, dtype=torch.bool, device=device),
    diagonal=seq_k - seq_q + 1,
)

# prefill (seq_q == seq_k == T): diagonal=1, 上三角屏蔽
# decode  (seq_q == 1):          diagonal=T, 全零不屏蔽

# 合并 padding mask
if attention_mask is not None:
    pad_mask = (attention_mask.squeeze(1).squeeze(1) < -1.0)  # float->bool
    causal_mask = causal_mask.unsqueeze(0) | pad_mask.unsqueeze(-2)  # (B, S_q, S_kv)
    atten_mask = causal_mask.unsqueeze(1)  # (B, 1, S_q, S_kv)
else:
    atten_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S_q, S_kv)
```

## npu_swiglu concat 顺序参考

```
npu_swiglu(x, dim=-1) = SiLU(x[..., :half]) * x[..., half:]

原始代码            concat 顺序
-----------          ------------
a1 * silu(a2)   -->  [a2, a1]     # SiLU 作用于前半 (a2)
silu(a1) * a2   -->  [a1, a2]     # SiLU 作用于前半 (a1)
```

## 环境变量

| 变量 | 用途 |
|------|------|
| `ASCEND_RT_VISIBLE_DEVICES={selected_npu}` | 限制单卡，避免多卡 device_map 问题；运行前先用 `npu-smi info` 选择空闲或低占用卡，不要默认 0 号卡 |

## 精度验收基准

**精度验收必须使用 pretrained 权重**，baseline（accuracy_run.py）与 perf（accuracy_run_perf.py）均需 `--use-pretrained`。

| 指标 | 可接受范围 | 说明 |
|------|------------|------|
| Logits 余弦相似度 | > 0.99 | 单步 forward 精度 |
| PPL 相对差异 | < 15% | 全序列累积误差 |
| 文本匹配率 | 不作为主要指标 | bf16 自回归解码发散是正常现象 |

## GQA (Grouped Query Attention) 支持

`npu_fusion_attention` 原生支持 GQA。直接传入不同 head 数的 q/k/v 即可，无需 `repeat_interleave`。

```python
# query: (B, S, num_heads, D)
# key/value: (B, S, num_kv_heads, D)  ← num_kv_heads < num_heads
npu_fusion_attention(query, key, value, num_heads, input_layout="BSND", ...)
```

## 环境变量调优

| 变量 | 值 | 作用 |
|------|------|------|
| `ASCEND_RT_VISIBLE_DEVICES={selected_npu}` | 0,1,... | 限制可见 NPU 卡；应先用 `npu-smi info` 确认所选卡当前空闲或低占用 |
| `TASK_QUEUE_ENABLE=1` | 0/1 | 异步算子下发，推理 +5~15% |
| `ASCEND_LAUNCH_BLOCKING=1` | 0/1 | 同步执行，仅用于调试 |
| `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:512` | MB | 减少显存碎片化 |

## 其他有用 API（按适用性分级）

### ⭐⭐⭐ 高适用性

| API | 用途 | 适用场景 | 约束 |
|-----|------|---------|------|
| `torch_npu.npu_add_layer_norm(x, residual, gamma, beta, eps)` | 融合 residual add + LayerNorm | Post-norm 架构（BERT/CamemBERT/ViT） | 需要 add + LN 在同一方法内 |
| `torch_npu.npu_rms_norm(x, weight, epsilon=eps)` | 替代手写 RMSNorm | 使用 RMSNorm 的模型（LLaMA/Qwen） | 返回 tuple，取 `[0]` |
| `torch_npu.npu_swiglu(x, dim=-1)` | SwiGLU: SiLU(first_half)*second_half | SwiGLU 门控 FFN | 注意 concat 顺序 |
| `torch_npu.npu_rotary_mul(x, cos, sin)` | 旋转位置编码 | RoPE 模型（Qwen/ModernBERT） | cos/sin 需 expand_as(x) |
| `torch_npu.npu_fusion_attention(q, k, v, num_heads, ...)` | 融合 attention | decoder-only fp16/bf16（Qwen/OPT） | fp32 encoder cosine~0.8，不可用 |
| `torch_npu.npu_ffn(x, w1, w2, activation, ...)` | 融合 FFN（matmul+act+matmul） | fp16/bf16 模型的 FFN 层 | **仅 fp16/bf16**，需提取 weight |
| `torch.npu.synchronize()` | 同步 NPU 计时 | 性能测试时必须调用 | — |

### ⭐⭐ 中等适用性

| API | 用途 | 适用场景 | 约束 |
|-----|------|---------|------|
| `torch_npu.npu_gelu(x, approximate='none')` | NPU 原生 GELU | GELU 激活模型（BERT/CamemBERT） | NPU 原生 gelu 默认 tanh 近似，此 API 可强制 erf 模式（精度修正） |
| `torch_npu.npu_gelu_mul(x, approximate='none')` | 融合 GELU + 逐元素乘 | GeGLU 门控 FFN（ModernBERT） | 末维 ≤ 1024 且为偶数 |
| `torch_npu.npu_prompt_flash_attention(q, k, v, ...)` | 全量 Flash Attention（标准 softmax） | fp16/bf16 attention（替代 npu_fusion_attention） | **仅 fp16/bf16**，fp32 不支持 |

### ⭐ 低适用性 / 特殊场景

| API | 用途 | 不适用的原因 |
|-----|------|-------------|
| `torch_npu.npu_scaled_masked_softmax(x, mask, scale)` | 融合 scale+mask+softmax | SDPA 已优化；eager path 较少使用；H/W 须 ≥32 且整除 32 |
| `torch_npu.npu_transpose_batchmatmul(x, w, ...)` | 融合 transpose + matmul | 仅 3D tensor；K/N 须整除 128 |
| `torch_npu.contrib.function.fuse_add_softmax_dropout(...)` | 融合 add+softmax+dropout | 仅训练有收益（dropout>0），推理 dropout=0 无意义 |
| `torch_npu.npu_interleave_rope(x, cos, sin)` | interleave 布局 RoPE | D 必须等于 64，约束太严 |
| `torch_npu.npu_fused_infer_attention_score(...)` | 增量+全量 FA | 面向 KV Cache 增量推理，encoder 不适用 |

### ❌ 已废弃 / 不适用

| API | 状态 | 替代方案 |
|-----|------|---------|
| `torch_npu.npu_silu(x)` | 计划废弃 | 用 `F.silu(x)` |
| `torch_npu.npu_dtype_cast(x, dtype)` | 计划废弃 | 用 `x.to(dtype)` |
| `torch_npu.npu_layer_norm_eval(x, w, b, eps)` | 报错 500001 | 用 `npu_add_layer_norm` 融合 residual+LN |
| `torch_npu.contrib.npu_fused_attention(...)` | 已废弃 | 用 `npu_fusion_attention` |
| `torch_npu.contrib.npu_fused_attention_with_layernorm(...)` | 已废弃 | — |
| 所有量化 API (`npu_quant_*`, `npu_dynamic_quant_*` 等) | 不在推理优化范围 | INT8/INT4 量化需专项优化 |
| 所有 MoE API (`npu_moe_*`) | MoE 专用 | 小模型不使用 MoE |

### 精度类型限制汇总（关键）

许多 torch_npu 融合算子仅支持 fp16/bf16，**fp32 推理模型可用算子极其有限**：

| API | fp16 | bf16 | fp32 |
|-----|:----:|:----:|:----:|
| `npu_add_layer_norm` | ✅ | ✅ | ✅ |
| `npu_rms_norm` | ✅ | ✅ | ✅ |
| `npu_rotary_mul` | ✅ | ✅ | ✅ |
| `npu_fusion_attention` | ✅ 精度好 | ✅ 精度好 | ❌ cosine~0.8 |
| `npu_ffn` | ✅ | ✅ | ❌ |
| `npu_prompt_flash_attention` | ✅ | ✅ | ❌ |
| `npu_swiglu` | ✅ | ✅ | ✅ |
| `npu_gelu` | ✅ | ✅ | ✅ |
| `npu_gelu_mul` | ✅ | ✅ | ✅ |
| `npu_scaled_masked_softmax` | ✅ | ✅ | ✅ |

## 参考文档（refer 目录）

本 skill 的 `refer/` 目录包含 torch_npu 官方 API 文档的本地副本，便于离线查阅：

| 文档 | 说明 |
|------|------|
| [refer/torch_npu_list.md](refer/torch_npu_list.md) | torch_npu 接口列表（创建 tensor、计算类操作） |
| [refer/torch_npu-contrib_list.md](refer/torch_npu-contrib_list.md) | torch_npu.contrib 亲和库接口列表（组合类接口） |

核心融合算子对应的详细 API 文档：

| 速查表中的 API | refer 中的文档 |
|----------------|----------------|
| `npu_rms_norm` | [（beta）torch_npu-npu_rms_norm.md](refer/（beta）torch_npu-npu_rms_norm.md) |
| `npu_swiglu` | [（beta）torch_npu-npu_swiglu.md](refer/（beta）torch_npu-npu_swiglu.md) |
| `npu_rotary_mul` | [torch_npu-npu_rotary_mul.md](refer/torch_npu-npu_rotary_mul.md) |
| `npu_fusion_attention` | [torch_npu-npu_fusion_attention.md](refer/torch_npu-npu_fusion_attention.md) |
| `npu_add_layer_norm` | 本地 refer 快照未单独收录；先查 [refer/torch_npu_list.md](refer/torch_npu_list.md) 与 [（beta）torch_npu-npu_layer_norm_eval.md](refer/（beta）torch_npu-npu_layer_norm_eval.md) |
| `npu_ffn` | [torch_npu-npu_ffn.md](refer/torch_npu-npu_ffn.md) |
| `npu_gelu` | [torch_npu-npu_gelu.md](refer/torch_npu-npu_gelu.md) |
| `npu_gelu_mul` | [torch_npu-npu_gelu_mul.md](refer/torch_npu-npu_gelu_mul.md) |
| `npu_prompt_flash_attention` | [torch_npu-npu_prompt_flash_attention.md](refer/torch_npu-npu_prompt_flash_attention.md) |
| `npu_scaled_masked_softmax` | [torch_npu-npu_scaled_masked_softmax.md](refer/torch_npu-npu_scaled_masked_softmax.md) |

## 优化瓶颈与突破方向

当前已优化的 5 个模型（BERT-small, CamemBERT-base, ModernBERT-base, OPT-125m, DINO ViT-B/16）的主要可用 API 已基本用尽。进一步优化方向：

1. **半精度推理**：BERT/CamemBERT/ViT 改为 fp16/bf16 推理后，可启用 `npu_ffn`、`npu_prompt_flash_attention` 等 fp16-only 算子，收益显著增加
2. **环境变量加速**：`TASK_QUEUE_ENABLE=1` 异步算子下发，无需改代码，额外 +5~15%
3. **更大模型**：7B+ 模型算子融合的收益更明显；小模型（<1B）融合收益被固定开销稀释

## 已知问题补充

- **NPU 上 GELU approximate 参数无效**：`F.gelu(x, approximate='none')` 在 NPU 上仍走 tanh 近似；需精确 erf 模式时用 `torch_npu.npu_gelu(x, approximate='none')`
- **npu_layer_norm_eval 已弃用**：报 SetPrecisionMode 500001 错误，改用 `npu_add_layer_norm`
