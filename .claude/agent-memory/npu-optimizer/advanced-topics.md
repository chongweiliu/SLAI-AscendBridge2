# NPU 优化进阶知识

## 1. GQA (Grouped Query Attention) 的 npu_fusion_attention 处理

现代大模型（LLaMA-2/3, Qwen-2, Mistral）大量使用 GQA，即 num_kv_heads < num_heads。

**关键**：`npu_fusion_attention` 原生支持 GQA，不需要手动 repeat_kv。

```python
# 错误做法：手动 expand kv_heads 到 num_heads
key = key.repeat_interleave(num_groups, dim=2)  # 浪费内存
value = value.repeat_interleave(num_groups, dim=2)

# 正确做法：直接传入不同 head 数的 q/k/v
# query: (B, S, num_heads, D), key/value: (B, S, num_kv_heads, D)
npu_out = torch_npu.npu_fusion_attention(
    query, key, value,
    num_heads,           # 传 query 的 head 数
    input_layout="BSND",
    ...
)[0]
```

`npu_fusion_attention` 内部自动处理 GQA 的 head 映射，比手动 repeat 更快且省内存。

## 2. npu_fusion_attention 的 sparse_mode 参数

默认 `sparse_mode=0` 即 defaultMask（全量 attention mask）。其他常用模式：

| sparse_mode | 含义 | 适用场景 |
|-------------|------|----------|
| 0 | defaultMask | 通用，需传 atten_mask |
| 1 | allMask | 无 mask（不推荐，精度问题） |
| 2 | leftUpCausal | 左上角因果 mask |
| 3 | rightDownCausal | 右下角因果 mask（标准 causal） |
| 4 | band | 带状 sparse attention |
| 5 | prefix | prefix-LM（前缀全 attend，后续 causal） |

**实测发现**：`sparse_mode=0` + 显式 bool mask 最稳定可靠。`sparse_mode=3` 理论上可以省掉手动构建 causal mask，但需要仔细验证 diagonal 行为是否与模型一致。建议先用 mode=0 跑通，再尝试 mode=3 看是否有额外性能收益。

## 3. 环境变量调优

| 变量 | 值 | 作用 |
|------|------|------|
| `ASCEND_RT_VISIBLE_DEVICES={selected_npu}` | 0,1,... | 限制可见 NPU 卡；运行前先用 `npu-smi info` 确认该卡空闲或低占用，避免默认抢 0 号卡 |
| `TASK_QUEUE_ENABLE=1` | 0/1 | 启用任务队列，异步下发算子，减少 Host 等待 |
| `ASCEND_LAUNCH_BLOCKING=1` | 0/1 | 同步执行，仅用于调试定位算子错误 |
| `HCCL_BUFFSIZE=120` | MB | HCCL 通信缓冲区大小，多卡通信调优 |
| `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:512` | MB | 显存分配策略，减少碎片化 |

**重要**：`TASK_QUEUE_ENABLE=1` 在推理场景通常能带来 5~15% 额外提升，因为算子异步下发减少了 Host-Device 同步等待。但在调试精度问题时应关闭。

## 4. 算子 fallback 检测

当某个 PyTorch 算子在 NPU 上没有对应实现时，会自动 fallback 到 CPU 执行（涉及 D2H + H2D 数据搬运），严重拖慢性能。

### 检测方法

```python
# 方法 1：环境变量开启 fallback 日志
# ASCEND_GLOBAL_LOG_LEVEL=1 会打印 warning 级别日志，其中包含 fallback 信息

# 方法 2：用 profiler 检测
import torch_npu
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.NPU],
    record_shapes=True,
) as prof:
    model(inputs)
# 查看 trace 中是否有 aten:: 前缀的算子在 CPU 上执行
print(prof.key_averages().table(sort_by="self_npu_time_total"))
```

### 常见 fallback 算子及解决

| fallback 算子 | 原因 | 解决 |
|--------------|------|------|
| `aten::_unique2` | NPU 不支持 | 改用等价实现或前置到 CPU |
| `aten::nonzero` | 动态 shape | 尽量避免，或接受 fallback |
| `aten::index_put_` | 部分场景不支持 | 用 scatter_ 替代 |
| `aten::multinomial` | 采样算子 | generation 时用 do_sample=False 绕过 |

## 5. torch.compile + NPU

**现状（2024-2025）**：`torch.compile` 对 NPU 的支持仍在演进中。

```python
# torchair 是华为提供的 NPU compile 后端
import torchair
config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)
model = torch.compile(model, backend=npu_backend)
```

**实测建议**：
- 对于 LLM 推理，手动算子替换（本 skill 的方式）目前比 torch.compile 更可控
- torch.compile 在 NPU 上可能遇到 dynamic shape 不支持、graph break 等问题
- 可以两者结合：先做算子替换，再尝试 compile 看是否有额外收益

## 6. KV Cache 优化

LLM 自回归生成时，KV Cache 的内存管理影响显著：

### 6.1 连续内存 KV Cache

```python
# 预分配连续内存，避免每步 cat 产生的碎片
max_seq_len = 2048
kv_cache = torch.zeros(2, batch, num_heads, max_seq_len, head_dim,
                       dtype=dtype, device="npu:0")
# 推理时 in-place 写入
kv_cache[0, :, :, seq_pos, :] = new_key
kv_cache[1, :, :, seq_pos, :] = new_value
```

### 6.2 PagedAttention

vLLM 等框架的 PagedAttention 在 NPU 上需要特殊适配，目前 MindIE 提供了部分支持。

## 7. npu_dtype_cast vs .to(dtype)

```python
# .to(dtype) 可能触发非最优路径
x = x.to(torch.float16)

# npu_dtype_cast 走 NPU 原生类型转换，更高效
x = torch_npu.npu_dtype_cast(x, torch.float16)
```

在频繁做类型转换的场景（如混合精度推理中的上下 cast），差异可能累积。

## 8. npu_fusion_attention fp32 精度问题（重要踩坑）

**发现**：`npu_fusion_attention` 在 fp32 精度下与标准 SDPA/eager attention 存在显著数值差异。

**实测数据**（BERT-small, 8 heads, head_dim=64, fp32）：
- SDPA vs npu_fusion_attention cosine: **0.807**（远低于 0.99 阈值）
- max abs error: **3.52**
- SDPA vs eager cosine: **1.000**（标准实现之间无差异）

**原因**：`npu_fusion_attention` 底层使用 flash attention 实现，在 fp32 下仍可能使用内部低精度累加，导致与标准数学实现有偏差。

**影响范围**：
- decoder-only 模型（bf16 推理）：影响较小，因为 bf16 本身精度较低
- encoder-only 模型（fp32 推理，如 BERT）：影响严重，cosine 从 1.0 跌到 0.8

**结论**：**fp32 推理的 encoder-only 模型（BERT、ViT 等）不应使用 npu_fusion_attention**。改用 `npu_add_layer_norm` 替代 residual + LayerNorm 融合，仍有 ~8% 提速。

## 9. 内置模型（Built-in）的 NPU 优化注入方式

对于 transformers 内置模型（BERT、GPT-2 等没有 custom modeling 代码的模型），`from_pretrained(MODEL_ID)` 不会加载 model_files 中的自定义代码。

**解决方案**：Monkey-patch transformers 模块中的类方法。

```python
import transformers.models.bert.modeling_bert as _bert_mod

# 在 from_pretrained 之前 patch
_orig_forward = _bert_mod.BertSelfOutput.forward
def _npu_patched(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    if _is_npu(hidden_states) and not self.training:
        return torch_npu.npu_add_layer_norm(
            hidden_states, input_tensor,
            self.LayerNorm.weight, self.LayerNorm.bias,
            self.LayerNorm.eps,
        )[0]
    hidden_states = self.dropout(hidden_states)
    return self.LayerNorm(hidden_states + input_tensor)

_bert_mod.BertSelfOutput.forward = _npu_patched
# 然后 from_pretrained 加载的实例会使用 patched forward
model = BertModel.from_pretrained(MODEL_ID, ...)
```

**注意**：transformers 5.x 中 `BertSdpaSelfAttention` 已被移除，attention 统一由 `BertSelfAttention` + `ALL_ATTENTION_FUNCTIONS` 调度。

## 10. CamemBERT / RoBERTa 家族优化模式

CamemBERT 是 RoBERTa 的子类，model_type=camembert，有独立的 modeling_camembert.py（复制 RoBERTa 代码，类名以 Camembert 前缀）。

### 结构特点

- `CamembertSelfOutput`: attention 残差 + LayerNorm（dense + dropout + LN(residual)）
- `CamembertOutput`: FFN 残差 + LayerNorm（dense + dropout + LN(residual)）
- 使用 GELU 激活、标准 LayerNorm（非 RMSNorm）
- fp32 推理

### 优化方法

两个类均可 patch `npu_add_layer_norm`：
```python
import transformers.models.camembert.modeling_camembert as _camembert_mod

def _npu_output_forward(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    if _is_npu(hidden_states) and not self.training:
        return torch_npu.npu_add_layer_norm(
            hidden_states, input_tensor,
            self.LayerNorm.weight, self.LayerNorm.bias,
            self.LayerNorm.eps,
        )[0]
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states

_camembert_mod.CamembertSelfOutput.forward = _npu_output_forward
_camembert_mod.CamembertOutput.forward = _npu_output_forward
```

### 实测数据（CamemBERT-base, fp32, NPU）

- 提速: +7.1%（单样本），+5.6%（100 样本平均）
- Cosine: 1.0, Max error: 1.86e-6
- 不使用 npu_fusion_attention（fp32 encoder-only 已知精度差）

## 11. ViT Pre-LN 架构下的 npu_add_layer_norm 融合策略

ViT（包括 DINO ViT、DeiT 等）使用 Pre-LN（pre-normalization）架构，与 BERT 的 Post-LN 不同：

```
ViT (Pre-LN):
  x_norm = LN(x)             # layernorm_before (standalone, cannot fuse)
  attn_out = Attention(x_norm)
  x = attn_out + x            # residual (no LN after)
  y = LN(x)                   # layernorm_after (CAN fuse with previous add!)
  y = FFN(y)
  out = y + x                 # residual (no LN after)

BERT (Post-LN):
  attn_out = Attention(x)
  x = LN(attn_out + x)        # residual + LN fused with npu_add_layer_norm
```

### 融合策略

虽然 ViT 的 Pre-LN 中 layernorm_before 和最终 layernorm 无法融合（它们没有紧邻的 residual add），但可以将 attention residual 和 layernorm_after 融合：

```python
# Original ViT:
#   hidden_states = attention_output + hidden_states  # residual
#   layer_output = self.layernorm_after(hidden_states)  # LN

# Fused:
layer_output = torch_npu.npu_add_layer_norm(
    attention_output, hidden_states,
    self.layernorm_after.weight, self.layernorm_after.bias,
    self.layernorm_after.eps,
)[0]
# = LN(attention_output + hidden_states) -- mathematically identical
```

**收益**：DINO ViT-B/16 (fp32, 12 layers) 提速 **+35.8%**（0.592s -> 0.380s），cosine 1.0, max_err 2.07e-05。

### npu_layer_norm_eval 已弃用

torch_npu 2.9.0 下 `npu_layer_norm_eval` 报 `SetPrecisionMode: error code is 500001`。使用 `npu_add_layer_norm` 替代独立 LayerNorm 不可行（它需要两个输入做加法）。对于无法融合的独立 LN，保持使用 `F.layer_norm`（已在 NPU 上优化）。

## 12. 内置模型 model_files 的 auto_map 方案

对于 transformers 内置模型（如 ViT、BERT），两种 NPU 优化注入方式：

### 方式 A：monkey-patch（适合小范围修改）

在 `accuracy_run_perf.py` 中、`from_pretrained` 之前 patch。

### 方式 B：model_files + auto_map（推荐，符合规范）

1. 从 transformers 复制 `modeling_*.py` 到 `model_files/`
2. 修复相对导入为绝对导入（`from ...` -> `from transformers.`）
3. 在 `config.json` 添加 `auto_map: {"AutoModel": "modeling_vit.ViTModel"}`
4. 在 `modeling_*.py` 中直接修改代码添加 NPU 优化
5. `from_pretrained(MODEL_PATH, trust_remote_code=True)` 加载本地文件

## 13. 性能分析实践

### 准确测时

```python
# 必须 synchronize 后再计时
torch.npu.synchronize()
start = time.perf_counter()
output = model(inputs)
torch.npu.synchronize()  # 必须！否则测的是下发时间不是执行时间
end = time.perf_counter()
```

### warmup 的重要性

NPU 首次执行涉及算子编译和图优化，通常比稳态慢 2~10x。至少 warmup 2~3 次后再开始正式计时。

### 推理前设置

```python
model.eval()
torch.no_grad().__enter__()  # 或用 @torch.no_grad() 装饰器
# 可选：关闭梯度计算图，节省内存
torch.set_grad_enabled(False)
```

## 14. refer 目录中所有 API 的适用性全景分析

通读 `.claude/skills/torch-npu-optimization/refer/` 下全部 130+ 个 API 文档后，按**对当前已优化模型的适用性**分级如下。

### 14.1 已使用且验证的 API（5 个）

| API | 模型 | 效果 |
|-----|------|------|
| `npu_add_layer_norm` | BERT, CamemBERT, ViT | +7~35% |
| `npu_rms_norm` | Qwen-7B | 稳定提速 |
| `npu_swiglu` | Qwen-7B | 稳定提速 |
| `npu_rotary_mul` | Qwen-7B, ModernBERT | 中等提速 |
| `npu_fusion_attention` | Qwen-7B, OPT-125m | +9~37% |

### 14.2 未使用但值得关注的 API

#### npu_ffn — 融合 FFN（P1 优先级）

**公式**：`out = activation(x·W₁+b₁)·W₂+b₂`，一次调用完成 FFN 全部计算。

**支持激活**：gelu, fastgelu, relu, silu, geglu, swiglu, reglu。

**适用性**：
- OPT-125m (fp16)：✅ 可行，使用 ReLU 激活，fp16 满足要求
- BERT/CamemBERT/ViT/ModernBERT (fp32)：❌ **仅支持 fp16/bf16**

**难点**：需从 `nn.Linear` 提取 `weight`/`bias` tensor，将 `self.linear(x)` → `self.linear2(act(self.linear(x)))` 改为 `npu_ffn(x, self.linear.weight.T, self.linear2.weight.T, activation, bias1=self.linear.bias, bias2=self.linear2.bias)`。

**门槛**：文档提到"激活层为 geglu/swiglu/reglu 时，性能使能需 vector 耗时 30us 且占比 10% 以上"。

#### npu_gelu — NPU 原生 GELU（P2 精度修正）

**关键发现**：NPU 原生 gelu 的 `approximate` 参数**不起作用，默认走 tanh 近似**。BERT/CamemBERT/ViT 的 `ACT2FN["gelu"]` 默认为 erf 模式。

```python
# PyTorch 代码意图用 erf 模式：
hidden = F.gelu(hidden)  # approximate='none'
# NPU 上实际走 tanh 近似（静默差异）！

# 修正：显式使用 npu_gelu
hidden = torch_npu.npu_gelu(hidden, approximate='none')
```

**收益**：主要价值是精度修正，不一定加速。

#### npu_gelu_mul — 融合 GELU + 乘（P2）

**公式**：`GELU(x[..., :half]) * x[..., half:]`，适合 GeGLU 门控结构。

**适用**：ModernBERT 使用 `GELU(gate) * up`，语义匹配。但末维约束 ≤ 1024，ModernBERT intermediate_size=1152 可能超限。

#### npu_prompt_flash_attention — 全量 FA（P3）

标准 softmax（非 online softmax），理论上 fp16 精度优于 `npu_fusion_attention`。但 **仅支持 fp16/bf16**，且 OPT 已有 npu_fusion_attention 方案。

### 14.3 受限不可用的 API

| API | 原因 |
|-----|------|
| `npu_scaled_masked_softmax` | SDPA 已优化；H/W 须 ≥32 且整除 32；小模型常不满足 |
| `npu_transpose_batchmatmul` | 仅 3D tensor；K/N 须整除 128 |
| `fuse_add_softmax_dropout` | 仅训练有收益；推理 dropout=0 无意义 |
| `npu_interleave_rope` | D 必须等于 64，约束太严 |
| `npu_fused_infer_attention_score/v2` | 面向 KV Cache 增量推理，encoder 不适用 |

### 14.4 已废弃 / 不适用

| API | 状态 | 替代 |
|-----|------|------|
| `npu_silu` | 计划废弃 | `F.silu` |
| `npu_dtype_cast` | 计划废弃 | `x.to(dtype)` |
| `npu_layer_norm_eval` | 报错 500001 | `npu_add_layer_norm` |
| `npu_fused_attention` (contrib) | 已废弃 | `npu_fusion_attention` |
| `npu_fused_attention_with_layernorm` (contrib) | 已废弃 | — |
| 量化 API (`npu_quant_*` 等) | 不在范围 | INT8/INT4 专项优化 |
| MoE API (`npu_moe_*`) | MoE 专用 | 小模型不使用 |

### 14.5 核心结论

> **fp32 推理模型可用算子极其有限**。许多高性能融合算子（`npu_ffn`、`npu_prompt_flash_attention`、`npu_fusion_attention`）仅支持 fp16/bf16。对于 fp32 encoder-only 模型，只有 `npu_add_layer_norm`、`npu_rotary_mul`、`npu_gelu` 等少数算子可用。
>
> **进一步优化的最大杠杆是改用半精度推理**，这能解锁更多融合算子。代码层面无需改动的快速加速方式是设置 `TASK_QUEUE_ENABLE=1`（异步算子下发，+5~15%）。

### 14.6 精度类型限制速查

| API | fp16 | bf16 | fp32 |
|-----|:----:|:----:|:----:|
| `npu_add_layer_norm` | ✅ | ✅ | ✅ |
| `npu_rms_norm` | ✅ | ✅ | ✅ |
| `npu_rotary_mul` | ✅ | ✅ | ✅ |
| `npu_fusion_attention` | ✅ 精度好 | ✅ 精度好 | ❌ cosine~0.8 |
| `npu_ffn` | ✅ | ✅ bias 需 float32 | ❌ |
| `npu_prompt_flash_attention` | ✅ | ✅ | ❌ |
| `npu_swiglu` | ✅ | ✅ | ✅ |
| `npu_gelu` | ✅ | ✅ | ✅ |
| `npu_gelu_mul` | ✅ | ✅ | ✅ |
| `npu_scaled_masked_softmax` | ✅ | ✅ | ✅ |

### 14.7 npu_ffn 实战经验

**OPT-125m 多精度优化验证（2026-03-16）**:

1. **npu_ffn 支持 ReLU 激活**：OPT 的 FFN 是 `fc1(x) -> relu -> fc2(x)`，对应 `npu_ffn(x, w1, w2, "relu")`。支持 activation: `"fastgelu", "gelu", "relu", "silu", "geglu", "swiglu", "reglu"`。

2. **weight 传递需转置**：`nn.Linear` 的 weight shape 是 `[out_features, in_features]`，但 npu_ffn 期望 `[K1, N1]`（即 `[in, out]`）。需要 `.t().contiguous()`。

3. **bf16 下 bias 必须转 float32**：当 `inner_precise=0`（高精度模式）且输入为 bf16 时，NPU 算子要求 bias 为 float32，否则报错：
   ```
   AclNN_Parameter_Error(EZ1001): Tensor ffnParams.bias1 not implemented for DT_BFLOAT16,
   should be in dtype support list [DT_FLOAT,].
   Detected high precision, bias dtype is not right!
   ```
   **修复**：`b1 = b1.float() if b1 is not None and x.dtype == torch.bfloat16 else b1`

4. **fp16 下 bias 可保持 fp16**：不需要特殊处理。

5. **monkey-patch 完整 FFN**：需要替换整个 `OPTDecoderLayer.forward`，不能只 patch FFN 部分，因为 FFN 中的 hidden_states 有 reshape/residual 逻辑需要保持。

6. **多精度支持最佳实践**：通过 `--dtype` 参数控制，fp16/bf16 启用 npu_fusion_attention + npu_ffn，fp32 仅用 TASK_QUEUE_ENABLE=1。

## 16. LongRoPE 的 npu_rotary_mul 兼容性

Phi-3.5-mini 使用 LongRoPE (`rope_scaling.type = "longrope"`)，cos/sin 包含 scaling_factor 缩放。

**实测结论**：npu_rotary_mul 完全兼容 LongRoPE，cosine similarity = 1.000000 (50 samples, bf16)。

**关键参数**：
- `max_position_embeddings = 131072`, `original_max_position_embeddings = 4096`
- `scale = 32.0`, `scaling_factor = 1.190238` (sqrt(1 + log(32)/log(4096)))
- LongRoPE 通过 short_factor/long_factor 动态调整 inv_freq，cos/sin 已包含 scaling_factor

**patch 方式**（trust_remote_code 模型）：
- 不能用 model traversal（`model.named_modules()`），因为 `apply_rotary_pos_emb` 是模块级函数
- 通过 `sys.modules` 查找 `"modeling_phi3"` 匹配的模块，直接替换 `module.apply_rotary_pos_emb`
- Phi-3.5-mini 的 transformers_modules 路径含 `microsoft.Phi_hyphen_3_dot_5_hyphen_mini_hyphen_instruct`

**monkey-patch bound method**：
- trust_remote_code 模型用 `module.forward = lambda self, hs: ...` 会报 `missing argument`（因为直接赋值不绑定 self）
- 必须用 `types.MethodType(func, module)` 绑定

## 17. generate kwargs 中的 use_cache 冲突

当 `inputs` dict 中已包含 `use_cache=False`，且 generate kwargs 中也显式传 `use_cache=False` 时，`dict(**inputs, use_cache=False)` 会报 `TypeError: dict() got multiple values for keyword argument 'use_cache'`。

**修复**：构建 gen_kwargs 时过滤掉 inputs 中的 use_cache：
```python
gen_inputs = {k: v for k, v in inputs.items() if k != "use_cache"}
gen_kwargs = dict(**gen_inputs, use_cache=False, ...)
```
