# Qwen-7B NPU 优化完整案例

## 模型信息

- 模型: DUTIR-BioNLP/Taiyi-LLM (Qwen-7B 架构)
- 精度: bf16
- 文件: `adaptations/dutir_bionlp_taiyi_llm/model_files/modeling_qwen.py`

## 优化内容

### 1. RMSNorm → npu_rms_norm

位置: `class RMSNorm.forward()`

```python
if _HAS_TORCH_NPU and str(x.device).startswith("npu"):
    return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]
```

### 2. SwiGLU → npu_swiglu

位置: `class QWenMLP.forward()`

原始: `a1 * F.silu(a2)`，替换为:
```python
gate_up = torch.cat([a2, a1], dim=-1)  # 注意顺序!
intermediate_parallel = torch_npu.npu_swiglu(gate_up, dim=-1)
```

### 3. Rotary Embedding → npu_rotary_mul

位置: `apply_rotary_pos_emb()`

```python
cos = cos.expand_as(t_)
sin = sin.expand_as(t_)
output = torch_npu.npu_rotary_mul(t_, cos, sin).type_as(t)
```

### 4. Attention → npu_fusion_attention

位置: `class QWenAttention.forward()`，在 flash_attn CUDA 分支之后添加。

关键点:
- query/key/value 从 `_split_heads` 出来已是 (B, S, H, D) 格式
- 使用 `input_layout="BSND"`
- causal mask 用 `torch.triu(..., diagonal=seq_k - seq_q + 1)`
- padding mask 从 float attention_mask 转换: `(attention_mask.squeeze(1).squeeze(1) < -1.0)`
- OR 合并两个 mask

## 测试结果

### 性能 (50 样本, pretrained, bf16)

| 指标 | Baseline | 优化后 | 提升 |
|------|----------|--------|------|
| Latency | 2.464 s/sample | 1.534 s/sample | +37.7% |
| Throughput | ~26 tok/s | 41.7 tok/s | +60% |

### 精度 (50 样本, pretrained, bf16)

| 指标 | 值 |
|------|-----|
| Logits 余弦相似度 | 0.993 (平均), 单样本 0.992~0.9999 |
| PPL 平均相对差异 | 9.73% |
| 文本匹配率 | 0% (正常，bf16 自回归发散) |

### PPL 逐样本对比

| 样本 | Baseline | 优化后 | 相对差异 |
|------|----------|--------|----------|
| [0] | 95.82 | 90.02 | 6.1% |
| [1] | 20.72 | 20.72 | 0.0% |
| [2] | 74.63 | 70.11 | 6.1% |
| [3] | 7.22 | 7.39 | 2.4% |
| [5] | 11.81 | 11.81 | 0.0% |
| [6] | 15.64 | 15.64 | 0.0% |

## 执行命令

```bash
# 先看卡占用并选择空闲卡
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES={selected_npu}

# 先跑 baseline
uv run python accuracy_run.py --use-pretrained --max-samples 50
# 再跑优化版（CUDA 和 NPU 各跑一次，产出 outputs_*_perf.pt）
uv run python accuracy_run_perf.py run --use-pretrained --max-samples 50
# 对比 CUDA/NPU 产出
uv run python accuracy_run_perf.py compare
```
