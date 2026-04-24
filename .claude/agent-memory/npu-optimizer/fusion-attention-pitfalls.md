# npu_fusion_attention 踩坑记录

## 坑 1: 去掉显式 causal mask 导致精度崩塌

### 现象

认为 `pre_tockens=65536, next_tockens=0` 足以实现 causal attention，去掉了显式 `atten_mask`。

### 结果

- Logits cosine 从 0.999+ 暴跌至 0.5~0.84
- 生成文本完全乱码

### 原因

`pre_tockens/next_tockens` 是滑动窗口参数，不等同于显式的上三角因果 mask。在某些场景下（尤其是 prefill 阶段），单独使用这两个参数无法正确屏蔽未来 token。

### 解决

始终使用显式 causal mask:
```python
causal_mask = torch.triu(
    torch.ones(seq_q, seq_k, dtype=torch.bool, device=device),
    diagonal=seq_k - seq_q + 1,
)
```

## 坑 2: 不合并 padding mask 导致 PPL 差异巨大

### 现象

只用 causal mask，忽略 Transformer 层传入的 `attention_mask`。

### 结果

- Logits cosine 还行 (0.992~0.999)
- 但 PPL 平均相对差异从 9.7% 飙升至 4092%
- 个别样本 PPL 从 95 跳到 28282

### 原因

Transformer 的 `attention_mask` 包含 padding 信息。不合并意味着 padding token 也参与了 attention 计算，在某些序列上会产生巨大的误差。

单步 forward 的 last-token logits 影响较小（因为最后一个 token 位置受 padding 影响小），但 PPL 的计算覆盖全序列，累积误差显著。

### 解决

```python
if attention_mask is not None:
    pad_mask = (attention_mask.squeeze(1).squeeze(1) < -1.0)
    causal_mask = causal_mask.unsqueeze(0) | pad_mask.unsqueeze(-2)
    atten_mask = causal_mask.unsqueeze(1)
else:
    atten_mask = causal_mask.unsqueeze(0).unsqueeze(0)
```

## 坑 3: mask 语义相反

### 现象

PyTorch 标准的 `attention_mask` 是 float 加法 mask (0.0=attend, -inf=mask)。

但 `npu_fusion_attention` 的 `atten_mask` 是 bool mask (True=屏蔽, False=参与)。

### 解决

转换时用阈值判断: `pad_mask = (attention_mask < -1.0)` 将 -inf 转为 True。

## 坑 4: input_layout 与 tensor shape 不匹配

### 现象

有些模型的 attention 在进入计算前先 permute 为 `(B, H, S, D)` 格式。

### 解决

- 如果 tensor 已经是 `(B, S, H, D)`: 用 `input_layout="BSND"`
- 如果 tensor 是 `(B, H, S, D)`: 用 `input_layout="BNSD"`
- 或者在 NPU 分支中跳过 permute，直接用原始的 `(B, S, H, D)` + `"BSND"`

## 坑 5: step1 profiler carryover 导致 step2 wall-clock 测量失真

### 现象

accuracy_run_perf.py 采用 step1(profiling) + step2(accuracy) 架构。step1 启用 torch_npu.profiler，产生 profiler context。step2 虽然不启用新的 profiler context，但 step1 的 profiler context 可能残留于 NPU runtime，导致 step2 的 wall-clock 测量（start_time=step1.start, end_time=step2.end）包含约 0.25s 额外开销。

导致：wall-clock speedup < 1.0（实际是 slowdown），但 per-sample latency speedup = 1.28x（真实提速）。

### 诊断

```python
# self-baseline step2 wall-clock = 0.3421s (无 profiler)
# perf step2 wall-clock = 0.5177s (step1 带 profiler)
# profiler carryover 额外 ~0.176s per sample × 50 = ~8.8s？ 实际约 0.25s total
```

### 根因

step1 的 `npu_profile` context 管理（`with _PerfMonitor`）结束后，NPU runtime 的 profiling 状态未完全清理，残留状态影响后续 inference 的调度和同步开销。

### 影响

board_ops 的 `_validate_optimization_metric_artifacts` 要求 `speedup_ratio = baseline_wall_clock_s / perf_wall_clock_s` 且 speedup > 1.0。profiler carryover 导致 perf wall-clock 偏高，wall-clock speedup < 1.0，无法通过 completed gate。

### 解决

无完美解决方案。可选方案：

1. **禁用 step1 profiling**：step1 不启用 profiler，只做单样本 warmup。缺点：无 trace 文件。
2. **两步测量**：step1 单独跑 profiling（用于 trace），step2 单独跑 accuracy（不 profiling）用于精确 timing。
3. **使用 per-sample latency 而非 wall-clock**：per-sample latency 不受 profiler carryover 影响，是真实提速证据。但 board_ops 强制要求 wall-clock speedup > 1.0。
4. **在 step1 和 step2 之间插入进程重启**：完全清除 NPU runtime 状态。但开销大。

目前最优：方案 1 或 2，但会导致 trace 文件缺失（影响调试）。建议在 optimization_notes 中明确说明 per-sample speedup 是真实值，wall-clock 失真是测量问题。
