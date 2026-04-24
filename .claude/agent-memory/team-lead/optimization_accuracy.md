# NPU 优化精度对比踩坑记录

## 核心发现：warmup 消耗 RNG 状态导致假性精度问题

### 现象
accuracy_run_perf.py（优化版）vs accuracy_run.py（baseline）对比时，余弦相似度仅 0.0006，
看似 npu_rms_norm/npu_gelu 算子替换破坏了精度。

### 根因
accuracy_run_perf.py 在 benchmark 前有 2 次 warmup 迭代，每次调用 `create_model_inputs()`
里的 `torch.randn()`，消耗了 torch RNG 状态。之后 step1/step2 的随机输入和 baseline
使用不同的随机数，导致输出完全不同。

### 修复
在 warmup 循环结束后、step1 开始前，重新设置所有随机种子：
```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.npu.manual_seed_all(SEED)
```

修复后余弦相似度从 0.0006 恢复到 0.9998。

### 经验教训
1. **任何会消耗 torch RNG 的操作（warmup、额外推理）都必须在之后重设种子**
2. **Torch RNG 和 Python random 是两个独立生成器**，warmup 里 torch.randn 只消耗 torch RNG
3. **不要仅凭低余弦相似度就判定算子替换有精度问题**，先排除 RNG 状态差异
4. **TASK_QUEUE_ENABLE=1 改变算子调度顺序**，但在重设种子后对精度影响可忽略（cosine=0.9998）

### 对比验证流程
```
无 patch 无 TASK_QUEUE → cosine=1.0（基线）
无 patch 有 TASK_QUEUE → cosine=1.0（TASK_QUEUE 不影响）
有 npu_gelu → cosine=1.0（npu_gelu 精度无损）
有 npu_rms_norm + npu_gelu + TASK_QUEUE → cosine=0.9998（全部优化，精度可接受）
```

## 模型
- Lightricks/LTX-2.3（22B DiT，48 层）
- Ascend 910_9382，bf16
- 优化加速 1.24x（5.45s → 4.41s）
