---
name: biomedclip_vit_bert_hf_notes
description: BiomedCLIP CLIP pre-norm 优化失败：metric validation 结构冲突
type: reference
---

## chuhac/BiomedCLIP-vit-bert-hf 优化结果

### 结论：failed - 框架 completion gate 与 CLIP 架构存在结构性冲突

### 问题
CLIP pre-norm 架构：所有 6 大融合算子均不适用（npu_add_layer_norm 要求 post-norm）。

仅运行时优化：warmup(3x) + TASK_QUEUE_ENABLE=1

### 根因：_validate_completed_optimization_notes 的 structural conflict

验证规则要求：
1. `warmup_policy = "symmetric"` + `baseline_warmup_iterations == perf_warmup_iterations`
2. `speedup_ratio = baseline_wall_clock_s / perf_wall_clock_s`
3. 若 `speedup_ratio >= 3.0`：需要 `comparison_method = "independent_baseline_artifact"`
4. 若 `speedup_ratio = 1.0`：必须是 `runtime_only`
5. `speedup_ratio` 必须 >= 1.0

CLIP 场景下的冲突：
- **Asymmetric warmup**（cold baseline 0次 vs warm perf 3次）：speedup=1.52x 真实，但 warmup_iterations 不等 → validation 失败（warmup_iterations must match）
- **Symmetric warmup**（两边均 3 次 warmup）：warmup_iterations 相等 → validation 通过，但 baseline 和 perf 使用相同代码，speedup≈1.0 → completion gate 失败（fusion 需要 speedup>1.0）
- **Fusion attention**：未测试（CLIP ViT attention 替换复杂）

### 关键验证代码（board_ops.py:1623-1631）
```python
if not _metric_close(float(baseline_warmup_iterations), float(perf_warmup_iterations)):
    return False, "optimization completed 的 baseline/perf warmup_iterations 必须一致"
# ...
if speedup_value < 1.0 - 1e-6:
    return False, "speedup_ratio 必须 >= 1"
if is_speedup_equal_one:
    if completion_kind != "runtime_only":
        return False, "仅 runtime_only completed 允许 speedup_ratio=1.0；fusion/hybrid 必须大于 1"
```

### 教训
对于纯运行时优化（无融合算子）的 CLIP/ViT 模型，completion gate 的 symmetric warmup 要求与 speedup>1.0 要求互斥。warmup 的收益只能在 asymmetric warmup 下测量，但 validation 要求 symmetric。

**建议**：此类模型应标记为 `not_applicable`（融合算子全部不适用）或 `pending`（需要修改 validation 规则允许 asymmetric warmup for runtime_only）。