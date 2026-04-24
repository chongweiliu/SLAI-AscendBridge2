---
name: qwen3_embedding_4b_failure
description: Qwen3-Embedding-4B fp16 优化失败：artifact 测量口径不一致导致 completed gate 无法通过
type: project
---

# Qwen3-Embedding-4B 优化失败：artifact 测量口径不一致

## 任务信息

- **model_id**: Qwen/Qwen3-Embedding-4B
- **adaptation_path**: adaptations/qwen_qwen3_embedding_4b
- **结果**: failed（artifact 测量不一致）

## 优化内容

- **优化项**: npu_rms_norm(36 layers) + npu_swiglu(36 layers) + warmup(3x) + TASK_QUEUE_ENABLE=1
- **跳过**: npu_fusion_attention（双向 embedding 模型，causal mask 不适用）、npu_rotary_mul
- **精度**: cosine similarity = 0.999893 > 0.99 ✓

## 失败根因

benchmark 工件（accuracy_run.py 产出）的 `latency_s` 与 wall-clock 时间不一致：

| 工件 | latency_s | wall_clock_actual | latency_s/50 | wall_clock/50 |
|------|-----------|-------------------|--------------|---------------|
| baseline | 1.621062 | 8.347s | 0.0324 | **0.1669** |
| perf | 0.970391 | 7.329s | 0.0194 | **0.1466** |

completed gate 要求：`baseline_latency_s == baseline_wall_clock_actual / num_samples`

即：1.621062 == 0.1669 → **FALSE**

参考案例 qwen3_14b 通过是因为：
- baseline_latency_s = 0.062254
- baseline_wall_clock_s = 3.113
- 3.113/50 = 0.06226 ≈ 0.062254 ✓

## 教训

**measurement_contract_v3 要求 artifacts 的 latency_s 必须等于 wall_clock / num_samples**。如果 benchmark 脚本测量的 latency_s 是"单次推理的累加和"而非"wall-clock 时间 / num_samples"，则该模型无法通过 optimization completed gate。

Qwen3-Embedding-4B 的 accuracy_run.py 产出中 latency_s 含义与 wall-clock 不一致，属于 benchmark 脚本层面的测量口径问题，而非 optimization 问题。优化本身正确（npu_rms_norm + npu_swiglu + warmup + TQE 全部生效，cosine similarity 0.999893）。

## 验证方法

```bash
# 检查 completed gate
uv run python optimization/scripts/check_accuracy_run_perf.py --adapt adaptations/qwen_qwen3_embedding_4b
# 失败: "results[0] 非生成类任务要求 baseline_wall_clock_s / num_samples 与 baseline_latency_s 一致"
```

## 建议

1. 修复 accuracy_run.py 使 latency_s = wall_clock / num_samples
2. 重新运行 benchmark 产出合规的 artifacts
3. 重新执行 optimization