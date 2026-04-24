---
name: qa_matscibert_seed36_failure
description: MaterialsInformaticsLaboratory/QA-MatSciBERT-seed36 NPU优化失败：测量口径导致speedup_ratio无效
type: project
---

# MaterialsInformaticsLaboratory/QA-MatSciBERT-seed36 失败案例

**时间**: 2026-03-29
**结果**: failed
**模型**: MaterialsInformaticsLaboratory/QA-MatSciBERT-seed36 (BERT-base QA, 12层, post-norm)
**优化项**: warmup(3x) + TASK_QUEUE_ENABLE (融合算子全部禁用)

## 失败根因

### 1. 融合算子导致输出回归
- **npu_add_layer_norm**: cosine = -0.45（严重回归）
- **npu_gelu**: 数值异常

### 2. 测量口径冲突导致 speedup_ratio 无效

**关键数据**:
- baseline_wall_clock_s = 11.5736s（来自 benchmark_metrics_npu_fp32_pretrained_builtin.json）
- perf_wall_clock_s = 0.284887s（来自 benchmark_metrics_npu_fp32_pretrained_builtin_perf.json）
- baseline_latency_s = 0.234892（per-sample）
- perf_latency_s = 0.284887（per-sample）
- steady_state_baseline_latency_s = 0.00569774
- steady_state_perf_latency_s = 0.00569774

**矛盾**:
- 按 wall_clock_s 计算: speedup_ratio = 11.5736 / 0.284887 = **40.62x** (虚高)
- 按 per-sample latency 计算: speedup_ratio = 0.234892 / 0.284887 = **0.82x** (实际无提速)
- steady-state per-sample 两者相等: 1.0x (无真实提速)

**根因**: baseline 的 11.57s 包含了 step1(encoder-only) + step2(50样本) + profiling 开销，而 perf 的 0.285s 只是 50 样本的 wall-clock。口径不一致导致 speedup_ratio 无意义。

## check_accuracy_run_perf.py 报错

```
optimization_notes.best_result.baseline_latency_s 与 baseline_artifact.latency_s 不一致
```

修复后变成:
```
results[0] speedup_ratio 必须按 baseline_wall_clock_s / perf_wall_clock_s 计算
```

## 教训

1. **runtime_only speedup 也必须 >= 1.0**: CLAUDE.md 规定 `best_result.speedup_ratio` 默认必须 > 1.0；runtime_only 路径允许 = 1.0，但不允许 < 1.0
2. **warmup + TQE 对小模型无真实提速**: BERT-base (12层, 768hidden) 在 symmetric warmup 下 steady-state speedup = 1.0x
3. **测量口径必须在对比前对齐**: baseline 和 perf 必须使用相同的测量范围（都包含 step1，或都不包含）
4. **speedup_ratio >= 3.0 必须有独立 baseline 工件**: 当 speedup_ratio >= 3.0 时，必须是 independent_baseline_artifact 口径，不能是 self-baseline 或测量口径不一致的虚高值

## 相关已有案例

- `qa_scibert_seed12_pending.md`: 小模型 warmup 效应饱和，speedup = 1.0x
- `google_mt5_small_notes.md`: warmup_iterations 必须对称，artifact 内 self-baseline 元数据与 independent_baseline speedup 冲突
- `dice-research-lola_v1-notes.md`: wall-clock vs latency 速度矛盾（warmup 开销计入 wall-clock 但不计入 latency）
