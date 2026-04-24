# MaterialsInformaticsLaboratory/QA-SciBERT-seed12 Optimization Pending

## 模型信息
- **model_id**: MaterialsInformaticsLaboratory/QA-SciBERT-seed12
- **架构**: BERT-based QA (Post-LN, 12层, ~110M params)
- **日期**: 2026-03-27

## 尝试的优化

### 方案: warmup(3x) + TASK_QUEUE_ENABLE (runtime_only)
- **结果**: speedup_ratio = 1.0 (对称 warmup)
- **问题**: completion gate 要求 speedup_ratio > 1.0

## 关键发现

1. **非对称 warmup 有效但无效**: 原始测量显示 3.38x per-sample speedup (0.357s → 0.106s)，但这是因为 baseline 冷启动 vs perf 热启动的对比
2. **对称 warmup 后 speedup = 1.0**: 当 baseline 和 perf 都有 3 次 warmup 时，编译开销已被消除，两者的 per-sample 延迟几乎相同
3. **小模型特性**: 对于小模型（~110M params），warmup 已经将 per-sample 延迟降到很低（0.106s），TQE 的额外收益在噪声范围内

## 根因分析

- **非对称 speedup**: baseline_latency = 0.357s (cold), perf_latency = 0.106s (warm) → 3.38x
- **对称 speedup**: baseline_latency = 0.106s (warm), perf_latency = 0.106s (warm) → 1.0x
- **wall-clock speedup**: 1.27x (包含加载时间)

## 结论

- **optimization_status**: pending (speedup_ratio = 1.0, 不满足 > 1.0 要求)
- **建议**: 标记为 not_applicable 或 skip，因为融合算子导致回归，且 runtime_only 无法提供可测量的提速
- **教训**: 对于小模型，warmup 效应饱和后，TQE 的收益有限