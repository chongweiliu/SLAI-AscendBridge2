# juwonna7/Qwen2.5-VL-7B-Scientific-VLM-post-pretrain 优化失败

## 模型信息
- **model_id**: juwonna7/Qwen2.5-VL-7B-Scientific-VLM-post-pretrain
- **架构**: Qwen2.5-VL (VLM, 7B params), RMSNorm + SwiGLU + mRoPE(3D) + GQA
- **日期**: 2026-03-27

## 尝试的优化

### 方案 1: npu_rms_norm + npu_swiglu + warmup + TQE
- **结果**: 初始测量显示 regression (0.05x, 即慢 19 倍)
- **根因**: `run_step1` 中的 profiling context 导致perf latency 虚高（9.5s vs baseline 0.5s）
- **修复**: 移除 profiling overhead 后，latency speedup = 1.08x (0.55s → 0.51s)

### 方案 2: runtime_only (warmup + TQE, 无 fusion patches)
- **结果**: speedup_ratio = 1.0 (wall-clock based)
- **问题**: completion gate 要求 speedup_ratio > 1.0，但 runtime_only 无法提供 wall-clock 提速
- **验证**: check_optimization_notes.py 通过，但 completed gate 模拟校验失败

## 关键发现

1. **profiling overhead 误判**: `torch_npu.profiler` 在 `run_step1` 中引入巨大开销（10x+），导致误以为 fusion patches 导致 regression
2. **runtime_only 限制**: warmup 带来的 latency 改善无法转化为 wall-clock 提速，因为 baseline 和 perf 的热身时间相同
3. **completion gate 规则**: `speedup_ratio = baseline_wall_clock_s / perf_wall_clock_s` 必须 > 1.0，latency speedup 不被认可

## 结论

- **optimization_status**: pending (speedup_ratio = 1.0, 不满足 > 1.0 要求)
- **建议**: 标记为 not_applicable 或 skip，因 Qwen2.5-VL 架构限制（mRoPE 使 npu_rotary_mul 不适用，fusion ops 对 VLM 收益有限）
- **教训**: 测量 NPU 性能时必须排除 profiling overhead，使用独立的 latency 测量而非依赖 step1 的 profiler-wrapped 测量