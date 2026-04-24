# Qwen3-30B-A3B Base 优化记录

**模型**: `Qwen/Qwen3-30B-A3B-Base`  
**目录**: `adaptations/qwen_qwen3_30b_a3b_base`  
**测试时间**: 2026-04-23  
**最终结果**: `speedup_ratio=1.046891`, `cosine=0.999986`, `max_abs_error=0.0`

## 最终可过 gate 方案

- 物理卡固定为 `ASCEND_RT_VISIBLE_DEVICES=12,13`
- baseline: `benchmark_metrics_npu_0_bf16_pretrained_wikitext.json`
- perf: `benchmark_metrics_npu_0_bf16_pretrained_wikitext_perf.json`
- workload: pretrained + `wikitext` + `50 samples` + `max_length=128` + `batch_size=1`
- 优化项:
  - `warmup(3x)`
  - `TASK_QUEUE_ENABLE=1`
  - `npu_swiglu`

## 关键结论

1. **这类 Qwen3 MoE 30B 不要默认全开 fusion patch。**
   单独 `npu_swiglu` 就能稳定提速且不破坏精度；把 `npu_rms_norm` 加进去会直接把 `max_abs_error` 打爆。

2. **`npu_rms_norm` 在该模型上不可作为 completed 方案。**
   单 patch 实测:
   - `speedup_ratio=1.103795`
   - `cosine=0.999623`
   - `min_cosine=0.994884`
   - `max_abs_error=2.9375`
   - `ppl_rel_diff=1.172%`
   虽然 wall-clock 更快，但 completed gate 要求 `max_abs_error < 0.001`，因此必须判定为坏 patch。

3. **`npu_swiglu` 是当前安全 patch。**
   单 patch 实测:
   - `speedup_ratio=1.046891`
   - `cosine=0.999986`
   - `min_cosine=0.999980`
   - `max_abs_error=0.0`
   - `ppl_rel_diff=0.0%`
   三道本地 gate 都能通过。

4. **旧的“Qwen3 30B 主要靠 RMSNorm”结论已经过时。**
   在当前 teacher-forcing `last_token_logits + perplexity` 合同下，`npu_rms_norm` 不是安全答案；默认 patch 集必须收敛到 `npu_swiglu`。

## 代码调整

- `accuracy_run.py`
  - 改成 teacher-forcing `last_token_logits + perplexity`
  - 强制使用 adaptation 私有 snapshot
  - 补齐 `wall_clock_s` / runtime metadata / 50-sample gate 字段

- `accuracy_run_perf.py`
  - 支持 `run` / `compare`
  - 默认 `enabled-patches=npu_swiglu`
  - 生成符合 completed gate 的 `optimization_notes.json`

- `model_files/npu_patches.py`
  - 增加选择性 patch 开关 `apply_npu_patches(enabled=...)`
  - 允许单 patch / 子集 patch 做定位

## 本地验证

- `uv run python benchmark/scripts/check_accuracy_run.py --adapt qwen_qwen3_30b_a3b_base`
- `uv run python optimization/scripts/check_accuracy_run_perf.py --adapt adaptations/qwen_qwen3_30b_a3b_base`
- `uv run python optimization/scripts/check_optimization_notes.py --adapt adaptations/qwen_qwen3_30b_a3b_base`

以上全部通过。
