---
name: dler_r1_7b_notes
description: nvidia/DLER-R1-7B-Research runtime-only completed path and compare contract fixes
type: reference
---

# nvidia/DLER-R1-7B-Research

## 最终结论

- 该模型可以走 **runtime_only completed**。
- 正式链路：本地 snapshot + `warmup(3x) + TASK_QUEUE_ENABLE=1 + batch_size=1` + teacher-forcing `last_token_logits + perplexity`
- 固定卡：`ASCEND_RT_VISIBLE_DEVICES=12`

## 最终结果

- baseline artifact: `benchmark_metrics_npu_bf16_pretrained_wikitext.json`
- perf artifact: `benchmark_metrics_npu_0_bf16_pretrained_wikitext_perf.json`
- `num_samples=50`
- `baseline_wall_clock_s=18.438`
- `perf_wall_clock_s=2.381667`
- `speedup_ratio=7.741636`
- `baseline_latency_s=0.368760`
- `perf_latency_s=0.047633`
- `cosine_similarity=0.999993`
- `min_cosine_similarity=0.999978`
- `ppl_avg_rel_diff_pct=0.0517`
- `max_abs_error=0.0`

## 这次真正卡住 completed 的是历史工件契约

### 1. baseline/perf device tag 不一致

- 新 perf 工件是 `npu_0`
- 旧 baseline 工件还是 `npu`
- compare 不能只按完全相同前缀配对

### 2. baseline outputs 的 logits 是 `list[Tensor]`

- 旧 baseline `outputs_*.pt` 里 `logits` 不是整块 tensor
- compare 前必须先标准化 `Tensor` / `list[Tensor]`

### 3. baseline metrics 缺 `wall_clock_s`

- 旧 baseline 只有 `latency_s` / `num_samples`
- compare 阶段要显式补 `wall_clock_s`，并写回工件，避免 gate 继续读到旧口径

### 4. teacher-forcing workload 不该保留 `ttft_ms/tpot_ms`

- 正式 workload 是 forward，不是流式 generate
- 混入 `ttft_ms/tpot_ms` 会触发 metadata health gate
- 这类脚本应直接写 `ttft_ms=null`、`tpot_ms=null`

### 5. compare 真实口径是 logits，就必须把 metadata 对齐成 logits

- baseline 历史 metrics 可能还是 `generated_text`
- 但 compare 实际消费的是 logits/perplexity
- baseline/perf/best_result 的 `output_type` 必须统一成真实 compare 口径

## 经验总结

- 这类“历史 baseline + 新 runtime-only perf”假失败，大多不是模型速度或精度不够，而是 **工件契约漂移**。
- 如果 speedup/cosine 已经健康，先排查：
  - device tag 配对
  - outputs 结构兼容
  - wall-clock 来源补齐
  - metadata health
  - output_type 对齐
