# facebook/esm2_t33_650m_ur50d

日期：2026-04-22
状态：stage3 completed

## 结论

- 旧的 hybrid `model_files` 路径能跑通，但不能作为正式 completed 结果
- 当前正式结果改为 runtime-only
- 单模型正式 speedup 证据链在同一卡 `13` 上串行完成

## 关键问题

- `model_files/modeling_esm.py` 为兼容当前 `transformers==5.2.0`，需要补：
  - `OutputRecorder`
  - `find_pruneable_heads_and_indices`
  - `prune_linear_layer`
  - `check_model_inputs`
  - `auto_docstring` no-op fallback
  - `_tied_weights_keys` 旧 list -> dict
  - `get_head_mask` 缺失 fallback
- NPU 上还需要强制 attention 走 eager，避免 `sdpa_attention_forward -> FlashAttentionScore` 因 mask shape 不支持而崩

## 为什么不能用 model_files 作为正式结果

- `model_files` 路径虽可运行并有高倍速度，但 protein logits 精度严重漂移：
  - `cosine_similarity ~= 0.802975`
  - `min_cosine_similarity ~= 0.726255`
  - `max_abs_error ~= 14.5945`
- 因此不满足 stage3 completed gate，不能写 completed

## 正式通过 gate 的方案

- `accuracy_run_perf.py` 支持：
  - `run` / `compare`
  - `--load-from {model_files,baseline_snapshot}`
  - `--task-queue-enable`
  - 从 perf 工件继承 `selected_npu(s)` / `device_topology` / `parallel_mode`
- 默认正式路径切到：
  - `--load-from baseline_snapshot`
  - `batch_size=1`
  - `warmup_iterations=3`
  - `TASK_QUEUE_ENABLE=0`

## 经验

- 对 ESM2 650M，这个模型的 `protein_logits` 对 batched inference 很敏感
- `batch_size=8` 时虽然 `cosine` 接近 1，但 `max_abs_error` 会到 `1e-2`，completed gate 仍失败
- `batch_size=1` 后恢复到可接受范围：
  - `baseline_wall_clock_s = 21.10865`
  - `perf_wall_clock_s = 1.749925`
  - `speedup_ratio = 12.062603`
  - `cosine_similarity = 1.0`
  - `min_cosine_similarity = 1.0`
  - `max_abs_error = 7.62939453125e-06`
  - `text_match_rate = 1.0`

## 写库前检查

- `benchmark/scripts/check_accuracy_run.py --adapt facebook_esm2_t33_650m_ur50d`
- `optimization/scripts/check_accuracy_run_perf.py --adapt facebook_esm2_t33_650m_ur50d`
- `optimization/scripts/check_optimization_notes.py --adapt adaptations/facebook_esm2_t33_650m_ur50d`

全部通过后，才能用 `board_ops.py update_optimization_status ... completed`。
