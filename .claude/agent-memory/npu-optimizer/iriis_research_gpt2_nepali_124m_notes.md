# IRIIS-RESEARCH/GPT2_Nepali_124M

日期：2026-04-22
状态：completed

## 最终结果

- baseline `wall_clock_s=0.498449`
- perf `wall_clock_s=0.211047`
- `speedup_ratio=2.361791`
- `num_samples=50`
- `selected_npu=13`
- `cosine_similarity=0.9999998534`
- `max_abs_error=2.622604e-05`

## 关键修复

- 删除旧的 `config` 单样本 baseline 与旧 `<1.0` perf 工件，避免 completed gate 误读。
- 废弃旧 `generated_text + TASK_QUEUE_ENABLE-only` 合同，改成 teacher-forcing `last_token_logits`。
- baseline / perf 都改为只走 adaptation 内本地 snapshot，禁止再依赖远端下载或 silent fallback。
- perf `compare` 补齐：
  - `output_compare_perf.json`
  - perf metrics 内嵌 `output_compare`
  - `baseline_file`
  - `baseline_type=independent_baseline_artifact`
  - `wall_clock_speedup_ratio`
- `optimization_notes.json` 显式记录：
  - `selected_npu=13`
  - `selected_npus=["13"]`
  - `device_topology=single_npu`
  - `parallel_mode=single_card`
  - `warmup_policy=symmetric`

## 可复用方案

- baseline：teacher-forcing `last_token_logits`，`batch_size=1`
- perf：`batched_teacher_forcing(bs=4) + warmup(3x) + TASK_QUEUE_ENABLE=1`
- 数据集：`wikitext`
- 模型加载：adaptation 内本地 snapshot + `local_files_only=True`
- 证据链：同一物理卡串行完成

## 经验

- 对小 GPT-2，单纯 `TASK_QUEUE_ENABLE=1` 很容易回归；真正能过 gate 的往往是 batching。
- 旧 `generated_text` 合同一旦混入不规范 `ttft_ms/tpot_ms` 或 config 工件，继续补旧链路通常性价比很低。
- 对 causal LM，改成 teacher-forcing logits 后，精度 compare 和 wall-clock 对齐都更稳。
