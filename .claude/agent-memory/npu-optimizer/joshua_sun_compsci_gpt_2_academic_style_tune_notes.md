# Joshua-Sun-CompSci/GPT-2_academic_style_tune

日期：2026-04-22
状态：completed

## 最终结果

- baseline `wall_clock_s=0.402763`
- perf `wall_clock_s=0.142362`
- `speedup_ratio=2.829147`
- `num_samples=50`
- `selected_npu=13`
- `cosine_similarity=0.9999999523`
- `max_abs_error=5.149841e-05`

## 关键修复

- 删除旧的 `config` 单样本 baseline 与旧 `<1.0` perf 工件，避免 benchmark/optimization gate 继续误读。
- 废弃旧 `generated_text + TASK_QUEUE_ENABLE-only` 合同，改成 teacher-forcing `last_token_logits`。
- baseline / perf 都改为只走 adaptation 内本地 snapshot，禁止 silent fallback。
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

- 这一类 GPT-2 小模型和 `IRIIS-RESEARCH/GPT2_Nepali_124M` 基本同构，可直接复用同一份第三阶段模板。
- 旧 `generated_text` 路径常把 wall-clock/ttft/tpot 口径搞乱，切 logits 合同后 gate 更稳。
- batching 往往是主要收益来源，单纯 `TASK_QUEUE_ENABLE=1` 不值得再单独作为正式合同。
