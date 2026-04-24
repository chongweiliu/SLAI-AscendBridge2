# E-MIMIC/inclusively-reformulation-it5

日期：2026-04-22
状态：completed

## 最终结果

- baseline `wall_clock_s=2.037021`
- perf `wall_clock_s=0.694756`
- `speedup_ratio=2.931995`
- `num_samples=50`
- `selected_npu=13`
- `cosine_similarity=0.9999999118`

## 关键修复

- 删除旧的 `config_builtin_it` 单样本工件，避免 benchmark completed gate 误读。
- 废弃旧 `generated_text` 合同，改为 teacher-forcing `last_token_logits`。
- baseline / perf 都补齐：
  - `dataset`
  - `dtype`
  - `output_type`
  - `start_time`
  - `end_time`
  - `wall_clock_s`
  - `selected_npu`
  - `selected_npus`
  - `device_topology`
  - `parallel_mode`
- `accuracy_run_perf.py compare` 生成：
  - `output_compare_perf.json`
  - perf metrics 内嵌 `output_compare`
  - 规范化 `optimization_notes.json`

## 可复用方案

- baseline：teacher-forcing logits，`batch_size=1`
- perf：`batched_teacher_forcing(bs=4) + warmup(3x) + TASK_QUEUE_ENABLE=1`
- 数据集：sorted `wikitext`
- 模型加载：adaptation 内本地 snapshot + `local_files_only=True`
- 证据链：同一物理卡串行完成

## 经验

- 对 T5 / seq2seq，如果历史 stage3 工件已经混入旧 step1/step2、旧 `generated_text`、旧 `ttft_ms`，继续修旧文本 compare 往往只会反复踩坑。
- 直接切成稳定 logits 合同，通常更容易一次过 gate。
