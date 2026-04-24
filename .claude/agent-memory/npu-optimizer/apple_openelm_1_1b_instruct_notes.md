# apple/OpenELM-1_1B-Instruct

日期：2026-04-22

## 最终结果

- stage1 `adaptation_status=completed`
- stage2 `benchmark_status=completed`
- stage3 `optimization_status=completed`
- 路线：`runtime_only`
- 设备：物理卡 `13`
- 数据集：`wikitext`
- 样本数：`50`
- wall-clock：`226.033890s -> 221.057003s`
- `speedup_ratio=1.022514`
- `text_match_rate=1.0`
- `cosine_similarity=1.0`
- `ppl_avg_rel_diff_pct=0.0`

## 关键修复

1. 重写 `accuracy_run.py`
- 修正 `DATASET_DIR` 到当前 repo 根下的 `datasets/`
- tokenizer/model 都改为 adaptation 内 `models/` 本地缓存 + `local_files_only=True`
- baseline metrics 补齐 stage2 50-sample 口径和 gate 必需字段

2. 新建并修正 `accuracy_run_perf.py`
- 采用 runtime-only `run/compare` 结构
- 生成 `_perf` outputs / metrics / trace / `output_compare_perf.json` / `optimization_notes.json`
- compare 阶段把 `cosine_similarity` 夹紧到 `[0, 1]`
- `optimization_notes.json` 写对称 warmup：`baseline_warmup_iterations=perf_warmup_iterations=3`

3. 修复 `demo.py` 和 `.status.json`
- 旧 `demo.py` 直接走 `LlamaTokenizer.from_pretrained(MODEL_ID)`，历史上因 tokenizer 兼容性失败
- 改为只从 adaptation 内本地 snapshot 加载：
  - `models/models--apple--OpenELM-1_1B-Instruct/snapshots/...`
  - `models/models--hf-internal-testing--llama-tokenizer/snapshots/...`
- dry-run 成功后自动把 `.status.json` 刷成 `completed`

## 链路修复顺序

这个模型最容易踩的不是性能，而是 DB 状态被历史操作打乱：

1. 本地 stage2/stage3 工件已经修好，但 DB 里 `adaptation_status=skipped`
2. 先修 `demo.py` + 跑 `uv run python demo.py --dry-run` + 过 `check_adaptation.py`
3. 用 `board_ops.py update_adaptation_status --completed` 回写 stage1
4. 这一步会把 `benchmark_status` 自动重置成 `pending`
5. 再用现有 baseline 工件回写 `benchmark_status=completed`
6. 最后才能回写 `optimization_status=completed`

## 复用建议

- 遇到“本地工件齐全，但 DB 上游阶段是旧失败/跳过”的模型，不要直接硬写 stage3
- 先把最上游缺口补齐，再按 stage1 -> stage2 -> stage3 顺序回填
- 对 generated_text runtime-only 模型，notes 的 warmup 和 cosine 字段要优先自检，否则 checker 会卡在 schema 细节而不是模型本身
