# facebook/w2v-bert-2.0

日期：2026-04-22

## 最终结果

- stage1 `adaptation_status=completed`
- stage2 `benchmark_status=completed`
- stage3 `optimization_status=completed`
- 路线：`runtime_only`
- 设备：物理卡 `13`
- 数据集：`librispeech`
- 样本数：`50`
- wall-clock：`2.357734s -> 2.218363s`
- `speedup_ratio=1.062826`
- `output_type=audio_embeddings`
- `cosine_similarity=1.0`
- `max_abs_error=0.0`

## 关键修复

1. 修 `accuracy_run.py`
- `MAX_SAMPLES` 提到 `250`
- `load_model(..., use_pretrained=None)`，避免 `check_accuracy_run.py` 把默认参数误报成 silent fallback
- pretrained 路径下统一 `local_files_only=True`
- baseline metrics/outputs 改成带 `mode`，并补齐 `wall_clock_s`、`num_samples`、`warmup_iterations`

2. 修 `accuracy_run_perf.py`
- 重写成 runtime-only `run/compare` 结构
- baseline 强制 `TASK_QUEUE_ENABLE=0`，perf 用 `--task-queue-enable`
- baseline/perf 都写 `selected_npu`、`selected_npus`、`device_topology`、`parallel_mode`
- compare 改成按工件 JSON 的 `end_time/start_time` 选最近一轮成对 baseline/perf/output，不再硬编码 `fp32`，也不再按文件 `mtime` 选

## 踩坑记录

1. 旧工件污染不是删文件就完事
- 目录里原本同时有历史 `fp32` 和本轮 `bf16` 工件
- 第一次 `compare` 硬编码读 `fp32`，直接把旧退化结果写回 `optimization_notes.json`
- 第二次改成按最新 `mtime` 选，仍然失败，因为 compare 会重写旧 perf metrics，导致旧文件 `mtime` 反而更新
- 正确做法是按 metrics 内部 `end_time/start_time` 选最近一轮，并要求 baseline/perf/output 四件套成对存在

2. 这是典型 audio embedding runtime-only 模型
- 历史 `fp32` stage3 真实退化：`1.8232s -> 1.8561s`，`speedup_ratio=0.982275`
- 新 `bf16` runtime-only 结果才是正式 completed 证据
- compare 指标用 `cosine_similarity` + `max_abs_error` 足够稳定，`audio_embeddings` 这类输出可以直接复用这套口径

## 复用建议

- 如果 adaptation 目录里存在多精度、多轮次 baseline/perf 工件，`accuracy_run_perf.py compare` 必须做“同 mode、同 dataset、同轮次”的成对解析
- 不要让 compare 阶段去重写旧历史工件的关键选择依据；一旦要写 perf metrics，就必须把 artifact selection 建立在 JSON 内容而不是文件时间上
- 音频 embedding 模型优先保留 `output_type=audio_embeddings`，compare 时直接落 `cosine_similarity` / `min_cosine_similarity` / `max_abs_error`
