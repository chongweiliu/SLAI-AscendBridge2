# MilosKosRad/ScientificNLIsrb

## 结论

- 路径：`runtime_only`
- 设备：`ASCEND_RT_VISIBLE_DEVICES=13`
- 数据集：`wikitext`
- speedup：`2.069174s -> 0.718568s`，`2.87958x`
- 精度：`cosine_similarity=1.0`，`text_match_rate=1.0`

## 有效做法

- 只从 adaptation 私有 snapshot 加载：
  - `models/models--MilosKosRad--ScientificNLIsrb/refs/main`
  - `models/models--MilosKosRad--ScientificNLIsrb/snapshots/<ref>`
- `AutoTokenizer/AutoConfig/AutoModelForSequenceClassification` 全部带 `local_files_only=True`
- perf 路径只做运行时优化：
  - `warmup(3x)`
  - `TASK_QUEUE_ENABLE=1`
  - `batched_inference(batch_size=8)`

## 踩坑

- 老的 baseline/perf 工件与 notes 不一致，且历史 `speedup_ratio < 1.0`，属于脏工件，不能沿用。
- `check_accuracy_run.py` / `check_accuracy_run_perf.py` 会校验：
  - `ttft_ms <= latency_s * 1000 + 1e-3`
- 如果把 `ttft_ms` 写成 `round(latency_s * 1000, 2)`，可能因为进位出现：
  - `latency_s=0.050636`
  - `ttft_ms=50.64`
  - 从而被判定元数据不可信。

## 修复

- `accuracy_run.py` 和 `accuracy_run_perf.py` 统一改成：
  - `ttft_ms = round(latency_s * 1000, 3)`
- 然后必须整轮重跑：
  - baseline
  - perf
  - compare
  - `check_accuracy_run.py --adapt ...`
  - `check_accuracy_run_perf.py --adapt ...`
  - `check_optimization_notes.py --adapt ...`
