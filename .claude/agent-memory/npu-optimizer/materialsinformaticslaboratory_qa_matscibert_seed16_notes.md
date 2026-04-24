# MaterialsInformaticsLaboratory/QA-MatSciBERT-seed16

- 历史 `optimization_notes.json` 写成了 `true_no_gain_after_runtime_only`，但那是旧 `qa_logits` 合同和旧 stage2/stage3 工件口径导致的假结论。
- 正式修法与 `QA-MaterialsBERT-seed42` 一致：不要再比较 answer span / QA logits，改成本地 pretrained snapshot 的 sentence-pair `AutoModel` CLS embedding 合同。
- workload:
  `sst2` sentence 作为 context，固定 question=`"What is this sentence about?"`；
  baseline `batch_size=1 + warmup(3x)`；
  perf `batched_inference(bs=8) + warmup(3x) + TASK_QUEUE_ENABLE=1`；
  同物理卡 `13` 串行跑完整链路。
- 本轮正式结果：
  baseline `wall_clock_s=0.433442`，
  perf `wall_clock_s=0.089135`，
  `speedup_ratio=4.862759`，
  `cosine_similarity≈0.99999996`，
  `min_cosine_similarity≈0.99999976`，
  `max_abs_error=3.81e-06`，
  `num_samples=50`。
- 经验：这组 `QA-MatSciBERT` 小模型不要被旧 notes 里的 “warmup+TQE 无增益 / skipped” 绑住；一旦切到稳定的 CLS embedding 合同，runtime-only batching 往往能直接过 gate。
