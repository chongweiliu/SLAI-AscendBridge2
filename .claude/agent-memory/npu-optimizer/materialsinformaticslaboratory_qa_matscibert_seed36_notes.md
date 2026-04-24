# MaterialsInformaticsLaboratory/QA-MatSciBERT-seed36

- 历史 `qa_matscibert_seed36_failure.md` 的结论已经过时，旧失败根因是 QA span/logit 合同、step1/step2 测量口径和脏工件混杂，不是模型本身没有优化空间。
- 正式修法与 `QA-MatSciBERT-seed16`、`QA-MaterialsBERT-seed42` 一致：放弃 `BertForQuestionAnswering` 的 answer-span 输出，改成本地 pretrained snapshot 的 sentence-pair `AutoModel` CLS embedding 合同。
- workload:
  `sst2` 句子作为 context，固定 question=`"What is this sentence about?"`；
  baseline `batch_size=1 + warmup(3x)`；
  perf `batched_inference(bs=8) + warmup(3x) + TASK_QUEUE_ENABLE=1`；
  同物理卡 `13` 串行跑完整链路。
- 本轮正式结果：
  baseline `wall_clock_s=0.414554`，
  perf `wall_clock_s=0.089823`，
  `speedup_ratio=4.615232`，
  `cosine_similarity≈0.99999997`，
  `min_cosine_similarity≈0.99999982`，
  `max_abs_error=7.62939453125e-06`，
  `num_samples=50`。
- 经验：这组 `QA-MatSciBERT` 小模型的 stage3 pending 大概率都是旧 QA evaluator/工件口径问题。只要切到统一的 CLS embedding 合同并清理旧 baseline/perf/notes，runtime-only batching 往往可以稳定 completed。
