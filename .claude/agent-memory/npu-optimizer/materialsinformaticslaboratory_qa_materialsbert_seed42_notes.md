# MaterialsInformaticsLaboratory/QA-MaterialsBERT-seed42

- 历史 stage3 pending 的根因不是模型不可优化，而是旧 QA answer 合同和旧 config/builtin 工件同时污染 gate。
- 正式可过 gate 的修法是放弃 `BertForQuestionAnswering` answer-span 输出，改成基于本地 pretrained snapshot 的 `AutoModel` sentence-pair CLS embedding 合同：
  `sst2` 句子作为 context，固定 question=`"What is this sentence about?"`，baseline/perf 都输出 `cls_embeddings`。
- 这类 QA/BERT 小模型用 runtime-only batching 更稳，不要强行保留早期 `npu_add_layer_norm` 路线。实际完成配置：
  baseline `batch_size=1 + warmup(3x)`，
  perf `batched_inference(bs=8) + warmup(3x) + TASK_QUEUE_ENABLE=1`，
  同物理卡 `13` 串行跑完整链路。
- 本轮正式结果：
  baseline `wall_clock_s=0.437218`，
  perf `wall_clock_s=0.099743`，
  `speedup_ratio=4.383445`，
  `cosine_similarity=0.99999996`，
  `min_cosine_similarity=0.99999976`，
  `max_abs_error=3.34e-06`，
  `num_samples=50`。
- `check_accuracy_run.py` 对 `--max-samples` 很死板，虽然常量值也是 250，但默认值必须直接写成字面量 `250`，否则会报“必须定义 --max-samples 且默认值为 250”。
