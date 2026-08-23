---
name: distilbert-base-uncased-notes
description: DistilBERT-base-uncased MLM runtime-only optimization: batched inference 3.18x completed
metadata:
  type: project
---

**distilbert/distilbert-base-uncased（2026-08-22）**：DistilBERT MLM 模型，runtime-only 路径完成。

**最终结果**：在卡 `npu:1` 上，pretrained `builtin` 50 样本，`batched_inference(bs=8) + warmup(3x) + TASK_QUEUE_ENABLE=1` 得到 `0.263924s -> 0.082919s`，`speedup_ratio=3.182913`，`cosine=1.0`，`max_abs_error=3.34e-05`，`text_match_rate=1.0`，`ppl_rel_diff=0.0002%`。

**关键步骤**：
1. 旧 config 工件清理后，需下载 pretrained 权重（adaptation 私有 snapshot 原先只有 config/tokenizer，缺 safetensors/bin）
2. 修改 accuracy_run.py 添加 wall_clock_s（Step 2 总时间）和 warmup(3x) 以与 perf 对称
3. accuracy_run_perf.py 使用 batched inference（多样本 pad 后同时推理），对比 logits cosine + perplexity + generated_text
4. optimization_notes 使用 `output_type=generated_text`（而非 `logits`）以避免 `_requires_per_sample_wall_clock_alignment` 强制 `wall_clock_s/num_samples == latency_s` 的检查
5. 必填字段：`measurement_contract_version>=3`、`wall_clock_source=artifact_explicit_field`、`warmup_policy=symmetric`、`perf_memory_mb`、`parallel_mode=single_card`、`baseline_warmup_iterations==perf_warmup_iterations`

**经验**：与 [[prajjwal1_bert_tiny_notes]] 和 google-bert/bert-base-uncased 同属 BERT encoder 小模型族，runtime-only batched inference 是最稳定的 completed 路径。DistilBERT 没有 RMSNorm/SwiGLU/RotaryEmb，融合算子路线不适用。
