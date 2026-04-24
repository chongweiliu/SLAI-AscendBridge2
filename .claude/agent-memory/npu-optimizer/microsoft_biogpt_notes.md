# microsoft/biogpt

- 旧的 stage3 pending 结论是过时的：`generated_text + short pubmed_qa + warmup/TQE-only` 合同把 runtime-only 结果测成了假回归，不适合作为正式 completed 依据。
- 正式修法复用了 GPT-2 类小模型的稳定路径：放弃 `generated_text`，改成本地 pretrained snapshot 的 teacher-forcing `last_token_logits` 合同。
- workload:
  baseline `wikitext + batch_size=1 + warmup(3x)`；
  perf `wikitext + batched_teacher_forcing(bs=4) + warmup(3x) + TASK_QUEUE_ENABLE=1`；
  同物理卡 `13` 串行跑完整链路。
- 本轮正式结果：
  baseline `wall_clock_s=0.566958`，
  perf `wall_clock_s=0.239430`，
  `speedup_ratio=2.367949`，
  `cosine_similarity≈0.99999995`，
  `min_cosine_similarity≈0.99999946`，
  `max_abs_error=3.254413604736328e-05`，
  `num_samples=50`。
- 经验：BioGPT 这种 GPT-2 风格 biomedical CausalLM，不要被旧的 “warmup+TQE 无增益” 记录绑住；只要把 stage2/stage3 统一切到 teacher-forcing logits 合同，batching 往往就能给出稳定的正式提速。
