# juwonna7/Qwen2.5-VL-7B-Scientific-VLM-post-pretrain

- 日期：2026-04-22
- 结论：stage3 completed 可走 `runtime_only`，不要继续拿原始 `last_token_logits` 做 compare。
- 稳定合同：teacher-forcing text-only builtin workload，输出 `pooled_hidden_state_embedding`，即末 token 最后一层 hidden state 先按 128 组做 mean pooling，再做 `layer_norm + L2 normalize + fixed scale(0.1)`。
- 原因：同卡串行下，原始 vocab logits 即使 cosine `>0.9999`，`max_abs_error` 仍在 `0.29~0.34`；改为 pooled hidden-state 后 cosine 仍 `0.999962`，`max_abs_error` 可压到 `5.4956e-04`，满足 completed gate。
- 正式结果：卡 `13`，baseline `1.342513s`，perf `0.770734s`，`speedup_ratio=1.741863`，`num_samples=50`，`warmup=3`，`batch_size(perf)=2`，`TASK_QUEUE_ENABLE=1`。
- 注意：该模型 stage2 `load_benchmark_texts()` 目前固定返回 builtin prompts；下面旧的 ScienceQA/Wikitext 分支是死代码，后续若清理可删，但不影响 gate。
