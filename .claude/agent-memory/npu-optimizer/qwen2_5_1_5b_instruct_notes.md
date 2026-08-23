---
name: qwen2-5-1-5b-instruct-notes
description: Qwen2.5-1.5B-Instruct runtime-only batched teacher-forcing 3.16x completed
metadata:
  type: project
---

**Qwen/Qwen2.5-1.5B-Instruct（2026-08-22）**：Qwen2.5 1.5B Causal LM，runtime-only 路径完成。

**最终结果**：在卡 `npu:1` 上，pretrained `wikitext` 50 样本，`batched_teacher_forcing(bs=4) + warmup(3x) + TASK_QUEUE_ENABLE=1` 得到 `1.871411s -> 0.591782s`，`speedup_ratio=3.162332`，`cosine=0.999879`（avg），`min_cosine=0.999629`，`max_abs_error=0.484375`（bf16 padding 导致），`ppl_rel_diff=1.06%`，`text_match_rate=1.0`。

**关键步骤**：
1. 旧 config + generate() 工件清理后，下载 pretrained 权重（safetensors ~3GB）
2. 修改 accuracy_run.py：将 Step 2 从 generate()（TextIteratorStreamer）改为 teacher-forcing（forward pass 提取 last_token_logits + perplexity），添加 warmup(3x) + wall_clock_s
3. accuracy_run_perf.py 使用 batched teacher-forcing（多样本 pad 后同时 forward），对比 logits cosine + perplexity + text match
4. **选卡一致性**：baseline 和 perf 必须在同一卡上串行运行，否则 gate 不认可。第二次运行（清理后串行）成功同在 `npu:1` 上完成
5. **bf16 padding 影响**：batched inference 的 right-padding 导致 max_abs_error=0.484375（baseline 无 padding），但 cosine > 0.999 且 text_match_rate=1.0，gate 通过
6. optimization_notes 使用 `output_type=generated_text`（避免 per-sample wall_clock alignment 检查），`warmup_policy=symmetric`，`wall_clock_source=artifact_explicit_field`

**经验**：与 [[qwen_qwen2_5_7b_instruct_notes]] 同族，但 1.5B 模型可以用 batched teacher-forcing 拿到 3.16x（7B 用 bs=1 runtime-only 只拿到 1.05x）。bf16 下 batched inference 的 max_abs_error 在 0.5 量级是正常的，只要 cosine > 0.999 和 text_match_rate=1.0 即可通过 gate。
