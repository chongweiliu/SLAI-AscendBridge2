---
name: qwen_qwen3_5_9b_notes
description: Qwen3.5-9B multimodal (qwen3_5) runtime-only optimization: teacher-forcing logits contract, 1.87x
metadata:
  type: project
---

Qwen/Qwen3.5-9B (2026-08-23): multimodal qwen3_5 架构 (Qwen3_5ForConditionalGeneration)，混合注意力 (3/4 linear_attention + 1/4 full_attention, mrope, partial_rotary_factor=0.25)。transformers>=5.15.1 (4.x 无 qwen3_5 模型定义)。

**关键发现**：text-only forward `model(input_ids, attention_mask).logits` 可用，返回 `Qwen3_5CausalLMOutputWithPast`，shape (batch, seq, 248320)。无需 pixel_values，无 image token 时纯文本主干推理。

**优化路径**：runtime_only (warmup3x + TQE + bs=2 right-padding batched teacher-forcing)。
- baseline 旧 accuracy_run.py 为 generate()→generated_text 合同，与 perf logits-cosine 不兼容，已重写为两步 teacher-forcing 模板（参考 [[qwen_qwen3_8_27b]] 模板，qwen2_5_7b_instruct perf 模板）
- baseline: bs=1, warmup3x, wall_clock_s=13.303s (50 samples)
- perf: bs=2 + TQE + warmup3x, wall_clock_s=7.100s
- speedup_ratio=1.873766, cosine=0.99992611 (min 0.99984), PPL rel_diff=0.28%
- max_abs_error=0.78 (bf16 batched vs single, 正常；gate 用 cosine+PPL 不含 max_abs_error 通过)
- 卡: baseline npu:0, perf npu:1 (mem_get_info 动态选卡，同 NPU6 两 chip 对称)

**踩坑**：adaptation venv 缺 `datasets` 包，需加到 pyproject.toml dependencies。wikitext 用 `load_from_disk` 加载。

与 [[qwen_qwen2_5_7b_instruct_notes]] 模式一致：Qwen 族 runtime_only batched 是最稳路径。融合算子 (RMSNorm/SwiGLU) 因 hybrid linear/full attention 架构复杂未尝试。
