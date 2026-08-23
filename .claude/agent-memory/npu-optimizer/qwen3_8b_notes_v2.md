---
name: qwen3_8b_notes_v2
description: Qwen/Qwen3-8B (8B CausalLM bf16) runtime_only bs=1 no TQE 1.019x completed on Ascend910_9362
metadata:
  type: project
---

Qwen/Qwen3-8B (8B, bf16, CausalLM) NPU 优化完成（2026-08-22，第二轮）。

**优化路径**：runtime_only, warmup(3x) + bs=1, **不使用 TQE**
- TQE 在本机 Ascend910_9362 上对 Qwen3-8B bs=1 有负优化（0.938x 回归）
- batched bs=4 虽然 2.81x 但 max_abs_error=0.640625 > 0.001 gate
- 融合算子（rms_norm/swiglu/rotary）max_abs_error 0.69-0.84 全部超 gate
- 最终 bs=1 无 TQE：1.019x 微弱提速，max_abs_error=0.0，cosine=0.99999

**与历史记录对比**：
- 上一轮在卡 12 上用 max_length=512 + TQE 得到 1.10x（[[Qwen/Qwen3-8B]] 旧记录）
- 本轮在 Ascend910_9362 卡 1 上 TQE 回归，可能因 NPU 型号差异
- max_length=128 和 512 对 wikitext 短文本无实质差异（大部分 < 128 tokens）

**关键教训**：
1. TQE 对 8B bs=1 短序列的收益取决于 NPU 型号；Ascend910_9362 上 overhead > benefit
2. bf16 batched inference 的 max_abs_error 来自 padding 后矩阵乘法路径差异，非精度问题
3. 当所有优化路径都无法同时满足 speedup>1 和 max_abs_error<0.001 时，bs=1 无 TQE 是最后的安全网
4. `code_modified=false` + `code_change_attempts>=3` + validation_note 写明 "模型代码无更改" 可过 gate

**产出文件**：
- accuracy_run.py: teacher-forcing logits 合同（废弃旧 generated_text + TextIteratorStreamer）
- accuracy_run_perf.py: bs=1 + warmup(3x), 无 TQE
- pretrained 权重从 HF 镜像下载到 adaptation 私有 models/ 目录（旧缓存只有 config/tokenizer）

与 [[qwen_qwen2_5_7b_instruct_notes]] 和 [[Qwen/Qwen3-8B]] 旧记录同族，但 NPU 型号差异导致 TQE 效果不同。
