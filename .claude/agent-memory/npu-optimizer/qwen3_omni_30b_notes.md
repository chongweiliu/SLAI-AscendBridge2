---
name: qwen3_omni_30b_captioner_failure
description: Qwen3OmniMoeForConditionalGeneration forward API不兼容，无法验证fusion ops优化
type: project
---

# Qwen/Qwen3-Omni-30B-A3B-Captioner 优化记录

## 模型信息
- **模型**: Qwen/Qwen3-Omni-30B-A3B-Captioner
- **架构**: Qwen3OmniMoeForConditionalGeneration (MoE, 128 experts, 8 active)
- **dtype**: bf16
- **问题**: forward() API 不接受 input_ids，需要 hidden_states

## 核心问题

### 1. forward() API 不兼容
- `Qwen3OmniMoeForConditionalGeneration.forward()` 签名：
  ```python
  def forward(hidden_states, cu_seqlens, rotary_pos_emb, ...)
  ```
- 不接受 `input_ids`，与标准 benchmark 范式不兼容
- `model(**inputs)` 报错：`TypeError: _forward_unimplemented() got an unexpected keyword argument 'input_ids'`

### 2. model.generate() 不返回 logits
- `model.generate()` 接受 `input_ids` 和 `thinker_max_new_tokens`/`talker_max_new_tokens` 参数
- 返回 `GenerateOutput(tensor)` 即 token IDs，不含 logits
- 无法计算 cosine similarity 和 PPL

### 3. 无法验证 fusion ops 优化
- `model_files/` 已创建 npu_rms_norm + npu_swiglu 代码
- 但因无法获取 logits，无法验证优化效果

## 尝试的修复

1. **修改 accuracy_run_perf.py step1**:
   - 将 `model(**inputs)` 改为 `model.generate(input_ids=..., thinker_max_new_tokens=1, talker_max_new_tokens=0, ...)`
   - 成功，但只返回 token IDs

2. **修改 warmup 函数**:
   - 将 `model(input_ids).logits` 改为使用 `model.generate()`

3. **修改 step2**:
   - 无法从 generate() 获取 logits，只能记录 generated_text

## 运行结果

- **warmup + TASK_QUEUE_ENABLE**: 19.7s latency (step1, 1 sample)
- **baseline**: 20.7s latency (step1, 1 sample, no warmup)
- **speedup**: 1.05x (但 warmup 不对称，无实际意义)

## 结论

**not_applicable**: 模型 API 设计不适合标准 benchmark 范式

- `model.forward()` 不接受 input_ids
- `model.generate()` 不返回 logits
- 无法计算 cosine similarity 和 PPL
- 无法验证 fusion ops 优化效果

## 经验教训

1. **MoE 模型需要特殊处理**: Qwen3OmniMoe 的 forward() 签名是针对 MoE 架构设计的，不适合标准 text generation benchmark

2. **generate() 不等于 forward()**: 某些模型的 generate() 实现不返回中间 logits，只能得到最终 token IDs

3. **warmup_policy 必须对称**: check_optimization_notes.py 要求 baseline_warmup_iterations == perf_warmup_iterations，且 warmup_policy="symmetric"

4. **cosine_similarity 不能为 null**: 即使是 runtime_only 优化，也需要提供有效的 cosine_similarity 值

## 相关主题

- `advanced-topics.md`: API 适用性分析
- `fusion-attention-pitfalls.md`: npu_fusion_attention 踩坑记录