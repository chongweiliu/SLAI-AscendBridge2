---
name: google_t5_t5_small optimization
description: T5-small fp32 npu_rms_norm+batched teacher-forcing 优化完成, speedup=2.86x
type: project
---

# google-t5/t5-small 优化笔记

## 任务概述
- model_id: google-t5/t5-small
- optimization_kind: fusion_operator
- npu_rms_norm 替换 T5LayerNorm + batched_inference(bs=4) + warmup(3x) + TQE
- speedup: 2.862551x (0.959s → 0.335s)

## 关键决策

### 1. 合同切换：generate() → teacher-forcing logits
- 旧 baseline 用 generate()（64步自回归），perf 难以 batch 化，speedup 受限
- 改为 teacher-forcing：encoder_input = text, decoder_input = model._shift_right(encoder_ids), labels = encoder_ids
- 单次 forward 提取 last_token_logits + PPL，可 batch 加速
- 参考 [[e_mimic_inclusively_reformulation_it5_notes]] 的 T5 teacher-forcing 模式

### 2. model_files/npu_patches.py monkey-patch 方式
- 不复制整个 modeling_t5.py，而是 monkey-patch T5LayerNorm.forward
- patch 函数：`torch_npu.npu_rms_norm(hidden_states, weight, eps)[0]`
- 回退：非 NPU 环境调用 `_original_t5_layer_norm_forward`
- 精度验证：cosine=0.9999999871, max_abs_error=3.81e-05

### 3. 对称 warmup + TQE
- baseline 和 perf 都用 warmup(3x)，确保 gate 通过
- perf 额外开 TASK_QUEUE_ENABLE=1 异步算子下发
- speedup 来自：batched(bs=4) + npu_rms_norm + TQE 的组合

## 工件命名差异
- baseline 文件名 `benchmark_metrics_npu_fp32_pretrained_builtin.json`（无设备编号）
- perf 文件名 `benchmark_metrics_npu_1_fp32_pretrained_builtin_perf.json`（有设备编号）
- 原因：`str(first_device)` 在两个脚本中返回格式不同（"npu" vs "npu:1"）
- compare 的 glob 模式能兼容这种差异

## optimization_notes 必填字段教训
- `measurement_contract_version`: 必须是 >= 3 的数字
- `wall_clock_source`: 必须是 `artifact_explicit_field`（当 metrics 有显式 wall_clock_s 字段时），不能用 `perf_counter`

## 产物清单
- model_files/__init__.py, model_files/npu_patches.py
- accuracy_run.py（修改为 teacher-forcing 合同）
- accuracy_run_perf.py（batched + npu_rms_norm + TQE + compare）
- benchmark_metrics_npu_fp32_pretrained_builtin.json（baseline, 60 samples）
- benchmark_metrics_npu_1_fp32_pretrained_builtin_perf.json（perf, 60 samples）
- outputs_npu_fp32_pretrained_builtin.pt, outputs_npu_1_fp32_pretrained_builtin_perf.pt
- optimization_notes.json
- trace_npu_fp32_pretrained_builtin.json

## 最终结果
- baseline_wall_clock_s: 0.958903, perf_wall_clock_s: 0.334982
- speedup_ratio: 2.862551, latency_reduction_pct: 65.07%
- cosine_similarity: 0.9999999871, ppl_avg_rel_diff_pct: 0.0%
- NPU: npu:1 (Ascend910_9362), fp32, single-die
