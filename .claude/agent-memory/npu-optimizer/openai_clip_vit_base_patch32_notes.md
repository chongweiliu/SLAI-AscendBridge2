---
name: openai_clip_vit_base_patch32 optimization
description: CLIP-ViT-B/patch32 fp32 runtime-only all-texts-one-batch 优化完成, speedup=3.76x
type: project
---

# openai/clip-vit-base-patch32 优化笔记

## 任务概述
- model_id: openai/clip-vit-base-patch32
- optimization_kind: runtime_only
- all_texts_one_batch(60) + warmup(3x) + TASK_QUEUE_ENABLE=1
- speedup: 3.761095x (0.069s → 0.018s)

## 关键决策

### 1. 一次性处理所有文本 vs 分块
- baseline 分 4 块（每块 16 文本），每块都重新编码图像
- perf 一次性处理所有 60 文本，图像只编码 1 次（从 4→1）
- 主要加速来自减少图像 ViT 前向次数

### 2. 与 CLIP-ViT-B/16 的差异
- patch16 的 image_embeddings 合同需要 bs=1 + TQE（batched 会 max_abs_error 不过）
- patch32 的 image_text_similarity 合同可以安全 batched（text encoder 的 padding 不影响精度）
- cosine=1.0, max_abs_error=2.098e-05

### 3. 从 config → pretrained
- 旧 baseline 是 config 模式（随机权重），相似度无意义
- pretrained 下 top1="a solid red square"（正确匹配合成红色方块图像）

## 产物清单
- accuracy_run.py（加 warmup(3x) + wall_clock_s）
- accuracy_run_perf.py（all texts one batch + warmup(3x) + TQE + compare）
- benchmark_metrics_npu_fp32_pretrained_builtin.json（baseline）
- benchmark_metrics_npu_fp32_pretrained_builtin_perf.json（perf）
- outputs_npu_fp32_pretrained_builtin.pt, outputs_npu_fp32_pretrained_builtin_perf.pt
- optimization_notes.json

## 最终结果
- baseline_wall_clock_s: 0.068561, perf_wall_clock_s: 0.018229
- speedup_ratio: 3.761095, latency_reduction_pct: 73.41%
- cosine_similarity: 1.0, max_abs_error: 2.098e-05
- top1: "a solid red square" (both baseline and perf, correct match)
- NPU: npu:1 (Ascend910), fp32, single-die, single_card
- dataset: builtin (60 texts × 1 synthetic image), output_type: image_text_similarity
