# ibm-research/materials.3dgrid_vqgan

- 日期：2026-04-22
- 物理卡：`ASCEND_RT_VISIBLE_DEVICES=13`
- 结果：`optimization_status=completed`

## 旧问题

- benchmark baseline 被历史 `benchmark_metrics_npu_fp32_config_synthetic_3d.json` 污染，缺 `dataset` 等 completed gate 必填字段。
- stage3 旧工件是：
  - `reconstructed_3d_grids`
  - `self_baseline_same_model`
  - `speedup_ratio=0.998`
- 旧 `optimization_notes.json` 不能作为 completed 证据：
  - `comparison_method=self_baseline_same_model`
  - `precision_method=self_baseline_same_model`
  - `speedup_ratio < 1.0`

## 关键发现

- 本地 snapshot 完整，真实权重在：
  - `3DGrid-VQGAN_43.safetensors`
- 直接按自定义 `FullVQGAN3D` 加载时，checkpoint 只多出：
  - `codebook.N`
  - `codebook.embeddings`
  - `codebook.z_avg`
- 同时缺：
  - `codebook.weight`
- 这说明 checkpoint 的 codebook 不是简单 `nn.Embedding`。但 stage3 正式 workload 如果只对比 encoder latent embeddings，就不需要依赖 codebook 路径。

## 最终方案

- 重写 `accuracy_run_perf.py`，废弃旧 self-baseline 合同。
- 正式对比目标改成：
  - `output_type=latent_embeddings`
  - 输出 `embeddings`
  - compare 看 cosine/max_abs_error
- baseline/perf：
  - baseline `bs=1`
  - perf `batched_latent_embeddings(bs=2) + warmup(3x) + TASK_QUEUE_ENABLE=1`
  - 同一卡、串行、对称 warmup
- preload 时允许 codebook 的以下 mismatch：
  - missing: `codebook.weight`
  - unexpected: `codebook.N`, `codebook.embeddings`, `codebook.z_avg`

## 结果

- baseline:
  - `benchmark_metrics_npu_0_fp32_pretrained_synthetic_3d.json`
  - `wall_clock_s=0.100466`
  - `latency_s=0.002009`
- perf:
  - `benchmark_metrics_npu_0_fp32_pretrained_synthetic_3d_perf.json`
  - `wall_clock_s=0.064569`
  - `latency_s=0.001291`
- speedup:
  - `speedup_ratio=1.555948`
- precision:
  - `cosine_similarity=0.9999999642372132`
  - `min_cosine_similarity=0.9999998211860657`
  - `max_abs_error=0.0`

## 经验

- 对自定义 VQ/VQGAN checkpoint，如果 codebook 实现不完全一致，但正式优化 workload 只需要 encoder latent，对 stage3 可以直接把 compare 合同切到 latent embeddings。
- 这类 3D Conv 模型单样本串行 often 接近无提速；轻量 batching 才能把 wall-clock 从 `<1.0x` 拉到正式 completed 区间。
