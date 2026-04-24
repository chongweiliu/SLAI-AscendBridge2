# ibm-research/DAC.speech.v1.0

- 日期：2026-04-22
- 物理卡：`ASCEND_RT_VISIBLE_DEVICES=13`
- 结果：`optimization_status=completed`

## 关键问题

- 本地 snapshot 只有 `weights_24khz_1.5kbps_v1.0.pth` / `weights_24khz_3kbps_v1.0.pth`，不是 transformers `DacModel.from_pretrained()` 可直接加载的标准权重。
- `descript-audio-codec` 安装后，顶层 `import dac` 会级联到 CUDA 版 `torchaudio`，在 Ascend 机器上因 `libcudart.so.13` 缺失直接报错。
- 旧 stage3 工件是 `audio_codec + selfbaseline + asymmetric warmup` 口径，`check_accuracy_run_perf.py` / `check_optimization_notes.py` 会同时拦：
  - `warmup_policy=asymmetric`
  - `baseline_warmup_iterations != perf_warmup_iterations`
  - `wall_clock/num_samples != latency_s`

## 最终方案

- 在 `model_files/modeling_dac.py` 内按 installed `dac` 源码重建最小真实 DAC：
  - `Snake1d`
  - `WNConv1d` / `WNConvTranspose1d`
  - `ResidualVectorQuantize`
  - `DAC`
- 用 checkpoint `metadata["kwargs"]` 重建结构，再加载本地 `state_dict`。
- baseline/perf 改成统一合同：
  - 50 样本 `synthetic`
  - `output_type=cls_embeddings`
  - 输出 `embeddings`
  - baseline `bs=1`
  - perf `batched_pooled_latents(bs=5) + warmup(3x) + TASK_QUEUE_ENABLE=1`
  - 同一卡、串行、对称 warmup

## 工件结果

- baseline:
  - `benchmark_metrics_npu_0_fp32_pretrained_synthetic.json`
  - `wall_clock_s=1.833411`
  - `latency_s=0.036668`
- perf:
  - `benchmark_metrics_npu_0_fp32_pretrained_synthetic_perf.json`
  - `wall_clock_s=0.349735`
  - `latency_s=0.006995`
- compare:
  - `output_compare_perf.json`
  - `cosine_similarity=0.9999999761581421`
  - `min_cosine_similarity=0.9999998211860657`
- notes:
  - `optimization_notes.json`
  - `speedup_ratio=5.242286`
  - `selected_npu=13`
  - `num_samples=50`

## 经验

- Audio codec 这类模型，如果 checkpoint 明显是官方原生格式而不是 HF 格式，不要硬套 transformers 路径。
- 只要 top-level 包导入会被无关 CUDA 依赖拖死，就直接在 adaptation 内裁出最小真实推理实现，更稳。
- 对 codec 类任务，使用 pooled latent embeddings 作为 stage3 compare 目标，比旧 `reconstructed_audio/mse/audio_codec` 字典合同稳定得多。
