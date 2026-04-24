# ibm-granite/granite-geospatial-biomass

日期：2026-04-22

## 结果

- `optimization_status=completed`
- 物理卡：`13`
- `optimization_kind=runtime_only`
- `speedup_ratio=2.6704`
- `num_samples=50`

## 核心修复

- 旧脚本把 TerraTorch 初始化异常直接退化成 `SimpleConvModel`，导致所谓 pretrained 结果其实不是真实 Granite Geospatial Biomass。
- 真实阻塞点在 `LightningInferenceModel.from_config()`：它会无条件实例化 datamodule，原始训练 config 里的 `GenericNonGeoPixelwiseRegressionDataModule` 在当前环境触发 `jsonargparse` / albumentations 配置错误。
- 解决方式：
  - adaptation 内临时生成 trimmed inference config
  - `data: null`
  - `trainer.logger: null`
  - `trainer.callbacks: []`
  - `trainer.accelerator: cpu`
  - `trainer.devices: 1`
  - 用 `LightningInferenceModel.from_config(trimmed_config, checkpoint)` 只加载 task/model
  - 真正推理时只调用 `inference_model.model`，并确保外层 task / 内层 `model.model` 都 `.eval()`

## 输出结构

- 模型输出为 `terratorch.models.model.ModelOutput`
- 正式张量在 `output.output`
- shape 为 `[B, 224, 224]`

## 正式口径

- baseline：
  - `benchmark_metrics_npu_0_fp32_pretrained_geospatial.json`
  - `wall_clock_s=2.051150`
  - `latency_s=0.041023`
- perf：
  - `benchmark_metrics_npu_0_fp32_pretrained_geospatial_perf.json`
  - `wall_clock_s=0.768106`
  - `latency_s=0.015362`
- compare：
  - `cosine_similarity=1.0`
  - `min_cosine_similarity=1.0`
  - `max_abs_error=0.014448`

## 经验

- 这类 geospatial segmentation / pixelwise regression 模型，即使没有融合算子可替换，runtime-only batching 也可能带来足够大的真实收益。
- `check_accuracy_run.py` 会严格校验 `latency_s * 1000 >= ttft_ms`。若 `ttft_ms` 普通四舍五入到 2 位小数，可能因为舍入放大而比 `latency_s*1000` 大 `0.01ms`，从而误杀；稳妥做法是对 `ttft_ms` 做向下截断。
