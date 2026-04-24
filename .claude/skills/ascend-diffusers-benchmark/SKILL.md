---
name: ascend-diffusers-benchmark
description: 为 diffusers / FLUX / Stable Diffusion / Wan 等图像与视频生成模型编写和执行 benchmark `accuracy_run.py`。处理 diffusion/video 的手写评测脚本、builtin prompts、latency-only workload、latent 统计输出与 trace/metrics 产物。触发词：diffusers benchmark、扩散模型评测、视频生成评测、FLUX benchmark、Wan benchmark、latency-only benchmark、builtin prompts、diffusion stats。
---

# Ascend Diffusers Benchmark Skill

本 skill 用于 **diffusers 图像/视频生成模型** 的 benchmark。它覆盖 `model_type in {"diffusion", "video"}` 的 `accuracy_run.py` 手写实现。

**与 benchmark-script 的关系**：`benchmark-script` 负责标准 transformers 模板路线；本 skill 负责 **diffusers / video 的手写 benchmark 路线**。遇到 `diffusion`、`video`、`FluxPipeline`、`WanPipeline`、`StableDiffusion*Pipeline` 时，不要从 Jinja2 模板起步。

## 什么时候用

出现以下任一情况时，优先使用本 skill：

- `scripts/dataset_mapping.py` 返回 `model_type=diffusion` 或 `model_type=video`
- benchmark 对象通过 `diffusers.Pipeline` 或 `*Pipeline.from_pretrained(...)` 加载
- 评测目标是图像生成、视频生成、latent 路径或 transformer/unet 主干
- 数据集更适合 **builtin prompts / synthetic inputs**，而不是标准 NLP/vision dataset 模板
- 需要 latency-only benchmark，但仍要产出可追溯的 `outputs_*.pt`

## 边界

- 仍遵守 benchmark-runner 的目录边界：副作用只允许落在当前 `adaptation_path`
- 只创建 / 修改 `accuracy_run.py` 及 benchmark 产物；**不要**创建 `model_files/` 或 `accuracy_run_perf.py`
- `cache_dir` 固定到 `adaptation_path/models/`
- 生成或修改后，**必须**运行 `uv run python benchmark/scripts/check_accuracy_run.py --adapt {sanitized_name}`

## 核心路线

1. **先看 adaptation 实现**
   - 优先复用 `demo.py` 的导入、pipeline 组装、设备选择和缓存路径
   - 真实 benchmark 路径必须与 adaptation 主路径一致，避免 benchmark 另起一套加载逻辑
2. **区分 benchmark 负载**
   - 图像模型：优先 text-to-image 或 transformer/unet 主干
   - 视频模型：优先 text-to-video 或 transformer 主干
   - 当整管线过重时，允许 benchmark 主干 latent 路径，但要在脚本头部与 metrics 中写清
3. **保留两步流程**
   - Step 1: trace + benchmark_metrics
   - Step 2: outputs
4. **输出用统计量，不保存大媒体**
   - 不要把完整图片 / 视频 tensor 大量落盘
   - 优先保存可比较的 latent 统计量，格式见 [references/benchmark-patterns.md](references/benchmark-patterns.md)
5. **再做 gate 校验**
   - `check_accuracy_run.py --adapt ...`
   - 样本数正式完成仍要满足 `num_samples >= 50`

## 数据与输出约定

- `dataset_mapping.py` 对 `diffusion` / `video` 给的是 **evaluation profile hint**，不是标准模板承诺
- 默认优先用 `builtin prompts`
- `benchmark_metrics_*.json` 中的 `output_type` 应反映真实 workload，如 `diffusion_latency`、`video_latency`、`generated_images`、`video_latents`
- 若后续需要 `benchmark_tool.py compare` 做跨设备输出对比，优先把 `outputs_*.pt` 写成 `diffusion_stats` 友好格式：

```python
{
  "diffusion_outputs": [
    {
      "prompt": "...",
      "latent_shape": [1, 4, 64, 64],
      "latent_mean": 0.0123,
      "latent_std": 0.9876,
    }
  ],
  "latencies": [0.91, 0.88],
  "avg_latency": 0.895,
}
```

## 手写 benchmark 时必须保留

- `--use-pretrained` 分支与 `from_config` / config-only 分支
- `--max-samples`，默认 `250`
- `get_dtype_str(next(model.parameters()).dtype)` 或等价的实际 dtype 推导
- `CACHE_DIR = Path(__file__).resolve().parent / "models"`
- 固定随机种子 + `torch.use_deterministic_algorithms(True, warn_only=True)`
- Step 1 导出 `trace_*.json` 与 `benchmark_metrics_*.json`
- Step 2 导出 `outputs_*.pt`
- 设备检测与清缓存逻辑

## 何时不要走本 skill

- 普通 `transformers` 文本 / 图像分类 / ASR / seq2seq benchmark：回到 `benchmark-script`
- 需要 NPU 优化版 benchmark：那是 `npu-optimizer` 与 `accuracy_run_perf.py` 的职责
- 模型并非 diffusers pipeline，而只是普通 PyTorch 模块：按原 benchmark 路线处理

## 参考文件

- [references/benchmark-patterns.md](references/benchmark-patterns.md): diffusers/video benchmark 的脚本骨架与输出格式
