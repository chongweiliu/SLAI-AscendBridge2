# Diffusers Benchmark Patterns

## 1. 先确定 benchmark 入口

优先复用 adaptation 里的真实加载方式：

- `demo.py` 若直接跑 pipeline，则 benchmark 也优先跑 pipeline
- `demo.py` 若 dry run 只验证主干 `transformer` / `unet`，benchmark 可按同一主干写 config-only 路径
- 不要为了套模板而把 diffusers benchmark 改写成普通 transformers benchmark

## 2. builtin prompts 优先

`diffusion` / `video` 往往没有仓库内统一数据集模板，优先使用 repo-local 的 builtin prompts：

```python
def load_benchmark_prompts(max_samples: int) -> tuple[list[str], str]:
    prompts = [
        "A sunset over the ocean",
        "A futuristic city at night",
        "A cat playing with yarn",
    ]
    return prompts[:max_samples], "builtin"
```

要求：

- 顺序固定
- 不依赖在线 dataset 下载
- 正式完成样本数仍需 `>= 50`

## 3. Step 2 输出优先保存统计量

为了同时满足 `outputs_*.pt` 产出和跨设备可比性，优先保存 diffusion stats：

```python
all_outputs.append(
    {
        "prompt": prompt,
        "latent_shape": list(final_latent.shape),
        "latent_mean": float(final_latent.mean()),
        "latent_std": float(final_latent.std()),
    }
)

torch.save(
    {
        "diffusion_outputs": all_outputs,
        "latencies": all_latencies,
        "avg_latency": sum(all_latencies) / len(all_latencies),
    },
    outputs_path,
)
```

`benchmark/scripts/benchmark_tool.py compare` 会把这种格式识别为 `diffusion_stats`。

## 4. metrics.output_type 的建议

`benchmark_metrics_*.json` 的 `output_type` 只要能真实描述 workload 即可，常见可写：

- `diffusion_latency`
- `video_latency`
- `generated_images`
- `video_latents`
- `transformer_hidden_states`

如果未来要稳定做 compare，重点不是 `metrics.output_type` 名字，而是 `outputs_*.pt` 保持上面的 `diffusion_outputs` 结构。

## 5. config-only 与 pretrained

diffusers benchmark 仍必须保留两条路径：

- `--use-pretrained`: 正式 baseline
- 默认 config-only: 轻量验证或无权重路径

禁止：

- `--use-pretrained` 失败后 silent fallback 到 config 继续产出正式结果
- 只写 pipeline preload 不做真实前向
- 直接把完整图片 / 视频批量保存到 `outputs_*.pt`

## 6. 常见模式

- 图像模型：保存最终 latent 或 transformer hidden states 的 mean/std
- 视频模型：保存视频 latent 的 shape/mean/std，不保存完整帧序列
- 若整管线太重：Step 1/Step 2 都可改为 benchmark 主干组件，但必须在脚本开头明确“当前 benchmark 的真实工作负载”
