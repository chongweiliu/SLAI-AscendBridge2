---
name: diffusers-sdxl-pattern
description: diffusers pipeline（SDXL 系）适配：dry-run 从 config 随机权重组装、diffusers0.39 to() 关键字坑、整管线单卡策略
metadata:
  type: project
---

diffusers 文生图适配（2026-08-22 于 stabilityai/stable-diffusion-xl-base-1.0 验证通过）。

- **dry-run 不下载权重**：`UNet2DConditionModel/AutoencoderKL/EulerDiscreteScheduler` 用
  `.load_config(model_id, subfolder=...)`（只下 config.json），transformers 文本编码器用
  `CLIPTextConfig.from_pretrained(model_id, subfolder="text_encoder")`，
  tokenizer 用 `CLIPTokenizer.from_pretrained(subfolder=...)`，然后手动组装
  `StableDiffusionXLPipeline(vae=..., text_encoder=..., text_encoder_2=..., tokenizer=..., tokenizer_2=..., unet=..., scheduler=...)`。
- **diffusers>=0.39 大坑**：`Pipeline.to()` 关键字参数是 `dtype=`/`device=`，
  `torch_dtype=` 被**静默忽略**（不报错），组件仍是 fp32 → UNet conv2d 报
  `Input type (Half) and bias type (float)`。正解：位置参数 `pipe.to(device, dtype)`
  （新旧版通用），并在 to 后断言 UNet 参数 dtype==fp16 防回归。
- **显存策略**：SDXL fp16 全组件 ~7GB，单卡 64GB 整管线 `.to()` 即可；
  不要 offload / `device_map="balanced"`（Ascend 兼容差）。
- **generator** 固定 `torch.Generator(device="cpu")`。
- SDXL 64x64×2 步（dry）与 512x512×1 步（full，~6.9GB fp16 variant）都能跑；
  1 步图已有可辨识结构（骑手+马+火星色调），可作为真实权重证据。
- 版本栈：torch 2.8.0 + torch-npu 2.8.0.post5 + diffusers 0.39.0 + transformers 4.57.6
  （见 [[dependency-pinning]]、[[qwen3_ascend_stack_verified]]）。

**Why:** diffusers 0.39 的 to() 静默忽略 torch_dtype 是极易漏检的隐错（不报错、运行时才炸）。
**How to apply:** 适配任何 diffusers pipeline（SD3/FLUX/Wan）时按本模式组装 dry-run 与 to() 调用。
