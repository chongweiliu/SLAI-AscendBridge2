---
name: ascend-diffusers-adaptation
description: 将 diffusers / FLUX / Stable Diffusion / Wan 等图像与视频生成 pipeline 适配到 Ascend NPU。处理 pipeline 组件拆分、offload、dispatch_model、VAE 跨设备、complex64 RoPE 替换与非标准 diffusers 包装。触发词：diffusers 适配、FLUX 迁移、Stable Diffusion 迁移、Wan 适配、text-to-image、image-to-video、text-to-video、pipeline migration、offload、dispatch_model。
---

# Ascend Diffusers Adaptation Skill

本 skill 用于处理 **diffusers pipeline** 的 Ascend 适配。它覆盖 `FLUX`、`Stable Diffusion`、`Wan` 以及类似的图像/视频生成模型。

**与 ascend-adaptation 的关系**：`ascend-adaptation` 负责通用 PyTorch / transformers 适配；本 skill 负责 **diffusers pipeline 结构、显存策略与常见兼容性坑**。

## 什么时候用

出现以下特征时，优先使用本 skill：

- 模型通过 `diffusers.Pipeline` 或 `*Pipeline.from_pretrained(...)` 加载
- 任务是 `text-to-image`、`image-to-image`、`image-to-video`、`text-to-video`
- 需要拆分 `text_encoder / transformer(or unet) / vae / scheduler`
- 需要 `offload`、`dispatch_model()`、`infer_auto_device_map()`
- 需要处理 `VAE decode` 跨设备
- 需要修复 `complex64` / `torch.polar` / `view_as_complex`

## 仓库边界

- 仍然遵守本仓库 adaptation 目录边界：所有缓存与副作用只允许落在当前 `adaptation_path`
- `demo.py`、`pyproject.toml`、`README.md`、`.status.json` 仍由 adapter 负责
- **不要**创建 `model_files/` 或 `accuracy_run_perf.py`；那是 `npu-optimizer` 的职责
- **不要**直接改 site-packages；适配期只在 adaptation 本地脚本中做 patch / wrapper

## 先做分流

开始前先判断模型属于哪种包装：

1. **标准 diffusers 多目录结构**
   - 常见于 `transformer/`, `unet/`, `vae/`, `scheduler/`, `tokenizer/`
   - 这是本 skill 的主路径
2. **单文件量化或特殊发布格式**
   - 如 NVFP4、缺少标准 `config.json` / 组件目录
   - 先确认是否仍能走 diffusers 正常加载；不能则按“非标准包装”处理
3. **需要额外源码或自定义仓库**
   - 若模型卡说明必须 clone 外部 repo，再回到通用 custom repo 流程

如果只是普通 `transformers.AutoModel*`，退出本 skill，回到 `ascend-adaptation`。

## 核心流程

1. **识别 pipeline 组件**
   - 先确认 pipeline 类、主要计算组件、权重体量与 dtype
   - 详细步骤见 [references/migration-workflow.md](references/migration-workflow.md)
2. **选择显存策略**
   - 在整管线 `.to("npu")`、`model-offload`、`sequential-offload`、多 NPU dispatch 间做选择
   - 选择规则见 [references/memory-strategies.md](references/memory-strategies.md)
3. **实现 adaptation-local 入口**
   - `demo.py --dry-run` 走最小可验证路径
   - `demo.py` full run 走真实 pipeline / 真实权重路径
   - 缓存目录固定为 `adaptation_path/models/`
4. **修兼容性问题**
   - 优先处理设备不一致、complex64、`device_map="balanced"`、generator 设备问题
   - 常见修复见 [references/common-fixes.md](references/common-fixes.md)
5. **回到仓库验证链路**
   - 跑通 `uv run python demo.py --dry-run`
   - 产出 `output.txt`
   - 用项目内 `check_adaptation.py` 验证

## Dry Run 原则

diffusers 模型的 dry run 不要求完整生成高质量图片/视频，但必须满足：

- 真正走到 NPU 或 CUDA 上的主要代码路径
- 至少覆盖主干组件的实例化与一次最小前向
- 不允许静默退回 CPU
- 不允许只打印配置而没有任何模型执行

如果整条 pipeline 的随机权重 dry run 代价太高，可在 `demo.py --dry-run` 中：

- 只实例化主干组件（如 `transformer` / `unet` / `WanTransformer3DModel`）
- 构造最小 latent / hidden states 做一次前向
- 明确在日志中说明 dry run 覆盖的是哪条主路径

## 非标准包装

遇到以下情况，不要硬套标准 diffusers 路线：

- 单文件量化格式，不具备标准 pipeline 目录结构
- 模型卡明确依赖 NVIDIA-only 特性
- 缺失关键组件配置，无法构造标准 `Pipeline.from_pretrained`

这类情况应在 adaptation 结果里明确写清：

- 为什么不是标准 diffusers 适配路径
- 现有证据是什么
- 下一步是改走自定义流程、补仓库信息，还是继续验证标准路径

**状态口径限制**：

- “非标准包装” 只表示 **先退出标准 diffusers 路线**，不等于可以直接标 `not_applicable`
- 只有命中 adapter 的格式预检规则，或已有明确证据证明该发布格式 / 平台依赖确实不适用于本仓库时，才允许走 `not_applicable`
- 仅因模型是图像生成、视频生成、FLUX、Stable Diffusion，或采用单文件 / 特殊包装，**都不足以**单独判 `not_applicable`

## 参考文件

- [references/migration-workflow.md](references/migration-workflow.md): pipeline 组件分析与 adaptation 实现骨架
- [references/memory-strategies.md](references/memory-strategies.md): `.to("npu")` / offload / dispatch 选择规则
- [references/common-fixes.md](references/common-fixes.md): VAE 跨设备、complex64、balanced device_map 等常见修复
