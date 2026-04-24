---
name: ascend-diffusers-optimization
description: 昇腾 NPU 上 diffusers / FLUX / Stable Diffusion / Wan 图像与视频生成 pipeline 的推理优化。处理 monkey-patch、DIFFUSERS_ATTN_BACKEND=_native_npu、non-causal attention、env tuning、offload 与通信瓶颈分析。触发词：diffusers 推理优化、FLUX 优化、Stable Diffusion 优化、Wan 优化、图像生成加速、视频生成加速、monkey-patch、DIFFUSERS_ATTN_BACKEND、non-causal attention。
---

# Ascend Diffusers Optimization Skill

本 skill 用于 **diffusers pipeline** 的推理优化。目标是让已适配、已建立 benchmark 基线的图像/视频生成模型在 Ascend 上跑得更快。

**与 torch-npu-optimization 的关系**：`torch-npu-optimization` 主要面向普通 PyTorch / transformers `modeling_*.py` 路线；本 skill 处理 **diffusers pipeline 在 site-packages 中、需要 monkey-patch、非因果 attention 与 offload/通信瓶颈** 的情况。

## 什么时候用

出现以下特征时，优先使用本 skill：

- 优化对象是 `diffusers` pipeline
- 代码热点主要在 `diffusers.models.*`，不是 adaptation 本地 `modeling_*.py`
- 模型是 `FLUX`、`Stable Diffusion`、`Wan` 或同类图像/视频生成模型
- 需要启用 `DIFFUSERS_ATTN_BACKEND=_native_npu`
- attention 是 **non-causal**，不能照搬 LLM causal mask 方案
- 性能瓶颈可能来自 offload 或多卡通信，而不是单个融合算子

## 仓库边界

- 仍遵守本仓库 optimization 规则：`accuracy_run_perf.py` 与 `model_files/` 仅由 `npu-optimizer` 创建
- 不要直接改 site-packages；优先用 `adaptation_path/model_files/` 下的 adaptation-local patch 模块
- diffusers monkey-patch 默认落在 `adaptation_path/model_files/`，例如 `model_files/npu_patches.py`
- `accuracy_run_perf.py` 应显式从 `model_files/` 导入并应用 patch；不要把 patch 散落在项目根、site-packages 或仓库外临时文件
- 只有在 `npu-optimizer` 已明确选择“克隆源码直接改写”路径时，才允许改 `adaptation_path/<custom_repo>/...`
- baseline / perf 仍必须使用 **pretrained** 权重做正式对比
- baseline / perf 要复用同一组卡和同一并行模式

## 先做分层判断

按成本从低到高判断优化层级：

1. **runtime-only**
   - `TASK_QUEUE_ENABLE`
   - allocator / async 配置
   - warmup 与稳定测量
2. **backend 切换**
   - `DIFFUSERS_ATTN_BACKEND=_native_npu`
3. **monkey-patch 融合算子**
   - `RMSNorm`、`GELU`、`SwiGLU`
4. **结构性瓶颈判断**
   - 若主要瓶颈是 offload 或跨卡通信，不要把时间耗在无效的小算子替换上

## 核心流程

1. **先确认 benchmark 基线**
   - adaptation 与 benchmark 已完成
   - pretrained baseline 能稳定复现
2. **优先做低侵入优化**
   - 先尝试 runtime-only 与 backend 配置
3. **只在有理由时做 monkey-patch**
   - 适用性、精度与收益都要自证
   - 详细模式见 [references/optimization-playbook.md](references/optimization-playbook.md)
4. **评估是否被通信 / offload 主导**
   - 大模型多卡 diffusers 常常不是算子融合主导
   - 典型案例见 [references/case-notes-flux-wan.md](references/case-notes-flux-wan.md)
5. **不要盲目追求“全都 patch”**
   - 某些模型中融合算子收益极小，甚至被额外 `cat` / 搬运开销抵消
   - 反例见 [references/negative-applicability-zimage.md](references/negative-applicability-zimage.md)

## diffusers 特有注意点

- 非因果 attention 不能直接套 LLM causal mask 方案
- `DIFFUSERS_ATTN_BACKEND` 要在相关导包或构图前设置
- 如果代码在 site-packages 中，优先 monkey-patch 类方法，并把 patch 放到 `adaptation_path/model_files/`，不要直接编辑安装包源码
- 多卡大模型上，per-step 小幅提速不一定能反映成端到端大提速

## 何时停止加 patch

满足以下任一情况时，应该停止继续堆融合算子：

- 主要耗时来自 CPU<->NPU offload
- 主要耗时来自多卡通信
- 模型原本就已使用 diffusers / torch_npu 的高效实现
- 新 patch 引入额外 `cat` / reshape / device move，抵消了融合收益
- 精度风险已经上升，但速度没有实质改进

## 参考文件

- [references/optimization-playbook.md](references/optimization-playbook.md): runtime-only、backend 与 monkey-patch 的选择顺序
- [references/case-notes-flux-wan.md](references/case-notes-flux-wan.md): FLUX / Wan 的经验归纳
- [references/negative-applicability-zimage.md](references/negative-applicability-zimage.md): 什么时候不要强推融合算子
