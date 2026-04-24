---
name: torchair-graph-mode
description: 在华为昇腾上使用 TorchAir 图模式（`torch.compile` + `torchair.get_npu_backend`）进行启用、模式选择、编译配置、日志排障、精度比对、性能分析与自定义算子入图。只要用户提到 TorchAir、图模式、`CompilerConfig`、`reduce-overhead`、`max-autotune`、动态图/静态图、graph dump、`ge_converter`、自定义算子入图、FA 静态下沉、图编译报错或图模式性能问题，就应主动使用这个 skill；不用于纯 Eager patch 性能优化或尚未建立单卡 Eager 基线的任务。
---

# TorchAir 图模式 Skill

本 skill 用于指导在昇腾 NPU 上使用 TorchAir 图模式。重点覆盖 5 类任务：

1. 让模型以 `torch.compile(..., backend=torchair.get_npu_backend(...))` 跑起来
2. 在 `reduce-overhead` 和 `max-autotune` 间做选择
3. 排查入图失败、断图、重复编译、静态图/动态图不符合预期
4. 做精度比对和性能分析
5. 让自定义算子或缺失 Converter 的算子进入图模式

## 先做任务分流

先判断用户属于哪一类需求，再沿对应路径处理：

- **启用图模式**：给出最小可运行示例，检查 import 顺序、版本、单卡约束
- **模式选择**：根据是否更关注 Host 调度开销、动态 shape、图优化能力来选模式
- **排障**：先确认 Eager NPU 跑通，再开日志、dump 图、定位是 Dynamo、TorchAir 还是 GE
- **精度问题**：优先做图模式 vs Eager 的数据 dump 与比对
- **性能问题**：先判断编译慢还是执行慢，再做 profiling
- **自定义算子入图**：检查 Schema、Meta、函数化转换、Converter

如果用户描述不完整，优先补这几个关键信息：

- 目标模型或脚本入口
- 期望使用的模式：未指定时默认按 `max-autotune` 处理
- 问题发生阶段：Eager、compile、第一次执行、重复执行、精度、性能
- 是否涉及自定义算子 / `torch.library` / `torch.ops.*`

## 基线约束

开始给方案前，默认先检查这些前提：

- **先跑通 Eager 模式**：TorchAir 文档明确要求先保证模型在 NPU Eager 模式可运行
- **导包顺序固定**：必须先 `import torch_npu`，再 `import torchair`
- **版本下限**：TorchAir 随 `torch_npu` 发布，建议 `torch_npu >= 7.3.0`，PyTorch 建议 `>= 2.6.0`
- **并行约束**：图模式支持单进程/多进程，但每个进程仅支持 1 张 NPU 卡
- **图模式定位**：当前版本重点面向推理场景

## 使用边界

先明确这个 skill 的适用边界，再决定是否继续走 TorchAir 路线：

- **适合直接使用**：单卡 NPU 推理、已跑通 Eager、需要启用 `torch.compile`、排查图编译报错、补 Converter、做图模式精度/性能分析
- **先不要直接使用**：用户只是想做 `torch_npu` Eager 融合或 `model_files` 优化，这类问题优先转 `torch-npu-optimization` 或 `model-files-override`
- **先补前置条件**：项目还没建立单卡 Eager 基线、入口依赖 `device_map="auto"` / `dispatch_model` 多卡分发、环境里 `torchair`/GE 初始化依赖不完整
- **不要直接承诺结果**：对于大模型 `generate`、MoE、`trust_remote_code=True` 路径，不要在未验证最小 compile probe 和真实入口可编译前承诺“整图成功”或“拿到真实加速比”
- **回退只用于诊断**：`torch._dynamo.config.suppress_errors = True` 只能用来观察哪些路径无法入图，不能把回退到 Eager 的结果当成 TorchAir 图模式加速结论

这些边界已经在真实项目里验证过，可直接当作默认经验：

- **环境侧边界**：部分 venv 里即使 `import torchair` 成功，GE 初始化阶段仍可能缺 Python 侧依赖；已知高频项包括 `setuptools<82`、`decorator`、`scipy`
- **模型侧边界**：Hugging Face MoE `generate` 路径默认视为高风险场景，常见阻塞点包括 `aten.histc.default` 缺 Converter，以及后续 GE `ConcatV2` shape infer 失败
- **测量侧边界**：只有编译对象与真实入口一致、未依赖 `suppress_errors=True` 回退、并且已做最小输出一致性确认时，才把结果记为“真实 TorchAir speedup”

如果项目较大或模型较重，优先先做一个最小 compile probe，再决定是否加载整模：

```python
import torch
import torch_npu
import torchair

config = torchair.CompilerConfig()
backend = torchair.get_npu_backend(compiler_config=config)
fn = torch.compile(lambda x, y: x + y, backend=backend, dynamic=True)

x = torch.randn(2, 2).npu()
y = torch.randn(2, 2).npu()
z = fn(x, y)
```

最小 probe 不通时，先解决环境或 GE 初始化问题；不要直接把大模型失败归因到模型本身。

## 分析现有 adaptation 时必须先检查

如果用户给的是 `adaptations/...` 里的现成项目，不要直接套最小模板。先显式核对这些事实，再给改造建议：

1. **入口脚本是否已经接入 TorchAir**：检查是否已有 `torchair`、`CompilerConfig`、`torch.compile`、`get_npu_backend`
2. **入口脚本是否已经接入 Eager patch**：检查 `demo.py` / `accuracy_run.py` 是否真的 `import` 并调用了 `model_files` 或 `npu_patches` 里的 patch 函数
3. **是否用了 `generate` / cache / MoE / `trust_remote_code=True`**：这些都会提升图捕获和排障复杂度
4. **是否用了 `device_map="auto"`、`infer_auto_device_map`、`dispatch_model`**：这类多设备逻辑要与「每进程仅支持 1 张 NPU 卡」的图模式约束对照说明

如果用户给的是项目化问题，最终回答里**必须明确写出**：

- 这个项目现在到底是 Eager、Eager+patch，还是已经接入 TorchAir
- 入口脚本与 patch 文件是否脱节
- 多卡调度策略是否与 TorchAir 单进程单卡约束冲突

## Transformers / LLM 场景特别注意

如果项目是 Hugging Face `AutoModelForCausalLM`、VLM 或 MoE 大模型，默认增加以下提醒：

- **`model.generate(...)` 不是“天然等于整图 compile 成功”**：先明确 compile 的对象是整模、`forward` 还是某个子模块
- **先稳住 Eager，再试图 compile**：不要在 Eager 还没跑通或 patch 还未接线时就把问题归咎为 TorchAir
- **先单卡验证图模式**：若原项目用了 `device_map="auto"` 或 `dispatch_model`，先给出单卡 POC 路径，再讨论多卡
- **Eager 融合 patch 与 TorchAir 是两层能力**：要明确说明二者是可组合但需要额外排障的关系，不要混写成“已经是图模式”
- **最小代码骨架要能直接映射到当前入口**：如果入口是 `model.generate(...)`，在回答里明确 compile 的对象与 generate 的关系，避免只贴一个脱离项目的 `MyModel()` 模板
- **MoE 路由是高风险区**：涉及 expert routing、`torch.histc`、grouped experts、动态 cache 拼接时，优先预期会遇到 Converter 缺失、断图或 GE shape 推导失败，不要默认按“普通 decoder-only LLM”处理

## 最小启用模板

优先从这个模板起步，再按任务加配置：

```python
import torch
import torch_npu
import torchair

model = MyModel().npu()

config = torchair.CompilerConfig()
# config.mode = "reduce-overhead"   # 不设时默认 max-autotune

npu_backend = torchair.get_npu_backend(compiler_config=config)
opt_model = torch.compile(model, backend=npu_backend, dynamic=True)

x = torch.randn(2, 2).npu()
y = torch.randn(2, 2).npu()
out = opt_model(x, y)
```

如果是在现有 adaptation 项目里给“最小可行骨架”，尽量不要把关键行写成注释占位。至少应显式写出：

- `torch.compile(..., backend=npu_backend)`
- 如果后文讨论了静态下沉或动态 shape，优先直接把 `dynamic=True` 放进骨架
- compile 的对象到底是 `model`、`model.model` 还是某个子模块

如需集合通信入图，可在编译前补：

```python
from torchair import patch_for_hcom

patch_for_hcom()
```

## 模式选择规则

### `max-autotune`

把它当作默认模式。适合：

- 需要更完整的图优化、Ascend IR 能力、图内多流、Tiling 调优、图内 dump
- 需要处理动态 shape、静态/动态混合图
- 需要自定义算子 Converter、GE 图级排障
- 未明确指定模式时

### `reduce-overhead`

把它当作“尽量减少 Host 调度开销”的模式。适合：

- 模型结构稳定，重复执行同一类图
- 更看重 Capture & Replay 带来的调度开销下降
- 希望使用 aclgraph 相关能力

### 选择建议

- 不确定时，先用 `max-autotune`
- 如果问题是“首轮编译后重复执行仍慢，但 shape 较稳定”，再评估 `reduce-overhead`
- 如果任务涉及自定义 GE 构图元素、`custom_op`、图内多流、SuperKernel、Tiling 下沉，优先 `max-autotune`

## 动态图 / 静态图判断

处理 dynamic shape 相关问题时，按文档规则理解：

- `dynamic=False`：输入维度更偏向固定常量，容易得到静态图
- `dynamic=True`：用户输入 tensor 的 shape 默认会符号化
- `torch._dynamo.mark_static(tensor)`：可在 `dynamic=True` 前提下局部静态化

如果用户目标是“包含 FA 算子但仍想整图静态下沉”，按文档给出以下组合：

1. `torch.compile(..., dynamic=True)`
2. 对关键输入做 `torch._dynamo.mark_static(inp)`
3. 打开 `config.experimental_config.tiling_schedule_optimize = True`

如果编译后仍是 GE 动态图，优先检查 dump 图里输入 shape 是否仍含 `-1`。

## 排障顺序

遇到“入图失败 / 断图 / backend 编译失败 / 图结果不对”时，按这个顺序排：

1. **先确认 Eager NPU 成功**
2. **再确认 TorchAir 后端是否接管**
3. **打开 Python 和 C++ 日志**
4. **导出 TorchAir dump 图**
5. **必要时联动 GE dump 图 / plog**
6. **判断是 Dynamo、TorchAir Converter 还是 GE/CANN 阶段**

如果是现有 adaptation 项目，尽量把排障步骤写成**项目内可执行动作**，例如：

- 先跑 `uv run python demo.py --dry-run`
- 再跑一个最小 TorchAir compile probe，确认 `torchair` 与 GE 初始化可用
- 再确认入口脚本是否实际调用了 patch
- 再加 TorchAir backend 与 compile
- 再开日志和 graph dump

不要只说“先确认 Eager，再看日志”，而不把它落到当前项目文件和命令上。

### Python 日志

```python
import logging
from torchair import logger

logger.setLevel(logging.DEBUG)
```

如需看 Dynamo 日志：

```python
import logging
import torch

torch._logging.set_logs(
    dynamo=logging.DEBUG,
    aot=logging.DEBUG,
    output_code=True,
    graph_code=True,
)
```

### C++ 日志

必须在 `import torchair` 前设置：

```python
import os
os.environ["TNG_LOG_LEVEL"] = "0"
```

或在环境里：

```bash
export TNG_LOG_LEVEL=0
```

### 图结构 dump

```python
import torchair

config = torchair.CompilerConfig()
config.debug.graph_dump.type = "pbtxt"
config.debug.graph_dump.path = "./torchair_dump"
```

经验规则：

- 想快速看文本结构，用 `py` 或 `txt`
- 想结合 Netron / TensorBoard 看图，用 `pbtxt`
- 如果是 `reduce-overhead`，要注意 dump 格式能力更受限

## 常见问题映射

优先按错误语义给出处理建议，而不是泛泛重装环境：

- **`ModuleNotFoundError: No module named 'pkg_resources'`**：常见于 `setuptools>=82` 移除了 `pkg_resources`，可优先尝试约束 `setuptools<82`
- **`Failed to initialize GE ... No module named 'decorator'` / `No module named 'scipy'`**：这是 GE Python 侧依赖未补齐，先补环境，再重新跑最小 compile probe
- **`torch.xxx ge_converter is not implemented!`**：缺 Converter，补 `register_fx_node_ge_converter(...)`
- **`unsupported operator`**：通常缺 Meta 推导函数
- **`Failed to converter aten.histc.default to AscendIR`**：MoE/路由统计类算子缺 Converter，默认不要再把“整模 forward compile”当作可立即成立的方案
- **`Found a custom (non-ATen) operator`**：In-place 自定义算子通常缺函数化转换，需要对应 out-of-place 版本和 functionalization
- **`tensor's device must be 'meta'`**：Meta 注册返回的 Tensor 不在 `meta` 设备
- **`op dtype is not same`**：对比 TorchAir dump 图和 GE dump 图，常见需要在 Converter 里显式 `Cast`
- **`Check output size failed`**：Meta 推导结果的 dtype / shape 不对，导致输出缓冲申请不匹配
- **GE `ConcatV2` / shape infer failed**：常见于部分入图后动态 shape、cache 拼接或空 tensor 分支不一致，先缩小 compile 范围或改为局部子模块 POC，再决定是否继续整图
- **固定权重地址后精度异常**：检查 parameter 是否为非连续 Tensor，必要时转 `contiguous()`
- **本地编译失败，提示 `libboundscheck` 下载失败**：优先检查网络权限；离线环境可改为本地 whl
- **`GeneratedDatabase()->Add(...)`**：通常是 TE 包与系统 CANN 版本不匹配

## 自定义算子入图清单

只要任务涉及自定义算子，就按完整链路检查：

1. 确认算子 Schema
2. 实现 NPU 侧算子
3. 先保证 Eager 模式可跑
4. 实现 Meta 推导函数
5. 如果是 In-place 算子，补函数化转换
6. 需要时实现 Converter
7. 再回到图模式验证

处理这类需求时，优先使用这些接口概念：

- `register_fx_node_ge_converter`
- `torchair.ge.Const`
- `torchair.ge.Cast`
- `torchair.ge.Clone`
- `torchair.ge.custom_op`
- `TensorSpec`
- `DataType`

若用户是在做 FX 图级融合而不是单个算子 Converter，可考虑 `register_replacement(...)`。

## 精度比对

如果问题是“图模式结果不准”，优先做**图模式 vs Eager**对比，而不是直接改算子：

1. 先确认 Eager 本身精度正常
2. 获取图模式 dump 数据
3. 获取 Eager / FX dump 数据
4. 用 `msit llm compare` 做差异比对
5. 将整网问题收敛到具体算子或局部结构

如果仓库里已有自己的 benchmark / compare 工具，优先复用本仓能力；`msit` 作为需要细粒度图模式对比时的补充方案。

## 性能分析

如果问题是“图模式慢”，先区分两类：

- **编译慢**：检查是否重复编译，比较多次 FX 图是否变化
- **执行慢**：做 profiling，比较 Eager 和图模式的 Host/Device 开销

分析顺序：

1. 先在脚本中打印关键阶段耗时
2. 再采集 profiling 数据
3. 判断问题更偏前端 / TorchAir / GE / NPU 执行

只有满足下面条件时，才把结果当成“真实 TorchAir speedup”：

- 编译对象与真实业务入口一致，而不是只编译一个脱离主路径的 toy 函数
- 没有依赖 `suppress_errors=True` 回退到 Eager 才跑通
- 输出质量或关键指标已与 Eager 做过最小一致性确认
- 对比双方的 batch、prompt、token 数、warmup 和设备完全一致

如需深挖 profiling，请联动 `ascend-profiling` skill。

## 回答风格要求

在真正回答用户时，不要只复述文档目录。应当：

- 先给可执行结论，再给解释
- 直接给最小代码修改或排障步骤
- 明确当前建议依赖的是哪条文档规律
- 如果用户只问一个报错，就聚焦那个报错，不扩展成整本手册总结

## 参考资料

遇到需要更细节的 API、模式和案例时，再读：

- `references/torchair-guide-notes.md`：按主题整理的文档要点
- `references/torchair-api-cheatsheet.md`：常用 TorchAir API 速查
