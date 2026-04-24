# TorchAir Guide Notes

基于 `docs/Ascend Extension for PyTorch 7.3.0 图模式使用指南(TorchAir) 01.pdf` 提炼，供 skill 在需要细节时再读。

## 1. 产品定位与约束

- TorchAir 是 `torch_npu` 的图模式扩展库，不是独立安装包
- 当前版本重点面向**推理场景**
- 需要先掌握 Ascend Extension for PyTorch 基础知识
- 每个进程仅支持 1 张 NPU 卡
- 建议 `torch_npu >= 7.3.0`，PyTorch `>= 2.6.0`

## 2. 最小可用流程

```python
import torch
import torch_npu
import torchair

config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)
model = torch.compile(model, backend=npu_backend)
```

关键约束：

- 必须先 `import torch_npu`，再 `import torchair`
- 图模式前先保证模型在 NPU Eager 模式成功
- `config.mode` 默认是 `max-autotune`

## 3. 两种模式

### `reduce-overhead`

- 也叫 aclgraph / 捕获模式
- 通过 Capture & Replay 降低 Host 调度开销
- 更偏重复执行、稳定图结构场景

### `max-autotune`

- 也叫 Ascend IR 模式
- 将 FX 图转为 Ascend IR，再由 GE 编译执行
- 支持更多图优化、动态 shape、图内多流、SuperKernel、Tiling 相关能力

## 4. `CompilerConfig` 主要分组

- `debug`
- `export`
- `dump_config`
- `fusion_config`
- `experimental_config`
- `inference_config`
- `ge_config`
- `aclgraph_config`
- `mode`

## 5. 常用调试手段

### Python 日志

```python
import logging
from torchair import logger

logger.setLevel(logging.DEBUG)
```

### Dynamo 日志

```python
torch._logging.set_logs(
    dynamo=logging.DEBUG,
    aot=logging.DEBUG,
    output_code=True,
    graph_code=True,
)
```

### C++ 日志

必须早于 `import torchair`：

```python
import os
os.environ["TNG_LOG_LEVEL"] = "0"
```

### 图结构 dump

```python
config.debug.graph_dump.type = "pbtxt"
config.debug.graph_dump.path = "./dump_dir"
```

可选格式：

- `py`
- `txt`
- `pbtxt`

## 6. 动态图 / 静态图

### Dynamo 侧

- `dynamic=False`：输入更偏固定常量
- `dynamic=True`：用户输入 tensor 的 shape 默认符号化
- `mark_static(tensor)`：对局部输入静态化

### GE 侧

- 所有输入 shape 固定，更容易得到静态图和下沉调度
- 动态 shape 图通常只能 Host 调度
- 静态图中也可能有部分节点仍采用 Host 调度

## 7. 含 FA 算子的静态下沉

若想让包含 Flash Attention 类值依赖算子的模型走整图静态下沉，优先采用这组条件：

1. `torch.compile(..., dynamic=True)`
2. 对关键输入执行 `torch._dynamo.mark_static(...)`
3. 打开 `config.experimental_config.tiling_schedule_optimize = True`

如果 dump 图里 shape 仍为 `-1`，通常说明对应输入没有静态化成功。

## 8. 入图失败定位流程

文档建议的总流程：

1. 先看模型在 NPU Eager 模式是否成功
2. 再看图模式 backend 是否成功
3. 若 FX 成图失败，优先排查脚本与 Dynamo 侧
4. 若 FX 成图成功但后续失败，开 TorchAir Python/C++ 日志
5. 结合 TorchAir dump 图、GE dump 图、CANN plog 定位

定位时优先收集：

- TorchAir Python 日志
- TorchAir C++ 日志
- TorchAir dump 图
- GE dump 图
- CANN plog

## 9. 常见错误与修复方向

### `op dtype is not same`

- 常见是 TorchAir / GE 的 dtype 推导不一致
- 处理方式通常是在 Converter 中补 `Cast`

### `Check output size failed`

- 常见是 Meta 推导出的 dtype 或 shape 不正确
- 导致输出缓冲大小申请与 GE 实际需求不一致

### `tensor's device must be 'meta'`

- Meta 注册函数返回的 tensor 不在 `meta` 设备

### `torch.xxx ge_converter is not implemented!`

- 缺少对应 Converter

### `unsupported operator`

- 常见是没有实现 Meta 推导函数

### `Found a custom (non-ATen) operator`

- In-place 自定义算子通常缺函数化转换

### 开启固定权重地址后精度异常

- 重点排查 parameter 是否非连续；必要时转 `contiguous()`

### `GeneratedDatabase()->Add(...)`

- 一般是 TE 包与当前 CANN 版本不匹配

## 10. 精度比对

文档推荐用 `msit` 系工具做 TorchAir 图模式与 Eager/FX 的 dump 对比：

- `get_ge_dump_config(...)`：拿图模式 dump
- `get_fx_dump_config()`：拿 Eager / FX dump
- `msit llm compare --my-path ... --golden-path ...`

适合把整网精度问题拆解到算子级。

## 11. 性能分析

图模式性能分析建议区分：

- 编译前 / 编译阶段 / 首次执行
- Host 侧开销 / TorchAir 开销 / GE 开销 / NPU 执行开销

Profiling 产物里常用：

- `api_statistic.csv`
- `kernel_details.csv`
- `op_statistic.csv`
- `step_trace_time.csv`
- `trace_view.json`

若需要更系统的 profiling 流程，应切换到仓库里的 `ascend-profiling` skill。

## 12. 自定义算子入图完整链路

文档给出的完整工作链路是：

1. 确定算子原型
2. 实现 NPU 算子
3. 注册并适配 Eager
4. 实现 Meta 推导
5. 若是 In-place，补函数化转换
6. 需要时实现 Converter
7. 功能验证
