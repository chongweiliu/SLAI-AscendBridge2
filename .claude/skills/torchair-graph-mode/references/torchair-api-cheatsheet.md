# TorchAir API Cheatsheet

只收录在适配、排障、入图和图优化中最常用的接口。

## 后端与编译

### `torchair.CompilerConfig`

TorchAir 图模式总配置入口。

常见分组：

- `debug`
- `dump_config`
- `fusion_config`
- `experimental_config`
- `inference_config`
- `ge_config`
- `aclgraph_config`
- `mode`

### `torchair.get_npu_backend(compiler_config=config)`

获取 TorchAir 的 NPU backend，供 `torch.compile(..., backend=...)` 使用。

### `torchair.get_compiler(...)`

获取编译器对象。需要面向更底层控制时再考虑，常规场景优先 `get_npu_backend(...)`。

### `torchair.dynamo_export(...)`

用于导图或导出 Dynamo 图相关场景。不是一般“让模型先跑起来”的首选接口。

## 集合通信

### `torchair.patch_for_hcom()`

用于集合通信相关入图能力补丁。多卡通信图优化场景可考虑使用。

## FX 图替换 / 融合

### `torchair.register_replacement(...)`

在 FX 图里识别 `search_fn` 模式并替换为 `replace_fn`。

适合：

- 已有一段算子组合，想整体替换为更优逻辑
- 做融合 Pass，而不是单个算子的 GE Converter

关键参数：

- `search_fn`
- `replace_fn`
- `example_inputs`
- `extra_check`

## Converter 注册

### `register_fx_node_ge_converter(...)`

为某个 FX 节点注册到 GE 的 Converter。

适合：

- 报错 `ge_converter is not implemented`
- 自定义算子需要从 FX 映射到 GE
- 想在 Converter 里插入 `Cast` / `Reshape` / `Const` 等 GE 构图元素

## `torchair.ge` 常用构图元素

### `DataType`

GE dtype 枚举。常用：

- `DT_FLOAT`
- `DT_FLOAT16`
- `DT_BF16`
- `DT_INT64`
- `DT_BOOL`

### `Format`

GE format 枚举。常用：

- `FORMAT_NCHW`
- `FORMAT_NHWC`
- `FORMAT_ND`

### `Tensor`

Converter 入参类型声明使用。

### `TensorSpec`

表示 Meta 推导得到的张量规格，常用于读取：

- `dtype`
- `rank`
- `size`

### `Const(v, dtype=..., node_name=...)`

在 GE 图中创建常量节点。

适合：

- 把 Python 常量显式变为图节点
- 给自定义 `custom_op` 传入常量属性/输入

### `Cast(x, dst_type=...)`

显式插入类型转换。处理 dtype 推导不一致时很常用。

### `Clone(x, dependencies=[...])`

显式做图上的拷贝。某些 in-place / 依赖顺序相关场景会用到。

### `custom_op(...)`

根据算子原型构造 GE 自定义节点。

适合：

- 已有自定义算子 IR / REG_OP
- 需要在 Converter 中直接拼 GE 自定义节点

## 推理与缓存

### `torchair.inference.cache_compile(...)`

用于模型编译缓存。

适合：

- 首次编译代价高
- 相同或兼容输入会重复执行

注意：

- 某些功能组合有限制，启用前检查与当前模式/配置是否兼容

### `torchair.inference.readable_cache(...)`

让缓存更可读，方便分析缓存内容。

### `torchair.inference.set_dim_gears(...)`

动态 shape 分档执行场景可用。

## 图内多流 / 范围控制

### `torchair.scope.npu_stream_switch(...)`

切换图内执行流标签。

### `torchair.scope.npu_wait_tensor(...)`

建立图内流间依赖，让后续算子等待前序算子结果。

### `torchair.scope.super_kernel(...)`

标定可融合为 SuperKernel 的上下文范围。

### `torchair.scope.limit_core_num(...)`

限制算子使用的 AI Core / Vector Core 上限。

### `torchair.scope.op_never_timeout(...)`

给 GE 图中的算子配置“不参与超时检测”属性。

## LLM DataDist

### `torchair.llm_datadist.create_npu_tensors(...)`

用一组 device 地址创建 NPU Tensor，主要用于 KV Cache 等分离部署场景。

## 选型建议

- **只是启用 TorchAir 图模式**：`CompilerConfig` + `get_npu_backend`
- **只是集合通信入图**：先看 `patch_for_hcom`
- **缺 Converter**：`register_fx_node_ge_converter`
- **做子图替换或融合**：`register_replacement`
- **需要显式 GE 节点**：`Const` / `Cast` / `Clone` / `custom_op`
- **需要缓存或动态分档**：`cache_compile` / `set_dim_gears`
- **需要图内多流或核数限制**：`npu_stream_switch` / `npu_wait_tensor` / `limit_core_num`
