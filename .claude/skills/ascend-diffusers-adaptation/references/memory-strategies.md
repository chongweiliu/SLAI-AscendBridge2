# Diffusers Memory Strategies

## 选择原则

先比较：

- 最大单组件大小
- 全部组件总大小
- 单卡 HBM 可用量
- 可用 NPU 卡数

## 1. 整管线 `.to("npu")`

适合：

- 最大组件和总组件都能放进单卡
- 希望保留最简单、最快的执行路径

优点：

- 代码最简单
- 没有频繁 CPU<->NPU 搬运

## 2. `model-offload`

适合：

- 单卡无法同时常驻所有组件
- 但主要组件可以轮流上卡

经验：

- 编码阶段结束后，及时把不再需要的组件移回 CPU
- 释放对象后再 `gc.collect()` 与 `torch.npu.empty_cache()`

## 3. `enable_sequential_cpu_offload`

适合：

- 单卡显存最紧张
- 速度不是第一优先级

注意：

- 这是兜底路径，通常最慢
- 某些 offload 场景下，`torch.Generator(device="cpu")` 更稳

## 4. 多 NPU `dispatch_model`

适合：

- 最大组件本身就超过单卡可承受范围
- 或单卡无法完成稳定测量

建议：

- 优先只 dispatch 最大的组件，不要先对整条 pipeline 做粗暴 `device_map`
- 常见做法是只 dispatch `transformer` / `unet`，把 `vae` 固定在主卡
- 预留部分 HBM 给中间激活，不要把设备容量配满

## 5. 不要优先用 `device_map="balanced"`

在 Ascend 路线上，pipeline 级 `device_map="balanced"` 兼容性较差。

默认策略：

- 先加载到 CPU
- 再对重组件单独 `infer_auto_device_map()` + `dispatch_model()`

## 6. 计时与清理

做性能判断或 OOM 排查时：

- 关键阶段前后做 `torch.npu.synchronize()`
- 组件迁移后及时释放无用对象
- 不要把多个大组件同时留在卡上“观望”
