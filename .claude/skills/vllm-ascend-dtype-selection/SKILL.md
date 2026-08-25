---
name: vllm-ascend-dtype-selection
description: Select and validate vLLM-Ascend inference dtype from model config, quantization metadata, hardware generation, and user constraints. Use before rendering commands or when dtype causes memory, accuracy, or kernel failures.
---

# dtype 选择

dtype 是约束决策，不是只按参数量猜测。优先级为：用户明确要求、官方同模型同硬件 recipe、
权重/量化元数据、硬件与镜像支持矩阵、最后才是保守候选。用
`scripts/select_dtype.py CONFIG --hardware A3` 生成可审计建议。

## 判断规则

- 已量化权重必须使用其声明的量化路径；不能把 FP16/BF16 参数覆盖到 W8A8、GPTQ、AWQ、
  FP8 等权重上。
- 未量化 LLM 默认比较 BF16 与 FP16：BF16 通常优先稳定性和动态范围，FP16 只在官方 recipe、
  显存/算子约束或实测证明需要时采用。
- 不要把 `dtype` 与 KV cache dtype、权重 dtype、激活 dtype 混为一谈；分别记录。
- A2/A3/310p、vLLM-Ascend 版本和量化 kernel 支持必须交叉核对；未知时停止自动部署并询问。

## 验收

固定提示完成单卡和目标 TP 的真实 forward，检查 NaN/Inf、空输出、长度异常、显存峰值和
任务语义。需要比较候选时读取 [references/validation.md](references/validation.md)。dtype 选择
可以缩小搜索空间，但没有同制品同拓扑实测时，不称为“最优”。
