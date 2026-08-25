# dtype 验证

## 分别记录

- checkpoint tensor dtype 与量化格式；
- `--dtype` 解析后的模型/激活 dtype；
- KV cache dtype；
- accumulation 或敏感算子的内部精度（可观察时）；
- 硬件、CANN、torch-npu、vLLM/vLLM-Ascend 和量化插件版本。

## A/B 方法

固定模型制品、拓扑、镜像、prompt、chat template、采样参数、长度和并发。每个候选先预热，
再重复真实 forward，检查任务契约、NaN/Inf、显存峰值、TTFT、TPOT 和吞吐。BF16 与 FP16
权重字节数通常接近，因此“FP16 一定更省内存”不能作为默认结论；OOM 还可能来自 KV cache、
图捕获、激活、batch token 或碎片。

FP32 可用于小规模故障隔离，但不应假定所有优化 kernel 都支持或高效。`auto` 必须在启动
日志中解析成具体 dtype 后才算已知。量化模型按其官方加载参数验证，不能仅改 `--dtype`
模拟量化。
