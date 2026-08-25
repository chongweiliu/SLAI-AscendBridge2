---
name: vllm-ascend-troubleshooting
description: Diagnose vLLM-Ascend startup, first-forward, runtime, HCCL, KV-transfer, memory, and API failures from reproducible evidence. Use when an Ascend vLLM service fails, hangs, falls back, or regresses.
---

# vLLM-Ascend 故障排查

按“现象 -> 最早根因 -> 最小复现 -> 修复证据”排查，不把最后一行的连锁异常当根因。
部署任务先保留 deployment ID、请求 JSON、镜像与 vLLM-Ascend 版本、硬件代际、拓扑、
环境变量、完整启动命令和首个真实请求；禁止用重启掩盖问题。

## 路由

遇到具体日志模式时读取 [references/failure-matrix.md](references/failure-matrix.md)，按其中的
观察项和最小复现隔离故障，不机械套用修复参数。

- 启动即退出、导入失败、算子缺失：先核对 Python/torch/torch-npu/vLLM/vLLM-Ascend/CANN
  版本和模型 `config.json`，再查模型适配 Skill。
- `/v1/models` 正常但首次 forward 失败：检查 dtype、TP/DP/EP、KV heads、量化和模型
  task；健康检查不代表拓扑可用。
- 多机卡住或 HCCL timeout：读取 [HCCL 验证](../ascend-hccl-validation/SKILL.md)，先做
  节点、端口、网卡、时间和 world-size 一致性检查，再决定是否做带宽测试。
- PD 请求无输出或 KV transfer 报错：分别检查 Prefill/Decode/Proxy 健康、Mooncake
  connector、`kv_role`、`kv_port`、P/D TP/DP 和 direct transport；先用普通非 PD 基线
  定位模型问题。
- OOM 或吞吐下降：记录输入/输出长度、并发、batch token、KV cache、量化、prefix caching
  和图模式，单变量调整；不要把一次调参结果宣称为全局最优。

## 证据采集

在目标机或容器中运行 `scripts/collect_vllm_ascend_env.py --output env.json`。该脚本只读，
会记录可用命令输出和版本，缺少命令不会伪造成功。随后收集启动日志、`npu-smi info`、
环境变量白名单、实际命令行和一次确定性请求的响应。敏感值只记录变量名，不记录 token、
密码或私钥。

## 结论门槛

每个结论必须标记 `observed`、`reproduced`、`inferred` 或 `unknown`。修复后至少重跑原始
失败请求和最小真实 forward；多机或 PD 还要验证通信/KV transfer。若只做静态检查，明确说明
没有目标 NPU/集群上的 E2E 证据。
