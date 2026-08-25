---
name: ascend-hccl-validation
description: Validate HCCL prerequisites for Ascend multi-node vLLM-Ascend and PD deployments. Use before distributed startup or when ranks hang, timeout, or disagree about topology.
---

# HCCL 通信验证

HCCL 验证是多机部署的前置门，不是性能寻优器。先验证“能通信且配置一致”，只有用户要求
性能基线或通信可疑时才做带宽测试。单机部署不强制运行多机带宽测试。

## 必查项

1. 所有节点硬件代际、CANN、driver/firmware、torch-npu、vLLM-Ascend 和镜像版本一致。
2. `world_size`、rank、TP/DP/EP、角色分组和 `ASCEND_RT_VISIBLE_DEVICES` 一致；PD 的
   Prefill/Decode 组分别核对，不能用普通多机公式替代角色公式。
3. HCCL 网卡/IP 从路由和平台注入信息确定；`HCCL_SOCKET_IFNAME`、`GLOO_SOCKET_IFNAME`
   和 `TP_SOCKET_IFNAME` 不得指向不可达或混用的接口。
4. 节点间 rendezvous、HCCL、KV transfer 和 Proxy 端口可达，DNS/host 映射稳定，时钟偏差
   可接受，防火墙/安全组没有拦截。
5. 先运行轻量 all-rank 初始化或最小真实 forward，再诊断带宽；`HCCL timeout` 不能直接
   通过增大 timeout 视为解决。

## 产出

用 `scripts/validate_hccl_plan.py plan.json` 检查静态通信契约。报告要列出每个节点的 rank、
角色、网卡/IP、端口、world-size 和失败项。静态通过不等于目标集群 E2E 通过；真实部署还要
保留 HCCL 初始化日志和首个 forward 证据。需要集合通信正确性或带宽测试时，再读取
[references/hccl-test.md](references/hccl-test.md)。
