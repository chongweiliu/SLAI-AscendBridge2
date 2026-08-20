# 硬件代际与官方 recipe

## 规则来源

2026-08-12 排查 DeepSeek-V4-Flash 自动部署发现：官方教程同时包含 A2 与 A3
命令，但旧 profile 将其压平成 `recommended_tp=[4,8]`，通用规划器随后选择最大
TP=8，导致 A2 recipe 被错误套到 A3。

短期通用方案：

1. 先检测 A2/A3/310p/Ascend950。
2. 从模型官方教程中选择当前硬件和部署模式对应的完整 recipe。
3. 同步使用 recipe 的 TP/DP/EP 与环境变量。
4. 没有对应 recipe 时才回退模型配置通用计算，并标记
   `topology_source=model_config_topology`。

## DSV4 证据案例

- A2 单节点：TP=8、DP=1、EP 开启。
- A3 单节点：TP=4、DP=4、EP 开启；教程使用 `HCCL_BUFFSIZE=1024`、
  `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`。
- A3 PD：Prefill TP=4/DP=4，Decode TP=1/DP=16。

若另行采用 A3 TP=8 等非官方场景，必须有同模型、同硬件、同版本、同拓扑的
真实推理证据。已知大 MoE TP8 AllGather 在某些 A3 环境默认 ring 会触发
`507014`/aicore timeout；经验证的规避配置为
`HCCL_ALGO="level0:NA;level1:fullmesh"`。该结论是诊断记忆，不应覆盖官方
recipe 优先级，也不能无条件应用到所有模型。
