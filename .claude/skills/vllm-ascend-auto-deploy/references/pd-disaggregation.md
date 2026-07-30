# 多机 Prefill/Decode 分离

仅当 `deployment_mode=multi_node` 且用户确认启用 PD 分离时读取本文件。PD 架构包含 Proxy、Prefill KV producer、Decode KV consumer。PD 的目标通常是改善 TTFT/TPOT 尾延迟和 goodput，不保证提升总吞吐。

## 唯一 PD 规模问题

用户确认启用 PD 后，只询问一次：`请提供几 P 几 D（例如 2P2D）？`

`P` 和 `D` 默认分别表示 Prefill/Decode 的单节点独立实例数，`P + D`
就是默认节点总数。用户提示中已经给出 `2P2D` 等表达时不再询问。

其余参数不得作为固定问题，必须自动计算：

1. 先做 profile-first 选择：检查 Skill `profiles/` 和项目 `adaptations/`
   中的真实推理成功证据。只有模型架构、权重路径与配置指纹、量化、镜像、vLLM-Ascend 版本、
   平台资源规格和 xPyD 全部一致才可命中；命中后复用完整的 P/D TP、DP、
   作用域、本地 rank 布局、关键模型长度、连接器和超时参数，并运行
   `--validated-profile` 模式再次校验。任何一项不匹配都回退通用计算，
   不得近似套用。配置摘要将来源标为 `validated_profile`；未命中时标为
   `generic_calculation`。
2. 未命中 profile 时读取模型 `config.json`，用安全 TP 规则确定两侧运行 NPU 和 TP；每实例
   DP 默认为 1。MoE 开启 EP，Dense 关闭 EP。
3. 默认两侧均为 `independent_instances`，避免没有成功证据时引入跨节点
   collective；同模型、同镜像、同后端已有真实成功证据时可自动采用对应
   `global_group` 配置。
4. 从目标镜像检查 vLLM/vLLM-Ascend 版本并选择连接器；
   `use_ascend_direct=true`，关闭 prefix caching。
   推测解码/MTP、动态 EPLB、图模式及特殊权重预取只有在同模型制品、
   同镜像、同拓扑的真实推理证据中通过后才可作为默认值；否则采用保守
   基线。候选模板若已有失败记录，失败证据优先，禁止再次自动启用。
5. 自动分配 engine_id、KV port、DP rank、service port、Proxy 后端和节点槽位；
   KV 端口必须避开设备保留段并在提交前检查占用。
6. Proxy 默认与 Prefill-1 共置。共享模型路径、HCCL 网卡和运行时 IP 通过
   本机/SSH/Kubernetes 预检自动发现并验证；只有发现缺失或多个冲突候选时才追问。

调用：

```bash
python scripts/plan_pd_topology.py \
  --config CONFIG.json \
  --prefill-count 2 --decode-count 2 \
  --allocation-npu-per-node 16 \
  --vllm-ascend-version 0.21
```

端口、engine_id、DP rank 和 Proxy 映射必须由当前 Skill 内置规划器一次性
写入计划，生成脚本不得再次手算，也不得依赖仓库外部的脚本或运行目录。

若用户要求 agent 自动选择最优 P:D 比例，而不是用户直接给出几 P 几 D，
才询问典型输入长度、输出长度、并发量及 TTFT/TPOT 目标；缺少这些信息不得
声称比例是性能最优。

SSH 部署还需要每个 P/D 节点的 IP、用户名、SSH 端口和会话内凭据。调度平台仍按 scheduler reference 收集平台连接、队列、镜像、挂载和网络等信息，不从历史任务推断。

## 硬约束

- `global_group` 角色校验 `TP × DP = role_node_count × runtime_npu_per_node`。
- `independent_instances` 角色校验 `instance_count × TP × DP_per_instance = role_node_count × runtime_npu_per_node`。不得为了用满资源把已验证的独立 Decode 实例自动合并成跨节点 DP/EP group；是否跨节点组网必须是显式拓扑字段。
- 调度申请量和运行量分离；两侧运行量均不得超过每节点申请量。
- 设置 `--no-enable-prefix-caching`，请求中必须是 `prefix_caching=false`。
- vLLM/vLLM-Ascend 0.21+ 通常使用内置 `MooncakeConnectorV1`；P/D 非对称 TP/DP 优先 `MooncakeHybridConnector`。0.19 使用 `MooncakeConnector` 及匹配模块路径。最终以当前镜像实际支持的连接器为准。
- 将 `use_ascend_direct` 写入请求和计划，不依赖隐式默认值。0.21+ 默认要求 `true`；`false` 只允许在用户明确接受的降级/故障隔离任务中使用，不是自动回退路径。
- A3 场景 KV 端口避开 `[20000, 20000 + allocation_npu_per_node × 1000)`；通常从 36000 起，并检查端口占用。
- 不在纯 PD 场景擅自切换到 AscendStore/Yuanrong；只有明确需要 prefix reuse 或 HBM swap 时才评估。
- 不在 PD 分离模式设置 balance scheduling。
- 不独立 import/初始化 Mooncake TransferEngine 做探测；缺少 NPU 上下文可能导致进程崩溃。
- 模型特定环境变量（例如部分 MoE+MTP 组合需要关闭 fused MC2）必须由同模型、同版本证据触发，不得全局硬编码。

## Direct 传输失败处理

若真实请求已显示各 rank KV transfer 后，Decode 在 `AscendDirectTransport`、`HcclCommPrepare` 或 `GlobalMemRegMgr` 内原生段错误：

1. 将该作业判为失败并停止，仅保留日志；不要把后续 HTTP 500 当作 Proxy 问题。
2. 查找同模型、同镜像、同拓扑的 direct 成功证据；检查节点是否有残留 Mooncake/ZMQ/HCCL 进程或端口占用。
3. 生成新的冻结目录和新 job name，换干净节点或完成节点级清理后保持 P/D 两侧 `use_ascend_direct=true` 重提；不在运行脚本上原地修改。
4. 至少一次干净节点复测仍失败时，报告 direct 阻塞和原生栈证据。禁止自行改成 `false`。
5. 仅当用户明确同意降级时，才能在另一冻结目录把 P/D 两侧同时改为 `false` 做故障隔离；结果必须标记 `degraded_transport=true`，且不能替代 direct 验收。
6. 只有 Proxy HTTP 200、所有 rank KV transfer 成功、无 fallback/EngineDead 且严格语义断言通过时才报告完整成功。

非 direct 是稳定性回退，不代表语义质量自动通过。若严格提示要求固定答案，必须断言解析后的文本；同时直接请求 Prefill 做对照。Prefill 单体和 PD 都产生同类异常文本时，将问题归到模型/量化/镜像运行配置，而不是 KV 传输。

## Kubernetes/A3 网络

优先使用平台已验证的 host network 模式。不要机械设置 `GLOO_SOCKET_IFNAME` 或 `TP_SOCKET_IFNAME`，显式绑定错误接口可能使 Gloo 失败。必须按当前平台网卡实测确认 `HCCL_IF_IP`、`HCCL_SOCKET_IFNAME`，Proxy 与 KV 地址必须是其他角色可达的运行时 IP。

P/D 的 DP group 必须来自同一启动代次。调度器若分批重启或换节点，旧 rank 不得继续等待新 rank 加入已经初始化的 communicator。使用 `stable_cluster_supervisor.py` 在加载权重前形成稳定代次；session/IP 变化时立即终止 P、D、Proxy 的整个本地进程组，共享存储或心跳短暂不可用时只在有限宽限期内重试。将状态目录按 job 冻结目录唯一化，禁止复用上一次任务的共享 IP/心跳目录；保留 `events/rank-*.jsonl` 以追溯容器重启前的原因。

对于跨节点 Prefill DP/EP，稳定代次只能保证同时开始，不能保证同时完成权重加载。共享权重的冷热缓存差异可能超过 PyTorch 默认 30 分钟，并在 `post_process_after_loading` 的 broadcast 中超时。启动时应显式设置 `--distributed-timeout-seconds`，取值必须覆盖最坏节点加载偏差并留有余量；它和 engine handshake timeout 是两个独立门限。当前镜像若存在 Ascend Worker 未透传该参数的问题，调用 `scripts/patch_vllm_distributed_timeout.py` 后再启动，且把补丁成功信息纳入日志验收。

`post_process_after_loading` 首次创建 HCCL communicator 时还受
`HCCL_CONNECT_TIMEOUT` 约束；快节点等待尚未交付建连接口的慢节点时，
即使 distributed timeout 足够也会独立失败。跨节点大模型需要同时按最坏
加载差设置 `HCCL_CONNECT_TIMEOUT`，并为长 collective 设置足够的
`HCCL_EXEC_TIMEOUT`。同时让 `--distributed-timeout-seconds` 严格大于
`HCCL_EXEC_TIMEOUT`，保留 HCCL plog 与协调退出的诊断窗口。

## 规划命令

```bash
python scripts/plan_pd_topology.py \
  --config /path/to/model/config.json \
  --prefill-count 2 --decode-count 2 \
  --allocation-npu-per-node 16 \
  --vllm-ascend-version 0.21
```

规划器负责资源恒等式、安全 TP/DP/EP、作用域、端口、engine、节点槽位和
连接器建议，不声称自动优化用户给出的 P:D 比例。用户在提示中明确指定
高级拓扑时，仍可使用规划器的手动参数模式覆盖默认值。

## 验收

1. 分别检查所有 Prefill 与 Decode 实例 `/v1/models`。
2. 检查 Proxy 后端注册和存活状态。
3. 先直接请求 Prefill/普通非 PD 基线，再通过 Proxy 发送答案唯一的确定性请求；两者都对解析后的文本做语义断言。
4. 检查两侧 KV transfer 日志，确认 engine/port 配对正确、传输成功且无 fallback。
5. 用 `scripts/validate_inference_result.py` 对固定答案提示做精确或锚定正则断言；HTTP 200 和非空内容只证明链路可返回，不能证明模型质量。
6. 检查所有角色无异常重启；失败时停止本次创建的全部 P、D、Proxy 资源并确认无残留。
