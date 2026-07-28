> 来源：vllm-ascend docs/source/user_guide/deployment_guide/index.md + using_mindie_motor.md + using_volcano_kthena.md（main 分支，抓取于 2026-07-28）

# 外部部署集成方式

vLLM-Ascend 的外部部署集成主要有两种：MindIE-Motor（一键 PD 分离/聚合）与 Volcano Kthena（K8s 编排）。

## 1. MindIE-Motor

[MindIE-Motor](https://gitcode.com/Ascend/MindIE-Motor) 为 vLLM-Ascend 在 Ascend NPU 上提供一键 **prefill–decode (PD) 分离**与 **PD 聚合**部署。它用高性能调度与负载均衡，配合 RAS（Reliability, Availability and Serviceability）能力，构建快速且高稳定的推理服务。

快速部署见 [MindIE-Motor Quick Start](https://gitcode.com/Ascend/MindIE-Motor/blob/master/docs/zh/user_guide/README.md)。

## 2. Volcano Kthena（K8s 上 PD 分离）

[Kthena](https://kthena.volcano.sh/) 是 K8s 原生 LLM 推理平台，用声明式模型生命周期管理与智能请求路由提供高性能、企业级可扩展性。vLLM 与 Kthena 集成见 [Deploy vLLM with Kthena](https://docs.vllm.ai/en/latest/deployment/integrations/kthena/)。

### 2.1 PD 分离是什么

LLM 推理分两阶段：
- **Prefill**：处理输入 token、构建 KV cache；batch 友好、高吞吐、适合并行 NPU。
- **Decode**：消费 KV cache 生成输出 token；延迟敏感、内存密集、更串行。
- 客户端视角仍是单个 Chat/Completions 端点。

### 2.2 三个关键 CRD

- `ModelServing` — 定义工作负载（prefill 与 decode 角色）
- `ModelServer` — 管理 PD 分组与内部路由
- `ModelRoute` — 暴露稳定的模型端点

示例用 `deepseek-ai/DeepSeek-V2-Lite`，可替换为任意 vLLM-Ascend 支持的模型。

### 2.3 前置条件

- 带 Ascend NPU 节点的 K8s 集群；不同 NPU Driver 资源名略有差异：
  - MindCluster：`huawei.com/Ascend310P` 或 `huawei.com/Ascend910`
  - 华为云 CCE + CCE AI Suite Plugin (Ascend NPU)：`huawei.com/ascend-310` 或 `huawei.com/ascend-1980`
- 已装 Kthena（见 [Kthena 安装指南](https://kthena.volcano.sh/docs/getting-started/installation)）

### 2.4 部署 PD 分离 DeepSeek-V2-Lite

官方示例文件：

```bash
kubectl apply -f https://raw.githubusercontent.com/volcano-sh/kthena/refs/heads/main/examples/model-serving/prefill-decode-disaggregation.yaml
```

关键字段（ModelServing manifest，prefill 角色）：

- `image: ghcr.io/volcano-sh/kthena-engine:vllm-ascend_v0.10.1rc1_mooncake_v0.3.5`
- 关键环境变量：
  - `HF_HUB_OFFLINE=1`
  - `HCCL_IF_IP` = `status.podIP`
  - `GLOO_SOCKET_IFNAME=eth0`、`TP_SOCKET_IFNAME=eth0`、`HCCL_SOCKET_IFNAME=eth0`
  - `VLLM_LOGGING_LEVEL=DEBUG`
  - `AscendRealDevices` = `metadata.annotations['huawei.com/AscendReal']`
- 关键 vLLM 参数（prefill）：
  - `--served-model-name deepseek-ai/DeepSeekV2`
  - `--tensor-parallel-size 2`
  - `--gpu-memory-utilization 0.8`
  - `--max-model-len 8192`
  - `--max-num-batched-tokens 8192`（prefill）
  - `--trust-remote-code`、`--enforce-eager`
  - `--kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_buffer_device":"npu","kv_role":"kv_producer","kv_parallel_size":1,"kv_port":"20001","kv_rank":0,"kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":2},"decode":{"dp_size":2,"tp_size":2}}}'`
- 资源：`limits/requests: cpu=8, memory=64Gi, huawei.com/ascend-1980=4`
- 健康探针：readiness `/health:8000`（initialDelaySeconds 5）；liveness `/health:8000`（initialDelaySeconds 900）
- 卷：models（hostPath）、hccn-config（`/etc/hccn.conf`）、shared-memory-volume（emptyDir Memory 256Mi）

decode 角色差异：
- `--max-num-batched-tokens 16384`
- `--no-enable-prefix-caching`
- `kv_role=kv_consumer`、`kv_port=20002`、`kv_rank=1`

预期 Pod：
- `deepseek-v2-lite-0-prefill-0-0`
- `deepseek-v2-lite-0-decode-0-0`

### 2.5 ModelServer：PD 分组管理

```bash
kubectl apply -f https://raw.githubusercontent.com/volcano-sh/kthena/refs/heads/main/examples/kthena-router/ModelServer-prefill-decode-disaggregation.yaml
```

manifest 关键字段：

- `kvConnector.type: nixl`
- `workloadSelector.matchLabels.modelserving.volcano.sh/name: deepseek-v2-lite`
- `pdGroup.groupKey: "modelserving.volcano.sh/group-name"`
- prefillLabels `role: prefill`、decodeLabels `role: decode`
- `workloadPort.port: 8000`
- `model: "deepseek-ai/DeepSeekV2"`、`inferenceEngine: "vLLM"`
- `trafficPolicy.timeout: 10s`

### 2.6 ModelRoute：用户端点

```yaml
apiVersion: networking.serving.volcano.sh/v1alpha1
kind: ModelRoute
metadata:
  name: deepseek-v2
  namespace: dev
spec:
  modelName: "deepseek-ai/DeepSeekV2"
  rules:
    - name: "default"
      targetModels:
        - modelServerName: "deepseek-v2"
```

### 2.7 验证

```bash
# 检查工作负载
kubectl get modelserving deepseek-v2-lite -n dev -o yaml | grep status -A 10
kubectl get pod -n dev -owide -l modelserving.volcano.sh/name=deepseek-v2-lite

# 测试 chat 端点
export ENDPOINT=$(kubectl get svc kthena-router -n kthena-system --output=jsonpath='{.status.loadBalancer.ingress[0].ip}:{.spec.ports[0].port}')
curl --location "http://${ENDPOINT}/v1/chat/completions" \
  --header "Content-Type: application/json" \
  --data '{
    "model": "deepseek-ai/DeepSeekV2",
    "messages": [{"role": "user", "content": "Where is the capital of China?"}],
    "stream": false
  }'
```

### 2.8 清理

```bash
kubectl delete modelroute deepseek-v2 -n dev
kubectl delete modelserver deepseek-v2 -n dev
kubectl delete modelserving deepseek-v2-lite -n dev
```
