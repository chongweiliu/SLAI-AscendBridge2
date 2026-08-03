> 来源：vllm-ascend docs（main 分支，抓取于 2026-07-28），整合自多个源文件

# PD Disaggregation 知识库（Prefill/Decode 分离部署）

本文件整合 vllm-ascend 官方文档中关于 Prefill/Decode 分离（PD disaggregation）的全部内容，覆盖单节点 Mooncake、多节点 Mooncake、PD-Colocated 多实例以及 EPD（Encoder-Prefill-Decode）设计与设计文档。所有命令、配置块、端口、环境变量均保留原文英文，仅说明性文字使用中文。

---

## PD Disaggregation Overview

PD 分离（Prefill-Decode disaggregation）是 vLLM-Ascend 在大规模推理服务中的核心架构：把 prefill（首 token 计算阶段，KV cache 计算密集）与 decode（自回归生成阶段，显存带宽密集）拆分到不同的 vLLM 实例，再通过专用 connector 在 P/D 节点之间传输 KV cache，从而对 TTFT 与 TPOT/ITL 做独立、细粒度的优化。

### 为什么要分离 Prefill

- **并行策略与实例数可独立调整**：P 节点与 D 节点可以分别配置 dp / tp / ep 以及实例数，针对 TTFT 与 TPOT 做针对性调优。
- **优化 TPOT**：若不分离，prefill 任务会插入 decode 流水，造成 decode 抖动与延迟。分离后可避免在 decode 期间插入 chunked prefill，省去 chunk size 的取舍，使 TPOT 更可控。

### 两种 KV cache 传输 Connector

- **MooncakeConnector**：D 节点主动从 P 节点 **pull** KV cache。请求先到 P 节点完成 prefill，再被 Proxy 转发到 D 节点，D 节点拉取远端 KV 后继续 decode。对应示例 `load_balance_proxy_server_example.py`。
- **MooncakeLayerwiseConnector**：P 节点以分层方式向 D 节点 **push** KV cache。请求先进入 D 节点，D 节点经 Metaserver 反向触发远端 prefill，P 节点逐层推送 KV，与计算重叠；传输完成后 D 节点无缝续接 decode。对应示例 `load_balance_proxy_layerwise_server_example.py`。

### 与 EPD（Encoder-Prefill-Decode）的关系

对于多模态模型，PD 分离通常扩展为 EPD：把 vision encoder 阶段也独立到单独 vLLM 实例。

- **Encoder 实例**：执行视觉编码。
- **PD 实例**：可以是 (E + PD) 单实例，也可以是 (E + P + D) 全分离三实例。
- 通过 **ECConnector** 在 encoder 与 PD 实例间传递 encoder-cache（EC）embedding。`vllm/distributed/ec_transfer` 下集中实现。
- EPD Load Balancing Proxy 采用多路径调度 + 实例级动态负载均衡（按活跃 token 负载的最小负载策略）。
- 在 vLLM-Ascend 中，EPD 默认使用 `MooncakeLayerwiseConnector`（`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py`）完成 P↔D 的 KV 传递。

### 规格与限制

- 兼容 A2 与 A3 硬件；支持 MLA、GQA；支持等 TP 与部分非等 TP（P_tp > D_tp 且 P_tp % D_tp == 0）。
- **不支持异构 P/D**（如 P 在 A2、D 在 A3）。
- 多模态跨进程缓存场景需关闭 `--mm-processor-cache-gb 0`（即不要设为 0）。
- 启用 speculative decoding 时，`num_speculative_tokens` 须满足：Hybrid Mamba 模型（Qwen-Next / Qwen3.5 系列）P 与 D 相等；其他模型 P 节点为 1、D 节点 >= 1。

### DFX 校验项

- KV transfer 配置项校验：检查 `kv_connector` 类型是否受支持；传输失败时给出清晰错误日志。
- 端口冲突检测：启动前对 `rpc_port`、`metrics_port`、`http_port/metaserver` 等做 bind 探测，已占用则 fail fast。
- PD 比例校验：非对称 PD 场景下校验 P↔D 的 TP 比例是否符合调度约束。

### 部署模式选择指引

| 场景 | 推荐模式 | Connector | 代理脚本 |
|------|----------|-----------|----------|
| 单机资源受限验证 | 1P1D 单节点 Mooncake | MooncakeConnectorV1 | `load_balance_proxy_server_example.py` |
| 多机大模型（如 DeepSeek）+ EP | 2P1D 多节点 | MooncakeLayerwiseConnector / MooncakeConnectorV1 | `load_balance_proxy_layerwise_server_example.py`（Layerwise）或 `load_balance_proxy_server_example.py`（非 Layerwise） |
| 跨实例/跨节点 KV 复用，节点内 colocate | PD-Colocated 多实例 | MooncakeConnectorStoreV1 | （单实例对外暴露，Proxy 可选） |
| 多模态（vision encoder + LM）| EPD | ECConnector + MooncakeLayerwiseConnector | `disagg_1e1pd` / `disagg_1e1p1d` |

---

## Mooncake Single Node

适用场景：单台 Atlas 800T A2，资源受限下做 "1P1D" 验证（一个 Prefiller + 一个 Decoder 在同一节点）。示例模型 Qwen2.5-VL-7B-Instruct，vllm-ascend 镜像对应版本，节点 IP 假设 192.0.0.1。

### 单节点通信环境校验

```bash
# Check the remote switch ports
for i in {0..7}; do hccn_tool -i $i -lldp -g | grep Ifname; done
# Get the link status of the Ethernet ports (UP or DOWN)
for i in {0..7}; do hccn_tool -i $i -link -g ; done
# Check the network health status
for i in {0..7}; do hccn_tool -i $i -net_health -g ; done
# View the network detected IP configuration
for i in {0..7}; do hccn_tool -i $i -netdetect -g ; done
# View gateway configuration
for i in {0..7}; do hccn_tool -i $i -gateway -g ; done
```

检查 HCCN 配置（Docker 需挂载进容器）：

```bash
cat /etc/hccn.conf
```

获取 NPU IP：

```bash
for i in {0..7}; do hccn_tool -i $i -ip -g;done
```

跨节点 PING 测试：

```bash
# Execute on the target node (replace 'x.x.x.x' with actual npu ip address).
for i in {0..7}; do hccn_tool -i $i -ping -g address x.x.x.x;done
```

检查 NPU TLS 配置：

```bash
# The tls settings should be consistent across all nodes
for i in {0..7}; do hccn_tool -i $i -tls -g ; done | grep switch
```

### 启动 Docker 容器

```bash
# Update the vllm-ascend image
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
export NAME=vllm-ascend

# Run the container using the defined variables
docker run --rm \
--name $NAME \
--net=host \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci4 \
--device /dev/davinci5 \
--device /dev/davinci6 \
--device /dev/davinci7 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /mnt/sfs_turbo/.cache:/root/.cache \
-it $IMAGE bash
```

### 安装 Mooncake

Mooncake 是 Kimi（Moonshot AI）的推理服务平台。安装与编译指南：<https://github.com/kvcache-ai/Mooncake?tab=readme-ov-file#build-and-use-binaries>。

```shell
git clone -b v0.3.9 --depth 1 https://github.com/kvcache-ai/Mooncake.git
```

（可选）网络较差时替换 go install URL：

```shell
cd Mooncake
sed -i 's|https://go.dev/dl/|https://golang.google.cn/dl/|g' dependencies.sh
```

安装 MPI：

```shell
apt-get install mpich libmpich-dev -y
```

安装相关依赖（不需要安装 Go）：

```shell
bash dependencies.sh -y
```

编译并安装（**`USE_ASCEND_DIRECT=ON` 是 Ascend 直连传输的关键开关**）：

```shell
mkdir build
cd build
cmake .. -DUSE_ASCEND_DIRECT=ON
make -j
make install
```

设置环境变量（按实际 Python 路径调整；确保 `/usr/local/lib` 与 `/usr/local/lib64` 在 `LD_LIBRARY_PATH` 中）：

```shell
export LD_LIBRARY_PATH=/usr/local/lib64/python3.12/site-packages/mooncake:$LD_LIBRARY_PATH
```

### Prefiller / Decoder 部署

#### Prefiller

```shell
export ASCEND_RT_VISIBLE_DEVICES=0
export HCCL_IF_IP=192.0.0.1  # node ip
export GLOO_SOCKET_IFNAME="eth0"  # network card name
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10

vllm serve /model/Qwen2.5-VL-7B-Instruct  \
  --host 0.0.0.0 \
  --port 13700 \
  --no-enable-prefix-caching \
  --tensor-parallel-size 1 \
  --seed 1024 \
  --served-model-name qwen25vl \
  --max-model-len 40000  \
  --max-num-batched-tokens 40000  \
  --trust-remote-code \
  --gpu-memory-utilization 0.9  \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_producer",
  "kv_port": "30000",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 1,
                    "tp_size": 1
             },
             "decode": {
                    "dp_size": 1,
                    "tp_size": 1
             }
      }
  }'
```

#### Decoder

```shell
export ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_IF_IP=192.0.0.1  # node ip
export GLOO_SOCKET_IFNAME="eth0"  # network card name
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10

vllm serve /model/Qwen2.5-VL-7B-Instruct  \
  --host 0.0.0.0 \
  --port 13701 \
  --no-enable-prefix-caching \
  --tensor-parallel-size 1 \
  --seed 1024 \
  --served-model-name qwen25vl \
  --max-model-len 40000  \
  --max-num-batched-tokens 40000  \
  --trust-remote-code \
  --gpu-memory-utilization 0.9  \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_consumer",
  "kv_port": "30100",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 1,
                    "tp_size": 1
             },
             "decode": {
                    "dp_size": 1,
                    "tp_size": 1
             }
      }
  }'
```

> 若要部署 "2P1D"，请为每个 P 进程设置不同的 `ASCEND_RT_VISIBLE_DEVICES` 和端口。

### 部署 Proxy

在与 prefiller 同节点运行代理，脚本来自 examples：[load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

```shell
python load_balance_proxy_server_example.py \
    --host 192.0.0.1 \
    --port 8080 \
    --prefiller-hosts 192.0.0.1 \
    --prefiller-port 13700 \
    --decoder-hosts 192.0.0.1 \
    --decoder-ports 13701
```

|Parameter  | Meaning |
| --- | --- |
| --port | Port of proxy |
| --prefiller-port | All ports of prefill |
| --decoder-ports | All ports of decoder |

### 验证

```shell
curl http://192.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen25vl",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
                {"type": "text", "text": "What is the text in the illustration?"}
            ]}
            ],
        "max_completion_tokens": 100,
        "temperature": 0
    }'
```

---

## Mooncake Multi Node

适用场景：多台 Atlas 800T A3，部署 "2P1D" + EP（Expert Parallel）。示例模型 DeepSeek-r1-w8a8，4 台服务器：prefill 节点 192.0.0.1/192.0.0.2，decoder 节点 192.0.0.3/192.0.0.4；每台 8 NPU、16 chip，部署一个服务实例。

### 多节点通信环境校验

**物理层要求**：所有物理机同 LAN 且网络互通；NPU 互联——节点内 HCCS，节点间 RDMA。

A3 校验流程（结果须全部 `success`、状态 `UP`）：

```bash
# Check the remote switch ports
for i in {0..15}; do hccn_tool -i $i -lldp -g | grep Ifname; done 
# Get the link status of the Ethernet ports (UP or DOWN)
for i in {0..15}; do hccn_tool -i $i -link -g ; done
# Check the network health status
for i in {0..15}; do hccn_tool -i $i -net_health -g ; done
# View the network detected IP configuration
for i in {0..15}; do hccn_tool -i $i -netdetect -g ; done
# View gateway configuration
for i in {0..15}; do hccn_tool -i $i -gateway -g ; done
```

```bash
cat /etc/hccn.conf
```

```bash
# Get virtual NPU IP.
for i in {0..15}; do hccn_tool -i $i -vnic -g;done
```

```bash
for i in {0..15}; do npu-smi info -t spod-info -i $i -c 0;npu-smi info -t spod-info -i $i -c 1;done
```

```bash
# Execute on the target node (replace 'x.x.x.x' with virtual NPU IP address).
for i in {0..15}; do hccn_tool -i $i -hccs_ping -g address x.x.x.x;done
```

```bash
# The TLS settings should be consistent across all nodes
for i in {0..15}; do hccn_tool -i $i -tls -g ; done | grep switch
```

A2 校验流程：

```bash
# Check the remote switch ports
for i in {0..7}; do hccn_tool -i $i -lldp -g | grep Ifname; done
# Get the link status of the Ethernet ports (UP or DOWN)
for i in {0..7}; do hccn_tool -i $i -link -g ; done
# Check the network health status
for i in {0..7}; do hccn_tool -i $i -net_health -g ; done
# View the network detected IP configuration
for i in {0..7}; do hccn_tool -i $i -netdetect -g ; done
# View gateway configuration
for i in {0..7}; do hccn_tool -i $i -gateway -g ; done
```

```bash
cat /etc/hccn.conf
```

```bash
for i in {0..7}; do hccn_tool -i $i -ip -g;done
```

```bash
# Execute on the target node (replace 'x.x.x.x' with actual npu ip address)
for i in {0..7}; do hccn_tool -i $i -ping -g address x.x.x.x;done
```

```bash
# The TLS settings should be consistent across all nodes
for i in {0..7}; do hccn_tool -i $i -tls -g ; done | grep switch
```

### 启动 Docker（每节点一个）

```bash
# Update the vllm-ascend image
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
export NAME=vllm-ascend

# Run the container using the defined variables
# Note: If you are running bridge network with Docker, please expose available ports for multiple nodes communication in advance.
docker run --rm \
--name $NAME \
--net=host \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci4 \
--device /dev/davinci5 \
--device /dev/davinci6 \
--device /dev/davinci7 \
--device /dev/davinci8 \
--device /dev/davinci9 \
--device /dev/davinci10 \
--device /dev/davinci11 \
--device /dev/davinci12 \
--device /dev/davinci13 \
--device /dev/davinci14 \
--device /dev/davinci15 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /mnt/sfs_turbo/.cache:/root/.cache \
-it $IMAGE bash
```

### 安装 Mooncake（多节点同单节点流程）

```shell
git clone -b v0.3.9 --depth 1 https://github.com/kvcache-ai/Mooncake.git
```

```shell
cd Mooncake
sed -i 's|https://go.dev/dl/|https://golang.google.cn/dl/|g' dependencies.sh
```

```shell
apt-get install mpich libmpich-dev -y
```

```shell
bash dependencies.sh -y
```

```shell
mkdir build
cd build
cmake .. -DUSE_ASCEND_DIRECT=ON
make -j
make install
```

```shell
export LD_LIBRARY_PATH=/usr/local/lib64/python3.12/site-packages/mooncake:$LD_LIBRARY_PATH
```

### kv_port 配置指南（重要）

每个 P/D 节点会占用 `kv_port` 到 `kv_port + num_chips` 的端口区间做 socket 监听，须避免冲突；同时各节点 `engine_id` 必须唯一。

在 Ascend NPU 上，Mooncake 使用 `AscendDirectTransport` 做 RDMA，会在 `[20000, 20000 + npu_per_node × 1000)` 区间随机分配端口。若 `kv_port` 与该区间重叠，可能出现间歇性端口冲突。按表配置：

| NPUs per Node | Reserved Port Range | Recommended kv_port |
|---------------|---------------------|---------------------|
| 8             | 20000 - 27999       | >= 28000            |
| 16            | 20000 - 35999       | >= 36000            |

> 警告：若启动期间偶现 `zmq.error.ZMQError: Address already in use`，多半是 `kv_port` 与 AscendDirectTransport 随机端口冲突，请把 `kv_port` 提到保留区间之上。

### 启动脚本

- 使用 `launch_online_dp.py` 启动 external dp vllm servers：[launch_online_dp.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/external_online_dp/launch_online_dp.py)
- 各节点修改 `run_dp_template.sh`：[run_dp_template.sh](https://github.com/vllm-project/vllm-ascend/blob/main/examples/external_online_dp/run_dp_template.sh)

> speculative decoding 约束：Hybrid Mamba（Qwen-Next、Qwen3.5 系列）P 与 D 的 `num_speculative_tokens` 相等；其他模型 P 节点为 1、D 节点 >= 1。

### Layerwise 部署（MooncakeLayerwiseConnector）

#### Prefiller node 1

```shell
nic_name="eth0"  # network card name
local_ip="192.0.0.1"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=256
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
vllm serve /path_to_weight/DeepSeek-r1_w8a8_mtp \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name ds_r1 \
  --max-model-len 40000 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 8 \
  --enforce-eager \
  --trust-remote-code \
  --gpu-memory-utilization 0.9  \
  --quantization ascend \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
  --additional-config '{"enable_shared_expert_dp": true}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeLayerwiseConnector",
  "kv_role": "kv_producer",
  "kv_port": "36000",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 32,
                    "tp_size": 1
             }
      }
  }'
```

#### Prefiller node 2

```shell
nic_name="eth0"  # network card name
local_ip="192.0.0.2"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=256
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
vllm serve /path_to_weight/DeepSeek-r1_w8a8_mtp \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name ds_r1 \
  --max-model-len 40000 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 8 \
  --enforce-eager \
  --trust-remote-code \
  --gpu-memory-utilization 0.9  \
  --quantization ascend \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
  --additional-config '{"enable_shared_expert_dp": true}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeLayerwiseConnector",
  "kv_role": "kv_producer",
  "kv_port": "36100",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 32,
                    "tp_size": 1
             }
      }
  }'
```

#### Decoder node 1

```shell
nic_name="eth0"  # network card name
local_ip="192.0.0.3"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=600
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
vllm serve /path_to_weight/DeepSeek-r1_w8a8_mtp \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name ds_r1 \
  --max-model-len 40000 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 40 \
  --trust-remote-code \
  --gpu-memory-utilization 0.94  \
  --quantization ascend \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
  --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": true,"finegrained_tp_config": {"lmhead_tensor_parallel_size":16}}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeLayerwiseConnector",
  "kv_role": "kv_consumer",
  "kv_port": "36200",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 32,
                    "tp_size": 1
             }
      }
  }'
```

#### Decoder node 2

```shell
nic_name="eth0"  # network card name
local_ip="192.0.0.4"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=600
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
vllm serve /path_to_weight/DeepSeek-r1_w8a8_mtp \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name ds_r1 \
  --max-model-len 40000 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 40 \
  --trust-remote-code \
  --gpu-memory-utilization 0.94  \
  --quantization ascend \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
  --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": true,"finegrained_tp_config": {"lmhead_tensor_parallel_size":16}}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeLayerwiseConnector",
  "kv_role": "kv_consumer",
  "kv_port": "36200",
  "kv_connector_extra_config": {

            "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 32,
                    "tp_size": 1
             }
      }
  }'
```

### Non-layerwise 部署（MooncakeConnectorV1）

#### Prefiller node 1

```shell
nic_name="eth0"  # network card name
local_ip="192.0.0.1"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=256
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
vllm serve /path_to_weight/DeepSeek-r1_w8a8_mtp \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name ds_r1 \
  --max-model-len 40000 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 8 \
  --enforce-eager \
  --trust-remote-code \
  --gpu-memory-utilization 0.9  \
  --quantization ascend \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
  --additional-config '{"enable_shared_expert_dp": true}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_producer",
  "kv_port": "36000",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 32,
                    "tp_size": 1
             }
      }
  }'
```

#### Prefiller node 2

```shell
nic_name="eth0"  # network card name
local_ip="192.0.0.2"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=256
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
vllm serve /path_to_weight/DeepSeek-r1_w8a8_mtp \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name ds_r1 \
  --max-model-len 40000 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 8 \
  --enforce-eager \
  --trust-remote-code \
  --gpu-memory-utilization 0.9  \
  --quantization ascend \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
  --additional-config '{"enable_shared_expert_dp": true}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_producer",
  "kv_port": "36100",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 32,
                    "tp_size": 1
             }
      }
  }'
```

#### Decoder node 1

```shell
nic_name="eth0"  # network card name
local_ip="192.0.0.3"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=600
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
vllm serve /path_to_weight/DeepSeek-r1_w8a8_mtp \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name ds_r1 \
  --max-model-len 40000 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 40 \
  --trust-remote-code \
  --gpu-memory-utilization 0.94  \
  --quantization ascend \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
  --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": true,"finegrained_tp_config": {"lmhead_tensor_parallel_size":16}}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_consumer",
  "kv_port": "36200",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 32,
                    "tp_size": 1
             }
      }
  }'
```

#### Decoder node 2

```shell
nic_name="eth0"  # network card name
local_ip="192.0.0.4"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=600
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
vllm serve /path_to_weight/DeepSeek-r1_w8a8_mtp \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name ds_r1 \
  --max-model-len 40000 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 40 \
  --trust-remote-code \
  --gpu-memory-utilization 0.94  \
  --quantization ascend \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
  --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": true,"finegrained_tp_config": {"lmhead_tensor_parallel_size":16}}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_consumer",
  "kv_port": "36200",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 32,
                    "tp_size": 1
             }
      }
  }'
```

### 启动服务（launch_online_dp.py 调用）

```bash
# on 192.0.0.1
python launch_online_dp.py --dp-size 2 --tp-size 8 --dp-size-local 2 --dp-rank-start 0 --dp-address 192.0.0.1 --dp-rpc-port 12321 --vllm-start-port 7100
# on 192.0.0.2
python launch_online_dp.py --dp-size 2 --tp-size 8 --dp-size-local 2 --dp-rank-start 0 --dp-address 192.0.0.2 --dp-rpc-port 12321 --vllm-start-port 7100
# on 192.0.0.3
python launch_online_dp.py --dp-size 32 --tp-size 1 --dp-size-local 16 --dp-rank-start 0 --dp-address 192.0.0.3 --dp-rpc-port 12321 --vllm-start-port 7100
# on 192.0.0.4
python launch_online_dp.py --dp-size 32 --tp-size 1 --dp-size-local 16 --dp-rank-start 16 --dp-address 192.0.0.3 --dp-rpc-port 12321 --vllm-start-port 7100
```

### 部署 Proxy（两种实现，路由行为不同）

- **`load_balance_proxy_layerwise_server_example.py`**：请求先路由到 D 节点，D 按需再转发到 P 节点。配合 MooncakeLayerwiseConnector。源码：[load_balance_proxy_layerwise_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py)
- **`load_balance_proxy_server_example.py`**：请求先路由到 P 节点，P 处理后再转发到 D 节点。配合 MooncakeConnector。源码：[load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

#### Layerwise Proxy

```shell
python load_balance_proxy_layerwise_server_example.py \
  --port 1999 \
  --host 192.0.0.1 \
  --prefiller-hosts \
    192.0.0.1 \
    192.0.0.1 \
    192.0.0.2 \
    192.0.0.2 \
  --prefiller-ports  \
    7100 7101 7100 7101 \
  --decoder-hosts \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
  --decoder-ports  \
    7100 7101 7102 7103 7104 7105 7106 7107 7108 7109 7110 7111 7112 7113 7114 7115\
    7100 7101 7102 7103 7104 7105 7106 7107 7108 7109 7110 7111 7112 7113 7114 7115\
```

#### Non-layerwise Proxy

```shell
python load_balance_proxy_server_example.py \
  --port 1999 \
  --host 192.0.0.1 \
  --prefiller-hosts \
    192.0.0.1 \
    192.0.0.1 \
    192.0.0.2 \
    192.0.0.2 \
  --prefiller-ports  \
    7100 7101 7100 7101 \
  --decoder-hosts \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
  --decoder-ports  \
    7100 7101 7102 7103 7104 7105 7106 7107 7108 7109 7110 7111 7112 7113 7114 7115\
    7100 7101 7102 7103 7104 7105 7106 7107 7108 7109 7110 7111 7112 7113 7114 7115\
```

|Parameter  | meaning |
| --- | --- |
| --port | Proxy service Port |
| --host | Proxy service Host IP|
| --prefiller-hosts | Hosts of prefiller nodes |
| --prefiller-ports | Ports of prefiller nodes |
| --decoder-hosts | Hosts of decoder nodes |
| --decoder-ports | Ports of decoder nodes |

### Benchmark（推荐 aisbench）

```shell
git clone https://github.com/AISBench/benchmark.git
cd benchmark/
pip3 install -e ./
```

```shell
# unset proxy
unset http_proxy
unset https_proxy
```

数据集与配置目录：`benchmark/ais_bench/datasets`、`benchmark/ais_bench/benchmark/configs/models/vllm_api`，以 `vllm_api_stream_chat.py` 为例：

```python
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="/root/.cache/ds_r1",
        model="dsr1",
        request_rate = 14,
        retry = 2,
        host_ip = "192.0.0.1", # Proxy service host IP
        host_port = 8000,  # Proxy service Port
        max_out_len = 10,
        batch_size=768,
        trust_remote_code=True,
        generation_kwargs = dict(
            temperature = 0,
            seed = 1024,
            ignore_eos=False,
        )
    )
]
```

以 gsm8k 评测为例：

```shell
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf  --debug  --mode perf
```

### FAQ

- **Prefiller 节点需要预热**：部分 NPU 算子需要多轮 warm-up 才能达到最佳性能，正式压测前建议先发若干请求预热。

### 验证

```shell
curl http://192.0.0.1:8080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "ds_r1",
        "prompt": "Who are you?",
        "max_completion_tokens": 100,
        "temperature": 0
    }'
```

---

## Colocated Mooncake Multi Instance

适用场景：PD-Colocated 部署（一个实例同时承担 P 与 D），多实例跨节点共享 KV cache 池，验证跨节点/跨实例 KV 复用性能。示例模型 Qwen2.5-72B-Instruct，两台 Atlas 800T A2，每实例 4 NPU 卡。

### 多节点通信环境校验

**物理层要求**：两节点须通过 RoCE 网络物理互联；缺 RoCE 会显著降低跨节点 KV cache 访问性能。节点内 HCCS，节点间 RoCE。

```bash
# Check the remote switch ports
for i in {0..7}; do hccn_tool -i $i -lldp -g | grep Ifname; done
# Get the link status of the Ethernet ports (UP or DOWN)
for i in {0..7}; do hccn_tool -i $i -link -g ; done
# Check the network health status
for i in {0..7}; do hccn_tool -i $i -net_health -g ; done
# View the network detected IP configuration
for i in {0..7}; do hccn_tool -i $i -netdetect -g ; done
# View gateway configuration
for i in {0..7}; do hccn_tool -i $i -gateway -g ; done
```

```bash
cat /etc/hccn.conf
```

```bash
for i in {0..7}; do hccn_tool -i $i -ip -g;done
```

```bash
# Execute the following command on each node, replacing x.x.x.x
# with the target node's NPU card address.
for i in {0..7}; do hccn_tool -i $i -ping -g address x.x.x.x; done
```

```bash
# The tls settings should be consistent across all nodes.
for i in {0..7}; do hccn_tool -i $i -tls -g ; done | grep switch
```

### 启动 Docker（每节点，4 卡）

```bash
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
export NAME=vllm-ascend

# Run the container using the defined variables
# This test uses four NPU cards to create the container.
# Mount the hccn.conf file from the host node into the container.
docker run --rm \
--name $NAME \
--net=host \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:\
/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /root/.cache:/root/.cache \
-it $IMAGE bash
```

### （可选）安装 Mooncake

新版镜像已预装 Mooncake，下列步骤可选。

```bash
git clone -b v0.3.9 --depth 1 https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
git submodule update --init --recursive
```

```bash
apt-get install mpich libmpich-dev -y
```

```bash
bash dependencies.sh -y
```

```bash
mkdir build
cd build
cmake .. -DUSE_ASCEND_DIRECT=ON
make -j
make install
```

校验安装：

```bash
python -c "import mooncake; print(mooncake.__file__)"
# Expected output path:
# /usr/local/Ascend/ascend-toolkit/latest/python/
# site-packages/mooncake/__init__.py
```

### 启动 Mooncake Master 服务

在其中一节点容器内启动：

```bash
docker exec -it vllm-ascend bash
cd /vllm-workspace/Mooncake
mooncake_master --port 50088 \
  --eviction_high_watermark_ratio 0.95 \
  --eviction_ratio 0.05
```

| Parameter                     | Value | Explanation                           |
| ----------------------------- | ----- | ------------------------------------- |
| port                          | 50088 | Port for the master service           |
| eviction_high_watermark_ratio | 0.95  | High watermark ratio (95% threshold)  |
| eviction_ratio                | 0.05  | Percentage to evict when full (5%)    |

### 创建 mooncake.json 配置

模板：

```json
{
    "metadata_server": "P2PHANDSHAKE",
    "protocol": "ascend",
    "device_name": "",
    "master_server_address": "<your_server_ip>:50088",
    "global_segment_size": 107374182400
}
```

| Parameter   | Value                  | Explanation                           |
| --------------| ------------------------| -----------------------------------|
| metadata_server | P2PHANDSHAKE              | Point-to-point handshake mode  |
| protocol              | ascend              | Ascend proprietary protocol    |
| master_server_address | 90.90.100.188:50088(for example) | Master server address|
| global_segment_size   | 107374182400    | Size per segment (100GB)      |

### vLLM 实例部署

节点 1 实例用 NPU [0-3]，节点 2 实例用另一台服务器的 [0-3]，验证跨节点跨实例 KV cache 复用与性能。

#### 部署实例 1

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/\
latest/python/site-packages:$LD_LIBRARY_PATH
export MOONCAKE_CONFIG_PATH="/vllm-workspace/mooncake.json"
# NPU buffer pool: quantity:size(MB)
# Allocates 4 buffers of 8MB each for KV transfer
export ASCEND_BUFFER_POOL=4:8

vllm serve <path_to_your_model>/Qwen2.5-72B-Instruct/ \
--served-model-name qwen \
--dtype bfloat16 \
--max-model-len 25600 \
--tensor-parallel-size 4 \
--host <your_server_ip> \
--port 8002 \
--max-num-batched-tokens 4096 \
--gpu-memory-utilization 0.9 \
--kv-transfer-config '{
      "kv_connector": "MooncakeConnectorStoreV1",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
          "use_layerwise": false,
          "mooncake_rpc_port": "0",
          "load_async": true,
          "register_buffer": true
      }
  }'
```

#### 部署实例 2

与实例 1 完全一致，仅按实例 2 配置修改 `--host` 与 `--port`。

#### 配置参数

| Parameter         | Value                 | Explanation                      |
| ----------------- | ----------------------| -------------------------------- |
| kv_connector      | MooncakeConnectorStoreV1 | Use StoreV1 version           |
| kv_role         | kv_both                | Enable both produce and consume  |
| use_layerwise     | false                | Transfer entire cache (see note) |
| mooncake_rpc_port | 0                    | Automatic port assignment        |
| load_async        | true                 | Enable asynchronous loading      |
| register_buffer   | true                 | Required for PD-colocated mode   |

**use_layerwise 说明**：

- `false`：传输完整 KV Cache（适合带宽充足的跨节点场景）
- `true`：逐层传输（适合单节点显存受限场景）

### Benchmark（三步法，全随机数据集）

数据集 A：input/output tokens 1024/10，共 100 请求，并发 25。

- **Step 1 Baseline（无 cache 命中）**：把 A 发给节点 1 实例 1，记录 TTFT 为 **TTFT1**。
- **准备 Step 2**：先向实例 1 发全随机数据集 B；统一片上内存/DRAM KV cache 采用 LRU 驱逐，B 把 A 从片上内存驱逐，A 仅留在节点 1 DRAM。
- **Step 2 本地 DRAM 命中**：再次向实例 1 发 A，记录 **TTFT2**。
- **Step 3 跨节点 DRAM 命中**：把 A 发给实例 2；经 Mooncake KV cache 池，从节点 1 DRAM 跨节点命中，记录 **TTFT3**。

模型配置：

```python
from ais_bench.benchmark.models import VLLMCustomAPIChatStream
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="<path_to_your_model>/Qwen2.5-72B-Instruct",
        model="qwen",
        request_rate = 0,
        retry = 2,
        host_ip = "<your_server_ip>",
        host_port = 8002,
        max_out_len = 10,
        batch_size= 25,
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 0,
            ignore_eos = True,
        ),
    )
]
```

压测命令：

```shell
ais_bench --models vllm_api_stream_chat \
  --datasets gsm8k_gen_0_shot_cot_str_perf \
  --debug --summarizer default_perf --mode perf
```

测试结果示例：

| Requests | Concur | TTFT1 (ms) | TTFT2 (ms) | TTFT3 (ms) |
| -------- | ------ | ---------- | ---------- | ---------- |
| 100      | 25     | 2322       | 739        | 948        |

---

## Disaggregated Prefill (Design Doc)

### 为什么需要 disaggregated prefill

针对大规模推理中 TPOT 与 TTFT 的优化，动机有二：

1. **为 P 与 D 节点灵活调整并行策略与实例数**：可分别对 P、D 设置 dp / tp / ep 及实例数，对 TTFT 与 TPOT 做更精细调优。
2. **优化 TPOT**：不分离时 prefill 会插入 decode 流，导致效率与延迟问题；分离后能更好地控制系统 TPOT，避免 chunk size 的取舍难题，使输出 token 时间更可控。

### 用法

vLLM Ascend 当前支持两类 KV cache 管理 connector：

- **MooncakeConnector**：D 节点从 P 节点 pull KV cache。
- **MooncakeLayerwiseConnector**：P 节点以分层方式向 D 节点 push KV cache。

部署指南参见 [PD disaggregation multi-node deployment guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/features/pd_disaggregation_mooncake_multi_node.html)。

### 工作原理

#### 1. 设计思路

disaggregated-prefill 下，一个全局 Proxy 接收外部请求：prefill 转发到 P 节点，decode 转发到 D 节点；P 与 D 之间通过 P2P 通信交换 KV cache。

设计图示意 pull 与 push 两种方案（见原文 `assets/disaggregated_prefill_pull.png` / `disaggregated_prefill_push.png`）。

#### 2. 实现设计

##### Mooncake Connector（pull 方案）

1. 请求发到 Proxy 的 `_handle_completions`。
2. Proxy 调用 `select_prefiller` 选 P 节点转发请求，配置 `kv_transfer_params`：`do_remote_decode=True`、`max_completion_tokens=1`、`min_tokens=1`。
3. P 节点 scheduler 完成 prefill 后，`update_from_output` 调用 schedule connector 的 `request_finished` 延迟 KV cache 释放，构造 `kv_transfer_params` 设 `do_remote_prefill=True`，返回 Proxy。
4. Proxy 调用 `select_decoder` 选 D 节点转发请求。
5. D 节点 scheduler 标记 `RequestStatus.WAITING_FOR_REMOTE_KV`，预分配 KV cache，调 `kv_connector_no_forward` 拉取远端 KV cache，再通知 P 节点释放 KV cache，继续 decode 返回结果。

##### Mooncake Layerwise Connector（push 方案）

1. 请求发到 Proxy 的 `_handle_completions`。
2. Proxy 调用 `select_decoder` 选 D 节点转发请求，配置 `kv_transfer_params`：`do_remote_prefill=True` 并设置 `metaserver` endpoint。
3. D 节点 scheduler 用 `kv_transfer_params` 标记 `RequestStatus.WAITING_FOR_REMOTE_KV`，预分配 KV cache，调 `kv_connector_no_forward` 向 metaserver 发请求并等待 KV cache 传输完成。
4. Proxy 的 `metaserver` endpoint 收到请求后，调 `select_prefiller` 选 P 节点，转发时 `kv_transfer_params` 设 `do_remote_decode=True`、`max_completion_tokens=1`、`min_tokens=1`。
5. 处理过程中 P 节点 scheduler 逐层 push KV cache；全部层推送完成后释放请求并通知 D 节点开始 decode。
6. D 节点执行 decode 并返回结果。

#### 3. 接口设计（以 MooncakeConnector 为例）

- **MooncakeConnector**：基类，提供核心接口。
- **MooncakeConnectorScheduler**：engine core 内调度 connector 的接口，管理 KV cache 传输需求与完成。
- **MooncakeConnectorWorker**：worker 进程中管理 KV cache 注册与传输的接口。

#### 4. 规格设计

支持 MLA 与 GQA 模型；兼容 A2、A3；支持等 TP 与部分不等 TP 的多 P/D 节点场景。

| Feature                       |      Status    |
|-------------------------------|----------------|
| A2                            | 🟢 Functional  |
| A3                            | 🟢 Functional  |
| equal TP configuration        | 🟢 Functional  |
| unequal TP configuration      | 🟢 Functional  |
| MLA                           | 🟢 Functional  |
| GQA                           | 🟢 Functional  |

状态含义：🟢 Functional 完全可用并持续优化；🔵 Experimental 实验性支持，接口可能变更；🚧 WIP 开发中即将支持；🟡 Planned 计划中；🔴 NO plan/Deprecated 无计划或已废弃。

### DFX 分析

1. **Config 参数校验**：检查 `kv_connector` 类型是否支持；传输失败时输出清晰错误日志。
2. **端口冲突检测**：启动前对配置端口（`rpc_port`、`metrics_port`、`http_port/metaserver`）做 bind 探测，已占用则 fail fast 并报错。
3. **PD 比例校验**：非对称 PD 场景下，校验 P↔D 的 tp 比例是否符合预期与调度约束。

### 限制

- 不支持异构 P/D（如 P 在 A2、D 在 A3）。
- 非对称 TP 仅支持 P 节点 TP 高于 D 节点且 P TP 为 D TP 整数倍（即 `P_tp > D_tp` 且 `P_tp % D_tp == 0`）。

---

## EPD Disaggregation（Encoder-Prefill-Decode）

EPD 是 PD 分离在多模态模型上的扩展：把 vision encoder 阶段也独立到单独 vLLM 实例。

- **Encoder 实例**：执行 vision 编码。
- **PD 实例**：运行语言 prefill + decode，可以是单实例 (E + PD) 或全分离 (E + P + D)。
- **ECConnector** 负责在 encoder 与 PD 实例间传递 encoder-cache（EC）embedding。`Scheduler role` 检查 cache 是否存在并调度加载；`Worker role` 把 embedding 载入显存。相关代码位于 `vllm/distributed/ec_transfer`。
- **EPD Load Balancing Proxy**：多路径调度策略（动态把多模态/文本请求分流到对应推理路径）+ 实例级动态负载均衡（基于活跃 token 负载的最小负载优先队列）。

### 为什么需要 disaggregated encoder

1. **独立细粒度扩缩容**：vision encoder 轻量，language model 大数个量级；语言模型可独立并行，不影响 encoder 集群；encoder 节点可独立增减。
2. **更低 TTFT**：纯文本请求完全绕过 vision encoder；encoder 输出仅在需要的 attention 层注入，缩短 prefill 关键路径。
3. **跨进程复用与缓存 encoder 输出**：进程内 encoder 复用局限于单 worker；远端共享 cache 让任意 worker 取已有 embedding，消除重复计算。

设计文档：<https://docs.google.com/document/d/1aed8KtC6XkXtdoV87pWT0a8OJlZ-CpnuLLzmR8l9BAE/edit>

### 用法

当前参考实现是 **ExampleConnector**。开箱即用脚本：

- 1 Encoder + 1 PD 实例：`examples/online_serving/disaggregated_encoder/disagg_1e1pd/`
- 1 Encoder + 1 Prefill + 1 Decode 实例：`examples/online_serving/disaggregated_encoder/disagg_1e1p1d/`

### 与 PD 分离的关系

vLLM-Ascend 中，EPD 默认使用 `MooncakeLayerwiseConnector`（`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py`）完成 P↔D 的 KV 传递，并参考 `examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py`。

使用 MooncakeLayerwiseConnector 的 PD 分离流程：请求先进入 Decoder 实例，Decoder 通过 Metaserver 反向触发远端 prefill；Prefill 节点执行推理并按层 push KV cache 到 Decoder，与计算重叠；传输完成后 Decoder 无缝续接后续 token 生成。设计思路见 `docs/source/developer_guide/Design_Documents/disaggregated_prefill.md`。

### 限制

- 若要使用跨进程缓存，需关闭 `--mm-processor-cache-gb 0`（即不要设为 0）。
- PD 分离部分遵循 PD 分解的限制（不支持异构 P/D；非对称 TP 仅支持 P_tp > D_tp 且 P_tp % D_tp == 0）。

---

## 源文件清单

| 源文件路径 | 状态 |
|------------|------|
| docs/source/tutorials/features/index.md | OK |
| docs/source/tutorials/features/pd_disaggregation_mooncake_single_node.md | OK |
| docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md | OK |
| docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md | OK |
| docs/source/user_guide/feature_guide/epd_disaggregation.md | OK |
| docs/source/developer_guide/Design_Documents/disaggregated_prefill.md | OK |
