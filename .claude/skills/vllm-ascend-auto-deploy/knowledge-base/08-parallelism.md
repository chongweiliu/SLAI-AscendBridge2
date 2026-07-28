> 来源：vllm-ascend docs（main 分支，抓取于 2026-07-28），整合自多个源文件

# 08 并行与部署优化

本篇整合 vllm-ascend 在并行策略与部署侧优化方面的 8 个特性文档：Context Parallel、Sequence Parallelism、Fine-grained TP、Dynamic Chunk Pipeline Parallel、Large Scale EP、Expert Parallelism Load Balancer、EPLB Swift Balancer 设计文档、CPU Binding。涵盖原理、适用场景、硬件/版本要求、启用方式与关键参数。

---

## Context Parallel

### 是什么

Decode Context Parallel (DCP) 在 Tensor Parallel (TP) 组内沿序列维度切分 KV cache，消除冗余 KV-cache 拷贝，可提升长上下文解码的可用 batch size。

注意：vLLM Ascend **不支持** Prefill Context Parallel，上游 `prefill_context_parallel_size` 必须保持默认值 `1`。DSA-CP 是独立的稀疏注意力优化，由 `additional_config.enable_dsa_cp` 控制。

### 适用场景与要求

DCP 支持 eager / graph 执行、prefix caching、chunked prefill、speculative decoding、P/D disaggregation、MLAPO（在 vLLM Ascend 文档列出的模型与硬件组合下）。SFA attention backend 支持投机解码；MLA 与 GQA attention backend 仅在 P/D disaggregation 部署场景下支持投机解码，混合部署不支持。

约束：
- MLA 模型（如 DeepSeek-R1）：`tensor_parallel_size >= decode_context_parallel_size`，且 `tensor_parallel_size % decode_context_parallel_size == 0`
- GQA 模型（如 Qwen3-235B）：`(tensor_parallel_size // num_key_value_heads) >= decode_context_parallel_size`，且 `(tensor_parallel_size // num_key_value_heads) % decode_context_parallel_size == 0`
- KV-cache 传输场景（KV pooling 或 P/D disaggregation）需设 `cp_kv_cache_interleave_size` 为 KV-cache `block_size`（默认 128）

DCP 复用 TP 设备，**不增加 world size**。

### 如何启用

离线示例：

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="deepseek-ai/DeepSeek-V2-Lite",
    tensor_parallel_size=2,
    decode_context_parallel_size=2,
)
outputs = llm.generate(prompts, sampling_params)
```

在线示例：

```bash
vllm serve deepseek-ai/DeepSeek-V2-Lite \
    --tensor-parallel-size 2 \
    --decode-context-parallel-size 2
```

KV-cache 传输场景：

```shell
vllm serve deepseek-ai/DeepSeek-V2-Lite \
    --tensor-parallel-size 2 \
    --decode-context-parallel-size 2 \
    --cp-kv-cache-interleave-size 128 \
    --kv-transfer-config '{...}'
```

---

## Sequence Parallelism

### 是什么

Sequence Parallelism (SP) 最早由 Megatron 提出，核心是把 `Allreduce->LayerNorm` 改为 `ReduceScatter->LayerNorm->Allgather`，用于推理场景。单纯拆分 Allreduce 收益有限；SP 的真正收益来自：1）INT8 量化场景下 Allgather 通信量减半；2）ReduceScatter/Allgather 可与前后 Matmul 融合成通信-计算并行算子，降低延迟。

### 适用场景与要求

vllm-ascend 已为 VL 类模型实现基于 Inductor pass 的 SP。SP 依赖 graph mode，**不支持 eager mode**。当前不支持量化（适配中）。

支持矩阵（无量化）：

|                      |  VL + Dense | VL + MoE | non-VL + Dense | non-VL + MoE |
| -------------------- | ----------- | -------- | -------------- | ------------ |
| Sequence Parallelism  | x           | x        | x              | x            |
| Flash Comm V1         | eager/graph | eager/graph | eager/graph | eager/graph  |

> 注：表中 "x" 表示 SP 在该列不支持/未适配；FC1 在所有组合均支持 eager/graph。

### 如何启用

```bash
vllm serve Qwen/Qwen3-VL-2B-Instruct \
    --tensor-parallel-size 2 \
    --compilation-config '{"pass_config": {"enable_sp": true , "sp_min_token_num": 1000}}'
```

参数：
- `"enable_sp"`：SP 开关，依赖 graph mode
- `sp_min_token_num`：token 数小于该值时 SP 反而带来负收益（通信量小时通信算子固定开销占比过大）。Ascend 默认 `1000`，一般无需修改。自定义示例：`--compilation-config '{"pass_config": {"enable_sp": true, "sp_min_token_num": 512}}'`。该值会被追加进 `compile_ranges_split_points`，按区间划分图编译范围并逐段检查 pass 是否适用

最简（推荐）启用方式（不改 `sp_min_token_num`）：

```bash
vllm serve Qwen/Qwen3-VL-2B-Instruct \
    --tensor-parallel-size 2 \
    --compilation-config '{"pass_config": {"enable_sp": true}}'
```

### SP 与 Flash Comm V1 的区别

Flash Comm V1 (FC1) 是 NPU 上 SP 的增强版：1）MLA 结构下 Allgather 推迟到 QKV projection 之后；2）MoE 模型下 Allgather 推迟到 Gating+DynamicQuant 之后，进一步降通信量。FC1 基于 Custom OP 实现，难以支持 VL 类模型，因此 FC1 与 SP 互补。

### Pass 设计

启用 SP 时依次运行 `SequenceParallelismPass` 与 `SequenceParallelismMoePass`。

`SequenceParallelismPass` 先跑 `NoOpEliminationPass` 消除冗余 view 类操作，再应用 AllReduce 模式：

| Pattern                                | Match                            | Replacement                                                                           |
| -------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------- |
| `MiddleAllReduceRMSNormPattern`        | `all_reduce` + `layernorm`       | `reduce_scatter` + `layernorm` + `all_gather`                                         |
| `LastAllReduceRMSNormPattern`          | Same (last layer, no residual)   | Same                                                                                  |
| `Qwen3VLMiddleAllReduceRMSNormPattern` | `all_reduce` + add + `layernorm` | `reduce_scatter` + chunk(`deepstack_input_embeds`) + add + `layernorm` + `all_gather` |

Qwen3-VL 中间层在 `all_reduce` 与 `layernorm` 间插入 `hidden_states=hidden_states + deepstack_input_embeds`；SP 下 `hidden_states` 被 reduce_scatter 到 `[seq_len/tp, hidden]`，而 `deepstack_input_embeds` 来自视觉/deepstack 路径保持全长 `[seq_len, hidden]`，需按 `tp_size` chunk `deepstack_input_embeds` 保证形状一致。

`SequenceParallelismMoePass` 处理：1）推迟 allgather 到 layernorm 之后（`MiddleLayerAllgatherAddRMSNormPattern` / `LastLayerAllgatherRMSNormPattern` / `Qwen3VLMiddleLayerAllgatherAddRMSNormPattern`）；2）`AllGatherChunkNoOpPattern` 消除 `all_gather` + `sequence_parallel_chunk_impl` 冗余 no-op。

### FAQ

SP 默认不启用，处于实验阶段，未来会默认开启。代码流：`pass_config` 中 `enable_sp` 与 `sp_min_token_num` 默认 `None`；`NPUPlatform.apply_config_platform_defaults` 在 `enable_sp=True` 且 `sp_min_token_num=None` 时设默认值（Dense 1000，MoE 1）；`VllmConfig._apply_optimization_level_defaults` 对 dense 模型设 `enable_sp=True`；`VllmConfig.__post_init__` 若 `sp_min_token_num` 仍为 `None` 则 `enable_sp` 置 `False`。

---

## Fine-grained TP

### 是什么

Fine-Grained Tensor Parallelism (Fine-grained TP) 扩展标准 TP，允许为不同模型组件配置**独立的 tensor-parallel size**。通过 `finegrained_tp_config` 为 embedding、lm_head、o_proj、MLP 等模块分别设 TP size，而非全局统一 `tensor_parallel_size`。

### 收益

- 降低单卡显存占用：将大权重矩阵（LM Head、o_proj）分片到多卡，降低峰值显存、支撑更大 batch，无需量化
- 加速 GEMM 显存访问：decode 负载常为 memory-bound，权重分片减少单卡权重搬运量，提升带宽效率（LM Head、o_proj 等延迟敏感层尤为明显）

### 适用场景与要求

模型无关，支持所有标准 dense transformer 架构（Llama、Qwen、DeepSeek base/dense 等）。

组件与执行模式支持：

| TP config     | Eager | Graph | Hybrid | Prefill | Decode |
| ------------- | ----- | ----- | ------ | ------- | ------ |
| **embedding** | ✅     | ✅     | ✅      | ✅       | ✅      |
| **o_proj**    | ❌     | ✅     | ❌      | ❌       | ✅      |
| **mlp**       | ✅     | ✅     | ✅      | ✅       | ✅      |
| **LMhead**    | ✅     | ✅     | ✅      | ✅       | ✅      |

> 注意：`o_proj` TP 仅在 Graph mode Decode 下支持（eager mode 下 dummy_run 不触发 o_proj）。`mlp` TP 支持 dense 模型或 MoE 模型的 dense 层（如 DeepSeek-R1 前三层 dense 层）。

配置限制：任一组件的 Fine-grained TP size 必须 `≤ DP size`，且 `dp_size % tp_size == 0`。

### 如何启用

通过 `--additional-config` 的 `finegrained_tp_config` 字段控制：

```bash
--additional-config '{
    "finegrained_tp_config": {
        "embedding_tensor_parallel_size": 8,
        "lmhead_tensor_parallel_size": 8,
        "oproj_tensor_parallel_size": 8,
        "mlp_tensor_parallel_size": 8
    }
}'
```

完整示例（DeepSeek-R1，DP16 + EP32，Fine-grained TP size 8）：

```bash
vllm serve deepseek-ai/DeepSeek-R1 \
    --data-parallel-size 16 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --additional-config '{
        "finegrained_tp_config": {
            "embedding_tensor_parallel_size": 8,
            "lmhead_tensor_parallel_size": 8,
            "mlp_tensor_parallel_size": 8
        }
    }'
```

### 部署建议

Fine-grained TP 在 **PD 分离的 decode 实例**上效果最佳（模型通常以全 DP 方式部署，分片权重重的层可减少冗余存储与显存压力）。

实验数据（DeepSeek-R1-W8A8，32 卡 Atlas A2 64G，DP32+EP32，Fine-grained TP size 8）：

| Module           | Memory Savings | TPOT Impact (batch=24)    |
| ---------------- | -------------- | ------------------------- |
| o_proj TP = 8    | 5.8 GB         | +1.5 ms (degradation) |
| LM head TP = 8   | 1.51 GB        | −1.2 ms (improvement) |
|  FFN TP = 8 | 0.9 GB         | −1.0 ms (improvement) |
| Embedding TP = 8 | 1.51 GB        | −1.0 ms (improvement) |
| **Total**        | **9.72 GB**    | —                         |

---

## Dynamic Chunk Pipeline Parallel

### 是什么

Dynamic Chunked Pipeline Parallel (CPP) 是基于 profiling 的动态分块策略，优化 Pipeline Parallelism (PP) 场景下长序列 prefill 性能。**CPP 设计用于 PD 分离部署中的 Prefiller (P) 节点**，通过 profiling 数据动态计算最优 chunk size，显著降低长序列 P 节点的 TTFT。

### 适用场景与要求

- **PD disaggregation P 节点**：在 Prefiller 上启用 CPP 优化长序列 prefill；Decoder 节点无需 CPP
- 变长序列服务：短序列无退化，长序列通过动态分块获益
- 超长序列推理：超过单机显存的序列（如 1M tokens），动态分块显著降低 pipeline 空闲

支持矩阵（聚焦 PD disaggregation P 节点 prefill 阶段）：

|         | Eager | Graph | Prefix <br> Cache | Chunked <br> Prefill |
| ------- | ----- | ----- | ------ | ------ |
| **CPP** | ✅    | ✅     | ✅      | ✅       |

约束：
- `--pipeline-parallel-size > 1`
- `--enable-chunked-prefill`
- 与 Balance Scheduling 不兼容：不能启用 `VLLM_ASCEND_BALANCE_SCHEDULING`
- 启动开销：profiling 约需 64 次前向（数十秒）

### 如何启用（PD disaggregation 1P1D 示例）

注意：
- `async-scheduling` 在 PP prefill 阶段可能造成性能退化，且对 prefill 收益甚微，建议 P 节点不开 async scheduling
- 推荐使用 `MooncakeConnectorV1` 作为 `kv_connector`，对 PP 支持更完整

P 节点（Prefiller，带 CPP）：

```shell
# For nic_name, run the `ifconfig` command to check the network adapter whose IP address is the same as that of the local host.
nic_name=<COMMAND_RESULT>
local_ip=<YOUR_MACHINE_IP>

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name 
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

vllm serve Qwen/Qwen3-30B-A3B \
    --host 0.0.0.0 \
    --port 13700 \
    --served-model-name "qwen" \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2 \
    --enforce-eager \
    --max-model-len 131072 \
    --max-num-batched-tokens 32768 \
    --enable-prefix-caching \
    --no-async-scheduling \
    --additional-config '{"scheduler_config": {"profiling_chunk_config": {"enabled": true}}}' \
    --kv-transfer-config \
    '{
        "kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_producer",
        "kv_port": "30000",
        "engine_id": "0",
        "kv_connector_extra_config": {
            "prefill": {
                "pp_size": 2,
                "dp_size": 1,
                "tp_size": 2
            },
            "decode": {
                "dp_size": 2,
                "tp_size": 2
            }
        }
    }'
```

D 节点（Decoder，不带 CPP）：

```shell
# For nic_name, run the `ifconfig` command to check the network adapter whose IP address is the same as that of the local host.
nic_name=<COMMAND_RESULT>
local_ip=<YOUR_MACHINE_IP>

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name 
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

vllm serve Qwen/Qwen3-30B-A3B \
    --host 0.0.0.0 \
    --port 13701 \
    --served-model-name "qwen" \
    --data-parallel-size 2 \
    --tensor-parallel-size 2 \
    --enable-prefix-caching \
    --max-model-len 131072 \
    --max-num-batched-tokens 256 \
    --gpu-memory-utilization 0.9 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --kv-transfer-config \
    '{
        "kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_consumer",
        "kv_port": "30000",
        "engine_id": "0",
        "kv_connector_extra_config": {
            "prefill": {
                "pp_size": 2,
                "dp_size": 1,
                "tp_size": 2
            },
            "decode": {
                "dp_size": 2,
                "tp_size": 2
            }
        }
    }'
```

代理服务器（与 Prefiller 同节点，程序见仓库 examples `load_balance_proxy_server_example.py`）：

```shell
python load_balance_proxy_server_example.py \
    --host <PROXY_IP> \
    --port 8080 \
    --prefiller-hosts <PREFILL_MACHINE_IP> \
    --prefiller-port 13700 \
    --decoder-hosts <DECODE_MACHINE_IP> \
    --decoder-ports 13701
```

健康检查：

```shell
curl http://<PROXY_IP>:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen",
        "messages": [
        {
            "role": "system",
            "content": "You are a useful AI assistant."
        },
        {
            "role": "user",
            "content": "Question: ... How much does she make?\nAnswer:"
        }
        ],
        "max_completion_tokens": 100,
        "temperature": 0
    }'
```

### 配置参数

`profiling_chunk_config` 字段：

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | False | Enable/disable Dynamic Chunked Pipeline Parallel |
| `smooth_factor` | float | 1.0 | Smoothing factor (0 < x ≤ 1.0). Higher values trust dynamic prediction more |
| `min_chunk` | int | 4096 | Minimum chunk size for dynamic calculation |
| `need_timing` | bool | True | Enable/disable Online Calibration |
| `max_fit_chunk` | int | 30 | Number of chunk-time data for Online Calibration |

调参：
- `smooth_factor`：`1.0` 严格遵循模型预测；`0.6~0.85` 平衡动态调整与调度开销；`0.0` 退化为固定分块
- `min_chunk`：一般无需调整，应小于 `max-num-batched-tokens`

### 推荐设置

`max-num-batched-tokens`（CPP 动态求解的初始 chunksize，对 TTFT 敏感）：

| Sequence Length | `max-num-batched-tokens` |
|-----------------|--------------------------|
| 64k             | 20480                    |
| 128k            | 32768                    |

Online Calibration 可用 aisbench 生成定长随机数据集，校准数据长度与 `max-model-len` 对齐，用 `batch_size=1`，开启 prefix caching 时数据需互异以避免缓存命中。

### 性能

DeepSeek-V3.1-W8A8 / Qwen3-235B，P 实例部署于 Atlas A3 64G：

- DeepSeek-V3.1-W8A8（定长，并发 1，输入 128k）：CPP TTFT 22.5s vs 静态 PP 27.0s
- Qwen3-235B（定长，并发 1，输入 256k）：CPP TTFT 53.5s vs 静态 PP 61.4s
- DeepSeek-V3.1-W8A8（变长 4k~64k，均值 32k，并发 4，prefix hit 99%）：CPP2TP8 输入吞吐 22424 tps/card，DP2TP8 16150 tps/card，TP16 18875 tps/card

---

## Large Scale EP

### 是什么

vLLM-Ascend 支持大规模 Expert Parallelism (EP) 场景下的 Prefill-Decode (PD) disaggregation，采用分布式 DP server 实现更优性能。P/D 节点可基于各自特性采用不同优化策略。示例：DeepSeek 模型用 8 台 Atlas 800T A3 服务器部署，前 4 台作 prefiller，后 4 台作 decoder，prefiller 各自独立作 master，decoder 以 192.0.0.5 为 master。

### 物理层要求

- 同一 LAN，网络互通
- 所有 NPU 互联：Atlas A2 代节点内 HCCS、节点间 RDMA；Atlas A3 代节点内与节点间均 HCCS

### 多机通信环境验证

A3 单节点验证（每节点 0..15 卡）：

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
# View NPU network configuration
cat /etc/hccn.conf
```

A3 获取 NPU IP / superpodid：

```bash
for i in {0..15}; do hccn_tool -i $i -vnic -g;done
for i in {0..15}; do npu-smi info -t spod-info -i $i -c 0;npu-smi info -t spod-info -i $i -c 1;done
```

A3 跨节点 PING（替换 `x.x.x.x` 为目标 NPU IP）：

```bash
for i in {0..15}; do hccn_tool -i $i -hccs_ping -g address x.x.x.x;done
```

A2 单节点验证（每节点 0..7 卡）：

```bash
for i in {0..7}; do hccn_tool -i $i -lldp -g | grep Ifname; done
for i in {0..7}; do hccn_tool -i $i -link -g ; done
for i in {0..7}; do hccn_tool -i $i -net_health -g ; done
for i in {0..7}; do hccn_tool -i $i -netdetect -g ; done
for i in {0..7}; do hccn_tool -i $i -gateway -g ; done
cat /etc/hccn.conf
```

A2 获取 NPU IP：

```bash
for i in {0..7}; do hccn_tool -i $i -ip -g;done
```

A2 跨节点 PING：

```bash
for i in {0..7}; do hccn_tool -i $i -ping -g address x.x.x.x;done
```

### 部署脚本

Prefiller 节点 `run_dp_template.sh`：

```shell
# run_dp_template.sh
#!/bin/sh

# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip
nic_name="xxxx"
local_ip="xxxx"

# basic configuration for HCCL and connection
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export HCCL_BUFFSIZE=256

# obtain parameters from distributed DP server
export VLLM_DP_SIZE=$1
export VLLM_DP_MASTER_IP=$2
export VLLM_DP_MASTER_PORT=$3
export VLLM_DP_RANK_LOCAL=$4
export VLLM_DP_RANK=$5
export VLLM_DP_SIZE_LOCAL=$7

#pytorch_npu settings and vllm settings
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_USE_MODELSCOPE="True"

# enable the distributed DP server
export VLLM_WORKER_MULTIPROC_METHOD="fork"
export VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED=1

# The w8a8 weight can be obtained from https://www.modelscope.cn/models/vllm-ascend/DeepSeek-R1-W8A8
# "--additional-config" is used to enable characteristics from vllm-ascend
vllm serve vllm-ascend/DeepSeek-R1-W8A8 \
    --host 0.0.0.0 \
    --port $6 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name deepseek_r1 \
    --max-model-len 17000 \
    --max-num-batched-tokens 16384 \
    --trust-remote-code \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeConnectorV1",
      "kv_buffer_device": "npu",
      "kv_role": "kv_producer",
      "kv_parallel_size": "1",
      "kv_port": "20001",
    }' \
    --additional-config '{"enable_weight_nz_layout":true,"enable_prefill_optimizations":true}'
```

Decoder 节点 `run_dp_template.sh`（`HCCL_BUFFSIZE=1024`，`--tensor-parallel-size 1`，`--max-num-batched-tokens 256`，无 `enable_prefill_optimizations`）：

```shell
# run_dp_template.sh
#!/bin/sh

nic_name="xxxx"
local_ip="xxxx"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export HCCL_BUFFSIZE=1024

export VLLM_DP_SIZE=$1
export VLLM_DP_MASTER_IP=$2
export VLLM_DP_MASTER_PORT=$3
export VLLM_DP_RANK_LOCAL=$4
export VLLM_DP_RANK=$5
export VLLM_DP_SIZE_LOCAL=$7

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_USE_MODELSCOPE="True"

export VLLM_WORKER_MULTIPROC_METHOD="fork"
export VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED=1

vllm serve vllm-ascend/DeepSeek-R1-W8A8 \
    --host 0.0.0.0 \
    --port $6 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name deepseek_r1 \
    --max-model-len 17000 \
    --max-num-batched-tokens 256 \
    --trust-remote-code \
    --max-num-seqs 28 \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
    --kv-transfer-config \
        '{"kv_connector": "MooncakeConnectorV1",
        "kv_buffer_device": "npu",
        "kv_role": "kv_consumer",
        "kv_parallel_size": "1",
        "kv_port": "20001",
        }' \
    --additional-config '{"enable_weight_nz_layout":true}'
```

### 启动分布式 DP server

在各节点执行 Python 启动脚本（推荐 v0.9.1 正式版使用）。Prefiller 节点示例（`dp_size=2`, `dp_size_local=2`, `dp_rank_start=0`, `dp_ip="192.0.0.1"`, `dp_port=13395`, `engine_port=9000`）：

```python
import multiprocessing
import os
import sys
dp_size = 2 # total number of DP engines for decode/prefill
dp_size_local = 2 # number of DP engines on the current node
dp_rank_start = 0 # starting DP rank for the current node
# dp_ip is different on prefiller nodes in this example
dp_ip = "192.0.0.1" # master node IP for DP communication
dp_port = 13395 # port used for DP communication
engine_port = 9000 # starting port for all DP groups on the current node
template_path = "./run_dp_template.sh"
if not os.path.exists(template_path):
  print(f"Template file {template_path} does not exist.")
  sys.exit(1)
def run_command(dp_rank_local, dp_rank, engine_port_):
  command = f"bash ./run_dp_template.sh {dp_size} {dp_ip} {dp_port} {dp_rank_local} {dp_rank} {engine_port_} {dp_size_local}"
  os.system(command)
processes = []
for i in range(dp_size_local):
  dp_rank = dp_rank_start + i
  dp_rank_local = i
  engine_port_ = engine_port + i
  process = multiprocessing.Process(target=run_command, args=(dp_rank_local, dp_rank, engine_port_))
  processes.append(process)
  process.start()
for process in processes:
  process.join()
```

Decoder 节点示例（`dp_size=64`, `dp_size_local=16`, `dp_rank_start` 可为 0/16/32/48, `dp_ip="192.0.0.5"`）：

```python
import multiprocessing
import os
import sys
dp_size = 64 # total number of DP engines for decode/prefill
dp_size_local = 16 # number of DP engines on the current node
dp_rank_start = 0 # starting DP rank for the current node. e.g. 0/16/32/48
# dp_ip is the same on decoder nodes in this example
dp_ip = "192.0.0.5" # master node IP for DP communication.
dp_port = 13395 # port used for DP communication
engine_port = 9000 # starting port for all DP groups on the current node
template_path = "./run_dp_template.sh"
if not os.path.exists(template_path):
  print(f"Template file {template_path} does not exist.")
  sys.exit(1)
def run_command(dp_rank_local, dp_rank, engine_port_):
  command = f"bash ./run_dp_template.sh {dp_size} {dp_ip} {dp_port} {dp_rank_local} {dp_rank} {engine_port_} {dp_size_local}"
  os.system(command)
processes = []
for i in range(dp_size_local):
  dp_rank = dp_rank_start + i
  dp_rank_local = i
  engine_port_ = engine_port + i
  process = multiprocessing.Process(target=run_command, args=(dp_rank_local, dp_rank, engine_port_))
  processes.append(process)
  process.start()
for process in processes:
  process.join()
```

### 代理服务器

```shell
python load_balance_proxy_server_example.py \
  --port 8000 \
  --host 0.0.0.0 \
  --prefiller-hosts \
    192.0.0.1 \
    192.0.0.2 \
    192.0.0.3 \
    192.0.0.4 \
  --prefiller-hosts-num \
    2 2 2 2 \
  --prefiller-ports \
    9000 9000 9000 9000 \
  --prefiller-ports-inc \
    2 2 2 2\
  --decoder-hosts \
    192.0.0.5 \
    192.0.0.6 \
    192.0.0.7 \
    192.0.0.8 \
  --decoder-hosts-num \
    16 16 16 16 \
  --decoder-ports  \
    9000 9000 9000 9000 \
  --decoder-ports-inc \
    16 16 16 16 \
```

|Parameter  | meaning |
| --- | --- |
| --port | Proxy service Port |
| --host | Proxy service Host IP|
| --prefiller-hosts | Hosts of prefiller nodes |
| --prefiller-hosts-num | Number of repetitions for prefiller node hosts |
| --prefiller-ports | Ports of prefiller nodes |
| --prefiller-ports-inc | Number of increments for prefiller node ports |
| --decoder-hosts | Hosts of decoder nodes |
| --decoder-hosts-num | Number of repetitions for decoder node hosts |
| --decoder-ports | Ports of decoder nodes |
| --decoder-ports-inc | Number of increments for decoder node ports |

### PD 配置详情

- Prefiller：`HCCL_BUFFSIZE=256`，加 `--enforce-eager`，`kv_role: kv_producer`，`--additional-config '{"enable_weight_nz_layout":true,"enable_prefill_optimizations":true}'`
- Decoder：`HCCL_BUFFSIZE=1024`，`kv_role: kv_consumer`，`--additional-config '{"enable_weight_nz_layout":true}'`

参数说明：
- `enable_weight_nz_layout`：量化权重转 NZ 格式加速矩阵乘
- `enable_prefill_optimizations`：DeepSeek 模型 prefill 优化

启用 MTP：

```shell
--speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}'
```

推荐配置（输入均长 3.5k、输出 1.1k、上下文 16k、数据集最大输入 7k，4 节点 prefill + 4 节点 decode）：

| node     | DP | TP | EP | max-model-len | max-num-batched-tokens | max-num-seqs |  gpu-memory-utilization |
|----------|----|----|----|---------------|------------------------|--------------|-----------|
| prefill  | 2  |  8 | 16 |     17000     |         16384          |      4       |    0.9    |
| decode   | 64 |  1 | 64 |     17000     |          256           |      28      |    0.9    |

### FAQ

Prefiller 节点需预热：部分 NPU 算子需多轮 warm-up 才达最佳性能，性能测试前建议先用部分请求预热。

---

## Expert Parallelism Load Balancer

### 是什么

Expert Parallelism Load Balancer (EPLB) 针对 MoE 模型推理的专家负载不均问题，采用冗余专家策略（复制高频专家）并启发式打包到 NPU，结合 group-limited expert routing 尽量把同组专家放同节点以减少跨节点流量，从而降低 TTFT/TPOT 抖动。

### 适用场景与要求

支持 vLLM-Ascend 所有 MoE 模型，但仅在 deepseek-v3.1/r1 上验证过性能。

> 重要：Ascend A5 不支持 EPLB 与 quant type "W4A8MXFP4"、"W4A16"、"W4A16MXFP4" 组合。

MoE QuantType 与硬件：

| QuantType                       | Supported Hardware          |
| ------------------------------- | --------------------------- |
| W8A8 / W8A8-Dynamic             | A2, A3 |
| W4A8 (with fused MC2 enabled)   | A2, A3 |
| MXFP4                           | Ascend 950 Products         |
| MXFP8                           | Ascend 950 Products         |

不建议使用 EPLB 的场景（收益可能无法覆盖开销）：
- P 节点输入序列短于 `1024` tokens
- D 节点每 die 专家数 `<= 8`（950DT 上 `<= 16`），或每 die 负载低于 `128` tokens

> 警告：满足上述条件可能造成性能退化。每 die 约 8 个专家时，EPLB 收益与开销可能相当，需 benchmark 确认有性能提升后再启用。

### 三种使用模式

| Mode | Config in `eplb_config` | Env Variable |
| ---- | ----------------------- | ------------ |
| **Dynamic EPLB** | `dynamic_eplb: true` | `DYNAMIC_EPLB=true` |
| **Recording** (generate expert map) | `expert_map_record_path` | `DYNAMIC_EPLB=true` 或 `EXPERT_MAP_RECORD=true` |
| **Static EPLB** (load pre-recorded map) | `expert_map_path` | none required |

> 重要：Dynamic EPLB 与 Recording 模式下，仅设 `dynamic_eplb: true` 不够，断言要求 `DYNAMIC_EPLB=true` 或 `EXPERT_MAP_RECORD=true`。Static EPLB（通过 `expert_map_path` 加载预记录 map）无需环境变量。

### Dynamic EPLB

需 `export DYNAMIC_EPLB="true"`。当前版本推荐使用 swift balancer 策略（policy 2）。

参数：

| Parameter | Description | Default |
| --- | --- | --- |
| dynamic_eplb | Enable dynamic EPLB. | False |
| expert_heat_collection_interval | Interval for collecting expert heat. | 600 |
| algorithm_execution_interval | Interval for executing the balancing algorithm. | 50 |
| eplb_policy_type | EPLB policy type. | 2 |
| num_redundant_experts | Number of redundant experts. | 0 |
| eplb_heat_collection_stage | Request stage used to collect expert heat. Available values: `all`, `prefill`, and `decode`. | `all` |

D 节点或 colocation 示例：

```shell
# D node or colocation
vllm serve Qwen/Qwen3-235B-A22 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --additional-config '{ "eplb_config": {
    "dynamic_eplb": true,
    "expert_heat_collection_interval": 600,
    "algorithm_execution_interval": 50,
    "eplb_policy_type": 2,
    "num_redundant_experts": 16
    }}'
```

P 节点示例（更短间隔）：

```shell
# P node
vllm serve Qwen/Qwen3-235B-A22 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --additional-config '{ "eplb_config": {
    "dynamic_eplb": true,
    "expert_heat_collection_interval": 50,
    "algorithm_execution_interval": 5,
    "eplb_policy_type": 2,
    "num_redundant_experts": 16
    }}'
```

#### Policy 类型

| Value | Policy | Description |
|-------|--------|-------------|
| `0` | Random | 随机交换专家，仅用于基础测试 |
| `1` | DefaultEplb | 开源 EPLB 算法，对最热专家加冗余，均衡分配 + 本地约束交换 |
| `2` | SwiftBalanceEplb | 低带宽环境优化，支持节点内与节点间冗余，联合优化专家放置（**推荐**） |
| `3` | FlashLB | 滑窗均值/方差/协方差统计，FlashTree 分层搜索最优副本分配，`minimize_redeploy` 增量调整，适合高频负载波动 |

#### 选择性专家热度采集

`eplb_heat_collection_stage` 用于 PD disaggregation：prefill 每轮 token 多，decode 每轮 token 少，两阶段专家负载分布不同，混合采集可能掩盖想优化的阶段的不均衡。

> 重要：选择性热度采集目前由 Ascend model runner V1 实现，Dynamic EPLB（含此选项）尚不被 Ascend model runner V2 支持。

| Value | Behavior | Typical use |
| ----- | -------- | ----------- |
| `all` | 同时采集 prefill 与 decode 热度 | 通用负载，默认 |
| `prefill` | 仅采集 prefill 迭代热度 | 优化 prefill 负载均衡与 TTFT |
| `decode` | 仅采集 decode 迭代热度 | 优化 decode 负载均衡与 TPOT |

调参起点：
- 典型输入序列长于 `1024` tokens：从 `prefill` 开始
- 典型输入短于 `1024` tokens 但并发高于 `1024`：试 `decode` 或 `all`
- 其他/混合：对 `all`/`prefill`/`decode` benchmark 后再定

仅采集 prefill 热度示例：

```shell
export DYNAMIC_EPLB="true"

vllm serve Qwen/Qwen3-235B-A22 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --additional-config '{ "eplb_config": {
    "dynamic_eplb": true,
    "expert_heat_collection_interval": 600,
    "algorithm_execution_interval": 50,
    "eplb_policy_type": 2,
    "num_redundant_experts": 16,
    "eplb_heat_collection_stage": "prefill"
  }}'
```

仅 decode：

```json
{
  "eplb_config": {
    "dynamic_eplb": true,
    "eplb_heat_collection_stage": "decode"
  }
}
```

> 说明：阶段选择仅作用于 dynamic EPLB 热度采集。vLLM-Ascend 内部按"每轮前向 padded scheduled token 数与最大 decode token 数比较"判定迭代属于 prefill 还是 decode（按前向迭代而非单个请求判定）。迭代不匹配所选阶段时不累计专家负载、不推进热度采集间隔；热度采集完成后，均衡计算与逐层专家权重更新正常进行。

### Static EPLB

> 警告：Static EPLB 计划在 v0.25.1 移除。

初始记录 expert map（需 `export EXPERT_MAP_RECORD="true"`）：

```shell
vllm serve Qwen/Qwen3-235B-A22 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --additional-config '{ "eplb_config": {
    "expert_map_record_path": "/path/to/eplb.json",
    "num_redundant_experts": 16,
    "expert_heat_collection_interval": 400,
    "algorithm_execution_interval": 30
  }}'
```

后续部署加载已记录 map：

```shell
vllm serve Qwen/Qwen3-235B-A22 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --additional-config '{
    "eplb_config": {"expert_map_path": "/path/to/eplb.json"}
  }'
```

### 关键注意

1. 调参：
   - `expert_heat_collection_interval`：稳定负载取大值（如 600+），波动负载取小值（如 50-100）
   - `algorithm_execution_interval`：应 `≥ 50`，避免启动期过早均衡
   - `num_redundant_experts`：须满足 `(num_experts + num_redundant_experts)` 能被 expert-parallel size 整除
2. 硬件：所有 NPU 显存与算力一致；网络带宽须支持专家重分布流量（建议 `≥ 10 Gbps`）；容器需挂载 shm
3. 监控：日志中搜 `[Expert Hotness]`（含 current/update peak-to-average ratio）；用 vLLM monitor 检测运行时不均衡；加载前用 jq 等校验 expert map JSON 结构

---

## EPLB Swift Balancer (Design)

### 为什么需要 EPLB

EP 下不同专家分配到不同 NPU，专家负载随工作量变化，需保持跨 NPU 负载均衡。采用冗余专家策略复制高频专家，再启发式打包以均衡负载；结合 MoE 的 group-limited expert routing，尽量把同组专家放同节点以减少跨节点流量。vLLM Ascend 在 `vllm_ascend/eplb/core/policy` 实现可部署的 EP 负载均衡算法，基于估计专家负载计算均衡的专家复制与放置方案（专家负载预测方法不在本仓库范围内，常用历史统计滑动平均）。

### 模块架构

```shell
vllm_ascend
├── eplb
│   ├── adaptor
│   │   └── vllm_adaptor.py
│   ├── core
│   │   ├── policy
│   │   │   ├── policy_abstract.py
│   │   │   ├── policy_default_eplb.py
│   │   │   ├── policy_factory.py
│   │   │   ├── policy_flashlb.py
│   │   │   ├── policy_random.py
│   │   │   └── policy_swift_balancer.py
│   │   ├── eplb_device_transfer_loader.py
│   │   ├── eplb_utils.py
│   │   └── eplb_worker.py
│   ├── eplb_updator.py
│   └── utils.py
└───────────
```

- **Adaptor Module**：`vllm_adaptor.py` 支持 Qwen3-MoE 与 DeepSeek，标准化 policy 算法参数处理
- **Core Module**：
  - Policy 子模块（工厂模式）：`policy_abstract.py`（抽象接口）、`policy_default_eplb.py`（开源 EPLB 论文算法）、`policy_swift_balancer.py`（低带宽设备如 A2 的优化版）、`policy_flashlb.py`（基于阈值的分层波动检测降开销）、`policy_random.py`（随机，基础测试）、`policy_factory.py`（策略工厂）
  - `eplb_device_transfer_loader.py`（专家表/权重传输与更新）、`eplb_utils.py`（专家表初始化与映射）、`eplb_worker.py`（异步算法编排与结果处理）
- **System Components**：`eplb_updator.py`（推理流中负载均衡中心协调器）、`utils.py`（EPLB 接口注册通用工具）

### 默认算法

- **分层负载均衡**：当 server 节点数能整除 expert group 数时使用，利用 group-limited expert routing，先按节点均衡打包 expert group，再节点内复制专家，最后复制专家打包到 NPU。适合 prefill 阶段、较小 expert-parallel size
- **全局负载均衡**：跨 expert group 全局复制专家再打包到 NPU。适合 decode 阶段、较大 expert-parallel size

### 扩展新 Policy

1. 继承 `policy_abstract.py` 的 `EplbPolicy` 抽象类，重写 `rebalance_experts`，输入参数 `current_expert_table`、`expert_workload`，返回 `newplacement`。示例：

    ```python
    class RandomLoadBalance(EplbPolicy):
        def rebalance_experts(self, current_expert_table, expert_workload):
            new_table = copy.deepcopy(current_expert_table)
            num_layers = len(current_expert_table)

            for i in range(num_layers):
                # randomly choose two card
                # indices = random.sample(range(num_card), 2)
                indices = [3, 1]

                # swap redundant experts
                expert_id_to_exchange = new_table[i][indices[0]][-1].clone()
                new_table[i][indices[0]][-1] = new_table[i][indices[1]][-1]
                new_table[i][indices[1]][-1] = expert_id_to_exchange

            return 1, [-i for i in range(num_layers)], new_table
    ```

2. 在 `policy_factory.py` 的 `PolicyFactory` 中注册 policy type 与实现类

### 集成新 MoE 模型

1. Adapter 修改：继承/修改 `vllm_ascend/eplb/adaptor/vllm_adaptor.py`，处理 `num_dense_layers`、`global_expert_num`、`num_roe_layers`，在 `model_register` 中同步参数。示例：

    ```python
    if self.model.config.model_type == "qwen3_moe":
     self.num_dense_layers = 0
     self.global_expert_num = self.model.config.num_experts
    ```

    ```python
    if config.model_type == "qwen3_moe":
        model.num_moe_layers = config.num_hidden_layers
    ```

2. MoE 特性集成：扩展 `vllm_ascend/eplb/utils.py`
3. 注册逻辑更新：在 `model_register` 中加 patch，保持向后兼容
4. 验证测试：跨层参数一致性、跨设备专家表通信、与基线（如 Qwen3-MoE）benchmark

### DFX

- 参数校验：整数参数须明确最值并校验（如 `expert_heat_collection_interval > 0`）；文件路径须校验合法性与读写权限、扩展名为 `.json`、存在性
- 函数规范：初始化时所有 EPLB 参数默认值与类型；通用函数须指定参数类型/默认值/默认返回处理，推荐 `try-except`
- 一致性：expert map 全局唯一，多节点初始化时用分布式通信校验各 rank map 一致性，不一致须告知用户哪些 rank 不一致；更新时若仅部分层/rank 变更须与 EPLB context 同步；更新专家权重时确保旧专家权重内存已释放或旧专家不再使用

### 局限

使用 EPLB 前启动脚本加 `export DYNAMIC_EPLB="true"`；负载/性能数据采集前加 `export EXPERT_MAP_RECORD="true"`。

---

## CPU Binding

### 是什么

**从 vllm-ascend v0.18.0rc1 起，ARM 架构 Ascend 服务器默认启用 CPU binding。** 通常无需手动配置，仅在禁用或显式声明默认行为时设 `enable_cpu_binding`。

### 收益

CPU Binding 优化多 socket ARM 服务器（配 Ascend NPU）的 host 侧调度，解决三类 host 侧推理性能问题：

- 降低跨 NUMA 流量：worker 进程贴近其活跃 NPU 对应的 CPU 与内存资源，减少远端 NUMA 访问
- 降低线程抢占的上下文切换开销：关键运行时线程跑在稳定 CPU 范围，减少调度迁移与 CPU 争用
- 更稳定的延迟与多 worker 隔离：独立 worker 不共享相同 CPU/NUMA 资源，降低尾延迟抖动，吞吐更可预测

> 注：这是 host 侧性能优化，**不改变模型执行逻辑或数值输出**。内存迁移不可用时 CPU 亲和仍生效，但内存局部性可能变差，延迟/吞吐可能退化。

### 如何启用/禁用

在线服务默认行为：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct
```

禁用：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --additional-config '{"enable_cpu_binding": false}'
```

离线推理默认：

```python
from vllm import LLM

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
```

禁用：

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    additional_config={"enable_cpu_binding": False},
)
```

### 要求

官方 vllm-ascend 镜像在 v0.18.0rc1 及更早已含 `util-linux` 与 `procps`/`procps-ng`，**v0.18.0rc1 起官方镜像另含 `numactl`**。非官方镜像需手动装：

```bash
# Ubuntu/Debian
sudo apt-get install -y util-linux numactl procps

# RHEL/CentOS/Alma/Rocky
sudo yum install -y util-linux numactl procps-ng

# openEuler
sudo dnf install -y util-linux numactl procps-ng
```

> 缺少 `numactl`/`migratepages` 时，vLLM Ascend 仅跳过内存迁移，worker 进程与运行时线程仍会 pin，但已落在远端 NUMA 的页不会迁移，**可能降低局部性、退化延迟/吞吐**。建议使用跨 NUMA 节点均匀分布的 cpuset，不均衡 cpuset 会削弱收益。

Ascend 950 细节：
- 用 `npu-smi info -t topo` 的 NPU-to-CPU 亲和选 worker 亲和 NUMA 节点；每个 worker 主进程 pin到该 NUMA 的一个 CPU cluster，cluster 大小由 `lscpu` `Thread(s) per core` 推导（值为 1 时 8 CPU，值为 2 时 16 CPU）
- Ascend 950 还把 host `uvb_poll_window_thread` 线程 pin 到 NUMA0（除 CPU0 外，受当前 cpuset 约束）。Docker 部署需加 `--pid=host` 以便 vLLM Ascend 发现并绑定这些 host 线程
- 950 在 `migratepages` 可用时仍可迁移内存页，但**不单独 pin ACL/release 线程，不做 IRQ binding**
- IRQ binding 需读 `/proc/interrupts` 并写 `/proc/irq/*/smp_affinity` 的权限；若 `irqbalance` 在跑且进程可用 `systemctl`，vLLM Ascend 会在应用 IRQ 亲和前停掉它；容器无 `systemctl` 时需在 host 停 `irqbalance`

需要稳定 IRQ 亲和时，启动 vLLM 前在 host 停 `irqbalance`：

```bash
sudo systemctl stop irqbalance
```

退出后按需恢复：

```bash
sudo systemctl start irqbalance
```

Ascend 950 日志含 `[irq] IRQ binding skipped on Ascend 950.`，950 分配日志用 `worker=[...]` 而非 `acl=[...]`/`release=[...]`；UVB 轮询线程被找到并绑定时日志还会报其 TID 与 CPU 池：

```text
Ascend 950 NPU0: worker=[...]
[cpu_bind_ascend_950] uvb_poll_window_thread tids=[...] cpus=[...]
```

### 故障排查

| Message | Meaning | Action |
| --- | --- | --- |
| `CPU binding skipped: non-ARM CPU detected.` | CPU binding only runs on ARM. | No action needed on x86_64. |
| `Can not get running npu info.` | No running NPU was found, or `ASCEND_RT_VISIBLE_DEVICES` filtered all NPUs. | Check visible NPU IDs and `npu-smi info`. |
| `Insufficient CPUs for binding...` | Fewer CPUs are available than the role split requires. Devices with IRQ binding need at least 5 CPUs per logical NPU. Ascend 950 needs one full cluster per worker. | Expand the cpuset or reduce visible NPUs. |
| `NPU topo affinity not found...` | Topology affinity is unavailable. | On Ascend 950, worker CPU binding is skipped. On other topo-affinity devices, vLLM Ascend falls back to `global_slice`. Check `npu-smi info -t topo` when topology affinity is expected. |
| `uvb_poll_window_thread not found... --pid=host` | Ascend 950 could not see host UVB polling threads. | Recreate the Docker container with `--pid=host`, then restart vLLM. |
| `failed to bind uvb_poll_window_thread... --pid=host` | Ascend 950 found a UVB polling thread but failed to bind it. | Check permissions and recreate the Docker container with `--pid=host` if running in Docker. |
| `The 'migratepages' command is not available...` | Memory migration is skipped, while CPU thread binding still proceeds. | Install `numactl` if NUMA locality or performance is affected. |
| `[irq] IRQ binding skipped on Ascend 950.` | Ascend 950 does not use the IRQ binding step. | No action needed. Worker main binding and memory migration still proceed. |
| `Bind cpus failed in rank...` | A binding step failed and CPU binding was skipped for that rank. | Check `taskset`, `lscpu`, `npu-smi`, cpuset size, and `/proc/irq` permissions. |

---

## 源文件清单

- docs/source/user_guide/feature_guide/context_parallel.md — OK
- docs/source/user_guide/feature_guide/sequence_parallelism.md — OK
- docs/source/user_guide/feature_guide/Fine_grained_TP.md — OK（注：team-lead 给的文件名 `fine_grained_tp(Fine_grained_TP).md` 实际为 404，GitHub 上真实文件名为 `Fine_grained_TP.md`，已按真实名抓取成功）
- docs/source/user_guide/feature_guide/dynamic_chunk_pipeline_parallel.md — OK
- docs/source/user_guide/feature_guide/large_scale_ep.md — OK
- docs/source/user_guide/feature_guide/expert_parallelism_load_balancer.md — OK
- docs/source/developer_guide/Design_Documents/eplb_swift_balancer.md — OK
- docs/source/user_guide/feature_guide/cpu_binding.md — OK

无抓取失败源。所有 8 个源文件均成功获取并整合。
