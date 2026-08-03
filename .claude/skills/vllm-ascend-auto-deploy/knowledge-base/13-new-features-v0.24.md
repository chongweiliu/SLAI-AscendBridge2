> 来源：vllm-ascend docs（releases/v0.24.0rc 分支，抓取于 2026-07-28）。本节整合 3 个 main 分支尚未合入的新特性文档。注意：v0.24.0rc 为候选分支，内容可能调整。

# v0.24.0rc 新增特性（main 尚未合入）

本章节收录 `releases/v0.24.0rc` 分支有而 `main` 没有的 3 个新文档，避免知识库滞后。命令/参数/env 保持英文原文，说明用中文。

---

## 长序列 Context Parallel — 多节点（DeepSeek）

> 来源：docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md（releases/v0.24.0rc）

### 关键信息速览

- **特性**：Context Parallel（CP），用于长序列场景。当前**仅 Atlas A3 支持**，A2 未来支持。
- **示例模型**：DeepSeek-V3.1-w8a8（mix mtp 量化版，ModelScope `Eco-Tech/DeepSeek-V3.1-w8a8`），需将 config.json 的 `torch_dtype` 从 `float16` 改为 `bfloat16`。
- **拓扑**：3 台 Atlas 800T A3，1P1D——Prefill 跨多机（prefill1=192.0.0.1, prefill2=192.0.0.2），Decode 单机（decoder1=192.0.0.3）；每台 8 NPU 16 chips 一个实例。
- **策略**：在 Prefill 节点启用 CP 改善 TTFT；Decode 节点不开 DCP（额外通信+小算子开销不划算）。

### 原文（完整保留）

# Long-Sequence Context Parallel (Deepseek)

## Getting Started

!!! note

    Context parallel feature currently is only supported on Atlas A3 device, and will be supported on Atlas A2 in the future.

vLLM-Ascend now supports long sequence with context parallel options. This guide takes one-by-one steps to verify these features with constrained resources.

Take the Deepseek-V3.1-w8a8 model as an example, use 3 Atlas 800T A3 servers to deploy the “1P1D” architecture. Node p is deployed across multiple machines, while node d is deployed on a single machine. Assume the IP of the prefiller server is 192.0.0.1 (prefill 1) and 192.0.0.2 (prefill 2), and the decoder servers are 192.0.0.3 (decoder 1). On each server, use 8 NPUs 16 chips to deploy one service instance. In the current example, we will enable the context parallel feature on node p to improve TTFT. Although enabling the DCP feature on node d can reduce memory usage, it would introduce additional communication and small operator overhead. Therefore, we will not enable the DCP feature on node d.

## Environment Preparation

### Model Weight

- `DeepSeek-V3.1_w8a8mix_mtp` (Quantized version with mix mtp): [Download model weight](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V3.1-w8a8). Please modify `torch_dtype` from `float16` to `bfloat16` in `config.json`.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Verify Multi-node Communication

Refer to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication) to verify multi-node communication.

### Installation

You can use our official Docker image to run `DeepSeek-V3.1` directly.

Select an image based on your machine type and start the Docker image on your node, refer to [using Docker](../../installation.md#set-up-using-docker).

```bash
# Update the vllm-ascend image according to your environment.
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
export NAME=vllm-ascend

# Run the container using the defined variables
# Note: If you are running bridge network with Docker, please expose available ports for multiple nodes communication in advance
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
-v /root/.cache:/root/.cache \
-it $IMAGE bash
```

You need to set up environment on each node.

## Prefiller/Decoder Deployment

We can run the following scripts to launch a server on the prefiller/decoder node, respectively. Please note that each P/D node will occupy ports ranging from kv_port to kv_port + num_chips to initialize socket listeners. To avoid any issues, port conflicts should be prevented. Additionally, ensure that each node's engine_id is uniquely assigned to avoid conflicts.

1. Run the following script to execute online 128k inference on three nodes respectively.

    === "Prefiller node 1"

            ```shell
            nic_name="eth0"  # network card name
            local_ip="192.0.0.1"
            master_addr="192.0.0.1"
            export HCCL_IF_IP=$local_ip
            export GLOO_SOCKET_IFNAME=$nic_name
            export TP_SOCKET_IFNAME=$nic_name
            export HCCL_SOCKET_IFNAME=$nic_name
            export HCCL_BUFFSIZE=768
            export OMP_PROC_BIND=false
            export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
            export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
            export OMP_NUM_THREADS=1
            export HCCL_OP_EXPANSION_MODE="AIV"
            export VLLM_USE_V1=1
            export TASK_QUEUE_ENABLE=1

            vllm serve /path_to_weight/DeepSeek-V3.1_w8a8mix_mtp \
              --host 0.0.0.0 \
              --port 8004 \
              --decode-context-parallel-size 8 \
              --prefill-context-parallel-size 2 \
              --cp-kv-cache-interleave-size 128 \
              --tensor-parallel-size 16 \
              --enable-expert-parallel \
              --quantization ascend \
              --enforce-eager \
              --served-model-name deepseek_v3 \
              --seed 1024 \
              --no-enable-chunked-prefill \
              --no-enable-prefix-caching \
              --max-num-seqs 1 \
              --max-model-len 136000 \
              --max-num-batched-tokens 136000 \
              --block-size 128 \
              --trust-remote-code \
              --gpu-memory-utilization 0.8 \
              --nnodes 2 \
              --node-rank 0 \
              --master-addr $master_addr \
              --master-port 7001 \
              --speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp"}' \
              --kv-transfer-config \
              '{"kv_connector": "MooncakeConnectorV1",
              "kv_role": "kv_producer",
              "kv_port": "30000",
              "kv_connector_extra_config": {
                        "use_ascend_direct": true,
                        "prefill": {
                                "dp_size": 1,
                                "tp_size": 16
                        },
                        "decode": {
                                "dp_size": 1,
                                "tp_size": 16
                        }
                  }
              }'
            ```

    === "Prefiller node 2"

            ```shell
            nic_name="eth0"  # network card name
            local_ip="192.0.0.2"
            master_addr="192.0.0.1"
            export HCCL_IF_IP=$local_ip
            export GLOO_SOCKET_IFNAME=$nic_name
            export TP_SOCKET_IFNAME=$nic_name
            export HCCL_SOCKET_IFNAME=$nic_name
            export HCCL_BUFFSIZE=768
            export OMP_PROC_BIND=false
            export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
            export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
            export OMP_NUM_THREADS=1
            export HCCL_OP_EXPANSION_MODE="AIV"
            export VLLM_USE_V1=1
            export TASK_QUEUE_ENABLE=1

            vllm serve /path_to_weight/DeepSeek-V3.1_w8a8mix_mtp \
              --host 0.0.0.0 \
              --port 8004 \
              --decode-context-parallel-size 8 \
              --prefill-context-parallel-size 2 \
              --cp-kv-cache-interleave-size 128 \
              --tensor-parallel-size 16 \
              --enable-expert-parallel \
              --quantization ascend \
              --enforce-eager \
              --served-model-name deepseek_v3 \
              --seed 1024 \
              --no-enable-chunked-prefill \
              --no-enable-prefix-caching \
              --max-num-seqs 1 \
              --max-model-len 136000 \
              --max-num-batched-tokens 136000 \
              --block-size 128 \
              --trust-remote-code \
              --gpu-memory-utilization 0.8 \
              --nnodes 2 \
              --node-rank 1 \
              --headless \
              --master-addr $master_addr \
              --master-port 7001 \
              --speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp"}' \
              --kv-transfer-config \
              '{"kv_connector": "MooncakeConnectorV1",
              "kv_role": "kv_producer",
              "kv_port": "30000",
              "kv_connector_extra_config": {
                        "use_ascend_direct": true,
                        "prefill": {
                                "dp_size": 1,
                                "tp_size": 16
                        },
                        "decode": {
                                "dp_size": 1,
                                "tp_size": 16
                        }
                  }
              }'
            ```

    === "Decoder node 1"

            ```shell
            nic_name="eth0"  # network card name
            local_ip="192.0.0.3"
            export HCCL_IF_IP=$local_ip
            export GLOO_SOCKET_IFNAME=$nic_name
            export TP_SOCKET_IFNAME=$nic_name
            export HCCL_SOCKET_IFNAME=$nic_name
            export HCCL_BUFFSIZE=768
            export OMP_PROC_BIND=false
            export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
            export OMP_NUM_THREADS=1
            export HCCL_OP_EXPANSION_MODE="AIV"
            export VLLM_USE_V1=1
            export TASK_QUEUE_ENABLE=1

            vllm serve /path_to_weight/DeepSeek-V3.1_w8a8mix_mtp \
              --host 0.0.0.0 \
              --port 8004 \
              --api-server-count 1 \
              --data-parallel-size 1 \
              --data-parallel-size-local 1 \
              --data-parallel-start-rank 0 \
              --data-parallel-address $local_ip \
              --data-parallel-rpc-port 5980  \
              --decode-context-parallel-size 1 \
              --tensor-parallel-size 16 \
              --enable-expert-parallel \
              --quantization ascend \
              --no-enable-prefix-caching \
              --distributed-executor-backend mp \
              --served-model-name deepseek_v3 \
              --seed 1024 \
              --max-model-len 136000 \
              --max-num-batched-tokens 128 \
              --enable-chunked-prefill \
              --max-num-seqs 4 \
              --trust-remote-code \
              --gpu-memory-utilization 0.96 \
              --additional-config '{"recompute_scheduler_enable": true}' \
              --speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp"}' \
              --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[1,2,4]}' \
              --kv-transfer-config \
              '{"kv_connector": "MooncakeConnectorV1",
              "kv_role": "kv_consumer",
              "kv_port": "30200",
              "kv_connector_extra_config": {
                        "prefill": {
                                "dp_size": 1,
                                "tp_size": 16
                        },
                        "decode": {
                                "dp_size": 1,
                                "tp_size": 16
                        }
                  }
              }'
            ```

2. Prefill master node `proxy.sh` script

    ```shell
    python load_balance_proxy_server_example.py \
      --port 8005 \
      --host 192.0.0.1 \
      --prefiller-hosts \
        192.0.0.1 \
      --prefiller-ports \
        8004 \
      --decoder-hosts \
        192.0.0.3 \
      --decoder-ports \
        8004
    ```

3. Run proxy

Run a proxy server on the same node with the prefiller service instance. You can get the proxy program in the repository's examples: [load\_balance\_proxy\_server\_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

```shell
cd vllm-ascend/examples/disaggregated_prefill_v1/
bash proxy.sh
```

**Notice:**
The parameters are explained as follows:

- `--tensor-parallel-size` 16 are common settings for tensor parallelism (TP) sizes.
- `--prefill-context-parallel-size` 2 is common setting for prefill context parallelism (PCP) sizes.
- `--decode-context-parallel-size` 8 are common settings for decode context parallelism (DCP) sizes.
- `--max-model-len` represents the context length, which is the maximum value of the input plus output for a single request.
- `--max-num-seqs` indicates the maximum number of requests that each DP group is allowed to process. If the number of requests sent to the service exceeds this limit, the excess requests will remain in a waiting state and will not be scheduled. Note that the time spent in the waiting state is also counted in metrics such as TTFT and TPOT. Therefore, when testing performance, it is generally recommended that `--max-num-seqs` * `--data-parallel-size` >= the actual total concurrency.
- `--max-num-batched-tokens` represents the maximum number of tokens that the model can process in a single step. Currently, vLLM v1 scheduling enables ChunkPrefill/SplitFuse by default, which means:
    - (1) If the input length of a request is greater than `--max-num-batched-tokens`, it will be divided into multiple rounds of computation according to `--max-num-batched-tokens`;
    - (2) Decode requests are prioritized for scheduling, and prefill requests are scheduled only if there is available capacity.
    - Generally, if `--max-num-batched-tokens` is set to a larger value, the overall latency will be lower, but the pressure on NPU memory (activation value usage) will be greater.
- `--gpu-memory-utilization` represents the proportion of HBM that vLLM will use for actual inference. Its essential function is to calculate the available kv_cache size. During the warm-up phase (referred to as profile run in vLLM), vLLM records the peak NPU memory usage during an inference process with an input size of `--max-num-batched-tokens`. The available kv_cache size is then calculated as: `--gpu-memory-utilization` * HBM size - peak NPU memory usage. Therefore, the larger the value of `--gpu-memory-utilization`, the more kv_cache can be used. However, since the NPU memory usage during the warm-up phase may differ from that during actual inference (e.g., due to uneven EP load), setting `--gpu-memory-utilization` too high may lead to OOM (Out of Memory) issues during actual inference. The default value is `0.9`.
- `--enable-expert-parallel` indicates that EP is enabled. Note that vLLM does not support a mixed approach of ETP and EP; that is, MoE can either use pure EP or pure TP.
- `--no-enable-prefix-caching` indicates that prefix caching is disabled. To enable it, remove this option.
- `--quantization` "ascend" indicates that quantization is used. To disable quantization, remove this option.
- `--additional-config '{"recompute_scheduler_enable": true}'`: enables the recomputation scheduler. When the Key-Value Cache (KV Cache) of the decode node is insufficient, requests will be sent to the prefill node to recompute the KV Cache. In the PD separation scenario, enable this configuration only on decode nodes.
- `--compilation-config` contains configurations related to the aclgraph graph mode. The most significant configurations are "cudagraph_mode" and "cudagraph_capture_sizes", which have the following meanings:
"cudagraph_mode": represents the specific graph mode. Currently, "PIECEWISE" and "FULL_DECODE_ONLY" are supported. The graph mode is mainly used to reduce the cost of operator dispatch. Currently, "FULL_DECODE_ONLY" is recommended.
- "cudagraph_capture_sizes": represents different levels of graph modes. The default value is [1, 2, 4, 8, 16, 24, 32, 40,..., `--max-num-seqs`]. In the graph mode, the input for graphs at different levels is fixed, and inputs between levels are automatically padded to the next level. Currently, the default setting is recommended. Only in some scenarios is it necessary to set this separately to achieve optimal performance.
- `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1` indicates that Flashcomm1 optimization is enabled. Currently, this optimization is only supported for MoE in scenarios where tensor-parallel-size > 1.

**Notice:**

- tensor-parallel-size needs to be divisible by decode-context-parallel-size.
- decode-context-parallel-size must be less than or equal to tensor-parallel-size.

## Accuracy Evaluation

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result, here is the result of `DeepSeek-V3.1-w8a8` for reference only.

| dataset  | version | metric | mode | vllm-api-general-chat |
|----------| ----- | ----- | ----- |-----------------------|
| aime2024 | - | accuracy | gen | 86.67 |

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `DeepSeek-V3.1-w8a8` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
export VLLM_USE_MODELSCOPE=True
vllm bench serve --model /path_to_weight/DeepSeek-V3.1_w8a8mix_mtp  --dataset-name random --random-input 131072 --num-prompts 20 --request-rate 0 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.

| dataset | version | metric      | mode | ttft   |
|---------| ----- |-------------|------|--------|
| random  | - | performance | perf | 20.7s |

---

## 长序列 Context Parallel — 单节点（Qwen3-235B-A22B）

> 来源：docs/source/tutorials/features/long_sequence_context_parallel_single_node.md（releases/v0.24.0rc）

### 关键信息速览

- **示例模型**：Qwen3-235B-A22B-w8a8（量化版，ModelScope `vllm-ascend/Qwen3-235B-A22B-W8A8`），需 1 台 Atlas 800 A3（64G×16）。
- **拓扑**：单节点 pd co-locate。

### 原文（完整保留）

# Long-Sequence Context Parallel (Qwen3-235B-A22B)

## Getting Started

vLLM-Ascend now supports long-sequence context parallel. This guide takes one-by-one steps to verify these features with constrained resources.

Using the `Qwen3-235B-A22B-w8a8` (Quantized version) model as an example, use 1 Atlas 800 A3 (64G × 16) server to deploy the single node "pd co-locate" architecture.

## Environment Preparation

### Model Weight

- `Qwen3-235B-A22B-w8a8` (Quantized version): requires 1 Atlas 800 A3 (64G × 16) node. [Download model weight](https://modelscope.cn/models/vllm-ascend/Qwen3-235B-A22B-W8A8)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Run with Docker

Start a Docker container on each node.

```bash
# Update the vllm-ascend image
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
export NAME=vllm-ascend

# Run the container using the defined variables
# Note: If you are running bridge network with Docker, please expose available ports for multiple nodes communication in advance
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

## Deployment

### Single-node Deployment

`Qwen3-235B-A22B-w8a8` can be deployed on 1 Atlas 800 A3（64G*16）.
Quantized version needs to start with parameter `--quantization ascend`.

Run the following script to execute online 128k inference.

```shell
#!/bin/sh
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True
# To reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export TASK_QUEUE_ENABLE=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

vllm serve vllm-ascend/Qwen3-235B-A22B-w8a8 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --prefill-context-parallel-size 2 \
  --decode-context-parallel-size 2 \
  --seed 1024 \
  --quantization ascend \
  --served-model-name qwen3 \
  --max-num-seqs 1 \
  --max-model-len 131072 \
  --max-num-batched-tokens 131072 \
  --enable-expert-parallel \
  --trust-remote-code \
  --gpu-memory-utilization 0.95 \
  --hf-overrides '{"rope_parameters": {"rope_type":"yarn","rope_theta":1000000,"factor":4,"original_max_position_embeddings":32768}}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[1,2,4,8]}' \
```

**Notice:**

- for vllm version below `v0.12.0` use parameter: `--rope_scaling '{"rope_type":"yarn","factor":4,"original_max_position_embeddings":32768}' \`
- for vllm version `v0.12.0` use parameter: `--hf-overrides '{"rope_parameters": {"rope_type":"yarn","rope_theta":1000000,"factor":4,"original_max_position_embeddings":32768}}' \`

The parameters are explained as follows:

- `--tensor-parallel-size` 8 are common settings for tensor parallelism (TP) sizes.
- `--prefill-context-parallel-size` 2 are common settings for prefill context parallelism (PCP) sizes.
- `--decode-context-parallel-size` 2 are common settings for decode context parallelism (DCP) sizes.
- `--max-model-len` represents the context length, which is the maximum value of the input plus output for a single request.
- `--max-num-seqs` indicates the maximum number of requests that each DP group is allowed to process. If the number of requests sent to the service exceeds this limit, the excess requests will remain in a waiting state and will not be scheduled. Note that the time spent in the waiting state is also counted in metrics such as TTFT and TPOT. Therefore, when testing performance, it is generally recommended that `--max-num-seqs` * `--data-parallel-size` >= the actual total concurrency.
- `--max-num-batched-tokens` represents the maximum number of tokens that the model can process in a single step. Currently, vLLM v1 scheduling enables ChunkPrefill/SplitFuse by default, which means:
    - (1) If the input length of a request is greater than `--max-num-batched-tokens`, it will be divided into multiple rounds of computation according to `--max-num-batched-tokens`;
    - (2) Decode requests are prioritized for scheduling, and prefill requests are scheduled only if there is available capacity.
    - Generally, if `--max-num-batched-tokens` is set to a larger value, the overall latency will be lower, but the pressure on GPU memory (activation value usage) will be greater.
- `--gpu-memory-utilization` represents the proportion of HBM that vLLM will use for actual inference. Its essential function is to calculate the available kv_cache size. During the warm-up phase (referred to as profile run in vLLM), vLLM records the peak GPU memory usage during an inference process with an input size of `--max-num-batched-tokens`. The available kv_cache size is then calculated as: `--gpu-memory-utilization` * HBM size - peak GPU memory usage. Therefore, the larger the value of `--gpu-memory-utilization`, the more kv_cache can be used. However, since the GPU memory usage during the warm-up phase may differ from that during actual inference (e.g., due to uneven EP load), setting `--gpu-memory-utilization` too high may lead to OOM (Out of Memory) issues during actual inference. The default value is `0.9`.
- `--enable-expert-parallel` indicates that EP is enabled. Note that vLLM does not support a mixed approach of EP and TP; that is, MoE can either use pure EP or pure TP.
- `--no-enable-prefix-caching` indicates that prefix caching is disabled. To enable it, remove this option.
- `--quantization` "ascend" indicates that quantization is used. To disable quantization, remove this option.
- `--compilation-config` contains configurations related to the aclgraph graph mode. The most significant configurations are "cudagraph_mode" and "cudagraph_capture_sizes", which have the following meanings:
"cudagraph_mode": represents the specific graph mode. Currently, "PIECEWISE" and "FULL_DECODE_ONLY" are supported. The graph mode is mainly used to reduce the cost of operator dispatch. Currently, "FULL_DECODE_ONLY" is recommended.
- "cudagraph_capture_sizes": represents different levels of graph modes. The default value is [1, 2, 4, 8, 16, 24, 32, 40,..., `--max-num-seqs`]. In the graph mode, the input for graphs at different levels is fixed, and inputs between levels are automatically padded to the next level. Currently, the default setting is recommended. Only in some scenarios is it necessary to set this separately to achieve optimal performance.
- `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1` indicates that Flashcomm1 optimization is enabled. Currently, this optimization is only supported for MoE in scenarios where tp_size > 1.

**Notice:**

- tp_size needs to be divisible by dcp_size
- decode context parallel size must be less than or equal to max_dcp_size, where max_dcp_size = tensor_parallel_size // total_num_kv_heads.

## Accuracy Evaluation

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result, here is the result of `Qwen3-235B-A22B-w8a8` for reference only.

| dataset  | version | metric | mode | vllm-api-general-chat |
|----------| ----- | ----- | ----- |-----------------------|
| aime2024 | - | accuracy | gen | 83.33 |

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `Qwen3-235B-A22B-w8a8` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
export VLLM_USE_MODELSCOPE=True
vllm bench serve --model vllm-ascend/Qwen3-235B-A22B-w8a8  --dataset-name random --random-input 131072 --num-prompts 1 --request-rate 1 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.

| dataset | version | metric      | mode | ttft   |
|---------| ----- |-------------|------|--------|
| random  | - | performance | perf | 17.36s |

---

## External DP（外部数据并行 + 负载均衡 Proxy）

> 来源：docs/source/user_guide/feature_guide/external_dp.md（releases/v0.24.0rc）

### 关键信息速览

- **特性**：External DP——把每个 DP rank 当独立 vLLM 部署，外部 router 按实时 telemetry 做 HTTP 请求负载均衡。vLLM 原生支持 external DP，vllm-ascend 额外提供：
  1. 一条命令拉起多个 vLLM 实例的 launch 脚本；
  2. request-length-aware（请求长度感知）负载均衡 proxy。
- **依赖**：Python 3.10+，`pip install fastapi httpx uvicorn`。

### 原文（完整保留）

# External DP

For larger-scale deployments especially, it can make sense to handle the orchestration and load balancing of data parallel ranks externally.

In this case, it's more convenient to treat each DP rank like a separate vLLM deployment, with its own endpoint, and have an external router balance HTTP requests between them, making use of appropriate real-time telemetry from each server for routing decisions.

## Getting Started

The functionality of [external DP](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/?h=external#external-load-balancing) is already natively supported by vLLM. In vllm-ascend we provide two enhanced functionalities:

1. A launch script that helps to launch multiple vLLM instances in one command.
2. A request-length-aware load-balancing proxy for external DP.

This tutorial will introduce the usage of them.

### Prerequisites

- Python 3.10+
- Install dependencies needed by load-balance proxy server:

```shell
pip install fastapi httpx uvicorn
```

## Starting External DP Servers

First, you need to have at least two vLLM servers running in data parallel. These can be mock servers or actual vLLM servers. Note that this proxy also works with only one vLLM server running, but will fall back to direct request forwarding which is meaningless.

You can start external vLLM DP servers one-by-one manually or using the launch script in `examples/external_online_dp`. For scenarios of large DP size across multiple nodes, we recommend using our launch script for convenience.

### Manually Launch

```shell
# This example shows how to manually launch a vLLM service with DP size 2 in one node.
vllm serve --host 0.0.0.0 --port 8100 --data-parallel-size 2 --data-parallel-rank 0 ... # vLLM DP0
vllm serve --host 0.0.0.0 --port 8101 --data-parallel-size 2 --data-parallel-rank 1 ... # vLLM DP1
```

### Use Launch Script

Firstly, you need to modify the `examples/external_online_dp/run_dp_template.sh` according to your vLLM configuration. Then you can use `examples/external_online_dp/launch_online_dp.py` to launch multiple vLLM instances in one command on each node. It will internally call `examples/external_online_dp/run_dp_template.sh` for each DP rank with proper DP-related parameters.

An example of running external DP in one single node:

```shell
cd examples/external_online_dp
# running DP4 TP4 in a node with 16 NPUs
python launch_online_dp.py --dp-size 4 --tp-size 4 --dp-size-local 4 --dp-rank-start 0 --dp-address x.x.x.x --dp-rpc-port 12342
```

An example of running external DP in two nodes:

```shell
cd examples/external_online_dp
# running DP4 TP4 in two nodes with 8 NPUs each
# Node 0 holds DP0 DP1 and node 1 holds DP2 DP3
# Here x.x.x.x:12342 is served as the common data parallel RPC address

# On node 0:
python launch_online_dp.py --dp-size 4 --tp-size 4 --dp-size-local 2 --dp-rank-start 0 --dp-address x.x.x.x --dp-rpc-port 12342

# On node 1:
python launch_online_dp.py --dp-size 4 --tp-size 4 --dp-size-local 2 --dp-rank-start 2 --dp-address x.x.x.x --dp-rpc-port 12342
```

## Starting Load-balance Proxy Server

After all vLLM DP instances are launched, you can now launch the load-balance proxy server, which serves as an entrypoint for coming requests and load-balances them between vLLM DP instances.

The proxy server has the following features:

- Load balances requests to multiple vLLM servers based on request length.
- Supports OpenAI-compatible `/v1/completions` and `/v1/chat/completions` endpoints.
- Streams responses from backend servers to clients.

To run the proxy server, you need to specify the host and port for each vLLM DP Instance:

```shell
# For example, we have already started two DP instances in single node:
# python launch_online_dp.py --dp-size 2 --tp-size 8 --dp-size-local 2 --dp-rank-start 0 --dp-address x.x.x.x --dp-rpc-port 12342
# By default, launch_online_dp.py will launch vLLM instances from starting port 9000,
# so the vLLM ports for DP0 and DP1 are 9000 and 9001 separately.
# Then you can start the load-balance proxy server by:
cd examples/external_online_dp
python dp_load_balance_proxy_server.py \
    --host 0.0.0.0 --port 8000 \
    --dp-hosts 127.0.0.1 127.0.0.1 \
    --dp-ports 9000 9001 \
```

After this, you can directly send requests to the proxy server and run DP with external load balancing.
