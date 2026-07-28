> 来源：vllm-ascend docs（main 分支，抓取于 2026-07-28），整合自多个源文件

# KV Cache 与 Offload 特性知识库

本文件整合 vllm-ascend 中与 KV Cache 池化、CPU/HBM 卸载、模型权重迁移、持久化缓存等相关的全部特性。每个 H2 对应一个特性，按"是什么 / 何时使用 / 如何启用 / 硬件与版本要求"四段组织；所有命令、CLI 参数、环境变量、配置片段、端口/主机字符串、版本要求均为英文原样保留。

---

## KV Pool

### 是什么

KV Cache Pool（AscendStoreConnector）是 vllm-ascend 提供的跨实例 KV 缓存池化能力，把 Prefill 节点算出的 KV cache 写入一个外部 Store（mooncake / memcache / yuanrong），Decode 节点或后续请求再按 prefix 命中读回，从而避免重复 prefill。支持 PD 分离（`kv_producer` / `kv_consumer`）和 PD 混合（`kv_both`）两种部署，可单挂 `AscendStoreConnector`，也可用 `MultiConnector` 同时挂载 `MooncakeConnectorV1`（做 PD 间 KV transfer）+ `AscendStoreConnector`（做 prefix-cache 池）。

`kv_load_failure_policy`（`kv-transfer-config` 顶层字段）控制 KV 加载失败行为：
- `recompute`：加载失败时回滚到最后一个有效 prefix 重新调度重算（hybrid attention 模型如 DeepSeekV4 / Qwen 3.5 暂不支持）。
- `fail`：加载失败时直接终止请求报错。
- 默认 `fail`；用 `MultiConnector` 时该字段配在 `MultiConnector` 顶层，而非子 connector。

`kv_connector_extra_config` 关键参数：

| Parameter | Description |
| :--- | :--- |
| `lookup_rpc_port` | Pooling scheduler 与 worker 进程间 RPC 端口；每个实例需唯一端口。 |
| `load_async` | 是否启用异步加载，默认 false。 |
| `backend` | 存储后端 `mooncake` / `memcache` / `yuanrong`，默认 `mooncake`。 |
| `consumer_is_to_put` | Decode 节点是否向 KV Pool 写入 KV cache，默认 false。 |
| `consumer_is_to_load` | Decode 节点是否从 KV Pool 加载 KV cache，默认 false。 |
| `use_layerwise` | 启用逐层 KV 存取，仅 Prefill 节点支持、且需 `memcache` 后端，默认 false。 |
| `prefill_pp_size` | Prefill 节点开启 PP 时需设置。 |
| `prefill_pp_layer_partition` | Prefill 节点开启 PP 时需设置。 |

启用 KV Pool 后必须同步所有节点的 `PYTHONHASHSEED` 以保证 hash 一致：

```bash
export PYTHONHASHSEED=0
```

### 何时使用

- PD 分离或 PD 混合部署，需要跨实例共享 prefix KV。
- 长上下文场景需要降低重复 prefill 成本。
- 多实例集群需要扩大可用 prefix cache 容量（超出单卡 HBM）。

### 如何启用

三种后端可选：

**Mooncake 后端**：检查 `/etc/hccn.conf`（Docker 内需挂载），安装 mooncake（glibc ≥ 2.35）：

```shell
ldd --version
python3 -m pip install mooncake-transfer-engine-npu==0.3.11.post1 --extra-index-url https://mirrors.aliyun.com/pypi/web/simple
```

按硬件配置环境变量：

| Hardware | Dependencies | Export Command | Description |
| :--- | :--- | :--- | :--- |
| 800 I/T A5 series | HDK >=25.6 with mooncake >= v0.3.11 <br>CANN >= 9.1.0 | # UBOE<br> `export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["uboe:device"]}'` <br> # UB<br>`export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'` | Configure the required environment variables based on the communication protocol to use. |
| 800 I/T A3 series | HDK >= 26.0<br>or HDK >= 25.5 with mooncake >= v0.3.11<br>CANN >= 9.0.0<br>LingQu Computing Network >= 1.5 | `export ASCEND_ENABLE_USE_FABRIC_MEM=1` | **Recommended**. Enables unified memory address direct transmission scheme. |
| 800 I/T A3 series | If any dependency above is not met | `export ASCEND_BUFFER_POOL=4:8` | Configures the number and size of buffers on the NPU Device for aggregation and KV transfer (e.g., `4:8` means 4 buffers of 8MB). |
| 800 I/T A2 series | HDK >= 25.5 is recommended | `export HCCL_INTRA_ROCE_ENABLE=1` | Required by direct transmission scheme on 800 I/T A2 series|

启动 mooncake_master（`MOONCAKE_CONFIG_PATH` 指向 mooncake.json）：

```shell
mooncake_master --port 50088 --eviction_high_watermark_ratio 0.9 --eviction_ratio 0.1 --default_kv_lease_ttl 11000
```

mooncake.json 关键字段：`metadata_server=P2PHANDSHAKE`、`protocol=ascend`、`master_server_address=xx.xx.xx.xx:50088`、`global_segment_size=1GB`（需对齐 1GB）。`master_server_address` 可由 `MOONCAKE_MASTER` 环境变量覆盖；`global_segment_size` 可由 `MOONCAKE_GLOBAL_SEGMENT_SIZE` 覆盖。

PD 分离场景下，Prefill 节点 `multi_producer.sh` 关键内容：

```shell
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTHONHASHSEED=0
export PYTHONPATH=$PYTHONPATH:/xxxxx/vllm
export MOONCAKE_CONFIG_PATH="/xxxxxx/mooncake.json"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export ACL_OP_INIT_MODE=1
#A3
export ASCEND_ENABLE_USE_FABRIC_MEM=1
#A2
#export HCCL_INTRA_ROCE_ENABLE=1
#A5 UBOE
#export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["uboe:device"]}'
#A5 UB
#export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

export HCCL_RDMA_TIMEOUT=17
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000

python3 -m vllm.entrypoints.openai.api_server \
    --model /xxxxx/Qwen2.5-7B-Instruct \
    --port 8100 \
    --trust-remote-code \
    --enforce-eager \
    --no-enable-prefix-caching \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --max-model-len 32768 \
    --block-size 128 \
    --max-num-batched-tokens 16384 \
    --kv-transfer-config \
    '{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_producer",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
        "connectors": [
            {
                "kv_connector": "MooncakeConnectorV1",
                "kv_role": "kv_producer",
                "kv_port": "20001",
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
            },
            {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "lookup_rpc_port":"0",
                    "backend": "mooncake"
                }
            }  
        ]
    }
    }'
```

Decode 节点改 `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7`、`--port 8200`、`kv_role: "kv_consumer"`、`kv_port: "20002"`。MLA 模型允许 Decode 写回 KV：在 `AscendStoreConnector` 加 `consumer_is_to_put: true`，Prefill 开 PP 时再设 `prefill_pp_size` / `prefill_pp_layer_partition`：

```python
{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_consumer",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
        "lookup_rpc_port": "0",
        "backend": "mooncake",
        "consumer_is_to_put": true,
        "prefill_pp_size": 2,
        "prefill_pp_layer_partition": "30,31"
    }
}
```

启动 proxy_server（把 localhost 换成真实 IP）：

```shell
python vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --host localhost \
    --prefiller-hosts localhost \
    --prefiller-ports 8100 \
    --decoder-hosts localhost \
    --decoder-ports 8200
```

PD 混合（`kv_both`）单实例即可，无需 proxy：

```shell
python3 -m vllm.entrypoints.openai.api_server \
    --model /xxxxx/Qwen2.5-7B-Instruct \
    --port 8100 \
    --trust-remote-code \
    --enforce-eager \
    --no-enable-prefix-caching \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --max-model-len 32768 \
    --block-size 128 \
    --max-num-batched-tokens 16384 \
    --kv-transfer-config \
    '{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
        "lookup_rpc_port":"1",
        "backend": "mooncake"
    }
}' > mix.log 2>&1
```

注意：开启 `ASCEND_BUFFER_POOL` 的 MooncakeStore，性能压测前建议预热——发起输入 8K、输出 1 的请求，总数为设备（卡/裸 die）数的 2–3 倍，以触发 HCCL 全互联 one-sided 连接建立（每连接一次性开销 + 4MB/连接 持久显存）。

**Mooncake SSD Offload（需 mooncake ≥ v0.3.11，Embedded Real Client 模式）**：master 加 `--enable_offload=true --client_ttl=120`：

```shell
mooncake_master --port 50088 --eviction_high_watermark_ratio 0.9 --eviction_ratio 0.1 --default_kv_lease_ttl 11000 --enable_offload=true --client_ttl=120
```

mooncake.json 加：

```json
{
    "enable_ssd_offload": true,
    "ssd_offload_path": "/nvme/mooncake_offload"
}
```

SSD 磁盘用量环境变量：

| Environment Variable | Default | Description |
| :--- | :--- | :--- |
| `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` | `1342177280` (1280MB) | Per-rank SSD read/write buffer size in bytes. **On A3 with `ASCEND_ENABLE_USE_FABRIC_MEM=1`, must be aligned to 1GB.** |
| `MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE` | `0` | Eviction threshold in bytes. `0` = 90% of physical disk. |
| `MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY` | `none` | `none` / `fifo` / `lru`. |
| `MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES` | `2199023255552` (2 TB) | **Per-rank** maximum disk usage. Always override to match real disk capacity. |

示例（800GB 盘、8 TP rank，每 rank ~100GB）：

```shell
# 800 GB total disk, 8 ranks, ~100 GB per rank
export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=$((100 * 1024 * 1024 * 1024))
export MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=$((100 * 1024 * 1024 * 1024))
export MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru
export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1073741824   # 1GB
```

**Memcache 后端**（依赖 MemFabric）：

```shell
pip install memfabric-hybrid
pip install memcache-hybrid
```

`mmc-meta.conf` 与 `mmc-local.conf` 关键字段：`ock.mmc.meta_service_url = tcp://xx.xx.xx.xx:5000`、`ock.mmc.local_service.protocol`（A2 推荐 `device_rdma`，A3 推荐 `device_sdma`）、`ock.mmc.local_service.dram.size`。启动 MetaService：

```shell
export MMC_META_CONFIG_PATH={INSTALL_PATH}/memcache_hybrid/config/mmc-meta.conf
python -c "from memcache_hybrid import MetaService; MetaService.main()"
```

PD 分离 / PD 混合 memcache 配置在 `kv-transfer-config` 中将 `AscendStoreConnector` 的 `backend` 设为 `memcache`，并设 `MMC_LOCAL_CONFIG_PATH`。Memcache SSD Cache 需 `memcache_hybrid >= 1.2.0` + UBS IO（1.2.0 起内置），在 `mmc-local.conf` 加 `ock.mmc.local_service.storage.enabled = true`、`ubsio.disk.path = /dev/nvmexn1:...`、`ubsio.mem.size_in_gb = 10`、`ubsio.standalone.device_count = 8`，并在启动 vLLM 时 `export UBSIO_CONFIG_PATH=${MMC_LOCAL_CONFIG_PATH}`。分离部署模式（MemCache 与 vLLM 异进程，让 MemCache 抢到更大 DRAM 池）见源文。

**Yuanrong 后端**（`openyuanrong-datasystem`）：先装、起 etcd（`etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://${ETCD_IP}:2379 ...`），再用 `dscli start -w --worker_address "${WORKER_IP}:31501" --etcd_address "${ETCD_IP}:2379" --shared_memory_size_mb 40960 --arena_per_tenant 1 --enable_huge_tlb true --enable_fallocate false --rpc_thread_num 64 --oc_thread_num 64 ...` 起 worker。环境变量 `PYTHONHASHSEED=0`、`DS_WORKER_ADDR`、`DATASYSTEM_CLIENT_LOG_DIR`、`DS_ENABLE_EXCLUSIVE_CONNECTION`、`DS_ENABLE_REMOTE_H2D`。运行：

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /xxxxx/Qwen2.5-7B-Instruct \
    --port 8100 \
    --trust-remote-code \
    --enforce-eager \
    --no-enable-prefix-caching \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --max-model-len 10000 \
    --block-size 128 \
    --max-num-batched-tokens 4096 \
    --kv-transfer-config \
    '{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
        "lookup_rpc_port": "1",
        "backend": "yuanrong"
    }
}'
```

### 硬件与版本要求

- 软件：CANN >= 8.5.0，vLLM main，vLLM-Ascend main，mooncake >= 0.3.11.post1。
- A5：HDK >= 25.6 + mooncake >= v0.3.11，CANN >= 9.1.0，需 UBOE/UB 环境变量并挂载 `/dev/ummu`、`/dev/uburma`、`/usr/bin/urma_admin`、`/lib/route.conf`、`/etc/hccl_rootinfo.json`。
- A3：HDK >= 26.0（或 HDK >= 25.5 + mooncake >= v0.3.11），CANN >= 9.0.0，灵緌计算网络 >= 1.5，推荐 `ASCEND_ENABLE_USE_FABRIC_MEM=1`（fabric mem 分配需对齐 1GB）。
- A2：HDK >= 25.5 推荐，需 `HCCL_INTRA_ROCE_ENABLE=1`。
- SSD offload 需 mooncake >= v0.3.11；Memcache SSD Cache 需 `memcache_hybrid >= 1.2.0`。
- DSv4 存在已知临时问题（见 issue #9975）。

---

## Layerwise KV Pool

### 是什么

Layerwise 模式是 AscendStore KV Pool 的优化：把 KV cache 逐层 save/load，而非一次性整块拷贝；通过把第 i 层传输与第 i+1 层 attention 计算重叠，消除"整块 KV 必须先到齐才能推进"的 stall。Saving（producer / kv_both）：第 i 层 attention 算完立即把该层 KV 发往后端；Loading（consumer / kv_both）：算第 i 层前 `wait_for_layer_load` 等该层 KV 到达，第 i+1 层传输与第 i 层计算重叠。支持 PD 混合与 PD 分离两种场景。

### 何时使用

- 长提示场景，整块 KV 传输带来明显 serialization stall。
- 想把 save/load 延迟摊到 forward 过程中而非集中阻塞。

### 如何启用

前置：必须使用 `memcache` 后端。准备 huge pages、source memcache 环境、统一 hash：

```bash
# Huge pages (required by memcache device transfer)
echo 200000 > /proc/sys/vm/nr_hugepages

# Source memcache environment
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh

# Uniform hashing across nodes
export PYTHONHASHSEED=0
```

在 `AscendStoreConnector` extra config 加 `use_layerwise: true`：

```json
{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "backend": "memcache",
        "mooncake_rpc_port": "0",
        "use_layerwise": true
    }
}
```

PD 分离改 `kv_role` 为 `kv_producer` / `kv_consumer`。

关键参数：

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `use_layerwise` | `false` | Enable layer-by-layer KV save/load. Requires `backend: "memcache"`. |
| `backend` | `"mooncake"` | Storage backend. Layerwise currently supports `"memcache"` only. |
| `mooncake_rpc_port` | `"0"` | RPC port for the scheduler↔worker lookup service. `"0"` = auto-assign. |
| `layerwise_prefetch_layers` | `1` | Number of layers to prefetch ahead of the compute frontier. |
| `layerwise_max_transfer_blocks` | `0` (unlimited) | Maximum number of KV blocks per transfer batch. |
| `layerwise_max_transfer_bytes` | `0` (unlimited) | Maximum bytes per transfer batch. |
| `h2d_stagger_us` | `0` | Stagger delay (microseconds) between H2D copies across TP ranks. |
| `discard_partial_chunks` | `true` (non-layerwise) / `false` (layerwise) | Whether to discard KV for incomplete chunk boundaries. |

PD 混合单实例（无 proxy）：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1

python -m vllm.entrypoints.openai.api_server \
    --model /path/to/DeepSeek-V2-Lite \
    --port 8100 \
    --trust-remote-code \
    --enforce-eager \
    --no-enable-prefix-caching \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --kv-transfer-config '{
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "backend": "memcache",
            "mooncake_rpc_port": "0",
            "use_layerwise": true
        }
    }'
```

PD 分离需要专用 layerwise proxy（不同于标准 disagg proxy，需提供 `/v1/metaserver` 端点，且 `--host` 不能是 `0.0.0.0` 通配）：

```bash
python examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py \
    --host 127.0.0.1 \
    --port 9000 \
    --prefiller-hosts 127.0.0.1 \
    --prefiller-ports 8100 \
    --decoder-hosts 127.0.0.1 \
    --decoder-ports 8200
```

调优：增大 `layerwise_prefetch_layers`（典型 1–4）增加重叠但多吃显存；`layerwise_max_transfer_blocks` / `layerwise_max_transfer_bytes` 限制单批防止大层独占总线；多 TP 部署设 `h2d_stagger_us`（如 `100`）缓解 PCIe/HCCS 总线争用。

### 硬件与版本要求

- 后端仅 `memcache`（`mooncake`、`yuanrong` 不支持 `use_layerwise`）。
- 模型：集成 `mla_v1` 与 `sfa_v1` attention backend，支持 DeepSeek-V2/V3 等 MLA 模型；`attention_v1` 与所有 CP 变体（`mla_cp` / `sfa_cp` / `attention_cp`）尚未集成。
- 限制：不支持 hybrid KV cache（多 KV cache group family，如 MLA + 滑窗），会 `NotImplementedError`；不支持 context parallel；PD 分离必须用专用 layerwise proxy。

---

## KV Cache CPU Offload

### 是什么

`OffloadingConnector` + `NPUOffloadingSpec` 把不活跃的 KV cache 块从 NPU 内存卸载到 CPU 内存，让 vLLM 在 NPU 显存受限时支持更长上下文或更多并发请求。NPU 侧 prefix cache miss 但 CPU 命中时，KV 异步回载到 NPU，减少重算延迟。核心概念：CPU Block Pool（可选 pinned 内存）、异步 D2H/H2D 传输走专用 NPU stream、LRU 淘汰策略。

### 何时使用

- NPU 显存不足以承载目标上下文长度 / 并发数，且 CPU 内存宽裕。
- prefix 复用率高，希望避免重算。

### 如何启用

Python API：

```python
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

kv_transfer_config = KVTransferConfig(
    kv_connector="OffloadingConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "num_cpu_blocks": 1000,
        "block_size": 128,
        "spec_name": "NPUOffloadingSpec",
        "spec_module_path": "vllm_ascend.kv_offload.npu",
    },
)

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    gpu_memory_utilization=0.5,
    kv_transfer_config=kv_transfer_config,
)

sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
outputs = llm.generate(["Hello, my name is"], sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
```

在线服务：

```bash
vllm serve Qwen/Qwen3-0.6B \
    --gpu-memory-utilization 0.5 \
    --kv-transfer-config '{
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "num_cpu_blocks": 1000,
            "block_size": 128,
            "spec_name": "NPUOffloadingSpec",
            "spec_module_path": "vllm_ascend.kv_offload.npu"
        }
    }'
```

配置参数：
- `kv_connector`: 必须为 `"OffloadingConnector"`。
- `kv_role`: 设 `"kv_both"` 同时启用存与读。
- `num_cpu_blocks`: CPU 内存块数；每块内存 ∝ `block_size × num_layers × (key_size + value_size)`。
- `block_size`: CPU 侧块大小，应为 NPU 块大小整数倍，典型 `128`。
- `spec_name`: 必须 `"NPUOffloadingSpec"`。
- `spec_module_path`: 必须 `"vllm_ascend.kv_offload.npu"`。

可选 KV cache events（监控/调试）：

```python
from vllm.config import KVEventsConfig

kv_events_config = KVEventsConfig(
    enable_kv_cache_events=True,
    publisher="zmq",
    endpoint="tcp://*:5555",
    topic="kv_events",
)
```

### 硬件与版本要求

- 需要 vLLM v1 engine。
- `num_cpu_blocks` 按可用 CPU 内存调整，过大可能 host OOM；pinned 内存可用时优先使用。
- `gpu_memory_utilization` 控制留给 KV cache 的 NPU 显存，值越小 NPU KV 越少、offload 越活跃。
- 生产环境建议按真实请求模式压测找最优 `num_cpu_blocks` 与 `block_size`。

---

## Recompute CPU Offload

### 是什么

`RecomputeCPUOffloadConnector` 保存被 Decode 侧 recompute scheduler 抢占（preempt）的请求的 KV cache。当 HBM KV 块不足时 `RecomputeScheduler` 会抢占正在运行的 Decode 请求；不开此 connector 时请求回退到原始重算路径，可能被送回 Prefill 节点重跑 prefill。开启后：HBM 块被复用前，已算的 KV 块从 HBM 拷到 CPU DRAM，请求再次被调度时再从 CPU 拷回 HBM。专为在线 P/D 分离设计——Decode 节点 `max_num_batched_tokens` 常围绕 `max_num_seqs * (1 + num_spec_tokens)` 调优，适合 decode 但太小不足以重算长 prompt。

### 何时使用

- 在线 P/D 分离，Decode 节点调优为 decode 吞吐，无法高效重算长 prefill。
- 抢占频繁、不希望被抢占请求回到 Prefill 节点重跑。

### 如何启用

通过 `kv-transfer-config` 配置 `RecomputeCPUOffloadConnector`：

| Parameter | Description |
| :--- | :--- |
| `kv_connector` | Must be set to `RecomputeCPUOffloadConnector`. |
| `kv_role` | Set to `kv_consumer` on Decode nodes. |
| `cpu_bytes_to_use_per_rank` | Optional and recommended. CPU memory budget in bytes used by each rank/card. Overrides `cpu_bytes_to_use / world_size`. |
| `cpu_bytes_to_use` | Optional. Total CPU memory budget in bytes for this vLLM instance. Divided by `world_size`. Default 8 GiB total. |
| `enable_offload_prefix_caching` | Optional. Enables CPU block sharing for full hashed blocks. Default `false`. |

优先用 `cpu_bytes_to_use_per_rank`（如 `17179869184` = 16 GiB/卡）。`cpu_bytes_to_use` 会被 `world_size` 除，DP2TP8 下设 16 GiB 实际每 DP rank 卡约 8 GiB。Decode 节点还必须在 `additional-config` 启用 recompute scheduler：

```bash
--additional-config '{"scheduler_config":{"recompute_scheduler_enable":true}}'
```

`scheduler_config.recompute_scheduler_enable` 仅在 P/D 分离（`kv_role` 为 `kv_producer`/`kv_consumer`）有效，禁止在 PD-mixed（`kv_both`）启用。

P/D 分离 Decode 节点用 `MultiConnector` 同时挂 P/D connector（如 `MooncakeConnectorV1`）+ `RecomputeCPUOffloadConnector`：

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /path/to/model \
    --port 8200 \
    --trust-remote-code \
    --enforce-eager \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --max-model-len 32768 \
    --block-size 128 \
    --max-num-batched-tokens 4096 \
    --additional-config '{"scheduler_config":{"recompute_scheduler_enable":true}}' \
    --kv-transfer-config \
    '{
      "kv_connector": "MultiConnector",
      "kv_role": "kv_consumer",
      "kv_connector_extra_config": {
        "connectors": [
          {
            "kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_consumer",
            "kv_port": "28000",
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
          },
          {
            "kv_connector": "RecomputeCPUOffloadConnector",
            "kv_role": "kv_consumer",
            "kv_connector_extra_config": {
              "cpu_bytes_to_use_per_rank": 17179869184,
            }
          }
        ]
      }
    }'
```

Prefill 节点保持原 P/D connector 配置（如 `MooncakeConnectorV1` + `kv_producer`）。`kv_load_failure_policy` 配在 `MultiConnector` 顶层。Docker 内需 `--shm-size=1024g`（A3 典型 16 GiB/卡 offload）：

```bash
export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
docker run --rm \
    --name vllm-ascend \
    --shm-size=1024g \
    --net=host \
    --device /dev/davinci0 \
    ... \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    ... \
    -it $IMAGE bash
```

最小独立 connector 片段（仅用于在 P/D 分离 Decode 节点验证 recompute-offload 路径，不是 PD-mixed 部署）：

```python
from vllm.config import KVTransferConfig

kv_transfer_config = KVTransferConfig(
    kv_connector="RecomputeCPUOffloadConnector",
    kv_role="kv_consumer",
    kv_connector_extra_config={
        "cpu_bytes_to_use_per_rank": 17179869184,
        "enable_offload_prefix_caching": False,
    },
)
```

活跃路径日志标志：`Recompute preemption offload enabled for request ...` / `Created recompute offload state for request ...` / `Prepared recompute offload H2D load for request ...`。

### 硬件与版本要求

- 需 vLLM V1 engine + vLLM-Ascend recompute scheduler。
- 仅 P/D 分离 Decode 节点支持；PD-mixed / 非 P/D 部署不支持，禁止启用 `recompute_scheduler_enable`。
- Docker 大 per-rank offload 需预留足够 shm；A3 典型 16 GiB/卡 用 `--shm-size=1024g`。
- `enable_offload_prefix_caching` 实验性，默认关。
- 当前 D2H/H2D 用基本 torch copy，正确性优先未优化吞吐。
- Qwen3.5 + async scheduling 未完全支持，需禁用 async scheduling。
- CPU 内存不足时跳过 offload 回退原始重算。

---

## LMCascade Ascend Deployment

### 是什么

LMCache-Ascend 是社区维护的、在 Ascend NPU 上运行 LMCache 的插件。提供动态 KVConnector `LMCacheAscendConnectorV1Dynamic`，通过 `kv-transfer-config` 接入，作为 prefix cache 加速在线与离线推理。

### 何时使用

- 需要 LMCache 风格的 prefix cache 持久化/共享加速，且运行在 Ascend NPU 上。
- 多轮对话 / 长上下文推理场景降低 TTFT。

### 如何启用

克隆仓库（含 kvcache ops 子模块）：

```bash
cd /workspace
git clone --recurse-submodules https://github.com/LMCache/LMCache-Ascend.git
```

Docker 构建：

```bash
cd /workspace/LMCache-Ascend
docker build -f docker/Dockerfile.a2.openEuler -t lmcache-ascend:v0.3.12-vllm-ascend-v0.11.0-openeuler .
```

运行：

```bash
DEVICE_LIST="0,1,2,3,4,5,6,7"
docker run -it \
    --privileged \
    --cap-add=SYS_RESOURCE \
    --cap-add=IPC_LOCK \
    -p 8000:8000 \
    -p 8001:8001 \
    --name lmcache-ascend-dev \
    -e ASCEND_VISIBLE_DEVICES=${DEVICE_LIST} \
    -e ASCEND_RT_VISIBLE_DEVICES=${DEVICE_LIST} \
    -e ASCEND_TOTAL_MEMORY_GB=32 \
    -e VLLM_TARGET_DEVICE=npu \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /etc/localtime:/etc/localtime \
    -v /var/log/npu:/var/log/npu \
    -v /dev/davinci_manager:/dev/davinci_manager \
    -v /dev/devmm_svm:/dev/devmm_svm \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /etc/hccn.conf:/etc/hccn.conf \
    lmcache-ascend:v0.3.12-vllm-ascend-v0.11.0-openeuler \
    /bin/bash
```

手动安装（假设 `/workspace` 且 vllm/vllm-ascend 已装）：

```bash
NO_CUDA_EXT=1 pip install lmcache==0.3.12
cd /workspace/LMCache-Ascend
python3 -m pip install --no-build-isolation -e .
```

在线服务：

```bash
python \
    -m vllm.entrypoints.openai.api_server \
    --port 8100 \
    --model /data/models/Qwen/Qwen3-32B \
    --trust-remote-code \
    --disable-log-requests \
    --block-size 128 \
    --kv-transfer-config '{"kv_connector":"LMCacheAscendConnector","kv_role":"kv_both"}'
```

离线：

```python
ktc = KVTransferConfig(
        kv_connector="LMCacheAscendConnector",
        kv_role="kv_both"
    )
```

### 硬件与版本要求

- Ascend NPU（镜像示例为 A2 openEuler，`Dockerfile.a2.openEuler`）。
- vllm / vllm-ascend 已安装。
- lmcache `0.3.12`、vllm-ascend `v0.11.0`（镜像 tag 体现）。
- 更多部署细节见 LMCache-Ascend 官方 README。

---

## Sleep Mode

### 是什么

Sleep Mode 是 vLLM 提供的、用于把模型权重从 NPU 卸载到 CPU 并丢弃 KV cache 的 API，面向 RL 后训练（PPO/GRPO/DPO 等）工作负载。生成阶段与训练阶段可能采用不同模型并行策略，训练时需释放 vLLM 占用的 KV cache 甚至模型权重，避免 NPU 资源争用。两级 sleep：
- Level 1 Sleep：卸载权重到 CPU、丢弃 KV cache；适合稍后复用同一模型；需 CPU 内存能装下权重。
- Level 2 Sleep：权重与 KV cache 内容都丢弃；适合切换/更新模型。

`enable_sleep_mode=True` 后内存管理在一个特定内存池中进行，加载模型与初始化 KV cache 时把内存打标 `{"weight": data, "kv_cache": data}`。因依赖底层 AscendCL，须按安装指南源码构建；< v0.12.0rc1 需 `export COMPILE_CUSTOM_KERNELS=1`。

### 何时使用

- RL 后训练，policy 模型 autoregressive 生成与训练交替、需在训练阶段回收 NPU 显存。
- 同一引擎稍后复用同一模型（Level 1）或切换/更新模型（Level 2）。

### 如何启用

可选 extra cleanup（RL 训练侧需更多 NPU 显存时）：

```python
llm = LLM(
    "Qwen/Qwen2.5-0.5B-Instruct",
    enable_sleep_mode=True,
    additional_config={"enable_sleep_mode_extra_cleanup": True},
)
```

在线服务：

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --enable-sleep-mode \
    --additional-config '{"enable_sleep_mode_extra_cleanup": true}'
```

`enable_sleep_mode_extra_cleanup` 开启时，`sleep()` 额外清理 ACL graph attention workspaces、失效 captured ACL graph cache、重置 model runner graph manager（重捕）、等待 pending PP send work、同步 NPU、销毁 HCCL process group；`wake_up()` 恢复 HCCL PG、刷新 MoE dispatcher HCCL metadata、恢复 sleep-mode allocator 内存、按需重捕 ACL graph。权衡：降低 sleep 时 NPU 显存占用，但延长 wakeup 延迟（开 ACL graph 时 wakeup 必须 `capture_model()`）。

Level 2 wakeup 可分两阶段：

```python
llm.wake_up(tags=["weights"])
# Reload or update model weights here.
llm.wake_up(tags=["kv_cache"])
```

开 extra cleanup 时，仅当 `tags` 为 `None` 或含 `"kv_cache"` 才重捕 ACL graph。MoE 非量化模型权重在 `process_weights_after_loading()` 时对 `w13_weight`/`w2_weight` 做 `transpose(1,2)` 转 `torch_npu.npu_grouped_matmul` 所需布局；`wake_up()` 恢复未转置内存后会按 tag `"weights"` 重新应用同样转置（dense 模型与量化模型跳过）。

离线示例：

```python
import os

import torch
from vllm import LLM, SamplingParams
from vllm.utils.mem_constants import GiB_bytes

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ASCEND_ENABLE_NZ"] = "0"

if __name__ == "__main__":
    prompt = "How are you?"

    free, total = torch.npu.mem_get_info()
    print(f"Free memory before sleep: {free / 1024 ** 3:.2f} GiB")
    used_bytes_baseline = total - free
    llm = LLM("Qwen/Qwen2.5-0.5B-Instruct", enable_sleep_mode=True)
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    output = llm.generate(prompt, sampling_params)

    llm.sleep(level=1)

    free_npu_bytes_after_sleep, total = torch.npu.mem_get_info()
    print(f"Free memory after sleep: {free_npu_bytes_after_sleep / 1024 ** 3:.2f} GiB")
    used_bytes = total - free_npu_bytes_after_sleep - used_bytes_baseline
    assert used_bytes < 1 * GiB_bytes

    llm.wake_up()
    output2 = llm.generate(prompt, sampling_params)
    assert output[0].outputs[0].text == output2[0].outputs[0].text
```

在线服务（须 dev-mode 并显式 `VLLM_SERVER_DEV_MODE` 以暴露 sleep/wake up 端点）：

```bash
export VLLM_SERVER_DEV_MODE="1"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_MODELSCOPE="True"
export VLLM_ASCEND_ENABLE_NZ="0"

vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-sleep-mode

# sleep level 1
curl -X POST http://127.0.0.1:8000/sleep \
    -H "Content-Type: application/json" \
    -d '{"level": "1"}'

curl -X GET http://127.0.0.1:8000/is_sleeping

# sleep level 2
curl -X POST http://127.0.0.1:8000/sleep \
    -H "Content-Type: application/json" \
    -d '{"level": "2"}'

# wake up
curl -X POST http://127.0.0.1:8000/wake_up

# wake up with tag, tags must be in ["weights", "kv_cache"]
curl -X POST "http://127.0.0.1:8000/wake_up?tags=weights"

curl -X GET http://127.0.0.1:8000/is_sleeping

curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "prompt": "The future of AI is",
        "max_tokens": 7,
        "temperature": 0
    }'
```

### 硬件与版本要求

- 依赖底层 AscendCL，须按安装指南源码构建；< v0.12.0rc1 需 `export COMPILE_CUSTOM_KERNELS=1`。
- Level 1 需 CPU 内存足以容纳模型权重。
- 示例模型 `Qwen2.5-0.5B-Instruct`，可配 `VLLM_USE_MODELSCOPE=True` 从 ModelScope 自动下载。
- 在线 sleep/wake up 端点需 `VLLM_SERVER_DEV_MODE="1"`（防恶意访问）。

---

## NetLoader

### 是什么

NetLoader 是 vLLM 0.10 `register_model_loader` API 实现的权重加载器插件，利用 NPU 卡间高带宽 P2P 传输加载模型权重。流程：server 预加载模型 → 新 client 实例请求权重传输 → 校验模型与分区一致后 client 用 HCCL collective 通信（send/recv）按模型存储顺序收权重。server 通过子线程与正常推理并存，借助 `stateless_init_torch_distributed_process_group`；client 无需从存储读权重即可初始化。

### 何时使用

- 降低启动延迟（复用已加载权重直接 NPU 卡间传输，快于远端/本地拉取）。
- 缓解网络与存储负载（避免反复下载权重文件）。
- 提升资源利用率、降低成本（减少待机节点依赖）。
- 故障恢复时新实例快速接管，提升业务连续性与高可用。

### 如何启用

`--load-format=netloader` + `--model-loader-extra-config`（JSON 字符串）。配置字段：

| Field Name | Type | Description | Allowed Values / Notes |
|---|---|---|---|
| **SOURCE** | List | Weight data sources. Each item: `device_id` + `sources` (IP:port). Example: `{"SOURCE": [{"device_id": 0, "sources": ["10.170.22.152:19374"]}, {"device_id": 1, "sources": ["10.170.22.152:11228"]}]}`. If omitted/empty, fallback to default loader. Second priority. | A list of objects with keys `device_id: int` and `sources: List[str]` |
| **MODEL** | String | The model name, used to verify consistency between client and server. | Defaults to the `--model` argument if not specified. |
| **LISTEN_PORT** | Integer | Base port for the server listener. Actual port = `LISTEN_PORT + RANK`. If omitted, a random valid port is chosen. Valid range: 1024–65535. If out of range, that server instance won't open a listener. |
| **INT8_CACHE** | String | Behavior for handling int8 parameters in quantized models. | One of `["hbm", "dram", "no"]`. Default: `"no"`. |
| **INT8_CACHE_NAME** | List | Names of parameters to which `INT8_CACHE` is applied (filtering). | Default: `None` (no filtering—all parameters). |
| **OUTPUT_PREFIX** | String | Prefix for writing per-rank listener address/port files in server mode. | If set, each rank writes to `{OUTPUT_PREFIX}{RANK}.txt` (text), content = `IP:Port`. |
| **CONFIG_FILE** | String | Path to a JSON file specifying the above configuration. | If provided, the SOURCE inside this file has **first priority** (overrides SOURCE in other configs). |

Server：

```shell
VLLM_SLEEP_WHEN_IDLE=1 vllm serve <model_file> \
  --tensor-parallel-size 1 \
  --served-model-name <model_name> \
  --enforce-eager \
  --port `<port>` \
  --load-format netloader
```

Client：

```shell
export NETLOADER_CONFIG='{"SOURCE":[{"device_id":0, "sources": ["<server_IP>:<server_Port>"]}]}'

VLLM_SLEEP_WHEN_IDLE=1 ASCEND_RT_VISIBLE_DEVICES=<device_id_diff_from_server> \
  vllm serve <model_file> \
  --tensor-parallel-size 1 \
  --served-model-name <model_name> \
  --enforce-eager \
  --port <client_port> \
  --load-format netloader \
  --model-loader-extra-config="${NETLOADER_CONFIG}"
```

启动后用 temperature=0 推理对比输出验证一致性。

### 硬件与版本要求

- vLLM 0.10+（`register_model_loader` API）。
- 每个 worker 进程必须绑定监听端口（用户指定或随机），用户指定时确保可用。
- 需额外片上内存建立 HCCL 连接（`HCCL_BUFFERSIZE`，默认 ~200MB），通过 `--gpu-memory-utilization` 预留。
- 推荐 `VLLM_SLEEP_WHEN_IDLE=1` 以缓解不稳定/慢连接（vLLM Issue #16660 / PR #16226）。

---

## RFORK

### 是什么

RFork 是 vLLM-Ascend 的暖启动权重加载路径。新实例不总从存储读权重，而是向外部 planner 请求一个兼容 seed 实例，通过 `YuanRong TransferEngine` 直接拉权重。流程：`--load-format rfork` → RFork 用模型身份+部署拓扑构造 seed key → 向 planner 请求匹配 seed → 命中则本地初始化模型结构、注册本地权重内存、取远端 transfer-engine 元数据、批量传输到本地参数缓冲 → 未命中或失败则清理回退默认 loader → 加载完成后启动本地 seed 服务并周期心跳上报 planner，供后续实例复用。

### 何时使用

- 首次成功加载后的横向扩容：第一实例从存储加载，后续相同部署身份的实例复用作为 seed 缩短启动时间。
- 弹性 serving 集群：实例动态创建/回收。
- 拓扑敏感部署：seed key 编码 `kv_role`、`node_rank`、可选 `pp_rank`、`tp_rank`、可选 `ep_rank`、可选 `draft` 角色，仅拓扑兼容实例才匹配。

### 如何启用

`--load-format rfork` + `--model-loader-extra-config`（JSON 字符串）。前置：
- 每个实例安装 `YuanRong TransferEngine`。
- 运行实现 RFork seed 协议的 planner 服务（mock 脚本 `examples/rfork/rfork_planner.py`）。

配置字段：

| Field Name | Type | Description | Allowed Values / Notes |
|---|---|---|---|
| **model_url** | String | Logical model identifier used to build the RFork seed key. | Required for RFork transfer. Instances that should share seeds must use the same value. |
| **model_deploy_strategy_name** | String | Deployment strategy identifier used together with `model_url` to build the seed key. | Required for RFork transfer. |
| **rfork_scheduler_url** | String | Base URL of the planner service used for seed allocation, release, and heartbeat. | Required for planner-based matching. Example: `http://127.0.0.1:1223`. |
| **rfork_seed_timeout_sec** | Number | Timeout for waiting until the local seed HTTP service becomes healthy after startup. | Optional. Default: `5.0`. Must be > `0`. Invalid values fall back to the default. |
| **rfork_seed_key_separator** | String | Separator used when building the RFork seed key string. | Optional. Default: `$`. |

seed key 由 `model_url`、`model_deploy_strategy_name`、由 `kv_transfer_config.kv_role` 或 `kv_both` 推导的分离模式、`node_rank`、`pp_rank`（PP>1 时）、`tp_rank`、`ep_rank`（MoE EP 时）、可选 `draft` 后缀组成。量化模型传输的是 Ascend weight post-processing 后的 tensor（含转置、NZ 格式、packed weights、derived scale、MLA/SFA 的 `W_UV`/`W_UK_T`），空 tensor 不入传输清单。

安装 TransferEngine：

```shell
pip install openyuanrong-transfer-engine
```

启动 planner：

```shell
python rfork_planner.py \
  --host 0.0.0.0 \
  --port <planner_port>
```

启动 vLLM 实例（首实例通常无兼容 seed，回退默认 loader；加载完成即上报自己为 seed）：

```shell
export RFORK_CONFIG='{
  "model_url": "<model_url>",
  "model_deploy_strategy_name": "<deploy_strategy>",
  "rfork_scheduler_url": "http://<planner_ip>:<planner_port>"
}'

vllm serve <model_path> \
  --tensor-parallel-size 1 \
  --served-model-name <served_model_name> \
  --port <port> \
  --load-format rfork \
  --model-loader-extra-config "${RFORK_CONFIG}"
```

成功传输日志：`transfer weights starts` / `transfer weights time`；回退日志：`RFork transfer failed`。已验证模型（A2）：

| Model | Precision / Quantization | Hardware | Validation Status | Notes |
|---|---|---|---|---|
| Qwen2.5-7B | BF16 | A2 | Tested | RFork transfer has been validated. |
| Qwen3-32B | BF16 | A2 | Tested | RFork transfer has been validated. |
| Qwen3-235B-A22B | BF16 | A2 | Tested | RFork transfer has been validated. |
| DeepSeek-V4-Flash-W8A8-MTP | W8A8 | A2 | Tested | RFork transfer with MTP draft model has been validated. |
| GLM5-W4A8 | W4A8 | A2 | Tested | RFork transfer has been validated. |
| Kimi2.5-W4A8 | W4A8 | A2 | Tested | RFork transfer has been validated. |

### 硬件与版本要求

- 运行时依赖 `YuanRong TransferEngine`，缺失则无法初始化传输后端。
- 每个 worker 进程必须绑定监听端口（随机分配）。
- 不支持 dynamic EPLB：若 `eplb_config.dynamic_eplb` 或 `eplb_config.expert_map_record_path` 启用动态 EPLB，RFork 传输被旁路走默认 loader。
- 改 RFork 代码或模型参数后须重启 planner 与所有 vLLM 实例，并换新 `model_deploy_strategy_name`。
- mock planner `rfork_planner.py` 仅供简单演示，生产需自实现 planner。

---

## UCM Deployment

### 是什么

Unified Cache Manager (UCM) 是面向 vLLM/vLLM-Ascend prefix-caching 的外部 KV-cache 存储层。区别于 KV Pooling（仅靠聚合设备内存扩容、受 HBM/DRAM 大小限制、无持久化），UCM 采用存算分离与分层设计：每节点本地 DRAM 作快速缓存，共享后端（NFS / 3FS / 企业存储）作持久 KV store。三层缓存 `HBM → DRAM → Storage Backend`。优势：突破设备内存容量上限、持久可靠（跨重启/故障/调度迁移）、多场景加速（prefix cache、训练-free 稀疏注意力 GSA/CacheBlend、PD 分离）、显著性能提升（多轮对话与长上下文推理延迟降低 3-10x，prefix cache TTFT 提升可达 8x）。

### 何时使用

- prefix cache 容量需超出 HBM/DRAM，且需持久化。
- 需跨实例/跨重启复用 KV cache。
- 极长序列推理需要 GSA / CacheBlend 稀疏注意力。
- PD 分离需要存算分离（Centralized PD 经共享存储，或 P2P PD 经 Mooncake + UCM prefix cache）。

### 如何启用

前置：Linux + Ascend NPU（典型 Atlas 800 A2 系列），vLLM main，vLLM-Ascend main。UCM 安装参考官方 Ascend NPU 安装指南。

**Centralized PD 分离**（KV 经统一存储池传输，跨节点需共享存储后端如 NFS/3FS）：

UCM 配置文件 `ucm_config_example.yaml`（PipelineStore 推荐，链式 Cache Store + Posix Store）：

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: "/mnt/test1"
      cache_buffer_capacity_gb: 64
enable_event_sync: true
use_layerwise: false
```

Prefill（`kv_both`，2P2D 示例：node 192.168.10.1 port 7800/7801）：

```bash
export PYTHONHASHSEED=123456
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
vllm serve /models/QwQ-32B \
    --host 0.0.0.0 \
    --port 7800 \
    --gpu-memory-utilization 0.92 \
    --data-parallel-size 1 \
    --tensor-parallel-size 4 \
    --seed 1024 \
    --max-model-len 17000 \
    --max-num-batched-tokens 8000 \
    --max-num-seqs 20 \
    --trust-remote-code \
    --enforce-eager \
    --kv-transfer-config \
    '{
        "kv_connector": "UCMConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/path/to/ucm_config_example.yaml"}
    }'
```

Decode（`kv_both`，node 192.168.10.2 port 7802/7803，加 `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`）：

```bash
export PYTHONHASHSEED=123456
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
vllm serve /models/QwQ-32B \
    --host 0.0.0.0 \
    --port 7802 \
    --gpu-memory-utilization 0.92 \
    --data-parallel-size 1 \
    --tensor-parallel-size 4 \
    --seed 1024 \
    --max-model-len 17000 \
    --max-num-batched-tokens 8000 \
    --max-num-seqs 20 \
    --trust-remote-code \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --kv-transfer-config \
    '{
        "kv_connector": "UCMConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/path/to/ucm_config_example.yaml"}
    }'
```

负载均衡：

```bash
python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --port 7805 \
    --host 0.0.0.0 \
    --prefiller-hosts 192.168.10.1 192.168.10.1 \
    --prefiller-ports 7800 7801 \
    --decoder-hosts 192.168.10.2 192.168.10.2 \
    --decoder-ports 7802 7803
```

压测：

```bash
vllm bench serve \
    --backend vllm \
    --model /models/QwQ-32B \
    --host 192.168.10.1 \
    --port 7805 \
    --seed 123456 \
    --dataset-name random \
    --num-prompts 10 \
    --random-input-len 8000 \
    --random-output-len 1000 \
    --request-rate inf \
    --ignore-eos
```

**Distributed PD (P2P)**（Mooncake 直传 KV + UCM 在 Prefill 上做 prefix cache）：Mooncake master 同 KV Pool 起法，mooncake.json `master_server_address=192.168.10.1:50088`。Prefill 用 `MultiConnector`（`MooncakeConnectorV1` + `UCMConnector`，`kv_producer`）；Decode 仅 `MooncakeConnectorV1`（`kv_consumer`）。从 vLLM-Ascend 0.11.0 起官方镜像预装 Mooncake。

**PD-Mixed**（同实例 Prefill/Decode 交织调度，UCM 提供持久 KV cache）：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
vllm serve /models/QwQ-32B \
    --host 0.0.0.0 \
    --port 7800 \
    --gpu-memory-utilization 0.92 \
    --data-parallel-size 2 \
    --tensor-parallel-size 4 \
    --seed 1024 \
    --max-model-len 17000 \
    --max-num-batched-tokens 8000 \
    --max-num-seqs 20 \
    --trust-remote-code \
    --enforce-eager \
    --block-size 128 \
    --kv-transfer-config \
    '{
        "kv_connector": "UCMConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/path/to/ucm_config_example.yaml"}
    }'
```

跑两次 bench，第二次 TTFT 应明显下降；日志看 `INFO ucm_connector.py:xxx: request_id: xxx, total_blocks_num: xxx, hit hbm: 0, hit external: xxx`。

**大尺度 Expert Parallelism PD 分离示例**：Prefill 4 节点 DP4TP8（每节点 1 DP 进程 TP8）、Decode 4 节点 DP8TP4（每节点 2 DP 进程 TP4），共 8 台 Atlas 800T A2（每台 8 张 910B3）。Prefill `prefill.sh` 关键环境变量与配置（`MultiConnector`：`MooncakeConnectorV1` + `UCMConnector`，`--enable-expert-parallel --quantization ascend --additional-config '{"enable_weight_nz_layout":true,"enable_prefill_optimizations":true}'`）；Decode `decode.sh` 仅 `MooncakeConnectorV1`（`kv_consumer`，`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`）。`run_multi_dp.sh` 按节点循环起多 DP（Prefill `dp_rank_start=0/1/2/3`，Decode `dp_rank_start=0/2/4/6`）。负载均衡 `--prefiller-hosts 192.168.10.1 192.168.10.2 192.168.10.3 192.168.10.4`、`--decoder-hosts 192.168.10.5 192.168.10.5 192.168.10.6 192.168.10.6 ...`。压测需先以 0.8 prefix 比例预 seed KV cache（输入 = 目标输入 × 0.8、输出 = 1），再正式测（目标输入 + 输出 1000）。UCM PC 相比 HBM PC 在 32K/64K/128K 输入下 TTFT 显著降低（如 128K+1K：Recalc 268016ms / HBM PC 267680ms / UCM PC 105083ms）。

### 硬件与版本要求

- OS：Linux；硬件：Ascend NPU（典型 Atlas 800 A2 系列）。
- 框架：vLLM main、vLLM-Ascend main（也支持 SGLang main）。
- 支持平台：CUDA（H100/H20/L40/L20）、CANN（Atlas A2 / A3 推理产品）、MUSA（Mthreads S5000）、MACA（MetaX C500）。
- PipelineStore 为推荐 connector；更多配置见 UCM PipelineStore 文档。
- vLLM-Ascend 0.11.0 起官方镜像预装 Mooncake（P2P PD 用）。
- 完整支持矩阵见 UCM Support Matrix。

---

## 源文件清单

| # | 路径 | 状态 |
|---|---|---|
| 1 | `docs/source/user_guide/feature_guide/kv_pool.md` | OK |
| 2 | `docs/source/user_guide/feature_guide/layerwise_kv_pool.md` | OK |
| 3 | `docs/source/user_guide/feature_guide/kv_cache_cpu_offload.md` | OK |
| 4 | `docs/source/user_guide/feature_guide/recompute_cpu_offload.md` | OK |
| 5 | `docs/source/user_guide/feature_guide/lmcache_ascend_deployment.md` | OK |
| 6 | `docs/source/user_guide/feature_guide/sleep_mode.md` | OK |
| 7 | `docs/source/user_guide/feature_guide/netloader.md` | OK |
| 8 | `docs/source/user_guide/feature_guide/rfork.md` | OK |
| 9 | `docs/source/user_guide/feature_guide/ucm_deployment.md` | OK |
