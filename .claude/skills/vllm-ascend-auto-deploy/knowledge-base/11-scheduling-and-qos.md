> 来源：vllm-ascend docs（main 分支，抓取于 2026-07-28），整合自多个源文件

# Scheduling & QoS 知识库（vLLM-Ascend）

本文件整合 vLLM-Ascend 的调度器与 QoS 相关特性：Batch Invariance（批不变性）、Batch-Job-Aware Scheduler（批作业感知调度器）、ShortRequestFirst（短请求优先 prefill）、AI QoS（流量隔离与差异化调度）、Ray 多机分布式部署、Suffix Speculative Decoding（后缀投机解码）。所有命令、CLI flag、环境变量名与取值、scheduler-arg、ray start 命令、配置片段、版本/硬件要求均按原文逐字保留；解释性文字用中文。

---

## Batch Invariance

**是什么**：Batch Invariance（批不变性）保证模型输出确定性，且与 batch size 或 batch 内请求顺序无关。同一输入无论怎样分批都会得到相同输出。

**何时使用**：框架/模型调试（确定性输出便于复现问题）、强化学习（RL rollout 可复现、稳定训练）、大规模推理系统（测试、验证、一致性保证）。开启后会牺牲部分性能换取可复现性（属于有意为之的 trade-off）。

**硬件/版本要求**（逐字）：
- `Batch invariance currently requires Ascend Atlas A2 and A3 inference products NPUs.`
- `We will support Ascend 950 Products and other NPUs in the future.`

**软件要求**（逐字）：
- `Batch invariance requires a custom operator library for Atlas A2 and A3 inference products, and users need to set `VLLM_BATCH_INVARIANT=1` before building vllm-ascend to install the batch invariance custom operator library during the installation process.`
- 该特性当前处于 beta：`Batch invariance is currently in beta. Some features are still under active development.`

**如何开启**：构建 vllm-ascend 前先设置环境变量，再在运行时设置。

```bash
export VLLM_BATCH_INVARIANT=1
```

Online Inference（Server Mode）—— 注意 `cudagraph_mode` 必须为 `PIECEWISE`：

```bash
VLLM_BATCH_INVARIANT=1 vllm serve Qwen/Qwen3-8B \
  --compilation-config '{"cudagraph_mode": "PIECEWISE"}'
```

OpenAI 兼容客户端示例：

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

# These requests will produce deterministic outputs
# regardless of batch size or order
response = client.completions.create(
    model="Qwen/Qwen3-8B",
    prompt="The future of AI is",
    max_tokens=100,
    temperature=0.7,
    seed=42,
)

print(response.choices[0].text)
```

Offline Inference：

```python
import os
os.environ["VLLM_BATCH_INVARIANT"] = "1"

from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
    "Machine learning enables",
    "Deep learning models can",
]

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=100,
    seed=42,
)

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=1,
    compilation_config={"cudagraph_mode": "PIECEWISE"},
)

# Outputs will be deterministic regardless of batch size
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}\n")
```

**已测试模型**（逐字）：
- `Qwen3 (Dense)`: `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-8B`
- `Qwen3 (MoE)`: `Qwen/Qwen3-30B-A3B`, `Qwen/Qwen3-235B-A22B`

**实现要点**：开启后 vLLM 使用确定性 kernel（attention 等）、保证跨 batch size 数值一致、并禁用可能引入非确定性的优化。批不变性 attention 算子当前不支持 `FULL`、`FULL_DECODE_ONLY` cudagraph mode。开启可能影响性能。跟踪 issue：https://github.com/vllm-project/vllm-ascend/issues/5487

---

## Batch Job Aware Scheduler

**是什么**：Batch-Job-Aware Scheduler 是面向**离线批推理**的专用调度器，目标是在并发处理多个 batch job（每个 job 有独立请求集合）时最大化吞吐与硬件利用率。它通过三类策略提升吞吐：

1. **LPT (Longest Processing Time first) scheduling**：优先调度长任务，再用短任务填补空隙（尤其 decode 步），提高每轮调度的平均 token 数。
2. **KV cache reservation**：为运行中的请求预估并预留 KV cache 预算，减少 preemption 开销。
3. **Job-aware request grouping**：按 **job name**（从 request ID 中提取）分组，每个 job 一个 bucket，按 KV cache 可用量动态调整 job 调度顺序——可用 token > threshold（默认 4096）时优先长 decode job，≤ threshold 时优先短 decode job。

**何时使用**：离线批处理、或“请求等待时间不敏感”的在线推理。**注意**：该调度器**不实现防饿死**，对有严格延迟或公平性要求的请求不适用。

**硬件/版本要求**（逐字）：
- `vLLM v1 engine` is required（构建在 v1 调度框架之上）。
- `Ascend NPU` with sufficient memory for the target model(s).

**如何开启**：通过 `additional_config` 的 `scheduler_config.batch_job_sched_config.enabled=true` 开启。**当前仅在 offline batch 模式支持**。

基础离线批：

```bash
python -m vllm.entrypoints.openai.run_batch \
    --model /path/to/model \
    -i /path/to/input.jsonl \
    -o /path/to/output.jsonl \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --additional-config '{"scheduler_config": {"batch_job_sched_config": {"enabled": true}}}'
```

**Request ID 格式**：通过 `#job_name[${JOB_NAME}]#` 标签把 job name 嵌入 request ID。无标签的请求归入 `__default__` job。批量输入文件示例：

```jsonl
# In your batch input file (e.g., input.jsonl):
{"custom_id": "#job_name[job_A]#req_001", "method": "POST", "url": "/v1/chat/completions", "body": {"request_id": "#job_name[job_A]#req_001", "messages": [{"role": "user", "content": "Hello"}], "n": 1}}
{"custom_id": "#job_name[job_A]#req_002", "method": "POST", "url": "/v1/chat/completions", "body": {"request_id": "#job_name[job_A]#req_002", "messages": [{"role": "user", "content": "What is AI?"}], "n": 1}}
{"custom_id": "#job_name[job_B]#req_003", "method": "POST", "url": "/v1/chat/completions", "body": {"request_id": "#job_name[job_B]#req_003", "messages": [{"role": "user", "content": "Explain quantum computing"}], "n": 1}}
```

**配置参数**（均在 `additional_config` 的 `scheduler_config.batch_job_sched_config` 下）：

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Enable the batch-job-aware scheduler |
| `max_jobs` | int | `20` | Maximum number of tracked jobs |
| `reserve_margin_blocks` | int | `2` | Extra block margin added to the KV cache reserve as safety buffer |
| `reserve_max_blocks` | int | `8` | Maximum number of blocks that can be reserved |
| `low_available_tokens_threshold` | int | `4096` | Threshold for prioritising long vs short decode jobs |
| `short_decode_token_threshold` | int | `32` | Threshold for classifying a job as "short decode" |

带自定义配置的离线批：

```bash
python -m vllm.entrypoints.openai.run_batch \
    --model /path/to/model \
    -i /path/to/input.jsonl \
    -o /path/to/output.jsonl \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --additional-config '{
        "scheduler_config": {
            "batch_job_sched_config": {
                "enabled": true,
                "max_jobs": 10,
                "reserve_margin_blocks": 4,
                "reserve_max_blocks": 12,
                "low_available_tokens_threshold": 2048,
                "short_decode_token_threshold": 32
            }
        }
    }'
```

Python API：

```python
from vllm import LLM

llm = LLM(
    model="/path/to/model",
    max_model_len=4096,
    gpu_memory_utilization=0.9,
    additional_config={
        "scheduler_config": {
            "batch_job_sched_config": {
                "enabled": True,
            },
        },
    },
)
```

**Decode 长度估计**：纯 **EWMA（指数加权移动平均）** 估计器——无样本时返回冷启动默认 128 tokens；首个观测初始化 EWMA；后续增量更新。据此区分长 decode job（资源充足时优先）与短 decode job（资源紧张时优先）。

**最佳实践**：在 request ID 中用 `#job_name[${JOB_NAME}]#` 前缀；长 decode 占比高时可调低 `low_available_tokens_threshold` 以保持长 job 优先，混合负载保持默认。

---

## Short Request First

**是什么**：ShortRequestFirst 在 prefill 阶段通过让短 prompt 优先于长 prompt 运行来减少队头阻塞（head-of-line blocking），面向少量长请求会拖慢大量短请求的混合 prompt 长度流量。

**何时使用**：请求长度高度倾斜；短请求 TTFT 比严格 FCFS 顺序更重要；服务使用 FCFS 调度器（同步或异步均可）。负载均匀或 FCFS 顺序更重要时保持关闭。

**如何开启**：在 `scheduler_config` 中添加 `short_request_first_config`。FCFS 同步/异步普通部署、PD 分离 prefill（P）节点、PD-mixed 部署均支持；**不需要** `recompute_scheduler_enable`。不支持 batch-job-aware、profiling-chunk 调度，也不支持 PD 分离 D 节点（`kv_role='kv_consumer'`）。

```json
{
  "scheduler_config": {
    "short_request_first_config": {
      "enabled": true,
      "threshold": 256,
      "long_max_wait_ms": 2000
    }
  }
}
```

显式关闭：

```bash
vllm serve <model> \
  --additional-config '{"scheduler_config": {"short_request_first_config": {"enabled": false}}}'
```

**字段**：

- `enabled` (bool, default `false`)：开关。
- `threshold` (int, default `256`)：`num_prompt_tokens <= threshold` 视为短请求。
- `long_max_wait_ms` (float, default `0`)：长请求在短请求之后最多等待多久后可被提升到前面；`0` 表示关闭长请求提升、保持严格短请求优先。

**threshold 调优**：太低则大多数请求落入 long 通道、分流失效；太高则大多数落入 short 通道、同样失效。应设在短请求主簇与长请求尾之间的谷底。无精细流量模型时，工程基线取 prompt 长度的 `P70-P85`，再用真实请求长度直方图细化。双峰负载下，threshold 通常靠近短请求簇上沿而非最长 prompt。

**long_max_wait_ms 调优**：这是公平性闸门而非吞吐优化旋钮。先调 `threshold`；若退化告警频繁，先怀疑 `threshold` 偏小，再考虑调大 `long_max_wait_ms`。流程：从 `long_max_wait_ms = 0` 建立严格短优先基线 → 测量长请求等待分布 → `W_normal` 取 `P90`/`P95` → `W_slo` 为可容忍的最大长请求排队延迟 → 将 `long_max_wait_ms` 设在 `[W_normal, W_slo]` 之间（短请求 TTFT 更重要则偏 `W_slo`，长请求公平更重要则偏 `W_normal`）。不要用短请求平均 TTFT 反推。若 aged-long 提升反复出现，先调大 `threshold` 再调大 `long_max_wait_ms`。

**调度行为**：等待队列分三通道——`immediate`（preempted 或已有计算 token 的恢复型请求）、`short`（prompt 长度 ≤ threshold）、`long`（prompt 长度 > threshold）。派发优先级：`immediate > aged-long > short > long`。

**退化告警**：若长请求连续 3 次派发被提升到短请求之前，vLLM Ascend 会告警，说明正漂移向长请求优先（通常是 threshold 对当前流量偏小）。告警时应调大 `short_request_first_config.threshold` 或关闭 `short_request_first_config.enabled`。每 5 秒输出一次聚合统计日志。

**调度器兼容性**：ShortRequestFirst 仅改等待队列策略，要求 FCFS 调度，支持同步/异步。`VLLM_ASCEND_BALANCE_SCHEDULING` 保持原行为（控制 balance 调度的跨 DP 准入逻辑），ShortRequestFirst 独立安装，balance 关闭路径仍委托 vLLM 调度但使用 ShortRequestFirst 等待队列。

---

## AI QoS

**是什么**：推理场景中存在算子下发、集合通信、KVCache 等多种流量，它们经网络传输相互影响、增加推理延迟。例如 Agentic AI 时代上下文增长使 KVCache 体积膨胀，为节省 HBM 而把 KVCache 卸载到 DDR，并用“计算掩盖 KVCache”的流水编排（在当前层计算/通信时预取下一层 KVCache），这会引入 KVCache 与算子下发/集合通信的流量冲突。AI QoS 通过 Virtual Lane（VL）在 UB switch 上隔离流量并按 VL 差异化调度（strict priority, SP）来缓解：节点上为不同 NPU channel 设置优先级、建立 NPU channel 优先级与 UB switch VL 的映射、按优先级在 UB switch 各 VL 间差异化调度。

**何时使用**：Atlas 800T A3 服务器 / Atlas 900 A3 SuperPoD 集群上，多类流量在 UB switch 上冲突、影响推理 SLO 时。需在特权容器中运行。

**硬件/版本要求**（逐字）：
- 支持 `Atlas 800T A3 server` 与 `Atlas 900 A3 SuperPod cluster`。
- 必须在特权容器中使用。
- 软件：

| Software     |             Matched Version              |
| :----------: | :--------------------------------------: |
| Ascend HDK   | 25.5.2 or later                          |
| UB Switch    | LingQu Computing Network 1.5.1 or later  |

- 使用约束：`the QoS configurations for AIV_H2D and AIV_D2D do not take effect currently`（受底层 driver 限制，未来随 driver 释放通过模块升级交付）。

**如何开启（构建 AI QoS 模块）**：在使用 `tools/ai_qos.py` 前先构建并安装 AI QoS 扩展。DSMI 头文件 `dsmi_common_interface.h` 与库文件 `libdrvdsmi_host.so` 路径与环境相关，先在本机定位再替换 `YOUR_DSMI_INCLUDE_DIR`（如 `/usr/local/Ascend/driver/include`）与 `YOUR_DSMI_LIBRARY_FILE`（如 `/usr/local/Ascend/driver/lib64/driver/libdrvdsmi_host.so`）。容器部署时创建容器需把 DSMI 头/库目录挂入容器文件系统，否则 CMake 找不到。在 vLLM-Ascend 仓库根目录执行：

```bash
cmake -S tools/ai_qos -B tools/ai_qos/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${PWD}/vllm_ascend \
  -DDSMI_INCLUDE_DIR=YOUR_DSMI_INCLUDE_DIR \
  -DDSMI_LIBRARY=YOUR_DSMI_LIBRARY_FILE
cmake --build tools/ai_qos/build -j
cmake --install tools/ai_qos/build
```

**使用方式**：AI QoS 支持 Auto 与 Manual 两种模式，进入 vLLM-Ascend 安装目录、在运行推理任务前执行。

1) Auto mode：

`python tools/ai_qos.py`

自动分类不同类型流量优先级并生成 QoS 标签，同时打印 UB switch 配置，复制输出登录 UB switch 配置（会覆盖现有 QoS 配置，已有配置请先备份）。

2) Manual mode：

```bash
python tools/ai_qos.py --mode manual --AIV_D2D {priority} --AIV_H2D {priority} --SDMA_D2D {priority} --SDMA_H2D {priority} --PCIEDMA_H2D {priority}
```

manual 模式可只指定一种流量类型的优先级。参数：

| Name              | Type | Default                                                      | Description                                                  |
| ----------------- | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| mode              | str  | auto                                                         | The mode of AI QoS, default mode is "auto", another mode is "manual", some parameters need to be configured if you choose "manual" mode. |
| AIV_D2D、AIV_H2D、SDMA_D2D、SDMA_H2D、PCIEDMA_H2D | str    | AIV_D2D: high, AIV_H2D: high, SDMA_D2D: high, SDMA_H2D: low, PCIEDMA_H2D: high | Parameters for "manual" mode, determined the QoS priority of different types of traffic. The default configuration is the same as "auto" mode. Typical traffic types: AIV_D2D: AIV-based Device-to-Device communication, such as dispatch and combine. AIV_H2D: AIV-based Operator Delivery. SDMA_D2D: SDMA-based Device-to-Device communication, such as Allreduce and Allgather. SDMA_H2D: SDMA-based Host-to-Device/Device-to-Host communication, such as KVCache offloading and prefetching. PCIEDMA_H2D: PCIe DMA-based Operator Delivery. You can change the priority with "high/middle/low" options available. Due to hardware restrictions, "PCIEDMA_H2D" only supports "high/low" priority. |

**关闭 AI QoS**：

```bash
python tools/ai_qos.py unset
```

会打印在 UB Switch 上禁用 AI QoS 的命令，登录 UB Switch 执行即可。

---

## Ray (multi-node)

**是什么**：多机分布式推理用于单机放不下模型的场景，可通过 tensor parallelism 或 pipeline parallelism 分布到多节点。部署需完成三步：验证多机通信环境、搭建并启动 Ray 集群、在多机上启动在线推理服务。本教程以 Qwen3-235B-A22B 为例。

**何时使用**：模型单机装不下、需要跨机 NPU 资源时。

**硬件/物理要求**（逐字）：
- 物理机在同一 LAN、网络互通。
- 所有 NPU 用光模块连接，连接状态正常。

### 验证多机通信环境

在每节点依次执行，结果须均为 `success`、状态 `UP`：

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
 # View NPU network configuration
 cat /etc/hccn.conf
```

NPU 互联验证：

1. 获取 NPU IP：

```bash
for i in {0..7}; do hccn_tool -i $i -ip -g | grep ipaddr; done
```

2. 跨节点 PING：

```bash
# Execute on the target node (replace with actual IP)
hccn_tool -i 0 -ping -g address 10.20.0.20
```

### 搭建并启动 Ray 集群

基础容器：为保持各节点环境一致（模型路径、Python 环境），推荐用 Docker 镜像、容器化部署。主从节点均启动容器，使用 `--net=host`。挂载 `/root/.cache` 必须是各节点共享目录。**在所有节点**执行：

```bash
# Update the vllm-ascend image
export IMAGE=quay.nju.edu.cn/ascend/vllm-ascend:{{ vllm_ascend_version }}
export NAME=vllm-ascend

# Run the container using the defined variables
# Note if you are running bridge network with docker, please expose available ports for multiple nodes communication in advance.
# IMPORTANT: The cache directory mounted at /root/.cache must be a shared directory accessible by all nodes.
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
-v /path/to/shared/cache:/root/.cache \
-it $IMAGE bash
```

启动 Ray 集群：选一台作主节点，其余作从节点。先用 `ip addr` 查 `nic_name`。设置 `ASCEND_RT_VISIBLE_DEVICES` 指定 NPU；Ray 2.1 以上还需设 `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES` 以避免设备识别问题。**环境变量须在启动 Ray 集群前设置**，修改后需重启 Ray 集群。

Primary node（Head）：

```shell
# Head node
export HCCL_IF_IP={local_ip}
export GLOO_SOCKET_IFNAME={nic_name}
export TP_SOCKET_IFNAME={nic_name}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ray start --head
```

Secondary node（Worker）：

```shell
# Worker node
export HCCL_IF_IP={local_ip}
export GLOO_SOCKET_IFNAME={nic_name}
export TP_SOCKET_IFNAME={nic_name}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ray start --address='{head_node_ip}:6379' --node-ip-address={local_ip}
```

启动后用 `ray status` 与 `ray list nodes` 验证，应看到正确数量的节点与 NPU。Dashboard 默认 http://localhost:8265。

### 多机启动在线推理服务

容器内可像单机一样使用 vLLM，vLLM 会利用 Ray 集群所有节点的 NPU。**只需在一个节点运行 vllm 命令**。常规做法：`tensor-parallel-size` = 每节点 NPU 数，`pipeline-parallel-size` = 节点数。例如 2 节点 16 NPU（每节点 8）：

```shell
vllm serve Qwen/Qwen3-235B-A22B \
  --distributed-executor-backend ray \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --seed 1024 \
  --max-model-len 8192  \
  --max-num-seqs 25 \
  --served-model-name qwen \
  --trust-remote-code \
  --gpu-memory-utilization 0.9
```

若只用 tensor parallelism，`tensor-parallel-size` = 集群总 NPU 数（2 节点 16 NPU 即 16）：

```shell
vllm serve Qwen/Qwen3-235B-A22B \
  --distributed-executor-backend ray \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --seed 1024 \
  --max-model-len 8192  \
  --max-num-seqs 25 \
  --served-model-name qwen \
  --trust-remote-code \
  --gpu-memory-utilization 0.9
```

查询：

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen",
        "prompt": "tell me how to sleep well",
        "max_completion_tokens": 100,
        "temperature": 0
    }'
```

---

## Suffix Speculative Decoding

**是什么**：Suffix Decoding 是基于模式匹配的投机解码优化技术，同时从 prompt 和已生成内容中检索重复序列，用频次统计预测最可能的 token 续接。与传统投机解码不同，它完全运行在 CPU 上，无需额外 GPU 资源或 draft model，对 AI agent、代码生成等重复性任务加速显著。本教程在 Atlas A2 硬件上、单台 Atlas 800T A2 节点 4 卡部署 Qwen3-32B 实例，使用 AISBench 在 HumanEval、ARC、gsm8k、SuperGLUE_BoolQ、AGIEval、ShareGPT 等数据集上基准测试。验证表明 Qwen3-32B 在多类真实数据集上开启 Suffix Decoding 后吞吐提升约 20%~80%。

**何时使用**：重复性输出任务（agent、代码生成、多轮对话、长上下文重复 pattern）；希望在 SLO TPOT < 50ms 下提升吞吐时。

**硬件/版本要求**（逐字）：
- Atlas A2 硬件，单台 Atlas 800T A2 节点 4 卡。
- 镜像版本 `v0.13.0rc1`（官方镜像）。

### 拉取镜像

```bash

docker pull quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
```

### Docker 启动容器

```bash

# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
export NAME=vllm-ascend

# Run the container using the defined variables
# This test uses four Atlas A2 NPU cards to create the container.
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

### 安装 arctic-inference

```bash
pip install arctic-inference
```

### vLLM 实例部署

投机解码通过 `--speculative-config` 开启，`method` 设为 `suffix`，本测试 `num_speculative_tokens` 统一为 `3`。

```bash
# set the NPU device number:
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
# Set the operator dispatch pipeline level to 1 and disable manual memory control in ACLGraph
export TASK_QUEUE_ENABLE=1
# Enable the AIVector core to directly schedule ROCE communication.
export HCCL_OP_EXPANSION_MODE="AIV"
# Enable FlashComm_v1 optimization when tensor parallel is enabled.
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /data/Qwen3-32B \
  --served-model-name qwen3 \
  --trust-remote-code \
  --distributed-executor-backend mp \
  --tensor-parallel-size 4 \
  --max-model-len 5500 \
  --max-num-batched-tokens 40960 \
  --speculative-config '{"method": "suffix", "num_speculative_tokens": 3}' \
  --gpu-memory-utilization 0.9 \
  --additional-config '{"pa_shape_list":[48,64,72,80], "weight_prefetch_config":{"enable":true}}' \
  --port 8011
```

### AISbench 基准测试

**Model Configuration**（`ignore_eos` 须为 `False`，`max_out_len` 设大值让模型自然输出完整）：

```bash
# "ignore_eos" must be set to "False", and "max_out_len" should be set to a large value to allow the model to output completely and naturally.

from ais_bench.benchmark.models import VLLMCustomAPIChatStream

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="<path_to_your_model>/Qwen3-32B",
        model="qwen3",
        request_rate = 0,
        retry = 2,
        host_ip = "<your_server_ip>",
        host_port = 8011,
        max_out_len = 4000,
        batch_size= 16,
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 0,
            ignore_eos = False
        )
    )
]
```

**Performance Benchmarking Commands**：

```bash
# Example command to test gsm8k dataset performance using the first 100 prompts. Commands for other datasets are similar.
ais_bench --models vllm-api-stream-chat \
  --datasets gsm8k_gen_0_shot_cot_str_perf \
  --debug --summarizer default_perf --mode perf --num-prompts 100
```

### 测试结果汇总

| **Dataset Category** | **Typical Representative** | **Throughput Improvement (BS=1-10)** | **SLO TPOT** |
| -------------------- | -------------------------- | ------------------------------------ | ------------ |
| **High Gain**        | AGIEval, GSM8K             | **> 50%**                            | < 50ms       |
| **Medium-Low Gain**  | ARC, ShareGPT              | **20% ~ 30%**                        | < 50ms       |

原始详细测试结果：

| Concurrency         | Avg Input | Avg Output | Requests | Base TPOT(ms) | Base Throughput(TPS) | Suffix TPOT(ms) | Suffix Throughput(TPS) | Accept Rate | TPOT Gain | TPS Gain |
| ------------------- | --------- | ---------- | -------- | ------------- | -------------------- | --------------- | ---------------------- | ----------- | --------- | -------- |
| **HumanEval**       |           |            |          |               |                      |                 |                        |             |           |          |
| 1                   | 150       | 2700       | 100      | 55.1          | 18.1                 | 37.9            | 26.3                   | 27.0%       | 45.2%     | 45.1%    |
| 15                  | 150       | 2700       | 100      | 61.6          | 233.8                | 45.8            | 318.2                  | 27.0%       | 34.6%     | 36.1%    |
| 26                  | 150       | 2700       | 100      | 64.7          | 403.8                | 50.9            | 519.2                  | 27.0%       | 27.2%     | 28.6%    |
| **ARC**             |           |            |          |               |                      |                 |                        |             |           |          |
| 1                   | 76        | 960        | 100      | 52.8          | 18.9                 | 39.5            | 25.4                   | 23.9%       | 33.7%     | 34.6%    |
| 8                   | 76        | 960        | 100      | 59.1          | 125.4                | 47.0            | 163.1                  | 23.9%       | 25.7%     | 30.0%    |
| 15                  | 76        | 960        | 100      | 59.8          | 245.8                | 48.9            | 311.7                  | 23.9%       | 22.3%     | 26.8%    |
| **GSM8K**           |           |            |          |               |                      |                 |                        |             |           |          |
| 1                   | 67        | 1570       | 100      | 55.5          | 18.0                 | 35.7            | 28.5                   | 31.1%       | 55.6%     | 58.4%    |
| 17                  | 67        | 1570       | 100      | 61.5          | 279.8                | 45.4            | 403.0                  | 31.1%       | 35.6%     | 44.0%    |
| 26                  | 67        | 1570       | 100      | 63.9          | 396.4                | 50.0            | 527.6                  | 31.1%       | 27.8%     | 33.1%    |
| **ShareGPT**        |           |            |          |               |                      |                 |                        |             |           |          |
| 1                   | 666       | 231        | 327      | 54.1          | 18.3                 | 39.2            | 24.1                   | 23.9%       | 37.9%     | 31.5%    |
| 8                   | 666       | 231        | 327      | 58.8          | 125.0                | 46.2            | 153.2                  | 23.9%       | 27.1%     | 22.5%    |
| 14                  | 666       | 231        | 327      | 61.8          | 227.0                | 49.9            | 273.9                  | 23.9%       | 23.8%     | 20.7%    |
| **SuperGLUE_BoolQ** |           |            |          |               |                      |                 |                        |             |           |          |
| 1                   | 207       | 314        | 100      | 54.1          | 18.4                 | 36.1            | 26.8                   | 33.4%       | 49.8%     | 45.6%    |
| 16                  | 207       | 314        | 100      | 60.0          | 229.7                | 43.5            | 303.9                  | 33.4%       | 38.0%     | 32.3%    |
| 32                  | 207       | 314        | 100      | 62.7          | 396.4                | 47.8            | 507.5                  | 33.4%       | 31.3%     | 28.0%    |
| **AGIEval**         |           |            |          |               |                      |                 |                        |             |           |          |
| 1                   | 735       | 1880       | 100      | 53.1          | 18.7                 | 31.8            | 34.1                   | 50.3%       | 66.8%     | 81.9%    |
| 24                  | 735       | 1880       | 100      | 64.0          | 381.2                | 43.3            | 629.0                  | 50.3%       | 47.8%     | 65.0%    |
| 34                  | 735       | 1880       | 100      | 70.0          | 494.6                | 50.2            | 768.4                  | 50.3%       | 39.4%     | 55.3%    |

---

## 源文件清单

| # | 路径 | 状态 |
|---|------|------|
| 1 | `docs/source/user_guide/feature_guide/batch_invariance.md` | OK |
| 2 | `docs/source/user_guide/feature_guide/batch_job_aware_scheduler.md` | OK |
| 3 | `docs/source/user_guide/feature_guide/short_request_first.md` | OK |
| 4 | `docs/source/user_guide/feature_guide/Ai_QoS_introduction_en.md` | OK |
| 5 | `docs/source/tutorials/features/ray.md` | OK |
| 6 | `docs/source/tutorials/features/suffix_speculative_decoding.md` | OK |
