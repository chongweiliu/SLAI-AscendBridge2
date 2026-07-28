> 来源：vllm-ascend docs（main 分支，抓取于 2026-07-28），整合自多个源文件

# vllm-ascend 设计文档整合笔记

本文件整合 vllm-ascend 开发者指南 `Design_Documents/` 下 7 篇设计文档，覆盖 ACL Graph、Patch 机制、KV Cache Pool、ModelRunner 输入准备、自定义 aclnn 算子、Balance Schedule 重构、npugraph_ex。每节给出机制概览、关键 API/hook、扩展开发者接入点、版本/硬件要求。代码、API 名、类/函数名、环境变量名与值、config 片段、版本/硬件要求均保留英文原文，叙述性解释用中文。

---

## ACL Graph

ACL Graph 是 vLLM 静态图执行在 Ascend 上的实现。vLLM 上游负责通用的图模式（`CUDAGraphMode`、运行时分发、batch descriptor、bucketing/padding、full/piecewise graph 定义），vllm-ascend 在此之上提供平台 wrapper、capture-size 裁剪、以及为 ACL graph replay 保持 attention 参数正确所需的更新逻辑。

设计目标与上游一致：降低中小 batch 的 host launch 开销。实现边界划分：vLLM 提供通用分发路径，`vllm-ascend` 提供平台 wrapper、capture-size trimming、attention 特定的更新逻辑。

### 接入点

`NPUPlatform.get_static_graph_wrapper_cls()` 返回 `vllm_ascend.compilation.acl_graph.ACLGraphWrapper`。`ACLGraphWrapper` 负责：

- 从 forward context 读取 runtime mode 与 `batch_descriptor`；
- 决定是 eager、capture 新 ACL graph，还是 replay 缓存的 ACL graph；
- 按 batch descriptor 缓存 graph entry；
- 维护 graph pool 与 Ascend backend 所需的 replay bookkeeping。

wrapper 本身不定义上游分发策略，假定 vLLM 已选好 runtime mode 与 batch descriptor，仅在其上施加 Ascend capture 或 replay。

### Capture Sizes and Bucketing

vLLM replay 要求 runtime shape 稳定，因此只准备有限 capture size 集合，将 runtime batch 派发到最近的 supported size；若 runtime batch 大于最大 capture size，则跳过 graph mode、回退 eager。

默认 capture sizes 构造规则：

- `1`, `2`, `4`
- 8 的倍数，从 `8` 到 `255`
- 16 的倍数，从 `256` 到 `max_cudagraph_capture_size`

```text
[1, 2, 4, 8, 16, 24, 32, ..., 248, 256, 272, 288, ...]
```

小 batch 用更小步长以降低延迟敏感区间的 padding 开销，大 batch 用更大步长以控制 captured graph 数量。Ascend 在此基础上可能进一步缩减：sequence-parallel 过滤、runtime 资源限制、某些 runtime mode 在 capture 前归一化。

### Ascend 特定约束

- **Capture breadth 受 runtime 资源约束**：与 CUDA Graph 不同，ACL graph capture 仍可能在选定 size 耗尽 backend runtime 资源时失败。Piecewise 模式最敏感（捕获多个 subgraph，总成本随 model depth 与 size coverage 扩展）。旧版有 `update_aclgraph_sizes()` heuristic 缩减 PIECEWISE capture-size 集，已移除；当前实现保留上游 sizing/dispatch 行为，在 `vllm_ascend/compilation/acl_graph.py` 拦截确认的 capture-time stream-resource signature 并以更清晰的缓解指引 re-raise。因此 `cudagraph_capture_sizes` 与 `max_cudagraph_capture_size` 是 capture 失败时主要调优杠杆。更新的 HDK/CANN 组合能显著提升 ACL graph 容量，通信密集配置可能仍需更小 size 集。
- **Platform mode 归一化更严**：`vllm_ascend.platform.NPUPlatform.check_and_update_config()` 收窄部分上游模式——encoder-decoder 模型强制 `PIECEWISE`；`use_inductor` 在 ACL graph 路径下禁用；`ASCEND_LAUNCH_BLOCKING=1` 与 ACL graph 启用互斥；Xlite graph mode 可能禁用 ACL graph full mode 或回退 `FULL_DECODE_ONLY`。

### Full graph replay 的 host-side attention 参数更新

Full graph replay 在 Ascend 上有上游文档未详述的额外问题：即使整体 graph 静态，某些 attention operator 仍需 runtime metadata 更新。实现把 graph capture 与 host-side task parameter update 分离：

1. capture 期间，attention backend 记录 per-graph task handle、event、workspace、以及待刷新 tensor/metadata 的 weak reference；
2. replay 前，`update_full_graph_params()` 调用 backend 特定 `update_graph_params()` 实现；
3. backend 在 update stream 上，用 `torch.npu.graph_task_update_begin(...)` 与 `torch.npu.graph_task_update_end(...)` 包住 attention operator launch 执行参数刷新；
4. 用 `torch.npu.ExternalEvent` 对象在 host-side update stream 与 replay stream 间强制顺序。

实现在以下 attention backend 中：

- `vllm_ascend/attention/attention_v1.py`
- `vllm_ascend/attention/mla_v1.py`
- `vllm_ascend/attention/context_parallel/attention_cp.py`
- `vllm_ascend/attention/context_parallel/mla_cp.py`

设计要点：Ascend full graph 支持**依赖 backend 提供的 `update_graph_params()` hook**，没有该 hook 时仅靠 capture 无法 replay 正确的 attention 状态。

### Replay ordering and synchronization

`ACLGraphWrapper` 在 common path 中 replay 前同步当前 stream，确保 host-side 参数更新与消费它的 graph 执行对齐（异步调度或多线程时尤其重要）。若顺序被破坏：iteration *i* 的参数更新可能被 iteration *i-1* 的 replay 观察到，或 iteration *i* 的 replay 在其参数更新完成前启动，导致 attention 用错配的 runtime metadata，引发结果错误、精度问题甚至挂死。代码为 main full-graph eagle case 保留更窄路径，但通用假设一致：replay 不得超前 pending 参数更新。

### Full vs Piecewise on Ascend

- **Piecewise mode**：保守路径，依赖 vLLM split execution 策略，对 compilation path 选出的非 attention 段施加 ACL graph capture。Ascend 上当前支持更广，但最易受 stream pressure 影响（captured graph 数随 model depth 增长）。
- **Full graph mode**：当 attention backend 能通过 `update_graph_params()` 支持 runtime parameter patching 时，性能更优。Ascend full graph 支持绑定到这些 attention 特定 update hook、workspace caching、与 replay ordering 保证。

### Diagnostics

- 确认 graph mode 启用：开启 cudagraph metrics 并保持 log stats。CLI 用 `--cudagraph-metrics` 且不传 `--disable-log-stats`；Python 用 `cudagraph_metrics=True`、`disable_log_stats=False`，查看输出 metrics/log。
- debug 模式下 `ACLGraphWrapper` assert replay 使用 capture 时记录的同一 tensor 地址。
- `ASCEND_LAUNCH_BLOCKING=1` 与 ACL graph 启用不兼容。
- `vllm_ascend.utils` 提供 graph-aware print helpers（developer diagnostic，非执行设计组成部分）。

### Related Files

- `vllm_ascend/platform.py` — mode 归一化、platform hooks、static graph wrapper 选择
- `vllm_ascend/compilation/acl_graph.py` — ACL graph wrapper、capture/replay cache、graph parameter container、full graph update dispatch
- `vllm_ascend/attention/attention_v1.py` — full graph attention 参数 capture/update
- `vllm_ascend/attention/mla_v1.py` — MLA full graph 参数 capture/update
- `vllm_ascend/attention/context_parallel/attention_cp.py`、`mla_cp.py` — context parallel 更新路径

---

## Patch (operator patching)

vllm-ascend 是 vLLM 的 platform plugin。由于 vLLM 与 vllm-ascend 发布周期不同且硬件有限制，需要 patch vLLM 部分代码以兼容。`vllm_ascend/patch` 模块承载这些 patch。

### Principle

Patch 不是最佳方案，仅是临时解决；最佳方式是把改动贡献回 vLLM 上游使其原生兼容。基本原则：

1. Less is more，除非唯一方式否则不 patch；
2. 一旦加入 patch，必须描述移除该 patch 的 future plan；
3. 任何时候欢迎清理 patch 代码。

### How it works

```shell
vllm_ascend/
└── patch/
    ├── platform/
    │   └── patch_xxx.py
    └── worker/
        └── patch_yyy.py
```

- **platform/**：patch vLLM main process 代码。由 `vllm_ascend/platform::NPUPlatform::pre_register_and_update` 在 vLLM 初始化极早期调用。
    - online 模式：在 `vllm/vllm/engine/arg_utils.py::AsyncEngineArgs.add_cli_args` 解析 CLI args 时调用 platform patch。
    - offline 模式：在 `vllm/vllm/engine/arg_utils.py::EngineArgs.create_engine_config` 解析输入参数时调用。
- **worker/**：patch vLLM worker process 代码。由 `vllm_ascend/worker/worker::NPUWorker::__init__` 在 worker 进程初始化时调用。
    - online/offline 均在 `vllm/vllm/worker/worker_base.py::WorkerWrapperBase.init_worker` 初始化 worker 时调用。

### How to write a patch

以 patch vLLM `distributed` 模块为例：

1. 决定要 patch 哪些 vLLM 版本（如同时 patch `0.10.0` 与 `main`）。
2. 决定 patch 哪个进程（`distributed` 属 main process → patch `platform`）。
3. 在正确目录创建 patch 文件，命名为 `patch_{module_name}.py`，如 `vllm_ascend/patch/platform/patch_distributed.py`。
4. 写入 patch 代码：

    ```python
    import vllm

    def patch_destroy_model_parallel():
        # your patch code
        ...

    vllm.distributed.parallel_state.destroy_model_parallel = patch_destroy_model_parallel
    ```

5. 在 `__init__.py` 导入 patch 文件，如 `import vllm_ascend.patch.platform.patch_distributed`。
6. 在 `vllm_ascend/patch/__init__.py` 加描述：

    ```python
    # ** File: <The patch file name> **
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   1. `<The target patch module in vLLM>`
    #    Why:
    #       <Describe the reason why we need to patch>
    #    How:
    #       <Describe the way to patch>
    #    Related PR (if no, explain why):
    #       <Add a link to the related PR in vLLM. If there is no related PR, explain why>
    #    Future Plan:
    #       <Describe the future plan to remove the patch>
    ```

7. 加 Unit Test 与 E2E Test（见 testing guide）。

### Limitations

1. V1 Engine 启动三类进程：Main process、EngineCore process、Worker process。vllm-ascend 默认只能 patch Main 与 Worker 进程。要 patch EngineCore 进程，需在 setup 时整进程 patch（找到 `vllm.v1.engine.core` 全部代码，整体 override `EngineCoreProc` 与 `DPEngineCoreProc`）。
2. 运行被编辑过的 vLLM 代码时，vLLM 版本会自动变化（如基于 v0.9.n 改的代码版本可能变成 v0.9.nxxx），vllm-ascend 无法区分版本，对应版本 patch 失效。可设环境变量 `VLLM_VERSION` 显式指定版本，使对应版本（如 v0.9.n）patch 生效。

---

## KV Cache Pool Guide

### Why

Prefix caching 能大幅降低 prefill 计算时间，但其收益高度依赖 cache hit rate；仅用片上 memory 存 KV cache 会限制命中率。KV Cache Pool 利用片上 memory、DRAM、SSD 等多级存储构成 KV Cache 池，使请求 prefix 跨所有 node 可见，提高全局命中率。

vllm-ascend 当前支持 [MooncakeStore](https://github.com/kvcache-ai/Mooncake) 作为 KV Cache 存储引擎。虽然可经 LMCache remote backend 在 vLLM V1 engine 上用 MooncakeStore（GPU 路径），但 vllm-ascend 集成了直接支持 MooncakeStore 且适配华为 NPU 数据传输策略的 connector——**MooncakeStoreConnectorV1**，设计上大量借鉴 **LMCacheConnectorV1**。

### Usage

配置 `kv-transfer-config` 并选 `MooncakeStoreConnector` 作为 KV Connector 即可启用。详细部署见 [KV Pool User Guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/kv_pool.html)。

### How it works

KV Cache Pool 通过 connector-based 架构整合多级内存（片上/DRAM/SSD）。每个 connector 实现统一接口，按访问频率与硬件带宽在各 tier 间 store/retrieve/transfer KV block。与 vLLM Prefix Caching 结合后，本地（片上 memory）与全局（Mooncake）皆可高效缓存，热 prefix 保持 hot，冷数据溢出到低成本 memory。

#### 1. 与片上 memory Prefix Caching 结合

vLLM V1 默认启用片上 memory Prefix Caching（除非传 `--no-enable-prefix-caching`）。引入 KV Connector V1 后可无缝结合 Mooncake-backed KV Pool。Workflow：

1. engine 先查片上 memory cache 的 prefix hit；
2. 得到片上 hit token 数后，经 connector 查 KV Pool。若 KV Pool 有额外 hit，**只取 additional blocks**（从 KV Pool），其余 block 直接取自片上 memory，以最小化数据传输延迟；
3. KV Pool 中的 KV cache 载入片上 memory 后，剩余流程同片上 Prefix Caching。

#### 2. 与 Mooncake PD Disaggregation 结合

与 Mooncake PD（Prefill-Decode）Disaggregation 配合可跨 device/node 解耦 prefill 与 decode。当前仅对 **Prefill Nodes** 做 KV Pool 的 put/get，Decode Nodes 从 Mooncake P2P KV Connector（即 MooncakeConnector）取 KV Cache。收益：Prefill Node 同时享受片上 + KV Pool 的 Prefix Caching 减计算，Prefill↔Decode 间用 P2P KV Connector 直接 NPU device 间传输，不牺牲传输效率。启用需用 vLLM 的 Multi Connector 以特定顺序组合 Mooncake Connector 与 MooncakeStore Connector。见 [Mooncake connector deployment guide](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/mooncake_connector_deployment_guide.md)。

### MooncakeStoreConnectorV1 实现

继承 vLLM V1 的 KV Connector V1 基类，实现其必需方法即可集成第三方 KV cache 传输/存储后端。借鉴 LMCacheConnectorV1 的 `Lookup Engine`/`Lookup Client` 设计与 `ChunkedTokenDatabase`（处理 token 成 prefix-aware hash 等 hash 设计），并新增 `KVTransferThread`（多线程异步 `get`/`put`）与 NPU 数据传输优化（如移除 LMCache 的 `LocalBuffer` 以消除冗余传输）。需实现的方法分两类：

**Scheduler-Side（V1 scheduler 调用）：**

- `get_num_new_matched_tokens`：查 KV pool 返回 prefix cache hit token 数。
- `update_states_after_alloc`：临时 buffer alloc 后更新 KVConnector state。
- `build_connector_meta`：把 connector metadata attach 到 request。
- `request_finished`：request 完成时决定 block 立即释放还是异步发送后释放。

**Worker-Side（V1 worker 调用）：**

- `register_kv_caches`：注册 KV cache transfer 所需 buffer。
- `start_load_kv`：把 KV cache 从 storage 传到 device。
- `wait_for_layer_load`（可选）：layerwise + async KV load 场景等待 layer load。
- `save_kv_layer`（可选）：layerwise 把 KV cache put 进 KV Pool。
- `wait_for_save`：异步 save/put 完成时等待。
- `get_finished`：取完成 KV transfer 的 request，`put` 完成返回 `done_sending`，`get` 完成返回 `done_receiving`。

### DFX

1. KV Pool 查 key 未命中时，该 block 无 hit，且不再继续查该 request 后续 block。
2. 向 KV Pool put block 失败时不再 put 后续 block（subject to change）。

### Limitations

1. 当前 MooncakeStore for vLLM-Ascend 只支持 DRAM 作为 KV Cache pool 存储。
2. 若查到 key 存在但 `get` 失败，当前仅打 log 并继续，可能影响该 request 精度。未来计划回退该 request 按 no prefix cache hit 重算（更优方案：只回退一个 block 并保留之前的 Prefix Cache）。

---

## ModelRunner prepare_inputs

模型前向需要两类信息：inputs 与对应 attention metadata。本文档解释 vLLM 如何准备这两者。

### 输入概览

**1. Obtain inputs**

1. Get `token positions`：每个 token 在其 request 序列内的相对位置。
2. Get `token indices`：每个 scheduled token 在 token table 中的 index。
3. Get `Token IDs`：用 token indices 从 **token id table** 取出 Token IDs。

最终 Token IDs 喂入模型，`positions` 用于构造 `RoPE`，二者皆为模型输入。Token IDs 也称 `Input IDs`。

**2. Build attention metadata**：`query start location`、`sequence length`、`number of computed tokens`、`number of requests`、`number of tokens`、`block table`、`max query len`、`slot mapping`、`attention mask`。

### 变量层级

- token level：每个 scheduled token 对应一个属性，长度 = scheduled token 数。
- request level：每个 scheduled request 对应一个属性，长度通常 = scheduled request 数（`query start location` 特殊，多一个元素）。
- system level：
  1. **Token IDs table**：存每个 request 的 Token IDs，shape `(max num request, max model len)`。`max num request` 为 forward batch 最大并发请求数，`max model len` 为单 request 序列最大 token 数。
  2. **Block table**：把 block 的逻辑地址（序列内）映射到 device memory 全局物理地址，shape `(max num request, max model len / block size)`。

两表均来自 `prepare inputs` 之前的 `_update_states`。

### 计算示例（详细 walkthrough）

假设：最大一次可调度 token 数 = 10；`block size` = 2；3 个 request，prompt 长度 3/2/8；`max model length` = 12。

#### Step 1: 全 prefill

Scheduled tokens `{'0': 3, '1': 2, '2': 5}`（`request_2` 用 chunked prefill，剩 3 个未调度）。

- `request indices`: `[0, 0, 0, 1, 1, 2, 2, 2, 2, 2]`
- `token positions`: 每个 request 内 `已计算 token 数 + 当前调度 token 相对位置`，拼接得 `[0, 1, 2, 0, 1, 0, 1, 2, 3, 4]`
- 设 `M = max model len`，`token indices = request indices * M + token positions` = `[0, 1, 2, 12, 13, 24, 25, 26, 27, 28]`
- `input_ids = token_table[token_indices]` = `[T_0_0, T_0_1, T_0_2, T_1_0, T_1_1, T_2_0, T_2_1, T_3_2, T_3_3, T_3_4]`

attention metadata（`K = max model len / block size = 6`，block_0 标记未使用）：

1. `block table indices = request indices * K + positions / block size` = `[0, 0, 1, 6, 6, 12, 12, 13, 13, 14]`
2. `device block number = block_table[block_table_indices]` = `[1, 1, 2, 3, 3, 4, 4, 5, 5, 6]`
3. `block offsets = positions % block size` = `[0, 1, 0, 0, 1, 0, 1, 0, 1, 0]`
4. `slot mapping = device block number * block size + block_offsets` = `[2, 3, 4, 6, 7, 8, 9, 10, 11, 12]`

request level：`query start location`（prefix sum）= `[0, 3, 5, 10]`；`sequence length` = `[3, 2, 5]`；`number of computed tokens` = `[0, 0, 0]`；`number of requests` = `3`；`number of tokens` = `[3, 2, 5]`；`max query len` = `5`；`slot mapping` = `[2, 3, 4, 6, 7, 8, 9, 10, 11, 12]`；`attention mask` shape `5 * 5`（所有 prefill 请求共用一个 mask）。

#### Step 2: Chunked prefill

Scheduled `{'0': 1, '1': 1, '2': 3}`，结果直接给出：

- `request indices`: `[0, 1, 2, 2, 2]`
- `token positions`: `[3, 2, 5, 6, 7]`
- `token indices`: `[3, 14, 29, 30, 31]`
- `Input IDs`: `[T_0_3, T_1_2, T_3_5, T_3_6, T_3_7]`（`T_0_3`、`T_1_2` 为模型输出采样的新 token）
- `block table indices`: `[1, 7, 14, 15, 15]`
- `device block number`: `[2, 7, 6, 8, 8]`
- `block offsets`: `[1, 0, 1, 0, 1]`
- `slot mapping`: `[5, 14, 13, 16, 17]`
- `query start location`: `[0, 1, 2, 5]`
- `sequence length`: `[4, 3, 8]`
- `number of computed tokens`: `[3, 2, 5]`
- `number of requests`: `3`
- `max query len`: `3`
- `attention mask`: `5 * 8`（每 token 一个 `1 * 8` 向量，共 5 个 scheduled token）

---

## add custom aclnn op

自定义 aclnn operation 在 vllm-ascend 构建时编译安装到 `vllm_ascend/cann_ops_custom` 目录，绑定到 `torch.ops._C_ascend` 模块，供 Python 调用。启用自定义算子：

```python
from vllm_ascend.utils import enable_custom_op

enable_custom_op()
```

### 添加步骤

- 在 `csrc` 目录下新建 operation folder。
- 创建 `op_host` 与 `op_kernel` 目录分别放 host/kernel 源码。
- 在 `csrc/build_aclnn.sh` 为支持的 SOC 添加 build option；多个 op 用 `;` 分隔，如 `CUSTOM_OPS="op1;op2;op3"`。
- 在 `csrc/torch_binding.cpp` 把 aclnn operator 绑定到 `torch.ops._C_ascend` 模块。
- 在 `csrc/torch_binding_meta.cpp` 写 meta 实现，使 op 可被捕获进 aclgraph。

成功 build vllm-ascend 后即可在 Python 代码中调用该自定义 aclnn operation。

---

## Balance Schedule Refactor

### TL;DR

旧 `patch_balance_schedule.py` 逐字复制了上游三大单元（`Scheduler.schedule()` ~520 行、`DPEngineCoreProc.run_busy_loop()` ~40 行、`EngineCoreCore.run_engine_core()` ~55 行）只为注入约 5 行真实逻辑。本次重构在**严格保留 `balance_flag` 语义**前提下，先删除已 stale 的 `run_busy_loop()`/`run_engine_core()` 两份副本（改为 `_has_global_unfinished_reqs` 上的 engine core hook + 模块级 name swap 条件激活）；`schedule()` 副本**暂留**——上游无更细粒度 hook 可借，删除依赖向上游贡献 override seam（Phase 2B，TODO）。文件因此不会缩到几十行：`schedule()` body 仍是上游逐字副本（**verbatim 对齐 release tag `v0.24.0`**，仅 3 处 balance delta），文件约 830 行。本次真正消除的是 `run_busy_loop`/`run_engine_core` 副本的 stale-drift 风险，并修复一个 balance-enabled deadlock（gather 从 `schedule()` 内 → `_process_engine_step` 内 → 最终放到 `_has_global_unfinished_reqs` cross-rank all-reduce 之后）。

> 注：文档中任何具体 `v0.24.0` 仅是 pin 文件当前值快照，pin 推进后即过期，**不能**作为版本权威；运行时需读 `.github/vllm-release-tag.commit`。

### Balance Scheduling 做什么

大 `data-parallel-size` 且并发 ≈ `DP × max-num-seqs` 时，请求易堆积在部分 DP rank：饱和 rank 同时承担 prefill+decode 变慢，其他 rank 持续 admit 新请求，差距扩大。Balance scheduling **不**主动重平衡每 rank running 计数，而是提供 **global admission gate**：只要**任一 rank** running 计数到达 cap，**所有 rank** 停止从 WAITING 队列 admit 新请求，给饱和 rank 排空机会。语义**不是**"让落后 rank 追上 leader"（该语义被显式拒绝）。启用：`additional_config.enable_balance_scheduling = true`（env `VLLM_ASCEND_BALANCE_SCHEDULING` 已 deprecated）；仅支持 PD-mixed 模式，校验在 `vllm_ascend/platform.py` 与 `vllm_ascend/ascend_config.py`。

### 真实逻辑（仅两处）

1. **running 计数 cross-rank sync**——每 engine step 一次 `all_gather`，收集各 rank `len(self.running)`：

    ```python
    def balance_gather(self):  # dp_group is injected into self.dp_group by the engine core
        running_tensor = torch.tensor([len(self.running)], dtype=torch.int, device="cpu")
        dist.all_gather(self.balance_queue, running_tensor, group=self.dp_group)
    ```

2. **WAITING 调度循环内的 admission gate**——每 rank 持相同 gathered 向量，故检查结果一致。任一 rank 上步末 running 计数达 cap 时，所有 rank 本步停止 admit 新 WAITING 请求：

    ```python
    balance_flag = max(t.item() for t in self.balance_queue) == self.max_num_running_reqs
    if balance_flag:
        break
    ```

   **必须 bit-for-bit 保留的语义**："leader-at-cap ⇒ global freeze of admission"，**不是** "make lagging ranks catch up to the leader"。

### 已落地（Phase 1 + 2A + 3）与修正

实现时发现两条起草假设不成立：

- 上游 `Scheduler` **无** `new_step_starts()` 生命周期 hook（那是 `kv_cache_manager` 方法），无 "per-step scheduling start" 可 override seam。故 gather 不能放 `schedule()` 内（见 deadlock 教训），改放 engine core 的 `_has_global_unfinished_reqs`。
- scheduler 不能惰性获取 DP group：`dp_group` 在 `_init_data_parallel` 产生（早于 scheduler 创建），无全局 registry。故 `BalanceDPEngineCoreProc` **不删除**而是精简为只 override `_has_global_unfinished_reqs`（注入 `dp_group` + 调一次 `balance_gather`）；`run_engine_core` 副本由 patch 模块级 `DPEngineCoreProc` name 替代（上游 `run_engine_core` 在运行时按模块全局名解析该类），swap **只在 balance 启用时发生**（conditional activation）。

### Step 1：把 gather 挂到 `_has_global_unfinished_reqs` 并删除 EngineCore 副本

三约束：

1. scheduler 自己拿不到 DP group → engine core 须把 `dp_group` 交给 scheduler。
2. `balance_gather` 必须在每次 active（非 idle）wave 的每个 rank 每次迭代运行；`schedule()` 不满足（本地排空的 rank 跑 dummy batch，不进 `schedule()`）。
3. balance 不得侵入未启用它的 config（如 PD-disaggregated recompute / `AsyncRecomputeScheduler`），故 `BalanceDPEngineCoreProc` swap 必须条件化。

最终落地：

- `balance_gather` 拉入 `BalanceScheduler`（无参签名，用 `self.dp_group`），但**不从 `schedule()` 调**——由 engine core 每步触发。
- `BalanceDPEngineCoreProc` 精简为一个 override：hook `_has_global_unfinished_reqs`，先 `super()._has_global_unfinished_reqs()`（每 32 步做一次 cross-rank all-reduce），再在同次调用内注入 `dp_group` 并调一次 `balance_gather()`。上游 `run_busy_loop` 每次非 idle 迭代（含 drained rank 跑 dummy batch 不进 `schedule()` 的迭代）恰好调一次 `_has_global_unfinished_reqs`，故每 rank 每 active step 都参与 gather。`run_busy_loop` body 不再复制。
- `run_engine_core` 副本整段删除。上游 `run_engine_core`（staticmethod）体内按模块全局名解析 `DPEngineCoreProc`（`engine_core = DPEngineCoreProc(*args, **kwargs)`），故用 thin wrapper：入口（`vllm_config` 可用时）经 `_balance_scheduling_enabled` 决定把模块级 `DPEngineCoreProc` swap 为 `BalanceDPEngineCoreProc` 还是恢复上游原版，再调原 `run_engine_core`。**conditional activation**——balance 关时上游实现逐字运行，signal handling、`SignalCallback`、numa、tracer 全部上游正确。

> **Lesson A — deadlock（gather 不能在 `schedule()`）**：早期把 `balance_gather` 放 `BalanceScheduler.schedule()` 顶部，论据"两步间 `self.running` 只在 `schedule()`/`update_from_output()` 变化，snapshot 等价、仍每步一次 `all_gather`、安全"——只对了一半：gate 看到的值确实等价，但忽略了 `all_gather` 是 collective，每 rank 必须同步参与。DP MoE 下，本地排空（`has_requests()` False）的 rank 跑 `execute_dummy_batch()` **不进 `schedule()`**，跳过该 `all_gather`，而忙 rank 调用它并永远等待——collective mismatch，deadlock。`_has_global_unfinished_reqs` 真正 all-reduce 仅每 32 步、`engines_running` 其间 sticky，进一步放大窗口。

> **Lesson B — deadlock（gather 也不能在 `_process_engine_step`，必须在 `_has_global_unfinished_reqs` all-reduce 之后）**：后一迭代把 gather 移入 `_process_engine_step`（每迭代调用，在 sync 与 idle `continue` gate 之前）"修"了 Lesson A，却引入另一种 deadlock：`_has_global_unfinished_reqs` 是 busy loop中唯一重新同步 rank wave/idle 状态的点。把每步 `all_gather` 放在 sync 之前（及 idle `continue` 之前）会把 gather 与 wave 协调解耦。wave 边界——各 rank 完成时机不同、`_process_input_queue` 阻塞下一 wave/新请求、`engines_running` sticky 长达 32 步——一个 rank 到达 gather 时另一个仍阻塞在 `_process_input_queue` 或 `future.result()`。`all_gather` 随之 deadlock；卡住的 EngineCore 无法排空 worker shared-memory 广播通道，worker 的 `sample_tokens` 响应无处落地，60s 后 engine 死于 `RPC call to sample_tokens timed out`（间歇出现——"5 次 GPQA OK，第 6 次挂"——因触发依赖每次运行完成时机）。这与 expert-parallel 是否跨 DP rank 无关：失败机制是 EngineCore↔worker shm 耗尽，不是 worker 前向。结论：**`balance_gather` 必须紧随 `super()._has_global_unfinished_reqs()`（唯一每迭代 cross-rank sync）之后，使 rank 在达成 `engines_running` 共识后进入 all-gather。** 因 `_has_global_unfinished_reqs` 只在未走 idle `continue` 的迭代调用，全 idle 时各 rank 一致跳过 gather（无 rank 多做一次）——与重构前 copied `run_busy_loop` 中 gather 紧跟 all-reduce 之后一致。

### Step 2：用 minimal seam 替换 `schedule()` 副本

`balance_flag` gate 被 inline 在 `schedule()` 中段，上游无 override seam。分两阶段：

**Phase 2A（过渡，不依赖上游）**：保留 `schedule()` override，但——override 匹配共享 supported signature `def schedule(self, throttle_prefills: bool = False)`（v0.24.0 与 e5588e49 均暴露），disabled path 经 `super().schedule(throttle_prefills)` 直接委托；balance 改动收敛为 3 处带注释 delta：(1) disabled-path early return 委托 `super()`；(2) WAITING 循环内 `balance_flag` gate；(3) `if request_queue is None: break`（上游为 `assert`）。因上游无更细 hook，body 仍须复制。**逐字比对可复现**：`schedule()` 副本对齐 release tag（仅 3 处 balance delta 不同），固定 tag 使"对上游逐字比对"每次 CI 同基线。新增"对 release tag 逐字 drift 检测"测试：运行时从 `.github/vllm-release-tag.commit` 读 tag（与 CI 同源，非硬编码、非读设计文档），pin 推进时测试自动比对新 tag 并转红，提示"副本需 re-sync"。pin 推进时重应用 3 delta 即可（持续到 Phase 2B 删除副本）。

**Phase 2B（目标，随上游贡献落地）**：贡献上游最小重构，把 WAITING 循环停止条件提取为可 override 方法：

```python
# upstream vllm/v1/core/sched/scheduler.py
def _should_stop_admitting_waiting(self) -> bool:
    return len(self.running) >= self.max_num_running_reqs
```

上游暴露该 seam 后，Ascend patch 收敛为：

```python
class BalanceScheduler(Scheduler):
    def _should_stop_admitting_waiting(self) -> bool:
        if super()._should_stop_admitting_waiting():
            return True
        return self._balance_enabled and (
            max(t.item() for t in self.balance_queue) >= self.max_num_running_reqs
        )
```

（`>=` 与 `==` 等价，因无 rank `len(running)` 超 `max_num_running_reqs`；contract 用 `==` 固化语义以显式表达 "leader-at-cap ⇒ freeze" 并拒绝 "catch up to leader" 重解读。）结果：~520 行副本永久删除，文件不再随上游 `schedule()` 编辑漂移。这是 `AGENTS.md` 要求的 "long-term plan to contribute upstream"。

### Step 3：归一化 config probing

`_balance_scheduling_enabled()` 收敛为 **两 fallback（AscendConfig → additional_config）**。删除 `run_engine_core` 副本后唯一 caller 是 `BalanceScheduler.__init__`，但此刻 AscendConfig 是否已化仍无保证（旧 top-of-file TODO 缘由），故保留 `additional_config` 作启动窗口 fallback，否则返回 `False`。本轮相对旧实现的收紧：

- **删除直接环境变量读取**。旧实现 fallback 到裸 `os.getenv("VLLM_ASCEND_BALANCE_SCHEDULING")`，违反 AGENTS.md "no scattered `os.getenv`"。该函数不再自己读环境——`VLLM_ASCEND_BALANCE_SCHEDULING` 由 `AscendConfig` 集中解析（作为 `additional_config` 的 deprecated fallback），经主路径 `get_ascend_config().enable_balance_scheduling` 生效，避免多入口。

### Behavior-preservation contract

重构**必须**严格保留以下不变量，任何偏离即 bug：

1. **Leader-at-cap ⇒ global freeze**。`balance_flag = max(balance_queue) == max_num_running_reqs`，各 rank 据上步 gathered `len(running)` 计算。为真时无 rank admit 新 WAITING 请求。比较用 `==` 而非 `>=`、非 "catch up to leader"。
2. **Same inputs ⇒ same outputs**。给定相同 `self.running`/`self.waiting`/`self.skipped_waiting`/`balance_queue`/token budget，重构后 `schedule()` 产出与当前实现 identical 的 `SchedulerOutput`（相同 scheduled/preempted/resumed 集与 `num_scheduled_tokens`、connector metadata）。
3. **Gather cadence unchanged**。每 active engine step 恰一次 `all_gather`，同 DP group，payload 仍 `len(self.running)`，全 idle 时各 rank 一致跳过。仅 call site 移动。
4. **Disabled path unchanged**。`enable_balance_scheduling` 为 false 时 `_balance_run_engine_core` 把模块级 `DPEngineCoreProc` 恢复为上游原版，engine core 逐字跑上游实现；`BalanceScheduler`（`_balance_enabled=False`）把 `schedule(throttle_prefills)` 委托 `super().schedule(throttle_prefills)`，不分配 `balance_queue`，不做 collective 通信。balance 关时不触碰任何 config（含 PD-disaggregated recompute / `AsyncRecomputeScheduler`，已在 `platform.py` 与 balance 互斥，此为第二层防御）。
5. **Existing constraints still apply**。`profiling_chunk_config` mutex（`vllm_ascend/ascend_config.py`）与 PD-mixed-mode 限制（`vllm_ascend/platform.py`）仍在原位执行。

### Post-refactor 文件形态

`schedule()` body 为 release tag `v0.24.0` 逐字副本（Phase 2B 前不可删），带三处 documented delta：

```python
# vllm_ascend/patch/platform/patch_balance_schedule.py
import torch
import torch.distributed as dist
import vllm.v1.core.sched.scheduler as _sched_mod
import vllm.v1.engine.core as _engine_core_mod
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc
# ... other vllm imports ...


def _balance_scheduling_enabled(vllm_config) -> bool:
    try:
        from vllm_ascend.ascend_config import get_ascend_config
        return bool(get_ascend_config().enable_balance_scheduling)
    except Exception:
        pass
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    if "enable_balance_scheduling" in additional_config:
        return bool(additional_config["enable_balance_scheduling"])
    return False  # no longer reads the env var itself; VLLM_ASCEND_BALANCE_SCHEDULING is parsed by AscendConfig


class BalanceScheduler(Scheduler):
    def __init__(self, ...):
        super().__init__(...)
        self._balance_enabled = _balance_scheduling_enabled(vllm_config)
        self.dp_group = None  # injected by BalanceDPEngineCoreProc before the first gather
        if self._balance_enabled:
            self.balance_queue = [torch.tensor([0], ...) for _ in range(dp_size)]

    def balance_gather(self):  # uses self.dp_group; no-op when disabled / not injected
        if not self._balance_enabled or self.dp_group is None:
            return
        running_tensor = torch.tensor([len(self.running)], dtype=torch.int, device="cpu")
        dist.all_gather(self.balance_queue, running_tensor, group=self.dp_group)

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:  # shared by v0.24.0 and e5588e49
        if not self._balance_enabled:  # delta 1: disabled-path early return
            return super().schedule(throttle_prefills)
        # NOTE: balance_gather is NOT called here -- see BalanceDPEngineCoreProc.
        # ... upstream schedule() body (verbatim-aligned to the v0.24.0 tag) ...
        #   # inside the WAITING loop (deltas 2, 3):
        #   if max(t.item() for t in self.balance_queue) == self.max_num_running_reqs:  # delta 2: leader-at-cap => global freeze
        #       break
        #   request_queue = self._select_waiting_queue_for_scheduling()
        #   if request_queue is None:  # delta 3: keep if-break (upstream has assert)
        #       break
        # ...


class BalanceDPEngineCoreProc(DPEngineCoreProc):
    """Hook _has_global_unfinished_reqs: inject dp_group + one balance_gather per
    active step. Gather MUST sit immediately after super()._has_global_unfinished_reqs()
    (the only per-iteration cross-rank sync) -- NOT inside _process_engine_step
    (which runs before that sync and before the idle continue gate, and would
    deadlock at wave boundaries -> sample_tokens timeout), and NOT inside
    schedule() (drained ranks skip schedule() and would miss the all_gather)."""

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        result = super()._has_global_unfinished_reqs(local_unfinished)
        self.scheduler.dp_group = self.dp_group
        self.scheduler.balance_gather()
        return result


_OriginalDPEngineCoreProc = _engine_core_mod.DPEngineCoreProc
_OriginalRunEngineCore = EngineCoreProc.run_engine_core


def _balance_run_engine_core(*args, dp_rank=0, local_dp_rank=0, **kwargs):
    # Conditional activation: swap the module-level DPEngineCoreProc only when balance is on.
    if _balance_scheduling_enabled(kwargs.get("vllm_config")):
        _engine_core_mod.DPEngineCoreProc = BalanceDPEngineCoreProc
    else:
        _engine_core_mod.DPEngineCoreProc = _OriginalDPEngineCoreProc
    return _OriginalRunEngineCore(*args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs)


# Scheduler is constructed by module-global name when scheduler_cls is unset
# (the PD-mixed balance path); recompute / dynamic-batch / profiling schedulers
# set scheduler_cls and bypass this name, which is correct.
_sched_mod.Scheduler = BalanceScheduler
EngineCoreProc.run_engine_core = staticmethod(_balance_run_engine_core)
```

### Phased rollout

| Phase | Scope | Risk | Depends on | Status |
|-------|-------|------|------------|--------|
| 1 | Hook gather onto `_has_global_unfinished_reqs`（在 cross-rank all-reduce 之后——避开 schedule()-skip 与 _process_engine_step wave-boundary 两种 deadlock）；精简 `BalanceDPEngineCoreProc` 至该 hook；删除 `run_engine_core`/`run_busy_loop` 副本；`run_engine_core` wrapper 条件激活 `DPEngineCoreProc`；模块级 `Scheduler` swap | Low | none | Done |
| 2A | Override 匹配共享 supported signature（v0.24.0 + e5588e49 上 `schedule(self, throttle_prefills=False)`）；**body 逐字对齐 release tag**（仅 3 处 balance delta）；disabled path 直接委托 `super()`；signature equality + intent-lock + release-tag verbatim drift tests | Low | none | Done |
| 3 | config probing 收敛两 fallback（AscendConfig → additional_config）；删除直接 env-var 读取（仍由 AscendConfig 集中解析） | Low | Phase 1 | Done |
| 2B | Upstream `_should_stop_admitting_waiting` PR；删除 `schedule()` 副本 | Med | upstream review | TODO |
| Tests | Drift regression / behavior equivalence / gather cadence / disabled path / NPU performance check | Low | Phase 1 + 2A | TODO (needs NPU) |

各阶段可独立发布与回滚。Phase 1/2A/3 可同发布；2B 待上游 PR 合入。

### Test plan

1. **Signature + intent lock + verbatim drift test（Phase 2A）**：(a) `BalanceScheduler.schedule` signature 等于 installed `Scheduler.schedule` signature；(b) 3 行 balance delta 必须存在；(c) `_balance_run_engine_core` wrapper 已装且 import 时 `DPEngineCoreProc` 未 swap（延迟到 wrapper 按条件 swap）；(d) 上游 `DPEngineCoreProc._has_global_unfinished_reqs` 仍存在（gather 注入点——每非 idle 迭代必须调，否则 all_gather deadlock）；(e) 上游 `Scheduler` seam 方法（`_build_kv_connector_meta`、`_inflight_prefill_reserved_blocks`）仍在；(f) **verbatim drift detection**——先从 `.github/vllm-release-tag.commit` 读 tag，`git show <tag>:vllm/v1/core/sched/scheduler.py` 取该 tag 的 `schedule()`，strip 同 3 delta 后 AST 逐字比对 `BalanceScheduler.schedule` 源码，必须 identical；pin 推进自动转比对新 tag 并转红；tag 不可达时 skip 不 fail。
2. **Behavior-equivalence test**：用 fake DP group 与手设 `balance_queue` 驱动 `schedule()` 多状态，assert `SchedulerOutput` identical（contract item 2）。复用 `tests/ut/test_platform.py` 脚手架。
3. **Gather cadence test**：mock `torch.distributed.all_gather`（注意 `balance_gather` 内 `dist = torch.distributed`，故 mock target 是 `torch.distributed.all_gather` 而非 `vllm.distributed.all_gather`）；assert 每次 `balance_gather()` 恰一次 `all_gather`，payload `len(self.running)`、用注入的 dp_group。
4. **Disabled-path test**：flag 关时 assert 不分配 `balance_queue`、不调 `all_gather`、`schedule(throttle_prefills)` 委托 `super().schedule(throttle_prefills)`。
5. **NPU performance check**：`max(t.item() for t in self.balance_queue)` 每步触发一次 host sync（不可避免，值驱动 host-side 控制流）。profile 确认重构不引入额外 sync。

---

## npugraph_ex

npugraph_ex 是基于 FX graph 的优化，可视为 aclgraph 模式的加速方案。代码见 [torchair source code repository](https://gitcode.com/Ascend/torchair)。

> **Atlas inference products**：Atlas inference products 与 Atlas 200I Pro 不支持 `enable_npugraph_ex`。设置 `--additional-config '{"ascend_compilation_config": {"enable_npugraph_ex":false}}'`。

### Default FX Graph Optimization

**FX Graph pass**

- 对模型中间节点，把非 in-place 算子替换为 in-place 算子以减少计算中内存搬运，提升性能。
- 对模型原始输入参数，若含 in-place 算子，Dynamo 的 Functionalize 会把 in-place 替换为"非 in-place + copy"形式；npugraph_ex 会反向还原 in-place，减少内存搬运。

**FX fusion pass**

npugraph_ex 提供若干算子融合 pass，未来会增加更多。满足替换规则的算子组合可替换为对应融合算子。默认融合 pass 列表见 [pattern_fusion_pass](https://www.hiascend.com/document/detail/zh/Pytorch/latest/modthirdparty/torchairuseguide/docs/zh/npugraph_ex/basic/pattern_fusion_pass.md#功能简介)。

### Custom fusion pass

用户可在 npugraph_ex 注册自定义 graph 融合 pass 以修改 PyTorch FX graph，依赖 `register_replacement` API：

```python
register_replacement(search_fn, replace_fn, example_inputs, trace_fn=fwd_only, extra_check=_return_true, search_fn_pattern=None)
```

| Parameter Name | Input/Output | Explanation | Is necessary |
|--|--|--|--|
| search_fn | Input | 想在 FX graph 中识别的算子组合/计算逻辑（如需融合的算子组合） | Yes |
| replace_fn | Input | 找到 search_fn 对应组合后，用此函数计算逻辑替换原 subgraph 以实现融合/优化 | Yes |
| example_inputs | Input | 用于 trace search_fn 与 replace_fn 的示例输入张量，shape/dtype 应与实际场景匹配 | Yes |
| trace_fn | Input | 默认只跟踪前向计算图，适合推理优化；训练场景可提供支持 backward 跟踪的函数 | No |
| extra_check | Input | 算子融合后的额外验证函数，输入须为 `torch._inductor.pattern_matcher.Match`，用于对匹配结果做进一步自定义检查（如融合算子是否同 stream、device 类型、input shape 等） | No |
| search_fn_pattern | Input | 通常无需提供；定义遵循原生 PyTorch MultiOutputPattern 规则；传入后不再用 search_fn 匹配，而直接用此参数作为匹配规则 | No |

Usage Example（把 add + npu_rms_norm 融合为 npu_add_rms_norm）：

```python
import functools
import torch, torch_npu, npugraph_ex

from torch._inductor.pattern_matcher import Match
from torch._subclasses.fake_tensor import FakeTensorMode
from npugraph_ex.core.utils import logger

# Assume fusing the add operator and the npu_rms_norm operator into the npu_add_rms_norm operator
# Define a search_fn to find the operator combinations in the original FX graph before fusion.
def search_fn(x1, x2, gamma):
    xOut = torch.add(x1, x2)
    y, _ = torch_npu.npu_rms_norm(xOut, gamma)
    return y, xOut

# Define a replace_fn, that is, a fusion operator, used to replace operator combinations in the FX graph
def replace_fn(x1, x2, gamma):
    y, _, xOut = torch_npu.npu_add_rms_norm(
        x1, x2, gamma
    )
    return y, xOut

# extra_check can pass in additional validation logic. Here, it is used to check whether the last dimension of the first input parameter x1 is a specific value; if it is not the specific value, fusion is not allowed.
def extra_check(match: Match):
    x1 = match.kwargs.get("x1")

    if x1 is None:
        return False
    if not hasattr(x1, "meta") or "val" not in x1.meta:
        return False

    a_shape = x1.meta["val"].shape
    return a_shape[-1] == 7168

# Define some sample inputs to trace search_fn and replace_fn into an FX graph
fake_mode = FakeTensorMode()
with fake_mode:
    # sizes/values don't actually matter for initial trace
    # once we get a possible match we re-trace with the actual values and verify the match still holds
    input_tensor = functools.partial(torch.empty, (1, 1, 2), device="npu", dtype=torch.float16)
    kwargs_tensor = functools.partial(torch.empty, 2, device="npu", dtype=torch.float16)

    # Call the npugraph_ex.register_replacement API with search_fn, replace_fn, and example_inputs. If there are additional validations, you can pass them in as extra_check.
    npugraph_ex.register_replacement(
        search_fn=search_fn,
        replace_fn=replace_fn,
        example_inputs=(input_tensor(), input_tensor(), kwargs_tensor()),
        extra_check=extra_check
    )
```

npugraph_ex 默认融合 pass 也基于此 API 实现。更多示例见 vllm-ascend 与 npugraph_ex 代码仓库。

### DFX

复用 PyTorch 社区 `TORCH_COMPILE_DEBUG` 环境变量，设 `TORCH_COMPILE_DEBUG=1` 会输出全流程 FX graph。

---

## 源文件清单

| 源路径 | 状态 |
|--------|------|
| docs/source/developer_guide/Design_Documents/ACL_Graph.md | OK |
| docs/source/developer_guide/Design_Documents/patch.md | OK |
| docs/source/developer_guide/Design_Documents/KV_Cache_Pool_Guide.md | OK |
| docs/source/developer_guide/Design_Documents/ModelRunner_prepare_inputs.md | OK |
| docs/source/developer_guide/Design_Documents/add_custom_aclnn_op.md | OK |
| docs/source/developer_guide/Design_Documents/balance_schedule_refactor.md | OK |
| docs/source/developer_guide/Design_Documents/npugraph_ex.md | OK |

全部 7 篇均 HTTP 200 抓取成功，无 抓取失败。
