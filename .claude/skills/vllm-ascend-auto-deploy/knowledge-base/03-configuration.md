> 来源：vllm-ascend docs/source/user_guide/configuration/index.md + additional_config.md + env_vars.md + vllm_ascend/envs.py（main 分支，抓取于 2026-07-28）
> env_vars.md 通过 `{{ include_code('vllm_ascend/envs.py') }}` 引用源码中的 env vars 定义块

# 配置指南

vLLM Ascend 通过两种机制配置：**环境变量**与 `--additional-config`（迁移中，推荐新部署用 additional-config）。

## 1. additional-config 用法

vLLM 提供的插件控内机制，vLLM Ascend 用它做灵活配置。

**在线模式（vllm serve）**：

```bash
vllm serve Qwen/Qwen3-8B --additional-config='{"config_key":"config_value"}'
```

**离线模式**：

```python
from vllm import LLM
LLM(model="Qwen/Qwen3-8B", additional_config={"config_key":"config_value"})
```

## 2. 环境变量 → additional-config 迁移

从 PR #9064 起，10 个环境变量迁移到 `--additional-config`。迁移期两者都支持，未来环境变量将被移除，仅保留 additional-config。

| 环境变量 | Config Key | 类型转换 |
|----------|------------|----------|
| `VLLM_ASCEND_BALANCE_SCHEDULING` | `scheduler_config.enable_balance_scheduling` | `"1"` → `true`，`"0"` → `false` |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | `enable_flashcomm1` | `"1"` → `true`，`"0"` → `false` |
| `MSMONITOR_USE_DAEMON` | `msmonitor_use_daemon` | `"1"` → `true`，`"0"` → `false` |
| `VLLM_ASCEND_ENABLE_MLAPO` | `enable_mlapo` | `"1"` → `true`，`"0"` → `false` |
| `VLLM_ASCEND_ENABLE_NZ` | `weight_nz_mode` | Integer（不变，字段名变） |
| `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL` | `enable_context_parallel` | `"1"` → `true`，`"0"` → `false` |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | `enable_fused_mc2` | Integer（不变） |
| `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` | `enable_transpose_kv_cache_by_block` | `"1"` → `true`，`"0"` → `false` |

迁移示例：

```bash
# 旧（环境变量）
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
vllm serve Qwen/Qwen3-8B

# 新（additional-config）
vllm serve Qwen/Qwen3-8B --additional-config='{"enable_flashcomm1": true}'
```

## 3. 构建期环境变量（vllm_ascend/envs.py）

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `MAX_JOBS` | None | 包构建最大编译线程数；None 表示用全部 CPU 核 |
| `CMAKE_BUILD_TYPE` | Release | 构建类型：Release / Debug / RelWithDebugInfo |
| `COMPILE_CUSTOM_KERNELS` | `1` | 是否编译自定义算子。设 `0` 仅用于无 NPU 环境跑 UT |
| `CXX_COMPILER` | None | C++ 编译器路径；None 用系统默认 |
| `C_COMPILER` | None | C 编译器路径；None 用系统默认 |
| `SOC_VERSION` | None（自动 npu-smi 探测） | Ascend 芯片版本，用于包构建。无 npu-smi 时必须手设 |
| `VERBOSE` | `0` | `1` 编译时打印详细日志 |
| `ASCEND_HOME_PATH` | `/usr/local/Ascend/ascend-toolkit/latest` | CANN toolkit home 路径 |
| `HCCL_SO_PATH` | `libhccl.so` | HCCL 库路径，pyhccl communicator backend 用 |
| `VLLM_VERSION` | None | 已装 vLLM 版本；开发/可编辑安装版冲突时手动设 `X.Y.Z` |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | `0` | [DEPRECATED] TP 时启用 FlashComm 优化；用 `enable_flashcomm1` |
| `MSMONITOR_USE_DAEMON` | `0` | `1` 用 daemon 模式 msMonitor 监控性能 |
| `VLLM_ASCEND_ENABLE_MLAPO` | `1` | DeepSeek W8A8 系列启用 MLAPO 优化（默认开，更耗内存，优先减内存则关） |
| `VLLM_ASCEND_ENABLE_NZ` | `1` | 权重 cast 到 FRACTAL_NZ：0=关；1=仅 quant；2=尽量启用 |
| `DYNAMIC_EPLB` | `false` | 是否启用动态 EPLB |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | `0` | 启用 fused MC2（`dispatch_ffn_combine`）：0/未设=默认 ALLTOALL+MC2；1=可能替换。仅限 moe + W8A8 + EP<=32 + non-mtp + non-dynamic-eplb |
| `VLLM_ASCEND_BALANCE_SCHEDULING` | `0` | [DEPRECATED] 用 `--additional-config '{"enable_balance_scheduling": true}'` |
| `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` | `1` | 用 fused op `transpose_kv_cache_by_block` |
| `VLLM_ASCEND_ENABLE_BATCH_MEMCPY` | None（自动从 CANN 头检测） | KV cache offloading 的 aclrtMemcpyBatchAsync 编译路径：`1` 强制开、`0` 强制关 |

## 4. additional-config 配置选项总表

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `xlite_graph_config` | dict | `{}` | Xlite graph 模式配置 |
| `finegrained_tp_config` | dict | `{}` | 模块级 tensor parallelism 配置 |
| `ascend_compilation_config` | dict | `{}` | Ascend 编译配置 |
| `eplb_config` | dict | `{}` | EPLB 配置 |
| `scheduler_config` | dict | `{}` | Ascend scheduler 扩展配置（balance scheduling、recompute scheduling、ShortRequestFirst、dynamic chunked pipeline parallel） |
| `refresh` | bool | `false` | 刷新全局 Ascend 配置内容；RLHF/UT/E2E 用 |
| `dump_config` | dict | `None` | 内联 msprobe dump 配置，会物化成临时 JSON 传给 debugger |
| `dump_config_path` | str | `None` | msprobe dump 配置文件路径（兼容旧选项） |
| `enable_shared_expert_dp` | bool | `False` | DP 中 expert 共享时性能更好但更耗内存 |
| `multistream_overlap_shared_expert` | bool | `False` | 多流共享 expert；仅对带 shared expert 的 MoE 生效 |
| `enable_cpu_binding` | bool | `True` | ARM 服务器上启用 Ascend 原生 CPU binding；设 `False` 关闭 |
| `enable_sleep_mode_extra_cleanup` | bool | `False` | RL 负载的额外 sleep-mode 清理（HCCL 进程组释放、ACL graph workspace 清理） |
| `pa_shape_list` | list | `[]` | page attention 算子自定义 shape 列表 |
| `enable_kv_nz` | bool | `False` | KV cache NZ 布局；仅 MLA 模型（如 DeepSeek）生效 |
| `enable_sparse_c8` | bool | `False` | DSA 模型 KV cache C8（DeepSeek V3.2、GLM5）；Ascend 950 暂不支持 |
| `c8_enable_reshape_optim` | bool | `False` | C8 下 StoreKVBlock 算子加速（需 `enable_sparse_c8`）；PD 分离仅 P 节点开 |
| `enable_mc2_hierarchy_comm` | bool | `False` | dispatch/combine op 节点间通信走 ROCE |
| `enable_prefill_mc2` | bool | `False` | 为 prefill batch 预留 mc2_token_capacity（用 `max_num_batched_tokens` 而非 decode-only capacity）；此时 `max_num_batched_tokens` 建议上限 `tp_size * 512`。临时开关 |
| `mega_moe_max_tokens` | int | `65536` | mega moe（dispatch_ffn_combine）融合算子 dispatch 后每 rank token 上限；超限 token 被丢弃。workspace 内存随此值线性增长，勿设过大 |
| `enable_flashcomm1` | bool | `False` | FlashComm1 优化（迁移期也可用 `VLLM_ASCEND_ENABLE_FLASHCOMM1`） |
| `msmonitor_use_daemon` | bool | `False` | msmonitor daemon 模式（迁移期也可用 `MSMONITOR_USE_DAEMON`） |
| `enable_mlapo` | bool | `True` | MLAPO（迁移期也可用 `VLLM_ASCEND_ENABLE_MLAPO`） |
| `weight_nz_mode` | int | `1` | Weight NZ 模式（迁移期也可用 `VLLM_ASCEND_ENABLE_NZ`） |
| `enable_context_parallel` | bool | `False` | context parallel（迁移期也可用 `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL`） |
| `enable_fused_mc2` | int | `0` | fused MC2（迁移期也可用 `VLLM_ASCEND_ENABLE_FUSED_MC2`） |
| `enable_transpose_kv_cache_by_block` | bool | `True` | transpose KV cache by block（迁移期也可用 `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK`） |
| `enable_dsa_cp` | bool | `False` | DeepSeek V3.2/V4 等同架构启用 dsa_cp；依赖 FLASHCOMM1，需先开 FLASHCOMM1 |
| `rejection_sampler_config` | dict | `{}` | rejection sampler（block verify / entropy verify）配置 |
| `multistream_dsv4_dsa_overlap` | bool | `True` | DeepSeek V4 dsa 多流 overlap |
| `enable_reduce_sample` | bool | `False` | reduce sample 优化：TP 时 logits 保持分区，仅通信 top-k 候选值/索引，替代全词表 all-to-all/all-gather |

## 5. 子配置详情

### 5.1 xlite_graph_config

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | bool | `False` | 启用 Xlite graph 模式；目前支持 Llama、Qwen dense 系列、Qwen3-VL |
| `full_mode` | bool | `False` | prefill 与 decode 都启用 Xlite；默认仅 decode |

### 5.2 finegrained_tp_config

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `lmhead_tensor_parallel_size` | int | `0` | lm_head 自定义 TP size |
| `oproj_tensor_parallel_size` | int | `0` | o_proj 自定义 TP size |
| `embedding_tensor_parallel_size` | int | `0` | embedding 自定义 TP size |
| `mlp_tensor_parallel_size` | int | `0` | mlp 自定义 TP size |

### 5.3 ascend_compilation_config

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enable_npugraph_ex` | bool | `True` | npugraph_ex 后端（Atlas 推理产品与 200I Pro 不支持，须设 `false`） |
| `enable_static_kernel` | bool | `False` | static kernel；适合 shape 变化少且有编译时间的场景 |
| `fuse_norm_quant` | bool | `True` | fuse_norm_quant pass |
| `fuse_qknorm_rope` | bool | `True` | fuse_qknorm_rope pass；环境无 Triton 时设 `False` |
| `fuse_muls_add` | bool | `True` | fuse_muls_add pass |

### 5.4 eplb_config

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `dynamic_eplb` | bool | `False` | 启用动态 EPLB |
| `expert_map_path` | str | `None` | MoE expert load balancing 的 expert map 路径 |
| `expert_heat_collection_interval` | int | `400` | EPLB 开始前的前向迭代数 |
| `algorithm_execution_interval` | int | `30` | EPLB worker 完成 CPU 任务的前向迭代数 |
| `expert_map_record_path` | str | `None` | 把 expert load 计算结果保存为 expert table 的目录 |
| `num_redundant_experts` | int | `0` | 初始化时指定冗余 expert 数 |
| `eplb_policy_type` | int | `1` | 平衡策略：0=Random、1=DefaultEplb（开源算法）、2=SwiftBalanceEplb（低带宽优化）、3=FlashLB（滑窗统计） |
| `eplb_heat_collection_stage` | str | `"all"` | 收集阶段：`"prefill"` / `"decode"` / `"all"`。PD colocation 时按阶段选择性收集可降低 expert 不均衡 |

### 5.5 scheduler_config

> 旧版顶层 `enable_balance_scheduling`、`recompute_scheduler_enable`、`short_request_first_config`、`profiling_chunk_config` 迁移期仍支持但 deprecated。同字段时 `scheduler_config` 优先。

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enable_balance_scheduling` | bool | `False` | balance scheduling（迁移期也可用 `VLLM_ASCEND_BALANCE_SCHEDULING`） |
| `recompute_scheduler_enable` | bool | `False` | recompute scheduler；**仅 PD 分离 D 节点**（`kv_role=kv_consumer`）。P 节点或 PD 混合模式启用会启动失败 |
| `profiling_chunk_config` | dict | `{}` | dynamic chunked pipeline parallel 配置 |
| `short_request_first_config` | dict | `{}` | ShortRequestFirst prefill 调度配置 |
| `batch_job_sched_config` | dict | `{}` | batch-job-aware scheduler 配置 |

#### scheduler_config.profiling_chunk_config

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | bool | `False` | 启用 dynamic chunked pipeline parallel；需 `pipeline-parallel-size > 1` |
| `smooth_factor` | float | `1.0` | 平滑因子 (0 < x ≤ 1.0)；越大越信任动态预测；`0.0` 关闭动态调整 |
| `min_chunk` | int | `4096` | 动态计算最小 chunk；应小于 `max-num-batched-tokens` |
| `need_timing` | bool | `True` | 在线校准开关 |
| `max_fit_chunk` | int | `30` | 在线校准用 chunk 时间数据条数 |

#### scheduler_config.short_request_first_config

ShortRequestFirst 是 FCFS 同步/异步 prefill 与 PD-mixed 路径的等待队列策略。不支持 batch-job-aware、profiling-chunk 或 PD 分离 D 节点调度。

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | bool | `False` | 启用 ShortRequestFirst 调度 |
| `threshold` | int | `256` | Prompt 长度阈值（token）；`<= threshold` 视为短 prefill 优先 |
| `long_max_wait_ms` | float | `0.0` | 长 prefill 在短 prefill 后最长等待时间；`0` 关闭长请求提升 |

#### scheduler_config.batch_job_sched_config

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | bool | `false` | 启用 batch-job-aware scheduler |
| `max_jobs` | int | `20` | 跟踪最大 job 数；`0` 无限 |
| `reserve_margin_blocks` | int | `2` | KV cache reserve 额外 block 余量 |
| `reserve_max_blocks` | int | `8` | 最大可预留 block 数 |
| `low_available_tokens_threshold` | int | `4096` | 长/短 decode job 优先级切换阈值；available > threshold 优先长 decode，<= 优先短 decode |
| `short_decode_token_threshold` | int | `32` | "短 decode" job 分类阈值 |

### 5.6 rejection_sampler_config

> block verify 与 entropy verify 以降低采样精度换取投机解码性能（更高接受率、更低延迟）。`posterior_alpha` 越大调整越激进——高熵 token 接受门槛更低，吞吐提升但质量下降。

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enable_block_verify` | bool | `False` | block verify 模式：用累积概率积评估所有 draft token，提高接受率 |
| `enable_entropy_verify` | bool | `False` | entropy verify 模式：按 target 分布熵调整接受门槛——高熵 token 门槛更低 |
| `posterior_threshold` | float | `0.95` | 熵调整接受门槛上限，(0, 1]；有效门槛 = `min(exp(-entropy*posterior_alpha), posterior_threshold)` |
| `posterior_alpha` | float | `0.4` | 熵缩放因子，>=0；越大高熵 token 越易接受 |

## 6. 配置示例

```python
{
    "finegrained_tp_config": {
        "lmhead_tensor_parallel_size": 8,
        "oproj_tensor_parallel_size": 8,
        "embedding_tensor_parallel_size": 8,
        "mlp_tensor_parallel_size": 8,
    },
    "enable_kv_nz": False,
    "multistream_overlap_shared_expert": True,
    "rejection_sampler_config": {
        "enable_block_verify": True,
        "enable_entropy_verify": True,
        "posterior_threshold": 0.95,
        "posterior_alpha": 0.4,
    },
    "refresh": False
}
```
