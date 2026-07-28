> 来源：vllm-ascend docs/source/faqs.md（main 分支，抓取于 2026-07-28）

# 常见问题（FAQs）

## 版本专属 FAQ

- [v0.23.0rc1](https://github.com/vllm-project/vllm-ascend/issues/12238)
- [v0.22.1rc1](https://github.com/vllm-project/vllm-ascend/issues/10593)
- [v0.21.0rc1](https://github.com/vllm-project/vllm-ascend/issues/9970)
- [v0.20.2rc1](https://github.com/vllm-project/vllm-ascend/issues/9586)
- [v0.19.1rc1](https://github.com/vllm-project/vllm-ascend/issues/8819)
- [v0.18.0](https://github.com/vllm-project/vllm-ascend/issues/8238)

## Q1. 支持哪些设备？

当前**仅** Atlas A2 系列（Ascend-cann-kernels-910b）、Atlas A3 系列与 Atlas 300I（Ascend-cann-kernels-310p）：
- Atlas A2 训练系列：Atlas 800T A2、Atlas 900 A2 PoD、Atlas 200T A2 Box16、Atlas 300T A2
- Atlas 800I A2 推理系列
- Atlas A3 训练系列：Atlas 800T A3、Atlas 900 A3 SuperPoD、Atlas 9000 A3 SuperPoD
- Atlas 800I A3 推理系列
- [Experimental] Atlas 300I 推理系列（Atlas 300I Duo）；310I Duo 稳定版为 `v0.10.0rc1`

暂不支持：Atlas 200I A2（310b）、Ascend 910 / 910 Pro B（910）均未排期。技术上只要 torch-npu 支持，vllm-ascend 就支持。

## Q2. 如何获取 Docker 容器？

Quay.io：[vllm-ascend](https://quay.io/repository/ascend/vllm-ascend?tab=tags)、[cann](https://quay.io/repository/ascend/cann?tab=tags)。

国内加速：

```bash
TAG=v0.9.1
docker pull m.daocloud.io/quay.io/ascend/vllm-ascend:$TAG
# 或
docker pull quay.nju.edu.cn/ascend/vllm-ascend:$TAG
```

离线环境导入导出：

```bash
# 联网机导出
TAG=v0.22.1rc1
docker pull quay.io/ascend/vllm-ascend:$TAG
docker save quay.io/ascend/vllm-ascend:$TAG | gzip > vllm-ascend-$TAG.tar.gz

# 离线机导入
docker load -i vllm-ascend-$TAG.tar.gz
docker images | grep vllm-ascend
```

## Q3. 支持哪些模型？

见 [supported_models](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/supported_models.html)。

## Q4. 如何联系社区？

- GitHub [issue](https://github.com/vllm-project/vllm-ascend/issues?page=1)
- [周会](https://docs.google.com/document/d/1hCSzRTMZhIB8vRq1_qOOjx4c9uYUxvdQvDsMV2JcSrw/edit?tab=t.0#heading=h.911qu8j8h35z)
- [WeChat 群](https://github.com/vllm-project/vllm-ascend/issues/227)
- vLLM 论坛 [ascend 频道](https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support/6)

## Q5. V1 引擎支持哪些特性？

见 [supported_features](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/supported_features.html)。

## Q6. "Failed to infer device type" 或 "libatb.so: cannot open shared object file"？

NPU 环境未配好：
1. `source /usr/local/Ascend/nnal/atb/set_env.sh` 启用 NNAL
2. `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 启用 CANN
3. `npu-smi info` 检查 NPU

Python 校验：

```python
import torch
import torch_npu
import vllm
```

仍不行就提 GitHub issue。

## Q7. vllm-ascend 与 vLLM 如何配合？

vllm-ascend 是 vLLM 的硬件插件。稳定 release 与同版本 vLLM 对齐；RC release 用对应 vLLM final release 版本。例如 `vllm-ascend v0.18.0rc1` 匹配 vLLM `v0.18.0`。main 分支保证每个 commit vllm-ascend 与 vLLM 兼容。

## Q8. 是否支持 Prefill-Decode 分离？

支持，用 Mooncake backend。见 [官方教程](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/features/pd_disaggregation_mooncake_multi_node.html)。

## Q9. 是否支持量化？

已支持 w8a8、w4a8、w4a4 量化。

## Q10. 如何测试？

三方面：功能（CI 含 vLLM 原生单测 + vllm-ascend 自有单测 + E2E）、性能（[benchmark 工具](https://github.com/vllm-project/vllm-ascend/tree/main/benchmarks)，每个 PR 发 perf 网站）、精度（接入 CI 中）； nightly 全量跑。

## Q11. 如何修复 "InvalidVersion"？

通常因装了开发/可编辑版 vLLM。设环境变量 `VLLM_VERSION` 为已装 vLLM 版本，格式 `X.Y.Z`。

## Q12. 如何处理 OOM？

参考 [vLLM OOM 文档](https://docs.vllm.ai/en/latest/usage/troubleshooting/#out-of-memory)。NPU 片上内存有限时动态分配易产生碎片导致 OOM，建议：
- 限制 `--max-model-len`：省 KV cache 初始化片上内存
- 调 `--gpu-memory-utilization`：默认 `0.9`，调低预留更多内存
- 配 `PYTORCH_NPU_ALLOC_CONF`：如 `export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` 启用虚拟内存缓解碎片（详见 [PYTORCH_NPU_ALLOC_CONF](https://www.hiascend.com/document/detail/zh/Pytorch/700/comref/Envvariables/Envir_012.html)）

## Q13. DeepSeek 启用 NPU graph 模式失败

MLA + NPU graph 时，每 KV head 的 query 数须为 32/64/128。DeepSeek-V2-Lite 仅 16 头 → 16 query/KV，超出范围，未来支持。

DeepSeek-V3/R1 确保 TP 切分后 `num_heads`/`num_kv_heads` ∈ {32, 64, 128}：

```
[rank0]: RuntimeError: EZ9999: Inner Error!
... numHeads / numKvHeads = 8, MLA only support {32, 64, 128}.
```

## Q14. 卸载后从源码重装失败

C/C++ 编译失败时用 `python setup.py install`（推荐）或 `python setup.py clean` 清缓存。

## Q15. 如何生成确定性结果？

1. 贪心采样 `temperature=0`：
   ```python
   from vllm import LLM, SamplingParams
   prompts = ["Hello, my name is", "..."]
   sampling_params = SamplingParams(temperature=0)
   llm = LLM(model="Qwen/Qwen3-0.6B")
   outputs = llm.generate(prompts, sampling_params)
   ```
2. 设环境变量：
   ```bash
   export LCCL_DETERMINISTIC=1
   export HCCL_DETERMINISTIC=true
   export ATB_MATMUL_SHUFFLE_K_ENABLE=0
   export ATB_LLM_LCOC_ENABLE=0
   ```

## Q16. 多模态 "ImportError: Please install vllm[audio]"

部分多模态模型需 `librosa`，装 `qwen-omni-utils`（会装 librosa 及依赖）：`pip install qwen-omni-utils`。

## Q17. stream 资源耗尽导致 size capture 失败

```
capture_begin:...NPU function error: ... error code is 207008
[Error]: Stream resources are insufficient.
Insufficient_Stream_Resources(EL0009)
```

缓解：
1. 升级更新 HDK/CANN 栈
2. 手动减小 graph size：`{"cudagraph_capture_sizes":[size1,size2,...]}`，或降低 `max_cudagraph_capture_size`
3. 主要为均匀 decode 时用 `FULL` / `FULL_DECODE_ONLY` 代替 `PIECEWISE`
4. PIECEWISE/FULL_AND_PIECEWISE 升级后仍失败：按真实负载设 `cudagraph_capture_sizes` 减覆盖
5. 调试启动失败：临时禁用 graph 模式（`cudagraph_mode="NONE"` / `enforce_eager=True`）

根因：PIECEWISE 场景 graph 数随模型深度与 capture 覆盖线性增长，可能超出软/硬栈运行时资源。vLLM Ascend 不再本地自动缩小 PIECEWISE capture-size。

## Q18. 如何安装自定义 torch_npu 版本？

torch-npu 会在装 vllm-ascend 时被覆盖。需特定版本时，在 vllm-ascend 装完后手动装指定版本 torch-npu。

## Q19. Kylin OS 上 docker pull 报 "invalid tar header"

```
failed to register layer: ... archive/tar: invalid tar header
```

用离线方式：在另一台标准 Ubuntu 拉起 ARM64 镜像打成 `.tar`：

```bash
export IMAGE_TAG=v0.10.0rc1-310p
export IMAGE_NAME="quay.io/ascend/vllm-ascend:${IMAGE_TAG}"
# 国内镜像：export IMAGE_NAME="m.daocloud.io/quay.io/ascend/vllm-ascend:${IMAGE_TAG}"
docker pull --platform linux/arm64 "${IMAGE_NAME}"
docker save -o "vllm_ascend_${IMAGE_TAG}.tar" "${IMAGE_NAME}"
```

把 `.tar` 拷到目标机加载。

## Q20. docker run 报 "operation not permitted"

用 `--shm-size` 时可能需加 `--privileged=true`（注意安全风险，仅信任容器源时用）。

## Q21. CPU-only 机器从源码构建如何设 SOC_VERSION？

无 `npu-smi` 时须手动设。参考 Dockerfile：

```bash
# Atlas A2
export SOC_VERSION="ascend910b1"
# Atlas A3
export SOC_VERSION="ascend910_9391"
# Atlas 300I
export SOC_VERSION="ascend310p1"
# Ascend 950 Products
export SOC_VERSION="<value starting with ascend950>"
```

## Q22. 并发增长 TPOT 暴涨？

并发增 4 时 TPOT 增 0.5~1ms 属正常；暴涨 10~100ms 多为 [PREEMPTION](https://docs.vllm.ai/en/latest/configuration/optimization/#preemption)。KV cache 达 100% 触发抢占，默认重算被抢占请求 KV。验证：
- 看日志 `GPU KV cache usage: 99.0%,`
- 启动日志 `GPU KV cache size: 66340 tokens`、`Maximum concurrency for 16,384 tokens per request: 4.05`

缓解（核心是增 KV cache）：增 `--gpu-memory-utilization`，或降 `--max-num-seqs` 与 `--max-num-batched-tokens`。

## Q23. 单节点 vs 多节点如何选？

模型放得进单节点 NPU 内存就用单节点。如 Qwen3-32B BF16 需 4×64G 卡，单节点多卡 TP 即可。仅当总 NPU 数超单节点容量才需多节点。

## Q24. 用哪种量化？

- **BF16**：精度最佳，内存最大；精度优先且内存充足时用
- **W8A8**：精度与内存平衡；大模型（如 32B）在内存受限硬件上用
- **W4A8 / W4A4**：内存最大压缩；在更小硬件部署更大模型，精度有折损

## Q25. 何时启用 FlashComm_v1？

用 TP（TP ≥ 2）+ 高并发时启用 `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`。它有阈值保护，低并发不会激活退化。

## Q26. FIA 与 PA 算子的区别？

FIA（Flash Attention）是 vLLM-Ascend 默认注意力算子。部分 batch（尤其中等并发）FIA 性能欠佳时，可通过 `--additional-config` 的 `pa_shape_list` 手动启用 PA（Page Attention）；运行时 batch 命中 `pa_shape_list` 值即切 PA。这是临时调优旋钮，未来 FIA 优化会废弃它。
