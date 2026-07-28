# vLLM-Ascend 部署知识库

本知识库由 `vllm-ascend-auto-deploy` Skill 维护，抓取自 vllm-project/vllm-ascend 官方文档（main 分支，抓取于 2026-07-28），覆盖安装、配置、特性、PD 分离、并行、KV cache 与全部模型部署教程，供 deployer 在自闭环部署中按需检索。

> 抓取源：`https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/`（用 GitHub raw 源避开 docs.vllm.ai 429 限流）
> 当前 main 分支对应 vLLM `v0.22.1` / vLLM-Ascend `v0.22.1rc1`，CANN 镜像 tag `9.0.1-910b-ubuntu22.04-py3.12`。

## 顶层章节

| 文件 | 主题 |
|------|------|
| [01-installation.md](01-installation.md) | 镜像/源码安装、CANN/torch_npu 版本矩阵、硬件要求（Atlas A2/A3/310P/A5）、SOC_VERSION 取值 |
| [02-quickstart.md](02-quickstart.md) | `vllm serve` 最小启动、OpenAI 兼容 API、健康检查 |
| [03-configuration.md](03-configuration.md) | 全部环境变量表（来自 `vllm_ascend/envs.py`）+ additional_config 参数表 |
| [04-deployment-guides.md](04-deployment-guides.md) | MindIE motor、Volcano KTheNa 外部部署集成 |
| [05-support-matrix.md](05-support-matrix.md) | 支持模型列表、支持特性矩阵、硬件/版本兼容表 |
| [06-faqs.md](06-faqs.md) | 常见问题 |
| [07-features-core.md](07-features-core.md) | flash attention、graph mode、quantization、speculative decoding、LoRA、sleep mode、structured output |
| [08-parallelism.md](08-parallelism.md) | context/sequence parallel、fine-grained TP、动态 chunk pipeline、large-scale EP、EPLB、CPU binding |
| [09-pd-disaggregation.md](09-pd-disaggregation.md) | **PD 分离完整指导**：mooncake 连接器、单机/多机 1P1D/2P1D 拓扑、`kv-transfer-config`、`use_ascend_direct`、端口（**部署核心，重点章节**） |
| [10-kv-cache-and-offload.md](10-kv-cache-and-offload.md) | KV pool / layerwise KV pool / CPU offload / recompute / lmcache / netloader / rfork / ucm |
| [11-scheduling-and-qos.md](11-scheduling-and-qos.md) | batch invariance、batch job aware scheduler、short request first、Ai QoS、ray、suffix speculative decoding |
| [12-design-notes.md](12-design-notes.md) | ACL Graph、patch 机制、KV Cache Pool Guide、ModelRunner prepare_inputs、自定义 aclnn 算子、balance schedule、npugraph |
| [13-new-features-v0.24.md](13-new-features-v0.24.md) | **v0.24.0rc 新增（main 未合入）**：长序列 Context Parallel（多节点 DeepSeek / 单节点 Qwen3-235B）、External DP + 负载均衡 proxy |

## 模型部署教程

[models/](models/) 下 42 个模型部署指南（每个含架构类型、推荐 TP/DP/EP、`vllm serve` 启动命令原文、量化方式、已知坑）。索引见 [models/INDEX.md](models/INDEX.md)。

覆盖：DeepSeek（R1/V3.1/V3.2/V4-Flash/V4-Pro/OCR2）、GLM（4.x/5/5.2）、Qwen3 全系（235B-A22B/30B-A3B/Coder/ASR/Dense/Embedding/Next/Omni/Reranker/VL ×4/3.5/3.6）、Kimi-K2（Thinking/2.5/2.6）+ **Kimi-K3**（取自 releases/v0.23.0，vLLM-Ascend 0.23.0 首次支持）、MiniMax-M2、Mixtral-8x7B、Gemma4、Hunyuan-A13B、Hy3-preview、InternVL3.5、LLaVA-OneVision、PaddleOCR-VL、Minitron-8B、gpt-oss-120b。

架构分布：MoE 22（含 Kimi-K3 多模态 MoE）、Dense 6、VLM 5、Embedding 3、Reranker 2、ASR 1、Omni 1。

> 注：知识库主体抓自 main 分支；Kimi-K3 仅在 `releases/v0.23.0` 分支，main 尚未合入。新模型可能先出现在 release 分支，重新抓取时需同时核对 `releases/v0.23.0`（及更新的 release 分支）与 main。

## deployer 检索约定

- **找某模型的启动命令**：先查 `models/<Model>.md`，命中即取 `vllm serve` 原文命令与推荐拓扑。
- **找环境变量/参数语义**：查 `03-configuration.md`。
- **PD 分离部署**：必查 `09-pd-disaggregation.md`（mooncake 连接器配置、端口、拓扑样例）。
- **硬件/版本兼容**：查 `05-support-matrix.md` 与 `01-installation.md`。
- **特性是否可用**：查 `07-features-core.md` 与 `08-parallelism.md`。

## 维护

- 知识库为快照，随 vllm-ascend main 分支更新而老化；deployer 在引用命令时应与目标镜像实际版本交叉核对（镜像内 `vllm-ascend` 版本以 `pip show` 为准）。
- 重新抓取：对每个源文件 `curl -s https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/<path>` 重新整理即可。
- 抓取失败记录：`models/_fetch-failures.md`（本次无失败）。
