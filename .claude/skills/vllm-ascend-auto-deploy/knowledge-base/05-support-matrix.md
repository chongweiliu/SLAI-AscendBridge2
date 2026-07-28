> 来源：vllm-ascend docs/source/user_guide/support_matrix/index.md + supported_models.md + supported_features.md（main 分支，抓取于 2026-07-28）
> 最新动态：<https://github.com/vllm-project/vllm-ascend/issues/1608>

# 支持矩阵

## 1. 图例

- ✅ = Supported（支持）
- 🔵 = Experimental（实验性支持，接口/功能可能变）
- ❌ = Not supported（不支持）
- 🟡 = Not tested or verified（未测试/未验证）

## 2. 支持的设备

当前**仅**支持 Atlas A2 系列（Ascend-cann-kernels-910b）、Atlas A3 系列（Atlas-A3-cann-kernels）与 Atlas 300I（Ascend-cann-kernels-310p）：

- Atlas A2 训练系列：Atlas 800T A2、Atlas 900 A2 PoD、Atlas 200T A2 Box16、Atlas 300T A2
- Atlas 800I A2 推理系列
- Atlas A3 训练系列：Atlas 800T A3、Atlas 900 A3 SuperPoD、Atlas 9000 A3 SuperPoD
- Atlas 800I A3 推理系列
- [Experimental] Atlas 300I 推理系列（Atlas 300I Duo）；310I Duo 稳定版为 vllm-ascend `v0.10.0rc1`

暂不支持：
- Atlas 200I A2（Ascend-cann-kernels-310b）——未排期
- Ascend 910、Ascend 910 Pro B（Ascend-cann-kernels-910）——未排期

技术上，只要 torch-npu 支持该设备，vllm-ascend 就支持；否则需用 custom ops 实现。

## 3. 支持的模型

### 3.1 纯文本语言模型 — 生成式 — 核心支持（A2/A3）

| Model | Support | BF16 | Hardware | W8A8 | Chunked Prefill | Auto Prefix Cache | LoRA | Spec Decoding | Async Sched | TP | PP | EP | DP | PD 分离 | Piecewise AclGraph | Fullgraph AclGraph | max-model-len |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek V4-Flash | 🔵 | ✅ | A2/A3 | ✅ | ✅ | ✅ | | ✅ | ✅ | ✅ | | ✅ | ✅ | ✅ | | ✅ | 1M |
| DeepSeek V4-Pro | 🔵 | ✅ | A2/A3 | ✅ | ✅ | ✅ | | ✅ | ✅ | ✅ | | ✅ | ✅ | ✅ | | ✅ | 1M |
| DeepSeek V3/3.1 | ✅ | ✅ | A2/A3 | ✅ | ✅ | ✅ | | ✅ | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 240k |
| DeepSeek V3.2 | 🔵 | ✅ | A2/A3 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 160k |
| DeepSeek R1 | ✅ | ✅ | A2/A3 | ✅ | ✅ | ✅ | | ✅ | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 128k |
| Qwen3-Dense | ✅ | ✅ | A2/A3 | ✅ | ✅ | ✅ | | | ✅ | ✅ | | | ✅ | | ✅ | ✅ | 128k |
| Qwen3-30B-A3B | ✅ | ✅ | A2/A3 | ✅ | ✅ | ✅ | | ✅ | ✅ | ✅ | | ✅ | ✅ | | ✅ | ✅ | |
| Qwen3-Coder-30B-A3B | ✅ | ✅ | A2/A3 | ✅ | ✅ | ✅ | | ✅ | ✅ | ✅ | | ✅ | ✅ | | ✅ | ✅ | |
| Qwen3-235B-A22B | ✅ | ✅ | A2/A3 | ✅ | ✅ | ✅ | | | ✅ | ✅ | | ✅ | ✅ | ✅ | ✅ | ✅ | 256k |
| Qwen3-Next | 🔵 | ✅ | A2/A3 | ✅ | | | | | | | ✅ | | | ✅ | | ✅ | ✅ | |
| GLM-4.x | 🔵 | | A2/A3 | ✅ | ✅ | ✅ | | ✅ | ✅ | ✅ | | ✅ | ✅ | ✅ | ✅ | ✅ | 198k |
| GLM-5/5.1 | 🔵 | ✅ | A2/A3 | ✅ | ✅ | ✅ | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 200k |
| GLM-5.2 | 🔵 | ✅ | A2/A3 | ✅ | ✅ | ✅ | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 200k |
| Gemma4 | 🔵 | ✅ | A2/A3/Ascend950 | | ✅ | ✅ | | | | ✅ | ✅ | | | ✅ | | ✅ | ✅ | |
| Kimi-K2-Thinking | 🔵 | | A2/A3 | | | | | | | | | | | | | | | |
| DeepSeekOCR2 | ✅ | ✅ | A2/A3 | | ✅ | | | | | ✅ | | | | | | | |
| MiniMax-M2.5/2.7 | ✅ | ✅ | A2/A3/Ascend950(950 exp) | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ | 🟡 | ✅ | 200k |
| Qwen2.5-Math-RM-72B | ✅ | ✅ | A2 | ✅ | 🟡 | 🟡 | ❌ | 🟡 | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 4096 |

### 3.2 纯文本语言模型 — 生成式 — 核心支持（Atlas 推理产品）

| Model | Support | BF16 | Hardware | W8A8 | Chunked Prefill | Auto Prefix Cache | LoRA | Spec Decoding | Async Sched | TP | PD 分离 | Piecewise AclGraph | Fullgraph AclGraph | max-model-len |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Dense | ✅ | ❌ | 310p | ✅ | ✅ | ✅ | ❌ | 🟡 | ✅ | ✅ | ❌ | ✅ | ✅ | 20k |
| Qwen3-30B-A3B | ✅ | ❌ | 310p | ✅ | ✅ | ✅ | ❌ | 🟡 | ✅ | ✅ | ❌ | ✅ | ✅ | 16k |

### 3.3 纯文本 — 扩展兼容模型（A2/A3）

| Model | Support | Note | Hardware |
|---|---|---|---|
| DeepSeek Distill (Qwen/Llama) | ✅ | | A2/A3 |
| Qwen3-based | ✅ | | A2/A3 |
| Qwen2 | ✅ | | A2/A3 |
| Qwen2.5 | ✅ | | A2/A3 |
| Qwen2-based | ✅ | | A2/A3 |
| QwQ-32B | ✅ | | A2/A3 |
| Llama2/3/3.1/3.2 | ✅ | | A2/A3 |
| InternLM | 🔵 | #1962 | A2/A3 |
| Baichuan / Baichuan2 | 🔵 | | A2/A3 |
| Phi-4-mini | 🔵 | | A2/A3 |
| MiniCPM / MiniCPM3 | 🔵 | | A2/A3 |
| Ernie4.5 / Ernie4.5-Moe | 🔵 | | A2/A3 |
| Gemma-2 / Gemma-3 | 🔵 | | A2/A3 |
| Phi-3/4 | 🔵 | | A2/A3 |
| Mistral/Mistral-Instruct | 🔵 | | A2/A3 |
| Hy3-preview | 🔵 | | A3 |
| DeepSeek V2.5 | 🟡 | Need test | |
| Mllama | 🟡 | Need test | |
| MiniMax-Text | 🟡 | Need test | |

### 3.4 纯文本 — Pooling 模型（Embedding/Reranker/RM）

**A2/A3**

| Model | Support | Note | Hardware | W8A8 |
|---|---|---|---|---|
| Qwen3-Embedding | 🔵 | | A2/A3 | 🟡 |
| Qwen3-VL-Embedding | 🔵 | | A2/A3 | 🔵 |
| Qwen3-Reranker | 🔵 | | A2/A3 | 🟡 |
| Qwen3-VL-Reranker | 🔵 | | A2/A3 | 🔵 |
| Molmo | 🔵 | #1942 | A2/A3 | 🟡 |
| XLM-RoBERTa-based | 🔵 | | A2/A3 | 🟡 |
| Bert | 🔵 | | A2/A3 | 🟡 |
| Qwen2.5-Math-RM-72B | ✅ | Reward Model, gsm8k_correctness accuracy=0.80 | A2 | |

**Atlas 推理产品**（均 FP16）

| Model | Support | Note | W8A8 |
|---|---|---|---|
| Qwen3-Embedding | 🔵 | FP16 | 🟡 |
| Qwen3-VL-Embedding | 🔵 | FP16 | 🔵 |
| Qwen3-Reranker | 🔵 | FP16 | 🟡 |
| Qwen3-VL-Reranker | 🔵 | FP16 | 🔵 |
| XLM-RoBERTa-based | 🔵 | embedding and scoring | 🟡 |
| Qwen2.5-based | 🔵 | FP16 classification | 🟡 |

### 3.5 多模态语言模型 — 生成式 — 核心支持（A2/A3）

| Model | Support | BF16 | Hardware | W8A8 | Chunked Prefill | Auto Prefix Cache | LoRA | Spec Decoding | Async Sched | TP | PP | EP | DP | PD 分离 | Piecewise AclGraph | Fullgraph AclGraph | max-model-len |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL | ✅ | | A2/A3 | | | | | | | | ✅ | | | | | | ✅ | ✅ | |
| Qwen3-VL-30B-A3B/235B-A22B | ✅ | ✅ | A2/A3 | ✅ | ✅ | ✅ | | | ✅ | ✅ | | ✅ | ✅ | ✅ | ✅ | ✅ | 262144 |
| Qwen3.5-397B-A17B | ✅ | ✅ | A2/A3 | ✅ | ✅ | ✅ | | ✅ | ✅ | ✅ | | ✅ | ✅ | ✅ | ✅ | ✅ | 1010000 |
| Qwen3.5-27B / Qwen3.6-27B | ✅ | ✅ | A2/A3 | ✅ | ✅ | ✅ | | ✅ | ✅ | ✅ | | ✅ | ✅ | ✅ | ✅ | ✅ | 262144 |
| Qwen3.6-35B-A3B | 🔵 | ✅ | A2/A3 | ✅ | ✅ | ✅ | | 🔵 | ✅ | ✅ | | ✅ | ✅ | ❌ | ✅ | ✅ | 262144 |
| Qwen3-Omni-30B-A3B-Thinking | 🔵 | | A2/A3 | | | | | | | | ✅ | | ✅ | | | | | |
| Kimi-K2.5/Kimi-K2.6 | ✅ | | A2/A3 | | ✅ | ✅ | | ✅ | ✅ | ✅ | | ✅ | ✅ | ✅ | ✅ | ✅ | 262144 |

### 3.6 多模态 — 生成式 — 核心支持（Atlas 推理产品）

| Model | Support | BF16 | Hardware | W8A8 | Chunked Prefill | Auto Prefix Cache | LoRA | Spec Decoding | Async Sched | TP | PD 分离 | Piecewise AclGraph | Fullgraph AclGraph | max-model-len |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL | ✅ | ❌ | 310p | ✅ | ✅ | ✅ | ❌ | 🟡 | ✅ | ✅ | ❌ | ✅ | ✅ | 16k |
| Qwen3.5-Dense | ✅ | ❌ | 310p | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | 256k |
| Qwen3.5-35B-A3B | ✅ | ❌ | 310p | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | 256k |
| Qwen3.6-27B | ✅ | ❌ | 310p | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | 256k |
| Qwen3.6-35B-A3B | ✅ | ❌ | 310p | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | 256k |
| PaddleOCR-VL | ✅ | ❌ | 310p | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | 16k |
| Qwen3-ASR | ✅ | ❌ | 310p | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | 🟡 | ❌ | ✅ | ✅ | 4096 |

### 3.7 多模态 — 扩展兼容模型（A2/A3）

| Model | Support | Note | Hardware |
|---|---|---|---|
| Qwen2-VL | ✅ | | A2/A3 |
| Qwen3-Omni | 🔵 | | A2/A3 |
| QVQ | 🔵 | | A2/A3 |
| Qwen2-Audio | 🔵 | | A2/A3 |
| Aria | 🔵 | | A2/A3 |
| LLaVA-Next / LLaVA-Next-Video | 🔵 | | A2/A3 |
| MiniCPM-V | 🔵 | | A2/A3 |
| Mistral3 | 🔵 | | A2/A3 |
| Phi-3-Vision/Phi-3.5-Vision | 🔵 | | A2/A3 |
| Gemma3 | 🔵 | | A2/A3 |
| Llama3.2 | 🔵 | | A2/A3 |
| PaddleOCR-VL | 🔵 | | A2/A3 |
| Llama4 | ❌ | #1972 | |
| Keye-VL-8B-Preview | ❌ | #1961 | |
| Florence-2 | ❌ | #2259 | |
| GLM-4V | ❌ | #2260 | |
| InternVL2.0/2.5/3.0、InternVideo2.5、Mono-InternVL | ❌ | #2064 | |
| Whisper | ❌ | #2262 | |
| Ultravox | 🟡 | Need test | |

## 4. 支持的特性矩阵

原则：与 vLLM 对齐，积极协作加速支持。

| Feature | Status | Next Step |
|---|---|---|
| Chunked Prefill | 🟢 Functional | 功能可用 |
| Automatic Prefix Caching | 🟢 Functional | #732 |
| LoRA | 🔵 Experimental | 接口/功能可能变 |
| Speculative decoding | 🟢 Functional | 基础支持 |
| Pooling | 🔵 Experimental | CI 需适配更多模型；V1 依赖 vLLM |
| Enc-dec | 🟡 Planned | 需 vLLM 先支持 |
| Multi Modality | 🟢 Functional | 优化适配更多模型 |
| LogProbs | 🟢 Functional | CI needed |
| Prompt LogProbs | 🟢 Functional | CI needed |
| Async output | 🟢 Functional | CI needed |
| Beam search | 🔵 Experimental | CI needed |
| Guided Decoding | 🟢 Functional | #177 |
| Tensor Parallel | 🟢 Functional | TP>4 在 graph 模式下工作 |
| Pipeline Parallel | 🟢 Functional | 官方 guide 待补 |
| Expert Parallel | 🟢 Functional | 支持动态 EPLB |
| Data Parallel | 🟢 Functional | Qwen3 MoE DP 支持 |
| Prefill Decode Disaggregation | 🟢 Functional | xPyD 支持 |
| Quantization | 🟢 Functional | W8A8 可用；W4A8 等开发中 |
| Graph Mode | 🟢 Functional | 功能可用 |
| Sleep Mode | 🟢 Functional | 功能可用 |
| Context Parallel | 🟢 Functional | 功能可用 |

状态图例：🟢 Functional（完全可用，持续优化）/ 🔵 Experimental（实验性）/ 🚧 WIP（开发中）/ 🟡 Planned（计划中）/ 🔴 NO plan/Deprecated。
