> 来源：vllm-ascend docs/source/tutorials/models/index.md + 各模型教程（main 分支，抓取于 2026-07-28；Kimi-K3 取自 releases/v0.23.0 分支，首次支持于 vLLM-Ascend 0.23.0）

# vLLM-Ascend 模型部署知识库索引

共收录 41 个模型部署教程。

| 模型名 | 架构类型 | 推荐TP | 是否有量化 | 知识库文件名 |
|--------|----------|--------|-----------|-------------|
| DeepSeek-R1 | MoE | 4 | 是 | DeepSeek-R1.md |
| DeepSeek-V3.1 | MoE | 4 | 是 | DeepSeek-V3.1.md |
| DeepSeek-V3.2 | MoE | 8/16 | 是 | DeepSeek-V3.2.md |
| DeepSeek-V4-Flash | MoE | 4/8 | 是 | DeepSeek-V4-Flash.md |
| DeepSeek-V4-Pro | MoE | 8/16 | 是 | DeepSeek-V4-Pro.md |
| DeepSeekOCR2 | VLM | 1 | 否 | DeepSeekOCR2.md |
| GLM4.x | MoE | 8 | 是 | GLM4.x.md |
| GLM5 | MoE | 8/16 | 是 | GLM5.md |
| GLM5.2 | MoE | 8/16 | 是 | GLM5.2.md |
| Gemma4 | MoE | 4 | 否 | Gemma4.md |
| Hunyuan-A13B-Instruct | MoE | 4 | 否 | Hunyuan-A13B-Instruct.md |
| Hy3-preview | MoE | 16 | 否 | Hy3-preview.md |
| InternVL3.5 | VLM | 4 | 是 | InternVL3.5.md |
| Kimi-K2-Thinking | MoE | 16 | 是 | Kimi-K2-Thinking.md |
| Kimi-K2.5 | MoE | 4 | 是 | Kimi-K2.5.md |
| Kimi-K2.6 | MoE | 4 | 是 | Kimi-K2.6.md |
| LLaVA-OneVision-Qwen2-0.5B-OV | VLM | - | 否 | LLaVA-OneVision-Qwen2-0.5B-OV.md |
| MiniMax-M2 | MoE | 4/8 | 是 | MiniMax-M2.md |
| Minitron-8B-Base | Dense | 1 | 否 | Minitron-8B-Base.md |
| Mixtral-8x7B-Instruct-v0.1 | MoE | 4 | 是 | Mixtral-8x7B-Instruct-v0.1.md |
| PaddleOCR-VL | VLM | - | 否 | PaddleOCR-VL.md |
| Qwen-VL-Dense | VLM | - | 否 | Qwen-VL-Dense.md |
| Qwen2.5-Math-RM-72B | Dense | - | 否 | Qwen2.5-Math-RM-72B.md |
| Qwen3-235B-A22B | MoE | 4/8/16 | 是 | Qwen3-235B-A22B.md |
| Qwen3-30B-A3B | MoE | 1/2/4 | 是 | Qwen3-30B-A3B.md |
| Qwen3-ASR-1.7B | ASR | 1 | 否 | Qwen3-ASR-1.7B.md |
| Qwen3-Coder-30B-A3B | MoE | 1/4 | 是 | Qwen3-Coder-30B-A3B.md |
| Qwen3-Dense | Dense | 1/2/4/8 | 是 | Qwen3-Dense.md |
| Qwen3-Embedding | Embedding | - | 否 | Qwen3-Embedding.md |
| Qwen3-Next | MoE | 4 | 否 | Qwen3-Next.md |
| Qwen3-Omni-30B-A3B-Thinking | Omni (多模态) | 1/2/4 | 是 | Qwen3-Omni-30B-A3B-Thinking.md |
| Qwen3-Reranker | Reranker | - | 否 | Qwen3-Reranker.md |
| Qwen3-VL-235B-A22B-Instruct | VLM/MoE | 4/8 | 是 | Qwen3-VL-235B-A22B-Instruct.md |
| Qwen3-VL-30B-A3B-Instruct | VLM/MoE | 2 | 否 | Qwen3-VL-30B-A3B-Instruct.md |
| Qwen3-VL-Embedding | Embedding | - | 否 | Qwen3-VL-Embedding.md |
| Qwen3-VL-Reranker | Reranker | - | 否 | Qwen3-VL-Reranker.md |
| Qwen3.5-27B-Qwen3.6-27B | Dense | 2/4 | 是 | Qwen3.5-27B-Qwen3.6-27B.md |
| Qwen3.5-397B-A17B | MoE | 2/8/16 | 是 | Qwen3.5-397B-A17B.md |
| Qwen3.5-Dense | Dense | 1 | 否 | Qwen3.5-Dense.md |
| Qwen3.6-35B-A3B | MoE | 2 | 是 | Qwen3.6-35B-A3B.md |
| gpt-oss-120b | Dense | 4 | 否 | gpt-oss-120b.md |
| Kimi-K3 | MoE/多模态 | 16 | 是(W4A8) | Kimi-K3.md |