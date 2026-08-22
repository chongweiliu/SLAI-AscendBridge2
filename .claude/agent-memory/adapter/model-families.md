---
name: model-families
description: 各模型族的适配要点（transformers 版本、架构坑）；已验证过的族直接复用结论
metadata:
  type: project
---

# 模型族适配经验

## qwen3_5 族（Qwen3.5 / Qwen3.8，多模态 + 混合注意力）

- `model_type=qwen3_5`，架构 `Qwen3_5ForConditionalGeneration`（视觉塔 + 文本主干，
  3/4 linear_attention + 1/4 full_attention，mrope）。tags 含 `image-text-to-text`。
- **transformers 版本硬要求**：`>=5.15.1`（4.57.6 无 qwen3_5 定义；判断方法：
  `ls <venv>/site-packages/transformers/models/ | grep qwen`）。pin `>=5.15.1,<6`。
- Auto 类：`AutoModelForImageTextToText`（5.15 中映射到 ConditionalGeneration）；
  get_model_info 报的 `AutoModelForMultimodalLM` 在 5.15.1 里不存在，别用。
- **不要装 torchaudio**（本机损坏）；transformers 对它的导入有 `is_torchaudio_available()`
  守卫，文本路径（AutoTokenizer）不受影响。
- Dry-run 缩层：`text_config.num_hidden_layers` 缩到 4，同时把 `layer_types`
  截断到同长度（保留 1 个 full_attention），vision depth 缩到 2。
- 文本-only 生成（input_ids 无图像）可直接跑，覆盖文本主干全部算子；混合注意力
  在 torch_npu 2.8.0.post5 上无需 patch 即可生成。
- 规模参考：9B bf16≈18GB 单卡；27B bf16≈54GB 单卡 64GB 紧张，用 `device_map="auto"`
  跨 2 卡（本机 2×910）。
- 已验证：Qwen3.5-9B（npu 单卡）、Qwen3.8-27B（npu 跨 2 卡），均 dry+full 通过。

## 标准 transformers 文本模型（如 qwen3）

- 模板直接可用；tokenizer 若为 GPT2 风格 BPE（vocab.json+merges.txt）无需
  sentencepiece/tiktoken。
