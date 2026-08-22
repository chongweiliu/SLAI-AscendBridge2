---
name: demo-patterns
description: 不同模型类型的 demo.py 写法要点（transformers 5.x 已验证），含 CLIP/seq2seq/embedding 的差异
metadata:
  type: reference
---

transformers 5.15.x 下已验证的 demo.py 模式（2026-08-22，本机 venv）：

- **from_config 只在 Auto 类上**：`CLIPModel.from_config` 会报
  `AttributeError ... Did you mean: '_from_config'?`；一律用
  `AutoModel(.ForCausalLM/ForSeq2SeqLM).from_config(config)`。
- **CLIP get_*_features 返回对象而非张量**（5.x 行为）：投影后特征在
  `.pooler_output`；用 `getattr(out, "pooler_output", out)` 做 4.x/5.x 双兼容。
- **复合 config 收缩层数**：CLIP 要同时收缩 `config.text_config` 与
  `config.vision_config` 的 `num_hidden_layers`；T5 用 `num_layers`/
  `num_decoder_layers`；GPT-2 用 `n_layer`。
- **seq2seq 随机权重 dry-run 输出可能为空**（只产 eos/pad），打印原始
  token ids 并加注说明，不要当成失败。
- **多模态/视觉验证用程序内合成输入**（PIL 画红方块等），避免外部数据下载；
  full-run 断言用稳健的语义对比（如 颜色匹配 > 不匹配），不要用边缘案例。
- 句向量验证：AutoModel + attention_mask mean pooling + L2 归一化 +
  相近/无关句余弦对比（不依赖 sentence-transformers 库）。

依赖组合与索引见 [[dependency-pinning]]；设备选择注意 [[npu-host-env-quirks]]。
