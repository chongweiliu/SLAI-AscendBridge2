---
name: transformers5-remote-code-patches
description: transformers 5.x 加载 4.4x 时代 trust_remote_code 模型的 3 个通用 patch 点（tokenizer/rope/tied_weights）+ NPU bool mask 索引坑
metadata:
  type: feedback
---

transformers 5.5.4 加载按 4.44 API 编写的 trust_remote_code 模型（实测：GenerTeam GENERanno-eukaryote-0.5b-base，2026-09-01，training-ws/GENERanno-cpt/）需要 3 个最小 patch。未来加载 GENERator 系列等其它老 remote code DNA/蛋白模型大概率同样适用。

**Why:** huggingface 生态 remote code 大量按 4.4x 写成直接赋值/老接口；5.x 改了 `PreTrainedTokenizer.__setattr__` 钩子、移除 `ROPE_INIT_FUNCTIONS['default']`、新增 `all_tied_weights_keys` 静态属性收集，均会在 `from_pretrained` / `__init__` 链上崩溃。

**How to apply:**（按报错顺序逐一打，均在模型目录的 remote code 文件内，不动 site-packages）

1. `AttributeError: XxxTokenizer has no attribute _special_tokens_map`
   → tokenizer `__init__` 里删掉 `super().__init__()` 之前的 `self.xxx_token = ...` 直接赋值，改为 `super().__init__(unk_token=..., bos_token=..., ...)` 传参；`*_token_id` 赋值移到 super 之后。
2. `KeyError: 'default'`（`ROPE_INIT_FUNCTIONS[self.rope_type]`）
   → 5.x keys 只剩 `['linear','dynamic','yarn','longrope','llama3','proportional']`；补等价 `_default_rope_init(config, device, **kwargs)`：`dim = config.hidden_size // config.num_attention_heads`（或 `head_dim`），`base = config.rope_theta`，`inv_freq = 1/base^(arange(0,dim,2)/dim)`，返回 `(inv_freq, 1.0)`（返回签名 5.x 未变，仍是 `(Tensor, float)` 元组）。
3. `AttributeError: ... no attribute 'all_tied_weights_keys'`（from_pretrained 内 `_move_missing_keys_from_meta_to_device`）
   → 在 `XxxPreTrainedModel.__init__` 的 `super().__init__(config)` 后补：`if not hasattr(self, "all_tied_weights_keys"): self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(all_submodels=False)`（try/except 兜底 `{}`）。

另附一个 NPU 侧坑：**CPU bool mask 索引 NPU tensor** 报
`RuntimeError: a Tensor with N elements cannot be converted to Scalar`（N=mask 总元素数）。
修复：随机数在 CPU generator 生成后，bool mask 必须 `.to(batch.device)` 再索引，或整体在 CPU 构造完再搬。
