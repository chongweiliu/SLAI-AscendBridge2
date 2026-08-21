# 多模态 checkpoint → 文本头权重重映射

## 何时需要
CPT 目标是文本（CausalLM），但用户的 ckpt 可能是多模态（ForConditionalGeneration，含 vision/merger/mtp）。
直接 `AutoModelForCausalLM.from_pretrained(multimodal_dir)` 会全 miss（键前缀不同）或加载错架构。

## 判定
读 `config.json`：
- `architectures` 含 `ForConditionalGeneration`、有 `text_config`、`image_token_id` → 多模态。
- `model_type` 如 `qwen3_5`（多模态）vs `qwen3_5_text`（文本）。

## Qwen3.5 实例（已验证）
- ckpt 键：`model.language_model.<x>` + `model.visual.*`(153) + `mtp.*`(15)。
- 文本头 `Qwen3_5ForCausalLM(tc)`（tc=cfg.text_config）期望键：`model.<x>` + `lm_head.weight`（tie 时缺）。
- 映射：strip `model.language_model.` → `model.`；丢 `visual`/`mtp`；`strict=False`。
- `tie_word_embeddings=True` → lm_head 缺失正常，`model.tie_weights()`。

## tie=False 时须保留 lm_head.weight（必读）
- **先 `cat config.json` 看 `tie_word_embeddings`**：同族不同规格模型可能不同（如 Qwen3.5-0.8B 是 `True`，Qwen3.5-9B 是 `False`）。
- **tie=False**：ckpt 顶层有独立的 `lm_head.weight` 张量，重映射时**保留**它（不 strip `model.language_model.` 之外的不动、不 drop），加载后 `miss_lm_head` 应为 0；**不要**调 `tie_weights()`（会把 lm_head 绑成 embed，错误覆盖）。
- **tie=True**：lm_head 与 embed_tokens 共享，ckpt 无独立 `lm_head.weight`，加载后 lm_head 缺失正常，**必须**调 `model.tie_weights()` 绑定。
- 判据：加载后看 `miss_lm_head` 数——tie=True 时为 1（正常），tie=False 时应为 0。

## 通用步骤
1. 读 ckpt 键前缀（`model.safetensors.index.json` 的 `weight_map` keys）。
2. 找文本子树前缀（如 `model.language_model.`、`text_model.`、`language_model.`）。
3. strip 到文本头期望前缀（通常 `model.`）。
4. 丢非文本键（`visual`/`vision`/`merger`/`mtp`/`multi_modal_projector` 等）。
5. `strict=False` 加载，看 miss/unexp；tie 时 lm_head 缺失正常。
6. 若 miss 多 → 前缀判断错，回查 ckpt 与文本头 `model.state_dict().keys()` 对齐。

## 代码骨架
```python
from transformers.models.qwen3_5 import Qwen3_5ForCausalLM  # 或对应文本头类
from safetensors.torch import load_file
cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
tc = cfg.text_config
model = Qwen3_5ForCausalLM(tc)
keys = set(model.state_dict().keys())
# 读所有 shard
files = sorted(set(json.load(open(MODEL+"/model.safetensors.index.json"))["weight_map"].values()))
ckpt = {}; [ckpt.update(load_file(MODEL+"/"+f, device="cpu")) for f in files]
# 映射：试几种常见前缀
PREFIX = "model.language_model."   # 据实际改
sd = {k.replace(PREFIX,"model."): v for k,v in ckpt.items()
      if "visual" not in k and "mtp" not in k and "vision" not in k and "merger" not in k
      and k.replace(PREFIX,"model.") in keys}
miss, unexp = model.load_state_dict(sd, strict=False)
if getattr(tc,"tie_word_embeddings",False): model.tie_weights()
```

## 退路
若用户模型是纯文本 LM（多数模型），`AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=float32)` 直接成功，**无需**本文档。
本文档仅用于多模态→文本头的特殊情况。
