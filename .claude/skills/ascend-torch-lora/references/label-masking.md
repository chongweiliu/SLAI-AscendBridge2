# SFT loss 掩码：只对 assistant token 算 loss

> SFT 的核心：user/system 的 token 不算 loss（标 -100），只对 assistant 回复 token 算 next-token CE。mask 做错 = 在做 CPT，loss 看似下降但学错东西。

## 三种方法（按优先级）

### 方法 A：`return_assistant_tokens_mask`（首选，但不可靠）

```python
enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                              return_assistant_tokens_mask=True, return_dict=True, return_tensors="pt")
input_ids = enc["input_ids"][0]
labels = input_ids.clone()
labels[enc["assistant_masks"][0] == 0] = -100
```
**坑**：Qwen3.5 等模板实现有 bug，返回的 `assistant_masks` 全 0 → label token = 0 → 静默学错（见 pitfalls #5）。**必须自检 `int(am.sum()) > 0`，否则 fallback 方法 B。**

### 方法 B：字符偏移映射法（最稳健，模板默认）

不依赖 chat template 的 mask 实现，直接在渲染字符串上定位 assistant 块：

```python
ASTART = "<|im_start|>assistant\n"   # 按目标模型调整(见下表)
AEND = "<|im_end|>"

rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
enc = tok(rendered, return_offsets_mapping=True, add_special_tokens=False)
input_ids = enc["input_ids"]
offsets = enc["offset_mapping"]

# 在字符串中 find 所有 assistant 块的字符区间
spans, pos = [], 0
while True:
    idx = rendered.find(ASTART, pos)
    if idx == -1: break
    eidx = rendered.find(AEND, idx)
    if eidx == -1: break
    end = eidx + len(AEND)
    if end < len(rendered) and rendered[end] == "\n":  # 含尾随换行
        end += 1
    spans.append((idx, end))
    pos = end

# token 的 offset 落入任一 assistant 字符区间 → label
labels = [-100] * len(input_ids)
for ti, (cs, ce) in enumerate(offsets):
    if cs == ce: continue
    for (sp_st, sp_en) in spans:
        if cs < sp_en and ce > sp_st:   # 区间重叠
            labels[ti] = input_ids[ti]
            break
```

**为何稳健**：①不依赖模板的 mask 分支实现；②用 `tokenize=False`（字符串）规避 BatchEncoding `len()` 坑（pitfalls #4）；③不依赖前缀一致性（pitfalls #8）。

### 方法 C：增量前缀法（不推荐，易错）

`[len(encode(msgs[:i])), len(encode(msgs[:i+1])))` 求 assistant 区间。坑多（#4 #8 #12），仅在方法 A/B 都不可用时用。

## assistant 块定界符速查

按目标模型的 chat template 调整 `ASTART`/`AEND`（环境变量 `ASSISTANT_START`/`ASSISTANT_END`）。不确定时先渲染一条看：
```python
print(repr(tok.apply_chat_template(msgs, tokenize=False)[:300]))
```

| 模型族 | ASTART | AEND |
|---|---|---|
| Qwen / GLM / ChatML | `<|im_start\|>assistant\n` | `<|im_end\|>` |
| Llama-3 / Llama-4 | `<\|start_header_id\|>assistant<\|end_header_id\|>\n\n` | `<\|eot_id\|>` |
| Llama-2 / vicuna | ` ASSISTANT:` | `</s>` |
| Mistral | `[INST] ... [/INST] ` (assistant 在 [/INST] 之后) | `<\|/INST\|>` 后到结束 |
| DeepSeek | `<｜Assistant｜>` | `<｜end▁of▁sentence｜>` 或 `<｜User｜>` |

> 实战：对 Qwen3.5 用 ChatML 定界符，自检 label token 含 `<|im_start|>assistant\n...<|im_end|>` 且不含 `<|im_start|>user` 即正确。

## 自检（训练前务必跑）

```python
n_label = int((torch.tensor(labels) != -100).sum())
assert n_label > 0, "label 全 -100！检查 ASTART/AEND 或用方法 A 失败"
decoded = tok.decode([t for t, lb in zip(input_ids, labels) if lb != -100][:40])
print("label tokens:", n_label, "解码:", decoded)
# 期望: 以 <|im_start|>assistant 开头, 不含 <|im_start|>user
assert "<|im_start|>user" not in decoded, "label 混入 user token！"
```

若 `n_label == 0`：①检查 `ASSISTANT_START/END` 是否匹配该模型 chat template；②若用方法 A，检查 `assistant_masks.sum()` 是否为 0（用方法 B）。
