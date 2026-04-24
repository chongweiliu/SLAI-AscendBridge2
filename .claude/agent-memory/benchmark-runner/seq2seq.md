# Seq2Seq 模型评测经验

## T5/Flan-T5 模型

### 模型类
- `AutoModelForSeq2SeqLM`

### 关键点
1. 使用 `generate()` 方法进行推理
2. 不需要特殊的 decoder_input_ids 处理（generate 自动处理）
3. 输出通过 tokenizer.decode() 解码为文本

### 评测脚本要点
```python
from transformers import AutoModelForSeq2SeqLM

# 模型加载
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, ...)

# 推理
output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)

# 解码
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

### 注意事项
- T5 模型相对较小，可以正常运行
- 随机权重模式下生成内容无意义（正常现象）
- TTFT/TPOT 较高因为是 encoder-decoder 架构

## 已评测的 Seq2Seq 模型

| 模型 | 设备 | 延迟 | TTFT | TPOT |
|------|------|------|------|------|
| google/flan-t5-base | npu:2 | 5.83s | 870.78ms | 870.78ms |
