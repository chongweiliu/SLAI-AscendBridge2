# Completed Gate 字段要求 与 seq2seq (T5) 模板改写详解

（从 MEMORY.md 索引展开的细节文档；2026-08-22 由 google-t5/t5-small 实跑验证）

## 1. completed gate 与模板的字段缺口（board_ops 强制）

`check_accuracy_run.py --adapt {name}` 会模拟 completed gate
（board_ops._validate_benchmark_metric_artifacts），对 NPU baseline
`benchmark_metrics*.json` 要求：

- `num_samples >= 50`（即使 team-lead 说"10 个样本即可"，要让 `--adapt`
  检查通过也必须跑 >=50；小模型直接跑 60 即可）
- 必含字段：`latency_s`(>0)、`num_samples`、`mode`(pretrained/config)、
  `dataset`、`dtype`(fp32/fp16/bf16)、`output_type`、`device`(含 npu)、
  `start_time`、`end_time`(晚于 start)
- `ttft_ms`/`tpot_ms` 必须为数值或 null；`ttft_ms <= latency_s*1000`
- **注意**：benchmark-script 模板写的 metrics 字典**缺少** `dataset` 和
  `dtype` 两个字段，手动补上（`"dataset": dataset_name`、`"dtype": dtype_str`）
- `num_samples` 应写实际参与统计的样本数 `min(len(texts), max_samples)`，
  而非全量 `len(texts)`
- 若 `--max-samples` 超过内置样本数，内置文本要备够（如 66 条），
  或接受按实际条数统计
- 文件名含 "cuda"/"perf" 的工件不参与 baseline gate

## 2. seq2seq (T5 等) 模板改写要点

模板的 `model(**inputs)` 对 encoder-decoder 会报错：

```
ValueError: You have to specify either decoder_input_ids or decoder_inputs_embeds
```

### 实跑验证过的方案（google-t5/t5-small，NPU fp32）

- **Step 1**: 用 `model.generate(**inputs, max_new_tokens=64, do_sample=False,
  num_beams=1, pad_token_id=tokenizer.pad_token_id)` 作为被 trace 的负载
- **Step 2**: 先非流式 greedy generate 拿 `generated_ids`（确定性），再
  `model(**inputs, decoder_input_ids=generated_ids[:, :-1])` 提取最后 token
  logits；PPL 以生成序列自身为目标 `CE(logits[0], generated_ids[:, 1:])`
- **TTFT/TPOT**: 第二次生成用 TextIteratorStreamer 线程流式测量（greedy
  两次结果一致）；退化序列（<2 token）用零向量 logits + nan ppl 占位
- T5 输入必须带任务前缀（"translate English to German: ..." /
  "summarize: ..."），内置样本按翻译/摘要画像准备（66 条内置样本足够跑 60）
- `torch_dtype="auto"` 下 T5 权重为 fp32，gate 的 dtype 白名单
  （fp32/fp16/bf16）可通过

### 替代方案（旧记录）：仅用 decoder_start_token_id 做单步 forward

```python
decoder_start_token_id = model.config.decoder_start_token_id
decoder_input_ids = torch.full((1, 1), decoder_start_token_id, dtype=torch.long, device=first_device)
forward_inputs = {k: v for k, v in inputs.items()}
forward_inputs["decoder_input_ids"] = decoder_input_ids
logits_output = model(**forward_inputs)
last_token_logits = logits_output.logits[0, -1, :].cpu()
```

### NPU OOM 处理

如果 NPU 上 OOM，改用 `--cpu` 参数在 CPU 上运行
（注意：契约若禁止 CPU 则先减 `--max-samples` / `max_length`）。

## 参考

- benchmark-runner.md 2.10 禁用手册 / 模板强制检查
- benchmark-script/SKILL.md 9.4 手动编写规范、9.5 常见错误
- dataset-mapping/SKILL.md 4.1 数据集加载方式
- scripts/board_ops.py `_validate_benchmark_metric_artifacts` /
  `_REQUIRED_COMPLETED_BENCHMARK_FIELDS`
