# TF-to-PyTorch Checkpoint Conversion

## TF-Only Models: Direct h5py Conversion

当模型只有 TF checkpoint（`tf_model.h5`）而没有 PyTorch 版本时，可使用 `h5py` 直接读取 TF 变量并转换为 PyTorch state_dict，无需安装 TensorFlow。

### 关键文件

- `convert_tf_to_pytorch.py`：自定义转换脚本，放在 adaptation 目录下
- 转换后权重保存到：`models/models--{org}--{model}/snapshots/{hash}/pytorch_model.bin`

### T5 权重转换经验（lblueee/t5-academic-title-generator-model）

**TF checkpoint 路径格式**：
```
decoder/tft5_for_conditional_generation/decoder/block_._N/layer_._M/...
```

**形状映射规律**（T5 small/base，12层，768hidden）：

| 权重类型 | TF 形状 | PyTorch 形状 | 是否转置 |
|---------|--------|------------|---------|
| 词嵌入 | (32128, 768) | (32128, 768) | 否 |
| q/k/v/o attention | (768, 768) | (768, 768) | 否 |
| wi (FFN up) | (768, 3072) | (3072, 768) | 是 |
| wo (FFN down) | (3072, 768) | (768, 3072) | 是 |
| relative_attention_bias | (32, 12) | (32, 12) | 否 |
| final_layer_norm | (768,) | (768,) | 否 |

**关键发现**：
- T5 的 embeddings 和 q/k/v/o 不需要转置（和 typical transformer 不同）
- 只有 FFN 的 wi/wo 需要转置
- `relative_attention_bias` 形状是 (num_buckets, num_heads)，不需要转置

### accuracy_run.py 中的加载逻辑

```python
SNAPSHOT_DIR = Path(cache_dir) / f"models--{MODEL_ID.replace('/', '--')}" / "snapshots" / "3bdef036a9f228b11d404da164c3504375714eb6"
if use_pretrained:
    loaded = False
    # 1. Try snapshot directory (converted PyTorch weights)
    if SNAPSHOT_DIR.exists():
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(str(SNAPSHOT_DIR), trust_remote_code=True, torch_dtype="auto")
            model = model.to(device); loaded = True
```

### 验证方法

1. 对比原始 TF 权重形状和 PyTorch 权重形状
2. 运行推理并对比 cls_embeddings cosine similarity
3. 确保 text match rate = 1.0

### transformers 版本兼容性

- transformers 5.x 移除了 `from_tf` 支持
- transformers 4.44.2 仍支持（但无 `from_tf`）
- 如果需要从 TF 加载，必须用 h5py 直接转换

### measurement_contract_version 3.0 注意事项

当使用 measurement_contract_version 3.0 时：
- artifact 中 `latency_s` = total_batch_latency（wall_clock）
- `per_sample_latency = latency_s / num_samples`
- optimization_notes.json 中的 `baseline_latency_s` 和 `perf_latency_s` 必须与 artifact 中的 per-sample 值一致
- `baseline_wall_clock_s = baseline_latency_s * num_samples`
- `perf_wall_clock_s = perf_latency_s * num_samples`
- **常见错误**：`best_result` 和 `results[0]` 中的 latency 值未同步更新，导致 validator 报 `baseline_artifact.latency_s != optimization_notes.baseline_latency_s`

## 其他模型转换注意事项

### BERT 类模型

通常不需要转置，因为 TF 和 PyTorch 使用相同的 weight shape 约定。

### LLaMA 类模型

- 词嵌入：通常需要转置 (vocab, hidden) → (hidden, vocab)
- q/k/v/o：通常不需要转置
- 参见具体模型转换脚本
