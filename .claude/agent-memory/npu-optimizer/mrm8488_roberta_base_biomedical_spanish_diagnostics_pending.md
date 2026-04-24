# mrm8488/roberta-base-biomedical-spanish-diagnostics

## 当前结论

- 状态：保持 `optimization_status=pending`
- 根因：pretrained 源为空壳，无法生成合法 `--use-pretrained` baseline/perf

## 本地证据

- adaptation 私有缓存：
  - `models/models--mrm8488--roberta-base-biomedical-spanish-diagnostics/refs/main`
  - 指向 `46e5ec3c3c3d327541f6014a61c07bb6f036c80e`
  - 但 `snapshots/46e5ec3c3c3d327541f6014a61c07bb6f036c80e/` 不存在
- adaptation 私有 `.no_exist/<sha>/` 下的这些文件全是 `0` 字节：
  - `config.json`
  - `tokenizer_config.json`
  - `vocab.json`
  - `merges.txt`
  - `tokenizer.json`
  - `pytorch_model.bin`
  - `model.safetensors`
- 全局 `~/.cache/huggingface/hub/models--mrm8488--roberta-base-biomedical-spanish-diagnostics/` 也是同样结构，只有 `.gitattributes` 真正存在。

## 影响

- `accuracy_run.py --use-pretrained` 无法建立可信的 pretrained 基线。
- 旧 benchmark/optimization 产物都是 `config` 或空壳缓存背景下生成的，不符合 stage-3 completed 口径。

## 处理原则

- 不允许拿 `config` 结果冒充 pretrained completed。
- 在拿到合法 pretrained 权重前，不继续写 completed。
- 记录结构化 `pending` 原因后，继续处理后续 pending 模型。
