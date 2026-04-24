# nasa-cisto-data-science-group/satvision-toa-giant-patch8-window8-128

- 日期：2026-04-23
- 结论：`completed`
- 路线：标准 `timm.models.swin_transformer_v2.SwinTransformerV2` + 本地 DeepSpeed checkpoint 重映射 + runtime-only batching

## 关键模式

- 不要继续沿用旧的 config/dry-run `swinv2_cr_giant_224` 路线。这个 checkpoint 实际更贴近标准 `SwinTransformerV2`，不是 `timm` 的 `swinv2_cr` 变体。
- 可行结构：
  - `img_size=128`
  - `patch_size=4`
  - `in_chans=14`
  - `embed_dim=512`
  - `depths=(2, 2, 42, 2)`
  - `num_heads=(16, 32, 64, 128)`
  - `window_size=8`
  - `num_classes=0`
- DeepSpeed checkpoint `mp_rank_00_model_states.pt` 可直接复用，只需要做 key remap：
  - 去掉前缀 `encoder.`
  - 忽略 `mask_token`
  - 忽略 `decoder.*`
  - `layers.{i}.downsample.* -> layers.{i+1}.downsample.*`
- 这样可直接对上 `831` 个模型参数中的 `822` 个；剩余未命中主要是非持久 buffer 与 stage 边界 downsample 索引问题。修正 downsample 偏移后可 `strict=False` 无 missing/unexpected。

## 第三阶段合同

- baseline：
  - `accuracy_run.py --use-pretrained --max-samples 50`
  - 输出 `embeddings`
  - 逐样本跑，得到可比较 baseline wall-clock
- perf：
  - `accuracy_run_perf.py run --use-pretrained --max-samples 50 --batch-size 4`
  - runtime-only：`warmup(3x) + TASK_QUEUE_ENABLE=1 + batched_inference`
- compare：
  - `accuracy_run_perf.py compare`
  - `output_compare` 显式写 `baseline_samples/perf_samples/cuda_samples/ascend_samples=50`
  - `precision_method=embedding_cosine_compare`
  - `comparison_method=independent_baseline_artifact`
  - `comparison_scope=steady_state`

## 实测结果

- 物理卡：`ASCEND_RT_VISIBLE_DEVICES=13`
- baseline：
  - `wall_clock_s=2.693628`
  - `latency_s=0.053873`
  - `num_samples=50`
- perf：
  - `wall_clock_s=1.373225`
  - `latency_s=0.027465`
  - `num_samples=50`
- speedup：
  - `1.961534x`
- 精度：
  - `cosine_similarity=1.0`
  - `max_abs_error=1.91e-06`

## Gate 注意点

- `accuracy_run.py` 静态 checker 会抓 `images[0]`，即使运行时样本不为空，也最好改成 `images[0] if images else fallback`。
- `embeddings` 属于 vector family，`perf` 工件和 `optimization_notes` 必须提供：
  - `cosine_similarity >= 0.999`
  - `max_abs_error < 1e-3`
  - `latency_s == wall_clock_s / num_samples`
- 旧 `config` 工件必须先删，否则 benchmark checker 会把历史 `num_samples=1` 工件也扫进去。
