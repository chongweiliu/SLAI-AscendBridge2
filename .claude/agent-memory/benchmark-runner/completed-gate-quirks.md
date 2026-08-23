# completed gate 与本机 benchmark 要点（2026-08-22，gpt2 实测）

## `check_accuracy_run.py --adapt {name}` 隐含 completed gate

`--adapt` 不仅做静态规则检查，还会模拟 `benchmark_status=completed` 门禁
（`board_ops._validate_benchmark_metric_artifacts`）：

1. 必须已存在 `benchmark_metrics_*.json`（名字不含 cuda/perf），且 device 含 `npu`；
2. 必填字段：`latency_s, num_samples, mode, dataset, dtype, output_type, device, start_time, end_time`
   —— 注意 **`dataset` 与 `dtype` 两个字段在模板 metrics dict 里没有**，手写/改模板时必须补上；
3. `num_samples >= 50`（即使 team-lead 说"跑 10 个就行"，10 个也会卡在门禁；需向 team-lead 说明后按 >=50 跑）；
4. `latency_s > 0`、`end_time >= start_time`、`ttft_ms <= latency_s*1000`、`tpot_ms >= 0`。
   为避免 ttft 大于 step1 单样本 forward 延迟，最终 `latency_s` 用 Step2 全样本平均
   端到端延迟覆盖（step1 值保留为 `step1_forward_latency_s` 诊断字段）。

## 本机（2×Ascend910）执行要点

- 模板 `get_device()` 默认选 `npu:0`、pretrained 分支用 `device_map="auto"`，
  本机多 runner 并发时必须改：`mem_get_info` 选空闲卡 + `torch.npu.set_device()`；
  `device_map="auto"` 去掉，小模型直接 `model.to(device)`。
- 项目根 `datasets/` 常为空 → 落到内置文本兜底（dataset_name=`builtin`，
  checker 白名单已含 `_builtin`）；内置文本需扩到 >=50 条才能过门禁。
- transformers 5.15.1 + torch 2.8.0+cpu + torch_npu 2.8.0.post4 组合对
  GPT-2 CausalLM 路线（streamer 测 TTFT/TPOT + NPU profiler trace）完全可用。
- gpt2 实测：55 样本（每样本 forward+64 token 流式生成）约 40s；
  latency_s≈0.62s/样本，TTFT≈15ms，TPOT≈9ms，peak≈497MB HBM。

## embedding（句向量）模型路线（2026-08-22，all-MiniLM-L6-v2 实测）

- `dataset_mapping` 返回 `model_type=embedding` → 不走模板，手写（bert 分支骨架）。
- 池化用 demo.py 同款：mean pooling + L2 normalize（sentence-transformers 语义），
  不是 CLS；但 metrics `output_type` 仍写 `cls_embeddings`（benchmark_tool 词表）。
- outputs .pt 用 dict：`{"texts": [...], "embeddings": [tensor(1, dim)...], "similarity_profile": {...}}`；
  `detect_output_type` 按 `"embeddings"` 键识别为 cls_embeddings，compare 走
  `compare_cls_embeddings`（flatten 后逐样本余弦，[1, dim] 与 [dim] 均可）。
- 语义画像（team-lead 要求）：固定 3 组 (anchor, paraphrase, unrelated) 三元组，
  计算余弦 + margin，pretrained 下断言所有 margin>0；config 模式只验前向路径不断言。
  实测 all-MiniLM-L6-v2 margins ≈ 0.87/0.71/0.90。
- 非生成式：ttft_ms/tpot_ms 写 null（gate 允许），latency_s 用 Step2 每样本编码延迟均值。
- 实测：60 样本 ~4.5ms/样本，peak≈103MB HBM，全程 ~40s（含模型加载与画像）。

## HF 镜像 env 必须在 import 前设置（2026-08-22，Qwen2.5-1.5B 实测）

- 模板本身**不设置** `HF_ENDPOINT`；手写/改模板时必须把
  `os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")` 与
  `HF_HUB_DISABLE_XET=1` 放在 `import transformers` **之前**（huggingface_hub
  在 import 时固化 endpoint）。放晚了 → `from_pretrained`/tokenizer 直连
  huggingface.co 超时（本机不通外网），浪费整轮重试。
- 边界契约（副作用仅限 adaptation_path）下的数据集方案：把
  `Salesforce/wikitext`（config `wikitext-2-raw-v1`, split=test，4358 行/2891
  非空）下载保存到 `adaptations/{name}/datasets/wikitext___wikitext-2-raw-v1`
  （`load_dataset(...).save_to_disk(...)`），`load_benchmark_texts()` 按
  「本地 datasets/ → 项目根 datasets/（只读）→ 内置」顺序查找。
- Qwen2.5 config 的 `torch_dtype=bfloat16` → `from_config` 直接产 bf16 模型，
  工件名即 `*_npu_bf16_config_*`（dtype_str 必须按模型实际 dtype）。
- Qwen2.5-1.5B-Instruct config 模式实测（npu:1 单卡，50 样本）：总耗时约 2min；
  latency_s≈2.12s/样本（forward+64token），TTFT≈41ms，TPOT≈32ms，peak≈2966MB HBM；
  NPU profiler trace 33MB/38350 事件（含 async_npu），export_chrome_trace 为
  JSON **数组**（`[...]`）而非对象。
