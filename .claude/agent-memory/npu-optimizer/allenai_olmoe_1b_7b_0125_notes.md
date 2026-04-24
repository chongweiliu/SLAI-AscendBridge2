# allenai/OLMoE-1B-7B-0125

日期：2026-04-22
状态：stage3 pending，继续推进中

## 当前结论

- `board.db` 脏 row 使用 `model_id="allenai/OLMoE"`，但真实仓库是 `allenai/OLMoE-1B-7B-0125`；当前 adaptation 目录为 `adaptations/allenai_olmoe_1b_7b_0125`
- stage1 / stage2 已修通并 completed；当前只处理 stage3
- stage3 completed gate 必须使用 `pretrained`，config-only 结果只能用于诊断
- `accuracy_run_perf.py` 现已支持：
  - baseline/perf 两路工件
  - `output_compare`
  - `optimization_notes.json`
  - `selected_npu(s)` / `device_topology` / `parallel_mode`
  - `--enable-model-patch`
  - `--patch-scope {mlp_only,experts_only,both}`
- `compare` 生成 notes 时必须从 perf 工件继承选卡元数据，不能依赖当前 shell 的 `ASCEND_RT_VISIBLE_DEVICES`

## 下载 / 权重侧经验

- 当前环境访问 HuggingFace 容易遇到 TLS reset / connection reset，不能指望在线 `from_pretrained()` 自行补齐
- `modelscope` 可用，且必须把产物落在 adaptation 内：
  - `models/olmoe_pretrained_snapshot/`
- `pretrained_snapshot_ready()` 现在要求：
  - `config.json`
  - `model.safetensors.index.json`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - index 中声明的全部 safetensors shard
- snapshot 未完整时，`--use-pretrained` 必须直接失败，不能 silent fallback 到 config

## Patch 侧经验

- OLMoE 当前 transformers 实现的热点：
  - `OlmoeExperts.forward`: `gate_up_proj -> chunk(2) -> SiLU(gate) * up`
  - `OlmoeDecoderLayer` 使用 `OlmoeRMSNorm`
- 当前已实现 `npu_swiglu` patch：
  - `OlmoeMLP.forward`
  - `OlmoeExperts.forward`
- 但 config-mode 下同时 patch `OlmoeMLP + OlmoeExperts` 精度明显异常，不能直接相信这一路
- 后续 pretrained 试验顺序：
  1. runtime_only
  2. `patch_scope=mlp_only`
  3. `patch_scope=experts_only`
  4. `patch_scope=both`
- 单模型正式测速必须保持：
  - 同一卡
  - 同映射
  - 串行 baseline/perf
  - 目前固定使用 `ASCEND_RT_VISIBLE_DEVICES=12`

## 写库前置条件

- 先通过：
  - `benchmark/scripts/check_accuracy_run.py --adapt allenai_olmoe_1b_7b_0125`
  - `optimization/scripts/check_accuracy_run_perf.py --adapt allenai_olmoe_1b_7b_0125`
  - `optimization/scripts/check_optimization_notes.py --adapt adaptations/allenai_olmoe_1b_7b_0125`
- 只有拿到真实 pretrained、`num_samples >= 50`、`speedup_ratio > 1.0` 且精度证据过线后，才允许用 `board_ops.py update_optimization_status ... completed`
