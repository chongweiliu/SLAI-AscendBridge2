# ibm-research/materials.mhg-ged

- 日期：2026-04-22
- 结论：`optimization_status=completed`
- 路线：`runtime_only`
- 物理卡：`ASCEND_RT_VISIBLE_DEVICES=13`

## 关键修复

- 旧 stage3 的根因不是“pretrained 不可恢复”，而是脚本把 `pytorch_model.bin` 当普通 `state_dict` 读，直接被缺失私有模块 `mhg_model` 卡死，然后退回了 config-only mock MLP。
- 用最小 stub module 占位 `mhg_model.graph_grammar.graph_grammar.hrg`、`mhg_model.graph_grammar.hypergraph`、`mhg_model.graph_grammar.graph_grammar.symbols`、`mhg_model.graph_grammar.graph_grammar.corpus`、`mhg_model.graph_grammar.algo.tree_decomposition` 后，可以从本地 snapshot 解出真实 checkpoint payload：
  - `num_features=9`
  - `num_edge_features=3`
  - `max_length=92`
  - `gnn_params.encoder_params.hidden_channels=256`
  - `gnn_params.encoder_params.proximity_size=3`
  - `model_state_dict`
- 正式修法是在 adaptation 内新增 `mhg_runtime.py`，重建最小可运行 encoder：
  - `trans.embedding_list[9]`
  - `mlist[3]` message-passing blocks
  - `hidden2mean`
  - `hidden2logvar`
- decoder / vocab 相关 checkpoint 键允许作为 `unexpected` 前缀忽略：
  - `tgt_embedding`
  - `decoder`
  - `latent2tgt_emb`
  - `latent2hidden_dict`
  - `dec2vocab`
- baseline/perf 都改为真正的 `pretrained` 工件，不再使用旧 `self_baseline_same_model`、不再写 config-only completed。

## 正式合同

- baseline：
  - `benchmark_metrics_npu_0_fp32_pretrained_builtin_smiles.json`
  - `outputs_npu_0_fp32_pretrained_builtin_smiles.pt`
  - `batch_size=1`
  - `warmup(3x)`
- perf：
  - `benchmark_metrics_npu_0_fp32_pretrained_builtin_smiles_perf.json`
  - `outputs_npu_0_fp32_pretrained_builtin_smiles_perf.pt`
  - `batched_pretrained_mhg_embeddings(bs=10)`
  - `warmup(3x)`
  - `TASK_QUEUE_ENABLE=1`
- compare：
  - `output_compare_perf.json`
  - `comparison_method=independent_baseline_artifact`
  - `precision_method=cosine_similarity`

## 结果

- baseline wall-clock：`0.106405s`
- perf wall-clock：`0.010934s`
- `speedup_ratio=9.731571`
- `forward_latency_speedup_ratio=9.716895`
- `num_samples=50`
- `cosine_similarity≈0.999999969`
- `min_cosine_similarity≈0.999999821`
- `max_abs_error=0.00018310546875`

## Gate 注意事项

- `accuracy_run.py` 的 `--max-samples` 默认值必须保留 `250`，否则 `check_accuracy_run.py` 静态规则会拦。
- embedding runtime-only notes 当前必须写：
  - `warmup_policy="symmetric"`
  - `baseline_warmup_iterations == perf_warmup_iterations`
- `selected_npu(s)`、`device_topology`、`parallel_mode` 要从工件继承并回写到 notes。
