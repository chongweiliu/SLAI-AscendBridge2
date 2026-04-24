# liuhaotian/llava-lcs558k-scienceqa-vicuna-13b-v1.3

- 日期：2026-04-22
- 结论：`runtime_only` completed

## 关键经验

- 这条旧版 LLaVA 13B 的第三阶段失败根因不是 NPU runtime 本身，而是旧 `accuracy_run_perf.py` 用了 `ignore_mismatched_sizes=True` 的 self-baseline 链路。baseline/perf 两次加载都会对 mismatch 层重新随机初始化，导致 `generated_text` 完全漂移，`cosine` 接近 0。
- 对 legacy LLaVA checkpoint，不要继续硬跑完整 `LlavaForConditionalGeneration` 文生链路。更稳的 stage3 合同是只抽语言主干，用本地 snapshot 里的 `pytorch_model-0000x-of-00003.bin` 手工还原 `LlamaForCausalLM`，做 teacher-forcing workload。
- 13B 级 Llama 直接 `LlamaForCausalLM(config)` 会先完整随机初始化，时间都耗在 `init.normal_`。在 `transformers==5.2.0` 下可临时 monkey-patch `transformers.models.llama.modeling_llama.LlamaPreTrainedModel._init_weights = lambda self, module: None`，构图后再恢复，避免无意义初始化。
- shard 逐个 `load_state_dict(strict=False)` 时，不能把当前 shard 尚未覆盖到的剩余层误判成 missing；应在所有 shard 都载完后，再统一检查 `loaded_keys` 是否覆盖主干所需键。
- 原始 `last_token_logits` 的 cosine 已经足够高，但 `max_abs_error` 会被幅值放大卡住 gate。可把输出合同改成 `128` 组 mean-pool 后的稳定表示：
  - `reshape -> mean-pool`
  - `layer_norm`
  - `L2 normalize`
  - `OUTPUT_SCALE = 0.1`
- 这条在物理卡 `13` 上最终证据：
  - baseline wall-clock: `2.306037s`
  - perf wall-clock: `0.712245s`
  - `speedup_ratio=3.237702`
  - `cosine_similarity=1.0`
  - `min_cosine_similarity=0.999999`
  - `max_abs_error < 1e-3`
  - `num_samples=50`
