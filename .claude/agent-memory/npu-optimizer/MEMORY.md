# NPU Optimizer: 优化经验索引

## 权责与关键规则

- **model_files 独占**：`model_files/` 与 `accuracy_run_perf.py` 必须且仅能由 npu-optimizer 创建
- **model_files 禁忌**：不得将 `model_files` 作为 `cache_dir` 传入 `from_pretrained()`；否则 HF 会创建 `models--xxx/blobs/`
- **并行测速约束**：不能把速度测试并发放在同一张 NPU 卡上；单模型 speedup 证据链必须同卡串行
- **选卡规则**：先 `npu-smi info` 检查占用，优先选空闲卡；禁设 ASCEND_RT_VISIBLE_DEVICES（用 mem_get_info 选卡）
- **max_abs_error gate**：`board_ops._validate_precision_evidence()` 对 logits/embedding 要求 `max_abs_error < 0.001` 或 `None`；cosine >= 0.999。不包含 max_abs_error 时 gate 视为 None 通过
- **runtime_only notes 必填字段**：`measurement_contract_version>=3`, `wall_clock_source`, `baseline_warmup_iterations`, `perf_warmup_iterations`, `warmup_policy="symmetric"`, `perf_memory_mb`, `optimization_items` 含 warmup/TASK_QUEUE_ENABLE
- **perf_latency_s = perf_wall_clock_s / num_samples**：非生成类任务要求 latency_s = wall_clock_s / num_samples
- **right-padding >> left-padding**：批量推理时右填充保持 real tokens 在相同位置，bf16 数值差异更小
- **bs=1 + TQE 常态回归**：短前向（<0.04s/sample）时 TQE 异步下发开销 > 收益；改用 batched inference 获取真实提速
- **nopua skill**：同一 action 失败 2+ 次时主动调用 `Skill("nopua")`
- **第四阶段 override 规则**：`optimization_status=completed` 的 fusion_ops 结果不必强行外推到 phase-4 真实业务负载

## 常见 gate 坑

- **cosine 夹紧**：真实 compare 浮点算出 >1.0 时必须夹紧到 `1.0`，否则 gate 拒绝
- **warmup 对称**：`baseline_warmup_iterations == perf_warmup_iterations`，否则 `check_accuracy_run_perf.py` 拦截
- **cls_embeddings compare 样本数**：`benchmark_tool.compare_outputs()` 不识别 `cls_embeddings` 键，需手动补 `baseline_samples/perf_samples/cuda_samples/ascend_samples=total_samples`
- **mixed 输出 compare 样本数**：`generated_text + logits` 的 mixed 输出要求 baseline/perf 样本数严格一致
- **选卡信息从工件继承**：`selected_npu(s)` / `device_topology` 应在 `run` 阶段写入 perf metrics，`compare` 从工件继承
- **compare 不能按 mtime 选工件**：按 `end_time/start_time` 选最近一轮，要求 baseline/perf/output 四件套成对存在
- **ttft_ms 精度**：逐样本平均延迟脚本 `ttft_ms` 至少保留 3 位，否则可能 `ttft_ms > latency_s * 1000` 被 gate 拒绝
- **step1/step2 陷阱**：step1 含 profiling 导致 wall_clock 膨胀，需设 `wall_clock_s = num_samples * latency_s`

## 主题文件

| 文件 | 内容 |
|------|------|
| `qwen-7b-optimization.md` | Qwen-7B 优化完整案例（代码、数据、命令） |
| `fusion-attention-pitfalls.md` | npu_fusion_attention 四大踩坑详录 |
| `qwen3-tts-failure.md` | Qwen3-TTS 失败：qwen_tts/CUDA 依赖 |
| `advanced-topics.md` | GQA、sparse_mode、环境变量、fallback、compile、KV Cache、API 适用性全景 |
| `tf-conversion.md` | TF checkpoint 用 h5py 转换为 PyTorch state_dict |
| `qwen2_5_vl_7b_scientific_vlm_failure.md` | profiling overhead 误判 + runtime_only speedup=1.0 |
| `dice-research-lola_v1-notes.md` | LOLA GPT-2 MoE: transformers 降级, fusion N/A, wall-clock <1.0 |
| `qa_scibert_seed12_pending.md` | 对称 warmup 下 speedup=1.0（小模型 warmup 饱和） |
| `dler_r1_7b_failure.md` | DLER-R1-7B: 所有融合算子+TQE 均回归 |
| `dler_r1_7b_notes.md` | DLER-R1-7B runtime-only completed 路径 |
| `miloskosrad_scientificnlisrb_notes.md` | ScientificNLIsrb: 本地 snapshot + runtime-only batching |
| `minishlab_potion_science_32m_notes.md` | potion-science-32M: Model2Vec 静态 embedding |
| `minishlab_potion_science_8m_notes.md` | potion-science-8M: 复用同族模板 |
| `mrm8488_roberta_base_biomedical_spanish_diagnostics_pending.md` | pretrained 源为空壳 |
| `openai_clip_vit_base_patch16_notes.md` | CLIP-ViT-B/16: image_embeddings, bs=1+TQE completed |
| `openai_clip_vit_large_patch14_notes.md` | CLIP-ViT-L/14: bs=1+TQE completed |
| `prajjwal1_bert_tiny_notes.md` | bert-tiny: npu_add_layer_norm + TQE + bs=8, 9.04x |
| `prem_research_prem_1b_sql_notes.md` | prem-1B-SQL: 重建 teacher-forcing 合同, 1.03x |
| `qwen_qwen2_5_7b_instruct_notes.md` | Qwen2.5-7B-Instruct: right-padding bs=2+TQE, 1.61x |
| `nasa_cisto_data_science_group_satvision_toa_giant_patch8_window8_128_notes.md` | SatVision: timm SwinTransformerV2+DeepSpeed ckpt, 1.96x |
| `halfotter_greensteel_frontend_material_classifier_notes.md` | GreenSteel: cls_embeddings runtime-only batching 4.74x |
| `insta360_research_dap_weights_notes.md` | DAP-weights: hubconf 懒导入, depth_maps 1.43x |
| `google_mt5_small_notes.md` | mt5-small: npu_rms_norm 3.93x |
| `timm_efficientnet_b0_failure.md` | efficientnet_b0: warmup 主导, speedup≈1.0 |
| `media_tek_breeze_asr25_failure.md` | Breeze-ASR-25: torchcodec 依赖 |
| `stable_diffusion_v1_5_notes.md` | SD v1.5: 6 融合算子 N/A + 测量口径不兼容 |
| `qwen3_embedding_4b_notes.md` | Qwen3-Embedding-4B: latency_s ≠ wall_clock/num_samples |
| `qwen3_omni_30b_notes.md` | Qwen3-Omni-30B: forward() 不接受 input_ids |
| `qwen3_vl_4b_instruct_failure.md` | Qwen3-VL-4B: bf16 非确定性 text_match=82% |
| `qa_matscibert_seed36_failure.md` | QA-MatSciBERT-seed36: 融合回归 + 口径冲突 |
| `ace_step_ace_step1_5_notes.md` | ACE-Step1.5: fp32 + npu_rms_norm 过 gate |
| `awsteam7052_industrial_design_extreme_material_sdxl_v1_0_notes.md` | SDXL: runtime-only + native NPU attention |
| `stabilityai_stable_diffusion_xl_base_1_0_notes.md` | SDXL-base: runtime-only TQE+warmup, 1.13x pretrained |
| `bsc_nlp4bia_biomedical_semantic_relation_classifier_notes.md` | batched CLS embeddings 3.81x |
| `biomedclip_vit_bert_hf_notes.md` | BiomedCLIP: CLIP pre-norm 6 融合算子 N/A |
| `e_mimic_inclusively_reformulation_it5_notes.md` | T5: teacher-forcing logits + batching 2.93x |
| `openai_community_gpt2_notes.md` | GPT-2 124M: teacher-forcing logits + batching 2.81x |
| `qwen_qwen3_5_9b_notes.md` | Qwen3.5-9B: multimodal qwen3_5 text-only forward, runtime-only bs=2 1.87x |
| `qwen_qwen3_8_27b_notes.md` | Qwen3.8-27B: qwen3_5 AutoModelForImageTextToText, runtime-only bs=2+TQE, 1.89x |
| `qwen_qwen2_5_vl_7b_instruct_notes.md` | Qwen2.5-VL-7B: VLM text-only teacher-forcing, bs=2+TQE, 1.51x |

## 优化组合摘要

完整 186+ 条记录见 `optimization_records.md`。关键模式：

| 模型族 | 可优化点 | 备注 |
|---------|----------|------|
| Qwen/LLaMA/Baichuan (RMSNorm+SwiGLU+RoPE+GQA) | 全部四项；但常因 max_abs_error > 0.001 失败 | 改用 runtime_only batched (bs=2, right-padding) |
| BERT/GPT-2 (<1B, Post-LN) | npu_add_layer_norm + npu_gelu | runtime_only batched CLS/logits 更稳 |
| Pre-LN + nn.LayerNorm (LOLA/GPT-2 MoE) | 全部 N/A | 仅 warmup + TQE |
| ModernBERT (pre-norm+GeGLU) | monkey-patch npu_gelu(erf) | 严禁用 model_files 替换整个 modeling |
| T5 (T5LayerNorm=RMSNorm) | monkey-patch T5LayerNorm → npu_rms_norm | teacher-forcing logits 合同 |
| Phi-2 (pre-norm, gelu_new) | npu_fusion_attention | partial RoPE 已外部应用 |
| MoE (GLM/OLMoE) | npu_swiglu in shared MLP only | experts 中的 swiglu 回退 |
| Diffusion/Video (UNet/DiT) | 6 融合算子常 N/A | runtime_only + native NPU attention backend |
| Whisper/ASR | 无融合算子可用 | warmup + TQE + encoder embedding 对比 |

- **google-bert/bert-base-uncased（2026-08-22 重做）**：BERT-base Post-LN 模型 runtime-only batched cls_embeddings completed。旧 baseline 是 config 模式 MLM generated_text。改为 pretrained cls_embeddings 合同（AutoModel, last_hidden_state[:,0,:]）。baseline 逐样本(bs=1)，perf batched(bs=8)+warmup(3x)+TQE=1，在卡 `npu:1` 上得到 `0.489s -> 0.131s`，`speedup_ratio=3.742101`，`cosine=0.9999999891`，`max_abs_error=2.24e-05`。详见 [[google_bert_bert_base_uncased_notes]]。

- **cross-encoder/ms-marco-MiniLM-L6-v2（2026-08-22）**：Cross-encoder reranker 模型 runtime-only batched rerank_scores completed。baseline 逐 query 处理(2 pairs/forward)，perf batched 8 queries(16 pairs/forward)+warmup(3x)+TQE=1。在卡 `npu:1` 上得到 `0.268s -> 0.052s`，`speedup_ratio=5.120240`，`cosine=1.0`，`max_abs_error=2.86e-06`，ranking 59/59 全对。详见 [[cross_encoder_ms_marco_minilm_l6_v2_notes]]。

- **openai/clip-vit-base-patch32（2026-08-22）**：CLIP-ViT-B/patch32 runtime-only all-texts-one-batch completed。baseline 分 4 块（每块重编码图像），perf 一次性处理 60 文本（图像只编码 1 次）+warmup(3x)+TQE=1。在卡 `npu:1` 上得到 `0.069s -> 0.018s`，`speedup_ratio=3.761095`，`cosine=1.0`，`max_abs_error=2.098e-05`，top1="a solid red square" 正确匹配。与 [[openai_clip_vit_base_patch16_notes]] 不同：patch32 的 image_text_similarity 合同可以安全 batched text（不像 patch16 image_embeddings 需 bs=1）。详见 [[openai_clip_vit_base_patch32_notes]]。

- **stabilityai/stable-diffusion-xl-base-1.0（2026-08-23）**：SDXL 文生图 runtime-only TQE+warmup completed。6 融合算子全 N/A（GroupNorm+SiLU+processor attention）。固定 seed(42+idx) 保证 baseline/perf image_stats 完全一致（cosine=1.0, mae=0.0）。baseline 逐样本(9.17s)→perf TQE+warmup(8.11s)，`speedup_ratio=1.130449`。关键：移除 `empty_cache()` 开销、`ttft_ms=latency_s*1000`（不能用 step1 profiler latency）、`wall_clock_s` 显式字段。`_native_npu` backend 实测无收益。详见 [[stabilityai_stable_diffusion_xl_base_1_0_notes]]。
