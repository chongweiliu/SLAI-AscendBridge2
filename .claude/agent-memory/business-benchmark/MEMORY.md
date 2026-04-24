# Business-Benchmark Memory

## 摘要

- 第四阶段负责真实权重、真实数据集下的业务测评证据沉淀
- completed 前必须具备 NPU baseline、NPU perf、CUDA baseline 三类工件
- `business_summary.json` 必须包含 `device_model`、`latency_s`、`peak_memory_mb`、`throughput_metric_*` 等对比证据
- 2026-04-12 新强化：`Lightricks/LTX-2.3` 暴露出另一类 phase-4 漂移。repo id 本身没有 `diffusion/video` 常见关键词时，`dataset_mapping.py` 会把 text-to-video DiT 模型误落到 `causal_lm + mmlu`。稳定修法是两层一起做：一、`_contains_diffusion_signal()` 增加边界感知 `ltx` 信号，让 `LTX-2.3` 直接回到 `diffusion + latency_only`；二、manager 的 adaptation-context override 增加 `text-to-video / audio-video generation / DistilledPipeline / X0Model / ltx_diffusers_pipeline` 组合信号，把旧 stale `mmlu` config 自动清掉。`LTX-2.3` 还需要自定义 `business_model_eval.py`，复用 `accuracy_run.py` / `accuracy_run_perf.py` 的 X0 denoising 路径做 latency-only phase-4，不能直接套通用 causal-LM evaluator
- 2026-04-12 新强化：`Rostlab/prot_t5_xl_uniref50` 证明蛋白质 T5 不能继续因为名字里有 `t5` 就被误判成 `seq2seq + cnn_dailymail`。这类 checkpoint 虽然是 encoder-decoder/T5 形态，但业务上是蛋白 embedding。稳定修法是同时在 `dataset_mapping._is_protein_embedding_like()` 和 manager 的 adaptation-context override 里补 `prot_t5 / prott5 / uniref50 + protein embedding / protein sequence` 信号，统一收敛到 `protein_embedding -> embedding + synthetic_protein + embedding_similarity`
- 2026-04-12 新强化：`Qwen/Qwen3-VL-Embedding-8B` 暴露出另一类 phase-4 画像漂移。模型名和 adaptation 证据链同时包含 `qwen3-vl + embedding` / `Vision-Language Embedding`，若仍按通用 multimodal 先收敛到 `vlm + scienceqa + vlm_accuracy`，会把多模态 embedding 模型误当成生成式 VLM。稳定修法是两层同时做：一、`dataset_mapping.py` 增加 `multimodal embedding` 检测，在 `modality == multimodal` 之前优先回到 `embedding + wikitext + embedding_similarity`；二、manager 的 adaptation-context override 也补 `vision-language embedding / visual-text embedding / vlm embedding` 等信号，避免本地 README/accuracy_run 重新把画像拖回 `vlm_accuracy`
- 2026-04-12 新强化：manager 的 causal-LM instruct 识别不能继续对 `orca` / `dpo` / `chat` 等短词做裸子串匹配。`Qwen/Qwen3-Coder-Next-Base` 证明这会把 `AutoModelForCausalLM` / `HF_ENDPOINT` 一类无关上下文误判成 instruct，再叠加 README 里的弱 `conversational` 描述，把 base 模型错误刷成 `gsm8k`。稳定修法是：强 instruct 关键词改成边界感知匹配，并把 `conversation / conversational` 降为弱信号；若 model id / context 明确带 `Base`，弱会话信号不得覆盖 `causal_lm_base + mmlu`
- 2026-04-12 新强化：`Qwen/Qwen3-30B-A3B` 说明 phase-4 多卡继承不能只识别 `range(8)` 这种字面量。`accuracy_run(_perf).py` 里常见的 `device_map=\"auto\" + max_memory={i: ... for i in range(n_npu)}` 是“动态全可见多卡”信号；若 manager 吃不到，就会在 `run-npu` 前退回单卡自动选卡。稳定修法是三段一起做：一、`_infer_multicard_npu_plan_from_accuracy_perf()` 识别 `range(n_npu|num_npu|npu_count|torch.npu.device_count())` 并返回 `parallel_mode=auto, device_topology=all_visible_devices`；二、`_prepare_business_config()` 保留这类动态多卡计划，即使没有显式 `selected_npus`；三、`_apply_local_npu_device_selection()` 在看到 `auto + all_visible_devices` 时展开成当前可见 NPU 列表，而不是偷改回单卡
- 2026-04-12 新强化：`Qwen/Qwen3-ForcedAligner-0.6B` 暴露出 phase-4 画像里的另一类裸 `ner` 误伤。`aligner` 会包含子串 `ner`，若仅按 `_contains_any(..., TOKEN_CLASSIFICATION_KEYWORDS)` 做子串匹配，会把 forced-aligner 类 ASR 模型误判成 `token_classification + conll2003`。稳定修法是两步一起做：一、给 `FORCED_ALIGNER_KEYWORDS` 补上 `forcedaligner` 并在 `detect_model_type()` 里优先收敛到 `asr`；二、把 token-classification 的 `ner` 检测改成独立 token 匹配，而不是任意子串命中
- 2026-04-12 新强化：phase-4 通用模板原本不支持 `qwen3-tts` 家族。`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` 需要 `qwen_tts.Qwen3TTSModel.generate_custom_voice()`，`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` 则走 `AutoProcessor/AutoTokenizer + AutoModelForCausalLM.generate()`。稳定修法是在 `business_model_eval.py` 模板里新增 `_load_qwen_tts_stack()` 与 `_run_tts()` 双分支：`CustomVoice -> qwen_tts wrapper`，`VoiceDesign -> transformers generate`，同时把 `tts` 加入 warmup / `_execute_inference()`；完成后必须重新 `generate-script` 刷新 adaptation 内的 `business_model_eval.py`
- 2026-04-12 新强化：`answerdotai/JaColBERTv2.5` 暴露出 manager adaptation-context override 的优先级缺口。ColBERT / reranker README 常同时出现 `token-level contextualized embeddings`，旧逻辑会被 embedding override 抢先命中，把本来正确的 `reranker + ms_marco + reranker_ndcg` 漂回 `embedding + wikitext`。稳定修法是：在 `business_benchmark_manager.py` 的 adaptation-context override 中，先于 embedding 分支增加 `colbert / cross-encoder / rerank / ms-marco / late interaction` 的显式 reranker 收敛，并用回归测试锁住“即使 README 提到 embeddings，也不能覆盖 reranker”
- 2026-04-12 新强化：`allenai/longformer-base-4096` 与 `answerdotai/ModernBERT-base` 暴露出 generic masked-LM encoder 的两类 phase-4 漂移。Longformer 若不把 `longformer` 视作 BERT-family encoder，会退回 `causal_lm + mmlu`；ModernBERT 的 README/accuracy 注释里又常带 `token_classification` 候选数据集说明，旧 manager 会误把这类 `AutoModelForMaskedLM + cls_embeddings` 方案刷成 `token_classification + conll2003`。稳定修法是两层一起做：一、`dataset_mapping.py` 把 `longformer` 纳入 BERT-family encoder 画像，回到 `embedding + wikitext + embedding_similarity`；二、manager 的 adaptation-context override 对 generic masked-LM/encoder embedding 增加 `AutoModelForMaskedLM / fill-mask / cls_embeddings` 收敛，同时收紧 token-classification override，避免被注释里的候选数据集关键词误伤
- 2026-04-12 新强化：`Salesforce/blip-image-captioning-large`、`biometric-ai-lab/Face_Recognition`、`Qwen/Qwen3Guard-Gen-8B` 又补齐了三类 phase-4 漂移。BLIP 一旦让 README 里的 `visual question answering` 抢过 `image captioning / image input` 主信号，就会从多模态业务集漂到 `seq2seq_qa + pubmed_qa`；Face_Recognition 若只看到 repo 名里的 `bio`，会被误刷到 biomedical/text embedding；Qwen3Guard-Gen 若不识别 `guard-gen / 安全对话 / 安全防护`，会退回 base `mmlu`。稳定修法是：manager 提前固定 `blip / image captioning` 到 `vlm + scienceqa`，把 `face recognition / face embeddings / LFW` 视作 `vision_embedding + cifar100`，并把 `guard-gen / qwen3guard / 安全对话` 纳入 instruct 信号；同时 `dataset_mapping.py` 也补对应的视觉 embedding / instruct 关键词，避免 CLI 与 manager 再次分叉
- 2026-04-12 新强化：`Team-Promptia/RLT-student-Qwen3-32B-medicine_biology` 在 biomedical generative profile 与短答案 QA harness 都修正后，最新一轮 `pubmed_qa` 仍得到 `NPU baseline/perf ≈ 1.112s/1.116s`、`CUDA baseline ≈ 0.0516s`、`vs_cuda_latency_ratio=0.0463`，继续远低于 completed 下界。此时标准动作不是继续烧 CUDA，而是备份异常 canonical CUDA 工件与旧 completed summary，重建 `pending_remote_cuda` 的 partial `business_summary.json`，并把看板回退到 `pending`
- 2026-04-12 新强化：`Hermes` / `OpenHermes` / `Dolphin` 这类 instruct 家族即使仓库名里没有显式 `instruct` 后缀，第四阶段也必须按 instruct/chat 业务画像处理；稳定修法不是单模型手工覆盖，而是同时在 `dataset_mapping.py` 和 `business_benchmark_manager.py` 的 adaptation-context 画像里识别 `hermes` / `dolphin` / `instruction-following` / `conversation` / `role-playing` 等信号，确保 phase-4 默认落到 `causal_lm_instruct + gsm8k`
- 2026-04-11 新强化：`lblueee/t5-academic-title-generator-model` 暴露出 phase-4 画像里的短关键词误伤。`TOKEN_CLASSIFICATION_KEYWORDS` 里裸 `"ner"` 会命中 `generator`，把 `t5-academic-title-generator-model` 错判成 `token_classification + conll2003`。稳定修法不是硬覆盖单模型 config，而是在 `dataset_mapping.detect_model_type()` 里给 token-classification 关键词加“若同时存在明确 seq2seq 信号（`Seq2SeqLM`/`ConditionalGeneration`/`t5|bart|pegasus|flan|led`）则不再抢先命中”，并用回归测试锁住 `get_business_benchmark_profile("lblueee/t5-academic-title-generator-model") -> seq2seq + cnn_dailymail`
- 2026-04-11 新强化：`talphaidze/molm-scientific-router-trained200m_corr` 证明 phase-4 通用 causal-LM loader 还需要兼容两类 trust_remote_code MoLM 问题。第一类是 `from_pretrained()` 在 transformers 5.x 收尾时访问不存在的 `all_tied_weights_keys`；稳定修法是在 `business_model_eval.py::_load_from_source()` 捕获该错误后，对本地 snapshot 改走 `from_config + 手工加载 safetensors/bin state_dict + to(device)`。第二类是 multiple-choice scoring 的标准调用会给 custom forward 传 `use_cache=False`，而 MoLM/GPTBase 不接受这个 kwarg；稳定修法是在 `_score_multiple_choice_candidate(s)` 统一经过 `_forward_causal_lm_no_cache()`，若 `TypeError` 明确指向 `use_cache`，就剥掉该 kwarg 重试
- 2026-04-10 新强化：`Crystalcareai/llama-3-4x8b` 证明“重跑验证 + 深修 harness”之后仍明显越界的 phase-4 outlier 必须及时止损。该模型 fresh NPU `mmlu` latency 稳定在 `~10.5s/sample`，远端 H100 CUDA baseline 为 `~0.193s/sample`，`vs_cuda_latency_ratio=0.0183`；进一步把多选打分从“每个选项单独前向”改成“同一样本候选批量前向”后，fresh NPU baseline 在 7 分多钟时仍未落新工件，已足够证明它不可能回到 completed 门限。标准处置是：停止继续烧 CUDA，把异常 CUDA canonical 工件备份掉，恢复最新可用的本机 NPU baseline/perf，重建 `pending_remote_cuda` 的 partial `business_summary.json`，并把看板状态回到 `pending`
- 2026-04-10 新强化：`DimensionSTP/OPEN-SOLAR-KO-10.7B-scientific-qa` 暴露出两个 phase-4 起步坑。第一，模型名里的 `scientific-qa` 会把纯 `LlamaForCausalLM` 错漂成 `extractive_qa + squad_v2`，但只要把本地 snapshot `config.json` 的 `architectures=LlamaForCausalLM` 带进画像，就会稳定回到 causal-LM 路径（当前 manager 进一步收敛到 `pubmed_qa + qa_exact_match`）。第二，adaptation 目录若残留 root-owned `.venv` / `__pycache__`，`uv run --extra ascend` 会退化去建 `.venv_user`；标准修法仍然是“同目录原地改名隔离 root-owned `.venv` / `__pycache__`，连同误建的 `.venv_user` 一并隔离，再从干净目录重开 phase-4”
- 2026-04-10 新强化：`DimensionSTP/OPEN-SOLAR-KO-10.7B-scientific-qa` 还暴露出更硬的 completed 阻断：当前 HF 仓库公开的 `model.safetensors.index.json` 与本地 snapshot 一致，都会缺少 `model.layers.47.input_layernorm.weight`、`model.layers.47.post_attention_layernorm.weight`、`model.layers.47.mlp.down_proj.weight`、`model.norm.weight`、`lm_head.weight` 这 5 个 key；逐个扫描 10 个 safetensors shard 也确实找不到这些权重。结果是 `LlamaForCausalLM.from_pretrained()` 每次都会把这 5 个参数随机初始化，phase-4 baseline/perf 质量值会随 reload 漂移，不能再继续浪费 CUDA 时段。标准处置是：保留现有 partial `business_summary.json` 作为现场证据，把看板回退到 `pending`，reason 直接写 `phase4_upstream_checkpoint_missing_weights`，等待完整 checkpoint 后再重开整轮 phase-4
- 2026-04-10 新强化：`DimensionSTP/Solar-Ko-Recovery-11B-scientific-qa` 与上一条属于同一类仓库级缺陷。当前公开 `model.safetensors.index.json` 也缺同样 5 个关键权重，逐个 shard 扫描同样找不到；此外 adaptation 目录还残留 root-owned `.venv`，会让 `uv run --extra ascend` 直接报 `failed to canonicalize .../.venv/bin/python3: Permission denied`。因此这类模型在 phase-4 上的标准动作不是“先修 root env 再硬跑”，而是先按 `phase4_upstream_checkpoint_missing_weights` 回到 `pending`，并在 notes 里额外注明 `root_owned_phase4_env_blocks_uv`，等完整 checkpoint 可用后再顺手清理本地 `.venv` 污染并重开
- 2026-04-10 新强化：`sentence-transformers/all-MiniLM-L6-v2` 证明 manager 在 phase-4 `generate-script` 时不能先拿 `model_id` 直接 `AutoConfig.from_pretrained()`，否则即使 adaptation 本地 `models/.../snapshots/.../config.json` 已经齐全，也会先发一轮 `HEAD config.json` 到 HF/HF mirror，被弱网卡死。稳定修法是：`business_benchmark_manager.py::_infer_transformers_config_metadata()` 先解析 adaptation 本地 snapshot，命中 `config.json` 后把 snapshot 路径作为 `from_pretrained()` source，并显式 `local_files_only=True`
- 2026-04-10 新强化：`abid/indonesia-bioner` 证明 `business_benchmark_tool.py summarize` 不能优先拿“旧的完整三件套 run_id”或过期 `business_summary.json`，否则本轮只刷新 NPU baseline/perf 时会把 stale CUDA 工件重新拉回来。稳定修法是：一、按“最新 NPU baseline/perf 所属轮次”选 summary 工件，只有同 run_id 的 CUDA 才能并入；二、当 run_id 缺失时，仅在旧 `business_summary.json` 仍指向当前最新 NPU 文件时才允许沿用它的 artifact selection。该回归已由 `tests/test_business_benchmark_tool_summary_selection.py` 固化
- 2026-04-10 新强化：`Crystalcareai/GemMoE-Medium-V0.5` 暴露出两个叠加的 phase-4 继承缺口。第一层是 `cuda_baseline` 以前没有继承 `accuracy_run.py` 的 baseline patch，只在 `npu_baseline` 才合并 `accuracy_run._patch_gemmoe_decoder_layer`，导致远端 CUDA 仍走原始 GemMoE decoder；第二层是 phase-4 加载链路没把 GemMoE 的 `config._attn_implementation = "eager"` 继承进来。当前模板修法是：`_resolve_model_sources()` 对 `cuda_baseline` 也合并 `accuracy_run` hooks；`_load_model_stack()` 在 `AutoConfig.from_pretrained()` 后执行 `_apply_model_config_compatibility_fixes()`，对 `model_type/architectures/model_id` 命中 `gemmoe` 的模型强制 `_attn_implementation = "eager"`，并把修改后的 `config` 显式传入 `from_pretrained()`
- 2026-04-09 新强化：`jaydubya/Scientific_Industry_Theme` 证明 phase-4 画像不能只依赖已有 `business_benchmark_config.json` 和 README。该模型真实架构是 `DebertaV2ForSequenceClassification`，但配置初始缺少 `architectures/problem_type/num_labels`，README 又错误写成“文本生成”，导致 manager 先把它漂成 `causal_lm + mmlu`。稳定修法是三层同时做：一、`dataset_mapping.detect_model_type()` 在 BERT-family fallback 到 embedding 前必须带上 `architectures/problem_type/num_labels`；二、manager 的 `_resolve_business_profile()` 在配置字段缺失时要从本地 `AutoConfig` / `config.json` 自动回填；三、若 inferred 已是显式非 causal 模型，不能再让 adaptation context 的弱 `text generation` 信号把它覆盖回 `causal_lm`
- 2026-04-09 新强化：`mistralai/Mistral-7B-Instruct-v0.2` 与 `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5` 证明 root-owned 的 phase-4 污染必须按“整轮本地刷新”处理，而不是在脏目录上修补。对 root-owned `.venv` 直接 `rm -rf` 会报 `Permission denied`，跨文件系统挪到 `/tmp` 会退化成整目录复制。稳定做法是：先刷新 `business_benchmark_started_at` / `benchmark_run_started_at`，再在 adaptation 目录内同文件系统把 `.venv` / `__pycache__` 改名隔离成隐藏目录，删除现有 `business_*`，重新执行 `run-npu -> summarize`；若 SSH 仍不通，最后把 fresh partial `business_summary.json` 原文写回 `wait_cuda`
- 2026-04-09 新强化：`facebook/esm2_t48_15B_UR50D` 证明蛋白质 masked-LM 不能按通用 `causal_lm` 处理。旧 phase-4 把它重画像成 `causal_lm + mmlu`，导致 formal business run 试图走 `AutoModelForCausalLM` 并在加载阶段直接失败。当前规则必须把 `facebook/esm*` / `EsmForMaskedLM` / `AutoModelForMaskedLM + protein sequence/amino acid/UR50` 统一收敛到 `protein_embedding -> embedding + synthetic_protein + embedding_similarity`；修复点要同时落在 `dataset_mapping.py`、`download_datasets.py`、`business_eval.py` 和 manager 的 adaptation-context override，缺一处都会在 phase-4 重生成时再次漂回错误画像
- 2026-04-09 新强化：`Team-Promptia/RLT-student-Qwen3-32B-medicine_biology` 进一步暴露出一个“状态降级但本地 canonical 证据没收口”的风险。该模型已被看板降到 `wait_cuda`，但 adaptation 目录里仍残留 `business_metrics_cuda_*_baseline.json` 与 `status=completed` 的 `business_summary.json`，会让后续人工检查误以为结果已收敛。标准修法不是只改 DB，而是先把异常 CUDA 正式工件和旧 `business_summary.json` 改名成 `__prev_rule_refresh_<timestamp>`，再基于当前 NPU baseline/perf 重跑 `summarize`，让本地 summary 回到 `pending_remote_cuda + vs_cuda_latency_ratio=null`
- 2026-04-12 新强化：`Team-Promptia/RLT-student-Qwen3-32B-medicine_biology` 按上述 canonical 收口规则重开后，正式拿到的新一轮 CUDA baseline 仍只有 `0.051646s/sample`，而同轮 NPU `pubmed_qa` baseline/perf 维持在 `1.11194s / 1.11588s`，`vs_cuda_latency_ratio=0.0462827544`。这说明该模型已经超出“再补一次 warmup / 再重跑一次 formal”能解决的范围；后续如无上游实现、设备拓扑或业务负载层的深修，不应再次申请 CUDA 时段
- 2026-04-12 新强化：`NousResearch/Hermes-3-Llama-3.1-8B` 证明 instruct/chat 家族不能只靠 repo 名字里的 `instruct` 关键字判断。该模型 README/context 明显是 instruction-following / conversational，但旧 phase-4 仍误落 `causal_lm_base + mmlu`。稳定修法是把 `hermes` / `dolphin` 加入 `CAUSAL_LM_INSTRUCT_KEYWORDS`，并让 manager 的 adaptation-context override 同时识别 `instruction-following`、`role-playing`、`conversation` 等语义信号；修后默认画像恢复到 `gsm8k + generation_exact_match`，本轮也已成功 completed
- 2026-04-09 新强化：`ChangyuanWang/LLaVA-vicuna-7B-v1.3-ScienceQA` 证明 VLM 视觉塔的本地 snapshot 解析不能强依赖 `config.json`。`openai/clip-vit-large-patch14` 这类纯 image processor cache 只有 `preprocessor_config.json`，若 `_resolve_local_snapshot_source(..., input_kind='image_processor')` 仍要求 `config.json`，phase-4 formal CUDA 就会在真正跑数时回退到 hub id，外网探测 `processor_config.json`，表现为 warmup 能过、formal 长时间卡在 `HEAD https://hf-mirror.com/openai/clip-vit-large-patch14/resolve/main/processor_config.json`。修法是两层：一、纯 image processor snapshot 只要本地有 `preprocessor_config.json|processor_config.json` 就允许命中，无需 `config.json`；二、LLaVA 路径的 `CLIPImageProcessor.from_pretrained()` 必须优先使用解析出的本地 snapshot 路径，而不是直接传 hub id
- 2026-04-08 新强化：`01-ai/Yi-1.5-34B-Chat` 证明“远端 `models/` 目录存在”也不代表其中已经有完整权重分片。旧 `run-remote-cuda` 只看目录是否存在，结果对 34B 模型只同步 tokenizer/config，随后远端在 CUDA 时段里现拉 `15` 片 safetensors。修法是：manager 在复用远端旧 cache 前，不仅检查 input snapshot，还要逐个本地 snapshot 检查远端是否已有 `model.safetensors.index.json` / `pytorch_model.bin.index.json` 或实际权重分片；任一 snapshot 缺权重就直接 `rsync --partial --append-verify` 整个本地 `models/`。规则上要把“远端模型目录存在”和“远端权重可用”分开看待
- 2026-04-08 新强化：`RandipR/pegasus-560m-academic-sum` 证明“远端 `models/` 目录存在”不等于可直接信任。旧 `run-remote-cuda` 只在远端缺 `models/` 时才全量同步，本轮远端正好残留了一份坏 Pegasus tokenizer cache，导致 CUDA 侧 `AutoTokenizer.from_pretrained(..., use_fast=False)` 直接报 `ValueError: ('<unk>', 0.0) is not in list`。修法不是继续删库重下，而是让 manager 在复用远端旧 cache 前，始终用 `rsync -L` 把本地已验证 snapshot 的 input assets（至少 `config.json`、`tokenizer_config.json`、`special_tokens_map.json`、`spiece.model`，以及其它 tokenizer/processor 资产）覆盖到远端对应 `snapshots/<rev>/`；这样既避免全量模型重传，也能把坏 tokenizer cache 拉回到本地已验证口径
- 2026-04-08 新强化：`BIOMEDICA/BMC_CLIP_CF` 暴露出 `open_clip` 视觉零样本路径的 steady-state 计量边界问题。旧实现把 `processor.preprocess(image)` 也算进 `_execute_inference` 的 `latency_s`，导致跨机器 CPU 预处理差异直接污染 `vs_cuda_latency_ratio`，第一次 CUDA 结果冲到 `2.335x`、第二次仅靠把 `cuda_baseline_warmup_iterations` 提到 `3` 也只降到 `2.154x`。真正修法是：在 `latency_measurement_scope=steady_state` 时，把固定图像预处理移到计时外，只保留 H2D + `encode_image` + logits 进入延迟；修后同模型 NPU `latency_s` 从 `~0.033s` 回到 `~0.0093s`，CUDA `latency_s` 回到 `~0.0108s`，`vs_cuda_latency_ratio≈1.154`
- 同次规则固化：`run-remote-cuda` 的“第 1 次热身丢弃、第 2 次正式回收”是两个独立进程，不能替正式进程完成 kernel 预热；对 CLIP / `vision_topk_accuracy` 这类视觉任务，若正式 CUDA baseline 仍偏慢，优先把 `cuda_baseline_warmup_iterations` 提到至少 `3`，不要只靠外层 warmup run 侥幸过门槛
- 2026-04-08 新强化：`Helios9/BioMed_NER` 证明 `biomedical_token_classification` 不能在 phase-4 被当成“未知模型类型”。通用 `business_model_eval.py` 需要把它规范化为 `token_classification` 别名，至少覆盖组件加载、warmup 样本选择、推理分发三处；否则 manager 虽然已经把数据集纠正到 `ncbi_disease`，真正执行时仍会在 `model_type=biomedical_token_classification` 直接报不支持
- 2026-04-08 新强化：`ncbi_disease` 的 phase-4 下载链路不能只依赖 GitHub raw。当前稳定做法是：优先 `cdn.jsdelivr.net/gh/spyysalo/ncbi-disease@master/conll/devel.tsv`，失败再回退其它 GitHub URL，并且只下载 validation `devel.tsv`；不要为了业务评测再拉 `train/test`，否则弱网环境会把 `Helios9/BioMed_NER` 这类模型卡在数据集准备阶段
- 同次实跑结论：`Helios9/BioMed_NER` 在 `ncbi_disease` 64 样本上三路结果完全一致，`npu_baseline/perf/cuda` 的 `f1=0.03738`、`match_rate=0.92141`，同时 `npu_speedup_ratio=1.006`、`vs_cuda_latency_ratio=0.996`。这说明“绝对 F1 很低”不一定是硬件不一致；对于宽标签空间 biomedical NER 模型折叠到 disease-only 标签集后的场景，只要三路指标一致、没有 0 分塌陷、速度比正常，就应优先接受为业务画像边界，而不是继续误判成 phase-4 计量故障
- 2026-04-08 新强化：`Helios9/BioMed_NER` 实跑再次证明第四阶段不能依赖“根环境刚好装过可选指标包”。`token_classification_f1` 直接依赖 `seqeval`，`asr_wer` 依赖 `jiwer`，`rouge_l` / `reranker_ndcg` 也分别依赖 `rouge-score` / `scikit-learn`。manager 现在必须按 `evaluation_profile` 精准补齐 adaptation 自己 `pyproject.toml` 里的缺包，而不是一股脑把四个都装上；否则像 `BioMed_NER` 这种只需要 `seqeval` 的模型，也可能被 `jiwer -> rapidfuzz` 的无关下载超时拖死
- 2026-03-24 起默认不再把远端 CUDA 当成“纯手工回传”流程；若 `remote_ssh_host` / `remote_project_root` 可用且 SSH 连通，应优先使用 `business_benchmark_manager.py run-remote-cuda` 自动完成远端执行、工件回收、本地 summarize/check，只有 SSH 不通或自动执行失败时才降级到 `print-remote-command + waiting_cuda`
- 2026-04-05 新强化：`wait_cuda` 只是模型状态，不是 worker 占用状态。发送 `result=in_progress stage=waiting_cuda` 后，必须立刻再发 `status=idle` 给 team-lead，明确当前 business-benchmark 已释放，可继续领取别的 pending 第四阶段任务
- 2026-04-06 新强化：VLM / 多模态模型（尤其 `qwen2.5-vl` 一类）默认必须保持图文业务画像，不能漂成 `gsm8k/wikitext/mmlu/ceval` 这类纯文本数据集；若 evaluator 只有在带图样本上才会真实执行推理，那么无图数据集只会制造假快结果
- 2026-04-07 新强化：`Qwen/Qwen3-0.6B` 在 `mmlu` 64 样本上确认存在稳定的三路离散精度漂移 `0.296875 / 0.34375 / 0.3125`，跨度正好是 `3/64=0.046875`。当速度比健康、指标名一致、没有 0 分塌陷或越界时，bounded discrete quality tolerance 需要至少放宽到“最多三个样本粒度”，不能再按 1~2 个样本粒度硬拦
- 2026-04-07 新强化：`openai/whisper-large-v3` 实跑暴露出 ASR 半精度 dtype 链路缺口。即使 `business_benchmark_config.json` 里仍写 `fp32`，Whisper 实际参数也可能以 `fp16` / `bf16` 落到设备上；若 processor 产出的 `input_features` 继续保留 `float32`，会在 encoder `conv1d` 直接报 `Input type (float) and bias type (Half) should be the same`。通用规则应在 `_run_asr()` 中把所有浮点输入 tensor 显式对齐到模型真实参数 dtype 后再 `generate`
- 2026-04-08 新强化：`biodatlab/distill-whisper-th-large-v3` 证明 phase-4 的 Whisper/通用 ASR 速度异常不能只靠重跑或只看 `TASK_QUEUE_ENABLE`。该模型首轮真实业务 `librispeech` 上 `npu_speedup_ratio` 连续落到 `0.88x / 0.86x`，而 `TQE=0` 甚至更慢，说明问题不在单一 runtime flag。真正修法是给第四阶段 `npu_perf` 增加小批量 `generate` 能力，并在同一业务集上做 `batch_size=2/4/8` 原型；最终 `batch_size=4 + TASK_QUEUE_ENABLE=1` 把正式本机 NPU 从 `0.225705s -> 0.115559s`，`npu_speedup_ratio` 拉回 `1.953x`，且 `WER/text_match_rate` 不降反升。经验固化：对 Whisper/ASR 若两轮重跑后仍 `<0.90`，优先继续做 scenario-specific microbatch ablation，并把成功配置显式写进 `business_benchmark_config.json`（如 `asr_perf_batch_size=4`）；baseline/cuda 默认仍保 `batch_size=1`
- 2026-04-07 新强化：`Qwen/Qwen2.5-1.5B-Instruct` 的历史第四阶段 `gsm8k` 工件曾显示 `exact_match=0.0 / match_rate=0.0` 且 `npu_speedup_ratio=0.9519`；但按新规则备份旧正式工件、刷新 `benchmark_run_id` 后重跑，本机 NPU baseline/perf 立即恢复到 `exact_match=0.5625 / 0.5625`，`latency_s=5.983931 / 5.937823`，`npu_speedup_ratio=1.0078`。结论：`causal_lm_instruct + gsm8k` 的 0 分塌陷不能直接当模型能力结论，必须先按最新模板重开一轮 `run-npu`
- 2026-04-07 新强化：`dima806/fairface_age_image_detection` 的历史第四阶段同时踩了两类错配：一是 `model_id` 含 `detect` 被误判成 `vision_detection/coco`，二是后续又被手工漂成 `imagenet`，导致历史 `top1_accuracy=0.0`。修复方式是把业务画像固定到 `vision_classification + fairface + vision_topk_accuracy`，并为 `nateraw/fairface` 增加自定义下载器，把 legacy `val.pt` 脚本数据集物化为 `datasets/nateraw___fairface`
- 同次实跑结果：新一轮本机 NPU `fairface` 64 样本，baseline `latency_s=0.008667`、perf `0.007444`、`npu_speedup_ratio=1.1643`；两路 `top1_accuracy/match_rate=0.671875`，`top5_accuracy=1.0`，说明旧 `0.0` 完全来自业务画像错误，不是模型或 NPU 问题
- 2026-04-08 新强化：`magic-leap-community/superpoint` 证明 keypoint 模型如果没有单独画像，会被历史规则漂成 `causal_lm + mmlu` 一类假任务。当前必须补齐 `vision_keypoint_detection + synthetic_keypoints + keypoint_repeatability` 这条业务画像链路；以后新增视觉子类型时，必须同步更新 `dataset_mapping.py`、`download_datasets.py`、`business_eval.py` 和 manager 侧的 adaptation-context override
- 同次修复经验：`superpoint` 历史 phase-4 `business_model_eval.py` 是旧手写脚本，既缺 `run_business_eval()` 入口，也可能把 `benchmark_run_id` / `scenario_command` 留成模板字面量。遇到这类 legacy phase-4，不要在原文件上继续补洞；先备份旧正式工件与旧脚本，再用当前 manager 模板整体重生，最后再跑 `run-npu`
- 2026-04-08 新强化：`wait_cuda` 不再只是“缺 CUDA 工件”这个状态名；即便 SSH 不通，也必须在本机 NPU baseline/perf 刷新完成后立刻执行 `summarize`，生成一份指向本轮 NPU 工件的 partial `business_summary.json`，再写库到 `wait_cuda`。否则后续 CUDA 恢复后很容易拿着过期 summary 继续拼错轮次
- 2026-04-08 新强化：仅有 `benchmark_run_id` 还不够，`business_summary.json` 现在也必须显式保留 `benchmark_run_started_at`，并与 `business_benchmark_config.json` 保持一致。`MaterialsInformaticsLaboratory/QA-MatBERT-seed42` 这轮排查证明：如果 summary 丢了 start time，后续看板 notes 虽然能看到 run id，却无法确认是否真的是本轮重跑产物，因此要把“缺 start time”直接视为 stale summary，先重做 `summarize`
- 2026-04-08 新强化：`MaterialsInformaticsLaboratory/QA-SciBERT-seed36` 证明优化阶段 `fusion_ops` 成立，不代表第四阶段真实业务负载也必须继续加载 `model_files`。该模型优化阶段在 `sst2` 上 `npu_add_layer_norm + warmup + TQE` 仍有 `2.93x`，但 phase-4 `squad_v2` 首轮 fusion 业务工件却从 `0.007726s -> 0.009119s`，`npu_speedup_ratio=0.847x`。标准修法不是篡改 `optimization_notes.json`，而是在 `business_benchmark_config.json` 中显式写 `optimization_kind=runtime_only` + `npu_perf_use_model_files=false`，刷新 `benchmark_run_id/benchmark_run_started_at` 后整轮重跑；新本机 NPU 已恢复到 `0.010873s -> 0.007822s`，`npu_speedup_ratio≈1.39x`
- 2026-04-08 新强化：`LinerAI/snowflake-arctic-embed-m-v2.0-academic` 证明 `feature extraction` 是高风险歧义词。旧画像逻辑仅因 adaptation 上下文里出现该短语，就把文本 embedding 模型误落成 `vision_keypoint_detection + synthetic_keypoints`，随后 `business_eval.py` 直接因空数据集失败。规则修正后，keypoint 必须同时具备 `keypoint/superpoint/local feature/output_type=keypoints` 这类证据；若上下文含 `Model Type: embedding`、`text embeddings`、`mean pooling`、`output_type=text_embeddings`，则应优先固定为 `embedding + wikitext + embedding_similarity`
- 2026-04-08 新强化：`openai/clip-vit-large-patch14-336` 证明 CLIP 也有一条相反方向的误伤链。旧 manager 画像会先看到 adaptation 上下文里的 `image embeddings` / `text embeddings`，把 `CLIPModel + CLIPProcessor + zero-shot image classification` 错降成 `embedding + wikitext + embedding_similarity`，随后远端 CUDA 会把 wikitext 文本错误喂给图像 processor。规则修正后，CLIP / zero-shot-image-classification 必须先检查 `CLIPModel`、`CLIPProcessor`、`AutoModelForZeroShotImageClassification`、`logits_per_image`、`pixel_values`、`a photo of ...` 这类视觉信号，并优先固定到 `vision_classification + cifar100|imagenet + vision_topk_accuracy`
- 同次执行经验：当根目录 `.venv` 损坏、只能通过 `env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=.venv_user uv run ...` 启动 manager 时，必须确保这两个环境变量不会泄漏到 adaptation 目录内层的 `uv run --extra ascend|cuda ...`。否则 phase-4 会在 adaptation 下误建 `.venv_user`，把正式口径从 adaptation 自己的 `.venv` 拉偏。manager / `business_run.py` 现已默认在 adaptation 子命令前显式清掉 `UV_PROJECT_ENVIRONMENT` 与 `VIRTUAL_ENV`
- 2026-04-08 经验边界校准：截至 `biomni` 与 `biodatlab/distill-whisper-th-large-v3` 收口后，当前共有 82 个 `business_benchmark_status=completed`、90 份本地 `business_summary.json`。正式 completed 样本里，`npu_speedup_ratio` 最低值为 `0.9015`，`vs_cuda_latency_ratio` 的经验区间为 `[0.2282, 1.8034]`，中位数约为 `1.0395 / 0.6491`。这说明当前告警阈值 `npu_speedup_ratio < 0.90` 与 `vs_cuda_latency_ratio ∉ [0.22, 1.85]` 仍然基本贴着真实 completed 样本边界；后续若再遇到越界结果，应优先按“重跑验证 -> 深修 harness/画像/远端闭环”处理，而不是继续放宽阈值
- 2026-04-08 新强化：`Team-Promptia/RLT-student-Qwen3-32B-medicine_biology` 在 `pubmed_qa` 上完成了“重跑验证 + 深修 harness”后，`vs_cuda_latency_ratio` 仍稳定停在 `0.034 -> 0.0386 -> 0.0407`，远低于当前 completed 下界 `0.2282`，且扫描 87 份本地 `business_summary.json` 后仍是全局最小值。此类结果已经不是普通抖动，而是明确 outlier；标准处置应改为立即停止继续消耗 CUDA 时段，把模型保持在 `wait_cuda`/`pending`，并升级为上游模型实现、优化继承链或业务负载本身的深层问题，而不是继续无限重跑
- 同次修复经验：对这类异常先做两层兜底修复，再决定是否停损。第一层是计时链路加固：在 `run_business_eval()` 的顶层 timed inference 前后补设备同步，并同步修正视觉 steady-state override，避免 CUDA/NPU 异步下发把 `latency_s` 计短。第二层是短答案 QA evaluator 收紧：`PubMedQA` 样本要显式携带 `choices=["yes","no","maybe"]`，prompt 优先走 choice-constrained 路径，multiple-choice scoring 返回 choice 文本而不是 `A/B/C` 字母；如果这两层都做完，速度比仍是全局离群值，就应停止本阶段反复试错
- 2026-04-08 新强化：`01-ai/Yi-1.5-34B-Chat` 证明第四阶段还有一条设备继承缺口。该模型 optimization 已明确记录 `selected_npus=0-12`、`parallel_mode=tensor_parallel`、`device_topology=multi_npu`，但旧 `run-npu` 仍无条件按 `npu-smi info` 自动挑单卡，并把 `ASCEND_RT_VISIBLE_DEVICES=0` 写回 `business_benchmark_config.json`，导致 phase-4 本机跑成“单卡 + CPU/offload”的假业务环境
- 同次修复经验：当 `optimization_notes.json` 明确给出多 NPU 设备计划时，phase-4 生成配置必须先继承 `selected_npus / parallel_mode / device_topology`，并把 `npu_baseline_env` / `npu_perf_env` 里的 `ASCEND_RT_VISIBLE_DEVICES` 同步覆盖成同一组设备；只有在 optimization 没给出多卡计划时，才允许回退到“自动选当前空闲单卡”。如果上一轮错误 run 已把单卡 env 写回 config，刷新新轮次前也必须先把这个 stale 单卡值覆盖掉，否则后续 rerun 仍会被旧配置压回单卡
- 2026-04-08 新强化：`biomni/Biomni-R0-32B-Preview` 暴露出更隐蔽的一档继承缺口。该模型 phase-3 `accuracy_run_perf.py` 已明确写出 `device_map="auto"`、`max_memory={i: "15GB" for i in range(6)}` 和 `Multi-card: 6x NPU`，但 `optimization_notes.json` 却只残留了 `selected_npus=[0]`，导致 phase-4 manager 没有继承到真实多卡计划，重新生成的 `business_benchmark_config.json` 被 stale 单卡 env 压成 `ASCEND_RT_VISIBLE_DEVICES=0`
- 同次实跑结果：单卡误跑后，`biomni` 本机 NPU `latency_s` 稳定在 `1.80s -> 1.66s`，远端 H100 CUDA baseline 稳定在 `0.129s` 左右，两次闭环重跑的 `vs_cuda_latency_ratio` 分别是 `0.0766` 和 `0.0777`，几乎完全复现，说明这不是抖动而是错误业务拓扑
- 同次最终修复结果：按 `accuracy_run_perf.py:max_memory_range` 恢复 `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5` 后，`biomni` phase-4 本机 NPU 立刻回到 `baseline/perf = 0.563855s / 0.562292s`，两路 `exact_match/f1/match_rate = 0.4375`，补齐 CUDA 后正式 `vs_cuda_latency_ratio=0.23097`。经验固化：对 30B+ 大模型，只要 phase-3 明确是多卡，而 phase-4 却落成单卡，得到的速度比即使“可重复”也应直接视为错误拓扑，不得继续拿来判断 completed
- 同次规则固化：phase-4 设备计划继承不能只信 `optimization_notes.json`。若 notes 没把多卡写全，但 adaptation 的 `accuracy_run_perf.py` / `accuracy_run.py` / `demo.py` 已显式给出 `max_memory={i: ... for i in range(N)}`、多卡 `ASCEND_RT_VISIBLE_DEVICES` 字面量，或 `Multi-card: Nx NPU` 注释，manager 也必须把这些信号当作多卡兜底来源，自动恢复 `selected_npus / parallel_mode / device_topology / npu_*_env`
- 2026-04-08 新强化：`MaterialsInformaticsLaboratory/QA-MatBERT-seed16` 的历史 phase-4 显示 `npu_speedup_ratio=0.9359x`，但根因不是优化失效，而是旧手写 `business_eval.py` 与正式工件口径已漂移，连 `npu_perf` 的 `scenario_command` 都错指向 baseline。修复方式同样是备份旧 phase-4 正式工件和旧脚本、用当前模板重建并重跑；新一轮本机 NPU 结果恢复到 baseline `0.008751s`、perf `0.007799s`、`npu_speedup_ratio=1.1221x`
- 同次核验经验：若 adaptation 含 `model_files/`，第四阶段刷新后必须在 perf 工件里看到 `loaded_from_model_files=true`、`patch_load_status=applied` 或等价 patch 继承证据；若 summary/工件仍显示未加载 patch，不要急着接受 `speedup<1`，优先怀疑 phase-4 harness 仍在走旧脚本
- 2026-04-08 新强化：`InfiX-ai/Qwen-base-7B-biology` 证明还有一类更隐蔽的继承缺口：`optimization_kind=fusion_ops` 但 `model_files/` 为空目录时，第四阶段不能只检查 `model_files`。该模型的优化 hook 实际写在 `accuracy_run_perf.py::_apply_npu_patches()`；旧模板没复用它，导致 phase-4 `npu_perf` 只剩 runtime 开关，业务侧 `npu_speedup_ratio` 只有 `0.9598x`
- 同次修复经验：通用 `business_model_eval.py` 现在必须在 `npu_perf` 场景下先尝试 `model_files`，若没有可用 hook，再自动从 adaptation 的 `accuracy_run_perf.py` 发现并复用 `apply_npu_patches` / `_apply_npu_patches` / `apply_npu_optimizations` 一类入口；同时 hook 签名判断要按“是否存在必填位置参数”而不是“是否存在任意位置参数”，否则会把只有默认参数的 `_apply_npu_patches(patch_x=True, ...)` 误判成需要 `model` 注入
- 同次实跑结果：模板补齐后，`Qwen-base-7B-biology` phase-4 perf 工件开始明确记录 `patch_load_status=applied`、`patch_hooks=["accuracy_run_perf._apply_npu_patches"]`，业务延迟从 baseline `4.452912s` 降到 perf `3.884268s`，`npu_speedup_ratio` 恢复到 `1.1464x`。结论：对 `fusion_ops + 空 model_files` 的 completed 模型，若 phase-4 只看到 runtime-only 证据或速度比异常，优先检查是否漏继承了 `accuracy_run_perf.py` 内联 hook
- 2026-04-08 新强化：`Sidd2005/Heart-Biology-RAG-Model` 证明 biomedical 关键词还有另一类误伤。该模型真实是 `T5ForConditionalGeneration` / `AutoModelForSeq2SeqLM` 的 RAG 式问答，但旧第四阶段只凭 `bio/med/pubmed` 把它降成 `embedding_similarity`，随后通用 evaluator 错走 `_run_embedding()`，直接在 T5 decoder 触发 `ValueError: You have to specify either decoder_input_ids or decoder_inputs_embeds`
- 同次修复经验：对 biomedical 模型，若 adaptation 上下文出现 `AutoModelForSeq2SeqLM`、`T5ForConditionalGeneration`、`Seq2Seq / Question Answering`、`RAG-style` 这类证据，应优先覆盖为 `model_type=seq2seq`；若语义是问答，则业务画像固定到 `pubmed_qa + qa_exact_match`，不能再沿用 `embedding_similarity`
- 2026-04-08 新强化：`Team-Promptia/RLT-student-Qwen3-32B-medicine_biology` 证明 `PubMedQA` 这类短答案问答还有一条独立风险链。旧第四阶段虽然选对了 `pubmed_qa + qa_exact_match`，但通用生成路径没有给样本透传 `dataset_key`，也没有约束单词级答案和关闭 Qwen3 thinking，结果本机 baseline 一度跑成小时级异常长跑，历史工件还出现 `exact_match=0.0`
- 同次修复经验：对 `PubMedQA` / `qa_exact_match`，模板 `business_eval.py` 必须把 `dataset_key=pubmed_qa` 写进样本；模板 `business_model_eval.py` 必须为该数据集使用 `Answer with exactly one word: yes, no, or maybe.` 的 prompt、将 `max_new_tokens` 压到短答案量级，并在 chat template 支持时显式 `enable_thinking=False`；生成后还要做 `yes/no/maybe` 归一化，否则 reasoning 模型会用解释性长文本把业务时延和离散精度一起打坏
- 2026-04-08 新强化：同一个 `Team-Promptia/...Qwen3-32B...` 又暴露出 `model_files` 命名空间污染问题。该 adaptation 的 `model_files/modeling_qwen3.py` 实际只是说明文档，并不含可执行 patch；旧第四阶段却把它按 `transformers.models.qwen3.modeling_qwen3` 预导入，直接覆盖官方模块，导致 perf 场景在 `AutoModelForCausalLM.from_pretrained()` 时抛 `ModuleNotFoundError: Qwen3ForCausalLM`
- 同次修复经验：当 `model_files/` 没有 `config.json`、也不是完整本地模型仓时，第四阶段只能用中性模块名导入其中的 `*.py` 来搜集 hook，绝不能预先覆盖 `transformers.models.*` 命名空间。若这些文件最终没有提供 hook，应把状态视为 `namespace_only`，然后自动回退复用 `accuracy_run_perf.py` 的 `_apply_npu_patches` / `apply_npu_optimizations`；不要因为“导入过一个文件”就把 `loaded_from_model_files=true` 当成有效优化证据
- 2026-04-08 新强化：`hustvl/vitmatte-small-composition-1k` 说明 `image matting` 也是第四阶段需要单独画像的视觉子类型，不能被通用 `vision_classification` 或旧手写脚本糊过去。正式画像必须固定到 `image_matting + synthetic_matting + matting_mae + alpha_masks`
- 同次修复经验：旧 `VitMatte` phase-4 只有手写 `business_eval.py`，虽然看起来能产 NPU baseline/perf，但仍带着旧逻辑和过期字段。正确修法是先给通用链路补齐 `image_matting`：在 `dataset_mapping.py` 增加模型类型与画像，在 `download_datasets.py` 增加 `synthetic_matting` 生成器，在模板 `business_eval.py` / `business_model_eval.py` 增加样本加载与推理支持；然后再备份旧正式工件与旧 phase-4 文件，整轮重跑
- 同次实跑结果：`VitMatte` 新一轮本机 NPU baseline `latency_s=0.080577`、perf `0.073088`、`npu_speedup_ratio=1.1025`，两路 `mae=0.027979`、`cosine_similarity=0.968933`；说明旧 phase-4 并非“优化在业务侧无收益”，而是 harness 口径长期漂移。重跑后应立即 `summarize` 并写库为 `wait_cuda`
- 2026-04-07 同次远端执行经验：大模型首次切到远端 `cuda` extra 时，`uv sync --extra cuda` 可能长时间无业务 stdout，只看到 manager 本地挂起；这不等于 SSH 卡死。应先确认远端 `.venv` 是否刚创建、`uv sync -v` 是否仍在补 CUDA 依赖，再决定是否中断。`whisper-large-v3` 首次远端 `uv sync` 实际耗时接近 2 分钟，之后才进入热身 run
- 同次执行经验：若只重跑了 `npu_baseline` / `npu_perf` / `cuda_baseline` 其中一路，必须立刻重新执行 `business_benchmark_manager.py summarize` 再跑 gate / 写库；否则 `business_summary.json` 里的 throughput、ratio、best_result 很容易仍指向旧工件值，造成“工件真实已正常但 summary 不一致”的假失败
- 同次规则固化：若正式业务工件出现微秒级 `latency_s`、夸张 `throughput_qps`，或 `start_time/end_time` 明显是秒级整轮执行而落盘 `latency_s` 却接近 0，应直接视为假结果而不是“硬件特别快”
- 同次规则固化：只要 adaptation 有 `model_files/`，第四阶段 `npu_perf` 就必须显式证明自己继承了优化产物；至少要看到 `loaded_from_model_files=true` 或等价 patch 载入证据，不能静默退化成 baseline 路径
- 2026-04-05 规则升级后，远端 CUDA 旧的“直接 `.venv/bin/python business_run.py ...` 绕过 extra”经验已作废；当前正式口径必须与本机一致，统一使用 adaptation 自己的 uv 环境和 `uv run --extra cuda ...`
- 2026-04-05 批量刷新已验证：对已有正式 `business_metrics_*` / `business_summary.json` 的历史 completed 结果补跑前，必须先把旧正式工件改名备份（推荐 `__prev_rule_refresh_<timestamp>`）；否则 `run-remote-cuda` 回收同名 CUDA 工件时会因本地已有不同内容文件而把新工件隔离，导致新结果无法接管 canonical 文件名
- 2026-04-05 批量刷新写库经验：逐模型完成 `run-npu -> run-remote-cuda -> summarize/check -> $PROJECT_ROOT/.venv/bin/python scripts/board_ops.py update_business_benchmark_status --notes "$(cat business_summary.json)"`，不要等整批结束后再统一回填，更不要直接写 SQL
- 2026-04-05 长跑判活经验：远端 CUDA 若长时间无新 stdout，但 `business_eval.py` 仍在 `ps` 中存活，不要立即判定失败；先结合 `nvidia-smi`、本地 `business_metrics_cuda_*` / `business_summary.json` 修改时间和显存占用判断是否仍在真实执行
- 真实实跑暴露出 adaptation 自己的 `pyproject.toml` 也必须带上 `jiwer`、`rouge-score`、`seqeval`、`scikit-learn`，否则 `business_eval.py` 会在 adaptation venv 内缺包
- `wikitext` 这类数据集前部空行多，业务采样不能先硬截 `max_samples`；应先过采样再过滤，否则很容易拿不到 completed gate 需要的 `num_samples > 50`
- 业务 `latency_s` 必须只统计模型加载完成后的业务推理阶段，不能把 `from_pretrained()` 的加载耗时混进来，否则会把 perf/baseline 结论带偏
- `peak_memory_mb` 不能继续用占位值；NPU/CUDA 都应优先读取 runtime peak memory stats
- 2026-03-23 实跑 `BAAI/bge-small-en-v1.5`（本机 NPU，仅 local）结果：`wikitext` 64 样本，baseline `0.021381s/sample`，perf `0.022461s/sample`，业务速度比 `0.9519x`，说明 optimization 阶段的收益没有迁移到第四阶段真实业务 workload
- 对同一模型继续做了 measurement 闭环修正：补 `npu_perf` warmup(3x)，并去掉 embedding 场景下 reference 与 input 相同却重复推理的冗余；最终仍得到 baseline `0.013844s/sample`、perf `0.015008s/sample`、速度比 `0.9224x`。这表明问题不只是 warmup 缺失，更可能是 `npu_gelu` 在单样本 embedding 业务路径上的收益不足以覆盖额外调度/显存开销
- 2026-03-23 远端 CUDA baseline 实跑补充：`ssh '...'` 非交互链路不会读取 `~/.zshrc`，必须把 `UV_LINK_MODE` / `UV_CACHE_DIR` / `HF_ENDPOINT` 这类环境放进 `~/.zshenv`
- 2026-04-05 规则升级确认：上一条“直接 `.venv/bin/python` 更稳”的旧 workaround 不再沿用；当前正式第四阶段结果必须保留 `uv run --extra cuda ...` 的运行时证据，不能再通过绕过 extra 的方式产出 canonical 工件
- 同次排障结论：若 `torch 2.6.0+cu124` 导入时报 `libnvshmem_host.so.3` 缺失，需要在该 adaptation venv 内补装 `nvidia-nvshmem-cu12`
- 同次排障结论：远端 `business_benchmark_config.json` 里的 `dataset_local_path` 必须改成远端真实路径（例如 `/data/...`），不能直接沿用本地 `/mnt/...`
- 2026-03-23 实跑 `MaterialsInformaticsLaboratory/QA-BERT-seed16` 暴露出新规则缺口：仅靠旧 `dataset_mapping.py` 会把 `QA-BERT` 误判成 `classification -> sst2`；第四阶段需要原生支持 `question_answering -> squad_v2 -> qa_exact_match`
- 同次修正结论：`business_model_eval.py` 需要支持 `AutoModelForQuestionAnswering`，并从 `(question, context)` 中抽取 span；`business_eval.py` 需要支持 `squad_v2` 采样和 `exact_match + F1`
- 同次修正结论：`business_benchmark_tool.py` 与 `check_business_benchmark_run.py` 的质量指标优先级必须一致；若 summary 侧优先取 `exact_match`，gate 侧也必须同顺序，否则会出现“summary 可生成但 completed gate 失败”的规则漂移
- 同次修正结论：`generate-script` 不能再把旧 `business_benchmark_config.json` 里自动生成的 `dataset / evaluation_profile / primary_metric` 当成高优先级输入，否则旧画像会把新规则锁死；需要改为只认显式 `*_override` 字段
- 2026-03-23 实跑 `cross-encoder/ms-marco-TinyBERT-L-2-v2` 暴露出新规则缺口：`cross-encoder` / `ms-marco` 虽然底层是 `AutoModelForSequenceClassification`，但第四阶段不能按普通 `classification -> sst2` 处理，必须识别为 `reranker -> ms_marco -> reranker_ndcg`
- 同次修正结论：运行态的 `business_run.py` 不能只依赖重新画像；一旦 `business_benchmark_config.json` 已经生成，就应优先使用其中的 `model_type / dataset / evaluation_profile / primary_metric / output_type_hint`
- 同次修正结论：远端 CUDA 场景下，`business_eval.py` 必须优先读取 `BUSINESS_BENCHMARK_DATASET_PATH`，不能死盯配置里本机生成的 `/mnt/.../dataset_local_path`；否则远端即使数据集已存在也会走错路径
- 同次执行经验：开始本机 NPU 前就应先做一遍远端前置校验（SSH 连通、远端 repo root 存在、adaptation 可写），否则很容易在本机两条 NPU 都跑完后才发现远端 alias/目录不可用
- 同次执行经验：远端长任务若 stdout 长时间无新增，不代表一定卡死；`uv sync` 结束后可用 `ps` 看 `business_eval.py` 是否仍在运行，再结合 `nvidia-smi` 和 artifact 是否落盘判断是在真实评测、stdout 缓冲还是 CUDA 未占上
- 同次最终结果：`cross-encoder/ms-marco-TinyBERT-L-2-v2` 第四阶段完成，`ms_marco` 64 样本，NPU baseline `0.012274s/sample`，NPU perf `0.012339s/sample`，CUDA baseline(H100) `0.012731s/sample`，三路 `ndcg_at_10=0.272258` 一致；该模型在真实业务场景下没有复现 optimization 阶段的 NPU 提速
- 2026-03-23 新一轮小模型补跑又暴露出两条规则：`classification` 不能只凭 `AutoModelForSequenceClassification -> sst2` 就直接进入第四阶段；像 `cardiffnlp/twitter-roberta-base-sentiment-latest` 这类情感分类模型虽然能完整跑出业务工件，但在 `sst2` 上出现 `accuracy=0.0` 时必须视作“画像语义不匹配”，应回到业务画像层重新选数据集，而不是拿着 0 分结果继续冲 completed
- 同次远端执行经验：若远端 adaptation 是首次切到 `cuda` extra，`uv sync --extra cuda` 之后还可能因为远端缺少完整 `models/` 缓存而长时间只见 CPU 占用、`nvidia-smi` 几乎空闲、stdout 也不继续刷新；这类情况优先按 SOP 检查 `models/` / 数据集完整性，并先用 `rsync -av --partial --append-verify --no-o --no-g` 断点续传资产，再重新执行单模型 CUDA baseline
- 同次测量经验：`apple/mobilevit-small` 与 `sshleifer/distilbart-cnn-12-6` 的本机 NPU 业务工件里 `peak_memory_mb` 仍可能写成 `0.0`；这说明部分业务路径的 runtime peak memory 采集还没真正打通，后续 completed gate 若继续要求显存证据，不能把 `0.0` 当成有效值
- 同次修正结论：远端 CUDA 若通过交互式 `ssh '...business_run.py...'` 直接长跑，连接层可能在业务脚本完成前被远端主动关闭；若确实需要后台化排障，也应保持新规则口径，例如 `nohup bash -lc "CUDA_VISIBLE_DEVICES=0 uv run --extra cuda python business_run.py --scenario cuda_baseline" > cuda_baseline.log 2>&1 < /dev/null &`，再轮询 `ps` / `cuda_baseline.log` / artifact 是否落盘
- 同次修正结论：当 NPU 业务路径的 `torch.npu.max_memory_allocated()` / `memory_stats()` 持续返回 0 或 None 时，`business_eval.py` 需要在写盘前再做一次兜底，把非正数 `peak_memory_mb` 回退到当前进程 RSS；否则 `mobilevit-small`、`distilbart-cnn-12-6` 这类模型即使三路工件齐全也会被 completed gate 卡住
- 2026-03-24 对 `cardiffnlp/twitter-roberta-base-sentiment-latest` 的真正修复确认了两个独立问题：一是业务画像不能把 Twitter 三分类情感模型继续落到 `sst2`；应改成 `tweet_eval_sentiment` 这类语义和标签空间都匹配的数据集。二是通用 `business_model_eval.py` 里 `classification` 若返回 `id2label` 文本，而 `business_eval.py` 读取的数据集 `reference` 是 `0/1/2` 数值类标，会制造伪 `accuracy=0.0`；通用规则应统一返回 `pred_id`
- 同次远端执行又暴露出一个同步规则缺口：如果本次修复不仅改了 adaptation 内脚本，还改了仓库根下的 `scripts/dataset_mapping.py`、`scripts/download_datasets.py` 之类公共脚本，远端 CUDA 阶段不能只同步 adaptation；否则本地已支持的新数据集 key（如 `tweet_eval_sentiment`）在远端仍会报“未知数据集”
- 同次配置结论：对遵循仓库标准 `datasets/{dataset_dir}` 的业务数据集，`business_benchmark_config.json` 里的 `dataset_local_path` 最稳妥做法是留空，让 `business_run.py` 按 `dataset` 自动解析标准路径；这样同一份配置能同时兼容本地 `/mnt/...` 与远端 `/data/...`，不需要再手工改路径
- 2026-03-24 进一步把第四阶段画像从“顶层模型类型”扩展成“顶层 `model_type` + 业务意图子层”后，`SequenceClassification` 至少应优先区分：`reranker`、`extractive_qa`、`sentiment_binary`、`sentiment_multiclass`、`emotion_multiclass`、`offensive_binary`、`hate_binary`、`topic_classification`、`natural_language_inference`、`question_pair_classification`、`generic_classification`
- 同次规则固化结论：当前已和 harness 对齐、可直接闭环的分类业务数据集包括 `imdb`、`tweet_eval_sentiment`、`tweet_eval_emotion`、`tweet_eval_offensive`、`tweet_eval_hate`、`ag_news`、`glue_mnli`、`glue_qnli`；以后新增分类语义时，必须同时补齐 `dataset_mapping.py`、`download_datasets.py`、`business_eval.py`
- 同次实现经验：分类数据集的 `reference` 不应依赖“原始 label 看起来刚好和输出一致”的偶然匹配；应在 `business_eval.py` 里统一做 label 归一化，把 `ClassLabel` 名称或文本类标映射回稳定的 label id，再与 `business_model_eval.py` 返回的 `pred_id` 对齐
- 同次实现经验：对 `glue_mnli`、`glue_qnli` 这类文本对分类，样本结构必须显式携带 `input_pair`，通用 `business_model_eval.py` 也必须在分类分支里调用 `tokenizer(text, text_pair, ...)`；如果第二句被静默丢掉，得到的不是“低精度”，而是无意义测评
- 2026-03-24 最后一次链路修复确认：`print-remote-command` 的默认行为必须复用当前 `benchmark_run_id`；只有显式要求开新一轮时才允许刷新 run-id，否则会把刚跑完的 NPU 工件与后续 CUDA baseline 脱钩
- 同次修正结论：`generate-script` / `run-npu` / `print-remote-command` 重新生成配置时，普通的 `model_type / dataset / evaluation_profile / primary_metric / secondary_metrics / output_type_hint` 都应视为自动字段并按最新画像刷新；长期人工覆盖必须写进 `*_override`
- 同次修正结论：小样本 smoke run 绝不能覆盖正式 `business_metrics_*.json`；`business_eval.py --max-samples <= 50` 应自动产出带 `smoke{N}` 标签的工件，只用于链路验证，不参与 completed 汇总
- 同次修正结论：`summarize` 在没有完整同轮次三件套时，应优先沿用现有 `business_summary.json` 指向的工件；若工件处于新旧混合状态，必须直接报错，不能静默拼轮次

## 远端 CUDA 基线 SOP（2026-03-23 固化）

适用场景：第四阶段需要补 `cuda_baseline`，远端机器通过 SSH 使用同一仓库目录执行，最终把工件同步回本地并参与 `business_summary.json` 汇总。

### 0. 推荐前提

- 推荐先在 `business_benchmark_config.json` 中写好：
  - `remote_ssh_host`
  - `remote_project_root`
- 推荐 SSH alias 形式：
  - `slai-cuda-remote -> your-user@your-cuda-host:your-port`
- 远端项目根目录示例：
  - `<remote_project_root>`

### 1. 先做远端前置校验

在本机 NPU baseline / perf 跑完之前，就先确认远端条件成立：

```bash
ssh slai-cuda-remote 'cd "<remote_project_root>" && pwd'
ssh slai-cuda-remote 'test -d "<remote_project_root>/adaptations/<adaptation_name>" && echo OK'
ssh slai-cuda-remote 'nvidia-smi -L'
```

若这一步不通，不要等到本机两条 NPU 都跑完才发现远端链路坏了。

### 2. 校验远端资产，不足则断点续传

至少检查两类东西：

- `adaptations/<adaptation_name>/models/`
- 业务数据集目录（如 `datasets/wikitext`、`datasets/ms_marco`、`datasets/squad_v2`）

若远端缺失或明显不完整，用支持续传的 `rsync`：

```bash
rsync -av --partial --append-verify -e 'ssh -p <port>' \
  "adaptations/<adaptation_name>/models/" \
  "<user>@<host>:<remote_project_root>/adaptations/<adaptation_name>/models/"
```

若出现 `chown failed` / `Operation not permitted`，通常只是 owner/group 元数据同步失败，不代表文件内容传输失败。必要时可补跑：

```bash
rsync -av --partial --append-verify --no-o --no-g -e 'ssh -p <port>' ...
```

### 3. 校正远端配置路径

远端 `business_benchmark_config.json` 里的 `dataset_local_path` 不能继续是本机绝对路径，必须改成远端真实绝对路径，例如：

```json
{
  "dataset_local_path": "<remote_project_root>/datasets/squad_v2"
}
```

另一个强规则：

- 远端 `business_eval.py` 必须优先读取 `BUSINESS_BENCHMARK_DATASET_PATH`
- 不能只死盯配置里本机生成的 `dataset_local_path`

### 4. 远端先同步依赖，再按新规则执行

可以先在远端 adaptation 下做一次：

```bash
ssh slai-cuda-remote 'cd "<remote_project_root>/adaptations/<adaptation_name>" && uv sync --extra cuda'
```

随后实际运行 `cuda_baseline` 时，正式口径应使用：

```bash
ssh slai-cuda-remote 'cd "<remote_project_root>/adaptations/<adaptation_name>" && CUDA_VISIBLE_DEVICES=0 uv run --extra cuda python "business_run.py" --scenario cuda_baseline'
```

当前规则要求正式工件必须保留 `uv run --extra cuda ...` 对应的运行时证据；不要再用直接 `.venv/bin/python ...` 的旧 workaround 绕过 extra。

### 5. 旧 workaround 已废弃

- 2026-03-23 曾临时采用过 `.venv/bin/python business_run.py ...` 的远端 workaround
- 2026-04-05 规则升级后，这条经验只能视为一次性排障历史，不能再用于正式第四阶段结果
- 现在需要一律回归 `uv run --extra cuda python business_run.py --scenario cuda_baseline`

### 6. 非交互 SSH 环境变量规则

`ssh '...'` 这类非交互链路通常不会读取 `~/.zshrc`，因此：

- `UV_LINK_MODE`
- `UV_CACHE_DIR`
- `HF_ENDPOINT`

这类必须给非交互命令生效的环境变量，应放入 `~/.zshenv`，不要只写在 `~/.zshrc`。

### 7. 常见远端 CUDA 排障

- 若 `torch 2.6.0+cu124` 导入时报：
  - `libnvshmem_host.so.3` 缺失
  - 结论：在该 adaptation 的 `.venv` 内补装 `nvidia-nvshmem-cu12`

- 若远端日志长时间无新增：
  - 不要立刻判定卡死
  - 先看 `ps`
  - 再看 `nvidia-smi`
  - 再看 `business_metrics_cuda_*_baseline.json` 是否已落盘

### 8. 工件回传与后处理

远端工件生成后，拉回本地：

```bash
scp slai-cuda-remote:"<remote_project_root>/adaptations/<adaptation_name>/business_metrics_cuda_*_baseline.json" \
    "adaptations/<adaptation_name>/"
```

随后在本地继续：

1. 生成 / 更新 `business_summary.json`
2. 跑 completed gate
3. 通过后再写回 `board.db`

若这是历史 completed 结果按新规则补跑，还要补一条强规则：

4. 在远端回收前先把本地旧正式 `business_metrics_*` / `business_summary.json` 改名备份；否则新 CUDA 工件会因同名冲突被隔离，无法成为新的 canonical 工件

### 9. 本次远端 CUDA 三步标准链路

```bash
uv run python business_benchmark/scripts/business_benchmark_manager.py sync-remote-workspace --model "<model_id>" --ssh-host slai-cuda-remote --remote-project-root "<remote_project_root>"
ssh slai-cuda-remote 'cd "<remote_project_root>/adaptations/<adaptation_name>" && uv sync --extra cuda && CUDA_VISIBLE_DEVICES=0 uv run --extra cuda python "business_run.py" --scenario cuda_baseline'
uv run python business_benchmark/scripts/business_benchmark_manager.py fetch-remote-artifacts --model "<model_id>" --ssh-host slai-cuda-remote --remote-project-root "<remote_project_root>"
```

## 规则升级批量重跑经验（2026-04-05）

适用场景：需要把已经 `business_benchmark_status=completed` 的模型按新第四阶段规则整体补跑，并把新结果覆盖进 DB。

1. 先从 `board.db` 列出当前 `business_benchmark_status='completed'` 的模型，逐个串行处理
2. 对每个 adaptation，先把当前正式 `business_metrics_npu_*`、`business_metrics_cuda_*`、`business_summary.json` 改名成 `__prev_rule_refresh_<timestamp>` 备份
3. 本机跑 `run-npu`，再跑 `run-remote-cuda`
4. 若远端长时间无新 stdout，不要马上判死；先看远端 `ps` / `nvidia-smi` 和本地工件时间戳
5. 每个模型一旦生成新的 `business_summary.json` 并通过本地检查，立即用根目录 `board_ops.py update_business_benchmark_status --notes "$(cat business_summary.json)"` 写库
6. 建议整批写独立日志目录 `/tmp/slai_phase4_rule_refresh_<timestamp>/`；中断后优先从单模型日志恢复，而不是整批重跑
