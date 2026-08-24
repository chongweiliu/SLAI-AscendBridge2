# 音频-LLM（audio-text-to-text）继续预训练 CPT

> 适用：`audio-text-to-text` 类模型——音频理解 LLM（音频输入→文本输出），如 Qwen2-Audio、Qwen-Audio、SALMONN、Whisper-based audio-LLM。
> 典型组件：`audio_tower`(Whisper 等音频编码器) + `multi_modal_projector` + `language_model`(CausalLM) + tokenizer，常无独立 `model_index.json`（`Qwen2AudioForConditionalGeneration` 等类）。
> 本文从 Qwen2-Audio-7B-Instruct + fleurs en_us 实战提炼（2×910C 模型并行全量 CPT 200 步，held-out CE 0.810→0.145）。

## 核心方法论（先内化，再动手）

音频-LLM CPT 与文本 LM / 扩散 CPT 是**不同范式**，但骨架一致——"拆出可训练 backbone(LLM) + 冻结编解码器(audio_tower+projector)，用原生损失(转写 CE)训练"：

1. **判定格式**：是标准 PyTorch/transformers，还是 GGUF/MLX/量化？GGUF/MLX NPU 无法训练（坑 #47，须换同源 PyTorch 基座）。
2. **拆组件**：可训练 = `language_model`(CausalLM，全量或 LoRA)；冻结 = `audio_tower`(音频编码器) + `multi_modal_projector`(音频→LLM 维度投影)。只训 LLM。
3. **数据流水线**：音频(flac/wav 16k) → `AutoProcessor` 算 mel(`input_features`) + chat template 组 prompt(audio+transcribe 指令) → token ids；text_encoder 即 LLM 自身(转写 CE)，无需额外 text encoder。
4. **损失**：转写 CE loss——chat template user(audio+"Transcribe")+assistant(transcript)，**labels mask prompt 段 + audio 特殊 token + pad**，只对 assistant(transcript) 算 loss。
5. **评估**：base vs CPT 在 held-out 音频+转写上算 **CE loss**(delta<0 有效)；定性可采样生成转写算 WER。**不要套 PPL/next-token 全序列**。

## 全流程（9 阶段，与 SKILL.md 对齐）

### 阶段 0-1 · 判定 + 环境
- **范式判定**：`config.json` `architectures` 含 `Qwen2AudioForConditionalGeneration`/`Audio`/`Whisper`+LLM，`pipeline_tag=audio-text-to-text` → **音频-LLM 范式**。
- **格式判定**：`*.gguf`/`mlx_*` → GGUF/MLX，NPU 不可训练，换同源 PyTorch 基座（坑 #47）。如 `NexaAI/OmniAudio-2.6B` 是 GGUF-only + Gemma 门控 → 换 `Qwen2-Audio-7B-Instruct`(PyTorch safetensors)。
- 依赖：`pip install peft soundfile datasets`（音视频解；decord/librosa 非必需，soundfile 读 flac→numpy→processor 算 mel 即可，MLS/fleurs 是 16k 无需重采样）。
- **libgomp TLS（坑 #59）**：transformers 5.x 的 generation 间接 `import sklearn`，sklearn 的 libgomp 与 torch_npu OMP 冲突 `cannot allocate memory in static TLS block`。**脚本顶部先 `import sklearn`**（早入静态 TLS）再 torch_npu。
- cgroup 检查：audio_tower(Whisper) forward 激活大，全上 NPU 不在 CPU 堆（坑 #44/#46）。

### 阶段 2 · 获取
- 模型：PyTorch safetensors 版（未门控优先）。GGUF-only 的换同源 PyTorch 基座。
- 数据：MLS English 在 HF `facebook/multilingual_librispeech` **无 english config**（只有 dutch/.../spanish）；无精确"MLS English 10k"。用 `google/fleurs` en_us（英语语音+转写，16kHz，parquet 直下，含内嵌音频字节）作英语语音语料替代。parquet 优于 streaming（streaming 对大数据集 list 慢/超时，坑 #48）。

### 阶段 3 · 数据流水线（用 `prepare` 内联或 `cpt_audio_llm.py.tmpl`）
- 读音频：`soundfile.read(io.BytesIO(audio_bytes))` → numpy float32 16k（flac 字节在 parquet 的 `audio` 列 `bytes` 字段）。
- chat template：`proc.apply_chat_template([{user:[{type:audio},{type:text:"Transcribe..."}]},{assistant:transcript}], add_generation_prompt=False)`。
- **processor 批量（坑 #56）**：`proc(text=[full,...], audio=[arr,...], padding=True, return_tensors="pt", sampling_rate=16000)`——kwarg 是 **`audio=`（单数，非 `audios=`）**；返回 `input_ids`+`attention_mask`+`input_features`(mel `[B,128,~3000]`)+`feature_attention_mask`。单样本无 padding：`audio=[arr]`。
- **labels mask（坑 #58，必读）**：`labels=ids.clone()`；① mask pad `labels[attention_mask==0]=-100`；② mask prompt 段（用 prompt-only tokenize 的 `attention_mask[i].sum()` 作 per-sample prompt len）；③ **显式 mask audio 特殊 token** `<|audio_bos|>/<|AUDIO|>/<|audio_eos|>`（`labels[ids==audio_id]=-100`）——它们是条件输入不该当预测目标，**漏 mask 会致 loss 虚高 ~8×**（bs≥2 + padding 时尤甚，因 audio token 落到非 prompt 区）。

### 阶段 4-5 · 选型 + 超参（**自动**, 对齐 parallel-strategy.md / hyperparam-selection.md）
**通用模型识别**：`cpt_audio_llm.py.tmpl` 按 `config.architectures` 自动识别音频-LLM 类（`Qwen2AudioForConditionalGeneration`/`QwenAudioForConditionalGeneration`/… 映射表，回退 `AutoModelForConditionalGeneration`）；audio 特殊 token 从 tokenizer 自动发现（名含 "audio" 的特殊 token，不写死 Qwen2-Audio 的 3 个）。换音频-LLM 不改模板。

**自动并行**（探测 `device_count`+`mem_get_info` free，估模型显存 `bf16 master≈8×P`/`fp32 master≈16×P` GB，见 parallel-strategy.md 选型表）：
- `8×P ≤ free×0.8`（单卡装得下）→ **单卡**。
- 单卡装不下 + `n_card≥2` 且 `8×P/2 ≤ free×0.8` → **2 卡模型并行 `device_map+max_memory`**（`max_memory` 按每卡 `free×0.5` 自动算，不写死）。
- 仍装不下 → **LoRA 回退**（冻结 base，训 q/k/v/o，~10M 可训；`peft` 不可用则退单卡全量并告警可能 OOM）。
- 可 `MODE=single|mp2|lora` 强制覆盖（默认 `auto`）。2 卡须 shell `export ASCEND_RT_VISIBLE_DEVICES=0,1`（#59）。

**自动超参**（对齐 hyperparam-selection.md）：
- lr = `1e-5 × sqrt(global_batch/32)` clamp `[5e-6, 5e-5]`；warmup ~10% 步数；cosine 衰减。
- bs：全量 CPT 大模型默认 1，LoRA 默认 2；`BATCH_SIZE=0` 自动；**OOM 回退阶梯**：bs→//2→1，仍 OOM 跳过该步（torch.npu.OutOfMemoryError 捕获）。
- grad_accum 凑有效 batch（`effective = bs×accum×n_card`）。
- bf16 master（fp32 master `16×P` 在 2×64GB 多数 OOM，#60）；grad-ckpt(`use_reentrant=False`)。
- 可 env 覆盖：`NUM_STEPS/LR/WARMUP/BATCH_SIZE/GRAD_ACCUM/N_KEEP`（0=自动）。

### 阶段 6 · 脚本 + smoke（用 `cpt_audio_llm.py.tmpl`）
- load model → freeze `audio_tower`+`multi_modal_projector`(按**子模块路径名**，勿用类名"Audio"——会误冻顶层 `Qwen2Audio*`，坑 #59) → grad-ckpt LLM → AdamW(非 foreach/fused，多 device 参数；非 NpuFusedAdamW，模型并行跨 device 不兼容)。
- **model forward（坑 #57）**：`model(input_ids=, attention_mask=, input_features=, feature_attention_mask=, labels=)`——必须传 `input_features` **和** `feature_attention_mask`（只传 input_features → NoneType.to 报错）。
- smoke 顺序：先单组件（dummy mel→audio_tower；dummy ids+emb→LLM 前向反向）再合练 2 步。

### 阶段 7 · 正式训练
- 每步：build batch(audio+transcript, label mask)→ forward → CE loss → backward → step。心跳/用时表同文本 CPT。存 `cpt_state.pt`(state_dict, bf16 减半) + `train_summary.json` + `losses.json`。
- 画 loss 曲线 + 公网直链（plot_loss.py 通用）。

### 阶段 8 · 评估（用 `eval_audio_llm.py` 同构）
- **量化 CE loss**：held-out 音频+转写，base(全新/ad adapter off) vs CPT(加载 state_dict，`load_state_dict(strict=False)` 验 miss/unexp=0)，固定同 σ/noise seed 公平。`delta<0` 训练有效。
- 定性：采样生成转写 → 算 WER（需 jiwer/edit-distance；可选）。

## 踩坑速查（详见 pitfalls.md #55-60）
- #55 libgomp TLS(sklearn+torch_npu) → import sklearn 先
- #56 processor kwarg `audio=`(单数) + 返回 `input_features`/`feature_attention_mask`
- #57 model forward 须传 `input_features` + `feature_attention_mask`
- #58 audio 特殊 token 必 mask in labels（漏则 loss 虚高 ~8×）
- #59 model-parallel device_map 不自动分卡需 max_memory；ASCEND_RT_VISIBLE_DEVICES=0,1 两卡；freeze 按子模块路径勿用类名"Audio"
- #60 7B 全量 CPT 2×64GB: DDP bf16 OOM, fp32 master OOM; 模型并行 bf16 fits

## 实战数据点（Qwen2-Audio-7B-Instruct + fleurs en_us）
- **2×Ascend910C 64GB, 模型并行 device_map+max_memory, bf16 master, 7.755B 全量可训(audio_tower+projector 642M 冻结)**。
- 50 步 bs1(75 样本): first5→last5 0.473→0.109, held-out CE 0.810→0.218 (-73%)。
- **200 步 bs2(1301 样本=全量一半): first5→last5 2.251→0.272 (-88%), held-out CE 0.810→0.145 (-82%)**。扩数据+加步数+bs2 显著优于 50 步(0.218→0.145)。
- 全程 NPU 算子通过（audio_tower Whisper + LLM attention）；~0.83s/step(bs2)。
