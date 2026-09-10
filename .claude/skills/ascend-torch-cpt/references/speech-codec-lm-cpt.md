# speech-codec-token LM CPT（TTS talker：Qwen3-TTS / CosyVoice 类）

> 判定特征：config 无 `model_index.json`、非 ForCausalLM 文本头；模型内含 **talker（只预测 codec 流第 1 码本）
> + code predictor（多头各预测 2~16 码本）**，文本仅作条件输入轨；语料为音频+文本对（TTS 数据）。
> 本范式 = 把「文本+音频 codec token 交错流」当语料，对**整个 codec 流**做 next-token CE（CPT 全 token 口径）+
> 多码本 sub-head CE。完整实证：Qwen3-TTS-12Hz-1.7B-Base + aishell3 1000 条，Ascend910 单卡，
> held-out codec PPL 87428 → 4.96、next-token acc 0.9% → 59.1%（2026-09-07）。

## 1. 权重与运行时
- 用户权重若是 **GGUF → 不可训**，解析 GGUF 头拿 `general.name` 定位官方仓，ModelScope 直链拉 HF 权重。
- `qwen-tts` 包（pin `transformers==4.57.3`）内含全部模型类，运行时 `AutoConfig.register("qwen3_tts", ...)`；
  `Qwen3TTSTokenizer` 必须指向 **`speech_tokenizer/` 子目录**。
- 模型加载：`Qwen3TTSModel.from_pretrained(dir, dtype=torch.float32, attn_implementation="sdpa", device_map="npu:0")`
  ——flash-attn 不可用；**fp32 主权重 + bf16 autocast**（纯 bf16 前向 #4）。`.model` 即 `Qwen3TTSForConditionalGeneration`。

## 2. 数据准备（`prep_speech_codec_data.py.tmpl`）
- 目标音频：`Qwen3TTSTokenizer.encode(wav路径)` **离线**批量编码（内部自动重采样 24k）→ `audio_codes` [t,16] 存回 jsonl。
  12Hz codec：帧率 12.5Hz、16 码本、码本词表 2048（+1024 特殊 token = talker codec0 词表 3072）。
- ref 音频：重采样 24kHz + **定长 3.0s**（截断/补零）→ 官方 mel（n_fft=1024, 128 mels, hop=256, fmax=12000）
  预算成单个 `.npy` 缓存（训练零音频 IO）。
- 切分：尾部固定 N 条 held-out（如 950/50），**绝不参与训练**；写入独立 jsonl 供评估。

## 3. 双轨序列布局（官方 collate 数学，照抄可信）
每样本一个位置一个 `(text_id, codec_id)` 对，T = max(len)+8：
- 文本轨：`[im_start,assistant,\n] + tts_pad×4 + tts_bos + text_body + tts_eos + tts_pad...`
- codec 轨：位置 3-7 = `[nothink, think_bos, think_eos, 0(speaker槽), codec_pad]`，文本区 codec_pad，
  `codec_bos`，音频 codec0 帧，`codec_eos`；**位置 6 在 embedding 相加后整体替换为 speaker_encoder(ref_mel) 的输出**（detach）。
- 附加 embedding：音频帧位置加 code predictor 的 15 个码本 embedding（codec_mask 掩码）。
- `codec_embedding_mask`（codec 轨有效区，位置 6 置 False）即 loss 的 valid 区来源。

## 4. loss 配对（本范式最大坑）
- **talker**：传**全长 inputs_embeds + 全长 own-position labels**（valid 处 = codec 流本位值，其余 -100）。
  `ForCausalLMLoss` 的 pad+内部 shift 恰好给出 logits[t]↔stream[t+1] 标准 next-token——与 generate() 契约一致
  （首个采样 token 来自 prefill 最后位置 hidden）。**勿学官方 sft_12hz.py 再 `[:, 1:]`**（双重错位成 t→t+2）。
- **sub-talker**：`forward_sub_talker_finetune` 的 labels 路径同样错位；复刻其 embedding 拼装
  （[hidden, c0, c1..c14]），`forward_finetune(labels=None)` 拿 logits 后**恒等配对 CE**
  （logits[j] = group j+1 | hidden, c0..c_j，与生成逐组条件一致），targets = 该帧 codec_ids[:, 1:]。
- 总 loss = talker CE + 0.3 × sub-talker CE。判据法：基座上分 t→t+1 / t→t+2 两个配对测 CE，低者为预训练目标，
  再以 generate() 采样代码终审。

## 5. 训练（`cpt_speech_codec.py.tmpl`）
- 全参（1.93B：talker 28L hidden2048 + code_predictor 5L hidden1024 + embeddings）；speaker_encoder/speech_tokenizer 无梯度自然冻结。
- NpuFusedAdamW betas(0.9,0.95) wd=0.01 eps=1e-8 clip=1.0；lr 1e-5 cosine warmup 10%；batch 8。
- 显存：fp32 权重 7.7G + speech_tokenizer fp32 2.7G + 优化器 15.4G + 激活 ≈ **峰值 43G / 稳态 29G**（64G 卡）。
- 实测步时：首步 ~32s（编译+优化器初始化），稳态 **0.68s/step**（seq≈52 tokens, batch 8）——smoke 首步耗时不可外推 #105。
- 产物三件套：`cpt_model_state.pt`（bf16 评估用）/ `ckpt_latest.pt`（含优化器）/ HF 目录
  （`save_pretrained` 对该包 config 会撞 `KeyError: 'dtype'`，用 `safetensors.torch.save_file` 直写 + 拷贝配置文件兜底）。

## 6. 评估（codec PPL / next-token acc / sub-head CE）
- teacher-forced：同 §4 配对，统计 valid 位的 CE→PPL=exp(CE)、top-1 acc；sub-head CE 同口径单列。
- **协议锚定注意**：base 在 custom-voice 布局上 CE 很高（~11 nats）不是权重坏了——base 预训练是 ICL 格式
  （ref codec token 进流），本 CPT 同时做「格式适配 + 音色域适配」，这正是任务目标；自洽性锚点 =
  base 在训练集上的 CE ≈ 训练起步 loss（实测 10.78 vs 10.84）。
- 过拟合检查：train 末端 CE vs held-out CE 存在间隙属预期（千级样本×3 epoch 全参），红旗仅是 held-out 不改善。
- 音频 codec 流 top-1 acc 59%（2048 类）即强预测——音频 token 天然高熵，勿用文本 acc 直觉评判。

## 7. 实证工作区
`training-ws/qwen3-tts-cpt/`（脚本、数据缓存、ckpt、eval_results.json、README 均在，training-ws 不入库）。
