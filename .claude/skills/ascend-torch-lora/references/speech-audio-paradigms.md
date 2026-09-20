# 语音与音频三范式 LoRA（asr / audio_cls / tts）

> 场景：目标是语音识别（Whisper 系 encoder-decoder）、音频分类（AST/ViT 系）或语音合成
> （SpeechT5 系 text→mel），数据是音频波形（flac/wav/parquet 内嵌字节）。
> 实测：2026-09-18 台账第 6 大类 3 模型，**3/3 改善**——whisper-large-v3-turbo+librispeech
> （CER 0.0091→0.0045，-51%）、ast-finetuned-audioset+esc50（top1 7/10→10/10，CE -91.5%）、
> speecht5_tts+ljspeech（teacher-forced mel MSE -37%，10/10 全改善）。
> 训练器模板：`scripts/lora_train_speech.py.tmpl`、验证模板：`scripts/validate_speech.py.tmpl`
> （范式经 `PARADIGM` 环境变量切换）；完整配套（切分/sweep/val_loss）实战副本在 `lora-ws/.cat6/`。

## 三范式速查

| PARADIGM | 模型加载 | 数据格式（jsonl 每行） | 任务头 | loss | 真实 batch | 验证主指标 |
|---|---|---|---|---|---|---|
| asr | `WhisperForConditionalGeneration` | `{"audio":绝对路径,"text":转写}` | –（预训练） | 转写 token CE（labels pad→-100） | 4（30s mel 重） | **CER**（greedy 生成，双侧规范化）+ ROUGE-L/字符重叠 |
| audio_cls | `ASTForAudioClassification`（num_labels=K + `ignore_mismatched_sizes`） | `{"audio","label":int,"category":str}` | **模型内** classifier（ASTMLPHead：只重初始化 `dense.*`） | 分类 CE | 16 | top1 命中 + CE（base 侧=head-only 基线，见 vision-ts-paradigms.md 协议） |
| tts | `SpeechT5ForTextToSpeech`（`use_guided_attention_loss=False`） | `{"audio","text":normalized_text}` | –（预训练） | teacher-forced mel 重建（labels=完整谱，**内部自动 shift**） | **1×grad_accum 8** | teacher-forced mel MSE（+L1） |

三范式共用文本批次同款骨架：NpuFusedAdamW + bf16 autocast + cosine/warmup + loss.jsonl。

## transformers 5.x 语音栈五坑（全实测修复，模板已内置；同类模型必再遇）

1. **WhisperFeatureExtractor 的 `max_length` 是采样点语义**（pitfalls #48）：
   `padding="max_length", max_length=3000` 不再把 mel pad 到 3000 帧，而是把**音频**截到 3000 采样点
   （→18 mel 帧）→ 模型报 `expects the mel input features to be of length 3000, but found 18`。
   **修复**：逐条 `fe(aud, sampling_rate=16000, return_tensors="np").input_features[0]` 提 mel 后
   **手动 pad/截到 3000 帧**（`np.pad(m, ((0,0),(0,3000-T)))`），版本无关。
2. **SpeechT5 reduction_factor=2 要求偶数帧**（#49①）：奇数帧 labels（如 603）→ 解码器输出帧数
   不匹配报 `size of tensor a (603) must match b (602)`。修复：谱截偶（`spec[:-1] if T%2`）。
3. **SpeechT5 `use_guided_attention_loss` 默认开且对 attention_mask=None 崩**（#49②）：
   `'bool' object has no attribute 'sum'`。修复：`config.use_guided_attention_loss=False`
   （标准 mel 重建不需要该正则）+ 显式传全 1 attention_mask。
4. **SpeechT5SpectrogramLoss 的 BCE pos_weight 停留 CPU**（#49③）：criterion 在 forward 内**现建**
   （`BCEWithLogitsLoss(pos_weight=torch.tensor(5.0))`），buffer 不随模型上 NPU → device mismatch。
   修复：monkey-patch criterion.forward，把 `pos_weight` 搬到 `logits.device`（模板 load_model 内置）。
5. **SpeechT5 5.x 提 mel 用 `audio_target=` 参数**：`fe(audio=...)` 返回的是归一化**波形**
   （键 `input_values`），`fe(audio_target=...)` 才返回 log-mel 谱（键也叫 `input_values`，[T,80]）——
   两个路径同键名不同语义，拿错会在 decoder prenet 报 `mat1 and mat2 shapes cannot be multiplied`。

另：**labels-only 调用**——5.x 的 `SpeechT5ForTextToSpeech.forward` 在 `decoder_input_values=None` 时
由 labels 自动 `shift_spectrograms_right`；手传 decoder_input_values 反而帧数对不上。

## 共性模式（跨范式复用）

1. **mel 预编码缓存（tts）**：重采样（22.05k→16k，`scipy.signal.resample_poly`）+ mel 提取约 0.3s/条，
   按 `md5(DATA_FILE|paradigm)` 缓存到 `<WS>/outputs/mel_cache_<key>.pt`（fp16 存），sweep 各组合与
   val 各自键控复用——与多模态批次的 latent 缓存同一原则（重编码成本高就缓存）。
2. **变长谱不能 padding 混批（tts）**：SpeechT5 的谱 loss 无 padding mask，pad 帧会污染 loss →
   `BATCH=1 + GRAD_ACCUM=8`（有效 batch 8）。通用规律：**loss 不感 mask 的回归型输出，宁可用
   batch1+累积，不要 pad**。
3. **音频加载统一入口**：`soundfile.read`（flac/wav 通吃）→ 多声道取均值 → 目标采样率
   `resample_poly(aud, up, down)`（gcd 约分）。parquet 内嵌字节先落盘再读（切分器负责，绝对路径 #41）。
4. **CER 的双侧规范化**：GT（LibriSpeech 大写无标点）与预测（正常大小写带标点）必须同口径——
   双侧 `lower + 去非字母数字 + 压空格` 后再算字符编辑距离/ROUGE-L；主指标 CER=编辑距离/GT 长度。
5. **generation prompt 显式化（asr）**：`processor.get_decoder_prompt_ids(language="en", task="transcribe")`
   → `forced_decoder_ids`（拿不到则回退默认 generation_config）；greedy、`skip_special_tokens=True`。
6. **speaker 向量口径（tts）**：归档无 xvector 时用 `zeros(1,512)`（单说话人数据集惯例），
   报告必须记录该口径；无 vocoder 归档时验证限定为 **teacher-forced mel 域**，结论表述为
   「文本→mel 重建精度提升」而非「合成音质提升」。
7. **新头范式入场券**：audio_cls 的 classifier 重初始化/peft 冻结/融合优化器/撞名问题与
   vision-ts 批次完全同源——见 `vision-ts-paradigms.md` 共性模式 1-4 与 pitfalls #45-#47、#50。
   **单批过拟合测试**（20 步@1e-3 应显著下降）先行，再进 sweep。

## 各范式实测要点（含结果量级）

- **asr**（whisper-large-v3-turbo 809M + librispeech clean-100 子集池 611）：CER 0.0091→0.0045（-51%），
  7 改/3 平（本就 0 错）/1 微退（assistants→assistance 一词）。**强基座边际改善判读**：LibriSpeech 朗读
  英语是 whisper"主场"（base CER 已 0.9%），LoRA 仍减半错误率且把 4 条非完美样本中 3 条修到零错——
  真实但要在报告里写明 base 已近满分，避免"提升 51%"的误导性标题。LoRA 挂 encoder+decoder 双侧
  （q/k/v/out_proj+fc1/fc2，13.9M/1.7%）。显存 44.7GB（30s mel×batch4 encoder 是大头）。
- **audio_cls**（AST 87M + esc50 池 600，50 类）：top1 7/10→10/10、CE 1.742→0.148（-91.5%）——对
  head-only 线性探针基线（7/10 已不弱，AudioSet 预训练骨干迁移性强）。ESC-50 归档=master.zip
  （`ESC-50-master/meta/esc50.csv`+`audio/*.wav`，csv 名不是 meta.csv）；类别 id 按类别名字典序固定映射
  并写入 split_meta.json。
- **tts**（speecht5 145M + ljspeech 池 3930/训 2000）：mel MSE 0.649→0.408（-37%），10/10 全改善且
  LoRA 侧方差收窄（0.53~0.91 → 0.39~0.43）。LoRA 目标除 q/k/v/out_proj 外经 `LORA_EXTRA_TARGETS`
  加 `intermediate_dense,output_dense`（SpeechT5 的 FFN 命名）。用 `normalized_text`（数字/符号已规范化）。

## 超参模式（sweep 60 步 × 5 组合，64 条 held-out val loss 判据）

- asr：lr=2e-4 显著优于 5e-5/1e-4（val CE 0.112 vs 0.253/0.186），r=16 略优于 r=32（0.1117 vs 0.1141）；
- audio_cls/tts：lr=2e-4/r=32（与"新头/域差大→激进"模式一致；tts 虽无新头但 mel 重建任务量大）。
- **被 bug 污染的 sweep 一律作废重跑**：AST 两轮 sweep（恒 ln50=3.9062 / val 高于机会水平 4.05+）
  全部作废，修复后第三轮（val 1.42-3.77 正常学习）才用于择优——**不要用可疑 sweep 的排序将就**。

## 数据切分注意（接 30%/held-out 协议）

- 音频提取落盘 + 绝对路径（#41）；ljspeech 保留原采样率落 wav（重采样放训练器，缓存才与采样率解耦）；
- whisper 池 611 < 200×4=800 消耗 → 有放回重采样属预期（池即 30% 上限），报告记录池大小；
- 其余同协议：seed 42/43/7/123、写盘前洗牌（#34）、split_meta.json、不相交 assert、
  probe 前 300 标签覆盖检查（audio_cls 50 类全覆盖才进 sweep）。
