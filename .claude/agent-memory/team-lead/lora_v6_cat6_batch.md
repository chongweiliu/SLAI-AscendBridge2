---
name: lora-v6-cat6-batch
description: 台账第6大类3模型(语音音频)LoRA实战（asr/audio_cls/tts三新范式/transformers 5.x语音栈5 bug/过拟合测试纪律）
metadata:
  type: project
---

# 台账第 6 大类 3 模型（语音与音频）LoRA 批量实战（2026-09-18）

whisper-large-v3-turbo/librispeech(asr)、ast-finetuned-audioset/esc50(audio_cls)、speecht5_tts/ljspeech(tts)。总报告 `lora-ws/.cat6/LoRA-v6-验证报告.md`。

## 结果速记（3/3 改善）
- **whisper**：CER 0.0091→**0.0045**(-51%)——LibriSpeech clean 是 base"主场"仍减半错误率，7改/3平(已0错)/1微退
- **AST**：top1 7/10→**10/10**、CE 1.74→**0.148**(-91.5%)（vs head-only 探针基线；AudioSet 骨干迁移 ESC-50 域差被 LoRA 弥合）
- **speecht5**：teacher-forced mel MSE 0.649→**0.408**(-37%)，10/10 全改善；**口径**：无 vocoder 归档→mel 域非波形、零 speaker 向量，报告如实限定表述
- 超参：whisper 2e-4/r16；AST/tts 2e-4/r32

## transformers 5.x 语音栈五个坑（全修复，同类模型必再遇）
1. **WhisperFeatureExtractor `max_length` 变采样点语义**：`padding="max_length",max_length=3000` 不再 pad mel 到 3000 帧而是截音频到 3000 点(→18帧)→报 "expects 3000 found 18"。修：逐条提 mel 后**手动 pad/截 3000 帧**
2. **SpeechT5 reduction_factor=2 要偶数帧**：奇数帧 labels 报 size 603vs602。修：谱截偶
3. **SpeechT5 `use_guided_attention_loss` 默认开且 mask=None 崩**（'bool' has no 'sum'）。修：config 关 + 显式 attention_mask
4. **SpeechT5SpectrogramLoss BCE pos_weight 留 CPU**（criterion 在 forward 内现建）。修：monkey-patch 搬 pos_weight 到 logits.device
5. **ASTMLPHead=LayerNorm→Dropout→dense**：按 dim==1 归零的重初始化把 layernorm.weight 清零→特征湮灭→CE 恒 ln(50)=3.906 且 dense 梯度恒 0；叠加 **LoRA 候选 'dense' 撞 classifier.dense**（头被 LoRA 包裹→head.pt 键名 base_layer.*→validate strict=False 静默不加载→随机头假退化）。修：只重初始化 dense.*、候选排除 dense、头保存键名规范化+过滤 lora_ 键。**教训：strict=False 加载后必须校验命中数>0**

## 纪律沉淀：新头范式先跑「单批过拟合测试」
1 批×20-30 步@lr1e-3，loss 必须显著下降才进 sweep。AST 若先做此测试可省两轮作废 sweep（恒 3.906 的 loss 在过拟合测试下 20 步即暴露）。凡 loss 钉死在 ln(类别数) → 头没在训练（冻结/梯度为0/特征湮灭三选一排查）。

## 范式实现要点
- asr：WhisperForConditionalGeneration LoRA 挂 encoder+decoder(q/k/v/out_proj+fc1/fc2)；30s mel pad；验证=greedy generate+`get_decoder_prompt_ids(en/transcribe)`；指标 CER(主)+ROUGE-L/字符重叠（GT 大写无标点→双侧 norm：小写去标点压空格）
- audio_cls：ASTForAudioClassification(num_labels=50,ignore_mismatched_sizes)；ASTFeatureExtractor→(1024,128)谱；head-only 基线协议同 [[lora-v5-cat5-batch]]
- tts：labels=完整 mel（5.x 内部自动 shift，**不要**手传 decoder_input_values）；mel 预编码缓存(md5(DATA)键,sweep复用)；**batch1+grad_accum8**（谱长可变,padding 污染 loss）；ljspeech 22.05k→16k resample_poly
- esc50 归档=master.zip(ESC-50-master/meta/esc50.csv+audio/*.wav)，csv 名不是 meta.csv

## 复用资产
`lora-ws/.cat6/`：make_split6.py(3切分)、scripts/{lora_train6(三范式+mel缓存+grad_accum),val_loss6,validate6,sweep6}、run_ast.sh。候选 skill 沉淀：三范式+5 bug+过拟合测试纪律。

## 关联
[[lora-v5-cat5-batch]]（head-only 协议/peft 冻头 bug 同源）；[[lora-v4-cat4-batch]]；环境抢修 [[venv-interpreter-loss-recovery]]。
