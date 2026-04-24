# pyannote/wespeaker-voxceleb-resnet34-LM Optimization Notes

**模型**: ResNet34 Speaker Embedding (CNN, BasicBlock + BatchNorm2d + Conv2d)
**日期**: 2026-03-29
**结果**: failed (num_samples < 50)

## 模型架构
- ResNet34 CNN: BasicBlock + BatchNorm2d + Conv2d
- 无 transformer 层
- 所有融合算子均不适用: npu_add_layer_norm, npu_rms_norm, npu_swiglu, npu_rotary_mul, npu_fusion_attention

## 优化项
- TASK_QUEUE_ENABLE=1 (异步算子下发)
- warmup(3x) 对称测量

## 测量结果
- speedup_ratio: 1.5172x (TQE only, symmetric warmup)
- cosine_similarity: 1.0
- latency_reduction_pct: 34.09%

## 失败原因
**BLOCKER**: builtin dataset 只有 8 个样本，completion 要求 num_samples >= 50

### 数据集问题详情
1. builtin dataset 仅有 8 个随机噪声音频样本
2. librispeech_asr___clean 存在但需要 torchcodec 解码 (NVIDIA only, 不可用)
3. openslr___librispeech_asr___clean 数据格式与 accuracy_run.py 不兼容

### measurement_contract_version: 3
- warmup_policy: symmetric
- comparison_method: independent_baseline_artifact
- comparison_scope: steady_state

## 关键教训
1. **CNN 模型无融合算子优化**: ResNet 等 CNN 架构没有 RMSNorm/SwiGLU/RoPE/Attention，所有 6 大融合算子均不适用
2. **数据集充分性检查**: 任务开始前必须确认数据集样本数 >= 50
3. **torchcodec 依赖**: 音频模型依赖 torchcodec 解码，该库为 NVIDIA only，NPU 环境不可用
4. **step1/step2 架构问题**: accuracy_run.py 的 step1 含 profiling，step2 不含，导致 wall_clock 与 latency_s 口径不一致

## 代码修改
- accuracy_run.py: 添加 --warmup 参数和 run_warmup() 函数 (支持对称 warmup 测量)
- model_files/modeling_wespeaker.py: 与 baseline 相同 (无融合算子补丁)