# 批次执行记录

## optimization-team-v14（2026-03-23 本轮）

### 当前状态
- optimization completed: 143
- optimization pending: 242（benchmark 已完成，等待 optimization）
- optimization in_progress: 4（google-bert/bert-base-uncased, UBIAI/en_scibert_ScienceIE, bharathsj/bio-medical-llama3-lsfv1, google/flan-t5-large）

### 本轮完成（143 个）

| # | 模型 | Optimizer | Speedup | 备注 |
|---|------|-----------|---------|------|
| 1 | facebook/contriever | opt-3 | 1.29x | npu_add_layer_norm(24) |
| 2 | distilbert/distilgpt2 | opt-2 | 24.94x | GPT-2 pre-norm |
| 3 | google-bert/bert-base-multilingual-cased | opt-3 | 1.21x | npu_add_layer_norm(24) |
| 4 | prajjwal1/bert-tiny | opt-2 | 4.82x | npu_add_layer_norm(4) |
| 5 | Qwen/Qwen3-30B-A3B-Instruct-2507 | opt-1 | 1.08x | npu_rms_norm + npu_rotary_mul + npu_fusion_attention |
| 6 | Nanbeige/Nanbeige4.1-3B | opt-4 | 25.60x | bf16，仅 TQE+warmup |
| 7 | sentence-transformers/paraphrase-MiniLM-L6-v2 | opt-3 | 2.23x | TQE+warmup |
| 8 | facebook/esmfold_v1 | opt-2 | 21.96x | Pre-norm，仅 warmup+TQE |
| 9 | patrickjohncyh/fashion-clip | opt-1 | 1.03x | CLIP Pre-LN + GELU |
| 10 | jonatasgrosman/wav2vec2-large-xlsr-53-greek | opt-1 | 1.03x | npu_add_layer_norm(48) |
| 11 | nomic-ai/nomic-embed-text-v2-moe | opt-4 | 1.07x | dispatch_model 阻止 patch |
| 12 | ProsusAI/finbert | opt-3 | 24.91x | npu_add_layer_norm(24) |
| 13 | usyd-community/vitpose-plus-base | opt-3 | 1.08x | Pre-LN + GELU + MoE |
| 14 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | opt-2 | 1.08x | npu_rms_norm + npu_swiglu + npu_fusion_attention |
| 15 | answerdotai/answerai-colbert-small-v1 | opt-4 | 1.02x | TQE+warmup |
| 16 | facebook/wav2vec2-base-960h | opt-1 | 1.12x | TQE+warmup |
| 17 | microsoft/wavlm-base-plus-sd | opt-2 | 1.07x | add-then-LN 模式 |
| 18 | briaai/RMBG-1.4 | opt-1 | 1.09x | 纯 CNN U-Net |
| 19 | NovaSearch/stella_en_400M_v5 | opt-4 | 1.46x | npu_rms_norm(49) + npu_rotary_mul |
| 20 | mixedbread-ai/mxbai-rerank-xsmall-v1 | opt-3 | 46.59x (cold) / 1.10x (steady) | npu_add_layer_norm(24) |

## optimization-team-v10（2026-03-22）

见 MEMORY.md 历史记录。
