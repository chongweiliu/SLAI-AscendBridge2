---
name: dataset-mapping
description: 根据模型类型自动选择评测数据集。检测模型类别（CausalLM、BERT、VLM、Diffusion 等），返回推荐数据集名称。
---

# Dataset Mapping Skill

本 skill 为 benchmark-runner 提供模型类型到数据集的自动映射能力，避免硬编码数据集选择。

**触发词**：`select dataset`、`choose dataset`、`dataset mapping`、`benchmark dataset`

---

## 1. 数据集映射表

| 模型类别 | 推荐 Dataset Key | 说明 |
|---------|-----------------|------|
| `causal_lm` | `wikitext` | 语言建模基准 |
| `seq2seq` | `cnn_dailymail` | 摘要/生成任务 |
| `bert` | `wikitext` | BERT 系列模型（embedding 提取） |
| `token_classification` | `conll2003` | NER |
| `embedding` | `wikitext` | 语义相似度 |
| `reranker` | `ms_marco` | 检索排序 |
| `vision_classification` | `cifar100` | 图像分类（比 imagenet 小） |
| `vision_detection` | `coco` | 目标检测 |
| `vlm` | `scienceqa` | 视觉问答 |
| `asr` | `librispeech` | 语音识别 |
| `audio_embedding` | `librispeech` | 音频嵌入 |
| `biomedical_nlp` | `pubmed_qa` | 生物医学 NLP |
| `diffusion` | `None` | 不需要数据集，仅测 latency |
| `tts` | `None` | 不需要数据集 |
| `video` | `None` | 不需要数据集 |
| `specialized` | `wikitext` | 默认回退 |

---

## 2. 模型类型检测逻辑

### 2.1 检测函数

```python
def detect_model_type(model_id: str, auto_model_class: str = "") -> str:
    """
    检测模型类型，返回 dataset category key。

    Args:
        model_id: HuggingFace 模型 ID，如 "Qwen/Qwen2-0.5B"
        auto_model_class: AutoModel 类名，如 "AutoModelForCausalLM"

    Returns:
        模型类别字符串，如 "causal_lm"、"asr"、"diffusion"
    """
    model_id_lower = model_id.lower()

    # === 1. 基于 AutoModel class 优先级最高 ===
    if "CausalLM" in auto_model_class:
        return "causal_lm"
    if "Seq2SeqLM" in auto_model_class or "Seq2Seq" in auto_model_class:
        return "seq2seq"
    if "TokenClassification" in auto_model_class:
        return "token_classification"
    if "SequenceClassification" in auto_model_class:
        return "bert"  # 文本分类模型也使用 BERT 类型的评测方式
    if "ImageClassification" in auto_model_class:
        return "vision_classification"
    if "ObjectDetection" in auto_model_class or "VisionEncoderDecoder" in auto_model_class:
        return "vision_detection"
    if "Vision2Seq" in auto_model_class or "VisionModel" in auto_model_class:
        # 可能是 VLM 或纯视觉模型，进一步用关键词判断
        pass

    # === 2. 基于 model_id 关键词 ===

    # Diffusion 模型（Stable Diffusion, SDXL, FLUX 等）
    if any(kw in model_id_lower for kw in [
        "stable-diffusion", "sdxl", "sd-", "flux", "diffusion",
        "controlnet", "dreamshaper", "realistic", "sdxl-turbo"
    ]):
        return "diffusion"

    # 视频生成模型
    if any(kw in model_id_lower for kw in ["wan", "video", "svd", "animatediff"]):
        return "video"

    # TTS 模型
    if any(kw in model_id_lower for kw in ["tts", "xtts", "vocos", "bark", "speecht5"]):
        return "tts"

    # ASR 模型（语音识别）
    if any(kw in model_id_lower for kw in ["whisper", "wav2vec", "asr", "speech-recognition"]):
        return "asr"

    # 音频嵌入模型
    if any(kw in model_id_lower for kw in ["clap", "wavlm", "hubert", "audio-embed"]):
        return "audio_embedding"

    # VLM（视觉语言模型）
    if any(kw in model_id_lower for kw in [
        "llava", "qwen-vl", "qwen2-vl", "internvl", "cogvlm",
        "paligemma", "idefics", "blip", "kosmos", "vlm"
    ]):
        return "vlm"

    # 图像分类（ViT, ResNet, MobileNet 等）
    if any(kw in model_id_lower for kw in [
        "vit", "resnet", "mobilenet", "dino", "convnext",
        "efficientnet", "swin", "deit", "beit"
    ]):
        if any(kw in model_id_lower for kw in ["dino", "detect", "yolo"]):
            return "vision_detection"
        return "vision_classification"

    # 目标检测
    if any(kw in model_id_lower for kw in [
        "grounding", "detr", "yolo", "faster-rcnn", "mask-rcnn",
        "owlvit", "conditional-detr"
    ]):
        return "vision_detection"

    # Embedding 模型
    if any(kw in model_id_lower for kw in [
        "embed", "bge-", "e5-", "gte-", "nomic-embed", "sentence-"
    ]):
        return "embedding"

    # Reranker 模型
    if any(kw in model_id_lower for kw in ["rerank", "re-rank"]):
        return "reranker"

    # 生物医学 NLP
    if any(kw in model_id_lower for kw in [
        "bio", "med", "pubmed", "clinical", "biomed", "chem"
    ]):
        if any(kw in model_id_lower for kw in ["clip", "sam", "segment"]):
            return "vision_classification"  # 生物医学视觉
        return "biomedical_nlp"

    # BERT 系列判断（可能是分类或 NER）
    if any(kw in model_id_lower for kw in ["bert", "roberta", "deberta", "distilbert", "albert"]):
        if any(kw in model_id_lower for kw in ["ner", "token", "pos-tag"]):
            return "token_classification"
        return "bert"  # BERT 系列模型使用 bert 类型的评测方式

    # Seq2Seq 模型（T5, BART, Pegasus）
    if any(kw in model_id_lower for kw in ["t5", "bart", "pegasus", "flan", "led"]):
        return "seq2seq"

    # === 3. 默认：CausalLM ===
    return "causal_lm"
```

### 2.2 数据集选择函数

```python
# 数据集映射字典
DATASET_MAPPING = {
    "causal_lm": "wikitext",
    "seq2seq": "cnn_dailymail",
    "bert": "wikitext",
    "token_classification": "conll2003",
    "embedding": "wikitext",
    "reranker": "ms_marco",
    "vision_classification": "cifar100",
    "vision_detection": "coco",
    "vlm": "scienceqa",
    "asr": "librispeech",
    "audio_embedding": "librispeech",
    "biomedical_nlp": "pubmed_qa",
    "diffusion": None,
    "tts": None,
    "video": None,
    "specialized": "wikitext",
}


def get_dataset_for_model(model_id: str, auto_model_class: str = "") -> str | None:
    """
    获取模型对应的推荐数据集。

    Args:
        model_id: HuggingFace 模型 ID
        auto_model_class: AutoModel 类名（可选）

    Returns:
        数据集 key（如 "wikitext"），或 None 表示不需要数据集
    """
    model_type = detect_model_type(model_id, auto_model_class)
    return DATASET_MAPPING.get(model_type)
```

---

## 3. 使用示例

### 3.1 在 benchmark-runner 中使用

```python
# benchmark-runner 生成 accuracy_run.py 时
from dataset_mapping import get_dataset_for_model

model_id = "Qwen/Qwen2-0.5B"
dataset_key = get_dataset_for_model(model_id, "AutoModelForCausalLM")
# dataset_key = "wikitext"

# 对于 Diffusion 模型
dataset_key = get_dataset_for_model("runwayml/stable-diffusion-v1-5", "StableDiffusionPipeline")
# dataset_key = None  # 不需要数据集
```

### 3.2 测试用例

```python
def test_detect_model_type():
    # CausalLM
    assert detect_model_type("Qwen/Qwen2-0.5B", "AutoModelForCausalLM") == "causal_lm"
    assert detect_model_type("meta-llama/Llama-2-7b", "AutoModelForCausalLM") == "causal_lm"

    # ASR
    assert detect_model_type("openai/whisper-base", "WhisperForConditionalGeneration") == "asr"

    # Vision Classification
    assert detect_model_type("google/vit-base-patch16-224", "AutoModelForImageClassification") == "vision_classification"

    # Diffusion
    assert detect_model_type("runwayml/stable-diffusion-v1-5", "") == "diffusion"
    assert detect_model_type("black-forest-labs/FLUX.1-schnell", "") == "diffusion"

    # VLM
    assert detect_model_type("Qwen/Qwen2-VL-7B-Instruct", "") == "vlm"

    # Embedding
    assert detect_model_type("BAAI/bge-base-en-v1.5", "") == "embedding"

    # TTS
    assert detect_model_type("coqui/XTTS-v2", "") == "tts"

    # Token Classification
    assert detect_model_type("dslim/bert-base-NER", "") == "token_classification"

    # Biomedical
    assert detect_model_type("microsoft/BiomedNLP-PubMedBERT", "") == "biomedical_nlp"
```

---

## 4. 与 accuracy_run.py 的集成

### 4.1 数据集加载方式（重要）

**优先使用 `load_from_disk`** 加载项目级 `datasets/` 目录，避免在线加载与 trust_remote_code 问题：

```python
from datasets import load_from_disk
from pathlib import Path

# 项目根目录的 datasets 文件夹
DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"

# 根据 dataset_key 加载（需先用 download_datasets.py 下载）
# 目录名格式：wikitext___wikitext-2-raw-v1, cnn_dailymail___3.0.0, cifar100 等
ds = load_from_disk(str(DATASET_DIR / "wikitext___wikitext-2-raw-v1"))
texts = sorted([s["text"] for s in ds if s.get("text", "").strip()])
```

**禁止**：

- 对 `load_dataset` 使用 `trust_remote_code=True`（HF datasets 2.16+ 自定义脚本数据集已弃用，会报错 "Dataset scripts are no longer supported"）
- 依赖 `load_dataset` 在线加载自定义脚本数据集（如 indonlu、pubmed_qa 等）

**若必须在线加载**：仅使用标准 Parquet 数据集（wikitext、sst2、cifar100、librispeech），**不传** `trust_remote_code`。

### 4.2 推理输入选择

```python
def get_benchmark_input(model_type: str, dataset_key: str | None, tokenizer=None):
    """根据模型类型获取基准测试输入"""
    if model_type == "causal_lm":
        return tokenizer("Hello, benchmark run.", return_tensors="pt")
    elif model_type == "vision_classification":
        # 返回 CIFAR100 的一个样本
        ...
    elif model_type == "diffusion":
        # 返回随机噪声 + prompt
        ...
    # ... 其他类型
```

---

## 5. 数据集下载

使用 `scripts/download_datasets.py` 下载所需数据集：

```bash
# 下载单个数据集
uv run python scripts/download_datasets.py wikitext

# 下载多个数据集
uv run python scripts/download_datasets.py wikitext cifar100 librispeech

# 使用国内镜像
uv run python scripts/download_datasets.py --mirror wikitext
```

---

## 6. 注意事项

1. **None 数据集**：`diffusion`、`tts`、`video` 类型返回 `None`，表示仅需测 latency，不需要真实数据集
2. **优先级**：`auto_model_class` 优先级高于 `model_id` 关键词匹配
3. **回退**：未知模型默认为 `causal_lm`，使用 `wikitext`
4. **扩展**：新增模型类型时，更新 `DATASET_MAPPING` 和 `detect_model_type` 函数
