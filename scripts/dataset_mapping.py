#!/usr/bin/env python3
"""
模型类型到数据集的自动映射。

用法:
    uv run python scripts/dataset_mapping.py --model_id "Qwen/Qwen2-0.5B"
    uv run python scripts/dataset_mapping.py --model_id "openai/whisper-base" --model_class "WhisperForConditionalGeneration"
"""

import argparse
import json
import re
from typing import Optional

MULTILINGUAL_ASR_DATASET_RULES = (
    (("wav2vec2-large-xlsr-53-arabic", "-arabic"), "fleurs_ar_eg", "ar"),
    (("wav2vec2-large-xlsr-53-chinese-zh-cn", "zh-cn"), "fleurs_cmn_hans_cn", "zh"),
    (("wav2vec2-large-xlsr-53-dutch", "-dutch"), "fleurs_nl_nl", "nl"),
    (("wav2vec2-large-xlsr-53-greek", "-greek"), "fleurs_el_gr", "el"),
    (("wav2vec2-large-xlsr-53-hungarian", "-hungarian"), "fleurs_hu_hu", "hu"),
    (("wav2vec2-large-xlsr-53-japanese", "-japanese"), "fleurs_ja_jp", "ja"),
    (("reazonspeech-nemo-v2", "reazonspeech-nemo"), "fleurs_ja_jp", "ja"),
    (("wav2vec2-large-xlsr-53-polish", "-polish"), "mcspeech_pl", "pl"),
    (("wav2vec2-large-xlsr-53-portuguese", "-portuguese"), "fleurs_pt_br", "pt"),
    (("wav2vec2-large-xlsr-53-russian", "-russian"), "fleurs_ru_ru", "ru"),
    (("romanian-wav2vec2", "-romanian"), "fleurs_ro_ro", "ro"),
    (("wav2vec2-base-vi-vlsp2020", "-vietnamese", "vlsp2020"), "fleurs_vi_vn", "vi"),
    (("wav2vec2-large-xls-r-300m-urdu", "-urdu"), "fleurs_ur_pk", "ur"),
)

MULTILINGUAL_ASR_MODEL_SIGNALS = (
    "asr",
    "speech",
    "wav2vec",
    "wavlm",
    "whisper",
    "hubert",
    "seamless",
    "conformer",
    "ctc",
    "nemo",
    "canary",
)


def _has_multilingual_asr_model_signal(normalized_model_id: str) -> bool:
    return any(signal in normalized_model_id for signal in MULTILINGUAL_ASR_MODEL_SIGNALS)


def resolve_multilingual_asr_dataset(model_id: str) -> Optional[dict]:
    normalized_model_id = _normalize_text(model_id)
    has_asr_signal = _has_multilingual_asr_model_signal(normalized_model_id)
    for signals, dataset_key, asr_language in MULTILINGUAL_ASR_DATASET_RULES:
        matched_signals = [signal for signal in signals if signal in normalized_model_id]
        if not matched_signals:
            continue
        # Language-only suffixes like "-portuguese" are too broad and can
        # accidentally match text models; require an explicit speech/ASR cue
        # unless the rule already matched a model-family-specific signal.
        has_specific_family_signal = any(
            any(ch.isdigit() for ch in signal) or signal.startswith("reazonspeech") or signal.startswith("romanian-wav2vec2")
            for signal in matched_signals
        )
        if has_specific_family_signal or has_asr_signal:
            return {
                "dataset_key": dataset_key,
                "asr_language": asr_language,
                "asr_task": "transcribe",
            }
    return None


# 数据集映射字典
DATASET_MAPPING = {
    "causal_lm": "wikitext",
    "seq2seq": "cnn_dailymail",
    "classification": "sst2",
    "question_answering": "squad_v2",
    "masked_lm": "wikitext",
    "token_classification": "conll2003",
    "biomedical_token_classification": "ncbi_disease",
    "discriminator": "wikitext",
    "embedding": "wikitext",
    "reranker": "ms_marco",
    "vision_classification": "cifar100",
    "vision_embedding": "cifar100",
    "vision_detection": "coco",
    "vision_text_ocr": "synthetic_ocr",
    "vision_keypoint_detection": "synthetic_keypoints",
    "image_matting": "synthetic_matting",
    "vlm": "scienceqa",
    "asr": "librispeech",
    "audio_embedding": "librispeech",
    "biomedical_nlp": "pubmed_qa",
    "diffusion": None,
    "tts": None,
    "video": None,
    "specialized": "wikitext",
}

DATASET_CANDIDATES = {
    "causal_lm": ["wikitext", "gsm8k", "mmlu", "ceval"],
    "seq2seq": ["cnn_dailymail", "xsum", "samsum"],
    "classification": ["sst2", "imdb", "ag_news", "tweet_eval_sentiment"],
    "question_answering": ["squad_v2", "pubmed_qa", "glue_qnli"],
    "masked_lm": ["wikitext", "pubmed_qa"],
    "token_classification": ["conll2003", "science_ie"],
    "biomedical_token_classification": ["ncbi_disease", "conll2003"],
    "discriminator": ["wikitext"],
    "embedding": ["wikitext", "pubmed_qa", "ms_marco"],
    "reranker": ["ms_marco", "wikitext"],
    "vision_classification": ["cifar100", "imagenet", "fairface"],
    "vision_embedding": ["cifar100", "imagenet"],
    "vision_detection": ["coco", "scienceqa"],
    "vision_text_ocr": ["synthetic_ocr"],
    "vision_keypoint_detection": ["synthetic_keypoints"],
    "image_matting": ["synthetic_matting"],
    "vlm": ["scienceqa", "coco"],
    "asr": ["librispeech"],
    "audio_embedding": ["librispeech"],
    "biomedical_nlp": ["pubmed_qa", "chemprot", "wikitext"],
    "diffusion": [],
    "tts": [],
    "video": [],
    "specialized": ["wikitext", "gsm8k"],
}

BUSINESS_EVAL_PROFILES = {
    "causal_lm": {
        "evaluation_profile": "generation_exact_match",
        "primary_metric": "exact_match",
        "secondary_metrics": ["latency_s", "throughput_qps", "match_rate"],
        "output_type_hint": "generated_text",
    },
    "mmlu": {
        "evaluation_profile": "mmlu",
        "primary_metric": "accuracy",
        "secondary_metrics": ["latency_s", "throughput_qps", "match_rate"],
        "output_type_hint": "class_labels",
    },
    "seq2seq": {
        "evaluation_profile": "summarization_rouge",
        "primary_metric": "rougeL",
        "secondary_metrics": ["latency_s", "throughput_qps", "match_rate"],
        "output_type_hint": "generated_text",
    },
    "classification": {
        "evaluation_profile": "classification_accuracy",
        "primary_metric": "accuracy",
        "secondary_metrics": ["latency_s", "top1_accuracy", "match_rate"],
        "output_type_hint": "class_labels",
    },
    "question_answering": {
        "evaluation_profile": "qa_exact_match",
        "primary_metric": "exact_match",
        "secondary_metrics": ["latency_s", "f1", "match_rate"],
        "output_type_hint": "qa_answers",
    },
    "masked_lm": {
        "evaluation_profile": "classification_accuracy",
        "primary_metric": "accuracy",
        "secondary_metrics": ["latency_s", "match_rate"],
        "output_type_hint": "predicted_tokens",
    },
    "token_classification": {
        "evaluation_profile": "token_classification_f1",
        "primary_metric": "f1",
        "secondary_metrics": ["latency_s", "precision", "recall", "match_rate"],
        "output_type_hint": "predicted_tokens",
    },
    "biomedical_token_classification": {
        "evaluation_profile": "token_classification_f1",
        "primary_metric": "f1",
        "secondary_metrics": ["latency_s", "precision", "recall", "match_rate"],
        "output_type_hint": "predicted_tokens",
    },
    "discriminator": {
        "evaluation_profile": "embedding_similarity",
        "primary_metric": "cosine_similarity",
        "secondary_metrics": ["latency_s", "throughput_qps", "match_rate"],
        "output_type_hint": "discriminator_logits",
    },
    "embedding": {
        "evaluation_profile": "embedding_similarity",
        "primary_metric": "cosine_similarity",
        "secondary_metrics": ["latency_s", "throughput_qps", "match_rate"],
        "output_type_hint": "embeddings",
    },
    "reranker": {
        "evaluation_profile": "reranker_ndcg",
        "primary_metric": "ndcg_at_10",
        "secondary_metrics": ["latency_s", "mrr", "match_rate"],
        "output_type_hint": "relevance_scores",
    },
    "vision_classification": {
        "evaluation_profile": "vision_topk_accuracy",
        "primary_metric": "top1_accuracy",
        "secondary_metrics": ["latency_s", "top5_accuracy", "match_rate"],
        "output_type_hint": "class_labels",
    },
    "vision_embedding": {
        "evaluation_profile": "embedding_similarity",
        "primary_metric": "cosine_similarity",
        "secondary_metrics": ["latency_s", "throughput_qps", "match_rate"],
        "output_type_hint": "image_embeddings",
    },
    "vision_detection": {
        "evaluation_profile": "detection_map",
        "primary_metric": "mAP",
        "secondary_metrics": ["latency_s", "map50", "match_rate"],
        "output_type_hint": "detection_boxes",
    },
    "vision_text_ocr": {
        "evaluation_profile": "generation_exact_match",
        "primary_metric": "exact_match",
        "secondary_metrics": ["latency_s", "match_rate", "text_match_rate"],
        "output_type_hint": "generated_text",
    },
    "vision_keypoint_detection": {
        "evaluation_profile": "keypoint_repeatability",
        "primary_metric": "keypoint_repeatability",
        "secondary_metrics": ["latency_s", "throughput_qps", "num_keypoints"],
        "output_type_hint": "keypoints",
    },
    "image_matting": {
        "evaluation_profile": "matting_mae",
        "primary_metric": "mae",
        "secondary_metrics": ["latency_s", "throughput_qps", "cosine_similarity"],
        "output_type_hint": "alpha_masks",
    },
    "vision_language_action": {
        "evaluation_profile": "latency_only",
        "primary_metric": "latency_s",
        "secondary_metrics": ["throughput_qps"],
        "output_type_hint": "generated_action",
    },
    "vlm": {
        "evaluation_profile": "vlm_accuracy",
        "primary_metric": "accuracy",
        "secondary_metrics": ["latency_s", "match_rate", "text_match_rate"],
        "output_type_hint": "generated_text",
    },
    "asr": {
        "evaluation_profile": "asr_wer",
        "primary_metric": "wer",
        "secondary_metrics": ["latency_s", "throughput_qps", "text_match_rate"],
        "output_type_hint": "transcriptions",
    },
    "audio_embedding": {
        "evaluation_profile": "audio_embedding_similarity",
        "primary_metric": "cosine_similarity",
        "secondary_metrics": ["latency_s", "throughput_qps", "match_rate"],
        "output_type_hint": "audio_embeddings",
    },
    "biomedical_nlp": {
        # Bare biomedical encoders and SetFit-style checkpoints in this bucket
        # expose CLS embeddings rather than extractive QA heads.
        "evaluation_profile": "embedding_similarity",
        "primary_metric": "cosine_similarity",
        "secondary_metrics": ["latency_s", "throughput_qps", "match_rate"],
        "output_type_hint": "cls_embeddings",
    },
    "diffusion": {
        "evaluation_profile": "latency_only",
        "primary_metric": "latency_s",
        "secondary_metrics": ["throughput_qps"],
        "output_type_hint": "diffusion_latency",
    },
    "tts": {
        "evaluation_profile": "latency_only",
        "primary_metric": "latency_s",
        "secondary_metrics": ["throughput_qps"],
        "output_type_hint": "audio_waveform",
    },
    "video": {
        "evaluation_profile": "latency_only",
        "primary_metric": "latency_s",
        "secondary_metrics": ["throughput_qps"],
        "output_type_hint": "video_latency",
    },
    "specialized": {
        "evaluation_profile": "generation_exact_match",
        "primary_metric": "exact_match",
        "secondary_metrics": ["latency_s", "throughput_qps", "match_rate"],
        "output_type_hint": "generated_text",
    },
}

BUSINESS_INTENT_SPECS = {
    "causal_lm": {"model_type": "causal_lm", "dataset_key": "mmlu"},
    "causal_lm_base": {"model_type": "causal_lm", "dataset_key": "mmlu"},
    "causal_lm_instruct": {"model_type": "causal_lm", "dataset_key": "gsm8k"},
    "safety_guard_generation": {"model_type": "causal_lm", "dataset_key": "tweet_eval_offensive", "evaluation_profile_key": "classification"},
    "biomedical_causal_lm": {"model_type": "causal_lm", "dataset_key": "pubmed_qa", "evaluation_profile_key": "question_answering"},
    "seq2seq_summarization": {"model_type": "seq2seq", "dataset_key": "cnn_dailymail"},
    "biomedical_seq2seq_qa": {"model_type": "seq2seq", "dataset_key": "pubmed_qa", "evaluation_profile_key": "question_answering"},
    "generic_classification": {"model_type": "classification", "dataset_key": "sst2"},
    "sentiment_binary": {"model_type": "classification", "dataset_key": "imdb"},
    "sentiment_multiclass": {"model_type": "classification", "dataset_key": "tweet_eval_sentiment"},
    "emotion_multiclass": {"model_type": "classification", "dataset_key": "tweet_eval_emotion"},
    "offensive_binary": {"model_type": "classification", "dataset_key": "tweet_eval_offensive"},
    "hate_binary": {"model_type": "classification", "dataset_key": "tweet_eval_hate"},
    "topic_classification": {"model_type": "classification", "dataset_key": "ag_news"},
    "natural_language_inference": {"model_type": "classification", "dataset_key": "glue_mnli"},
    "question_pair_classification": {"model_type": "classification", "dataset_key": "glue_qnli"},
    "extractive_qa": {"model_type": "question_answering", "dataset_key": "squad_v2"},
    "masked_language_modeling": {"model_type": "masked_lm", "dataset_key": "wikitext", "evaluation_profile_key": "masked_lm"},
    "biomedical_masked_language_modeling": {"model_type": "masked_lm", "dataset_key": "pubmed_qa", "evaluation_profile_key": "masked_lm"},
    "token_classification": {"model_type": "token_classification", "dataset_key": "conll2003"},
    "scientific_token_classification": {"model_type": "token_classification", "dataset_key": "science_ie"},
    "biomedical_token_classification": {"model_type": "biomedical_token_classification", "dataset_key": "ncbi_disease"},
    "discriminator": {"model_type": "discriminator", "dataset_key": "wikitext"},
    "embedding": {"model_type": "embedding", "dataset_key": "wikitext"},
    "reranker": {"model_type": "reranker", "dataset_key": "ms_marco"},
    "vision_classification": {"model_type": "vision_classification", "dataset_key": "cifar100"},
    "vision_embedding": {"model_type": "vision_embedding", "dataset_key": "cifar100"},
    "face_age_classification": {"model_type": "vision_classification", "dataset_key": "fairface"},
    "vision_detection": {"model_type": "vision_detection", "dataset_key": "coco"},
    "table_detection": {"model_type": "vision_detection", "dataset_key": "pubtables_detection_1500"},
    "vision_keypoint_detection": {"model_type": "vision_keypoint_detection", "dataset_key": "synthetic_keypoints"},
    "image_matting": {"model_type": "image_matting", "dataset_key": "synthetic_matting"},
    "vision_language_action": {"model_type": "vlm", "dataset_key": None, "evaluation_profile_key": "vision_language_action"},
    "protein_embedding": {"model_type": "embedding", "dataset_key": "synthetic_protein"},
    "audio_embedding": {"model_type": "audio_embedding", "dataset_key": "librispeech"},
    "vlm": {"model_type": "vlm", "dataset_key": "scienceqa"},
    "asr": {"model_type": "asr", "dataset_key": "librispeech"},
    "biomedical_nlp": {"model_type": "embedding", "dataset_key": "pubmed_qa"},
    "diffusion": {"model_type": "diffusion", "dataset_key": None},
    "tts": {"model_type": "tts", "dataset_key": None},
    "video": {"model_type": "video", "dataset_key": None},
    "specialized": {"model_type": "specialized", "dataset_key": "wikitext"},
}

# 模型类型的友好名称
MODEL_TYPE_NAMES = {
    "causal_lm": "Causal Language Model",
    "seq2seq": "Sequence-to-Sequence",
    "classification": "Text Classification",
    "question_answering": "Question Answering",
    "masked_lm": "Masked Language Model",
    "token_classification": "Token Classification (NER)",
    "discriminator": "Token Discriminator",
    "embedding": "Text Embedding",
    "reranker": "Reranker",
    "vision_classification": "Image Classification",
    "vision_embedding": "Image Embedding",
    "vision_detection": "Object Detection",
    "vision_keypoint_detection": "Keypoint Detection",
    "image_matting": "Image Matting",
    "vision_language_action": "Vision-Language-Action",
    "vlm": "Vision-Language Model",
    "asr": "Automatic Speech Recognition",
    "audio_embedding": "Audio Embedding",
    "biomedical_nlp": "Biomedical NLP",
    "diffusion": "Diffusion Model",
    "tts": "Text-to-Speech",
    "video": "Video Generation",
    "specialized": "Specialized Model",
}

BUSINESS_INTENT_NAMES = {
    "causal_lm": "Generative Reasoning",
    "causal_lm_base": "Base Model Reasoning",
    "causal_lm_instruct": "Instruction Following",
    "safety_guard_generation": "Safety Prompt Moderation",
    "biomedical_causal_lm": "Biomedical Generative QA",
    "seq2seq_summarization": "Summarization",
    "biomedical_seq2seq_qa": "Biomedical Seq2Seq QA",
    "generic_classification": "Generic Classification",
    "sentiment_binary": "Binary Sentiment Classification",
    "sentiment_multiclass": "Multi-class Sentiment Classification",
    "emotion_multiclass": "Emotion Classification",
    "offensive_binary": "Offensive Content Classification",
    "hate_binary": "Hate Speech Classification",
    "topic_classification": "Topic Classification",
    "natural_language_inference": "Natural Language Inference",
    "question_pair_classification": "Question Pair Classification",
    "extractive_qa": "Extractive Question Answering",
    "masked_language_modeling": "Masked Token Prediction",
    "biomedical_masked_language_modeling": "Biomedical Masked Token Prediction",
    "token_classification": "Token Classification",
    "scientific_token_classification": "Scientific Token Classification",
    "biomedical_token_classification": "Biomedical Token Classification",
    "discriminator": "Token Replacement Discrimination",
    "embedding": "Embedding Similarity",
    "reranker": "Passage Reranking",
    "vision_classification": "Image Classification",
    "vision_embedding": "Image Embedding Similarity",
    "face_age_classification": "Face Age Classification",
    "vision_detection": "Object Detection",
    "table_detection": "Table Detection",
    "vision_keypoint_detection": "Keypoint Detection",
    "image_matting": "Image Matting",
    "vision_language_action": "Vision-Language-Action Latency",
    "protein_embedding": "Protein Embedding Similarity",
    "audio_embedding": "Audio Embedding",
    "vlm": "Vision-Language Understanding",
    "asr": "Speech Recognition",
    "biomedical_nlp": "Biomedical NLP",
    "diffusion": "Diffusion Latency",
    "tts": "Text-to-Speech Latency",
    "video": "Video Generation Latency",
    "specialized": "Specialized Fallback",
}

RERANKER_KEYWORDS = ("cross-encoder", "rerank", "re-rank", "ms-marco", "colbert", "bge-reranker")
QUESTION_ANSWERING_KEYWORDS = ("question-answer", "question_answer", "qa-", "qna", "squad", "qa_", "-qa", "reader")
TOKEN_CLASSIFICATION_KEYWORDS = ("ner", "token", "pos-tag", "slot", "scienceie")
EMBEDDING_KEYWORDS = ("embed", "embedding", "bge-", "e5-", "gte-", "nomic-embed", "sentence-")
AUDIO_EMBEDDING_KEYWORDS = (
    "clap",
    "wavlm",
    "hubert",
    "audio-embed",
    "muq",
    "wespeaker",
    "voxceleb",
    "speaker embedding",
    "speaker-embedding",
    "speaker verification",
    "speaker-verification",
    "speaker_encoder",
)
VLM_KEYWORDS = (
    "llava",
    "qwen-vl",
    "qwen2-vl",
    "qwen2.5-vl",
    "qwen2_5-vl",
    "qwen2_5_vl",
    "qwen2.5-omni",
    "qwen2_5_omni",
    "qwen2-5-omni",
    "omni",
    "qwen3-vl",
    "internvl",
    "cogvlm",
    "paligemma",
    "idefics",
    "blip",
    "kosmos",
    "vlm",
    "deepseek-ocr",
    "deepseekocr",
    "deepseek_vl_v2",
    "deepseek-vl-v2",
)
VISION_CLASSIFICATION_KEYWORDS = ("vit", "resnet", "mobilenet", "convnext", "efficientnet", "swin", "deit", "beit", "mobilevit")
VISION_EMBEDDING_KEYWORDS = (
    "sam-vit",
    "segment-anything",
    "segmentanything",
    "sammodel",
    "samvisionmodel",
    "imageencodervit",
    "sam image encoder",
    "dinov2",
    "dinov2model",
    "dino-vit",
    "dinovit",
    "face recognition",
    "face-recognition",
    "face_recognition",
    "face embedding",
    "face embeddings",
    "face_embeddings",
    "lfw",
)
VISION_DETECTION_KEYWORDS = ("grounding", "detr", "yolo", "faster-rcnn", "mask-rcnn", "owlvit", "conditional-detr", "detect")
KEYPOINT_DETECTION_KEYWORDS = ("superpoint", "keypoint", "local-feature", "local feature")
MATTING_KEYWORDS = ("matting", "trimap", "vitmatte", "alpha mask", "alpha_masks", "alpha matte", "background removal", "background-removal", "birefnet")
FAIRFACE_AGE_KEYWORDS = ("fairface", "age_detection", "age-detection", "age detection")
ASR_KEYWORDS = ("whisper", "wav2vec", "asr", "speech-recognition")
FORCED_ALIGNER_KEYWORDS = ("forcedaligner", "forced-aligner", "forced_aligner", "forced aligner")
TTS_KEYWORDS = ("tts", "xtts", "vocos", "bark", "speecht5", "musicgen", "text-to-audio", "text to audio", "text-to-music", "text to music")
VIDEO_KEYWORDS = ("wan", "video", "svd", "animatediff")
DIFFUSION_KEYWORDS = ("stable-diffusion", "sdxl", "flux", "diffusion", "controlnet", "dreamshaper", "realistic", "sdxl-turbo")
BIOMEDICAL_KEYWORDS = ("bio", "pubmed", "clinical", "biomed", "chem", "medical", "medic")
BERT_FAMILY_KEYWORDS = ("bert", "roberta", "deberta", "distilbert", "albert", "longformer")
SEQ2SEQ_KEYWORDS = ("t5", "bart", "pegasus", "flan", "led")
SENTIMENT_KEYWORDS = ("sentiment", "polarity", "sst2", "sst-2")
EMOTION_KEYWORDS = ("emotion", "emotions", "affect")
OFFENSIVE_KEYWORDS = ("offensive", "offense", "toxic", "toxicity", "abuse")
HATE_KEYWORDS = ("hate", "hate-speech", "hatespeech")
TOPIC_KEYWORDS = ("topic", "topics", "theme", "themes", "ag_news", "ag-news", "news-classification", "news_topic", "industry_theme", "industry-theme")
NLI_KEYWORDS = ("mnli", "xnli", "nli", "entailment", "contradiction")
QUESTION_PAIR_KEYWORDS = ("qnli", "question-pair", "question_pair", "pair-classification")
GENERIC_CLASSIFICATION_KEYWORDS = ("classifier", "classification")
# Common instruct-tuned families do not always carry an explicit "instruct"
# suffix in the repo name (for example Hermes / OpenHermes / Dolphin).
CAUSAL_LM_INSTRUCT_KEYWORDS = ("instruct", "chat", "assistant", "alpaca", "orca", "rlhf", "dpo", "sft", "hermes", "dolphin", "tulu", "chatglm")
GUARD_GENERATION_KEYWORDS = ("guard-gen", "guard gen", "qwen3guard", "qwen3 guard")
PROTEIN_MODEL_KEYWORDS = ("esm", "ur50", "protein", "amino", "protbert", "prott5", "prot_t5", "uniref")
AUDIO_EMBEDDING_BACKBONE_MARKERS = (
    "wav2vec2forpretraining",
    "wav2vec2model",
    "hubertmodel",
    "wavlmmodel",
    "unispeechmodel",
    "unispeechsatmodel",
)
AUDIO_ASR_HEAD_MARKERS = (
    "ctc",
    "speechseq2seq",
    "conditionalgeneration",
    "whisperfor",
    "speechencoderdecoder",
)
MULTIMODAL_EMBEDDING_KEYWORDS = (
    "vision-language embedding",
    "visual-language embedding",
    "visual-text embedding",
    "vision-text embedding",
    "multimodal embedding",
    "multi-modal embedding",
    "vlm embedding",
    "joint embedding",
)
MULTIMODAL_EMBEDDING_EXCLUDE_KEYWORDS = (
    "clip",
    "open_clip",
    "openclip",
    "zero-shot-image-classification",
    "zeroshotimageclassification",
    "logits_per_image",
    "logits per image",
    "vision classification",
)


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _normalize_text(value: object) -> str:
    return str(value or "").strip().lower()


def _contains_standalone_keyword(text: str, keyword: str) -> bool:
    if not text or not keyword:
        return False
    tokens = [token for token in re.split(r"[^a-z0-9]+", text) if token]
    return keyword in tokens


def _contains_token_classification_signal(text: str) -> bool:
    normalized = _normalize_text(text)
    if _contains_any(normalized, tuple(keyword for keyword in TOKEN_CLASSIFICATION_KEYWORDS if keyword != "ner")):
        return True
    return _contains_standalone_keyword(normalized, "ner")


def _contains_diffusion_signal(text: str) -> bool:
    normalized = _normalize_text(text)
    if _contains_any(normalized, DIFFUSION_KEYWORDS):
        return True
    if re.search(r"(?<![a-z0-9])sd-", normalized) is not None:
        return True
    # LTX text-to-video checkpoints do not carry explicit diffusion keywords in the repo id,
    # but they are still DiT denoising models for phase-4 latency-only evaluation.
    return re.search(r"(?<![a-z0-9])ltx(?:[-_ ]|$)", normalized) is not None


def _parse_int(value: object) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def _looks_like_model_class(auto_model_class: str, *needles: str) -> bool:
    return any(needle in auto_model_class for needle in needles)


def _is_protein_embedding_like(
    model_id: str,
    auto_model_class: str = "",
    *,
    architectures: str = "",
    problem_type: str = "",
) -> bool:
    model_id_lower = _normalize_text(model_id)
    auto_model_class_lower = _normalize_text(auto_model_class)
    architectures_lower = _normalize_text(architectures)
    problem_type_lower = _normalize_text(problem_type)
    combined = " ".join(part for part in (model_id_lower, auto_model_class_lower, architectures_lower, problem_type_lower) if part)

    has_protein_signal = _contains_any(combined, PROTEIN_MODEL_KEYWORDS)
    has_masked_lm_signal = any(
        signal in combined
        for signal in (
            "maskedlm",
            "masked language model",
            "fill-mask",
            "fill mask",
            "automodelformaskedlm",
            "esmformaskedlm",
        )
    )
    has_protein_seq2seq_signal = has_protein_signal and any(
        signal in combined
        for signal in (
            "prot_t5",
            "prott5",
            "t5",
            "seq2seq",
            "encoder-decoder",
            "encodermodel",
            "uniref50",
        )
    )
    if "esm" in combined:
        return True
    return has_protein_signal and (has_masked_lm_signal or has_protein_seq2seq_signal)


def _is_discriminator_like(
    model_id: str,
    auto_model_class: str = "",
    *,
    architectures: str = "",
) -> bool:
    model_id_lower = _normalize_text(model_id)
    auto_model_class_lower = _normalize_text(auto_model_class)
    architectures_lower = _normalize_text(architectures)
    combined = " ".join(part for part in (model_id_lower, auto_model_class_lower, architectures_lower) if part)

    if "electraforpretraining" in combined:
        return True
    if "automodelforpretraining" in auto_model_class_lower and ("electra" in combined or "discriminator" in combined):
        return True
    if "discriminator" in combined and "classification" not in combined and "sequenceclassification" not in combined:
        return True
    if "electra" in combined and any(signal in combined for signal in ("real/fake", "token replacement", "replaced token", "pretraining")):
        return True
    return False


def _is_audio_embedding_like(
    model_id: str,
    auto_model_class: str = "",
    *,
    architectures: str = "",
    problem_type: str = "",
) -> bool:
    model_id_lower = _normalize_text(model_id)
    auto_model_class_lower = _normalize_text(auto_model_class)
    architectures_lower = _normalize_text(architectures)
    problem_type_lower = _normalize_text(problem_type)
    combined = " ".join(part for part in (model_id_lower, auto_model_class_lower, architectures_lower, problem_type_lower) if part)

    if any(marker in combined for marker in AUDIO_ASR_HEAD_MARKERS):
        return False
    if any(marker in combined for marker in AUDIO_EMBEDDING_BACKBONE_MARKERS):
        return True
    return False


def _is_vision_embedding_like(
    model_id: str,
    auto_model_class: str = "",
    *,
    architectures: str = "",
    problem_type: str = "",
) -> bool:
    model_id_lower = _normalize_text(model_id)
    auto_model_class_lower = _normalize_text(auto_model_class)
    architectures_lower = _normalize_text(architectures)
    problem_type_lower = _normalize_text(problem_type)
    combined = " ".join(part for part in (model_id_lower, auto_model_class_lower, architectures_lower, problem_type_lower) if part)

    if any(token in combined for token in ("classification", "objectdetection", "questionanswering")):
        return False
    if _contains_any(combined, VISION_EMBEDDING_KEYWORDS):
        return True
    return "sam-vit" in model_id_lower


def _is_multimodal_embedding_like(
    model_id: str,
    auto_model_class: str = "",
    *,
    architectures: str = "",
    problem_type: str = "",
) -> bool:
    model_id_lower = _normalize_text(model_id)
    auto_model_class_lower = _normalize_text(auto_model_class)
    architectures_lower = _normalize_text(architectures)
    problem_type_lower = _normalize_text(problem_type)
    combined = " ".join(part for part in (model_id_lower, auto_model_class_lower, architectures_lower, problem_type_lower) if part)

    if not (_contains_any(combined, VLM_KEYWORDS) or "visiontextdualencoder" in combined):
        return False
    if _contains_any(combined, MULTIMODAL_EMBEDDING_EXCLUDE_KEYWORDS):
        return False
    if _contains_any(combined, RERANKER_KEYWORDS + QUESTION_ANSWERING_KEYWORDS):
        return False
    return _contains_any(combined, EMBEDDING_KEYWORDS + MULTIMODAL_EMBEDDING_KEYWORDS)


def _should_treat_bert_family_as_embedding(
    model_id: str,
    auto_model_class: str = "",
    *,
    architectures: str = "",
    problem_type: str = "",
    num_labels: object = None,
) -> bool:
    model_id_lower = _normalize_text(model_id)
    auto_model_class_lower = _normalize_text(auto_model_class)
    architectures_lower = _normalize_text(architectures)
    problem_type_lower = _normalize_text(problem_type)
    num_labels_value = _parse_int(num_labels)

    bert_like = any(
        _contains_any(text, BERT_FAMILY_KEYWORDS)
        for text in (model_id_lower, auto_model_class_lower, architectures_lower)
    )
    if not bert_like:
        return False

    if _looks_like_model_class(
        auto_model_class,
        "SequenceClassification",
        "QuestionAnswering",
        "TokenClassification",
        "MaskedLM",
        "CausalLM",
        "Seq2SeqLM",
        "ConditionalGeneration",
    ):
        return False

    task_keywords = (
        RERANKER_KEYWORDS
        + QUESTION_ANSWERING_KEYWORDS
        + SENTIMENT_KEYWORDS
        + EMOTION_KEYWORDS
        + OFFENSIVE_KEYWORDS
        + HATE_KEYWORDS
        + TOPIC_KEYWORDS
        + NLI_KEYWORDS
        + QUESTION_PAIR_KEYWORDS
        + GENERIC_CLASSIFICATION_KEYWORDS
    )
    if _contains_token_classification_signal(model_id_lower) or _contains_any(model_id_lower, task_keywords):
        return False

    if any(term in problem_type_lower for term in ("classification", "regression", "label")):
        return False

    # Bare encoder checkpoints often expose a default num_labels=2 in config metadata.
    # Treat only clearly task-specific label counts as a classification hint.
    if num_labels_value is not None and num_labels_value > 2:
        return False

    return True


def _is_instruct_chat_causal_lm(model_id: str) -> bool:
    return _contains_any(_normalize_text(model_id), CAUSAL_LM_INSTRUCT_KEYWORDS)


def _is_guard_generation_safety_model(model_id: str) -> bool:
    return _contains_any(_normalize_text(model_id), GUARD_GENERATION_KEYWORDS)


def _is_table_detection_like(
    model_id: str,
    auto_model_class: str = "",
    *,
    architectures: str = "",
    num_labels: object = None,
) -> bool:
    model_id_lower = _normalize_text(model_id)
    auto_model_class_lower = _normalize_text(auto_model_class)
    architectures_lower = _normalize_text(architectures)
    combined = " ".join(part for part in (model_id_lower, auto_model_class_lower, architectures_lower) if part)
    num_labels_value = _parse_int(num_labels)

    if "table-transformer-detection" in model_id_lower:
        return True
    if "table-transformer-structure-recognition" in model_id_lower:
        return False
    if "tabletransformerforobjectdetection" not in combined:
        return False
    if "table" not in combined:
        return False
    return num_labels_value in {None, 1, 2}


def detect_modality(model_id: str, auto_model_class: str = "", *, architectures: str = "") -> str:
    model_id_lower = _normalize_text(model_id)
    auto_model_class_lower = _normalize_text(auto_model_class)
    architectures_lower = _normalize_text(architectures)
    combined_model_signals = " ".join(
        part
        for part in (
            str(auto_model_class or ""),
            str(architectures or ""),
        )
        if part and str(part).strip()
    )
    combined_model_signals_lower = " ".join(
        part
        for part in (
            auto_model_class_lower,
            architectures_lower,
        )
        if part and str(part).strip()
    )
    multimodal_signal_text = " ".join(part for part in (model_id_lower, combined_model_signals_lower) if part)
    if _looks_like_model_class(combined_model_signals, "Vision2Seq", "VisionTextDualEncoder", "Omni", "Qwen2_5Omni") or _contains_any(multimodal_signal_text, VLM_KEYWORDS):
        return "multimodal"
    if _looks_like_model_class(combined_model_signals, "Image", "Vision", "ObjectDetection", "Keypoint", "VitMatte") or _contains_any(model_id_lower, VISION_CLASSIFICATION_KEYWORDS + VISION_DETECTION_KEYWORDS + KEYPOINT_DETECTION_KEYWORDS + MATTING_KEYWORDS):
        return "vision"
    if _looks_like_model_class(combined_model_signals, "Audio", "Speech", "Whisper", "Wav2Vec", "Hubert") or _contains_any(model_id_lower, ASR_KEYWORDS + AUDIO_EMBEDDING_KEYWORDS + TTS_KEYWORDS + FORCED_ALIGNER_KEYWORDS):
        return "audio"
    return "text"


def detect_model_type(
    model_id: str,
    auto_model_class: str = "",
    *,
    architectures: str = "",
    problem_type: str = "",
    num_labels: object = None,
) -> str:
    """
    检测模型类型，返回 dataset category key。

    Args:
        model_id: HuggingFace 模型 ID，如 "Qwen/Qwen2-0.5B"
        auto_model_class: AutoModel 类名，如 "AutoModelForCausalLM"

    Returns:
        模型类别字符串，如 "causal_lm"、"asr"、"diffusion"
    """
    model_id_lower = _normalize_text(model_id)
    modality = detect_modality(model_id, auto_model_class, architectures=architectures)

    combined_model_class = " ".join(
        text.strip()
        for text in (
            str(auto_model_class or ""),
            str(architectures or ""),
        )
        if text and text.strip()
    )
    has_explicit_generative_model_class = _looks_like_model_class(combined_model_class, "CausalLM", "Seq2SeqLM", "ConditionalGeneration", "Seq2Seq")

    if _contains_any(model_id_lower, RERANKER_KEYWORDS):
        return "reranker"
    has_seq2seq_signal = _looks_like_model_class(combined_model_class, "Seq2SeqLM", "ConditionalGeneration", "Seq2Seq") or _contains_any(model_id_lower, SEQ2SEQ_KEYWORDS)

    architecture_lower = _normalize_text(architectures)
    has_tts_signal = (
        _contains_any(model_id_lower, TTS_KEYWORDS)
        or _contains_any(architecture_lower, ("musicgen", "texttowaveform"))
        or _looks_like_model_class(combined_model_class, "TextToWaveform")
    )
    if _contains_any(model_id_lower, FORCED_ALIGNER_KEYWORDS):
        return "asr"
    if modality == "audio" and (
        _contains_any(model_id_lower, FORCED_ALIGNER_KEYWORDS)
        or _contains_any(architecture_lower, AUDIO_ASR_HEAD_MARKERS)
        or _looks_like_model_class(combined_model_class, "ForCTC", "SpeechSeq2Seq", "SpeechEncoderDecoder", "Whisper")
    ) and not has_tts_signal:
        return "asr"

    if _looks_like_model_class(combined_model_class, "QuestionAnswering"):
        return "question_answering"
    if _looks_like_model_class(combined_model_class, "MaskedLM"):
        return "masked_lm"
    if _looks_like_model_class(combined_model_class, "TokenClassification") or (_contains_token_classification_signal(model_id_lower) and not has_seq2seq_signal):
        return "token_classification"
    if _looks_like_model_class(combined_model_class, "ImageMatting", "VitMatte") or _contains_any(model_id_lower, MATTING_KEYWORDS) or _contains_any(_normalize_text(architectures), MATTING_KEYWORDS):
        return "image_matting"
    if _looks_like_model_class(combined_model_class, "Keypoint") or _contains_any(model_id_lower, KEYPOINT_DETECTION_KEYWORDS) or _contains_any(_normalize_text(architectures), KEYPOINT_DETECTION_KEYWORDS):
        return "vision_keypoint_detection"
    if _is_vision_embedding_like(
        model_id,
        auto_model_class,
        architectures=architectures,
        problem_type=problem_type,
    ):
        return "vision_embedding"
    if _is_discriminator_like(model_id, auto_model_class, architectures=architectures):
        return "discriminator"
    if _is_audio_embedding_like(
        model_id,
        auto_model_class,
        architectures=architectures,
        problem_type=problem_type,
    ):
        return "audio_embedding"
    if _is_multimodal_embedding_like(
        model_id,
        auto_model_class,
        architectures=architectures,
        problem_type=problem_type,
    ):
        return "embedding"
    if modality == "audio" and has_tts_signal:
        return "tts"
    if modality == "multimodal":
        return "vlm"
    if _looks_like_model_class(combined_model_class, "CausalLM"):
        return "causal_lm"
    # Multimodal checkpoints like Llava / DeepSeek-OCR can expose generation-style
    # heads, but their business workload should follow VLM evaluation instead of
    # text-only causal LM or seq2seq routes.
    if _looks_like_model_class(combined_model_class, "Seq2SeqLM", "ConditionalGeneration", "Seq2Seq"):
        return "asr" if _contains_any(model_id_lower, ASR_KEYWORDS) else "seq2seq"
    if _looks_like_model_class(combined_model_class, "SequenceClassification"):
        return "classification"
    if _contains_any(model_id_lower, QUESTION_ANSWERING_KEYWORDS) and not has_explicit_generative_model_class:
        return "question_answering"
    if _looks_like_model_class(combined_model_class, "ImageClassification"):
        return "vision_classification"
    if _looks_like_model_class(combined_model_class, "VisionEncoderDecoder"):
        vision_encoder_decoder_signals = " ".join(part for part in (model_id_lower, architecture_lower, _normalize_text(problem_type)) if part)
        if _contains_any(vision_encoder_decoder_signals, ("trocr", "ocr", "handwritten", "text-recognition", "text recognition")):
            return "vision_text_ocr"
        return "vision_detection"
    if _looks_like_model_class(combined_model_class, "ObjectDetection", "VisionEncoderDecoder"):
        return "vision_detection"
    if _contains_any(model_id_lower, FAIRFACE_AGE_KEYWORDS):
        return "vision_classification"
    if _contains_diffusion_signal(model_id_lower):
        return "diffusion"
    if _contains_any(model_id_lower, VIDEO_KEYWORDS):
        return "video"
    if _contains_any(model_id_lower, TTS_KEYWORDS):
        return "tts"
    if _contains_any(model_id_lower, ASR_KEYWORDS):
        return "asr"
    if _contains_any(model_id_lower, AUDIO_EMBEDDING_KEYWORDS):
        return "audio_embedding"
    if _contains_any(model_id_lower, VISION_DETECTION_KEYWORDS):
        return "vision_detection"
    if _contains_any(model_id_lower, VISION_CLASSIFICATION_KEYWORDS):
        return "vision_classification"
    if _contains_any(model_id_lower, EMBEDDING_KEYWORDS):
        return "embedding"
    if _should_treat_bert_family_as_embedding(
        model_id,
        auto_model_class,
        architectures=architectures,
        problem_type=problem_type,
        num_labels=num_labels,
    ):
        return "embedding"
    if _contains_any(model_id_lower, BIOMEDICAL_KEYWORDS):
        if _contains_any(model_id_lower, ("clip", "sam", "segment")):
            return "vision_classification"
        return "biomedical_nlp"
    if _contains_any(model_id_lower, BERT_FAMILY_KEYWORDS):
        return "classification"
    if _contains_any(model_id_lower, SEQ2SEQ_KEYWORDS):
        return "seq2seq"
    return "causal_lm"


def _detect_classification_intent(
    model_id: str,
    *,
    problem_type: str = "",
    num_labels: object = None,
) -> str:
    model_id_lower = _normalize_text(model_id)
    num_labels_value = _parse_int(num_labels)
    problem_type_lower = _normalize_text(problem_type)

    if _contains_any(model_id_lower, QUESTION_PAIR_KEYWORDS):
        return "question_pair_classification"
    if _contains_any(model_id_lower, NLI_KEYWORDS):
        return "natural_language_inference"
    if _contains_any(model_id_lower, EMOTION_KEYWORDS):
        return "emotion_multiclass"
    if _contains_any(model_id_lower, HATE_KEYWORDS):
        return "hate_binary"
    if _contains_any(model_id_lower, OFFENSIVE_KEYWORDS):
        return "offensive_binary"
    if _contains_any(model_id_lower, SENTIMENT_KEYWORDS):
        if (num_labels_value is not None and num_labels_value > 2) or _contains_any(model_id_lower, ("twitter", "tweet", "cardiffnlp", "multiclass", "three-way", "3class")):
            return "sentiment_multiclass"
        return "sentiment_binary"
    if _contains_any(model_id_lower, TOPIC_KEYWORDS):
        return "topic_classification"
    if num_labels_value is not None and num_labels_value > 2 and _contains_any(model_id_lower, ("intent", "category", "label", "labels")):
        return "topic_classification"
    if "regression" in problem_type_lower or "multi_label" in problem_type_lower:
        return "generic_classification"
    return "generic_classification"


def detect_business_intent(
    model_id: str,
    auto_model_class: str = "",
    *,
    architectures: str = "",
    problem_type: str = "",
    num_labels: object = None,
) -> str:
    model_id_lower = _normalize_text(model_id)
    if _is_protein_embedding_like(
        model_id,
        auto_model_class,
        architectures=architectures,
        problem_type=problem_type,
    ):
        return "protein_embedding"
    if _is_table_detection_like(
        model_id,
        auto_model_class,
        architectures=architectures,
        num_labels=num_labels,
    ):
        return "table_detection"
    model_type = detect_model_type(
        model_id,
        auto_model_class,
        architectures=architectures,
        problem_type=problem_type,
        num_labels=num_labels,
    )
    if model_type == "reranker":
        return "reranker"
    if model_type == "question_answering":
        return "extractive_qa"
    if model_type == "masked_lm":
        if _contains_any(model_id_lower, BIOMEDICAL_KEYWORDS):
            return "biomedical_masked_language_modeling"
        return "masked_language_modeling"
    if model_type == "token_classification":
        if "scienceie" in model_id_lower:
            return "scientific_token_classification"
        if _contains_any(model_id_lower, BIOMEDICAL_KEYWORDS):
            return "biomedical_token_classification"
        return "token_classification"
    if model_type == "discriminator":
        return "discriminator"
    if model_type == "embedding":
        return "embedding"
    if model_type == "audio_embedding":
        return "audio_embedding"
    if model_type == "vision_classification":
        if _contains_any(_normalize_text(model_id), FAIRFACE_AGE_KEYWORDS):
            return "face_age_classification"
        return "vision_classification"
    if model_type == "vision_embedding":
        return "vision_embedding"
    if model_type == "vision_detection":
        return "vision_detection"
    if model_type == "vision_keypoint_detection":
        return "vision_keypoint_detection"
    if model_type == "image_matting":
        return "image_matting"
    if _contains_any(
        model_id_lower + " " + _normalize_text(auto_model_class) + " " + _normalize_text(architectures),
        (
            "openvla",
            "vision-language-action",
            "vision language action",
            "action prediction",
            "actionprediction",
            "openvlaforactionprediction",
        ),
    ):
        return "vision_language_action"
    if model_type == "seq2seq":
        if _contains_any(_normalize_text(model_id), BIOMEDICAL_KEYWORDS):
            return "biomedical_seq2seq_qa"
        return "seq2seq_summarization"
    if model_type == "causal_lm":
        if _is_guard_generation_safety_model(model_id):
            return "safety_guard_generation"
        if _contains_any(_normalize_text(model_id), BIOMEDICAL_KEYWORDS):
            return "biomedical_causal_lm"
        return "causal_lm_instruct" if _is_instruct_chat_causal_lm(model_id) else "causal_lm_base"
    if model_type == "classification":
        if _should_treat_bert_family_as_embedding(
            model_id,
            auto_model_class,
            architectures=architectures,
            problem_type=problem_type,
            num_labels=num_labels,
        ):
            return "embedding"
        return _detect_classification_intent(model_id, problem_type=problem_type, num_labels=num_labels)
    if model_type == "biomedical_nlp":
        return "biomedical_nlp"
    return model_type


def _resolve_business_spec(
    model_id: str,
    auto_model_class: str = "",
    *,
    architectures: str = "",
    problem_type: str = "",
    num_labels: object = None,
) -> dict:
    business_intent = detect_business_intent(
        model_id,
        auto_model_class,
        architectures=architectures,
        problem_type=problem_type,
        num_labels=num_labels,
    )
    spec = dict(BUSINESS_INTENT_SPECS.get(business_intent, BUSINESS_INTENT_SPECS["specialized"]))
    multilingual_asr = resolve_multilingual_asr_dataset(model_id) if business_intent == "asr" else None
    if multilingual_asr:
        spec["dataset_key"] = multilingual_asr["dataset_key"]
    model_type = str(spec["model_type"])
    eval_profile_key = str(spec.get("evaluation_profile_key") or model_type)
    eval_profile = BUSINESS_EVAL_PROFILES.get(eval_profile_key, BUSINESS_EVAL_PROFILES["specialized"])
    if spec.get("dataset_key") == "mmlu":
        eval_profile = BUSINESS_EVAL_PROFILES["mmlu"]
    return {
        "business_intent": business_intent,
        "business_intent_name": BUSINESS_INTENT_NAMES.get(business_intent, business_intent),
        "model_type": model_type,
        "dataset_key": spec.get("dataset_key"),
        "evaluation_profile": eval_profile["evaluation_profile"],
        "primary_metric": eval_profile["primary_metric"],
        "secondary_metrics": list(eval_profile["secondary_metrics"]),
        "output_type_hint": eval_profile["output_type_hint"],
    }


def get_dataset_for_model(
    model_id: str,
    auto_model_class: str = "",
    *,
    architectures: str = "",
    problem_type: str = "",
    num_labels: object = None,
) -> Optional[str]:
    """
    获取模型对应的推荐数据集。

    Args:
        model_id: HuggingFace 模型 ID
        auto_model_class: AutoModel 类名（可选）

    Returns:
        数据集 key（如 "wikitext"），或 None 表示不需要数据集
    """
    model_type = detect_model_type(
        model_id,
        auto_model_class,
        architectures=architectures,
        problem_type=problem_type,
        num_labels=num_labels,
    )
    return DATASET_MAPPING.get(model_type)


def get_dataset_candidates_for_model(
    model_id: str,
    auto_model_class: str = "",
    *,
    architectures: str = "",
    problem_type: str = "",
    num_labels: object = None,
) -> list[str]:
    """获取模型可尝试的数据集候选，首项为默认推荐。"""
    model_type = detect_model_type(
        model_id,
        auto_model_class,
        architectures=architectures,
        problem_type=problem_type,
        num_labels=num_labels,
    )
    resolved = _resolve_business_spec(
        model_id,
        auto_model_class,
        architectures=architectures,
        problem_type=problem_type,
        num_labels=num_labels,
    )
    multilingual_asr = resolve_multilingual_asr_dataset(model_id) if resolved.get("model_type") == "asr" else None
    primary = multilingual_asr["dataset_key"] if multilingual_asr else (resolved.get("dataset_key") or DATASET_MAPPING.get(model_type))
    candidates = DATASET_CANDIDATES.get(model_type, [])
    ordered: list[str] = []
    seen: set[str] = set()
    for key in [primary, *candidates]:
        if not key:
            continue
        normalized = str(key).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def get_model_type_name(model_type: str) -> str:
    """获取模型类型的友好名称"""
    return MODEL_TYPE_NAMES.get(model_type, model_type)


def get_business_benchmark_profile(
    model_id: str,
    auto_model_class: str = "",
    *,
    architectures: str = "",
    problem_type: str = "",
    num_labels: object = None,
) -> dict:
    """返回第四阶段业务测评的数据集与评测画像。"""
    resolved = _resolve_business_spec(
        model_id,
        auto_model_class,
        architectures=architectures,
        problem_type=problem_type,
        num_labels=num_labels,
    )
    multilingual_asr = resolve_multilingual_asr_dataset(model_id) if resolved["model_type"] == "asr" else None
    return {
        "model_id": model_id,
        "model_class": auto_model_class or None,
        "model_type": resolved["model_type"],
        "model_type_name": get_model_type_name(str(resolved["model_type"])),
        "business_intent": resolved["business_intent"],
        "business_intent_name": resolved["business_intent_name"],
        "dataset_key": multilingual_asr["dataset_key"] if multilingual_asr else resolved["dataset_key"],
        "dataset_required": (multilingual_asr["dataset_key"] if multilingual_asr else resolved["dataset_key"]) is not None,
        "evaluation_profile": resolved["evaluation_profile"],
        "primary_metric": resolved["primary_metric"],
        "secondary_metrics": list(resolved["secondary_metrics"]),
        "output_type_hint": resolved["output_type_hint"],
        "asr_language": multilingual_asr["asr_language"] if multilingual_asr else None,
        "asr_task": multilingual_asr["asr_task"] if multilingual_asr else None,
    }


def main():
    parser = argparse.ArgumentParser(description="模型类型到数据集映射")
    parser.add_argument("--model_id", required=True, help="HuggingFace 模型 ID")
    parser.add_argument("--model_class", default="", help="AutoModel 类名（可选）")
    parser.add_argument("--architectures", default="", help="config.architectures（可选）")
    parser.add_argument("--problem-type", default="", help="config.problem_type（可选）")
    parser.add_argument("--num-labels", type=int, default=None, help="config.num_labels（可选）")
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式")
    parser.add_argument("--business-profile", action="store_true", help="输出第四阶段业务测评画像")
    parser.add_argument("--candidates", action="store_true", help="输出候选数据集列表，供样本不足时切换")
    args = parser.parse_args()

    model_type = detect_model_type(
        args.model_id,
        args.model_class,
        architectures=args.architectures,
        problem_type=args.problem_type,
        num_labels=args.num_labels,
    )
    dataset_key = get_dataset_for_model(
        args.model_id,
        args.model_class,
        architectures=args.architectures,
        problem_type=args.problem_type,
        num_labels=args.num_labels,
    )
    dataset_candidates = get_dataset_candidates_for_model(
        args.model_id,
        args.model_class,
        architectures=args.architectures,
        problem_type=args.problem_type,
        num_labels=args.num_labels,
    )
    type_name = get_model_type_name(model_type)

    if args.business_profile:
        profile = get_business_benchmark_profile(
            args.model_id,
            args.model_class,
            architectures=args.architectures,
            problem_type=args.problem_type,
            num_labels=args.num_labels,
        )
        if args.json:
            print(json.dumps(profile, indent=2, ensure_ascii=False))
        else:
            print(f"模型 ID: {profile['model_id']}")
            print(f"模型类别: {profile['model_type_name']} ({profile['model_type']})")
            print(f"业务意图: {profile['business_intent_name']} ({profile['business_intent']})")
            print(f"业务数据集: {profile['dataset_key'] or 'None (仅 latency)'}")
            print(f"业务评测画像: {profile['evaluation_profile']}")
            print(f"主指标: {profile['primary_metric']}")
            print(f"次指标: {', '.join(profile['secondary_metrics'])}")
            print(f"输出类型提示: {profile['output_type_hint']}")
        return

    if args.json:
        result = {
            "model_id": args.model_id,
            "model_class": args.model_class or None,
            "model_type": model_type,
            "model_type_name": type_name,
            "dataset_key": dataset_key,
            "dataset_candidates": dataset_candidates if args.candidates else None,
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"模型 ID: {args.model_id}")
        print(f"模型类别: {type_name} ({model_type})")
        if dataset_key:
            print(f"推荐数据集: {dataset_key}")
        else:
            print("推荐数据集: None (仅需 latency 测试)")
        if args.candidates:
            print(f"候选数据集: {', '.join(dataset_candidates) if dataset_candidates else 'None'}")


if __name__ == "__main__":
    main()
