#!/usr/bin/env python3
# CURSOR-MANAGED-BUSINESS-MODEL-EVAL
from __future__ import annotations

import ast
import importlib
import inspect
import io
import json
import math
import os
import random
import re
import sys
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ADAPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ADAPT_DIR.parents[1]
CACHE_DIR = ADAPT_DIR / "models"
MODEL_FILES_DIR = ADAPT_DIR / "model_files"
ACCURACY_RUN_PATH = ADAPT_DIR / "accuracy_run.py"
ACCURACY_RUN_PERF_PATH = ADAPT_DIR / "accuracy_run_perf.py"
LEGACY_OLMO_HELPER_PATHS = (
    ADAPT_DIR / "modeling_olmo_v1.py",
    MODEL_FILES_DIR / "modeling_olmo_v1.py",
    PROJECT_ROOT / "business_benchmark" / "templates" / "modeling_olmo_v1.py",
)
_MISSING_ATTR = object()
LABEL_TEXT_ALIASES = {
    "notentailment": {"notentailment", "nonentailment"},
    "notoffensive": {"notoffensive", "nonoffensive"},
    "nothate": {"nothate", "nonhate"},
    "sciencetechnology": {"sciencetechnology", "scitech", "technologyscience"},
}
OPEN_CLIP_MODEL_SPECS = {
    "BIOMEDICA/BMC_CLIP_CF": {
        "arch": "ViT-L-14",
        "checkpoint": "BMC_CLIP_CF.pt",
        "label_template": "a photo of a {}",
    }
}
_GENERATION_MIXIN_SHIM_CLASSES: dict[type, type] = {}
_TRANSFORMERS_PYTORCH_UTILS_COMPAT_APPLIED = False
_ORIGINAL_TORCH_LINSPACE = torch.linspace
_TORCH_LINSPACE_META_COMPAT_APPLIED = False
_ORIGINAL_TORCH_MODULE_GETATTR = torch.nn.Module.__getattr__
_TORCH_MODULE_TIED_WEIGHTS_COMPAT_APPLIED = False
_BIREFNET_NPU_DEFORM_CONV_PATCHED_CLASSES: set[int] = set()
_HF_OLMO_TIE_WEIGHTS_COMPAT_APPLIED = False
_ORIGINAL_HF_OLMO_TIE_WEIGHTS = None
QWEN_VLM_RERANKER_SYSTEM_PROMPT = 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
QWEN_VLM_RERANKER_DEFAULT_INSTRUCTION = "Given a search query, retrieve relevant candidates that answer the query."
QWEN3_GUARD_SAFETY_LABELS = ("safe", "unsafe", "controversial")
F5_TTS_DEFAULT_PROMPTS = (
    "ยินดีที่ได้รู้จักคุณวันนี้อากาศดีมาก",
    "ระบบสังเคราะห์เสียงภาษาไทยกำลังทำงานบนชิป Ascend",
    "ขอบคุณที่ช่วยทดสอบประสิทธิภาพของโมเดลนี้",
    "การวัดผลครั้งนี้ต้องใช้พารามิเตอร์และข้อมูลชุดเดียวกัน",
    "เราต้องการผลลัพธ์ที่เปรียบเทียบได้อย่างยุติธรรม",
    "แบบจำลองนี้ใช้สถาปัตยกรรม F5 TTS สำหรับภาษาไทย",
)


def _apply_transformers_pytorch_utils_compatibility_shims() -> None:
    global _TRANSFORMERS_PYTORCH_UTILS_COMPAT_APPLIED
    if _TRANSFORMERS_PYTORCH_UTILS_COMPAT_APPLIED:
        return
    try:
        import transformers.pytorch_utils as pytorch_utils
    except Exception:
        return

    conv1d_cls = getattr(pytorch_utils, "Conv1D", None)
    if conv1d_cls is None:
        return

    if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):

        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size, dtype=torch.bool)
            heads = set(heads) - set(already_pruned_heads)
            for head in heads:
                head = head - sum(1 if pruned_head < head else 0 for pruned_head in already_pruned_heads)
                mask[head] = False
            index = torch.arange(n_heads * head_size, dtype=torch.long)[mask.view(-1)]
            return heads, index

        pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    if not hasattr(pytorch_utils, "prune_conv1d_layer"):

        def prune_conv1d_layer(layer, index, dim=1):
            index = index.to(layer.weight.device)
            weight = layer.weight.index_select(dim, index).clone().detach()
            if dim == 0:
                bias = layer.bias[index].clone().detach()
            else:
                bias = layer.bias.clone().detach()
            new_size = list(layer.weight.size())
            new_size[dim] = index.numel()
            new_layer = conv1d_cls(new_size[1], new_size[0]).to(layer.weight.device)
            new_layer.weight.requires_grad = False
            new_layer.bias.requires_grad = False
            new_layer.weight.copy_(weight.contiguous())
            new_layer.bias.copy_(bias.contiguous())
            new_layer.weight.requires_grad = True
            new_layer.bias.requires_grad = True
            return new_layer

        pytorch_utils.prune_conv1d_layer = prune_conv1d_layer

    if "transformers.utils.model_parallel_utils" not in sys.modules:

        def get_device_map(n_layers: int, devices) -> dict[Any, list[int]]:
            device_list = list(devices)
            if not device_list:
                return {}
            layers = list(range(int(n_layers)))
            per_device = max(1, math.ceil(len(layers) / len(device_list)))
            return {device: layers[idx * per_device : (idx + 1) * per_device] for idx, device in enumerate(device_list) if layers[idx * per_device : (idx + 1) * per_device]}

        def assert_device_map(device_map: Mapping[Any, list[int]], num_blocks: int) -> None:
            if not isinstance(device_map, Mapping):
                raise ValueError(f"device_map must be a mapping, got {type(device_map)!r}")
            flattened: list[int] = []
            for blocks in device_map.values():
                flattened.extend(int(block) for block in blocks)
            expected = list(range(int(num_blocks)))
            if sorted(flattened) != expected:
                raise ValueError(f"device_map must cover each block exactly once; expected={expected}, got={sorted(flattened)}")

        model_parallel_utils = ModuleType("transformers.utils.model_parallel_utils")
        model_parallel_utils.assert_device_map = assert_device_map
        model_parallel_utils.get_device_map = get_device_map
        sys.modules["transformers.utils.model_parallel_utils"] = model_parallel_utils
        try:
            import transformers.utils as transformers_utils

            setattr(transformers_utils, "model_parallel_utils", model_parallel_utils)
        except Exception:
            pass

    _TRANSFORMERS_PYTORCH_UTILS_COMPAT_APPLIED = True


def _apply_torch_linspace_meta_compatibility_shim() -> None:
    global _TORCH_LINSPACE_META_COMPAT_APPLIED
    if _TORCH_LINSPACE_META_COMPAT_APPLIED:
        return

    def _patched_linspace(*args, device=None, **kwargs):
        linspace_kwargs = dict(kwargs)
        effective_device = device if device is not None else linspace_kwargs.get("device")
        if device is not None:
            linspace_kwargs["device"] = device
        result = _ORIGINAL_TORCH_LINSPACE(*args, **linspace_kwargs)
        if not getattr(result, "is_meta", False):
            return result
        fallback_device = effective_device
        if fallback_device is None or str(fallback_device) == "meta":
            fallback_device = "cpu"
        linspace_kwargs["device"] = fallback_device
        return _ORIGINAL_TORCH_LINSPACE(*args, **linspace_kwargs)

    torch.linspace = _patched_linspace
    _TORCH_LINSPACE_META_COMPAT_APPLIED = True


def _apply_torch_module_tied_weights_compatibility_shim() -> None:
    global _TORCH_MODULE_TIED_WEIGHTS_COMPAT_APPLIED
    if _TORCH_MODULE_TIED_WEIGHTS_COMPAT_APPLIED:
        return

    def _patched_module_getattr(self, name):
        if name == "all_tied_weights_keys":
            return {}
        return _ORIGINAL_TORCH_MODULE_GETATTR(self, name)

    torch.nn.Module.__getattr__ = _patched_module_getattr
    _TORCH_MODULE_TIED_WEIGHTS_COMPAT_APPLIED = True


def _ensure_model_has_all_tied_weights_keys(model) -> None:
    if hasattr(model, "all_tied_weights_keys"):
        return
    try:
        setattr(model, "all_tied_weights_keys", getattr(model, "_tied_weights_keys", {}) or {})
    except Exception:
        pass


@contextmanager
def _temporary_transformers_tied_weights_loading_compatibility():
    _apply_torch_module_tied_weights_compatibility_shim()
    try:
        import transformers.modeling_utils as modeling_utils
    except Exception:
        yield
        return

    pre_trained_model_cls = getattr(modeling_utils, "PreTrainedModel", None)
    original_finalize = getattr(pre_trained_model_cls, "_finalize_model_loading", None) if pre_trained_model_cls is not None else None
    original_adjust = getattr(pre_trained_model_cls, "_adjust_tied_keys_with_tied_pointers", None) if pre_trained_model_cls is not None else None
    accelerate_module = sys.modules.get("transformers.integrations.accelerate")
    original_infer = getattr(accelerate_module, "infer_auto_device_map", None) if accelerate_module is not None else None

    def _safe_finalize(self, *args, **kwargs):
        _ensure_model_has_all_tied_weights_keys(self)
        return original_finalize(self, *args, **kwargs)

    def _safe_adjust(self, *args, **kwargs):
        _ensure_model_has_all_tied_weights_keys(self)
        return original_adjust(self, *args, **kwargs)

    def _safe_infer(model, *args, **kwargs):
        _ensure_model_has_all_tied_weights_keys(model)
        return original_infer(model, *args, **kwargs)

    try:
        if callable(original_finalize):
            pre_trained_model_cls._finalize_model_loading = _safe_finalize
        if callable(original_adjust):
            pre_trained_model_cls._adjust_tied_keys_with_tied_pointers = _safe_adjust
        if callable(original_infer) and accelerate_module is not None:
            accelerate_module.infer_auto_device_map = _safe_infer
        yield
    finally:
        if callable(original_finalize):
            pre_trained_model_cls._finalize_model_loading = original_finalize
        if callable(original_adjust):
            pre_trained_model_cls._adjust_tied_keys_with_tied_pointers = original_adjust
        if callable(original_infer) and accelerate_module is not None:
            accelerate_module.infer_auto_device_map = original_infer


def _apply_birefnet_npu_deform_conv_compat(candidate_modules: list[Any]) -> list[str]:
    try:
        import torch_npu
    except Exception:
        return []

    applied: list[str] = []

    def _as_pair(value: Any) -> tuple[int, int]:
        if isinstance(value, tuple):
            return int(value[0]), int(value[1])
        if isinstance(value, list):
            if len(value) >= 2:
                return int(value[0]), int(value[1])
            if len(value) == 1:
                return int(value[0]), int(value[0])
        return int(value), int(value)

    for candidate in candidate_modules:
        if candidate is None:
            continue
        try:
            module_iter = candidate.modules()
        except Exception:
            continue
        for submodule in module_iter:
            submodule_cls = getattr(submodule, "__class__", None)
            if submodule_cls is None or getattr(submodule_cls, "__name__", "") != "DeformableConv2d":
                continue
            if not all(hasattr(submodule, attr) for attr in ("offset_conv", "modulator_conv", "regular_conv")):
                continue
            cls_id = id(submodule_cls)
            if cls_id in _BIREFNET_NPU_DEFORM_CONV_PATCHED_CLASSES:
                continue
            original_forward = getattr(submodule_cls, "forward", None)
            if not callable(original_forward):
                continue

            def _patched_forward(self, x, _original_forward=original_forward):
                if not isinstance(x, torch.Tensor) or not str(getattr(x, "device", "")).startswith("npu"):
                    return _original_forward(self, x)
                if not hasattr(torch, "npu") or not torch.npu.is_available():
                    return _original_forward(self, x)

                kernel_h, kernel_w = _as_pair(getattr(self.regular_conv, "kernel_size", 3))
                stride_h, stride_w = _as_pair(getattr(self, "stride", getattr(self.regular_conv, "stride", 1)))
                padding_h, padding_w = _as_pair(getattr(self, "padding", getattr(self.regular_conv, "padding", 0)))
                dilation_h, dilation_w = _as_pair(getattr(self.regular_conv, "dilation", 1))

                offset = self.offset_conv(x)
                modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))

                if offset.shape[1] % 2 != 0:
                    return _original_forward(self, x)
                offset_y = offset[:, 0::2, :, :]
                offset_x = offset[:, 1::2, :, :]
                if offset_x.shape[1] != offset_y.shape[1] or modulator.shape[1] != offset_x.shape[1]:
                    return _original_forward(self, x)

                bias = self.regular_conv.bias
                compute_dtype = torch.float32
                input_nchw = x.contiguous().to(dtype=compute_dtype)
                weight_oihw = self.regular_conv.weight.contiguous().to(dtype=compute_dtype)
                offset_all = torch.cat([offset_x, offset_y, modulator], dim=1).contiguous().to(dtype=compute_dtype)
                bias_fp32 = bias.to(dtype=compute_dtype) if bias is not None else None
                deformable_groups = max(1, int(offset_x.shape[1] // max(1, kernel_h * kernel_w)))

                output, _ = torch_npu.npu_deformable_conv2d(
                    input_nchw,
                    weight_oihw,
                    offset_all,
                    bias_fp32,
                    kernel_size=[kernel_h, kernel_w],
                    stride=[1, stride_h, stride_w, 1],
                    padding=[padding_h, padding_h, padding_w, padding_w],
                    dilation=[1, dilation_h, dilation_w, 1],
                    groups=int(getattr(self.regular_conv, "groups", 1)),
                    deformable_groups=deformable_groups,
                    modulated=True,
                )
                if torch.is_floating_point(x) and output.dtype != x.dtype:
                    output = output.to(dtype=x.dtype)
                return output

            submodule_cls.forward = _patched_forward
            _BIREFNET_NPU_DEFORM_CONV_PATCHED_CLASSES.add(cls_id)
            applied.append(f"{submodule_cls.__module__}.{submodule_cls.__name__}.forward")
    return applied


def _apply_hf_olmo_compatibility_shims() -> None:
    global _HF_OLMO_TIE_WEIGHTS_COMPAT_APPLIED, _ORIGINAL_HF_OLMO_TIE_WEIGHTS
    if _HF_OLMO_TIE_WEIGHTS_COMPAT_APPLIED:
        return
    try:
        from hf_olmo import OLMoForCausalLM
    except Exception:
        return

    original_tie_weights = getattr(OLMoForCausalLM, "tie_weights", None)
    if callable(original_tie_weights):
        _ORIGINAL_HF_OLMO_TIE_WEIGHTS = original_tie_weights

        def _patched_tie_weights(self, *args, **kwargs):
            return _ORIGINAL_HF_OLMO_TIE_WEIGHTS(self)

        OLMoForCausalLM.tie_weights = _patched_tie_weights

    _HF_OLMO_TIE_WEIGHTS_COMPAT_APPLIED = True


class _OpenClipInputAdapter:
    def __init__(self, preprocess, tokenizer, label_template: str):
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.label_template = label_template

    def build_label_texts(self, label_space: list[str]) -> list[str]:
        prompts: list[str] = []
        for label in label_space:
            label_text = str(label).strip().replace("_", " ")
            if "{}" in self.label_template:
                prompts.append(self.label_template.format(label_text))
            else:
                prompts.append(f"{self.label_template} {label_text}".strip())
        return prompts


class _TransformersZeroShotVisionInputAdapter:
    def __init__(self, image_processor, tokenizer, label_template: str, text_preprocessor=None):
        self.image_processor = image_processor
        self.text_tokenizer = tokenizer
        self.label_template = label_template
        self.text_preprocessor = text_preprocessor

    def __call__(self, *args, **kwargs):
        return self.image_processor(*args, **kwargs)

    def tokenizer(self, texts):
        if callable(self.text_preprocessor):
            encoded = self.text_preprocessor(text=texts, return_tensors="pt")
            if hasattr(encoded, "items"):
                allowed_keys = set(getattr(self.text_tokenizer, "model_input_names", []) or [])
                allowed_keys.update({"position_ids", "token_type_ids"})
                filtered = {key: value for key, value in encoded.items() if not allowed_keys or key in allowed_keys}
                if filtered:
                    return filtered
            return encoded
        return self.text_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    def preprocess(self, image):
        processed = self.image_processor(images=image, return_tensors="pt")
        return processed

    def build_label_texts(self, label_space: list[str]) -> list[str]:
        prompts: list[str] = []
        for label in label_space:
            label_text = str(label).strip().replace("_", " ")
            if "{}" in self.label_template:
                prompts.append(self.label_template.format(label_text))
            else:
                prompts.append(f"{self.label_template} {label_text}".strip())
        return prompts


class _BiRefNetImageMattingProcessor:
    def __init__(self, resolution: tuple[int, int] = (1024, 1024)) -> None:
        self.resolution = resolution
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, images=None, trimaps=None, return_tensors: str = "pt", **_: Any):
        from PIL import Image as PILImage

        if return_tensors != "pt":
            raise ValueError(f"BiRefNet processor only supports return_tensors='pt', got {return_tensors!r}")
        image = images[0] if isinstance(images, (list, tuple)) else images
        if image is None:
            raise RuntimeError("BiRefNet image input is empty")
        if not isinstance(image, PILImage.Image):
            image = PILImage.fromarray(np.array(image, copy=True))
        image = image.convert("RGB")
        image = image.resize(self.resolution, resample=PILImage.BILINEAR)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = (image_array - self.mean) / self.std
        pixel_values = torch.from_numpy(np.transpose(image_array, (2, 0, 1))).unsqueeze(0)
        return {"pixel_values": pixel_values}


class _VocosInputAdapter:
    def __init__(self, *, sample_rate: int, n_mels: int, hop_length: int, duration_seconds: float) -> None:
        self.backend = "vocos"
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.duration_seconds = duration_seconds


def _is_birefnet_model(model_id: str, model_config: Any | None = None) -> bool:
    model_id_text = str(model_id or "").strip().lower()
    if model_id_text == "zhengpeng7/birefnet":
        return True
    if model_config is None:
        return False
    architectures = [str(item).strip().lower() for item in list(getattr(model_config, "architectures", []) or []) if str(item).strip()]
    auto_map = getattr(model_config, "auto_map", {}) or {}
    auto_map_values = " ".join(str(value).strip().lower() for value in auto_map.values())
    return any("birefnet" in arch for arch in architectures) or "birefnet" in auto_map_values


def _looks_like_birefnet_request(config: dict[str, Any] | None, model_id: str) -> bool:
    model_id_text = str(model_id or "").strip().lower()
    if "birefnet" in model_id_text:
        return True
    if not isinstance(config, dict):
        return False
    model_type = str(config.get("model_type") or "").strip().lower()
    evaluation_profile = str(config.get("evaluation_profile") or "").strip().lower()
    output_type_hint = str(config.get("output_type_hint") or "").strip().lower()
    architectures = str(config.get("architectures") or "").strip().lower()
    return model_type == "image_matting" or evaluation_profile == "image_matting" or output_type_hint in {"alpha_matte", "alpha_mask", "matte", "image_matting_mask"} or "birefnet" in architectures


def _looks_like_clipseg_request(config: dict[str, Any] | None, model_id: str, *, model_config: Any | None = None) -> bool:
    signal_parts = [str(model_id or "").strip().lower()]
    if isinstance(config, dict):
        signal_parts.extend(
            str(config.get(key) or "").strip().lower()
            for key in (
                "model_type",
                "model_backend",
                "business_intent",
                "evaluation_profile",
                "output_type_hint",
                "architectures",
                "model_class",
            )
        )
    if model_config is not None:
        signal_parts.append(str(getattr(model_config, "model_type", "") or "").strip().lower())
        signal_parts.extend(str(item).strip().lower() for item in list(getattr(model_config, "architectures", []) or []) if str(item).strip())
    combined = " ".join(part for part in signal_parts if part)
    return "clipseg" in combined or "clipsegforimagesegmentation" in combined


def _looks_like_vocos_request(config: dict[str, Any] | None, model_id: str) -> bool:
    model_id_text = str(model_id or "").strip().lower()
    if "vocos" in model_id_text:
        return True
    if not isinstance(config, dict):
        return False
    backend = str(config.get("model_backend") or "").strip().lower()
    architectures = str(config.get("architectures") or "").strip().lower()
    model_file = str(config.get("model_file") or "").strip().lower()
    return any("vocos" in candidate for candidate in (backend, architectures, model_file))


def _load_vocos_stack(config: dict, model_id: str, scenario: str, target_device: torch.device, load_context: dict[str, Any]):
    if not _looks_like_vocos_request(config, model_id):
        return None

    try:
        from vocos import Vocos
    except ImportError:
        if "vocos" not in str(model_id or "").strip().lower():
            return None
        raise

    model_source, _, load_context = _resolve_model_sources(model_id, config, scenario, "processor")
    torch_dtype = _get_torch_dtype(config)
    _apply_model_load_seed(config)
    model = Vocos.from_pretrained(model_source)
    move_kwargs: dict[str, Any] = {"device": target_device}
    if torch_dtype != "auto":
        move_kwargs["dtype"] = torch_dtype
    model = model.to(**move_kwargs)
    patch_metadata = load_context.get("_patch_metadata")
    if isinstance(patch_metadata, dict):
        _apply_deferred_model_files_hooks(model, patch_metadata)
        load_context["patch_load_status"] = str(patch_metadata.get("status") or load_context.get("patch_load_status") or "disabled")
        load_context["patch_modules"] = list(patch_metadata.get("imported_modules") or [])
        load_context["patch_hooks"] = list(patch_metadata.get("called_hooks") or [])
        load_context["patch_errors"] = list(patch_metadata.get("errors") or [])

    sample_rate_candidates = (
        config.get("sample_rate"),
        config.get("tts_sample_rate"),
        getattr(getattr(model, "feature_extractor", None), "sample_rate", None),
        getattr(getattr(model, "config", None), "sample_rate", None),
        getattr(getattr(model, "config", None), "sampling_rate", None),
    )
    sample_rate = 24000
    for candidate in sample_rate_candidates:
        if isinstance(candidate, (int, float)) and int(candidate) > 0:
            sample_rate = int(candidate)
            break
    n_mels = int(config.get("tts_n_mels") or config.get("vocoder_n_mels") or 100)
    hop_length = int(config.get("tts_hop_length") or config.get("vocoder_hop_length") or 256)
    duration_seconds = float(config.get("tts_duration_seconds") or config.get("vocoder_duration_seconds") or 1.0)
    load_context["input_source"] = "builtin_vocos_runtime_adapter"
    load_context["input_source_kind"] = "vocos_runtime_adapter"
    model.eval()
    model_config = SimpleNamespace(id2label={}, num_labels=0, model_type="vocos", sample_rate=sample_rate)
    input_adapter = _VocosInputAdapter(
        sample_rate=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length,
        duration_seconds=duration_seconds,
    )
    return model, input_adapter, model_config, "tts", load_context


def _load_birefnet_image_matting_stack(config: dict, model_id: str, scenario: str, target_device: torch.device, load_context: dict[str, Any]):
    if not _looks_like_birefnet_request(config, model_id):
        return None

    from transformers import AutoConfig, AutoModelForImageSegmentation

    trust_remote_code = _as_bool(config.get("trust_remote_code"), default=False)
    model_source, input_source, load_context = _resolve_model_sources(model_id, config, scenario, "image_processor")
    try:
        model_config = AutoConfig.from_pretrained(model_source, cache_dir=str(CACHE_DIR), trust_remote_code=trust_remote_code)
    except Exception:
        if "birefnet" not in str(model_id or "").strip().lower():
            return None
        raise
    if not _is_birefnet_model(model_id, model_config=model_config):
        return None

    _apply_model_config_compatibility_fixes(model_config, model_id=model_id)
    torch_dtype = _get_torch_dtype(config)
    load_kwargs = {
        "cache_dir": str(CACHE_DIR),
        "trust_remote_code": trust_remote_code,
        "config": model_config,
    }
    if torch_dtype != "auto":
        load_kwargs["torch_dtype"] = torch_dtype
    _apply_model_load_seed(config)
    model = AutoModelForImageSegmentation.from_pretrained(model_source, **load_kwargs)
    move_kwargs: dict[str, Any] = {"device": target_device}
    if torch_dtype != "auto":
        move_kwargs["dtype"] = torch_dtype
    model = model.to(**move_kwargs)
    load_context["runtime_compatibility_shims"] = _apply_model_runtime_compatibility_shims(model)
    patch_metadata = load_context.get("_patch_metadata")
    if isinstance(patch_metadata, dict):
        _apply_deferred_model_files_hooks(model, patch_metadata)
        load_context["patch_load_status"] = str(patch_metadata.get("status") or load_context.get("patch_load_status") or "disabled")
        load_context["patch_modules"] = list(patch_metadata.get("imported_modules") or [])
        load_context["patch_hooks"] = list(patch_metadata.get("called_hooks") or [])
        load_context["patch_errors"] = list(patch_metadata.get("errors") or [])
    load_context["activation_name_shims"] = _stabilize_activation_name_shims(model)
    load_context["input_source"] = str(input_source or model_source)
    load_context["input_source_kind"] = "custom_birefnet_processor"
    model.eval()
    return model, _BiRefNetImageMattingProcessor(), model_config, "image_matting", load_context


class _TimmImageInputAdapter:
    def __init__(self, transform):
        self.backend = "timm_image_classification"
        self.transform = transform


class _SegmentAnythingImageInputAdapter:
    def __init__(self, image_size: int = 1024):
        self.backend = "segment_anything_image_encoder"
        self.image_size = image_size

    def preprocess(self, image):
        decoded = _decode_image_sample(image)
        if decoded is None:
            return None
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("segment-anything 业务测评需要 Pillow 支持") from exc
        if not isinstance(decoded, Image.Image):
            if isinstance(decoded, np.ndarray):
                decoded = Image.fromarray(decoded.astype(np.uint8)).convert("RGB")
            else:
                raise RuntimeError(f"segment-anything image encoder 不支持的输入类型: {type(decoded)!r}")
        resized = decoded.convert("RGB").resize((self.image_size, self.image_size))
        tensor = torch.from_numpy(np.asarray(resized, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
        return tensor


class _SegmentAnythingImageEncoderWrapper(torch.nn.Module):
    def __init__(self, sam):
        super().__init__()
        self.sam = sam
        self.image_encoder = sam.image_encoder

    def forward(self, pixel_values):
        if hasattr(self.sam, "preprocess"):
            pixel_values = self.sam.preprocess(pixel_values)
        return self.image_encoder(pixel_values)


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _apply_model_load_seed(config: dict[str, Any] | None) -> int:
    seed = 20260413
    if isinstance(config, dict):
        try:
            configured_seed = int(config.get("model_load_seed", seed))
        except (TypeError, ValueError):
            configured_seed = seed
        if configured_seed >= 0:
            seed = configured_seed
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "npu") and hasattr(torch.npu, "manual_seed_all"):
        try:
            torch.npu.manual_seed_all(seed)
        except Exception:
            pass
    return seed


def _normalize_label_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return "".join(ch for ch in text if ch.isalnum())


def _label_candidate_keys(value: Any) -> set[str]:
    normalized = _normalize_label_text(value)
    if not normalized:
        return set()
    candidates = {normalized}
    for alias_group in LABEL_TEXT_ALIASES.values():
        if normalized in alias_group:
            candidates.update(alias_group)
    return candidates


def _parse_label_id(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return str(int(value))
    text = str(value).strip()
    if not text:
        return None
    try:
        return str(int(float(text)))
    except Exception:
        return None


def _build_dataset_label_map(samples: list[dict]) -> dict[str, str]:
    label_map: dict[str, str] = {}
    for sample in samples:
        dataset_label_names = sample.get("dataset_label_names")
        if isinstance(dataset_label_names, list):
            for idx, label_name in enumerate(dataset_label_names):
                for key in _label_candidate_keys(label_name):
                    label_map.setdefault(key, str(idx))
        reference_id = _parse_label_id(sample.get("reference"))
        reference_label_name = sample.get("reference_label_name")
        if reference_id is None:
            continue
        for key in _label_candidate_keys(reference_label_name):
            label_map.setdefault(key, reference_id)
    return label_map


def _map_model_label_to_dataset_id(pred_id: int, config, dataset_label_map: dict[str, str]) -> tuple[str, str]:
    if not dataset_label_map:
        return str(pred_id), "raw_id"
    id2label_raw = getattr(config, "id2label", None) or {}
    model_label_text = ""
    if isinstance(id2label_raw, dict):
        model_label_text = str(id2label_raw.get(pred_id, "")).strip()
        if not model_label_text:
            model_label_text = str(id2label_raw.get(str(pred_id), "")).strip()
    for key in _label_candidate_keys(model_label_text):
        mapped_id = dataset_label_map.get(key)
        if mapped_id is not None:
            return mapped_id, "semantic_label"
    return str(pred_id), "raw_id"


def _load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to create module spec for {module_name}")
    module = importlib.util.module_from_spec(spec)
    module.__dict__.setdefault("torch", torch)
    parent_name, _, attr_name = module_name.rpartition(".")
    old_module = sys.modules.get(module_name)
    parent_module = None
    old_parent_attr = _MISSING_ATTR
    if parent_name:
        parent_module = importlib.import_module(parent_name)
        old_parent_attr = getattr(parent_module, attr_name, _MISSING_ATTR)
    sys.modules[module_name] = module
    if parent_module is not None:
        setattr(parent_module, attr_name, module)
    try:
        spec.loader.exec_module(module)
    except Exception:
        if old_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = old_module
        if parent_module is not None:
            if old_parent_attr is _MISSING_ATTR:
                try:
                    delattr(parent_module, attr_name)
                except AttributeError:
                    pass
            else:
                setattr(parent_module, attr_name, old_parent_attr)
        raise
    return module


def _iter_patch_hook_names(module) -> list[str]:
    preferred_names = (
        "apply_npu_patches",
        "_apply_npu_patches",
        "apply_npu_optimizations",
        "_apply_npu_optimizations",
        "apply_model_patches",
        "_apply_model_patches",
        "patch_model",
        "_patch_model",
    )
    discovered_names: list[str] = []
    seen_names: set[str] = set()
    for hook_name in preferred_names:
        if callable(getattr(module, hook_name, None)):
            discovered_names.append(hook_name)
            seen_names.add(hook_name)
    for hook_name in sorted(dir(module)):
        if hook_name in seen_names or hook_name.startswith("__"):
            continue
        is_apply_style = hook_name.startswith("apply_") or hook_name.startswith("_apply_")
        is_patch_style = hook_name.startswith("patch_") or hook_name.startswith("_patch_")
        if not (is_apply_style or is_patch_style):
            continue
        normalized = hook_name.lower()
        if is_apply_style and not any(token in normalized for token in ("npu", "patch", "optim")):
            continue
        if callable(getattr(module, hook_name, None)):
            discovered_names.append(hook_name)
            seen_names.add(hook_name)
    return discovered_names


def _register_patch_hooks(display_module_name: str, module, metadata: dict[str, Any]) -> None:
    for hook_name in _iter_patch_hook_names(module):
        hook = getattr(module, hook_name, None)
        if not callable(hook):
            continue
        hook_ref = f"{display_module_name}.{hook_name}"
        try:
            signature = inspect.signature(hook)
        except (TypeError, ValueError):
            signature = None
        if signature is not None:
            required_positional_params = [parameter for parameter in signature.parameters.values() if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) and parameter.default is inspect._empty]
            if len(required_positional_params) > 1:
                required_names = ", ".join(parameter.name for parameter in required_positional_params)
                metadata["errors"].append(f"{hook_ref}: unsupported required args ({required_names})")
                continue
            if len(required_positional_params) == 1:
                metadata["deferred_hooks"].append((hook_ref, hook))
                continue
        try:
            hook()
            metadata["called_hooks"].append(hook_ref)
        except Exception as exc:
            if "required positional argument" in str(exc):
                metadata["deferred_hooks"].append((hook_ref, hook))
                continue
            metadata["errors"].append(f"{hook_ref}: {exc}")


def _merge_patch_metadata(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "status": str(primary.get("status") or "missing"),
        "imported_modules": list(primary.get("imported_modules") or []),
        "called_hooks": list(primary.get("called_hooks") or []),
        "deferred_hooks": list(primary.get("deferred_hooks") or []),
        "errors": list(primary.get("errors") or []),
    }
    for key in ("imported_modules", "called_hooks", "errors"):
        for item in secondary.get(key) or []:
            if item not in merged[key]:
                merged[key].append(item)
    merged["deferred_hooks"].extend(secondary.get("deferred_hooks") or [])
    secondary_status = str(secondary.get("status") or "missing")
    if merged["called_hooks"]:
        merged["status"] = "applied"
    elif merged["deferred_hooks"]:
        merged["status"] = "loaded"
    elif secondary_status not in {"missing", "disabled"}:
        merged["status"] = secondary_status
    return merged


def _prepare_model_files_for_perf():
    metadata = {
        "status": "missing",
        "imported_modules": [],
        "called_hooks": [],
        "deferred_hooks": [],
        "errors": [],
    }
    if not MODEL_FILES_DIR.exists():
        return metadata
    if str(ADAPT_DIR) not in sys.path:
        sys.path.insert(0, str(ADAPT_DIR))
    if str(MODEL_FILES_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL_FILES_DIR))

    def _iter_module_paths():
        for module_path in sorted(MODEL_FILES_DIR.glob("*.py")):
            if module_path.name == "__init__.py" or module_path.name.startswith("_"):
                continue
            yield module_path

    module_specs: list[tuple[str, Path | None]] = []
    if (MODEL_FILES_DIR / "__init__.py").exists():
        try:
            import model_files as _model_files_pkg

            del _model_files_pkg  # unused
            module_specs.append(("model_files", None))
            for module_path in _iter_module_paths():
                module_specs.append((f"model_files.{module_path.stem}", None))
        except Exception as exc:
            metadata["errors"].append(f"import model_files: {exc}")
    if not module_specs:
        for module_path in _iter_module_paths():
            module_specs.append((f"_business_model_files_{module_path.stem}", module_path))
    if not module_specs:
        metadata["status"] = "namespace_only"
        return metadata

    for module_name, module_path in module_specs:
        try:
            if module_path is None:
                module = importlib.import_module(module_name)
            else:
                module = _load_module_from_path(module_name, module_path)
            metadata["imported_modules"].append(module_name)
        except Exception as exc:
            metadata["errors"].append(f"import {module_name}: {exc}")
            continue
        _register_patch_hooks(module_name, module, metadata)
    if metadata["called_hooks"]:
        metadata["status"] = "applied"
    elif metadata["deferred_hooks"]:
        metadata["status"] = "loaded"
    elif metadata["imported_modules"]:
        metadata["status"] = "namespace_only"
    else:
        metadata["status"] = "namespace_only"
    return metadata


def _prepare_accuracy_run_hooks(script_path: Path, *, display_module_name: str, import_module_name: str):
    metadata = {
        "status": "missing",
        "imported_modules": [],
        "called_hooks": [],
        "deferred_hooks": [],
        "errors": [],
    }
    if not script_path.exists():
        return metadata
    if str(ADAPT_DIR) not in sys.path:
        sys.path.insert(0, str(ADAPT_DIR))
    try:
        module = _load_module_from_path(import_module_name, script_path)
        metadata["imported_modules"].append(display_module_name)
    except Exception as exc:
        metadata["errors"].append(f"import {display_module_name}: {exc}")
        return metadata
    _register_patch_hooks(display_module_name, module, metadata)
    if metadata["called_hooks"]:
        metadata["status"] = "applied"
    elif metadata["deferred_hooks"] or metadata["imported_modules"]:
        metadata["status"] = "loaded"
    return metadata


def _prepare_accuracy_run_perf_hooks():
    return _prepare_accuracy_run_hooks(
        ACCURACY_RUN_PERF_PATH,
        display_module_name="accuracy_run_perf",
        import_module_name="_business_perf_accuracy_run_perf",
    )


def _should_use_model_files_for_perf(config: dict[str, Any]) -> bool:
    if not MODEL_FILES_DIR.exists():
        return False
    return _as_bool(config.get("npu_perf_use_model_files"), default=True)


def _normalize_runtime_only_patch_metadata(config: dict[str, Any], scenario: str, metadata: dict[str, Any]) -> dict[str, Any]:
    if scenario != "npu_perf" or _should_use_model_files_for_perf(config):
        return metadata
    # runtime_only 明确要求 phase-4 不继承 model_files / accuracy_run_perf patch 证据，
    # 但仍允许继承 accuracy_run.py 里的 runtime compatibility 修复。
    imported_modules = [str(module_name) for module_name in metadata.get("imported_modules") or []]
    if any(module_name in {"model_files", "accuracy_run_perf"} or module_name.startswith("model_files.") for module_name in imported_modules):
        metadata["status"] = "disabled"
        metadata["imported_modules"] = []
        metadata["called_hooks"] = []
        metadata["deferred_hooks"] = []
    return metadata


def _apply_deferred_model_files_hooks(model, metadata: dict[str, Any]) -> dict[str, Any]:
    deferred_hooks = list(metadata.get("deferred_hooks") or [])
    if not deferred_hooks:
        return metadata
    for hook_ref, hook in deferred_hooks:
        try:
            hook(model)
            metadata["called_hooks"].append(hook_ref)
        except Exception as exc:
            metadata["errors"].append(f"{hook_ref}: {exc}")
    metadata["deferred_hooks"] = []
    if metadata["called_hooks"]:
        metadata["status"] = "applied"
    return metadata


def _ensure_qwen3_next_rmsnorm_epsilon_compat(metadata: dict[str, Any] | None = None) -> None:
    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm
    except Exception:
        return

    current_forward = getattr(Qwen3NextRMSNorm, "forward", None)
    if current_forward is None or getattr(current_forward, "_slai_qwen3_next_eps_compat", False):
        return

    def _compat_forward(self, *args, **kwargs):
        if not hasattr(self, "variance_epsilon") and hasattr(self, "eps"):
            self.variance_epsilon = self.eps
        return current_forward(self, *args, **kwargs)

    _compat_forward._slai_qwen3_next_eps_compat = True  # type: ignore[attr-defined]
    Qwen3NextRMSNorm.forward = _compat_forward
    if isinstance(metadata, dict):
        metadata.setdefault("compat_patches", [])
        if "qwen3_next_rmsnorm_eps_alias" not in metadata["compat_patches"]:
            metadata["compat_patches"].append("qwen3_next_rmsnorm_eps_alias")


def _has_local_input_assets(input_kind: str) -> bool:
    if input_kind == "tokenizer":
        candidates = (
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
            "vocab.txt",
            "merges.txt",
            "sentencepiece.bpe.model",
            "special_tokens_map.json",
        )
    elif input_kind == "image_processor":
        candidates = ("preprocessor_config.json", "processor_config.json")
    else:
        candidates = (
            "processor_config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
            "vocab.txt",
            "merges.txt",
            "special_tokens_map.json",
        )
    return any((MODEL_FILES_DIR / candidate).exists() for candidate in candidates)


def _model_files_is_full_model_repo() -> bool:
    if not MODEL_FILES_DIR.exists():
        return False
    if not (MODEL_FILES_DIR / "config.json").exists():
        return False
    return _snapshot_has_model_assets(MODEL_FILES_DIR)


def _model_files_can_be_used_as_model_source(metadata: dict[str, Any]) -> bool:
    if not _model_files_is_full_model_repo():
        return False
    errors = [str(item or "").strip() for item in list(metadata.get("errors") or [])]
    blocking_prefixes = ("import model_files.configuration_", "import model_files.modeling_")
    if any(error.startswith(blocking_prefixes) for error in errors):
        return False
    return True


def _load_tokenizer_from_local_vocab_fallback(tokenizer_source: str):
    source_path = Path(str(tokenizer_source))
    if not source_path.is_dir():
        return None
    vocab_path = source_path / "vocab.txt"
    sentencepiece_path = None
    for candidate_name in ("tokenizer.model", "spiece.model", "sentencepiece.bpe.model"):
        candidate_path = source_path / candidate_name
        if candidate_path.exists():
            sentencepiece_path = candidate_path
            break

    tokenizer_class = ""
    tokenizer_kwargs: dict[str, Any] = {}
    tokenizer_config_path = source_path / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        try:
            tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        except Exception:
            tokenizer_config = {}
        tokenizer_class = str(tokenizer_config.get("tokenizer_class") or "").strip()
        for key in ("do_lower_case", "strip_accents", "tokenize_chinese_chars"):
            value = tokenizer_config.get(key)
            if value is not None:
                tokenizer_kwargs[key] = value

    if sentencepiece_path is not None and tokenizer_class in {"", "LlamaTokenizer", "LlamaTokenizerFast", "YayiTokenizer"}:
        from transformers import LlamaTokenizer

        for key in ("bos_token", "eos_token", "unk_token", "pad_token", "add_bos_token", "add_eos_token"):
            value = tokenizer_config.get(key) if tokenizer_config_path.exists() else None
            if isinstance(value, dict):
                value = value.get("content")
            if value is not None:
                tokenizer_kwargs[key] = value
        return LlamaTokenizer(str(sentencepiece_path), **tokenizer_kwargs)

    if not vocab_path.exists():
        return None

    if tokenizer_class not in {"", "BertTokenizer", "BertTokenizerFast"}:
        return None

    from transformers import BertTokenizer

    return BertTokenizer(str(vocab_path), **tokenizer_kwargs)


def _iter_accuracy_run_fallback_tokenizer_ids() -> list[str]:
    fallback_ids: list[str] = []
    seen: set[str] = set()

    for candidate_path in (ACCURACY_RUN_PATH, ACCURACY_RUN_PERF_PATH):
        if not candidate_path.exists():
            continue
        try:
            module_ast = ast.parse(candidate_path.read_text(encoding="utf-8"), filename=str(candidate_path))
        except Exception:
            continue
        for node in ast.walk(module_ast):
            value_node = None
            if isinstance(node, ast.Assign):
                if any(isinstance(target, ast.Name) and target.id == "FALLBACK_TOKENIZER_ID" for target in node.targets):
                    value_node = node.value
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id == "FALLBACK_TOKENIZER_ID":
                    value_node = node.value
            if not isinstance(value_node, ast.Constant) or not isinstance(value_node.value, str):
                continue
            fallback_id = value_node.value.strip()
            if fallback_id and fallback_id not in seen:
                seen.add(fallback_id)
                fallback_ids.append(fallback_id)
    return fallback_ids


def _load_tokenizer_from_accuracy_run_source_fallback(trust_remote_code: bool):
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None

    seen_sources: set[str] = set()
    for fallback_id in _iter_accuracy_run_fallback_tokenizer_ids():
        local_snapshot_source = _resolve_local_snapshot_source(fallback_id, input_kind="tokenizer")
        for candidate_source in (local_snapshot_source, fallback_id):
            candidate_text = str(candidate_source or "").strip()
            if not candidate_text or candidate_text in seen_sources:
                continue
            seen_sources.add(candidate_text)
            try:
                return AutoTokenizer.from_pretrained(
                    candidate_text,
                    cache_dir=str(CACHE_DIR),
                    trust_remote_code=trust_remote_code,
                )
            except Exception:
                continue
    return None


def _load_custom_tokenizer_from_accuracy_run_fallback(tokenizer_source: str):
    source_path = Path(str(tokenizer_source))
    if not source_path.is_dir():
        return None

    config_path = source_path / "config.json"
    tokenizer_config_path = source_path / "tokenizer_config.json"
    if not config_path.exists() or not tokenizer_config_path.exists():
        return None

    try:
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        tokenizer_config_payload = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if "char_ords" not in tokenizer_config_payload:
        return None

    config = SimpleNamespace(**config_payload)
    tokenizer_config = getattr(config, "tokenizer_config", None)
    if not isinstance(tokenizer_config, Mapping):
        tokenizer_config = tokenizer_config_payload
        setattr(config, "tokenizer_config", tokenizer_config_payload)

    for candidate_name, import_name in (
        ("accuracy_run.py", "_business_tokenizer_accuracy_run"),
        ("accuracy_run_perf.py", "_business_tokenizer_accuracy_run_perf"),
        ("demo.py", "_business_tokenizer_demo"),
    ):
        candidate_path = ADAPT_DIR / candidate_name
        if not candidate_path.exists():
            continue
        try:
            module = _load_module_from_path(import_name, candidate_path)
        except Exception:
            continue
        tokenizer_cls = getattr(module, "DNATokenizer", None)
        if tokenizer_cls is None:
            continue
        try:
            return tokenizer_cls(
                char_ords=list(tokenizer_config_payload.get("char_ords") or tokenizer_config.get("char_ords") or [68, 78, 65, 84, 71, 67]),
                vocab_size=int(getattr(config, "vocab_size", 512) or 512),
                bos_token=int(getattr(config, "bos_token_id", 11) or 11),
                eos_token=int(getattr(config, "eos_token_id", 10) or 10),
                pad_token=int(getattr(config, "pad_token_id", 9) or 9),
                sep_token=int(getattr(config, "sep_token_id", 7) or 7),
            )
        except Exception:
            continue
    return None


def _load_image_processor_from_local_legacy_fallback(image_source: str, *, trust_remote_code: bool):
    source_path = Path(str(image_source))
    if not source_path.is_dir():
        return None

    preprocessor_payload: dict[str, Any] = {}
    config_payload: dict[str, Any] = {}
    for candidate_name, target in (
        ("preprocessor_config.json", preprocessor_payload),
        ("config.json", config_payload),
    ):
        candidate_path = source_path / candidate_name
        if not candidate_path.exists():
            continue
        try:
            target.update(json.loads(candidate_path.read_text(encoding="utf-8")))
        except Exception:
            continue

    model_type = str(config_payload.get("model_type") or "").strip().lower()
    architectures = [str(item or "").strip().lower() for item in list(config_payload.get("architectures") or [])]
    feature_extractor_type = str(preprocessor_payload.get("feature_extractor_type") or "").strip().lower()

    # Older repos may still expose only legacy feature extractor metadata while
    # the runtime already moved to image processors. RMBG-1.4 is one example.
    if "segformer" in model_type or any("bria" in item and "rmbg" in item for item in architectures) or feature_extractor_type == "segformerfeatureextractor":
        from transformers import SegformerImageProcessor

        return SegformerImageProcessor.from_pretrained(
            str(source_path),
            cache_dir=str(CACHE_DIR),
            trust_remote_code=trust_remote_code,
        )
    return None


def _is_optional_asr_lm_dependency_error(exc: Exception) -> bool:
    message = str(exc or "").strip().lower()
    optional_dependency_markers = (
        "wav2vec2processorwithlm",
        "pyctcdecode",
        "kenlm",
        "beamsearchdecoderctc",
    )
    return any(marker in message for marker in optional_dependency_markers)


def _cleanup_asr_lm_sidecar_files(processor_source: str):
    source_path = Path(str(processor_source))
    if not source_path.is_dir():
        return

    preprocessor_path = source_path / "preprocessor_config.json"
    if not preprocessor_path.exists():
        return
    try:
        preprocessor_payload = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    except Exception:
        return

    processor_class = str(preprocessor_payload.get("processor_class") or "").strip()
    if processor_class != "Wav2Vec2ProcessorWithLM":
        return

    language_model_dir = source_path / "language_model"
    if not language_model_dir.is_dir():
        return

    removed_sidecars: list[str] = []
    for sidecar_path in sorted(language_model_dir.glob("*.rclonelink")):
        try:
            sidecar_path.unlink()
        except OSError:
            continue
        removed_sidecars.append(sidecar_path.name)
    if removed_sidecars:
        print(f"[business][asr] removed LM sidecars from {language_model_dir}: {removed_sidecars}")


def _load_asr_processor_without_optional_lm(processor_source: str, *, trust_remote_code: bool):
    source_path = Path(str(processor_source))
    if not source_path.is_dir():
        return None

    preprocessor_path = source_path / "preprocessor_config.json"
    if not preprocessor_path.exists():
        return None
    try:
        preprocessor_payload = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    processor_class = str(preprocessor_payload.get("processor_class") or "").strip()
    if processor_class != "Wav2Vec2ProcessorWithLM":
        return None

    from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

    load_kwargs = {"cache_dir": str(CACHE_DIR)}
    if trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    try:
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(str(source_path), **load_kwargs)
    except TypeError:
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(str(source_path), cache_dir=str(CACHE_DIR))

    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(source_path), **load_kwargs)
    except TypeError:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(source_path), cache_dir=str(CACHE_DIR))

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    setattr(processor, "_business_optional_lm_decoder_disabled", True)
    return processor


def _iter_snapshot_asset_dirs(snapshot_dir: Path, *, max_depth: int = 4):
    if not snapshot_dir.exists():
        return
    yield snapshot_dir
    nested_dirs: list[Path] = []
    try:
        for candidate in snapshot_dir.rglob("*"):
            if not candidate.is_dir():
                continue
            try:
                depth = len(candidate.relative_to(snapshot_dir).parts)
            except ValueError:
                continue
            if depth > max_depth:
                continue
            nested_dirs.append(candidate)
    except Exception:
        return
    for candidate in sorted(nested_dirs, key=lambda path: (len(path.relative_to(snapshot_dir).parts), str(path))):
        yield candidate


def _snapshot_dir_has_model_assets(snapshot_dir: Path) -> bool:
    weight_patterns = (
        "model.safetensors",
        "model-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model-*.bin",
        "last.ckpt",
        "*.ckpt",
        "tf_model.h5",
        "flax_model.msgpack",
    )
    for pattern in weight_patterns:
        for candidate in snapshot_dir.glob(pattern):
            if candidate.exists():
                return True
    return False


def _snapshot_has_model_assets(snapshot_dir: Path) -> bool:
    return any(_snapshot_dir_has_model_assets(candidate_dir) for candidate_dir in _iter_snapshot_asset_dirs(snapshot_dir))


def _install_bmfm_targets_pickle_stubs() -> None:
    if "bmfm_targets" in sys.modules:
        return

    class PickleCompat:
        def __init__(self, *args, **kwargs):
            if args:
                self._pickle_args = args
            if kwargs:
                self.__dict__.update(kwargs)

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
                return
            self._pickle_state = state

        def __getstate__(self):
            return dict(self.__dict__)

    def _ensure_module(name: str) -> ModuleType:
        module = sys.modules.get(name)
        if isinstance(module, ModuleType):
            return module
        module = ModuleType(name)
        sys.modules[name] = module
        return module

    root_module = _ensure_module("bmfm_targets")
    root_module.__path__ = []  # type: ignore[attr-defined]
    config_module = _ensure_module("bmfm_targets.config")
    config_module.__path__ = []  # type: ignore[attr-defined]
    compat_module = _ensure_module("bmfm_targets.config._compat")
    model_config_module = _ensure_module("bmfm_targets.config.model_config")
    tokenization_module = _ensure_module("bmfm_targets.config.tokenization_config")
    training_module = _ensure_module("bmfm_targets.config.training_config")

    class SCBertConfig(PickleCompat):
        pass

    class FieldInfo(PickleCompat):
        pass

    class LabelColumnInfo(PickleCompat):
        pass

    class TrainerConfig(PickleCompat):
        pass

    compat_module.PickleCompat = PickleCompat
    model_config_module.SCBertConfig = SCBertConfig
    tokenization_module.FieldInfo = FieldInfo
    tokenization_module.LabelColumnInfo = LabelColumnInfo
    training_module.TrainerConfig = TrainerConfig
    config_module.PickleCompat = PickleCompat
    config_module.SCBertConfig = SCBertConfig
    config_module.FieldInfo = FieldInfo
    config_module.LabelColumnInfo = LabelColumnInfo
    config_module.TrainerConfig = TrainerConfig
    config_module._compat = compat_module
    config_module.model_config = model_config_module
    config_module.tokenization_config = tokenization_module
    config_module.training_config = training_module
    root_module.config = config_module


@contextmanager
def _temporary_sys_path_entries(*entries: Path):
    added_entries: list[str] = []
    try:
        for entry in entries:
            entry_text = str(entry)
            if not entry_text or entry_text in sys.path:
                continue
            sys.path.insert(0, entry_text)
            added_entries.append(entry_text)
        if not (MODEL_FILES_DIR / "bmfm_targets").exists():
            _install_bmfm_targets_pickle_stubs()
        yield
    finally:
        for entry_text in reversed(added_entries):
            try:
                sys.path.remove(entry_text)
            except ValueError:
                continue


@contextmanager
def _temporary_env_overrides(overrides: Mapping[str, Any]):
    previous_values: dict[str, str | None] = {}
    try:
        for key, value in overrides.items():
            if not key:
                continue
            previous_values[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, previous_value in previous_values.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value


def _iter_local_custom_module_entries(*sources: str | Path | None):
    seen: set[str] = set()
    for source in sources:
        if source is None:
            continue
        source_path = Path(str(source))
        candidate_dirs: list[Path] = []
        if source_path.is_dir():
            candidate_dirs.append(source_path)
            custom_modules_dir = source_path / "custom_modules"
            if custom_modules_dir.is_dir():
                candidate_dirs.append(custom_modules_dir)
        models_custom_modules_dir = CACHE_DIR / "custom_modules"
        if models_custom_modules_dir.is_dir():
            candidate_dirs.append(models_custom_modules_dir)
        if MODEL_FILES_DIR.is_dir():
            candidate_dirs.append(MODEL_FILES_DIR)
        candidate_dirs.append(ADAPT_DIR)
        for candidate_dir in candidate_dirs:
            candidate_text = str(candidate_dir)
            if candidate_text in seen or not candidate_dir.exists():
                continue
            seen.add(candidate_text)
            yield candidate_dir


def _load_state_dict_from_snapshot(snapshot_dir: Path) -> dict[str, Any]:
    def _load_safetensors_state_dict(checkpoint_path: Path) -> dict[str, Any]:
        from safetensors import safe_open

        state_dict: dict[str, Any] = {}
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            has_bnb_int8_scale = any(key.endswith(".SCB") for key in keys)
            for key in keys:
                if key.endswith(".SCB") or key.endswith(".weight_format"):
                    continue
                value = handle.get_tensor(key)
                if has_bnb_int8_scale and getattr(value, "dtype", None) == torch.int8 and key.endswith(".weight"):
                    scb_key = key.replace(".weight", ".SCB")
                    if scb_key in keys:
                        scb = handle.get_tensor(scb_key).float()
                        value = (value.float() * scb.view(-1, 1) * (1.0 / 127.0)).half()
                state_dict[key] = value
        return state_dict

    if snapshot_dir.is_file():
        checkpoint_path = snapshot_dir
        if checkpoint_path.suffix == ".safetensors":
            return _load_safetensors_state_dict(checkpoint_path)
        if checkpoint_path.suffix == ".bin":
            checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
            checkpoint_state_dict = checkpoint_payload.get("state_dict") if isinstance(checkpoint_payload, dict) and isinstance(checkpoint_payload.get("state_dict"), dict) else checkpoint_payload
            if not isinstance(checkpoint_state_dict, dict):
                raise RuntimeError(f"checkpoint 缺少 state_dict: {checkpoint_path}")
            return checkpoint_state_dict
        if checkpoint_path.suffix == ".ckpt":
            with _temporary_sys_path_entries(ADAPT_DIR, MODEL_FILES_DIR):
                checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint_payload, dict) and isinstance(checkpoint_payload.get("state_dict"), dict):
                checkpoint_state_dict = checkpoint_payload["state_dict"]
            elif isinstance(checkpoint_payload, dict) and isinstance(checkpoint_payload.get("model"), dict):
                checkpoint_state_dict = checkpoint_payload["model"]
            elif isinstance(checkpoint_payload, dict):
                checkpoint_state_dict = checkpoint_payload
            else:
                raise RuntimeError(f"checkpoint 缺少 state_dict: {checkpoint_path}")
            state_dict = {}
            for key, value in checkpoint_state_dict.items():
                normalized_key = str(key)
                if normalized_key.startswith("model."):
                    normalized_key = normalized_key[6:]
                state_dict[normalized_key] = value
            return state_dict
        raise RuntimeError(f"不支持的 checkpoint 文件: {checkpoint_path}")

    safetensors_paths = sorted(snapshot_dir.glob("model-*.safetensors"))
    if not safetensors_paths:
        direct_safetensors = snapshot_dir / "model.safetensors"
        if direct_safetensors.exists():
            safetensors_paths = [direct_safetensors]
    if safetensors_paths:
        state_dict: dict[str, Any] = {}
        for checkpoint_path in safetensors_paths:
            shard_state_dict = _load_safetensors_state_dict(checkpoint_path)
            state_dict.update(shard_state_dict)
        return state_dict

    bin_paths = sorted(snapshot_dir.glob("pytorch_model-*.bin"))
    if not bin_paths:
        direct_bin = snapshot_dir / "pytorch_model.bin"
        if direct_bin.exists():
            bin_paths = [direct_bin]
    if bin_paths:
        state_dict = {}
        for checkpoint_path in bin_paths:
            shard_payload = torch.load(checkpoint_path, map_location="cpu")
            shard_state_dict = shard_payload.get("state_dict") if isinstance(shard_payload, dict) and isinstance(shard_payload.get("state_dict"), dict) else shard_payload
            if not isinstance(shard_state_dict, dict):
                raise RuntimeError(f"checkpoint 缺少 state_dict: {checkpoint_path}")
            state_dict.update(shard_state_dict)
        return state_dict

    ckpt_paths = sorted(snapshot_dir.glob("*.ckpt"))
    if ckpt_paths:
        state_dict = {}
        for checkpoint_path in ckpt_paths:
            with _temporary_sys_path_entries(ADAPT_DIR, MODEL_FILES_DIR):
                checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint_payload, dict) and isinstance(checkpoint_payload.get("state_dict"), dict):
                shard_state_dict = checkpoint_payload["state_dict"]
            elif isinstance(checkpoint_payload, dict) and isinstance(checkpoint_payload.get("model"), dict):
                shard_state_dict = checkpoint_payload["model"]
            elif isinstance(checkpoint_payload, dict):
                shard_state_dict = checkpoint_payload
            else:
                raise RuntimeError(f"checkpoint 缺少 state_dict: {checkpoint_path}")
            for key, value in shard_state_dict.items():
                normalized_key = str(key)
                if normalized_key.startswith("model."):
                    normalized_key = normalized_key[6:]
                state_dict[normalized_key] = value
        return state_dict

    raise RuntimeError(f"未在 {snapshot_dir} 找到可加载的权重文件")


def _should_retry_with_manual_state_dict_load(exc: Exception, source: str) -> bool:
    source_path = Path(str(source))
    if not source_path.is_dir():
        return False
    if any(source_path.glob("*.ckpt")):
        return True
    error_text = str(exc or "")
    if "all_tied_weights_keys" in error_text:
        return True
    if "meta device context manager" in error_text:
        return True
    if "Only Tensors of floating point and complex dtype can require gradients" in error_text:
        return True
    if "requires the latest version of bitsandbytes" in error_text:
        return True
    return False


def _snapshot_dir_has_input_assets(snapshot_dir: Path, input_kind: str) -> bool:
    if input_kind == "tokenizer":
        candidates = (
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
            "vocab.txt",
            "merges.txt",
            "sentencepiece.bpe.model",
            "spiece.model",
            "special_tokens_map.json",
        )
    elif input_kind == "image_processor":
        candidates = ("preprocessor_config.json", "processor_config.json")
    else:
        candidates = (
            "processor_config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
            "vocab.txt",
            "merges.txt",
            "special_tokens_map.json",
        )
    return any((snapshot_dir / candidate).exists() for candidate in candidates)


def _snapshot_has_input_assets(snapshot_dir: Path, input_kind: str) -> bool:
    return any(_snapshot_dir_has_input_assets(candidate_dir, input_kind) for candidate_dir in _iter_snapshot_asset_dirs(snapshot_dir))


def _load_transformers_components(model_type: str):
    _apply_transformers_pytorch_utils_compatibility_shims()
    import transformers as transformers_module

    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModel,
        AutoModelForCTC,
        AutoModelForCausalLM,
        AutoModelForDepthEstimation,
        AutoModelForImageClassification,
        AutoModelForImageSegmentation,
        AutoModelForMaskedLM,
        AutoModelForObjectDetection,
        AutoModelForPreTraining,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForSpeechSeq2Seq,
        AutoModelForTokenClassification,
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        AutoTokenizer,
        VisionEncoderDecoderModel,
    )

    VitPoseForPoseEstimation = getattr(transformers_module, "VitPoseForPoseEstimation", None)
    VitMatteForImageMatting = getattr(transformers_module, "VitMatteForImageMatting", None)

    canonical_model_type = "token_classification" if model_type == "biomedical_token_classification" else model_type

    model_class_map = {
        "causal_lm": ("tokenizer", AutoModelForCausalLM),
        "seq2seq": ("tokenizer", AutoModelForSeq2SeqLM),
        "tts": ("processor", AutoModelForCausalLM),
        "asr": ("processor", (AutoModelForSpeechSeq2Seq, AutoModelForCTC)),
        "classification": ("tokenizer", AutoModelForSequenceClassification),
        "masked_lm": ("tokenizer", AutoModelForMaskedLM),
        "question_answering": ("tokenizer", AutoModelForQuestionAnswering),
        "token_classification": ("tokenizer", AutoModelForTokenClassification),
        "discriminator": ("tokenizer", AutoModelForPreTraining),
        "reranker": ("tokenizer", AutoModelForSequenceClassification),
        "vision_classification": ("image_processor", AutoModelForImageClassification),
        "vision_embedding": ("image_processor", (AutoModel, AutoModelForDepthEstimation)),
        "vision_text_ocr": ("processor", (VisionEncoderDecoderModel, AutoModelForCausalLM)),
        "embedding": ("tokenizer", AutoModel),
        "audio_embedding": ("processor", AutoModel),
        "vision_detection": ("processor", (AutoModelForObjectDetection, AutoModelForZeroShotObjectDetection)),
        "vision_keypoint_detection": ("image_processor", VitPoseForPoseEstimation),
        "image_matting": ("image_processor", (VitMatteForImageMatting, AutoModelForImageSegmentation)),
        "semantic_segmentation": ("image_processor", AutoModelForImageSegmentation),
    }
    if canonical_model_type not in model_class_map:
        raise RuntimeError(
            f"通用 business_model_eval.py 目前仅支持 causal_lm / seq2seq / tts / asr / classification / masked_lm / question_answering / token_classification / biomedical_token_classification / discriminator / reranker / vision_classification / vision_embedding / vision_text_ocr / embedding / audio_embedding / vision_detection / vision_keypoint_detection / image_matting / semantic_segmentation，当前 model_type={model_type} 需要在 adaptation 目录自行定制业务测评代码。"
        )
    input_kind, model_cls = model_class_map[canonical_model_type]
    if isinstance(model_cls, tuple):
        model_cls = tuple(candidate for candidate in model_cls if candidate is not None)
        if len(model_cls) == 1:
            model_cls = model_cls[0]
    if model_cls is None or (isinstance(model_cls, tuple) and not model_cls):
        raise RuntimeError(f"当前 transformers 版本缺少 {model_type} 所需模型类，请升级 adaptation 目录的 transformers 依赖或为该模型提供自定义 business_model_eval.py。")
    input_loader_cls = AutoTokenizer if input_kind == "tokenizer" else AutoImageProcessor if input_kind == "image_processor" else AutoProcessor
    return AutoConfig, input_loader_cls, model_cls, input_kind


def _looks_like_legacy_olmo_checkpoint(model_config, *, model_id: str) -> bool:
    model_type = str(getattr(model_config, "model_type", "") or "").strip().lower()
    if model_type != "olmo":
        return False
    architectures = [str(item).strip().lower() for item in list(getattr(model_config, "architectures", []) or []) if str(item).strip()]
    if any("olmoforcausallm" in arch for arch in architectures):
        return True
    model_id_text = str(model_id or "").strip().lower()
    if "olmo" in model_id_text:
        return True
    return any(hasattr(model_config, attr) for attr in ("d_model", "n_layers", "n_heads", "activation_type"))


def _resolve_legacy_causal_lm_model_class(model_cls, model_config, *, model_id: str):
    if not _looks_like_legacy_olmo_checkpoint(model_config, model_id=model_id):
        return model_cls, ""
    for index, helper_path in enumerate(LEGACY_OLMO_HELPER_PATHS):
        if not helper_path.is_file():
            continue
        try:
            helper_module = _load_module_from_path(f"_business_legacy_olmo_helper_{index}", helper_path)
        except Exception:
            continue
        helper_model_cls = getattr(helper_module, "OLMoForCausalLM", None)
        if helper_model_cls is not None:
            return helper_model_cls, f"local_helper:{helper_path}"
    _apply_torch_module_tied_weights_compatibility_shim()
    _apply_hf_olmo_compatibility_shims()
    try:
        from hf_olmo import OLMoForCausalLM
    except Exception as exc:
        raise RuntimeError(
            "检测到 legacy OLMo checkpoint，但当前业务测评环境既没有本地 lightweight OLMo helper，也缺少 ai2-olmo 运行时。请先补齐 business_benchmark/templates/modeling_olmo_v1.py，或让 business_benchmark_manager 为该 adaptation 安装 ai2-olmo。"
        ) from exc
    return OLMoForCausalLM, "hf_olmo"


def _normalize_model_family_token(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").strip().lower())


def _load_config_payload_from_source(source: str) -> dict[str, Any]:
    source_path = Path(str(source))
    config_path = source_path / "config.json"
    if not config_path.is_file():
        return {}
    try:
        payload = json.loads(config_path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_gliner_config_payload_from_source(source: str) -> dict[str, Any]:
    source_path = Path(str(source))
    config_path = source_path / "gliner_config.json"
    if not config_path.is_file():
        return {}
    try:
        payload = json.loads(config_path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _looks_like_span_marker_checkpoint(config_payload: Mapping[str, Any] | None = None, *, model_id: str = "") -> bool:
    payload = config_payload if isinstance(config_payload, Mapping) else {}
    payload_model_type = _normalize_model_family_token(payload.get("model_type"))
    payload_architectures = [_normalize_model_family_token(item) for item in list(payload.get("architectures") or []) if str(item).strip()]
    model_id_text = _normalize_model_family_token(model_id)
    return payload_model_type == "spanmarker" or "spanmarkermodel" in payload_architectures or "spanmarker" in model_id_text


def _looks_like_gliner_checkpoint(source: str | None = None, *, model_id: str = "", model_backend: str = "") -> bool:
    normalized_backend = _normalize_model_family_token(model_backend)
    if normalized_backend == "gliner":
        return True
    model_id_text = _normalize_model_family_token(model_id)
    if "gliner" in model_id_text:
        return True
    source_text = str(source or "").strip()
    if not source_text:
        return False
    source_path = Path(source_text)
    if source_path.exists() and (source_path / "gliner_config.json").is_file():
        return True
    return "gliner" in _normalize_model_family_token(source_text)


def _resolve_local_snapshot_source(model_id: str, *, require_model_assets: bool = False, input_kind: str | None = None) -> str | None:
    if not model_id or "/" not in model_id:
        return None
    org, name = model_id.split("/", 1)
    cache_dir = CACHE_DIR / f"models--{org.replace('/', '--')}--{name.replace('/', '--')}"
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    snapshot_candidates = sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), reverse=True)
    if not snapshot_candidates:
        return None
    for candidate in snapshot_candidates:
        for asset_dir in _iter_snapshot_asset_dirs(candidate):
            has_input_assets = bool(input_kind and _snapshot_dir_has_input_assets(asset_dir, input_kind))
            has_primary_config = (asset_dir / "config.json").exists() or (asset_dir / "gliner_config.json").exists()
            if not has_primary_config and (require_model_assets or not has_input_assets):
                continue
            if require_model_assets and not _snapshot_dir_has_model_assets(asset_dir):
                continue
            if input_kind and not has_input_assets:
                continue
            return str(asset_dir)
    return None


def _resolve_cached_hf_snapshot_source(model_id: str) -> str | None:
    model_id_text = str(model_id or "").strip()
    if not model_id_text:
        return None
    normalized = model_id_text.replace("/", "--")
    candidate_roots = [CACHE_DIR / f"models--{normalized}"]
    if "/" not in model_id_text:
        candidate_roots.extend(sorted(CACHE_DIR.glob(f"models--*--{normalized}")))
    seen: set[str] = set()
    for candidate_root in candidate_roots:
        candidate_key = str(candidate_root)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        snapshots_dir = candidate_root / "snapshots"
        if not snapshots_dir.is_dir():
            continue
        for snapshot_dir in sorted(snapshots_dir.iterdir()):
            if snapshot_dir.is_dir():
                return str(snapshot_dir)
    return None


def _resolve_configured_source(source_value: Any, *, require_model_assets: bool = False, input_kind: str | None = None) -> tuple[str | None, str | None]:
    source_text = str(source_value or "").strip()
    if not source_text:
        return None, None
    local_snapshot_source = _resolve_local_snapshot_source(source_text, require_model_assets=require_model_assets, input_kind=input_kind)
    if local_snapshot_source:
        return local_snapshot_source, "config_override_local_snapshot"
    return source_text, "config_override"


def _resolve_local_adapter_snapshot_source(source: str) -> str | None:
    source_path = Path(str(source))
    if source_path.exists():
        if source_path.is_dir() and (source_path / "adapter_config.json").is_file():
            return str(source_path)
        return None
    source_text = str(source or "").strip()
    if not source_text or "/" not in source_text:
        return None
    org, name = source_text.split("/", 1)
    cache_dir = CACHE_DIR / f"models--{org.replace('/', '--')}--{name.replace('/', '--')}"
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    snapshot_candidates = sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), reverse=True)
    for candidate in snapshot_candidates:
        for asset_dir in _iter_snapshot_asset_dirs(candidate):
            if not (asset_dir / "adapter_config.json").is_file():
                continue
            return str(asset_dir)
    return None


def _extract_base_model_id_from_local_runtime_scripts() -> str:
    pattern = re.compile(r"(?m)^\s*BASE_MODEL_ID\s*=\s*['\"]([^'\"]+)['\"]")
    for candidate_path in (ACCURACY_RUN_PERF_PATH, ACCURACY_RUN_PATH, ADAPT_DIR / "demo.py"):
        if not candidate_path.is_file():
            continue
        try:
            content = candidate_path.read_text(encoding="utf-8")
        except Exception:
            continue
        match = pattern.search(content)
        if match is not None:
            base_model_id = str(match.group(1) or "").strip()
            if base_model_id:
                return base_model_id
    return ""


def _load_peft_config_from_source(source: str):
    source_text = str(source or "").strip()
    if not source_text:
        return None
    try:
        from peft import PeftConfig
    except Exception:
        return None
    for candidate in dict.fromkeys(
        item
        for item in (
            source_text,
            _resolve_local_adapter_snapshot_source(source_text),
        )
        if str(item or "").strip()
    ):
        try:
            return PeftConfig.from_pretrained(str(candidate), cache_dir=str(CACHE_DIR))
        except TypeError:
            try:
                return PeftConfig.from_pretrained(str(candidate))
            except Exception:
                continue
        except Exception:
            continue
    return None


def _resolve_peft_adapter_spec(config: dict[str, Any], model_id: str, model_source: str) -> dict[str, Any] | None:
    adapter_candidates = []
    for candidate in (
        model_source,
        model_id,
        _resolve_local_adapter_snapshot_source(model_source),
        _resolve_local_adapter_snapshot_source(model_id),
    ):
        candidate_text = str(candidate or "").strip()
        if candidate_text and candidate_text not in adapter_candidates:
            adapter_candidates.append(candidate_text)

    adapter_source = ""
    peft_config = None
    for candidate in adapter_candidates:
        peft_config = _load_peft_config_from_source(candidate)
        if peft_config is not None:
            adapter_source = candidate
            break
    if peft_config is None or not adapter_source:
        return None

    explicit_base_model_id = str(config.get("base_model_id") or "").strip()
    runtime_base_model_id = _extract_base_model_id_from_local_runtime_scripts()
    peft_base_model_id = str(getattr(peft_config, "base_model_name_or_path", "") or "").strip()
    base_model_id = ""
    base_model_source = ""
    base_model_source_kind = ""
    for candidate in (explicit_base_model_id, runtime_base_model_id, peft_base_model_id):
        candidate_text = str(candidate or "").strip()
        if not candidate_text:
            continue
        resolved_source, resolved_kind = _resolve_configured_source(candidate_text, require_model_assets=True)
        if not resolved_source:
            continue
        base_model_id = candidate_text
        base_model_source = resolved_source
        base_model_source_kind = resolved_kind or ("local_snapshot" if Path(str(resolved_source)).exists() else "hub")
        break
    if not base_model_source:
        return None

    adapter_source_kind = "local_snapshot" if Path(str(adapter_source)).exists() else "hub"
    return {
        "adapter_source": adapter_source,
        "adapter_source_kind": adapter_source_kind,
        "base_model_id": base_model_id,
        "base_model_source": base_model_source,
        "base_model_source_kind": base_model_source_kind,
        "peft_type": str(getattr(peft_config, "peft_type", "") or "").strip(),
        "raw_base_model_name_or_path": peft_base_model_id,
    }


def _resolve_model_sources(model_id: str, config: dict[str, Any], scenario: str, input_kind: str) -> tuple[str, str, dict]:
    explicit_base_model_id = str(config.get("base_model_id") or "").strip()
    # base_model_id is commonly used for tokenizer / processor inheritance (for example TrOCR),
    # but the runnable weights still come from model_id unless an explicit model_source override is set.
    explicit_model_source_value = config.get("model_source_override") or config.get("model_source")
    explicit_input_source_value = config.get("tokenizer_source_override") or config.get("input_source_override") or config.get("input_source") or (explicit_base_model_id if input_kind == "tokenizer" else "")
    configured_model_source, configured_model_source_kind = _resolve_configured_source(
        explicit_model_source_value,
        require_model_assets=True,
    )
    configured_input_source, configured_input_source_kind = _resolve_configured_source(
        explicit_input_source_value,
        input_kind=input_kind,
    )
    local_model_snapshot_source = _resolve_local_snapshot_source(model_id, require_model_assets=True)
    local_input_snapshot_source = _resolve_local_snapshot_source(model_id, input_kind=input_kind)
    model_source = configured_model_source or local_model_snapshot_source or model_id
    input_source = configured_input_source or local_input_snapshot_source or local_model_snapshot_source or configured_model_source or model_id
    patch_metadata = {
        "status": "disabled",
        "imported_modules": [],
        "called_hooks": [],
        "errors": [],
    }
    model_source_kind = configured_model_source_kind or ("local_snapshot" if local_model_snapshot_source or local_input_snapshot_source else "hub")
    input_source_kind = configured_input_source_kind or ("local_snapshot" if local_input_snapshot_source or local_model_snapshot_source else "hub")
    context = {
        "model_source": model_source,
        "model_source_kind": model_source_kind,
        "input_source": input_source,
        "input_source_kind": input_source_kind,
        "used_model_files": False,
        "patch_load_status": "disabled",
        "patch_modules": [],
        "patch_hooks": [],
        "patch_errors": [],
        "_patch_metadata": patch_metadata,
    }
    if scenario == "npu_perf" and _should_use_model_files_for_perf(config):
        patch_metadata = _prepare_model_files_for_perf()
        if _model_files_can_be_used_as_model_source(patch_metadata):
            model_source = str(MODEL_FILES_DIR)
            context["model_source_kind"] = "model_files"
            context["used_model_files"] = True
        if _has_local_input_assets(input_kind):
            input_source = str(MODEL_FILES_DIR)
            context["input_source_kind"] = "model_files"
            context["used_model_files"] = True
        if str(patch_metadata.get("status") or "").lower() in {"loaded", "applied"}:
            context["used_model_files"] = True
    if scenario == "npu_perf" and _should_use_model_files_for_perf(config) and not list(patch_metadata.get("called_hooks") or []) and not list(patch_metadata.get("deferred_hooks") or []):
        patch_metadata = _merge_patch_metadata(patch_metadata, _prepare_accuracy_run_perf_hooks())
    if scenario == "npu_perf" and not _should_use_model_files_for_perf(config) and not list(patch_metadata.get("called_hooks") or []) and not list(patch_metadata.get("deferred_hooks") or []):
        patch_metadata = _merge_patch_metadata(
            patch_metadata,
            _prepare_accuracy_run_hooks(
                ACCURACY_RUN_PATH,
                display_module_name="accuracy_run",
                import_module_name="_business_perf_accuracy_run",
            ),
        )
    if scenario in {"npu_baseline", "cuda_baseline"} and not list(patch_metadata.get("called_hooks") or []) and not list(patch_metadata.get("deferred_hooks") or []):
        patch_metadata = _merge_patch_metadata(
            patch_metadata,
            _prepare_accuracy_run_hooks(
                ACCURACY_RUN_PATH,
                display_module_name="accuracy_run",
                import_module_name="_business_baseline_accuracy_run",
            ),
        )
    if scenario == "npu_perf" and str(patch_metadata.get("status") or "").lower() in {"loaded", "applied"}:
        _ensure_qwen3_next_rmsnorm_epsilon_compat(patch_metadata)
    if scenario == "npu_perf" and _should_use_model_files_for_perf(config):
        imported_modules = [str(module_name) for module_name in patch_metadata.get("imported_modules") or []]
        if any(module_name == "model_files" or module_name.startswith("model_files.") for module_name in imported_modules):
            context["used_model_files"] = True
        elif str(patch_metadata.get("status") or "").lower() in {"loaded", "applied"}:
            context["used_model_files"] = True
    patch_metadata = _normalize_runtime_only_patch_metadata(config, scenario, patch_metadata)
    context["model_source"] = model_source
    context["input_source"] = input_source
    context["patch_load_status"] = str(patch_metadata.get("status") or "disabled")
    context["patch_modules"] = list(patch_metadata.get("imported_modules") or [])
    context["patch_hooks"] = list(patch_metadata.get("called_hooks") or [])
    context["patch_errors"] = list(patch_metadata.get("errors") or [])
    context["_patch_metadata"] = patch_metadata
    return model_source, input_source, context


def _get_torch_dtype(config: dict):
    dtype_key = str(config.get("torch_dtype") or config.get("dtype") or "").strip().lower()
    if dtype_key == "fp16":
        return torch.float16
    if dtype_key == "bf16":
        return torch.bfloat16
    return "auto"


def _accelerate_available() -> bool:
    return importlib.util.find_spec("accelerate") is not None


def _get_first_parameter(model):
    if hasattr(model, "parameters"):
        try:
            return next(model.parameters())
        except Exception:
            pass
    inner_model = getattr(model, "model", None)
    if inner_model is not None and hasattr(inner_model, "parameters"):
        try:
            return next(inner_model.parameters())
        except Exception:
            pass
    return None


def _get_model_device(model) -> torch.device:
    parameter = _get_first_parameter(model)
    if parameter is not None:
        return parameter.device
    return torch.device("cpu")


def _format_model_dtype(model) -> str:
    param = _get_first_parameter(model)
    if param is None:
        return "fp32"
    dtype = getattr(param, "dtype", None)
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    return "fp32"


def _get_model_floating_dtype(model) -> torch.dtype | None:
    param = _get_first_parameter(model)
    if param is None:
        return None
    try:
        if torch.is_floating_point(param):
            return param.dtype
    except Exception:
        return None
    return None


def _move_runtime_tensor(value, device: torch.device, *, dtype: torch.dtype | None = None):
    if not hasattr(value, "to"):
        return value
    move_kwargs: dict[str, Any] = {"device": device}
    if dtype is not None:
        try:
            if value.is_floating_point():
                move_kwargs["dtype"] = dtype
        except Exception:
            pass
    return value.to(**move_kwargs)


def _looks_like_openvla_request(config: dict[str, Any] | None, model_id: str, *, model_config: Any | None = None) -> bool:
    signal_parts = [str(model_id or "").strip().lower()]
    if isinstance(config, dict):
        signal_parts.extend(
            str(config.get(key) or "").strip().lower()
            for key in (
                "model_backend",
                "business_intent",
                "output_type_hint",
                "architectures",
                "model_class",
            )
        )
    if model_config is not None:
        signal_parts.extend(
            str(value or "").strip().lower()
            for value in (
                getattr(model_config, "model_type", None),
                getattr(model_config, "architectures", None),
            )
        )
    signal_text = " ".join(signal_parts)
    return any(
        token in signal_text
        for token in (
            "openvla",
            "openvlaforactionprediction",
            "vision-language-action",
            "generated_action",
            "openvla_action_prediction",
        )
    )


def _infer_activation_name(activation: Any) -> str | None:
    current_name = getattr(activation, "__name__", None)
    if isinstance(current_name, str) and current_name.strip():
        return current_name.strip()
    class_name = str(getattr(activation.__class__, "__name__", "") or "").strip().lower()
    if not class_name:
        return None
    alias_map = {
        "gelu": "gelu",
        "relu": "relu",
        "silu": "silu",
        "swish": "silu",
        "tanh": "tanh",
        "sigmoid": "sigmoid",
    }
    for token, alias in alias_map.items():
        if token in class_name:
            return alias
    return class_name


class _NamedActivationWrapper(torch.nn.Module):
    def __init__(self, activation: Any, name: str):
        super().__init__()
        self.activation = activation
        self.__name__ = name

    def forward(self, *args, **kwargs):
        return self.activation(*args, **kwargs)


def _stabilize_activation_name_shims(model) -> list[str]:
    patched_modules: list[str] = []
    attr_candidates = ("intermediate_act_fn", "activation_fn", "act_fn")
    for module_name, module in model.named_modules():
        for attr_name in attr_candidates:
            activation = getattr(module, attr_name, None)
            if activation is None or not callable(activation):
                continue
            activation_name = _infer_activation_name(activation)
            if not activation_name:
                continue
            current_name = getattr(activation, "__name__", None)
            if isinstance(current_name, str) and current_name.strip():
                continue
            try:
                setattr(activation, "__name__", activation_name)
            except Exception:
                if isinstance(activation, torch.nn.Module):
                    setattr(module, attr_name, _NamedActivationWrapper(activation, activation_name))
                else:

                    def _wrapped(*args, _activation=activation, **kwargs):
                        return _activation(*args, **kwargs)

                    _wrapped.__name__ = activation_name
                    setattr(module, attr_name, _wrapped)
            patched_modules.append(f"{module_name or '<root>'}.{attr_name}={activation_name}")
    return patched_modules


def _get_device_for_scenario(scenario: str) -> torch.device:
    if scenario == "cuda_baseline":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA baseline 场景要求当前环境可用 CUDA")
        return torch.device("cuda:0")
    try:
        import torch_npu  # noqa: F401

        npu_module = getattr(torch, "npu", None)
        if npu_module is not None and npu_module.is_available():
            return torch.device("npu:0")
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _resolve_open_clip_spec(model_id: str, config: dict[str, Any]) -> dict[str, str] | None:
    arch = str(config.get("open_clip_model_arch") or "").strip()
    checkpoint = str(config.get("open_clip_checkpoint") or "").strip()
    label_template = str(config.get("open_clip_label_template") or "").strip()
    if arch and checkpoint:
        return {
            "arch": arch,
            "checkpoint": checkpoint,
            "label_template": label_template or "a photo of a {}",
        }
    default_spec = OPEN_CLIP_MODEL_SPECS.get(model_id)
    if default_spec is None:
        return None
    return dict(default_spec)


def _infer_open_clip_loader_id(model_id: str, config: dict[str, Any]) -> str:
    explicit = str(config.get("open_clip_loader_id") or "").strip()
    if explicit:
        return explicit
    for candidate_name in ("accuracy_run_perf.py", "accuracy_run.py", "demo.py"):
        candidate_path = ADAPT_DIR / candidate_name
        if not candidate_path.exists():
            continue
        try:
            source = candidate_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pattern in (
            r'open_clip\.create_model_and_transforms\(\s*["\']([^"\']+)["\']',
            r'open_clip\.get_tokenizer\(\s*["\']([^"\']+)["\']',
        ):
            match = re.search(pattern, source)
            if match:
                value = str(match.group(1) or "").strip()
                if value.startswith("hf-hub:"):
                    return value
    snapshot_source = _resolve_open_clip_snapshot_source(model_id)
    if snapshot_source is not None:
        return f"hf-hub:{model_id}"
    return ""


def _infer_open_clip_arch(model_id: str, config: dict[str, Any]) -> str:
    explicit = str(config.get("open_clip_model_arch") or "").strip()
    if explicit:
        return explicit
    default_spec = OPEN_CLIP_MODEL_SPECS.get(model_id)
    if isinstance(default_spec, dict):
        default_arch = str(default_spec.get("arch") or "").strip()
        if default_arch:
            return default_arch
    for candidate_name in ("accuracy_run_perf.py", "accuracy_run.py", "demo.py"):
        candidate_path = ADAPT_DIR / candidate_name
        if not candidate_path.exists():
            continue
        try:
            source = candidate_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pattern in (
            r'open_clip\.create_model_and_transforms\(\s*["\']([^"\']+)["\']',
            r'open_clip\.get_tokenizer\(\s*["\']([^"\']+)["\']',
        ):
            match = re.search(pattern, source)
            if match:
                value = str(match.group(1) or "").strip()
                if value and not value.startswith("hf-hub:"):
                    return value
    return ""


def _resolve_open_clip_snapshot_source(model_id: str) -> str | None:
    if not model_id or "/" not in model_id:
        return None
    org, name = model_id.split("/", 1)
    cache_dir = CACHE_DIR / f"models--{org.replace('/', '--')}--{name.replace('/', '--')}"
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    snapshot_candidates = sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), reverse=True)
    for candidate in snapshot_candidates:
        for asset_dir in _iter_snapshot_asset_dirs(candidate):
            if not (asset_dir / "open_clip_config.json").exists():
                continue
            if any((asset_dir / filename).exists() for filename in ("open_clip_pytorch_model.bin", "open_clip_model.safetensors")):
                return str(asset_dir)
    return None


def _resolve_open_clip_checkpoint_name(snapshot_source: str) -> str:
    source_path = Path(snapshot_source)
    for filename in ("open_clip_pytorch_model.bin", "open_clip_model.safetensors"):
        if (source_path / filename).exists():
            return filename
    return ""


def _resolve_zero_shot_label_template(config: dict[str, Any], default: str = "a photo of a {}") -> str:
    template = str(config.get("zero_shot_label_template") or config.get("open_clip_label_template") or "").strip()
    return template or default


def _load_clipseg_stack(config: dict[str, Any], model_id: str, scenario: str, target_device: torch.device, load_context: dict[str, Any]):
    trust_remote_code = _as_bool(config.get("trust_remote_code"), default=False)
    model_source, input_source, load_context = _resolve_model_sources(model_id, config, scenario, "processor")
    try:
        from transformers import AutoConfig, AutoProcessor, CLIPSegForImageSegmentation
    except Exception:
        if "clipseg" not in str(model_id or "").strip().lower():
            return None
        raise

    try:
        model_config = AutoConfig.from_pretrained(model_source, cache_dir=str(CACHE_DIR), trust_remote_code=trust_remote_code)
    except Exception:
        if "clipseg" not in str(model_id or "").strip().lower():
            return None
        raise
    if not _looks_like_clipseg_request(config, model_id, model_config=model_config):
        return None

    torch_dtype = _get_torch_dtype(config)
    processor = AutoProcessor.from_pretrained(input_source, cache_dir=str(CACHE_DIR), trust_remote_code=trust_remote_code)
    load_kwargs = {
        "cache_dir": str(CACHE_DIR),
        "trust_remote_code": trust_remote_code,
        "config": model_config,
    }
    if torch_dtype != "auto":
        load_kwargs["torch_dtype"] = torch_dtype
    _apply_model_load_seed(config)
    model = CLIPSegForImageSegmentation.from_pretrained(model_source, **load_kwargs)
    move_kwargs: dict[str, Any] = {"device": target_device}
    if torch_dtype != "auto":
        move_kwargs["dtype"] = torch_dtype
    model = model.to(**move_kwargs)
    patch_metadata = load_context.get("_patch_metadata")
    if isinstance(patch_metadata, dict):
        _apply_deferred_model_files_hooks(model, patch_metadata)
        load_context["patch_load_status"] = str(patch_metadata.get("status") or load_context.get("patch_load_status") or "disabled")
        load_context["patch_modules"] = list(patch_metadata.get("imported_modules") or [])
        load_context["patch_hooks"] = list(patch_metadata.get("called_hooks") or [])
        load_context["patch_errors"] = list(patch_metadata.get("errors") or [])
    load_context["input_source"] = str(input_source or model_source)
    load_context["input_source_kind"] = "processor"
    load_context["runtime_compatibility_shims"] = _apply_model_runtime_compatibility_shims(model)
    model.eval()
    return model, processor, model_config, "semantic_segmentation", load_context


def _resolve_open_clip_business_model_type(config: dict[str, Any]) -> str:
    configured_model_type = str(config.get("model_type") or "").strip().lower()
    if configured_model_type in {"vision_classification", "vision_embedding"}:
        return configured_model_type
    evaluation_profile = str(config.get("evaluation_profile") or "").strip().lower()
    output_type_hint = str(config.get("output_type_hint") or "").strip().lower()
    if evaluation_profile == "embedding_similarity" and output_type_hint in {"image_embeddings", "embeddings"}:
        return "vision_embedding"
    return "vision_classification"


def _infer_ultralytics_model_file(config: dict[str, Any]) -> str:
    explicit = str(config.get("model_file") or "").strip()
    if explicit:
        return explicit
    for candidate_name in ("accuracy_run.py", "accuracy_run_perf.py", "demo.py"):
        candidate_path = ADAPT_DIR / candidate_name
        if not candidate_path.exists():
            continue
        try:
            source = candidate_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        match = re.search(r'MODEL_FILE\s*=\s*["\']([^"\']+)["\']', source)
        if match:
            return match.group(1).strip()
    return ""


def _infer_timm_model_name(model_id: str, config: dict[str, Any]) -> str:
    explicit = str(config.get("timm_model_name") or "").strip()
    if explicit:
        return explicit
    if model_id.startswith("timm/"):
        _, model_name = model_id.split("/", 1)
        if model_name:
            return model_name
    for candidate_name in ("accuracy_run_perf.py", "accuracy_run.py", "demo.py"):
        candidate_path = ADAPT_DIR / candidate_name
        if not candidate_path.exists():
            continue
        try:
            source = candidate_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        match = re.search(r'MODEL_NAME\s*=\s*["\']([^"\']+)["\']', source)
        if match:
            return match.group(1).strip()
        match = re.search(r'timm\.create_model\(\s*["\']([^"\']+)["\']', source)
        if match:
            return match.group(1).strip()
    return model_id


def _infer_segment_anything_variant(model_id: str, config: dict[str, Any]) -> str:
    explicit = str(config.get("sam_model_type") or "").strip()
    if explicit:
        return explicit
    for candidate_name in ("accuracy_run_perf.py", "accuracy_run.py", "demo.py"):
        candidate_path = ADAPT_DIR / candidate_name
        if not candidate_path.exists():
            continue
        try:
            source = candidate_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        match = re.search(r'sam_model_registry\[\s*["\']([^"\']+)["\']\s*\]', source)
        if match:
            return match.group(1).strip()
    model_id_lower = model_id.lower()
    if "vit-h" in model_id_lower or "vit_h" in model_id_lower:
        return "vit_h"
    if "vit-l" in model_id_lower or "vit_l" in model_id_lower:
        return "vit_l"
    return "vit_b"


def _infer_segment_anything_checkpoint_name(model_id: str, config: dict[str, Any]) -> str:
    explicit = str(config.get("model_file") or config.get("checkpoint_filename") or "").strip()
    if explicit:
        return explicit
    for candidate_name in ("accuracy_run_perf.py", "accuracy_run.py", "demo.py"):
        candidate_path = ADAPT_DIR / candidate_name
        if not candidate_path.exists():
            continue
        try:
            source = candidate_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        match = re.search(r'filename\s*=\s*["\']([^"\']+\.pth)["\']', source)
        if match:
            return match.group(1).strip()
    variant = _infer_segment_anything_variant(model_id, config)
    fallback_map = {
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_b": "sam_vit_b_01ec64.pth",
    }
    return fallback_map.get(variant, "sam_vit_h_4b8939.pth")


def _load_segment_anything_stack(config: dict[str, Any], model_id: str, target_device: torch.device, load_context: dict[str, Any]):
    backend = str(config.get("model_backend") or "").strip().lower()
    if backend != "segment_anything_image_encoder":
        return None

    from huggingface_hub import hf_hub_download
    from segment_anything import sam_model_registry

    checkpoint_name = _infer_segment_anything_checkpoint_name(model_id, config)
    variant = _infer_segment_anything_variant(model_id, config)
    checkpoint_path = Path(hf_hub_download(repo_id=model_id, filename=checkpoint_name, cache_dir=str(CACHE_DIR)))
    sam = sam_model_registry[variant](checkpoint=str(checkpoint_path))
    model = _SegmentAnythingImageEncoderWrapper(sam).to(target_device)
    model.eval()
    load_context["model_source"] = str(checkpoint_path)
    load_context["model_source_kind"] = "hub_checkpoint"
    load_context["input_source"] = variant
    load_context["input_source_kind"] = "segment_anything_builtin"
    load_context["segment_anything_variant"] = variant
    load_context["segment_anything_checkpoint"] = checkpoint_path.name
    model_config = SimpleNamespace(id2label={}, num_labels=0, model_type="segment_anything_image_encoder")
    return model, _SegmentAnythingImageInputAdapter(image_size=1024), model_config, "vision_embedding", load_context


def _load_ultralytics_stack(config: dict[str, Any], model_id: str, target_device: torch.device, load_context: dict[str, Any]):
    backend = str(config.get("model_backend") or "").strip().lower()
    if backend != "ultralytics_yolo":
        return None

    model_file = _infer_ultralytics_model_file(config)
    if not model_file:
        raise RuntimeError("ultralytics_yolo 业务测评缺少 model_file 配置，且无法从 adaptation 脚本推断")

    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    checkpoint_path = Path(hf_hub_download(repo_id=model_id, filename=model_file, cache_dir=str(CACHE_DIR)))
    model = YOLO(str(checkpoint_path))
    model.to(str(target_device))
    inner_model = getattr(model, "model", None)
    if inner_model is not None and hasattr(inner_model, "eval"):
        inner_model.eval()
    load_context["model_source_kind"] = "hub_checkpoint"
    load_context["input_source_kind"] = "ultralytics_builtin"
    load_context["model_source"] = str(checkpoint_path)
    load_context["input_source"] = "ultralytics_yolo"
    load_context["ultralytics_model_file"] = checkpoint_path.name
    model_config = SimpleNamespace(id2label={}, num_labels=0, model_type="ultralytics_yolo")
    return model, SimpleNamespace(backend="ultralytics_yolo", model_file=checkpoint_path.name), model_config, "vision_detection", load_context


def _load_timm_stack(config: dict[str, Any], model_id: str, scenario: str, target_device: torch.device, load_context: dict[str, Any]):
    backend = str(config.get("model_backend") or "").strip().lower()
    if backend != "timm_image_classification":
        return None

    import timm

    model_name = _infer_timm_model_name(model_id, config)
    _, _, load_context = _resolve_model_sources(model_id, config, scenario, "image_processor")
    os.environ["TIMM_CACHE"] = str(CACHE_DIR)
    os.environ["HF_HOME"] = str(CACHE_DIR)
    model = timm.create_model(model_name, pretrained=True, cache_dir=str(CACHE_DIR))
    model = model.to(target_device)

    patch_metadata = load_context.get("_patch_metadata")
    if isinstance(patch_metadata, dict):
        _apply_deferred_model_files_hooks(model, patch_metadata)
        load_context["patch_load_status"] = str(patch_metadata.get("status") or load_context.get("patch_load_status") or "disabled")
        load_context["patch_modules"] = list(patch_metadata.get("imported_modules") or [])
        load_context["patch_hooks"] = list(patch_metadata.get("called_hooks") or [])
        load_context["patch_errors"] = list(patch_metadata.get("errors") or [])
    load_context["activation_name_shims"] = _stabilize_activation_name_shims(model)

    model.eval()
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    num_classes = int(getattr(model, "num_classes", 0) or 0)
    load_context["model_source"] = model_id
    load_context["model_source_kind"] = "timm_builtin"
    load_context["input_source"] = model_name
    load_context["input_source_kind"] = "timm_transform"
    load_context["timm_model_name"] = model_name
    model_config = SimpleNamespace(
        id2label={idx: str(idx) for idx in range(num_classes)},
        num_labels=num_classes,
        model_type="timm_image_classification",
    )
    return model, _TimmImageInputAdapter(transform), model_config, "vision_classification", load_context


def _load_open_clip_stack(config: dict[str, Any], model_id: str, model_source: str, target_device: torch.device, load_context: dict[str, Any]):
    spec = _resolve_open_clip_spec(model_id, config)
    loader_id = _infer_open_clip_loader_id(model_id, config)
    snapshot_source = _resolve_open_clip_snapshot_source(model_id)
    runtime_model_type = _resolve_open_clip_business_model_type(config)
    if loader_id.startswith("hf-hub:") and snapshot_source:
        inferred_arch = _infer_open_clip_arch(model_id, config)
        inferred_checkpoint = _resolve_open_clip_checkpoint_name(snapshot_source)
        if inferred_arch and inferred_checkpoint:
            spec = {
                "arch": inferred_arch,
                "checkpoint": inferred_checkpoint,
                "label_template": _resolve_zero_shot_label_template(config),
            }
            loader_id = ""
    if spec is None and not loader_id:
        return None

    import open_clip
    from huggingface_hub import hf_hub_download

    if loader_id:
        label_template = _resolve_zero_shot_label_template(config)
        model, _, preprocess = open_clip.create_model_and_transforms(loader_id, cache_dir=str(CACHE_DIR))
        tokenizer = open_clip.get_tokenizer(loader_id)
        model = model.to(target_device)
        model.eval()
        load_context["open_clip_loader_id"] = loader_id
        load_context["open_clip_arch"] = loader_id
        load_context["model_source_kind"] = "open_clip_builtin"
        load_context["input_source_kind"] = "open_clip_builtin"
        load_context["model_source"] = loader_id
        load_context["input_source"] = loader_id
        adapter = _OpenClipInputAdapter(preprocess=preprocess, tokenizer=tokenizer, label_template=label_template)
        model_config = SimpleNamespace(id2label={}, num_labels=0, model_type=runtime_model_type)
        return model, adapter, model_config, runtime_model_type, load_context

    if spec is None:
        return None

    model, _, preprocess = open_clip.create_model_and_transforms(spec["arch"])
    tokenizer = open_clip.get_tokenizer(spec["arch"])

    checkpoint_name = spec["checkpoint"]
    candidate_source = Path(model_source)
    if candidate_source.exists():
        if candidate_source.is_file():
            checkpoint_path = candidate_source
        else:
            checkpoint_path = candidate_source / checkpoint_name
    else:
        inferred_arch = _infer_open_clip_arch(model_id, config)
        inferred_checkpoint = _resolve_open_clip_checkpoint_name(snapshot_source) if snapshot_source else ""
        if snapshot_source and inferred_arch and inferred_checkpoint:
            checkpoint_path = Path(snapshot_source) / inferred_checkpoint
            load_context["open_clip_arch"] = inferred_arch
            model = open_clip.create_model(inferred_arch)
            tokenizer = open_clip.get_tokenizer(inferred_arch)
        else:
            checkpoint_path = Path(hf_hub_download(model_id, checkpoint_name, cache_dir=str(CACHE_DIR)))
            load_context["model_source_kind"] = "hub_checkpoint"
            load_context["input_source_kind"] = "open_clip_builtin"
    if not checkpoint_path.exists():
        raise RuntimeError(f"open_clip checkpoint 不存在: {checkpoint_path}")

    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(str(checkpoint_path))
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        if isinstance(checkpoint.get("state_dict"), dict):
            raw_state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint.get("model"), dict):
            raw_state_dict = checkpoint["model"]
        elif checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            raw_state_dict = checkpoint
        else:
            raw_state_dict = None
    else:
        raw_state_dict = checkpoint
    if not isinstance(raw_state_dict, dict):
        raise RuntimeError(f"open_clip checkpoint 缺少 state_dict: {checkpoint_path}")
    state_dict = {}
    for key, value in raw_state_dict.items():
        normalized_key = key[7:] if str(key).startswith("module.") else key
        state_dict[normalized_key] = value
    model.load_state_dict(state_dict, strict=False)
    model = model.to(target_device)
    model.eval()
    load_context["open_clip_arch"] = spec["arch"]
    load_context["open_clip_checkpoint"] = checkpoint_path.name
    load_context["input_source_kind"] = "open_clip_builtin"
    load_context["model_source"] = str(checkpoint_path)
    load_context["input_source"] = spec["arch"]
    adapter = _OpenClipInputAdapter(preprocess=preprocess, tokenizer=tokenizer, label_template=spec["label_template"])
    model_config = SimpleNamespace(id2label={}, num_labels=0, model_type=runtime_model_type)
    return model, adapter, model_config, runtime_model_type, load_context


def _load_siglip_stack(
    config: dict[str, Any],
    model_id: str,
    model_source: str,
    input_source: str,
    target_device: torch.device,
    load_context: dict[str, Any],
):
    if str(config.get("model_type") or "").strip() != "vision_classification":
        return None

    architecture_hint = " ".join(
        str(value)
        for value in (
            config.get("architectures"),
            config.get("model_class"),
            config.get("model_loader"),
            model_id,
        )
        if value
    ).lower()
    if "siglip" not in architecture_hint:
        return None

    from transformers import AutoModel, AutoProcessor, SiglipImageProcessor, SiglipModel, SiglipTokenizer

    trust_remote_code = _as_bool(config.get("trust_remote_code"), default=False)
    torch_dtype = _get_torch_dtype(config)
    load_kwargs = {
        "cache_dir": str(CACHE_DIR),
        "trust_remote_code": trust_remote_code,
    }
    if torch_dtype != "auto":
        load_kwargs["torch_dtype"] = torch_dtype

    text_preprocessor = None
    if "siglip2" in architecture_hint:
        processor_bundle = AutoProcessor.from_pretrained(input_source, cache_dir=str(CACHE_DIR), trust_remote_code=trust_remote_code)
        tokenizer = getattr(processor_bundle, "tokenizer", None) or processor_bundle
        image_processor = processor_bundle
        text_preprocessor = processor_bundle
        model = AutoModel.from_pretrained(model_source, **load_kwargs)
        load_context["siglip_family"] = "siglip2"
        load_context["siglip_loader"] = "AutoModel.from_pretrained"
    else:
        tokenizer = SiglipTokenizer.from_pretrained(input_source, cache_dir=str(CACHE_DIR), trust_remote_code=trust_remote_code)
        image_processor = SiglipImageProcessor.from_pretrained(input_source, cache_dir=str(CACHE_DIR), trust_remote_code=trust_remote_code)
        model = SiglipModel.from_pretrained(model_source, **load_kwargs)
        load_context["siglip_family"] = "siglip"
        load_context["siglip_loader"] = "SiglipModel.from_pretrained"
    model = model.to(target_device)

    if not hasattr(model, "encode_image") and hasattr(model, "get_image_features"):

        def _encode_image(self, pixel_values):
            if isinstance(pixel_values, Mapping):
                image_outputs = self.get_image_features(**pixel_values)
            else:
                image_outputs = self.get_image_features(pixel_values=pixel_values)
            return image_outputs.pooler_output if hasattr(image_outputs, "pooler_output") else image_outputs

        model.encode_image = MethodType(_encode_image, model)
    if not hasattr(model, "encode_text") and hasattr(model, "get_text_features"):

        def _encode_text(self, text_inputs):
            if isinstance(text_inputs, Mapping):
                text_outputs = self.get_text_features(**text_inputs)
                return text_outputs.pooler_output if hasattr(text_outputs, "pooler_output") else text_outputs
            raise RuntimeError(f"SigLIP text inputs 必须是 Mapping，当前得到 {type(text_inputs)!r}")

        model.encode_text = MethodType(_encode_text, model)

    model.eval()

    load_context["model_source"] = model_source
    load_context["input_source"] = input_source
    # SigLIP checkpoints behave better on CIFAR-style zero-shot classification
    # with bare labels than with CLIP's default "a photo of a {}" prompt.
    load_context["siglip_label_template"] = _resolve_zero_shot_label_template(config, default="{}")
    adapter = _TransformersZeroShotVisionInputAdapter(
        image_processor=image_processor,
        tokenizer=tokenizer,
        label_template=load_context["siglip_label_template"],
        text_preprocessor=text_preprocessor,
    )
    model_config = SimpleNamespace(id2label={}, num_labels=0, model_type="siglip")
    return model, adapter, model_config, "vision_classification", load_context


def _load_timeseries_stack(config: dict[str, Any], model_id: str, model_source: str, target_device: torch.device, load_context: dict[str, Any]):
    if str(config.get("model_type") or "").strip().lower() != "timeseries":
        return None

    trust_remote_code = _as_bool(config.get("trust_remote_code"), default=False)
    try:
        from transformers import AutoConfig
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
    except ImportError as exc:
        raise RuntimeError("timeseries 业务测评依赖 granite-tsfm / tsfm_public，请先确保环境依赖已安装。") from exc

    model_config, effective_trust_remote_code = _load_model_config_with_known_fallbacks(
        AutoConfig,
        model_source,
        model_id=model_id,
        trust_remote_code=trust_remote_code,
    )
    _apply_model_config_compatibility_fixes(model_config, model_id=model_id)

    model = TinyTimeMixerForPrediction.from_pretrained(model_source, cache_dir=str(CACHE_DIR))
    model = model.to(target_device)
    model.eval()

    resolved_model_source = str(model_source)
    load_context["model_source"] = resolved_model_source
    load_context["model_source_kind"] = "local_snapshot" if Path(resolved_model_source).exists() else "hub"
    load_context["input_source"] = ""
    load_context["input_source_kind"] = "none"
    load_context["timeseries_family"] = str(getattr(model_config, "model_type", "") or "tinytimemixer")
    return model, None, model_config, "timeseries", load_context


def _fallback_model_source_candidates(model_id: str, model_type: str) -> list[str]:
    candidates: list[str] = []
    if model_type == "embedding" and model_id.startswith("Xenova/"):
        _, model_name = model_id.split("/", 1)
        sentence_transformers_id = f"sentence-transformers/{model_name}"
        local_cache_dir = CACHE_DIR / f"models--sentence-transformers--{model_name.replace('/', '--')}"
        if local_cache_dir.exists():
            candidates.append(sentence_transformers_id)
    return candidates


def _resolve_explicit_torch_dtype(value: Any) -> torch.dtype | None:
    if isinstance(value, torch.dtype):
        return value
    text = str(value or "").strip().lower()
    dtype_map = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return dtype_map.get(text)


def _resolve_llava_snapshot_dir(model_source: str, model_id: str) -> Path | None:
    model_source_path = Path(str(model_source))
    if model_source_path.is_dir() and (model_source_path / "config.json").exists():
        return model_source_path
    snapshot_source = _resolve_local_snapshot_source(model_id, require_model_assets=True)
    if snapshot_source is None:
        return None
    snapshot_path = Path(snapshot_source)
    if snapshot_path.is_dir() and (snapshot_path / "config.json").exists():
        return snapshot_path
    return None


def _list_legacy_llava_checkpoint_paths(snapshot_dir: Path) -> list[Path]:
    checkpoint_paths: list[Path] = []
    index_path = snapshot_dir / "pytorch_model.bin.index.json"
    if index_path.exists():
        try:
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            index_payload = {}
        weight_map = index_payload.get("weight_map")
        if isinstance(weight_map, dict):
            for shard_name in dict.fromkeys(str(value) for value in weight_map.values()):
                shard_path = snapshot_dir / shard_name
                if shard_path.exists():
                    checkpoint_paths.append(shard_path)
    if not checkpoint_paths:
        checkpoint_paths.extend(sorted(snapshot_dir.glob("pytorch_model*.bin")))
    projector_path = snapshot_dir / "mm_projector.bin"
    if projector_path.exists():
        checkpoint_paths.append(projector_path)
    return checkpoint_paths


def _is_legacy_llava_checkpoint(snapshot_dir: Path, model_config) -> bool:
    architectures = [str(item).strip().lower() for item in getattr(model_config, "architectures", []) or []]
    if any("llavallamaforcausallm" in item for item in architectures):
        return True
    if (snapshot_dir / "mm_projector.bin").exists():
        return True
    transformers_version = str(getattr(model_config, "transformers_version", "") or "").strip()
    return transformers_version.startswith("4.31")


def _infer_legacy_llava_image_size(snapshot_dir: Path, *, patch_size: int) -> int | None:
    position_embedding_key = "model.vision_tower.vision_tower.vision_model.embeddings.position_embedding.weight"
    for checkpoint_path in _list_legacy_llava_checkpoint_paths(snapshot_dir):
        if checkpoint_path.name == "mm_projector.bin":
            continue
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        position_embedding = state_dict.get(position_embedding_key)
        del state_dict
        if position_embedding is None or len(getattr(position_embedding, "shape", ())) < 2:
            continue
        num_positions = int(position_embedding.shape[0])
        patch_tokens = num_positions - 1
        patch_grid = int(round(patch_tokens**0.5))
        if patch_grid > 0 and patch_grid * patch_grid == patch_tokens:
            return int(patch_grid * patch_size)
    return None


def _load_legacy_llava_snapshot_config(snapshot_dir: Path) -> dict[str, Any]:
    config_path = snapshot_dir / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"legacy Llava snapshot 缺少 config.json: {snapshot_dir}")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"读取 legacy Llava config 失败: {config_path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"legacy Llava config.json 不是 object: {config_path}")
    return payload


def _build_legacy_llava_text_config(model_config, snapshot_config: Mapping[str, Any]):
    from transformers import LlamaConfig

    existing_text_config = getattr(model_config, "text_config", None)
    if existing_text_config is not None and hasattr(existing_text_config, "to_dict"):
        text_config_payload = dict(existing_text_config.to_dict())
    else:
        text_config_payload = {}

    for field_name in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
        "vocab_size",
        "rms_norm_eps",
        "rope_theta",
        "rope_scaling",
        "hidden_act",
        "head_dim",
        "pretraining_tp",
        "attention_bias",
        "attention_dropout",
        "mlp_bias",
        "initializer_range",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "tie_word_embeddings",
        "use_cache",
        "torch_dtype",
    ):
        field_value = snapshot_config.get(field_name, _MISSING_ATTR)
        if field_value is not _MISSING_ATTR:
            text_config_payload[field_name] = field_value

    text_config_payload.setdefault("model_type", "llama")
    return LlamaConfig(**text_config_payload)


def _build_legacy_llava_model_config(model_config, snapshot_dir: Path, *, image_token_id: int):
    from transformers import LlavaConfig

    snapshot_config = _load_legacy_llava_snapshot_config(snapshot_dir)
    config_payload = dict(model_config.to_dict()) if hasattr(model_config, "to_dict") else {}

    vision_config = getattr(model_config, "vision_config", None)
    vision_config_payload = dict(vision_config.to_dict()) if vision_config is not None and hasattr(vision_config, "to_dict") else {}
    patch_size = int(vision_config_payload.get("patch_size") or getattr(vision_config, "patch_size", 0) or 14)
    image_size = _infer_legacy_llava_image_size(snapshot_dir, patch_size=patch_size) or int(vision_config_payload.get("image_size") or getattr(vision_config, "image_size", 0) or 336)
    vision_config_payload["image_size"] = image_size
    vision_config_payload["patch_size"] = patch_size

    legacy_feature = str(snapshot_config.get("mm_vision_select_feature") or getattr(model_config, "mm_vision_select_feature", "") or "").strip().lower()
    vision_feature_select_strategy = "default" if legacy_feature == "patch" else (legacy_feature or str(config_payload.get("vision_feature_select_strategy") or "default"))
    legacy_feature_layer = snapshot_config.get("mm_vision_select_layer", getattr(model_config, "mm_vision_select_layer", None))

    text_config = _build_legacy_llava_text_config(model_config, snapshot_config)
    config_payload["text_config"] = text_config.to_dict()
    config_payload["vision_config"] = vision_config_payload
    config_payload["hidden_size"] = int(snapshot_config.get("hidden_size") or config_payload.get("hidden_size") or text_config.hidden_size)
    config_payload["vocab_size"] = int(snapshot_config.get("vocab_size") or config_payload.get("vocab_size") or text_config.vocab_size)
    config_payload["image_seq_length"] = int((image_size // patch_size) ** 2)
    config_payload["image_token_index"] = int(image_token_id)
    config_payload["projector_hidden_act"] = str(snapshot_config.get("projector_hidden_act") or config_payload.get("projector_hidden_act") or "linear")
    config_payload["multimodal_projector_bias"] = bool(snapshot_config.get("multimodal_projector_bias", config_payload.get("multimodal_projector_bias", True)))
    config_payload["vision_feature_select_strategy"] = vision_feature_select_strategy
    if legacy_feature_layer is not None:
        config_payload["vision_feature_layer"] = int(legacy_feature_layer)
    try:
        config_payload["image_token_id"] = int(image_token_id)
    except Exception:
        pass

    for field_name in (
        "mm_hidden_size",
        "mm_projector_type",
        "mm_vision_tower",
        "mm_use_im_patch_token",
        "mm_use_im_start_end",
        "mm_patch_merge_type",
        "image_aspect_ratio",
        "architectures",
    ):
        field_value = snapshot_config.get(field_name, _MISSING_ATTR)
        if field_value is not _MISSING_ATTR:
            config_payload[field_name] = field_value

    return LlavaConfig(**config_payload)


def _configure_llava_image_processor(image_processor, model_config) -> int:
    target_image_size = int(getattr(getattr(model_config, "vision_config", None), "image_size", 0) or 336)
    size_value = getattr(image_processor, "size", None)
    if isinstance(size_value, dict) and "shortest_edge" not in size_value:
        image_processor.size = {"height": target_image_size, "width": target_image_size}
    else:
        image_processor.size = {"shortest_edge": target_image_size}
    image_processor.crop_size = {"height": target_image_size, "width": target_image_size}
    if hasattr(image_processor, "do_center_crop"):
        image_processor.do_center_crop = True
    return target_image_size


def _remap_legacy_llava_state_key(key: str) -> str:
    if key.startswith("model.layers.") or key.startswith("model.embed_tokens.") or key.startswith("model.norm."):
        return key.replace("model.", "model.language_model.", 1)
    if key.startswith("model.vision_tower.vision_tower."):
        return key.replace("model.vision_tower.vision_tower.", "model.vision_tower.", 1)
    if key == "model.mm_projector.weight":
        return "model.multi_modal_projector.linear_1.weight"
    if key == "model.mm_projector.bias":
        return "model.multi_modal_projector.linear_1.bias"
    return key


def _initialize_identity_linear(linear) -> None:
    with torch.no_grad():
        linear.weight.zero_()
        diagonal_size = min(int(linear.weight.shape[0]), int(linear.weight.shape[1]))
        diagonal_indices = torch.arange(diagonal_size, device=linear.weight.device)
        linear.weight[diagonal_indices, diagonal_indices] = 1
        if getattr(linear, "bias", None) is not None:
            linear.bias.zero_()


def _load_legacy_llava_vision_tower(model, model_config, *, cache_dir: Path) -> dict[str, Any]:
    from transformers import CLIPVisionModel

    configured_vision_source = str(getattr(model_config, "mm_vision_tower", "") or "").strip()
    if not configured_vision_source:
        raise RuntimeError("legacy Llava 缺少 mm_vision_tower，无法补齐 vision tower 权重")

    candidate_sources: list[str] = []
    if configured_vision_source == "openai/clip-vit-large-patch14":
        candidate_sources.append("openai/clip-vit-large-patch14-336")
    candidate_sources.append(configured_vision_source)

    vision_tower = None
    resolved_vision_source = None
    last_error = None
    for candidate_source in candidate_sources:
        resolved_candidate_source = _resolve_local_snapshot_source(candidate_source, require_model_assets=True) or candidate_source
        vision_load_kwargs = {"cache_dir": str(cache_dir)}
        if Path(str(resolved_candidate_source)).exists():
            vision_load_kwargs["local_files_only"] = True
        try:
            vision_tower = CLIPVisionModel.from_pretrained(resolved_candidate_source, **vision_load_kwargs)
            resolved_vision_source = str(resolved_candidate_source)
            break
        except Exception as exc:
            last_error = exc
    if vision_tower is None:
        raise RuntimeError(f"legacy Llava vision tower 加载失败: source={configured_vision_source}, error={last_error}") from last_error

    missing_keys, unexpected_keys = model.model.vision_tower.load_state_dict(vision_tower.state_dict(), strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(f"legacy Llava vision tower load incomplete: missing={missing_keys[:8]}, unexpected={unexpected_keys[:8]}")
    return {
        "legacy_llava_vision_source": str(resolved_vision_source),
        "legacy_llava_vision_loader": "clip_vision_from_pretrained",
    }


def _load_legacy_llava_model(snapshot_dir: Path, model_config, *, target_device: torch.device, requested_torch_dtype: Any):
    from transformers import LlavaForConditionalGeneration

    load_dtype = _resolve_explicit_torch_dtype(requested_torch_dtype) or _resolve_explicit_torch_dtype(getattr(model_config, "torch_dtype", None))
    model = LlavaForConditionalGeneration(model_config)
    if load_dtype is not None:
        model = model.to(dtype=load_dtype)

    _initialize_identity_linear(model.model.multi_modal_projector.linear_2)

    model_state_keys = set(model.state_dict().keys())
    loaded_keys: set[str] = set()
    unexpected_keys: list[str] = []
    ignorable_unexpected_suffixes = ("rotary_emb.inv_freq",)

    for checkpoint_path in _list_legacy_llava_checkpoint_paths(snapshot_dir):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        remapped_state_dict: dict[str, Any] = {}
        for key, value in state_dict.items():
            mapped_key = _remap_legacy_llava_state_key(str(key))
            if mapped_key not in model_state_keys:
                if mapped_key.endswith(ignorable_unexpected_suffixes):
                    continue
                unexpected_keys.append(f"{key}->{mapped_key}")
                continue
            remapped_state_dict[mapped_key] = value
        model.load_state_dict(remapped_state_dict, strict=False)
        loaded_keys.update(remapped_state_dict.keys())
        del state_dict
        del remapped_state_dict

    allowed_missing_keys = {
        "model.multi_modal_projector.linear_2.weight",
        "model.multi_modal_projector.linear_2.bias",
    }
    missing_keys = [key for key in sorted(model_state_keys - loaded_keys - allowed_missing_keys) if not key.endswith("position_ids")]
    vision_missing_prefix = "model.vision_tower."
    vision_missing_keys = [key for key in missing_keys if key.startswith(vision_missing_prefix)]
    non_vision_missing_keys = [key for key in missing_keys if not key.startswith(vision_missing_prefix)]
    if non_vision_missing_keys or unexpected_keys:
        raise RuntimeError(f"legacy Llava checkpoint remap incomplete: missing={non_vision_missing_keys[:8]}, unexpected={unexpected_keys[:8]}")

    vision_context: dict[str, Any] = {}
    if vision_missing_keys:
        vision_context = _load_legacy_llava_vision_tower(model, model_config, cache_dir=CACHE_DIR)
        loaded_keys.update(vision_missing_keys)
        missing_keys = [key for key in sorted(model_state_keys - loaded_keys - allowed_missing_keys) if not key.endswith("position_ids")]
    if missing_keys or unexpected_keys:
        raise RuntimeError(f"legacy Llava checkpoint remap incomplete: missing={missing_keys[:8]}, unexpected={unexpected_keys[:8]}")

    model = model.to(target_device)
    compat_context = {
        "legacy_llava_loader": "manual_state_dict_remap",
        "legacy_llava_checkpoint_files": [path.name for path in _list_legacy_llava_checkpoint_paths(snapshot_dir)],
        "legacy_llava_missing_keys": missing_keys,
        "legacy_llava_unexpected_keys": unexpected_keys,
    }
    compat_context.update(vision_context)
    return model, compat_context


def _resize_token_embeddings_if_needed(model, tokenizer) -> None:
    if not hasattr(model, "get_input_embeddings") or not hasattr(model, "resize_token_embeddings"):
        return
    try:
        embeddings = model.get_input_embeddings()
    except NotImplementedError:
        return
    current_vocab_size = int(getattr(embeddings, "num_embeddings", 0) or 0) if embeddings is not None else 0
    target_vocab_size = len(tokenizer)
    if current_vocab_size > 0 and target_vocab_size > current_vocab_size:
        try:
            model.resize_token_embeddings(target_vocab_size, mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(target_vocab_size)


def _resolve_known_vlm_direct_loader(transformers_module, vlm_family: str):
    loader_name = {
        "blip": "BlipForConditionalGeneration",
        "deepseek_vl_v2": "AutoModel",
        "glm_ocr": "GlmOcrForConditionalGeneration",
        "idefics3": "Idefics3ForConditionalGeneration",
        "internvl_chat": "AutoModel",
        "qwen2_5_omni": "Qwen2_5OmniForConditionalGeneration",
        "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration",
        "qwen2_vl": "Qwen2VLForConditionalGeneration",
        "qwen3_vl_moe": "Qwen3VLMoeForConditionalGeneration",
        "qwen3_vl": "Qwen3VLForConditionalGeneration",
    }.get(str(vlm_family or "").strip().lower())
    if not loader_name:
        return None
    direct_loader = getattr(transformers_module, loader_name, None)
    if direct_loader is not None:
        return direct_loader
    if str(vlm_family or "").strip().lower() == "glm_ocr":
        try:
            from transformers.models.glm_ocr.modeling_glm_ocr import GlmOcrForConditionalGeneration

            return GlmOcrForConditionalGeneration
        except Exception:
            return None
    if str(vlm_family or "").strip().lower() == "qwen3_vl_moe":
        try:
            from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration

            return Qwen3VLMoeForConditionalGeneration
        except Exception:
            return None
    return None


def _is_moondream_vlm_family(vlm_family: str, *, model_config: Any | None = None) -> bool:
    normalized_family = str(vlm_family or "").strip().lower()
    if normalized_family == "moondream1":
        return True

    architecture_names = [str(name or "").strip().lower() for name in getattr(model_config, "architectures", []) or []]
    if any("moondream" in name for name in architecture_names):
        return True

    auto_map = getattr(model_config, "auto_map", {}) or {}
    if isinstance(auto_map, Mapping):
        auto_map_values = " ".join(str(value or "").strip().lower() for value in auto_map.values())
        if "moondream" in auto_map_values:
            return True
    return False


def _build_qwen_image_only_video_processor(transformers_module):
    base_video_processor_cls = getattr(transformers_module, "BaseVideoProcessor", None)
    if base_video_processor_cls is None:
        raise RuntimeError("当前 transformers 缺少 BaseVideoProcessor，无法构造 Qwen VLM video processor shim")

    class _QwenImageOnlyVideoProcessor(base_video_processor_cls):
        model_input_names = ["pixel_values_videos"]

        def __init__(self):
            self.merge_size = 2
            self.temporal_patch_size = 2

        def __call__(self, videos=None, **kwargs):
            raise RuntimeError("当前环境缺少 torchvision，Qwen VLM 第四阶段仅支持图像输入；如需视频输入请安装 torchvision")

        preprocess = __call__

    return _QwenImageOnlyVideoProcessor()


def _load_qwen_vl_processor(input_source: str, *, trust_remote_code: bool, vlm_family: str):
    import transformers as transformers_module
    from transformers import AutoTokenizer, Qwen2VLImageProcessor

    processor_cls_name = {
        "qwen2_vl": "Qwen2VLProcessor",
        "qwen2_5_vl": "Qwen2_5_VLProcessor",
        "qwen3_vl_moe": "Qwen3VLProcessor",
        "qwen3_vl": "Qwen3VLProcessor",
    }.get(str(vlm_family or "").strip().lower())
    processor_cls = getattr(transformers_module, processor_cls_name or "", None)
    if processor_cls is None:
        raise RuntimeError(f"当前 transformers 版本缺少 {processor_cls_name or '<unknown>'}，无法加载 Qwen VLM processor")

    source_path = Path(str(input_source))
    local_files_only = source_path.exists()
    tokenizer = AutoTokenizer.from_pretrained(
        input_source,
        cache_dir=str(CACHE_DIR),
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    image_processor = Qwen2VLImageProcessor.from_pretrained(
        input_source,
        cache_dir=str(CACHE_DIR),
        local_files_only=local_files_only,
    )
    video_processor_error = None
    try:
        from transformers import Qwen2VLVideoProcessor

        video_processor = Qwen2VLVideoProcessor.from_pretrained(
            input_source,
            cache_dir=str(CACHE_DIR),
            local_files_only=local_files_only,
        )
    except Exception as exc:
        video_processor_error = exc
        video_processor = _build_qwen_image_only_video_processor(transformers_module)
    chat_template = getattr(tokenizer, "chat_template", None)
    processor = processor_cls(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=chat_template,
    )
    if video_processor_error is not None:
        setattr(processor, "_business_video_processor_fallback_reason", str(video_processor_error))
    return processor


def _load_qwen_omni_processor(input_source: str, *, trust_remote_code: bool):
    import transformers as transformers_module
    from transformers import AutoFeatureExtractor, AutoTokenizer, Qwen2VLImageProcessor

    processor_cls = getattr(transformers_module, "Qwen2_5OmniProcessor", None)
    if processor_cls is None:
        raise RuntimeError("当前 transformers 版本缺少 Qwen2_5OmniProcessor，无法加载 Qwen Omni processor")

    source_path = Path(str(input_source))
    local_files_only = source_path.exists()
    tokenizer = AutoTokenizer.from_pretrained(
        input_source,
        cache_dir=str(CACHE_DIR),
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        input_source,
        cache_dir=str(CACHE_DIR),
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    image_processor = Qwen2VLImageProcessor.from_pretrained(
        input_source,
        cache_dir=str(CACHE_DIR),
        local_files_only=local_files_only,
    )
    video_processor_error = None
    try:
        from transformers import Qwen2VLVideoProcessor

        video_processor = Qwen2VLVideoProcessor.from_pretrained(
            input_source,
            cache_dir=str(CACHE_DIR),
            local_files_only=local_files_only,
        )
    except Exception as exc:
        video_processor_error = exc
        video_processor = _build_qwen_image_only_video_processor(transformers_module)
    processor = processor_cls(
        image_processor=image_processor,
        video_processor=video_processor,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        chat_template=getattr(tokenizer, "chat_template", None),
    )
    if video_processor_error is not None:
        setattr(processor, "_business_video_processor_fallback_reason", str(video_processor_error))
    return processor


def _configure_deepseek_vl_v2_input_adapter(input_adapter, input_source: str) -> None:
    processor_config_path = Path(str(input_source)) / "processor_config.json"
    processor_config: dict[str, Any] = {}
    try:
        if processor_config_path.is_file():
            processor_config = json.loads(processor_config_path.read_text(encoding="utf-8"))
    except Exception:
        processor_config = {}

    candidate_resolutions = processor_config.get("candidate_resolutions") or []
    first_resolution = candidate_resolutions[0] if isinstance(candidate_resolutions, list) and candidate_resolutions else None
    base_size = 1024
    if isinstance(first_resolution, (list, tuple)) and first_resolution:
        try:
            base_size = max(int(first_resolution[0]), int(first_resolution[-1]))
        except Exception:
            base_size = 1024

    setattr(input_adapter, "_business_deepseek_base_size", int(base_size))
    setattr(input_adapter, "_business_deepseek_image_size", int(min(base_size, 640)))
    setattr(input_adapter, "_business_deepseek_patch_size", int(processor_config.get("patch_size") or 16))
    setattr(input_adapter, "_business_deepseek_downsample_ratio", int(processor_config.get("downsample_ratio") or 4))
    setattr(input_adapter, "_business_deepseek_crop_mode", True)
    setattr(input_adapter, "_business_image_token", str(processor_config.get("image_token") or "<image>").strip() or "<image>")


def _load_vlm_stack(config: dict[str, Any], scenario: str, target_device: torch.device, load_context: dict[str, Any]):
    _apply_torch_linspace_meta_compatibility_shim()
    _apply_torch_module_tied_weights_compatibility_shim()
    import transformers as transformers_module
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, CLIPImageProcessor, LlavaForConditionalGeneration, LlavaProcessor

    try:
        from transformers import AutoModelForVision2Seq
    except ImportError:
        AutoModelForVision2Seq = None

    model_id = str(config.get("model_id") or "").strip()
    trust_remote_code = _as_bool(config.get("trust_remote_code"), default=False)
    torch_dtype = _get_torch_dtype(config)
    model_source, input_source, load_context = _resolve_model_sources(model_id, config, scenario, "processor")
    model_source_path = Path(str(model_source))
    input_source_path = Path(str(input_source))
    model_source_is_local = model_source_path.exists()
    input_source_is_local = input_source_path.exists()
    config_load_kwargs = {
        "cache_dir": str(CACHE_DIR),
        "trust_remote_code": trust_remote_code,
    }
    if model_source_is_local:
        config_load_kwargs["local_files_only"] = True
    model_config = AutoConfig.from_pretrained(model_source, **config_load_kwargs)

    load_kwargs = {
        "cache_dir": str(CACHE_DIR),
        "trust_remote_code": trust_remote_code,
    }
    vlm_family = str(getattr(model_config, "model_type", "") or "").strip().lower()
    if torch_dtype != "auto":
        load_kwargs["torch_dtype"] = torch_dtype
    if vlm_family == "openvla" or _looks_like_openvla_request(config, model_id, model_config=model_config):
        # OpenVLA custom attention path is known to require eager attention to
        # avoid SDPA compatibility failures during weight loading/inference.
        load_kwargs["attn_implementation"] = "eager"
    uses_device_map = target_device.type in {"cuda", "npu"} and _accelerate_available()
    if uses_device_map:
        load_kwargs["device_map"] = "auto"
    direct_model_loader = _resolve_known_vlm_direct_loader(transformers_module, vlm_family)
    uses_moondream_causal_loader = _is_moondream_vlm_family(vlm_family, model_config=model_config)
    default_image_token = "<|image|>" if vlm_family == "glm_ocr" else "<image>"
    image_token = str(getattr(model_config, "image_token", "") or config.get("vlm_image_token") or default_image_token).strip() or default_image_token

    if vlm_family == "llava":
        tokenizer_load_kwargs = {
            "cache_dir": str(CACHE_DIR),
            "trust_remote_code": trust_remote_code,
        }
        if input_source_is_local:
            tokenizer_load_kwargs["local_files_only"] = True
        tokenizer = AutoTokenizer.from_pretrained(input_source, **tokenizer_load_kwargs)
        image_token_id = _ensure_vlm_image_token(tokenizer, image_token=image_token)
        legacy_snapshot_dir = _resolve_llava_snapshot_dir(model_source, model_id)
        use_legacy_loader = legacy_snapshot_dir is not None and _is_legacy_llava_checkpoint(legacy_snapshot_dir, model_config)
        if use_legacy_loader:
            model_config = _build_legacy_llava_model_config(model_config, legacy_snapshot_dir, image_token_id=image_token_id)
        image_processor_source = str(getattr(model_config, "mm_vision_tower", "") or config.get("vision_tower") or "openai/clip-vit-large-patch14")
        resolved_image_processor_source = _resolve_local_snapshot_source(image_processor_source, input_kind="image_processor") or image_processor_source
        image_processor_load_kwargs = {"cache_dir": str(CACHE_DIR)}
        if Path(str(resolved_image_processor_source)).exists():
            image_processor_load_kwargs["local_files_only"] = True
        image_processor = CLIPImageProcessor.from_pretrained(resolved_image_processor_source, **image_processor_load_kwargs)
        image_processor_size = _configure_llava_image_processor(image_processor, model_config)
        input_adapter = LlavaProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            patch_size=int(getattr(getattr(model_config, "vision_config", None), "patch_size", 14) or 14),
            vision_feature_select_strategy=str(getattr(model_config, "vision_feature_select_strategy", "default") or "default"),
            image_token=image_token,
            num_additional_image_tokens=int(config.get("vlm_num_additional_image_tokens") or 1),
        )

        if use_legacy_loader:
            if legacy_snapshot_dir is None:
                raise RuntimeError("legacy Llava checkpoint 缺少本地 snapshot，无法执行兼容加载")
            model, compat_context = _load_legacy_llava_model(
                legacy_snapshot_dir,
                model_config,
                target_device=target_device,
                requested_torch_dtype=torch_dtype,
            )
            load_context.update(compat_context)
            load_context["model_source"] = str(legacy_snapshot_dir)
            load_context["model_source_kind"] = "legacy_local_snapshot"
        else:
            candidate_sources = [model_source]
            if not model_source_is_local and model_source != model_id:
                candidate_sources.append(model_id)

            model = None
            last_error = None
            for index, candidate_source in enumerate(candidate_sources):
                current_kwargs = dict(load_kwargs)
                candidate_source_is_local = Path(str(candidate_source)).exists()
                if candidate_source_is_local:
                    current_kwargs["local_files_only"] = True
                if index > 0:
                    current_kwargs.pop("device_map", None)
                try:
                    model = LlavaForConditionalGeneration.from_pretrained(candidate_source, **current_kwargs)
                    if "device_map" not in current_kwargs:
                        model = model.to(target_device)
                    if candidate_source != model_source:
                        load_context["model_source"] = candidate_source
                        load_context["model_source_kind"] = "hub_fallback"
                    break
                except Exception as exc:
                    last_error = exc
            if model is None:
                raise RuntimeError(f"加载 VLM 模型失败: source={model_source}, error={last_error}") from last_error

        _resize_token_embeddings_if_needed(model, input_adapter.tokenizer)
        model.config.image_token_index = image_token_id
        try:
            model.config.image_token_id = image_token_id
        except Exception:
            pass
        input_adapter._business_vlm_family = "llava"
        input_adapter._business_image_token = image_token
        load_context["image_processor_source"] = resolved_image_processor_source
        load_context["image_processor_size"] = image_processor_size
    else:
        if AutoModelForVision2Seq is None and direct_model_loader is None and not uses_moondream_causal_loader:
            raise RuntimeError(f"当前 transformers 版本不支持 AutoModelForVision2Seq，且该 VLM 未命中已知直连加载路径: model_type={vlm_family or '<unknown>'}")
        if vlm_family in {"qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe"}:
            input_adapter = _load_qwen_vl_processor(input_source, trust_remote_code=trust_remote_code, vlm_family=vlm_family)
        elif vlm_family == "qwen2_5_omni":
            input_adapter = _load_qwen_omni_processor(input_source, trust_remote_code=trust_remote_code)
        elif vlm_family == "glm_ocr":
            from transformers import AutoTokenizer, Glm46VImageProcessor, Glm46VProcessor

            source_path = Path(str(input_source))
            local_files_only = source_path.exists()
            input_tokenizer = AutoTokenizer.from_pretrained(
                input_source,
                cache_dir=str(CACHE_DIR),
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
            image_processor = Glm46VImageProcessor.from_pretrained(
                input_source,
                cache_dir=str(CACHE_DIR),
                local_files_only=local_files_only,
            )
            input_adapter = Glm46VProcessor(
                image_processor=image_processor,
                tokenizer=input_tokenizer,
                video_processor=_build_qwen_image_only_video_processor(transformers_module),
                chat_template=getattr(input_tokenizer, "chat_template", None),
            )
        elif vlm_family == "internvl_chat":
            tokenizer_load_kwargs = {
                "cache_dir": str(CACHE_DIR),
                "trust_remote_code": trust_remote_code,
            }
            if input_source_is_local:
                tokenizer_load_kwargs["local_files_only"] = True
            input_tokenizer = AutoTokenizer.from_pretrained(input_source, **tokenizer_load_kwargs)
            image_processor_source = _resolve_local_snapshot_source(input_source, input_kind="image_processor") or input_source
            image_processor_load_kwargs = {"cache_dir": str(CACHE_DIR)}
            if Path(str(image_processor_source)).exists():
                image_processor_load_kwargs["local_files_only"] = True
            image_processor = CLIPImageProcessor.from_pretrained(image_processor_source, **image_processor_load_kwargs)
            input_adapter = SimpleNamespace(
                tokenizer=input_tokenizer,
                image_processor=image_processor,
            )
            load_context["image_processor_source"] = str(image_processor_source)
        elif uses_moondream_causal_loader:
            tokenizer_load_kwargs = {
                "cache_dir": str(CACHE_DIR),
                "trust_remote_code": trust_remote_code,
            }
            if input_source_is_local:
                tokenizer_load_kwargs["local_files_only"] = True
            input_adapter = SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(input_source, **tokenizer_load_kwargs))
            load_context["input_source_kind"] = "tokenizer_only_moondream_vlm"
        else:
            processor_load_kwargs = {
                "cache_dir": str(CACHE_DIR),
                "trust_remote_code": trust_remote_code,
            }
            if input_source_is_local:
                processor_load_kwargs["local_files_only"] = True
            input_adapter = AutoProcessor.from_pretrained(input_source, **processor_load_kwargs)
            if vlm_family == "deepseek_vl_v2":
                _configure_deepseek_vl_v2_input_adapter(input_adapter, input_source)

        candidate_sources = [model_source]
        if not model_source_is_local and model_source != model_id:
            candidate_sources.append(model_id)

        model = None
        last_error = None
        for index, candidate_source in enumerate(candidate_sources):
            current_kwargs = dict(load_kwargs)
            candidate_source_is_local = Path(str(candidate_source)).exists()
            if candidate_source_is_local:
                current_kwargs["local_files_only"] = True
            if index > 0:
                current_kwargs.pop("device_map", None)
            if _should_disable_vlm_device_map_auto(vlm_family):
                current_kwargs.pop("device_map", None)
            if _should_disable_vlm_low_cpu_mem_usage(vlm_family):
                current_kwargs["low_cpu_mem_usage"] = False
            try:
                if direct_model_loader is not None:
                    model = direct_model_loader.from_pretrained(candidate_source, **current_kwargs)
                elif uses_moondream_causal_loader:
                    model = AutoModelForCausalLM.from_pretrained(candidate_source, **current_kwargs)
                else:
                    model = AutoModelForVision2Seq.from_pretrained(candidate_source, **current_kwargs)
                if "device_map" not in current_kwargs:
                    model = model.to(target_device)
                if candidate_source != model_source:
                    load_context["model_source"] = candidate_source
                    load_context["model_source_kind"] = "hub_fallback"
                break
            except Exception as exc:
                last_error = exc
        if model is None:
            raise RuntimeError(f"加载 VLM 模型失败: source={model_source}, error={last_error}") from last_error

        if hasattr(input_adapter, "tokenizer") and getattr(input_adapter, "tokenizer", None) is not None:
            _ensure_vlm_image_token(input_adapter.tokenizer, image_token=image_token)
            _resize_token_embeddings_if_needed(model, input_adapter.tokenizer)
        input_adapter._business_vlm_family = vlm_family

    load_context["runtime_compatibility_shims"] = _apply_model_runtime_compatibility_shims(model)
    tokenizer_candidate = getattr(input_adapter, "tokenizer", None)
    if tokenizer_candidate is None:
        tokenizer_candidate = input_adapter
    load_context["tokenizer_compatibility_shims"] = _apply_tokenizer_runtime_compatibility_shims(tokenizer_candidate)
    patch_metadata = load_context.get("_patch_metadata")
    if isinstance(patch_metadata, dict):
        _apply_deferred_model_files_hooks(model, patch_metadata)
        load_context["patch_load_status"] = str(patch_metadata.get("status") or load_context.get("patch_load_status") or "disabled")
        load_context["patch_modules"] = list(patch_metadata.get("imported_modules") or [])
        load_context["patch_hooks"] = list(patch_metadata.get("called_hooks") or [])
        load_context["patch_errors"] = list(patch_metadata.get("errors") or [])
    load_context["activation_name_shims"] = _stabilize_activation_name_shims(model)
    model.eval()
    return model, input_adapter, model_config, "vlm", load_context


def _should_disable_vlm_device_map_auto(vlm_family: str) -> bool:
    return str(vlm_family or "").strip().lower() in {"internvl_chat", "moondream1"}


def _should_disable_vlm_low_cpu_mem_usage(vlm_family: str) -> bool:
    return str(vlm_family or "").strip().lower() in {"internvl_chat"}


def _looks_like_qwen_vlm_reranker(model_id: str, config: dict[str, Any], *, model_config: Any | None = None) -> bool:
    if str(config.get("model_type") or "").strip() != "reranker":
        return False

    backend = str(config.get("model_backend") or "").strip().lower()
    if backend in {"qwen_vl_reranker", "qwen3_vl_reranker"}:
        return True

    supported_vlm_families = {"qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe"}
    config_model_type = str(getattr(model_config, "model_type", "") or "").strip().lower()
    if config_model_type in supported_vlm_families:
        return True

    signal_parts: list[str] = []
    for raw_value in (
        model_id,
        config.get("architectures"),
        config.get("model_class"),
        config.get("output_type_hint"),
        config.get("business_intent"),
        getattr(model_config, "architectures", None) if model_config is not None else None,
    ):
        if isinstance(raw_value, (list, tuple, set)):
            signal_parts.extend(str(item or "").strip().lower() for item in raw_value if str(item or "").strip())
        elif raw_value:
            signal_parts.append(str(raw_value).strip().lower())
    signal_text = " ".join(signal_parts)
    return "vl-reranker" in signal_text or "qwen3-vl-reranker" in signal_text or ("reranker" in signal_text and "qwen3vl" in signal_text)


def _looks_like_qwen3_guard_gen(model_id: str, config: dict[str, Any], *, model_config: Any | None = None) -> bool:
    if str(config.get("model_type") or "").strip() != "causal_lm":
        return False

    backend = str(config.get("model_backend") or "").strip().lower()
    if backend in {"qwen3_guard_gen", "qwen3guard_gen"}:
        return True

    signal_parts: list[str] = []
    for raw_value in (
        model_id,
        config.get("architectures"),
        config.get("model_class"),
        config.get("business_intent"),
        getattr(model_config, "architectures", None) if model_config is not None else None,
        getattr(model_config, "model_type", None) if model_config is not None else None,
    ):
        if isinstance(raw_value, (list, tuple, set)):
            signal_parts.extend(str(item or "").strip().lower() for item in raw_value if str(item or "").strip())
        elif raw_value:
            signal_parts.append(str(raw_value).strip().lower())
    signal_text = " ".join(signal_parts)
    return "guard-gen" in signal_text or "qwen3guard" in signal_text


def _resolve_single_token_id(tokenizer, *candidate_tokens: str) -> int:
    vocab = {}
    try:
        vocab = dict(tokenizer.get_vocab() or {})
    except Exception:
        vocab = {}

    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    for candidate_token in candidate_tokens:
        if not candidate_token:
            continue
        token_id = vocab.get(candidate_token)
        if isinstance(token_id, int) and token_id >= 0:
            return token_id
        try:
            token_id = tokenizer.convert_tokens_to_ids(candidate_token)
        except Exception:
            token_id = None
        if isinstance(token_id, int) and token_id >= 0 and token_id != unk_token_id:
            return token_id
        try:
            encoded = tokenizer.encode(candidate_token, add_special_tokens=False)
        except Exception:
            encoded = []
        if isinstance(encoded, list) and len(encoded) == 1 and isinstance(encoded[0], int):
            if encoded[0] >= 0 and encoded[0] != unk_token_id:
                return int(encoded[0])
    raise RuntimeError(f"无法在 tokenizer 中定位单 token 词元: candidates={candidate_tokens}")


def _build_qwen_vlm_reranker_score_head(model, processor) -> tuple[torch.nn.Module, int, int]:
    tokenizer = getattr(processor, "tokenizer", processor)
    token_yes_id = _resolve_single_token_id(tokenizer, "yes", "Yes", " yes", " Yes")
    token_no_id = _resolve_single_token_id(tokenizer, "no", "No", " no", " No")
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None or not hasattr(lm_head, "weight"):
        raise RuntimeError("Qwen VLM reranker 缺少 lm_head.weight，无法构造 yes/no 评分头")
    weight = lm_head.weight.detach()
    hidden_size = int(weight.shape[-1])
    score_head = torch.nn.Linear(hidden_size, 1, bias=False)
    with torch.no_grad():
        score_head.weight[0].copy_(weight[token_yes_id] - weight[token_no_id])
    score_head = score_head.to(device=weight.device, dtype=weight.dtype)
    score_head.eval()
    return score_head, token_yes_id, token_no_id


def _load_qwen_vlm_reranker_stack(config: dict[str, Any], scenario: str, target_device: torch.device, load_context: dict[str, Any]):
    model_id = str(config.get("model_id") or "").strip()
    if not model_id or str(config.get("model_type") or "").strip() != "reranker":
        return None

    from transformers import AutoConfig

    trust_remote_code = _as_bool(config.get("trust_remote_code"), default=False)
    model_source, _, load_context = _resolve_model_sources(model_id, config, scenario, "processor")
    model_config = AutoConfig.from_pretrained(model_source, cache_dir=str(CACHE_DIR), trust_remote_code=trust_remote_code)
    _apply_model_config_compatibility_fixes(model_config, model_id=model_id)
    if not _looks_like_qwen_vlm_reranker(model_id, config, model_config=model_config):
        return None

    vlm_config = dict(config)
    vlm_config["model_type"] = "vlm"
    model, input_adapter, model_config, _, load_context = _load_vlm_stack(vlm_config, scenario, target_device, load_context)

    tokenizer = getattr(input_adapter, "tokenizer", None)
    if tokenizer is not None:
        try:
            tokenizer.padding_side = "left"
        except Exception:
            pass
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token

    score_head, token_yes_id, token_no_id = _build_qwen_vlm_reranker_score_head(model, input_adapter)
    model._slai_qwen_vlm_reranker_enabled = True
    model._slai_qwen_vlm_reranker_score_head = score_head
    model._slai_qwen_vlm_reranker_backbone = getattr(model, "model", model)
    model._slai_qwen_vlm_reranker_yes_token_id = int(token_yes_id)
    model._slai_qwen_vlm_reranker_no_token_id = int(token_no_id)
    model._slai_qwen_vlm_reranker_instruction = str(config.get("reranker_instruction") or QWEN_VLM_RERANKER_DEFAULT_INSTRUCTION)
    model._slai_qwen_vlm_reranker_max_length = int(config.get("reranker_max_length") or 8192)

    load_context["reranker_backend"] = "qwen_vlm_yes_no"
    load_context["reranker_score_tokens"] = {"yes": int(token_yes_id), "no": int(token_no_id)}
    return model, input_adapter, model_config, "reranker", load_context


def _load_model_stack(config: dict, scenario: str):
    model_id = str(config.get("model_id") or "").strip()
    model_type = str(config.get("model_type") or "").strip()
    if not model_id or not model_type:
        raise RuntimeError("business_benchmark_config.json 缺少 model_id 或 model_type")

    target_device = _get_device_for_scenario(scenario)
    load_context = {
        "model_source": model_id,
        "model_source_kind": "hub",
        "input_source": model_id,
        "input_source_kind": "hub",
        "used_model_files": False,
        "patch_load_status": "disabled",
        "patch_modules": [],
        "patch_hooks": [],
        "patch_errors": [],
        "_patch_metadata": {
            "status": "disabled",
            "imported_modules": [],
            "called_hooks": [],
            "errors": [],
        },
    }
    _apply_model_load_seed(config)

    f5_tts_stack = _load_f5_tts_stack(config, model_id, scenario, target_device, load_context)
    if f5_tts_stack is not None:
        return f5_tts_stack

    qwen_tts_stack = _load_qwen_tts_stack(config, model_id, scenario, target_device, load_context)
    if qwen_tts_stack is not None:
        return qwen_tts_stack

    qwen_forced_aligner_stack = _load_qwen_forced_aligner_stack(config, model_id, scenario, target_device, load_context)
    if qwen_forced_aligner_stack is not None:
        return qwen_forced_aligner_stack

    qwen_asr_stack = _load_qwen_asr_stack(config, model_id, scenario, target_device, load_context)
    if qwen_asr_stack is not None:
        return qwen_asr_stack

    nemo_asr_stack = _load_nemo_asr_stack(config, model_id, scenario, target_device, load_context)
    if nemo_asr_stack is not None:
        return nemo_asr_stack

    ultralytics_stack = _load_ultralytics_stack(config, model_id, target_device, load_context)
    if ultralytics_stack is not None:
        return ultralytics_stack

    vocos_stack = _load_vocos_stack(config, model_id, scenario, target_device, load_context)
    if vocos_stack is not None:
        return vocos_stack

    timm_stack = _load_timm_stack(config, model_id, scenario, target_device, load_context)
    if timm_stack is not None:
        return timm_stack

    segment_anything_stack = _load_segment_anything_stack(config, model_id, target_device, load_context)
    if segment_anything_stack is not None:
        return segment_anything_stack

    birefnet_stack = _load_birefnet_image_matting_stack(config, model_id, scenario, target_device, load_context)
    if birefnet_stack is not None:
        return birefnet_stack

    clipseg_stack = _load_clipseg_stack(config, model_id, scenario, target_device, load_context)
    if clipseg_stack is not None:
        return clipseg_stack

    if model_type == "timeseries":
        model_source, _timeseries_input_source, load_context = _resolve_model_sources(model_id, config, scenario, "processor")
        timeseries_stack = _load_timeseries_stack(config, model_id, model_source, target_device, load_context)
        if timeseries_stack is not None:
            return timeseries_stack

    if model_type == "vlm":
        return _load_vlm_stack(config, scenario, target_device, load_context)

    qwen_vlm_reranker_stack = _load_qwen_vlm_reranker_stack(config, scenario, target_device, load_context)
    if qwen_vlm_reranker_stack is not None:
        return qwen_vlm_reranker_stack

    colbert_stack = _load_colbert_stack(config, model_id, scenario, target_device, load_context)
    if colbert_stack is not None:
        return colbert_stack

    gliner_stack = _load_gliner_stack(config, model_id, scenario, target_device, load_context)
    if gliner_stack is not None:
        return gliner_stack

    span_marker_stack = _load_span_marker_stack(config, model_id, scenario, target_device, load_context)
    if span_marker_stack is not None:
        return span_marker_stack

    accuracy_latency_bridge_stack = _load_accuracy_latency_bridge_stack(config, model_id, scenario, target_device, load_context)
    if accuracy_latency_bridge_stack is not None:
        return accuracy_latency_bridge_stack

    local_embedding_bridge_stack = _load_local_embedding_bridge_stack(config, model_id, scenario, target_device, load_context)
    if local_embedding_bridge_stack is not None:
        return local_embedding_bridge_stack

    local_vision_bridge_stack = _load_local_vision_classification_bridge_stack(config, model_id, scenario, target_device, load_context)
    if local_vision_bridge_stack is not None:
        return local_vision_bridge_stack

    AutoConfig, input_loader_cls, model_cls, input_kind = _load_transformers_components(model_type)
    model_source, tokenizer_source, load_context = _resolve_model_sources(model_id, config, scenario, input_kind)
    trust_remote_code = _as_bool(config.get("trust_remote_code"), default=False)
    torch_dtype = _get_torch_dtype(config)
    peft_spec = _resolve_peft_adapter_spec(config, model_id, model_source)
    if peft_spec is not None:
        load_context["adapter_source"] = peft_spec["adapter_source"]
        load_context["adapter_source_kind"] = peft_spec["adapter_source_kind"]
        load_context["peft_type"] = peft_spec["peft_type"]
        load_context["raw_base_model_name_or_path"] = peft_spec["raw_base_model_name_or_path"]
        load_context["base_model_id"] = peft_spec["base_model_id"]
        load_context["base_model_source"] = peft_spec["base_model_source"]
        load_context["base_model_source_kind"] = peft_spec["base_model_source_kind"]
        model_source = str(peft_spec["base_model_source"])
        load_context["model_source"] = model_source
        load_context["model_source_kind"] = str(peft_spec["base_model_source_kind"] or load_context.get("model_source_kind") or "hub")
    custom_module_entries = tuple(
        _iter_local_custom_module_entries(
            model_source,
            tokenizer_source,
            None if peft_spec is None else str(peft_spec["adapter_source"]),
        )
    )
    generator_source_override = _resolve_masked_lm_generator_source(model_source, model_id, model_type)

    open_clip_stack = _load_open_clip_stack(config, model_id, model_source, target_device, load_context)
    if open_clip_stack is not None:
        return open_clip_stack

    siglip_stack = _load_siglip_stack(config, model_id, model_source, tokenizer_source, target_device, load_context)
    if siglip_stack is not None:
        return siglip_stack

    input_loader_kwargs = {
        "cache_dir": str(CACHE_DIR),
        "trust_remote_code": trust_remote_code,
    }
    effective_trust_remote_code = trust_remote_code
    if input_kind == "processor":
        _cleanup_asr_lm_sidecar_files(tokenizer_source)
    try:
        with _temporary_sys_path_entries(*custom_module_entries):
            input_adapter = input_loader_cls.from_pretrained(tokenizer_source, **input_loader_kwargs)
    except Exception as exc:
        retry_exc = exc
        if effective_trust_remote_code and _should_retry_without_trust_remote_code(exc):
            effective_trust_remote_code = False
            input_loader_kwargs["trust_remote_code"] = False
            try:
                with _temporary_sys_path_entries(*custom_module_entries):
                    input_adapter = input_loader_cls.from_pretrained(tokenizer_source, **input_loader_kwargs)
            except Exception as second_exc:
                retry_exc = second_exc
            else:
                retry_exc = None
        if retry_exc is not None:
            if input_kind == "image_processor":
                input_adapter = _load_image_processor_from_local_legacy_fallback(
                    tokenizer_source,
                    trust_remote_code=effective_trust_remote_code,
                )
                if input_adapter is None:
                    raise retry_exc
            else:
                if input_kind == "processor" and _is_optional_asr_lm_dependency_error(retry_exc):
                    input_adapter = _load_asr_processor_without_optional_lm(
                        tokenizer_source,
                        trust_remote_code=effective_trust_remote_code,
                    )
                    if input_adapter is None:
                        raise retry_exc
                elif input_kind == "processor" and model_type == "audio_embedding":
                    try:
                        from transformers import AutoFeatureExtractor

                        input_adapter = AutoFeatureExtractor.from_pretrained(tokenizer_source, **input_loader_kwargs)
                    except Exception:
                        raise retry_exc
                else:
                    if input_kind != "tokenizer":
                        raise retry_exc
                    # Some tokenizers need a slow-tokenizer fallback when fast assets or
                    # optional conversion deps (protobuf / sentencepiece / tiktoken) are absent.
                    try:
                        with _temporary_sys_path_entries(*custom_module_entries):
                            input_adapter = input_loader_cls.from_pretrained(tokenizer_source, use_fast=False, **input_loader_kwargs)
                    except Exception:
                        input_adapter = _load_tokenizer_from_accuracy_run_source_fallback(
                            trust_remote_code=effective_trust_remote_code,
                        )
                        if input_adapter is None:
                            input_adapter = _load_tokenizer_from_local_vocab_fallback(tokenizer_source)
                        if input_adapter is None:
                            input_adapter = _load_custom_tokenizer_from_accuracy_run_fallback(tokenizer_source)
                        if input_adapter is None:
                            raise retry_exc
    if input_kind == "processor" and model_type == "vision_text_ocr":
        adapter_cls_name = type(input_adapter).__name__
        if "Processor" not in adapter_cls_name:
            try:
                from transformers import TrOCRProcessor

                input_adapter = TrOCRProcessor.from_pretrained(tokenizer_source, **input_loader_kwargs)
            except Exception:
                pass
    config_source = generator_source_override[1] if generator_source_override is not None else model_source
    with _temporary_sys_path_entries(*custom_module_entries):
        model_config, effective_trust_remote_code = _load_model_config_with_known_fallbacks(
            AutoConfig,
            config_source,
            model_id=model_id,
            trust_remote_code=effective_trust_remote_code,
        )
    _apply_model_config_compatibility_fixes(model_config, model_id=model_id)
    model_cls, legacy_model_loader = _resolve_legacy_causal_lm_model_class(model_cls, model_config, model_id=model_id)
    if legacy_model_loader:
        load_context["legacy_model_loader"] = legacy_model_loader
    encoder_checkpoint_override = str(config.get("encoder_checkpoint_override") or "").strip()
    if encoder_checkpoint_override and hasattr(model_config, "encoder_checkpoint"):
        resolved_encoder_source, resolved_encoder_source_kind = _resolve_configured_source(
            encoder_checkpoint_override,
            require_model_assets=True,
        )
        if resolved_encoder_source:
            model_config.encoder_checkpoint = resolved_encoder_source
            load_context["encoder_checkpoint_source"] = resolved_encoder_source
            load_context["encoder_checkpoint_source_kind"] = resolved_encoder_source_kind or (
                "local_snapshot" if Path(str(resolved_encoder_source)).exists() else "hub"
            )
    generic_model_family = str(getattr(model_config, "model_type", "") or "").strip().lower()
    eager_attn_required = generic_model_family in {"florence2", "phi3"} or any(
        token in str(model_id or "").strip().lower() for token in ("florence", "phi-3", "phi3")
    )
    strip_quantization_config = target_device.type == "npu" and getattr(model_config, "quantization_config", None) is not None
    if strip_quantization_config:
        try:
            delattr(model_config, "quantization_config")
        except Exception:
            setattr(model_config, "quantization_config", None)
    visible_accelerator_devices = [token.strip() for token in str(os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or "").split(",") if token.strip()]
    force_single_device_load = model_type == "vision_text_ocr" or (target_device.type == "npu" and len(visible_accelerator_devices) == 1)
    uses_device_map = target_device.type in {"cuda", "npu"} and _accelerate_available() and not force_single_device_load

    load_kwargs = {
        "cache_dir": str(CACHE_DIR),
        "trust_remote_code": effective_trust_remote_code,
        "config": model_config,
    }
    if torch_dtype != "auto":
        load_kwargs["torch_dtype"] = torch_dtype
    if eager_attn_required:
        load_kwargs["attn_implementation"] = "eager"
    if strip_quantization_config:
        load_kwargs["quantization_config"] = None
        load_kwargs["ignore_mismatched_sizes"] = True
    if force_single_device_load:
        load_kwargs["low_cpu_mem_usage"] = False
        if model_type == "vision_text_ocr":
            load_context["single_device_load_reason"] = "vision_text_ocr_meta_tensor_compatibility"
        elif target_device.type == "npu" and len(visible_accelerator_devices) == 1:
            load_context["single_device_load_reason"] = "single_visible_npu_device"
    if uses_device_map:
        load_kwargs["device_map"] = "auto"
    if legacy_model_loader:
        load_kwargs.pop("config", None)
        load_kwargs.pop("trust_remote_code", None)
        load_kwargs.pop("device_map", None)

    def _maybe_attach_peft_adapter(base_model):
        if peft_spec is None:
            return base_model
        try:
            from peft import PeftModel
        except Exception as exc:
            raise RuntimeError("检测到 PEFT/LoRA adapter，但当前业务测评环境缺少 peft 依赖。") from exc
        with _temporary_sys_path_entries(*custom_module_entries):
            return PeftModel.from_pretrained(
                base_model,
                str(peft_spec["adapter_source"]),
                cache_dir=str(CACHE_DIR),
            )

    def _load_from_config_with_local_state_dict(model_class, source: str):
        _apply_model_load_seed(config)
        source_path = Path(str(source))
        state_dict = _load_state_dict_from_snapshot(source_path)
        with _temporary_sys_path_entries(*custom_module_entries):
            with _temporary_transformers_tied_weights_loading_compatibility():
                from_config_kwargs = {"trust_remote_code": effective_trust_remote_code}
                if eager_attn_required:
                    from_config_kwargs["attn_implementation"] = "eager"
                model = model_class.from_config(model_config, **from_config_kwargs)
        force_manual_deberta_generator_load = _should_force_manual_deberta_generator_load(source, model_id=model_id, model_type=model_type, model_config=model_config)
        if force_manual_deberta_generator_load:
            state_dict = _remap_deberta_generator_masked_lm_state_dict(state_dict)
            load_context["masked_lm_loader"] = "manual_deberta_generator_remap"
            load_context["masked_lm_generator_source"] = str(source_path)
        _ensure_model_has_all_tied_weights_keys(model)
        model.load_state_dict(state_dict, strict=False)
        if force_manual_deberta_generator_load and hasattr(model, "tie_weights"):
            try:
                model.tie_weights()
            except Exception:
                pass
        del state_dict
        move_kwargs: dict[str, Any] = {"device": target_device}
        if torch_dtype != "auto":
            move_kwargs["dtype"] = torch_dtype
        model = model.to(**move_kwargs)
        return _maybe_attach_peft_adapter(model)

    def _load_from_source(source: str):
        _apply_model_load_seed(config)
        force_manual_deberta_generator_load = _should_force_manual_deberta_generator_load(source, model_id=model_id, model_type=model_type, model_config=model_config)
        if isinstance(model_cls, tuple):
            last_error = None
            for candidate in model_cls:
                try:
                    if force_manual_deberta_generator_load:
                        return _load_from_config_with_local_state_dict(candidate, source)
                    with _temporary_sys_path_entries(*custom_module_entries):
                        with _temporary_transformers_tied_weights_loading_compatibility():
                            model = candidate.from_pretrained(source, **load_kwargs)
                    _ensure_model_has_all_tied_weights_keys(model)
                    return _maybe_attach_peft_adapter(model)
                except Exception as exc:
                    if force_manual_deberta_generator_load:
                        last_error = exc
                        continue
                    if _should_retry_with_manual_state_dict_load(exc, source):
                        return _load_from_config_with_local_state_dict(candidate, source)
                    last_error = exc
            if last_error is not None:
                raise last_error
            raise RuntimeError(f"没有可用模型类加载 {source}")
        try:
            if force_manual_deberta_generator_load:
                return _load_from_config_with_local_state_dict(model_cls, source)
            with _temporary_sys_path_entries(*custom_module_entries):
                with _temporary_transformers_tied_weights_loading_compatibility():
                    model = model_cls.from_pretrained(source, **load_kwargs)
            _ensure_model_has_all_tied_weights_keys(model)
            return _maybe_attach_peft_adapter(model)
        except Exception as exc:
            if force_manual_deberta_generator_load:
                raise
            if _should_retry_with_manual_state_dict_load(exc, source):
                return _load_from_config_with_local_state_dict(model_cls, source)
            raise

    def _ensure_model_target_device(current_model):
        if target_device.type not in {"cuda", "npu"}:
            return current_model
        first_parameter = _get_first_parameter(current_model)
        if first_parameter is not None and first_parameter.device.type != "cpu":
            return current_model
        hf_device_map = getattr(current_model, "hf_device_map", None)
        if isinstance(hf_device_map, Mapping):
            has_accelerator_shards = any(any(token in str(location).strip().lower() for token in ("cuda", "npu", "xpu", "mps")) for location in hf_device_map.values())
            if has_accelerator_shards:
                return current_model
        move_kwargs: dict[str, Any] = {"device": target_device}
        if torch_dtype != "auto":
            move_kwargs["dtype"] = torch_dtype
        try:
            return current_model.to(**move_kwargs)
        except TypeError:
            return current_model.to(target_device)

    candidate_sources = []
    if generator_source_override is not None:
        candidate_sources.append(generator_source_override[0])
    if model_source not in candidate_sources:
        candidate_sources.append(model_source)
    fallback_model_id = str(peft_spec["base_model_id"]) if peft_spec is not None else model_id
    if peft_spec is None and model_source != model_id and model_id not in candidate_sources:
        candidate_sources.append(model_id)
    if peft_spec is not None and fallback_model_id and fallback_model_id not in candidate_sources:
        candidate_sources.append(fallback_model_id)
    for candidate in _fallback_model_source_candidates(fallback_model_id or model_id, model_type):
        if candidate not in candidate_sources:
            candidate_sources.append(candidate)

    model = None
    last_error = None
    for index, candidate_source in enumerate(candidate_sources):
        if index > 0:
            load_kwargs.pop("device_map", None)
        try:
            model = _load_from_source(candidate_source)
            if legacy_model_loader:
                move_kwargs: dict[str, Any] = {"device": target_device}
                if torch_dtype != "auto":
                    move_kwargs["dtype"] = torch_dtype
                model = model.to(**move_kwargs)
            elif not uses_device_map or index > 0:
                model = model.to(target_device)
            else:
                model = _ensure_model_target_device(model)
            if generator_source_override is not None and candidate_source == generator_source_override[0]:
                load_context["model_source"] = candidate_source
                load_context["model_source_kind"] = "local_generator_checkpoint"
                load_context["generator_config_source"] = config_source
            elif index > 0:
                load_context["model_source"] = candidate_source
                load_context["model_source_kind"] = "hub_fallback"
            break
        except Exception as exc:
            last_error = exc
    if model is None:
        fallback_text = ", ".join(candidate_sources[1:]) or model_source
        raise RuntimeError(f"加载业务测评模型失败: source={model_source}, fallback={fallback_text}, error={last_error}") from last_error

    load_context["runtime_compatibility_shims"] = _apply_model_runtime_compatibility_shims(model)
    patch_metadata = load_context.get("_patch_metadata")
    if isinstance(patch_metadata, dict):
        _apply_deferred_model_files_hooks(model, patch_metadata)
        load_context["patch_load_status"] = str(patch_metadata.get("status") or load_context.get("patch_load_status") or "disabled")
        load_context["patch_modules"] = list(patch_metadata.get("imported_modules") or [])
        load_context["patch_hooks"] = list(patch_metadata.get("called_hooks") or [])
        load_context["patch_errors"] = list(patch_metadata.get("errors") or [])
    model = _ensure_model_target_device(model)
    load_context["activation_name_shims"] = _stabilize_activation_name_shims(model)

    model.eval()
    model = _ensure_model_target_device(model)
    if model_type == "vision_text_ocr":
        try:
            embed_positions = model.decoder.model.decoder.embed_positions
            weights = getattr(embed_positions, "weights", None)
            if torch.is_tensor(weights) and getattr(weights, "is_meta", False):
                padding_idx = getattr(embed_positions, "padding_idx", None)
                embedding_dim = int(getattr(embed_positions, "embedding_dim", weights.shape[-1]))
                num_positions = int(weights.shape[0])
                rebuilt = embed_positions.get_embedding(num_positions, embedding_dim, padding_idx)
                rebuilt = rebuilt.to(device=target_device, dtype=weights.dtype if getattr(weights, "dtype", None) is not None else rebuilt.dtype)
                embed_positions.weights = rebuilt
                load_context["trocr_positional_embedding_fix"] = "materialized_meta_weights"
        except Exception as exc:
            load_context["trocr_positional_embedding_fix"] = f"failed:{exc}"
    if input_kind == "tokenizer" and getattr(input_adapter, "pad_token", None) is None and getattr(input_adapter, "eos_token", None) is not None:
        input_adapter.pad_token = input_adapter.eos_token
    return model, input_adapter, model_config, model_type, load_context


def _load_gliner_stack(config: dict[str, Any], model_id: str, scenario: str, target_device: torch.device, load_context: dict[str, Any]):
    requested_model_type = str(config.get("model_type") or "").strip()
    canonical_model_type = "token_classification" if requested_model_type == "biomedical_token_classification" else requested_model_type
    if canonical_model_type != "token_classification":
        return None

    model_backend = str(config.get("model_backend") or "").strip()
    model_source, tokenizer_source, load_context = _resolve_model_sources(model_id, config, scenario, "tokenizer")
    if not _looks_like_gliner_checkpoint(model_source, model_id=model_id, model_backend=model_backend):
        return None

    try:
        from gliner import GLiNER
    except Exception as exc:
        raise RuntimeError("检测到 GLiNER checkpoint，但当前业务测评环境缺少 gliner 依赖。") from exc

    _apply_model_load_seed(config)
    source_path = Path(str(model_source))
    load_kwargs: dict[str, Any] = {
        "map_location": "cpu",
    }
    offline_env: dict[str, str] = {}
    if source_path.exists():
        offline_env["HF_HOME"] = str(CACHE_DIR)
        offline_env["HF_HUB_CACHE"] = str(CACHE_DIR)
        offline_env["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
        load_kwargs["local_files_only"] = True
    else:
        load_kwargs["cache_dir"] = str(CACHE_DIR)

    with _temporary_env_overrides(offline_env):
        model = GLiNER.from_pretrained(model_source, **load_kwargs)

    inner_model = getattr(model, "model", None)
    if inner_model is None:
        raise RuntimeError("GLiNER runtime 缺少内部 torch model，无法迁移到目标设备。")

    torch_dtype = _get_torch_dtype(config)
    move_kwargs: dict[str, Any] = {"device": target_device}
    if torch_dtype != "auto":
        move_kwargs["dtype"] = torch_dtype
    try:
        inner_model = inner_model.to(**move_kwargs)
    except TypeError:
        inner_model = inner_model.to(target_device)
    model.model = inner_model

    model_config = getattr(model, "config", None)
    if model_config is None:
        model_config = SimpleNamespace()
    gliner_config_payload = _load_gliner_config_payload_from_source(model_source)
    if not getattr(model_config, "model_type", None):
        setattr(model_config, "model_type", "gliner")
    if not getattr(model_config, "architectures", None):
        setattr(model_config, "architectures", ["GLiNER"])
    setattr(model_config, "_slai_business_backend", "gliner")
    setattr(model_config, "_slai_raw_config_payload", dict(gliner_config_payload))
    _apply_model_config_compatibility_fixes(model_config, model_id=model_id)

    load_context["runtime_compatibility_shims"] = _apply_model_runtime_compatibility_shims(model)
    patch_metadata = load_context.get("_patch_metadata")
    if isinstance(patch_metadata, dict):
        _apply_deferred_model_files_hooks(model, patch_metadata)
        load_context["patch_load_status"] = str(patch_metadata.get("status") or load_context.get("patch_load_status") or "disabled")
        load_context["patch_modules"] = list(patch_metadata.get("imported_modules") or [])
        load_context["patch_hooks"] = list(patch_metadata.get("called_hooks") or [])
        load_context["patch_errors"] = list(patch_metadata.get("errors") or [])
    load_context["activation_name_shims"] = _stabilize_activation_name_shims(model)
    load_context["model_source"] = model_source
    load_context["input_source"] = tokenizer_source
    load_context["token_classification_backend"] = "gliner"

    input_adapter = getattr(model, "tokenizer", None)
    if input_adapter is None:
        input_adapter = getattr(getattr(model, "data_processor", None), "transformer_tokenizer", None)
    if hasattr(model, "eval"):
        model.eval()
    return model, input_adapter, model_config, requested_model_type, load_context


def _load_span_marker_stack(config: dict[str, Any], model_id: str, scenario: str, target_device: torch.device, load_context: dict[str, Any]):
    requested_model_type = str(config.get("model_type") or "").strip()
    canonical_model_type = "token_classification" if requested_model_type == "biomedical_token_classification" else requested_model_type
    if canonical_model_type != "token_classification":
        return None

    model_source, tokenizer_source, load_context = _resolve_model_sources(model_id, config, scenario, "tokenizer")
    config_payload = _load_config_payload_from_source(model_source)
    if not _looks_like_span_marker_checkpoint(config_payload, model_id=model_id):
        return None

    try:
        from span_marker import SpanMarkerModel
    except Exception as exc:
        raise RuntimeError("检测到 span-marker checkpoint，但当前业务测评环境缺少 span-marker 依赖。") from exc

    _apply_model_load_seed(config)
    load_kwargs: dict[str, Any] = {}
    if not Path(str(model_source)).exists():
        load_kwargs["cache_dir"] = str(CACHE_DIR)
    encoder_payload = config_payload.get("encoder") if isinstance(config_payload.get("encoder"), Mapping) else {}
    encoder_source = str(encoder_payload.get("_name_or_path") or "").strip()
    offline_env: dict[str, str] = {}
    if Path(str(model_source)).exists():
        offline_env["HF_HOME"] = str(CACHE_DIR)
        offline_env["HF_HUB_CACHE"] = str(CACHE_DIR)
        offline_env["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
    if encoder_source and _resolve_cached_hf_snapshot_source(encoder_source):
        offline_env["HF_HUB_OFFLINE"] = "1"
        offline_env["TRANSFORMERS_OFFLINE"] = "1"
    with _temporary_env_overrides(offline_env):
        model = SpanMarkerModel.from_pretrained(model_source, **load_kwargs)

    torch_dtype = _get_torch_dtype(config)
    move_kwargs: dict[str, Any] = {"device": target_device}
    if torch_dtype != "auto":
        move_kwargs["dtype"] = torch_dtype
    try:
        model = model.to(**move_kwargs)
    except TypeError:
        model = model.to(target_device)
    except Exception:
        inner_model = getattr(model, "model", None) or getattr(model, "encoder", None)
        if inner_model is None:
            raise
        try:
            inner_model.to(**move_kwargs)
        except TypeError:
            inner_model.to(target_device)

    model_config = getattr(model, "config", None)
    if model_config is None:
        model_config = SimpleNamespace()
    if config_payload:
        if not getattr(model_config, "model_type", None):
            setattr(model_config, "model_type", str(config_payload.get("model_type") or "span-marker"))
        if not getattr(model_config, "architectures", None):
            setattr(model_config, "architectures", list(config_payload.get("architectures") or []))
        if not getattr(model_config, "id2label", None):
            setattr(model_config, "id2label", dict(config_payload.get("id2label") or {}))
        if not getattr(model_config, "label2id", None):
            setattr(model_config, "label2id", dict(config_payload.get("label2id") or {}))
    setattr(model_config, "_slai_business_backend", "span_marker")
    setattr(model_config, "_slai_raw_config_payload", dict(config_payload))
    _apply_model_config_compatibility_fixes(model_config, model_id=model_id)

    load_context["runtime_compatibility_shims"] = _apply_model_runtime_compatibility_shims(model)
    patch_metadata = load_context.get("_patch_metadata")
    if isinstance(patch_metadata, dict):
        _apply_deferred_model_files_hooks(model, patch_metadata)
        load_context["patch_load_status"] = str(patch_metadata.get("status") or load_context.get("patch_load_status") or "disabled")
        load_context["patch_modules"] = list(patch_metadata.get("imported_modules") or [])
        load_context["patch_hooks"] = list(patch_metadata.get("called_hooks") or [])
        load_context["patch_errors"] = list(patch_metadata.get("errors") or [])
    load_context["activation_name_shims"] = _stabilize_activation_name_shims(model)
    load_context["model_source"] = model_source
    load_context["input_source"] = tokenizer_source
    load_context["token_classification_backend"] = "span_marker"

    input_adapter = getattr(model, "tokenizer", None)
    if hasattr(model, "eval"):
        model.eval()
    if input_adapter is not None and getattr(input_adapter, "pad_token", None) is None and getattr(input_adapter, "eos_token", None) is not None:
        input_adapter.pad_token = input_adapter.eos_token
    return model, input_adapter, model_config, requested_model_type, load_context


def _apply_model_config_compatibility_fixes(model_config, *, model_id: str) -> None:
    model_type = str(getattr(model_config, "model_type", "") or "").strip().lower()
    architectures = [str(item).strip().lower() for item in list(getattr(model_config, "architectures", []) or []) if str(item).strip()]
    model_id_text = str(model_id or "").strip().lower()
    if model_type == "olmo":
        legacy_hidden_size = getattr(model_config, "d_model", None)
        legacy_layer_count = getattr(model_config, "n_layers", None)
        legacy_head_count = getattr(model_config, "n_heads", None)
        legacy_vocab_size = getattr(model_config, "embedding_size", None) or getattr(model_config, "vocab_size", None)
        legacy_max_positions = getattr(model_config, "max_sequence_length", None)
        legacy_attention_bias = getattr(model_config, "include_bias", None)
        legacy_weight_tying = getattr(model_config, "weight_tying", None)
        legacy_mlp_hidden_size = getattr(model_config, "mlp_hidden_size", None)
        legacy_mlp_ratio = getattr(model_config, "mlp_ratio", None)
        legacy_activation = str(getattr(model_config, "activation_type", "") or "").strip().lower()
        try:
            if legacy_hidden_size:
                setattr(model_config, "hidden_size", int(legacy_hidden_size))
            if legacy_layer_count:
                setattr(model_config, "num_hidden_layers", int(legacy_layer_count))
            if legacy_head_count:
                setattr(model_config, "num_attention_heads", int(legacy_head_count))
                if not getattr(model_config, "multi_query_attention", False):
                    setattr(model_config, "num_key_value_heads", int(legacy_head_count))
            if legacy_vocab_size:
                setattr(model_config, "vocab_size", int(legacy_vocab_size))
            if legacy_max_positions:
                setattr(model_config, "max_position_embeddings", int(legacy_max_positions))
            if legacy_attention_bias is not None:
                setattr(model_config, "attention_bias", bool(legacy_attention_bias))
            if legacy_weight_tying is not None:
                setattr(model_config, "tie_word_embeddings", bool(legacy_weight_tying))
            if legacy_mlp_hidden_size:
                setattr(model_config, "intermediate_size", int(legacy_mlp_hidden_size))
            elif legacy_hidden_size and legacy_mlp_ratio:
                intermediate_size = int(float(legacy_hidden_size) * float(legacy_mlp_ratio))
                if legacy_activation in {"swiglu", "geglu", "glu"}:
                    intermediate_size //= 2
                if intermediate_size > 0:
                    setattr(model_config, "intermediate_size", intermediate_size)
        except Exception:
            pass
    if model_type == "gemmoe" or any("gemmoe" in arch for arch in architectures) or "gemmoe" in model_id_text:
        setattr(model_config, "_attn_implementation", "eager")
    if model_type == "phi3" or any("phi3" in arch for arch in architectures) or any(token in model_id_text for token in ("phi-3", "phi3")):
        setattr(model_config, "_attn_implementation", "eager")
        rope_scaling = getattr(model_config, "rope_scaling", None)
        if isinstance(rope_scaling, Mapping) and "type" not in rope_scaling:
            normalized_rope_scaling = dict(rope_scaling)
            rope_type = str(normalized_rope_scaling.get("rope_type") or "").strip()
            if rope_type and rope_type != "default":
                normalized_rope_scaling["type"] = rope_type
            else:
                normalized_rope_scaling = None
            setattr(model_config, "rope_scaling", normalized_rope_scaling)
    if model_type == "crystalcoder" or any("crystalcoder" in arch for arch in architectures) or "llm360/crystal" in model_id_text:
        if not hasattr(model_config, "add_cross_attention"):
            setattr(model_config, "add_cross_attention", False)


def _should_retry_without_trust_remote_code(exc: Exception) -> bool:
    message = str(exc or "")
    if "does not appear to have a file named" in message:
        return True
    normalized_message = message.lower()
    offline_markers = (
        "couldn't find them in the cached files",
        "outgoing traffic has been disabled",
        "couldn't connect to",
        "localentrynotfounderror",
    )
    if any(marker in normalized_message for marker in offline_markers):
        return True
    module_markers = (
        "configuration_",
        "modeling_",
        "tokenization_",
        "processing_",
        "feature_extraction_",
    )
    if any(marker in normalized_message for marker in module_markers):
        return True
    return False


def _peek_config_model_type_from_source(source: str) -> str:
    payload = _load_config_payload_from_source(source)
    return str(payload.get("model_type") or "").strip()


def _resolve_masked_lm_generator_source(model_source: str, model_id: str, model_type: str) -> tuple[str, str] | None:
    if str(model_type or "").strip() != "masked_lm":
        return None
    source_path = Path(str(model_source))
    if not source_path.is_dir():
        return None
    generator_weights_path = source_path / "pytorch_model.generator.bin"
    generator_config_path = source_path / "generator_config.json"
    if not generator_weights_path.is_file() or not generator_config_path.is_file():
        return None
    model_id_text = str(model_id or "").strip().lower()
    config_model_type = _peek_config_model_type_from_source(str(source_path)).lower()
    if "deberta-v3" not in model_id_text and config_model_type != "deberta-v2":
        return None
    return str(generator_weights_path), str(generator_config_path)


def _should_force_manual_deberta_generator_load(source: str, *, model_id: str, model_type: str, model_config) -> bool:
    if str(model_type or "").strip() != "masked_lm":
        return False
    source_path = Path(str(source))
    has_supported_source = False
    if source_path.is_file():
        has_supported_source = source_path.name == "pytorch_model.generator.bin" or source_path.name == "pytorch_model.bin" or source_path.suffix == ".safetensors"
    elif source_path.is_dir():
        has_supported_source = (
            source_path.joinpath("generator_config.json").is_file()
            or source_path.joinpath("pytorch_model.bin").is_file()
            or source_path.joinpath("model.safetensors").is_file()
            or any(source_path.glob("pytorch_model-*.bin"))
            or any(source_path.glob("model-*.safetensors"))
        )
    if not has_supported_source:
        return False
    model_id_text = str(model_id or "").strip().lower()
    config_model_type = str(getattr(model_config, "model_type", "") or "").strip().lower()
    return "deberta-v3" in model_id_text or config_model_type == "deberta-v2"


def _remap_deberta_generator_masked_lm_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    source_to_target_keys = {
        "lm_predictions.lm_head.dense.weight": ("cls.predictions.transform.dense.weight",),
        "lm_predictions.lm_head.dense.bias": ("cls.predictions.transform.dense.bias",),
        "lm_predictions.lm_head.LayerNorm.weight": ("cls.predictions.transform.LayerNorm.weight",),
        "lm_predictions.lm_head.LayerNorm.bias": ("cls.predictions.transform.LayerNorm.bias",),
        "lm_predictions.lm_head.bias": ("cls.predictions.bias", "cls.predictions.decoder.bias"),
    }
    if all(any(target_key in state_dict for target_key in target_keys) for target_keys in source_to_target_keys.values()):
        return state_dict
    if not any(source_key in state_dict for source_key in source_to_target_keys):
        return state_dict
    missing_source_keys = [key for key in source_to_target_keys if key not in state_dict]
    if missing_source_keys:
        raise RuntimeError(f"deberta generator checkpoint 缺少关键 MLM 头权重: {missing_source_keys}")
    remapped_state_dict = dict(state_dict)
    for source_key, target_keys in source_to_target_keys.items():
        source_value = state_dict[source_key]
        remapped_state_dict.pop(source_key, None)
        for target_key in target_keys:
            remapped_state_dict[target_key] = source_value
    return remapped_state_dict


def _load_model_config_with_known_fallbacks(auto_config_cls, source: str, *, model_id: str, trust_remote_code: bool):
    effective_trust_remote_code = trust_remote_code
    last_error: Exception | None = None
    try:
        return auto_config_cls.from_pretrained(source, cache_dir=str(CACHE_DIR), trust_remote_code=effective_trust_remote_code), effective_trust_remote_code
    except (OSError, ValueError) as exc:
        last_error = exc
    if effective_trust_remote_code and _should_retry_without_trust_remote_code(last_error):
        effective_trust_remote_code = False
        try:
            return auto_config_cls.from_pretrained(source, cache_dir=str(CACHE_DIR), trust_remote_code=False), effective_trust_remote_code
        except (OSError, ValueError) as exc:
            last_error = exc

    config_model_type = _peek_config_model_type_from_source(source)
    if not config_model_type:
        match = re.search(r"model type `([^`]+)`", str(last_error or ""))
        if match is not None:
            config_model_type = str(match.group(1) or "").strip()

    normalized_model_type = config_model_type.lower()
    model_id_text = str(model_id or "").strip().lower()
    if normalized_model_type == "scbert" or "biomed.rna.bert" in model_id_text:
        try:
            from transformers import BertConfig

            return BertConfig.from_pretrained(source, cache_dir=str(CACHE_DIR), trust_remote_code=effective_trust_remote_code), effective_trust_remote_code
        except Exception as exc:
            last_error = exc

    if last_error is None:
        raise RuntimeError(f"无法加载配置: {source}")
    raise last_error


def _load_colbert_stack(config: dict[str, Any], model_id: str, scenario: str, target_device: torch.device, load_context: dict[str, Any]):
    if str(config.get("model_type") or "").strip() != "reranker":
        return None

    from transformers import AutoConfig, AutoModel, AutoTokenizer

    model_source, tokenizer_source, load_context = _resolve_model_sources(model_id, config, scenario, "tokenizer")
    trust_remote_code = _as_bool(config.get("trust_remote_code"), default=False)
    torch_dtype = _get_torch_dtype(config)

    input_loader_kwargs = {
        "cache_dir": str(CACHE_DIR),
        "trust_remote_code": trust_remote_code,
    }
    try:
        input_adapter = AutoTokenizer.from_pretrained(tokenizer_source, **input_loader_kwargs)
    except (ValueError, ImportError) as exc:
        try:
            input_adapter = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False, **input_loader_kwargs)
        except (ValueError, ImportError):
            input_adapter = _load_tokenizer_from_local_vocab_fallback(tokenizer_source)
            if input_adapter is None:
                raise exc

    model_config = AutoConfig.from_pretrained(model_source, cache_dir=str(CACHE_DIR), trust_remote_code=trust_remote_code)
    _apply_model_config_compatibility_fixes(model_config, model_id=model_id)
    if not _looks_like_colbert_reranker(model_id, config, model_config=model_config):
        return None

    load_kwargs = {
        "cache_dir": str(CACHE_DIR),
        "trust_remote_code": trust_remote_code,
        "config": model_config,
    }
    if torch_dtype != "auto":
        load_kwargs["torch_dtype"] = torch_dtype
    if target_device.type in {"cuda", "npu"} and _accelerate_available():
        load_kwargs["device_map"] = "auto"

    def _load_from_source(source: str):
        _apply_model_load_seed(config)
        return AutoModel.from_pretrained(source, **load_kwargs)

    candidate_sources = [model_source]
    if model_source != model_id:
        candidate_sources.append(model_id)
    for candidate in _fallback_model_source_candidates(model_id, "embedding"):
        if candidate not in candidate_sources:
            candidate_sources.append(candidate)

    model = None
    last_error = None
    for index, candidate_source in enumerate(candidate_sources):
        if index > 0:
            load_kwargs.pop("device_map", None)
        try:
            model = _load_from_source(candidate_source)
            if index > 0:
                model = model.to(target_device)
                load_context["model_source"] = candidate_source
                load_context["model_source_kind"] = "hub_fallback"
            break
        except Exception as exc:
            last_error = exc
    if model is None:
        fallback_text = ", ".join(candidate_sources[1:]) or model_source
        raise RuntimeError(f"加载 ColBERT reranker 失败: source={model_source}, fallback={fallback_text}, error={last_error}") from last_error

    load_context["runtime_compatibility_shims"] = _apply_model_runtime_compatibility_shims(model)
    patch_metadata = load_context.get("_patch_metadata")
    if isinstance(patch_metadata, dict):
        _apply_deferred_model_files_hooks(model, patch_metadata)
        load_context["patch_load_status"] = str(patch_metadata.get("status") or load_context.get("patch_load_status") or "disabled")
        load_context["patch_modules"] = list(patch_metadata.get("imported_modules") or [])
        load_context["patch_hooks"] = list(patch_metadata.get("called_hooks") or [])
        load_context["patch_errors"] = list(patch_metadata.get("errors") or [])
    load_context["activation_name_shims"] = _stabilize_activation_name_shims(model)

    projection_source = model_source if Path(str(model_source)).is_dir() else _resolve_local_snapshot_source(model_id, require_model_assets=True)
    projection, projection_dim = _load_colbert_projection(projection_source, target_device=target_device, torch_dtype=torch_dtype)
    query_marker, doc_marker, marker_strategy = _resolve_colbert_marker_tokens(input_adapter, config)

    model._slai_colbert_enabled = True
    model._slai_colbert_projection = projection
    model._slai_colbert_projection_dim = int(projection_dim or getattr(model_config, "hidden_size", 0) or 0)
    model._slai_colbert_query_marker = query_marker
    model._slai_colbert_doc_marker = doc_marker
    model._slai_colbert_marker_strategy = marker_strategy
    model._slai_colbert_query_max_length = int(config.get("colbert_query_max_length") or 64)
    model._slai_colbert_doc_max_length = int(config.get("colbert_doc_max_length") or 256)
    model._slai_colbert_skip_token_ids = _build_colbert_skip_token_ids(input_adapter, query_marker=query_marker, doc_marker=doc_marker)

    load_context["reranker_backend"] = "colbert_late_interaction"
    load_context["colbert_projection_source"] = projection_source
    load_context["colbert_projection_dim"] = model._slai_colbert_projection_dim
    load_context["colbert_query_marker"] = query_marker
    load_context["colbert_doc_marker"] = doc_marker
    load_context["colbert_marker_strategy"] = marker_strategy
    load_context["colbert_query_max_length"] = model._slai_colbert_query_max_length
    load_context["colbert_doc_max_length"] = model._slai_colbert_doc_max_length

    model.eval()
    if getattr(input_adapter, "pad_token", None) is None and getattr(input_adapter, "eos_token", None) is not None:
        input_adapter.pad_token = input_adapter.eos_token
    return model, input_adapter, model_config, "reranker", load_context


def _looks_like_qwen_forced_aligner_model(model_id: str, config: dict[str, Any]) -> bool:
    backend = str(config.get("model_backend") or "").strip().lower()
    if backend in {"qwen_forced_aligner", "qwen3_forced_aligner"}:
        return True
    signal_text = " ".join(
        str(value or "").strip().lower()
        for value in (
            model_id,
            config.get("architectures"),
            config.get("model_class"),
            config.get("business_intent"),
            config.get("output_type_hint"),
            config.get("asr_task"),
        )
        if value
    )
    return "forcedaligner" in signal_text or "forced aligner" in signal_text or "forced-aligner" in signal_text or "forced_aligner" in signal_text


def _looks_like_qwen_asr_model(model_id: str, config: dict[str, Any]) -> bool:
    backend = str(config.get("model_backend") or "").strip().lower()
    if backend == "qwen_asr":
        return True
    if _looks_like_qwen_forced_aligner_model(model_id, config):
        return False
    signal_text = " ".join(
        str(value or "").strip().lower()
        for value in (
            model_id,
            config.get("architectures"),
            config.get("model_class"),
            config.get("business_intent"),
            config.get("output_type_hint"),
        )
        if value
    )
    return "qwen3-asr" in signal_text or "qwen3_asr" in signal_text


def _looks_like_nemo_asr_model(model_id: str, config: dict[str, Any]) -> bool:
    backend = str(config.get("model_backend") or "").strip().lower()
    if backend == "nemo_asr":
        return True
    signal_text = " ".join(
        str(value or "").strip().lower()
        for value in (
            model_id,
            config.get("architectures"),
            config.get("model_class"),
            config.get("business_intent"),
            config.get("output_type_hint"),
        )
        if value
    )
    return "reazonspeech-nemo" in signal_text or "encdecrnntbpemodel" in signal_text


def _looks_like_qwen_tts_model(model_id: str, config: dict[str, Any]) -> bool:
    backend = str(config.get("model_backend") or "").strip().lower()
    if backend == "qwen_tts":
        return True
    model_id_text = str(model_id or "").strip().lower()
    return "qwen3-tts" in model_id_text or "qwen3_tts" in model_id_text


def _looks_like_f5_tts_model(model_id: str, config: dict[str, Any]) -> bool:
    backend = str(config.get("model_backend") or "").strip().lower()
    if backend.startswith("f5_tts"):
        return True
    signal_text = " ".join(
        str(value or "").strip().lower()
        for value in (
            model_id,
            config.get("architectures"),
            config.get("model_class"),
            config.get("business_intent"),
            config.get("output_type_hint"),
        )
        if value
    )
    return any(token in signal_text for token in ("thonburiantts", "thonburian tts", "f5-tts", "f5_tts"))


def _contains_colbert_signal(*values: Any) -> bool:
    lowered = " ".join(str(value or "").strip().lower() for value in values if value)
    return "colbert" in lowered or "late interaction" in lowered


def _looks_like_colbert_reranker(model_id: str, config: dict[str, Any], *, model_config: Any | None = None) -> bool:
    if str(config.get("model_type") or "").strip() != "reranker":
        return False
    signal_values: list[Any] = [
        model_id,
        config.get("architectures"),
        config.get("model_class"),
        config.get("output_type_hint"),
        config.get("business_intent"),
    ]
    if model_config is not None:
        signal_values.extend(
            (
                getattr(model_config, "architectures", None),
                getattr(model_config, "model_type", None),
            )
        )
    return _contains_colbert_signal(*signal_values)


def _tokenizer_has_exact_token(tokenizer, token: str | None) -> bool:
    token_text = str(token or "").strip()
    if not token_text:
        return False
    try:
        token_id = tokenizer.convert_tokens_to_ids(token_text)
    except Exception:
        return False
    if token_id is None:
        return False
    unk_token = getattr(tokenizer, "unk_token", None)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if unk_token is not None and unk_token_id is not None and int(token_id) == int(unk_token_id) and token_text != str(unk_token):
        return False
    try:
        recovered = tokenizer.convert_ids_to_tokens([int(token_id)])
    except Exception:
        return False
    return bool(recovered) and str(recovered[0]) == token_text


def _resolve_colbert_marker_tokens(tokenizer, config: dict[str, Any]) -> tuple[str | None, str | None, str]:
    configured_query = str(config.get("query_token") or "").strip() or None
    configured_doc = str(config.get("doc_token") or "").strip() or None
    if configured_query and configured_doc and _tokenizer_has_exact_token(tokenizer, configured_query) and _tokenizer_has_exact_token(tokenizer, configured_doc):
        return configured_query, configured_doc, "config"
    if _tokenizer_has_exact_token(tokenizer, "[Q]") and _tokenizer_has_exact_token(tokenizer, "[D]"):
        return "[Q]", "[D]", "registered_markers"
    if _tokenizer_has_exact_token(tokenizer, "[unused0]") and _tokenizer_has_exact_token(tokenizer, "[unused1]"):
        return "[unused0]", "[unused1]", "bert_unused_markers"
    return None, None, "plain_text"


def _build_colbert_skip_token_ids(tokenizer, *, query_marker: str | None, doc_marker: str | None) -> set[int]:
    token_ids: set[int] = set()
    for attr_name in ("pad_token_id", "cls_token_id", "sep_token_id", "mask_token_id"):
        token_id = getattr(tokenizer, attr_name, None)
        if token_id is None:
            continue
        try:
            token_ids.add(int(token_id))
        except Exception:
            continue
    for marker in (query_marker, doc_marker):
        if not marker:
            continue
        if _tokenizer_has_exact_token(tokenizer, marker):
            try:
                token_ids.add(int(tokenizer.convert_tokens_to_ids(marker)))
            except Exception:
                continue
    return token_ids


def _extract_last_hidden_state(model_output) -> torch.Tensor:
    hidden = getattr(model_output, "last_hidden_state", None)
    if hidden is None and isinstance(model_output, (tuple, list)) and model_output:
        hidden = model_output[0]
    if hidden is None or not torch.is_tensor(hidden):
        raise RuntimeError("ColBERT reranker 输出缺少 last_hidden_state，无法计算 late interaction score")
    return hidden


def _load_colbert_projection(source: str | None, *, target_device: torch.device, torch_dtype: torch.dtype | str) -> tuple[torch.nn.Module | None, int | None]:
    source_path = Path(str(source or ""))
    if not source_path.is_dir():
        return None, None
    state_dict = _load_state_dict_from_snapshot(source_path)
    weight = state_dict.get("linear.weight")
    bias = state_dict.get("linear.bias")
    if weight is None or not torch.is_tensor(weight) or weight.ndim != 2:
        return None, None

    projection = torch.nn.Linear(int(weight.shape[1]), int(weight.shape[0]), bias=torch.is_tensor(bias))
    projection.weight.data.copy_(weight.to(dtype=projection.weight.dtype))
    if torch.is_tensor(bias) and projection.bias is not None:
        projection.bias.data.copy_(bias.to(dtype=projection.bias.dtype))
    if torch_dtype != "auto":
        projection = projection.to(dtype=torch_dtype)
    projection = projection.to(target_device)
    projection.eval()
    del state_dict
    return projection, int(weight.shape[0])


def _ensure_qwen_tts_namespace_packages() -> None:
    try:
        package_spec = importlib.util.find_spec("qwen_tts")
    except Exception as exc:  # pragma: no cover - runtime import resolution
        raise ImportError("未安装 qwen_tts") from exc
    if package_spec is None or not package_spec.submodule_search_locations:
        raise ImportError("未找到 qwen_tts 包目录")

    package_dir = Path(next(iter(package_spec.submodule_search_locations)))

    def _register_namespace(name: str, directory: Path) -> None:
        module = ModuleType(name)
        module.__file__ = str(directory / "__init__.py")
        module.__package__ = name
        module.__path__ = [str(directory)]  # type: ignore[attr-defined]
        sys.modules[name] = module

    for module_name in list(sys.modules):
        if module_name == "qwen_tts" or module_name.startswith("qwen_tts."):
            del sys.modules[module_name]

    _register_namespace("qwen_tts", package_dir)
    for child_name in ("core", "inference"):
        child_dir = package_dir / child_name
        if child_dir.exists():
            _register_namespace(f"qwen_tts.{child_name}", child_dir)

    def _unsupported_torchaudio_kaldi(*args, **kwargs):
        raise RuntimeError("torchaudio.compliance.kaldi is unavailable in this runtime; qwen_tts tokenizer_25hz should not be used for 12Hz CustomVoice")

    for module_name in list(sys.modules):
        if module_name == "torchaudio" or module_name.startswith("torchaudio."):
            del sys.modules[module_name]

    torchaudio_module = ModuleType("torchaudio")
    torchaudio_module.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None, is_package=True)
    torchaudio_module.__path__ = []  # type: ignore[attr-defined]
    compliance_module = ModuleType("torchaudio.compliance")
    compliance_module.__spec__ = importlib.machinery.ModuleSpec("torchaudio.compliance", loader=None, is_package=True)
    compliance_module.__path__ = []  # type: ignore[attr-defined]
    kaldi_module = ModuleType("torchaudio.compliance.kaldi")
    kaldi_module.__spec__ = importlib.machinery.ModuleSpec("torchaudio.compliance.kaldi", loader=None, is_package=False)
    kaldi_module.fbank = _unsupported_torchaudio_kaldi  # type: ignore[attr-defined]
    compliance_module.kaldi = kaldi_module  # type: ignore[attr-defined]
    torchaudio_module.compliance = compliance_module  # type: ignore[attr-defined]
    sys.modules["torchaudio"] = torchaudio_module
    sys.modules["torchaudio.compliance"] = compliance_module
    sys.modules["torchaudio.compliance.kaldi"] = kaldi_module

    # Qwen3-TTS 12Hz CustomVoice only needs the V2 tokenizer at runtime, but
    # upstream eagerly registers both V1/V2 classes. Import V1 under a stubbed
    # torchaudio namespace so registration can complete without pulling CUDA
    # torchaudio binaries into the Ascend-only environment.
    core_module = sys.modules.get("qwen_tts.core")
    if core_module is not None:
        tokenizer_v1_config_mod = importlib.import_module("qwen_tts.core.tokenizer_25hz.configuration_qwen3_tts_tokenizer_v1")
        tokenizer_v1_model_mod = importlib.import_module("qwen_tts.core.tokenizer_25hz.modeling_qwen3_tts_tokenizer_v1")
        tokenizer_v2_config_mod = importlib.import_module("qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2")
        tokenizer_v2_model_mod = importlib.import_module("qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2")
        v1_config = getattr(tokenizer_v1_config_mod, "Qwen3TTSTokenizerV1Config")
        v1_model = getattr(tokenizer_v1_model_mod, "Qwen3TTSTokenizerV1Model")
        v2_config = getattr(tokenizer_v2_config_mod, "Qwen3TTSTokenizerV2Config")
        v2_model = getattr(tokenizer_v2_model_mod, "Qwen3TTSTokenizerV2Model")
        setattr(core_module, "Qwen3TTSTokenizerV1Config", v1_config)
        setattr(core_module, "Qwen3TTSTokenizerV1Model", v1_model)
        setattr(core_module, "Qwen3TTSTokenizerV2Config", v2_config)
        setattr(core_module, "Qwen3TTSTokenizerV2Model", v2_model)


def _load_qwen_tts_stack(config: dict[str, Any], model_id: str, scenario: str, target_device: torch.device, load_context: dict[str, Any]):
    if not _looks_like_qwen_tts_model(model_id, config):
        return None

    model_id_text = str(model_id or "").strip().lower()
    model_source, _, load_context = _resolve_model_sources(model_id, config, scenario, "processor")
    source = model_source if Path(str(model_source)).exists() else model_id
    source_path = Path(str(source))
    source_kind = "local_snapshot" if source_path.exists() else "hub"

    torch_dtype = _get_torch_dtype(config)
    if torch_dtype == "auto":
        torch_dtype = torch.bfloat16 if target_device.type in {"npu", "cuda"} else torch.float32

    qwen_tts_mode = "custom_voice" if "customvoice" in model_id_text or "custom_voice" in model_id_text else "voice_design" if "voicedesign" in model_id_text or "voice_design" in model_id_text else "generic"
    # `qwen_tts.__init__` and `qwen_tts.core.__init__` eagerly import the 25Hz
    # tokenizer stack, which drags in torchaudio even for 12Hz variants.
    _ensure_qwen_tts_namespace_packages()
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        source,
        dtype=torch_dtype,
        device_map=str(target_device),
        attn_implementation=str(config.get("attn_implementation") or "eager"),
        cache_dir=str(CACHE_DIR),
    )
    input_adapter = SimpleNamespace(backend="qwen_tts", generation_mode=qwen_tts_mode)
    inner_model = getattr(model, "model", None) or getattr(model, "_model", None) or model

    patch_metadata = load_context.get("_patch_metadata")
    if isinstance(patch_metadata, dict):
        _apply_deferred_model_files_hooks(inner_model, patch_metadata)
        load_context["patch_load_status"] = str(patch_metadata.get("status") or load_context.get("patch_load_status") or "disabled")
        load_context["patch_modules"] = list(patch_metadata.get("imported_modules") or [])
        load_context["patch_hooks"] = list(patch_metadata.get("called_hooks") or [])
        load_context["patch_errors"] = list(patch_metadata.get("errors") or [])
    load_context["activation_name_shims"] = _stabilize_activation_name_shims(inner_model)

    if hasattr(inner_model, "eval"):
        inner_model.eval()

    load_context["model_source"] = str(source)
    load_context["model_source_kind"] = source_kind
    load_context["input_source"] = str(source)
    load_context["input_source_kind"] = source_kind

    model_config = SimpleNamespace(
        id2label={},
        num_labels=0,
        model_type="qwen3_tts",
        architectures=["Qwen3TTSForConditionalGeneration"],
    )
    return model, input_adapter, model_config, "tts", load_context


def _load_f5_tts_stack(config: dict[str, Any], model_id: str, scenario: str, target_device: torch.device, load_context: dict[str, Any]):
    if not _looks_like_f5_tts_model(model_id, config):
        return None
    if not ACCURACY_RUN_PATH.is_file():
        raise RuntimeError(f"F5-TTS 业务测评需要 accuracy_run.py: {ACCURACY_RUN_PATH}")

    accuracy_module = _load_module_from_path(f"_business_accuracy_f5_tts_{abs(hash(str(ACCURACY_RUN_PATH.resolve())))}", ACCURACY_RUN_PATH)
    create_api = getattr(accuracy_module, "create_api_from_pretrained", None)
    run_single_infer = getattr(accuracy_module, "run_single_infer", None)
    reference_audio = getattr(accuracy_module, "REFERENCE_AUDIO", None)
    reference_text = str(getattr(accuracy_module, "REFERENCE_TEXT", "") or "").strip()
    default_prompts = tuple(getattr(accuracy_module, "BUILTIN_THAI_TEXTS", ()) or F5_TTS_DEFAULT_PROMPTS)
    default_nfe_step = int(getattr(accuracy_module, "DEFAULT_NFE_STEP", 4) or 4)
    if not callable(create_api) or not callable(run_single_infer):
        raise RuntimeError("F5-TTS accuracy_run.py 缺少 create_api_from_pretrained()/run_single_infer()，无法复用真实推理链路")
    if reference_audio is not None and hasattr(accuracy_module, "ensure_reference_audio"):
        accuracy_module.ensure_reference_audio()

    perf_module = None
    perf_prepare_reference = None
    perf_build_sample_plan = None
    perf_run_single_infer_fast = None
    if ACCURACY_RUN_PERF_PATH.is_file():
        perf_module = _load_module_from_path(f"_business_accuracy_f5_tts_perf_{abs(hash(str(ACCURACY_RUN_PERF_PATH.resolve())))}", ACCURACY_RUN_PERF_PATH)
        perf_prepare_reference = getattr(perf_module, "prepare_reference_for_perf", None)
        perf_build_sample_plan = getattr(perf_module, "build_sample_plan", None)
        perf_run_single_infer_fast = getattr(perf_module, "run_single_infer_fast", None)

    api = create_api(str(target_device))
    runtime_model = getattr(api, "ema_model", None) or getattr(api, "model", None) or api
    if hasattr(runtime_model, "eval"):
        runtime_model.eval()

    patch_metadata = load_context.get("_patch_metadata")
    if isinstance(patch_metadata, dict):
        _apply_deferred_model_files_hooks(runtime_model, patch_metadata)
        load_context["patch_load_status"] = str(patch_metadata.get("status") or load_context.get("patch_load_status") or "disabled")
        load_context["patch_modules"] = list(patch_metadata.get("imported_modules") or [])
        load_context["patch_hooks"] = list(patch_metadata.get("called_hooks") or [])
        load_context["patch_errors"] = list(patch_metadata.get("errors") or [])

    load_context["model_source"] = str(getattr(accuracy_module, "PRETRAINED_CKPT", model_id) or model_id)
    load_context["model_source_kind"] = "local_snapshot"
    load_context["input_source"] = str(reference_audio or model_id)
    load_context["input_source_kind"] = "reference_audio"

    input_adapter = SimpleNamespace(
        backend="f5_tts",
        api=api,
        inference_fn=run_single_infer,
        perf_prepare_reference=perf_prepare_reference,
        perf_build_sample_plan=perf_build_sample_plan,
        perf_inference_fn=perf_run_single_infer_fast,
        reference_audio_path=str(reference_audio or ""),
        reference_audio_name=Path(reference_audio).name if reference_audio else "",
        reference_text=reference_text,
        nfe_step=default_nfe_step,
        default_prompts=default_prompts,
    )
    model_config = SimpleNamespace(
        id2label={},
        num_labels=0,
        model_type="f5_tts",
        architectures=["F5TTSForConditionalGeneration"],
    )
    return runtime_model, input_adapter, model_config, "tts", load_context


def _looks_like_accuracy_latency_bridge(config: dict[str, Any]) -> bool:
    model_type = str(config.get("model_type") or "").strip().lower()
    evaluation_profile = str(config.get("evaluation_profile") or "").strip().lower()
    output_type_hint = str(config.get("output_type_hint") or "").strip().lower()
    return bool(
        ACCURACY_RUN_PATH.is_file()
        and model_type in {"diffusion", "video"}
        and (evaluation_profile == "latency_only" or output_type_hint in {"video_latency", "diffusion_latency", "generated_images", "image_tensors"})
    )


def _looks_like_local_embedding_bridge(config: dict[str, Any]) -> bool:
    return bool(
        ACCURACY_RUN_PATH.is_file()
        and str(config.get("model_type") or "").strip().lower() == "embedding"
        and str(config.get("evaluation_profile") or "").strip().lower() == "embedding_similarity"
        and str(config.get("output_type_hint") or "").strip().lower() == "embeddings"
    )


def _module_supports_local_embedding_bridge(*modules: Any, config: dict[str, Any]) -> bool:
    architecture_text = " ".join(str(item or "").strip() for item in (config.get("architectures") or [] if isinstance(config.get("architectures"), list) else [config.get("architectures")]))
    lowered_architecture_text = architecture_text.lower()
    if any(marker in lowered_architecture_text for marker in ("maskedlm", "sequenceclassification", "tokenclassification", "causallm", "questionanswering")):
        return False

    has_embedding_runtime_helpers = False
    for module in modules:
        if module is None:
            continue
        if getattr(module, "AutoTokenizer", None) is not None:
            return False
        if any(callable(getattr(module, name, None)) for name in ("load_benchmark_embeddings", "run_embeddings")):
            has_embedding_runtime_helpers = True

    return has_embedding_runtime_helpers


def _call_accuracy_bridge_setup_model(module, *, target_device: torch.device):
    setup_model = getattr(module, "setup_model", None)
    if not callable(setup_model):
        raise RuntimeError(f"{getattr(module, '__file__', 'accuracy_run.py')} 缺少 setup_model()，无法复用 latency bridge")
    setup_signature = inspect.signature(setup_model)
    kwargs: dict[str, Any] = {}
    if "use_pretrained" in setup_signature.parameters:
        kwargs["use_pretrained"] = True
    if "device" in setup_signature.parameters:
        kwargs["device"] = str(target_device)
    if "cache_dir" in setup_signature.parameters:
        kwargs["cache_dir"] = str(CACHE_DIR)
    return setup_model(**kwargs)


def _load_accuracy_latency_bridge_stack(config: dict[str, Any], model_id: str, scenario: str, target_device: torch.device, load_context: dict[str, Any]):
    if not _looks_like_accuracy_latency_bridge(config):
        return None

    accuracy_module = _load_module_from_path(f"_business_accuracy_latency_bridge_{abs(hash(str(ACCURACY_RUN_PATH.resolve())))}", ACCURACY_RUN_PATH)
    runtime_module = accuracy_module
    perf_module = None
    if ACCURACY_RUN_PERF_PATH.is_file():
        perf_module = _load_module_from_path(f"_business_accuracy_latency_perf_bridge_{abs(hash(str(ACCURACY_RUN_PERF_PATH.resolve())))}", ACCURACY_RUN_PERF_PATH)
        if scenario == "npu_perf":
            apply_npu_optimizations = getattr(perf_module, "apply_npu_optimizations", None)
            if callable(apply_npu_optimizations):
                apply_npu_optimizations()
            runtime_module = perf_module

    model = _call_accuracy_bridge_setup_model(runtime_module, target_device=target_device)
    if hasattr(model, "eval"):
        model.eval()

    load_context["model_source"] = str(model_id)
    load_context["model_source_kind"] = "accuracy_run_bridge"
    load_context["input_source"] = str(getattr(runtime_module, "__file__", ACCURACY_RUN_PATH))
    load_context["input_source_kind"] = "local_runtime_script"

    input_adapter = SimpleNamespace(
        backend="accuracy_latency_bridge",
        runtime_module=runtime_module,
        accuracy_module=accuracy_module,
        perf_module=perf_module,
        run_single_fn=getattr(perf_module, "run_single", getattr(runtime_module, "run_single", None)),
        empty_cache_fn=getattr(perf_module, "empty_cache", getattr(runtime_module, "empty_cache", getattr(accuracy_module, "empty_cache", None))),
        requested_model_type=str(config.get("model_type") or "").strip().lower(),
        output_type_hint=str(config.get("output_type_hint") or "").strip().lower(),
    )
    model_config = SimpleNamespace(
        id2label={},
        num_labels=0,
        model_type=str(config.get("model_type") or "").strip().lower() or "diffusion",
        architectures=["AccuracyRunLatencyBridge"],
    )
    return model, input_adapter, model_config, model_config.model_type, load_context


def _call_local_embedding_bridge_setup_model(runtime_module, accuracy_module, *, target_device: torch.device):
    setup_model = getattr(runtime_module, "setup_model", None)
    if not callable(setup_model):
        setup_model = getattr(accuracy_module, "setup_model", None)
    if not callable(setup_model):
        raise RuntimeError(f"{getattr(accuracy_module, '__file__', 'accuracy_run.py')} 缺少 setup_model()，无法复用 local embedding bridge")
    setup_signature = inspect.signature(setup_model)
    kwargs: dict[str, Any] = {}
    if "use_pretrained" in setup_signature.parameters:
        kwargs["use_pretrained"] = True
    if "device" in setup_signature.parameters:
        kwargs["device"] = str(target_device)
    if "cache_dir" in setup_signature.parameters:
        kwargs["cache_dir"] = str(CACHE_DIR)
    return setup_model(**kwargs)


def _load_local_embedding_bridge_stack(config: dict[str, Any], model_id: str, scenario: str, target_device: torch.device, load_context: dict[str, Any]):
    if not _looks_like_local_embedding_bridge(config):
        return None

    accuracy_module = _load_module_from_path(f"_business_local_embedding_accuracy_bridge_{abs(hash(str(ACCURACY_RUN_PATH.resolve())))}", ACCURACY_RUN_PATH)
    runtime_module = accuracy_module
    perf_module = None
    if ACCURACY_RUN_PERF_PATH.is_file():
        perf_module = _load_module_from_path(f"_business_local_embedding_perf_bridge_{abs(hash(str(ACCURACY_RUN_PERF_PATH.resolve())))}", ACCURACY_RUN_PERF_PATH)
        if scenario == "npu_perf":
            apply_npu_optimizations = getattr(perf_module, "apply_npu_optimizations", None)
            if callable(apply_npu_optimizations):
                apply_npu_optimizations()
            runtime_module = perf_module

    if not _module_supports_local_embedding_bridge(accuracy_module, perf_module, runtime_module, config=config):
        return None

    model, _model_config = _call_local_embedding_bridge_setup_model(runtime_module, accuracy_module, target_device=target_device)
    if hasattr(model, "eval"):
        model.eval()

    load_context["model_source"] = str(model_id)
    load_context["model_source_kind"] = "accuracy_run_bridge"
    load_context["input_source"] = str(getattr(runtime_module, "__file__", ACCURACY_RUN_PATH))
    load_context["input_source_kind"] = "local_runtime_script"

    input_adapter = SimpleNamespace(
        backend="local_embedding_bridge",
        runtime_module=runtime_module,
        accuracy_module=accuracy_module,
        perf_module=perf_module,
        requested_model_type="embedding",
        output_type_hint="embeddings",
        empty_cache_fn=getattr(perf_module, "empty_cache", getattr(runtime_module, "empty_cache", getattr(accuracy_module, "empty_cache", None))),
    )
    model_config = SimpleNamespace(
        id2label={},
        num_labels=0,
        model_type="embedding",
        architectures=["LocalEmbeddingBridge"],
    )
    return model, input_adapter, model_config, "accuracy_latency_bridge", load_context


def _looks_like_local_vision_classification_bridge(config: dict[str, Any]) -> bool:
    return bool(
        ACCURACY_RUN_PATH.is_file()
        and str(config.get("model_type") or "").strip().lower() == "vision_classification"
    )


def _resolve_local_vision_helper_module(accuracy_module):
    build_model = getattr(accuracy_module, "build_model", None)
    helper_module_name = str(getattr(build_model, "__module__", "") or "").strip()
    if not helper_module_name or helper_module_name == getattr(accuracy_module, "__name__", ""):
        return accuracy_module
    helper_module = sys.modules.get(helper_module_name)
    if helper_module is not None:
        return helper_module
    helper_path = ADAPT_DIR / f"{helper_module_name.rsplit('.', 1)[-1]}.py"
    if helper_path.is_file():
        return _load_module_from_path(f"_business_local_vision_helper_{abs(hash(str(helper_path.resolve())))}", helper_path)
    return accuracy_module


def _call_local_vision_bridge_setup_model(runtime_module, accuracy_module, *, target_device: torch.device):
    setup_model = getattr(runtime_module, "setup_model", None)
    if not callable(setup_model):
        setup_model = getattr(accuracy_module, "setup_model", None)
    if callable(setup_model):
        setup_signature = inspect.signature(setup_model)
        kwargs: dict[str, Any] = {}
        if "use_pretrained" in setup_signature.parameters:
            kwargs["use_pretrained"] = True
        if "device" in setup_signature.parameters:
            kwargs["device"] = str(target_device)
        if "requested_dtype" in setup_signature.parameters:
            kwargs["requested_dtype"] = None
        return setup_model(**kwargs)

    build_model = getattr(runtime_module, "build_model", None)
    if not callable(build_model):
        build_model = getattr(accuracy_module, "build_model", None)
    if not callable(build_model):
        raise RuntimeError(f"{getattr(accuracy_module, '__file__', 'accuracy_run.py')} 缺少 setup_model()/build_model()，无法复用 local vision bridge")
    build_signature = inspect.signature(build_model)
    kwargs = {}
    if "use_pretrained" in build_signature.parameters:
        kwargs["use_pretrained"] = True
    if "device" in build_signature.parameters:
        kwargs["device"] = str(target_device)
    if "requested_dtype" in build_signature.parameters:
        kwargs["requested_dtype"] = None
    return build_model(**kwargs)


def _load_local_vision_classification_bridge_stack(config: dict[str, Any], model_id: str, scenario: str, target_device: torch.device, load_context: dict[str, Any]):
    if not _looks_like_local_vision_classification_bridge(config):
        return None

    accuracy_module = _load_module_from_path(f"_business_local_vision_accuracy_bridge_{abs(hash(str(ACCURACY_RUN_PATH.resolve())))}", ACCURACY_RUN_PATH)
    runtime_module = accuracy_module
    perf_module = None
    if ACCURACY_RUN_PERF_PATH.is_file():
        perf_module = _load_module_from_path(f"_business_local_vision_perf_bridge_{abs(hash(str(ACCURACY_RUN_PERF_PATH.resolve())))}", ACCURACY_RUN_PERF_PATH)
        if scenario == "npu_perf":
            apply_npu_optimizations = getattr(perf_module, "apply_npu_optimizations", None)
            if callable(apply_npu_optimizations):
                apply_npu_optimizations()
            runtime_module = perf_module

    helper_module = _resolve_local_vision_helper_module(accuracy_module)
    build_transform = getattr(helper_module, "build_transform", None)
    if not callable(build_transform):
        return None

    model, class_labels = _call_local_vision_bridge_setup_model(runtime_module, accuracy_module, target_device=target_device)
    if hasattr(model, "eval"):
        model.eval()
    if not isinstance(class_labels, list):
        class_labels = list(class_labels or [])

    load_context["model_source"] = str(model_id)
    load_context["model_source_kind"] = "accuracy_run_bridge"
    load_context["input_source"] = str(getattr(helper_module, "__file__", getattr(runtime_module, "__file__", ACCURACY_RUN_PATH)))
    load_context["input_source_kind"] = "local_runtime_script"

    input_adapter = SimpleNamespace(
        backend="local_vision_classification_bridge",
        runtime_module=runtime_module,
        accuracy_module=accuracy_module,
        perf_module=perf_module,
        helper_module=helper_module,
        transform=build_transform(),
        empty_cache_fn=getattr(helper_module, "empty_cache", getattr(runtime_module, "empty_cache", getattr(accuracy_module, "empty_cache", None))),
        class_labels=[str(label) for label in class_labels],
    )
    model_config = SimpleNamespace(
        id2label={idx: str(label) for idx, label in enumerate(class_labels)},
        num_labels=len(class_labels),
        model_type="vision_classification",
        architectures=["LocalVisionClassificationBridge"],
    )
    return model, input_adapter, model_config, "vision_classification", load_context


def _load_qwen_forced_aligner_stack(config: dict[str, Any], model_id: str, scenario: str, target_device: torch.device, load_context: dict[str, Any]):
    if not _looks_like_qwen_forced_aligner_model(model_id, config):
        return None

    from qwen_asr import Qwen3ForcedAligner

    model_source, _, load_context = _resolve_model_sources(model_id, config, scenario, "processor")
    source = model_source if Path(str(model_source)).exists() else model_id
    source_path = Path(str(source))
    source_kind = "local_snapshot" if source_path.exists() else "hub"

    torch_dtype = _get_torch_dtype(config)
    if torch_dtype == "auto":
        torch_dtype = torch.bfloat16 if target_device.type in {"npu", "cuda"} else torch.float32

    model = Qwen3ForcedAligner.from_pretrained(
        source,
        torch_dtype=torch_dtype,
        device_map=str(target_device),
        cache_dir=str(CACHE_DIR),
    )
    inner_model = getattr(model, "model", None) or getattr(model, "_model", None) or model
    if hasattr(inner_model, "to") and not hasattr(inner_model, "hf_device_map"):
        inner_model = inner_model.to(target_device)
    try:
        setattr(inner_model, "device", target_device)
    except Exception:
        pass

    patch_metadata = load_context.get("_patch_metadata")
    if isinstance(patch_metadata, dict):
        _apply_deferred_model_files_hooks(inner_model, patch_metadata)
        load_context["patch_load_status"] = str(patch_metadata.get("status") or load_context.get("patch_load_status") or "disabled")
        load_context["patch_modules"] = list(patch_metadata.get("imported_modules") or [])
        load_context["patch_hooks"] = list(patch_metadata.get("called_hooks") or [])
        load_context["patch_errors"] = list(patch_metadata.get("errors") or [])
    load_context["activation_name_shims"] = _stabilize_activation_name_shims(inner_model)

    if hasattr(inner_model, "eval"):
        inner_model.eval()

    load_context["model_source"] = str(source)
    load_context["model_source_kind"] = source_kind
    load_context["input_source"] = str(source)
    load_context["input_source_kind"] = source_kind

    model_config = SimpleNamespace(
        id2label={},
        num_labels=0,
        model_type="qwen3_forced_aligner",
        architectures=["Qwen3ForcedAligner"],
    )
    input_adapter = SimpleNamespace(backend="qwen_forced_aligner")
    return model, input_adapter, model_config, "asr", load_context


def _load_qwen_asr_stack(config: dict[str, Any], model_id: str, scenario: str, target_device: torch.device, load_context: dict[str, Any]):
    if not _looks_like_qwen_asr_model(model_id, config):
        return None

    from qwen_asr import Qwen3ASRModel

    model_source, _, load_context = _resolve_model_sources(model_id, config, scenario, "processor")
    source = model_source if Path(str(model_source)).exists() else model_id
    source_path = Path(str(source))
    source_kind = "local_snapshot" if source_path.exists() else "hub"

    torch_dtype = _get_torch_dtype(config)
    if torch_dtype == "auto":
        torch_dtype = torch.bfloat16 if target_device.type in {"npu", "cuda"} else torch.float32

    model = Qwen3ASRModel.from_pretrained(
        source,
        dtype=torch_dtype,
        device_map=str(target_device),
        max_inference_batch_size=max(_get_asr_batch_size(config, scenario), 1),
        max_new_tokens=int(config.get("asr_max_new_tokens") or config.get("max_new_tokens") or 256),
        cache_dir=str(CACHE_DIR),
    )
    inner_model = getattr(model, "model", None) or getattr(model, "_model", None) or model

    patch_metadata = load_context.get("_patch_metadata")
    if isinstance(patch_metadata, dict):
        _apply_deferred_model_files_hooks(inner_model, patch_metadata)
        load_context["patch_load_status"] = str(patch_metadata.get("status") or load_context.get("patch_load_status") or "disabled")
        load_context["patch_modules"] = list(patch_metadata.get("imported_modules") or [])
        load_context["patch_hooks"] = list(patch_metadata.get("called_hooks") or [])
        load_context["patch_errors"] = list(patch_metadata.get("errors") or [])
    load_context["activation_name_shims"] = _stabilize_activation_name_shims(inner_model)

    if hasattr(inner_model, "eval"):
        inner_model.eval()

    load_context["model_source"] = str(source)
    load_context["model_source_kind"] = source_kind
    load_context["input_source"] = str(source)
    load_context["input_source_kind"] = source_kind

    model_config = SimpleNamespace(
        id2label={},
        num_labels=0,
        model_type="qwen3_asr",
        architectures=["Qwen3ASRForConditionalGeneration"],
    )
    input_adapter = SimpleNamespace(backend="qwen_asr")
    return model, input_adapter, model_config, "asr", load_context


def _resolve_nemo_checkpoint_path(model_id: str, model_source: str, config: dict[str, Any]) -> tuple[Path, str]:
    configured_model_file = str(config.get("model_file") or "").strip()
    source_path = Path(str(model_source))

    if source_path.is_file() and source_path.suffix == ".nemo":
        return source_path, "local_checkpoint"

    if source_path.is_dir():
        local_candidates = sorted(source_path.rglob("*.nemo"))
        if configured_model_file:
            for candidate in local_candidates:
                if candidate.name == configured_model_file:
                    return candidate, "local_checkpoint"
        if len(local_candidates) == 1:
            return local_candidates[0], "local_checkpoint"

    from huggingface_hub import hf_hub_download

    checkpoint_name = configured_model_file or f"{str(model_id).strip().split('/')[-1]}.nemo"
    checkpoint_path = Path(hf_hub_download(repo_id=model_id, filename=checkpoint_name, cache_dir=str(CACHE_DIR)))
    return checkpoint_path, "hub_checkpoint"


def _load_nemo_asr_stack(config: dict[str, Any], model_id: str, scenario: str, target_device: torch.device, load_context: dict[str, Any]):
    if not _looks_like_nemo_asr_model(model_id, config):
        return None

    import nemo.collections.asr as nemo_asr

    model_source, _, load_context = _resolve_model_sources(model_id, config, scenario, "processor")
    checkpoint_path, source_kind = _resolve_nemo_checkpoint_path(model_id, model_source, config)

    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(str(checkpoint_path))
    model = model.to(target_device)

    patch_metadata = load_context.get("_patch_metadata")
    if isinstance(patch_metadata, dict):
        _apply_deferred_model_files_hooks(model, patch_metadata)
        load_context["patch_load_status"] = str(patch_metadata.get("status") or load_context.get("patch_load_status") or "disabled")
        load_context["patch_modules"] = list(patch_metadata.get("imported_modules") or [])
        load_context["patch_hooks"] = list(patch_metadata.get("called_hooks") or [])
        load_context["patch_errors"] = list(patch_metadata.get("errors") or [])
    load_context["activation_name_shims"] = _stabilize_activation_name_shims(model)

    model.eval()
    sample_rate = int(
        config.get("sample_rate")
        or getattr(getattr(getattr(model, "cfg", None), "preprocessor", None), "sample_rate", None)
        or getattr(getattr(model, "cfg", None), "sample_rate", None)
        or 16000
    )
    load_context["model_source"] = str(checkpoint_path)
    load_context["model_source_kind"] = source_kind
    load_context["input_source"] = "nemo_runtime_adapter"
    load_context["input_source_kind"] = "nemo_runtime_adapter"
    load_context["nemo_checkpoint"] = checkpoint_path.name

    model_config = SimpleNamespace(
        id2label={},
        num_labels=0,
        model_type="nemo_asr",
        architectures=["EncDecRNNTBPEModel"],
    )
    input_adapter = SimpleNamespace(backend="nemo_asr", sample_rate=sample_rate)
    return model, input_adapter, model_config, "asr", load_context


def _apply_model_runtime_compatibility_shims(model) -> list[str]:
    applied: list[str] = []
    candidate_modules: list[Any] = []
    seen_candidates: set[int] = set()

    def _push_candidate(candidate: Any) -> None:
        if candidate is None:
            return
        candidate_id = id(candidate)
        if candidate_id in seen_candidates:
            return
        seen_candidates.add(candidate_id)
        candidate_modules.append(candidate)

    _push_candidate(model)
    _push_candidate(getattr(model, "transformer", None))
    nested_model = getattr(model, "model", None)
    _push_candidate(nested_model)
    _push_candidate(getattr(model, "language_model", None))
    _push_candidate(getattr(model, "llm", None))
    if nested_model is not None:
        _push_candidate(getattr(nested_model, "language_model", None))
        _push_candidate(getattr(nested_model, "llm", None))
    applied.extend(_apply_birefnet_npu_deform_conv_compat(candidate_modules))

    def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is None:
            return [None] * num_hidden_layers
        if hasattr(self, "_convert_head_mask_to_5d"):
            converted = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        elif head_mask.dim() == 1:
            converted = head_mask.view(1, 1, -1, 1, 1).expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            converted = head_mask[:, None, :, None, None]
        else:
            raise ValueError(f"Unsupported head_mask ndim without _convert_head_mask_to_5d: {head_mask.dim()}")
        if is_attention_chunked:
            converted = converted.unsqueeze(-1)
        return converted

    generation_config_cls = None
    generation_mixin = None
    try:
        from transformers import GenerationConfig as _GenerationConfig
    except Exception:
        _GenerationConfig = None
    try:
        from transformers.generation import GenerationMixin as _GenerationMixin
    except Exception:
        try:
            from transformers import GenerationMixin as _GenerationMixin
        except Exception:
            _GenerationMixin = None
    generation_config_cls = _GenerationConfig
    generation_mixin = _GenerationMixin

    def _ensure_generation_config(candidate: Any, candidate_cls: type) -> None:
        if generation_config_cls is None or hasattr(candidate, "generation_config"):
            return
        candidate_config = getattr(candidate, "config", None)
        try:
            if candidate_config is not None:
                candidate.generation_config = generation_config_cls.from_model_config(candidate_config)
            else:
                candidate.generation_config = generation_config_cls()
            applied.append(f"{candidate_cls.__name__}.generation_config")
        except Exception:
            try:
                candidate.generation_config = generation_config_cls()
                applied.append(f"{candidate_cls.__name__}.generation_config")
            except Exception:
                pass

    def _should_preserve_prepare_inputs_cache(candidate: Any, past_key_values: Any) -> bool:
        if past_key_values is None or isinstance(past_key_values, tuple):
            return False

        candidate_config = getattr(candidate, "config", None)
        candidate_model_type = str(getattr(candidate_config, "model_type", "") or "").strip().lower()
        candidate_architectures = [str(item).strip().lower() for item in (getattr(candidate_config, "architectures", None) or []) if str(item).strip()]
        candidate_hints = " ".join(
            part
            for part in (
                candidate_model_type,
                candidate.__class__.__name__.lower(),
                " ".join(candidate_architectures),
            )
            if part
        )
        if any(token in candidate_hints for token in ("internlm2", "internvl")):
            return False

        # transformers 5.x cache objects expose masking helpers that break if we
        # coerce them back to legacy tuples before prepare_inputs_for_generation.
        if hasattr(past_key_values, "get_mask_sizes"):
            return True

        if any(token in candidate_hints for token in ("qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe")):
            return True
        return False

    def _wrap_prepare_inputs_for_generation(candidate: Any, candidate_cls: type) -> None:
        if not hasattr(candidate, "prepare_inputs_for_generation"):
            return
        if getattr(candidate, "_slai_prepare_inputs_dynamic_cache_compat", False):
            return
        original_prepare = candidate.prepare_inputs_for_generation

        def _wrapped_prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
            if past_key_values is not None and not isinstance(past_key_values, tuple) and not _should_preserve_prepare_inputs_cache(self, past_key_values):
                if hasattr(past_key_values, "to_legacy_cache"):
                    try:
                        past_key_values = past_key_values.to_legacy_cache()
                    except Exception:
                        pass
                elif hasattr(past_key_values, "layers"):
                    try:
                        legacy_layers = []
                        for layer in list(getattr(past_key_values, "layers", []) or []):
                            key_states = getattr(layer, "keys", None)
                            value_states = getattr(layer, "values", None)
                            if key_states is None or value_states is None:
                                continue
                            legacy_layers.append((key_states, value_states))
                        past_key_values = tuple(legacy_layers) if legacy_layers else None
                    except Exception:
                        pass
            return original_prepare(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        candidate.prepare_inputs_for_generation = MethodType(_wrapped_prepare_inputs_for_generation, candidate)
        candidate._slai_prepare_inputs_dynamic_cache_compat = True
        applied.append(f"{candidate_cls.__name__}.prepare_inputs_for_generation")

    for candidate in candidate_modules:
        if candidate is None or hasattr(candidate, "get_head_mask"):
            candidate_config = getattr(candidate, "config", None)
        else:
            candidate_config = getattr(candidate, "config", None)
            num_hidden_layers = getattr(candidate_config, "num_hidden_layers", None)
            if num_hidden_layers is None:
                num_hidden_layers = getattr(candidate_config, "n_layer", None)
            if num_hidden_layers is not None:
                candidate.get_head_mask = MethodType(_get_head_mask, candidate)
                applied.append(f"{candidate.__class__.__name__}.get_head_mask")

        if generation_mixin is None:
            _wrap_prepare_inputs_for_generation(candidate, candidate.__class__)
            continue
        if hasattr(candidate, "generate") or not hasattr(candidate, "prepare_inputs_for_generation"):
            _wrap_prepare_inputs_for_generation(candidate, candidate.__class__)
            continue

        candidate_cls = candidate.__class__
        if not issubclass(candidate_cls, generation_mixin):
            shim_cls = _GENERATION_MIXIN_SHIM_CLASSES.get(candidate_cls)
            if shim_cls is None:
                shim_attrs: dict[str, Any] = {
                    "__module__": candidate_cls.__module__,
                    "_slai_generation_mixin_shim": True,
                }
                if not hasattr(candidate_cls, "can_generate"):

                    @classmethod
                    def _can_generate(cls) -> bool:
                        return True

                    shim_attrs["can_generate"] = _can_generate
                shim_cls = type(f"{candidate_cls.__name__}GenerationShim", (candidate_cls, generation_mixin), shim_attrs)
                _GENERATION_MIXIN_SHIM_CLASSES[candidate_cls] = shim_cls
            try:
                if candidate.__class__ is not shim_cls:
                    candidate.__class__ = shim_cls
                _wrap_prepare_inputs_for_generation(candidate, candidate_cls)
                _ensure_generation_config(candidate, candidate_cls)
                applied.append(f"{candidate_cls.__name__}.GenerationMixin")
                continue
            except TypeError:
                pass

        fallback_methods: list[str] = []
        for method_name in dir(generation_mixin):
            if method_name.startswith("__") or hasattr(candidate, method_name):
                continue
            method_impl = getattr(generation_mixin, method_name, None)
            if isinstance(method_impl, property) or not callable(method_impl):
                continue
            setattr(candidate, method_name, MethodType(method_impl, candidate))
            fallback_methods.append(method_name)
        if not hasattr(candidate, "can_generate"):
            candidate.can_generate = MethodType(lambda self: True, candidate)
            fallback_methods.append("can_generate")
        _wrap_prepare_inputs_for_generation(candidate, candidate_cls)
        _ensure_generation_config(candidate, candidate_cls)
        if fallback_methods:
            applied.append(f"{candidate_cls.__name__}.GenerationFallback[{','.join(sorted(fallback_methods))}]")
    return applied


def _apply_tokenizer_runtime_compatibility_shims(tokenizer) -> list[str]:
    if tokenizer is None:
        return []

    applied: list[str] = []
    if not hasattr(tokenizer, "clean_up_tokenization"):

        def _clean_up_tokenization(self, text: str) -> str:
            cleaned = str(text)
            if not getattr(self, "clean_up_tokenization_spaces", False):
                return cleaned
            replacements = (
                (" .", "."),
                (" ?", "?"),
                (" !", "!"),
                (" ,", ","),
                (" ' ", "'"),
                (" n't", "n't"),
                (" 'm", "'m"),
                (" 's", "'s"),
                (" 've", "'ve"),
                (" 're", "'re"),
                (" 'd", "'d"),
                (" 'll", "'ll"),
            )
            for source, target in replacements:
                cleaned = cleaned.replace(source, target)
            return cleaned

        tokenizer.clean_up_tokenization = MethodType(_clean_up_tokenization, tokenizer)
        applied.append(f"{tokenizer.__class__.__name__}.clean_up_tokenization")
    return applied


def _apply_deepseek_moe_npu_infer_patch(model) -> list[str]:
    applied: list[str] = []

    def _patched_moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        if getattr(x, "device", None) is None or str(x.device).split(":", 1)[0] != "npu":
            return self._slai_original_moe_infer(x, flat_expert_indices, flat_expert_weights)

        expert_cache = torch.zeros_like(x)
        token_indices = torch.arange(flat_expert_indices.numel(), device=flat_expert_indices.device, dtype=torch.long)
        token_indices = torch.div(token_indices, int(self.num_experts_per_tok), rounding_mode="floor")
        flat_weights = flat_expert_weights.view(-1, 1)

        for expert_idx, expert in enumerate(self.experts):
            expert_mask = flat_expert_indices.eq(expert_idx)
            expert_token_idx = token_indices[expert_mask]
            if expert_token_idx.numel() == 0:
                continue
            expert_tokens = x.index_select(0, expert_token_idx)
            expert_out = expert(expert_tokens)
            expert_out = expert_out * flat_weights[expert_mask]
            expert_cache.index_add_(0, expert_token_idx, expert_out)
        return expert_cache

    for module in model.modules():
        if module.__class__.__name__ != "DeepseekMoE":
            continue
        if getattr(module, "_slai_moe_infer_index_add_patch", False):
            continue
        original_moe_infer = getattr(module, "moe_infer", None)
        if not callable(original_moe_infer):
            continue
        module._slai_original_moe_infer = original_moe_infer
        module.moe_infer = MethodType(_patched_moe_infer, module)
        module._slai_moe_infer_index_add_patch = True
        applied.append(f"{module.__class__.__name__}.moe_infer_index_add")
    return applied


@contextmanager
def _temporary_tensor_cuda_redirect(device: torch.device):
    if str(getattr(device, "type", "") or "") == "cuda":
        yield
        return

    original_cuda = torch.Tensor.cuda

    def _redirect_cuda(self, device_arg=None, non_blocking=False, memory_format=torch.preserve_format):
        target_device = device
        if device_arg is not None:
            try:
                requested_device = torch.device(device_arg)
            except Exception:
                requested_device = None
            if requested_device is not None and requested_device.type != "cuda":
                target_device = requested_device
        move_kwargs: dict[str, Any] = {
            "device": target_device,
            "non_blocking": non_blocking,
        }
        try:
            return self.to(memory_format=memory_format, **move_kwargs)
        except TypeError:
            return self.to(**move_kwargs)

    torch.Tensor.cuda = _redirect_cuda
    try:
        yield
    finally:
        torch.Tensor.cuda = original_cuda


def _decode_generated(tokens, tokenizer, input_len: int | None = None) -> str:
    if input_len is not None and tokens.ndim == 1:
        tokens = tokens[input_len:]
    return tokenizer.decode(tokens, skip_special_tokens=True).strip()


def _sample_has_choices(sample: dict) -> bool:
    choices = sample.get("choices")
    return isinstance(choices, list) and any(str(choice).strip() for choice in choices)


def _sample_dataset_key(sample: dict) -> str:
    return str(sample.get("dataset_key") or "").strip().lower()


def _decode_image_sample(value: Any):
    if value is None:
        return None
    try:
        from PIL import Image
    except ImportError:
        Image = None
    if Image is not None and isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        raw_bytes = value.get("bytes")
        if isinstance(raw_bytes, (bytes, bytearray)) and raw_bytes:
            if Image is None:
                return None
            return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        raw_path = str(value.get("path") or "").strip()
        if raw_path and Path(raw_path).is_file():
            if Image is None:
                return None
            return Image.open(raw_path).convert("RGB")
    return value


def _tokenizer_has_token(tokenizer, token: str) -> bool:
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        vocab = {}
    if token in vocab:
        return True
    try:
        added_vocab = tokenizer.get_added_vocab()
    except Exception:
        added_vocab = {}
    return token in added_vocab


def _ensure_vlm_image_token(tokenizer, *, image_token: str = "<image>") -> int:
    resolved_image_token = str(image_token or "<image>").strip() or "<image>"
    token_id = tokenizer.convert_tokens_to_ids(resolved_image_token)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    token_missing = not _tokenizer_has_token(tokenizer, resolved_image_token) or token_id is None or (unk_token_id is not None and token_id == unk_token_id)
    if token_missing and hasattr(tokenizer, "add_special_tokens"):
        tokenizer.add_special_tokens({"additional_special_tokens": [resolved_image_token]})
        token_id = tokenizer.convert_tokens_to_ids(resolved_image_token)
    setattr(tokenizer, "image_token", resolved_image_token)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if token_id is None or (unk_token_id is not None and token_id == unk_token_id):
        raise RuntimeError(f"无法为 VLM tokenizer 注册 image token: {resolved_image_token}")
    return int(token_id)


def _deepseek_find_closest_aspect_ratio(aspect_ratio: float, target_ratios: list[tuple[int, int]], width: int, height: int, image_size: int) -> tuple[int, int]:
    best_ratio = (1, 1)
    best_ratio_diff = float("inf")
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
            continue
        if ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def _deepseek_dynamic_preprocess(image, *, min_num: int = 2, max_num: int = 9, image_size: int = 640):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / max(orig_height, 1)
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if min_num <= i * j <= max_num},
        key=lambda value: value[0] * value[1],
    )
    target_aspect_ratio = _deepseek_find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        orig_width,
        orig_height,
        image_size,
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for idx in range(blocks):
        box = (
            (idx % (target_width // image_size)) * image_size,
            (idx // (target_width // image_size)) * image_size,
            ((idx % (target_width // image_size)) + 1) * image_size,
            ((idx // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    return processed_images, target_aspect_ratio


def _deepseek_text_encode(tokenizer, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
    token_ids = list(tokenizer.encode(text, add_special_tokens=False))
    bos_id = getattr(tokenizer, "bos_token_id", 0)
    eos_id = getattr(tokenizer, "eos_token_id", 1)
    if bos:
        token_ids = [0 if bos_id is None else int(bos_id)] + token_ids
    if eos:
        token_ids = token_ids + [1 if eos_id is None else int(eos_id)]
    return token_ids


def _prepare_deepseek_vl_v2_inputs(tokenizer, sample: dict, *, model_device: torch.device, model_dtype: torch.dtype | None):
    from PIL import ImageOps

    image = _decode_image_sample(sample.get("image"))
    prompt_text = _build_vlm_prompt(sample)
    if image is not None:
        prompt_text = f"<image>\n{prompt_text}".strip()

    image_token = str(getattr(tokenizer, "_business_image_token", "") or "<image>").strip() or "<image>"
    image_token_id = _ensure_vlm_image_token(tokenizer, image_token=image_token)
    base_size = int(getattr(tokenizer, "_business_deepseek_base_size", 1024) or 1024)
    image_size = int(getattr(tokenizer, "_business_deepseek_image_size", min(base_size, 640)) or min(base_size, 640))
    patch_size = int(getattr(tokenizer, "_business_deepseek_patch_size", 16) or 16)
    downsample_ratio = int(getattr(tokenizer, "_business_deepseek_downsample_ratio", 4) or 4)
    crop_mode = _as_bool(getattr(tokenizer, "_business_deepseek_crop_mode", True), default=True)
    try:
        from torchvision import transforms
    except ImportError:
        transforms = None

    if transforms is not None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
    else:

        def transform(image_value):
            array = np.asarray(image_value.convert("RGB"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            mean = torch.tensor((0.5, 0.5, 0.5), dtype=tensor.dtype).view(3, 1, 1)
            std = torch.tensor((0.5, 0.5, 0.5), dtype=tensor.dtype).view(3, 1, 1)
            return (tensor - mean) / std

    tokenized_str: list[int] = []
    images_seq_mask: list[bool] = []
    images_list: list[torch.Tensor] = []
    images_crop_list: list[torch.Tensor] = []
    images_spatial_crop: list[list[int]] = []
    image_samples = [image] if image is not None else []
    text_splits = prompt_text.split(image_token)
    fill_color = (127, 127, 127)
    if model_device.type in {"cuda", "npu"}:
        vision_dtype = model_dtype if model_dtype in {torch.float16, torch.bfloat16} else torch.bfloat16
    else:
        vision_dtype = torch.float32

    for text_sep, image_value in zip(text_splits, image_samples):
        tokenized_sep = _deepseek_text_encode(tokenizer, text_sep, bos=False, eos=False)
        tokenized_str.extend(tokenized_sep)
        images_seq_mask.extend([False] * len(tokenized_sep))

        images_crop_raw = []
        if crop_mode and max(image_value.size) > image_size:
            images_crop_raw, crop_ratio = _deepseek_dynamic_preprocess(image_value, image_size=image_size)
        else:
            crop_ratio = (1, 1)

        global_view = ImageOps.pad(image_value, (base_size, base_size), color=fill_color)
        images_list.append(transform(global_view).to(dtype=vision_dtype))

        width_crop_num, height_crop_num = int(crop_ratio[0]), int(crop_ratio[1])
        images_spatial_crop.append([width_crop_num, height_crop_num])
        if width_crop_num > 1 or height_crop_num > 1:
            for crop_image in images_crop_raw:
                images_crop_list.append(transform(crop_image).to(dtype=vision_dtype))

        num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
        num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)
        tokenized_image = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base
        tokenized_image += [image_token_id]
        if width_crop_num > 1 or height_crop_num > 1:
            tokenized_image += ([image_token_id] * (num_queries * width_crop_num) + [image_token_id]) * (num_queries * height_crop_num)
        tokenized_str.extend(tokenized_image)
        images_seq_mask.extend([True] * len(tokenized_image))

    last_text_split = text_splits[-1] if text_splits else prompt_text
    tokenized_sep = _deepseek_text_encode(tokenizer, last_text_split, bos=False, eos=False)
    tokenized_str = [int(getattr(tokenizer, "bos_token_id", 0) or 0)] + tokenized_str + tokenized_sep
    images_seq_mask = [False] + images_seq_mask + [False] * len(tokenized_sep)

    input_ids = torch.tensor(tokenized_str, dtype=torch.long)
    images_seq_mask_tensor = torch.tensor(images_seq_mask, dtype=torch.bool)

    if images_list:
        images_ori = torch.stack(images_list, dim=0)
        images_spatial_crop_tensor = torch.tensor(images_spatial_crop, dtype=torch.long)
        if images_crop_list:
            images_crop = torch.stack(images_crop_list, dim=0)
        else:
            images_crop = torch.zeros((1, 3, base_size, base_size), dtype=vision_dtype)
    else:
        images_ori = torch.zeros((1, 3, image_size, image_size), dtype=vision_dtype)
        images_spatial_crop_tensor = torch.zeros((1, 2), dtype=torch.long)
        images_crop = torch.zeros((1, 3, base_size, base_size), dtype=vision_dtype)

    processed = {
        "input_ids": input_ids.unsqueeze(0).to(model_device),
        "attention_mask": torch.ones((1, input_ids.shape[0]), dtype=torch.long, device=model_device),
        "images": [(images_crop.to(model_device), images_ori.to(model_device))],
        "images_seq_mask": images_seq_mask_tensor.unsqueeze(0).to(model_device),
        "images_spatial_crop": images_spatial_crop_tensor,
    }
    return processed, int(input_ids.shape[0])


def _run_deepseek_vl_v2(model, tokenizer, samples: list[dict], config: dict[str, Any] | None = None) -> tuple[list[str], dict]:
    predictions: list[str] = []
    model_device = _get_model_device(model)
    model_dtype = _get_model_floating_dtype(model)

    for sample in samples:
        processed, prompt_length = _prepare_deepseek_vl_v2_inputs(
            tokenizer,
            sample,
            model_device=model_device,
            model_dtype=model_dtype,
        )
        generation_kwargs = _generation_kwargs_for_sample(sample, tokenizer, config=config)
        generation_kwargs.setdefault("use_cache", True)

        with torch.no_grad():
            with _temporary_tensor_cuda_redirect(model_device):
                outputs = model.generate(**processed, **generation_kwargs)

        generated_ids = outputs
        if len(outputs.shape) == 2 and outputs.shape[1] > prompt_length:
            generated_ids = outputs[:, prompt_length:]
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if _sample_dataset_key(sample) == "scienceqa" or _sample_has_choices(sample):
            predictions.append(_normalize_scienceqa_prediction(decoded, sample))
        else:
            predictions.append(_extract_assistant_text(decoded) or str(decoded or "").strip())

    return predictions, {"inference_strategy": "deepseek_vl_v2_generate"}


def _normalize_generated_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _extract_assistant_text(decoded_text: str) -> str:
    text = str(decoded_text or "").strip()
    if not text:
        return ""
    for marker in ("\nassistant\n", "\nassistant:", "assistant\n", "assistant:"):
        lower_text = text.lower()
        idx = lower_text.rfind(marker)
        if idx >= 0:
            text = text[idx + len(marker) :].strip()
    return text


def _extract_choice_letter(answer_text: str, num_choices: int) -> str | None:
    if num_choices <= 0:
        return None
    patterns = (
        r"^\s*([A-Z])\s*[\.\):\-]",
        r"\b(?:answer|option|choice)\s*[:：]?\s*([A-Z])\b",
        r"\(([A-Z])\)",
        r"\b([A-Z])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, answer_text, flags=re.IGNORECASE)
        if not match:
            continue
        letter = str(match.group(1) or "").upper()
        if "A" <= letter < chr(ord("A") + num_choices):
            return letter
    return None


def _has_rejection_pattern(answer_text: str) -> bool:
    normalized = _normalize_generated_text(answer_text)
    patterns = (
        "not provided in the given choices",
        "not possible to select an answer",
        "cannot determine",
        "can't determine",
        "unable to determine",
        "none of the above",
        "neither option",
        "neither a nor b",
        "neither b nor a",
        "do not match",
    )
    return any(pattern in normalized for pattern in patterns)


def _normalize_scienceqa_prediction(decoded_text: str, sample: dict) -> str:
    text = _extract_assistant_text(decoded_text)
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidate_texts: list[str] = []
    if lines:
        candidate_texts.append(lines[0])
        candidate_texts.append(lines[-1])
    candidate_texts.append(text.strip())

    choices = sample.get("choices")
    if isinstance(choices, list) and choices:
        cleaned_choices = [str(choice or "").strip() for choice in choices]
        normalized_choice_map = {_normalize_generated_text(choice): choice for choice in cleaned_choices if choice}

        for candidate in candidate_texts:
            normalized_candidate = _normalize_generated_text(candidate)
            if normalized_candidate in normalized_choice_map:
                return normalized_choice_map[normalized_candidate]

        for candidate in candidate_texts:
            if _has_rejection_pattern(candidate):
                continue
            letter = _extract_choice_letter(candidate, len(cleaned_choices))
            if letter is None:
                continue
            choice_index = ord(letter) - ord("A")
            if 0 <= choice_index < len(cleaned_choices):
                return cleaned_choices[choice_index]

        for candidate in candidate_texts:
            if _has_rejection_pattern(candidate):
                continue
            normalized_candidate = _normalize_generated_text(candidate)
            for normalized_choice, original_choice in normalized_choice_map.items():
                if normalized_choice and normalized_choice in normalized_candidate:
                    return original_choice

    return lines[-1] if lines else text


def _extract_pubmed_qa_decision(raw_text: Any) -> str:
    text = _normalize_generated_text(raw_text)
    if not text:
        return ""
    direct_match = re.fullmatch(r"(yes|no|maybe)", text)
    if direct_match:
        return direct_match.group(1)
    answer_match = re.search(r"(?:final answer|answer)\s*[:：-]?\s*(yes|no|maybe)\b", text)
    if answer_match:
        return answer_match.group(1)
    token_match = re.search(r"\b(yes|no|maybe)\b", text)
    if token_match:
        return token_match.group(1)
    return text


def _build_generation_prompt(sample: dict) -> str:
    prompt = str(sample.get("input") or "").strip()
    dataset_key = _sample_dataset_key(sample)
    if dataset_key == "gsm8k" and prompt:
        return "\n".join(
            [
                "Solve the following math word problem.",
                "Return only the final numeric answer.",
                "Do not include reasoning or explanation.",
                "Use the format `#### <answer>`.",
                "",
                prompt,
            ]
        ).strip()
    choices = sample.get("choices")
    if isinstance(choices, list) and choices:
        option_lines: list[str] = []
        for idx, choice in enumerate(choices):
            choice_text = str(choice).strip()
            if not choice_text:
                continue
            label = chr(65 + idx) if 0 <= idx < 26 else str(idx)
            option_lines.append(f"{label}. {choice_text}")
        if option_lines:
            prompt = "\n".join(
                [
                    prompt,
                    *option_lines,
                    "Answer with only the single correct option letter.",
                    "Do not add any explanation.",
                    "Answer:",
                ]
            ).strip()
    if dataset_key == "pubmed_qa" and prompt:
        return "\n".join(
            [
                prompt,
                "Answer with exactly one word: yes, no, or maybe.",
                "Do not add any explanation.",
                "Answer:",
            ]
        ).strip()
    return prompt


def _build_vlm_prompt(sample: dict) -> str:
    prompt = str(sample.get("input") or "").strip()
    choices = sample.get("choices")
    if isinstance(choices, list) and choices:
        option_lines: list[str] = []
        for idx, choice in enumerate(choices):
            choice_text = str(choice).strip()
            if not choice_text:
                continue
            label = chr(65 + idx) if 0 <= idx < 26 else str(idx)
            option_lines.append(f"{label}. {choice_text}")
        if option_lines:
            prompt = "\n".join(
                [
                    prompt,
                    *option_lines,
                    "Answer with only the single correct option letter.",
                    "Do not add any explanation.",
                    "Answer:",
                ]
            ).strip()
    return prompt


def _render_chat_prompt(tokenizer, prompt: str, *, disable_thinking: bool = False) -> str:
    has_chat_template = hasattr(tokenizer, "apply_chat_template") and callable(getattr(tokenizer, "apply_chat_template", None)) and getattr(tokenizer, "chat_template", None) is not None
    if not has_chat_template:
        return prompt

    messages = [{"role": "user", "content": prompt}]
    render_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if disable_thinking:
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=False, **render_kwargs)
        except TypeError:
            pass
        except Exception:
            pass
    return tokenizer.apply_chat_template(messages, **render_kwargs)


def _render_vlm_prompt(processor, prompt: str, *, has_image: bool, model_family: str = "") -> str:
    if model_family == "llava":
        if has_image:
            return f"USER: <image>\n{prompt}\nASSISTANT:"
        return f"USER: {prompt}\nASSISTANT:"

    if model_family in {"qwen2_5_vl", "qwen2_vl", "qwen3_vl", "qwen3_vl_moe"}:
        user_prefix = "<|vision_start|><|image_pad|><|vision_end|>\n" if has_image else ""
        return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user_prefix}{prompt}<|im_end|>\n<|im_start|>assistant\n"

    has_chat_template = hasattr(processor, "apply_chat_template") and callable(getattr(processor, "apply_chat_template", None)) and getattr(processor, "chat_template", None) is not None
    if has_chat_template:
        if has_image:
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        try:
            return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except TypeError:
            pass
        except Exception:
            pass

    if has_image:
        return f"<image>\n{prompt}"
    return prompt


def _generation_kwargs_for_sample(sample: dict, tokenizer, config: dict[str, Any] | None = None) -> dict[str, Any]:
    max_new_tokens = 256
    dataset_key = _sample_dataset_key(sample)
    if dataset_key == "gsm8k":
        max_new_tokens = 64
    elif dataset_key == "pubmed_qa":
        # PubMedQA ultimately normalizes to yes/no/maybe, so a short decode budget
        # reduces cross-device drift from verbose generations.
        max_new_tokens = 4
    if isinstance(config, dict):
        override_value = config.get(f"{dataset_key}_max_new_tokens")
        if override_value is None and dataset_key == "pubmed_qa":
            override_value = config.get("qa_max_new_tokens")
        if override_value is None:
            override_value = config.get("max_new_tokens")
        try:
            parsed_override = int(override_value)
        except (TypeError, ValueError):
            parsed_override = None
        if parsed_override and parsed_override > 0:
            max_new_tokens = parsed_override
    choices = sample.get("choices")
    if isinstance(choices, list) and choices:
        # Multiple-choice tasks such as MMLU only need a short option label.
        max_new_tokens = 16

    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    return kwargs


def _should_disable_generation_cache(model, config: dict[str, Any] | None = None) -> bool:
    model_config = getattr(model, "config", None)
    architectures = " ".join(
        str(item).strip().lower()
        for item in list(getattr(model_config, "architectures", []) or [])
        if str(item).strip()
    )
    hint_text = " ".join(
        str(part).strip().lower()
        for part in (
            getattr(model, "name_or_path", None),
            getattr(model_config, "model_type", None),
            architectures,
            config.get("model_id") if isinstance(config, dict) else None,
            config.get("model_class") if isinstance(config, dict) else None,
            config.get("architectures") if isinstance(config, dict) else None,
        )
        if str(part or "").strip()
    )
    return any(token in hint_text for token in ("phi-3", "phi3", "quiet-star", "quiet_star", "modeling_quiet", "quietforcausallm"))


def _should_skip_multiple_choice_scoring(model, config: dict[str, Any] | None = None) -> bool:
    model_config = getattr(model, "config", None)
    architectures = " ".join(
        str(item).strip().lower()
        for item in list(getattr(model_config, "architectures", []) or [])
        if str(item).strip()
    )
    hint_text = " ".join(
        str(part).strip().lower()
        for part in (
            getattr(model, "name_or_path", None),
            getattr(model_config, "model_type", None),
            architectures,
            config.get("model_id") if isinstance(config, dict) else None,
            config.get("model_class") if isinstance(config, dict) else None,
            config.get("architectures") if isinstance(config, dict) else None,
        )
        if str(part or "").strip()
    )
    return any(token in hint_text for token in ("crystalcareai/quiet-star-custom", "quiet-star", "quiet_star", "modeling_quiet", "quietforcausallm"))


def _forward_causal_lm_no_cache(model_like, *, input_ids, attention_mask=None):
    forward_kwargs = {"input_ids": input_ids}
    if attention_mask is not None:
        forward_kwargs["attention_mask"] = attention_mask
    try:
        return model_like(**forward_kwargs, use_cache=False)
    except TypeError as exc:
        if "use_cache" not in str(exc):
            raise
        return model_like(**forward_kwargs)


def _score_multiple_choice_candidate(model, tokenizer, prompt_text: str, label: str) -> float:
    model_device = next(model.parameters()).device
    candidate_text = f" {label}"
    prompt_inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        add_special_tokens=False,
    )
    full_inputs = tokenizer(
        prompt_text + candidate_text,
        return_tensors="pt",
        truncation=True,
        max_length=1040,
        add_special_tokens=False,
    )
    prompt_len = int(prompt_inputs["input_ids"].shape[-1])
    input_ids = full_inputs["input_ids"].to(model_device)
    attention_mask = full_inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model_device)

    if input_ids.shape[-1] <= prompt_len or prompt_len <= 0:
        return float("-inf")

    with torch.no_grad():
        try:
            outputs = _forward_causal_lm_no_cache(model, input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        except Exception as exc:
            # Some trust_remote_code causal LMs still assume the legacy cache API
            # inside `model(...)` even for scoring. Fall back to the inner decoder
            # path that benchmark scripts already use for no-cache evaluation.
            if not hasattr(model, "model") or not hasattr(model, "lm_head"):
                raise
            if not any(token in str(exc) for token in ("get_usable_length", "DynamicCache", "from_legacy_cache")):
                raise
            model_outputs = _forward_causal_lm_no_cache(model.model, input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = model_outputs.last_hidden_state if hasattr(model_outputs, "last_hidden_state") else model_outputs[0]
            logits = model.lm_head(hidden_states)

    target_ids = input_ids[:, prompt_len:]
    candidate_logits = logits[:, prompt_len - 1 : -1, :]
    if candidate_logits.shape[1] <= 0 or target_ids.shape[1] <= 0:
        return float("-inf")
    if candidate_logits.shape[1] != target_ids.shape[1]:
        common_len = min(candidate_logits.shape[1], target_ids.shape[1])
        candidate_logits = candidate_logits[:, :common_len, :]
        target_ids = target_ids[:, :common_len]

    log_probs = torch.log_softmax(candidate_logits, dim=-1)
    gathered = torch.gather(log_probs, -1, target_ids.unsqueeze(-1)).squeeze(-1)
    return float(gathered.sum().item())


def _encode_multiple_choice_label_token(tokenizer, label: str) -> int | None:
    candidate_text = f" {label}"
    try:
        token_ids = tokenizer.encode(candidate_text, add_special_tokens=False)
    except Exception:
        token_ids = []
    if len(token_ids) != 1:
        try:
            token_ids = tokenizer.encode(str(label), add_special_tokens=False)
        except Exception:
            token_ids = []
    if len(token_ids) != 1:
        return None
    return int(token_ids[0])


def _score_multiple_choice_single_token_candidates(model, tokenizer, prompt_text: str, labels: list[str]) -> list[float] | None:
    if not labels:
        return []

    label_token_ids: list[int] = []
    for label in labels:
        token_id = _encode_multiple_choice_label_token(tokenizer, label)
        if token_id is None:
            return None
        label_token_ids.append(token_id)

    model_device = next(model.parameters()).device
    prompt_inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        add_special_tokens=False,
    )
    if not isinstance(prompt_inputs, Mapping):
        return None

    input_ids = prompt_inputs.get("input_ids")
    if input_ids is None or input_ids.shape[-1] <= 0:
        return None
    input_ids = input_ids.to(model_device)
    attention_mask = prompt_inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model_device)

    with torch.no_grad():
        try:
            outputs = _forward_causal_lm_no_cache(model, input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        except Exception as exc:
            if not hasattr(model, "model") or not hasattr(model, "lm_head"):
                raise
            if not any(token in str(exc) for token in ("get_usable_length", "DynamicCache", "from_legacy_cache")):
                raise
            model_outputs = _forward_causal_lm_no_cache(model.model, input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = model_outputs.last_hidden_state if hasattr(model_outputs, "last_hidden_state") else model_outputs[0]
            logits = model.lm_head(hidden_states)

    last_token_logits = logits[:, -1, :]
    log_probs = torch.log_softmax(last_token_logits, dim=-1)
    index_tensor = torch.tensor(label_token_ids, device=log_probs.device, dtype=torch.long).unsqueeze(0)
    scores = torch.gather(log_probs, -1, index_tensor).squeeze(0)
    return [float(score) for score in scores.detach().cpu().tolist()]


def _score_multiple_choice_single_token_candidates_batched(model, tokenizer, prompt_texts: list[str], labels: list[str]) -> list[list[float]] | None:
    if not prompt_texts or not labels:
        return [[] for _ in prompt_texts]

    label_token_ids: list[int] = []
    for label in labels:
        token_id = _encode_multiple_choice_label_token(tokenizer, label)
        if token_id is None:
            return None
        label_token_ids.append(token_id)

    model_device = next(model.parameters()).device
    prompt_inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=1024,
        add_special_tokens=False,
    )
    if not isinstance(prompt_inputs, Mapping):
        return None

    input_ids = prompt_inputs.get("input_ids")
    if input_ids is None or input_ids.shape[-1] <= 0:
        return None
    input_ids = input_ids.to(model_device)
    attention_mask = prompt_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(model_device)
    last_indices = attention_mask.sum(dim=1) - 1

    with torch.no_grad():
        try:
            outputs = _forward_causal_lm_no_cache(model, input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        except Exception as exc:
            if not hasattr(model, "model") or not hasattr(model, "lm_head"):
                raise
            if not any(token in str(exc) for token in ("get_usable_length", "DynamicCache", "from_legacy_cache")):
                raise
            model_outputs = _forward_causal_lm_no_cache(model.model, input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = model_outputs.last_hidden_state if hasattr(model_outputs, "last_hidden_state") else model_outputs[0]
            logits = model.lm_head(hidden_states)

    batch_indices = torch.arange(input_ids.shape[0], device=logits.device)
    last_token_logits = logits[batch_indices, last_indices, :]
    log_probs = torch.log_softmax(last_token_logits, dim=-1)
    index_tensor = torch.tensor(label_token_ids, device=log_probs.device, dtype=torch.long).unsqueeze(0).expand(log_probs.shape[0], -1)
    scores = torch.gather(log_probs, -1, index_tensor)
    return [[float(score) for score in row] for row in scores.detach().cpu().tolist()]


def _get_multiple_choice_batch_size(config: dict[str, Any] | None = None) -> int:
    config = config or {}
    scenario = str(os.environ.get("BUSINESS_BENCHMARK_SCENARIO") or "").strip()
    field_by_scenario = {
        "npu_baseline": "multiple_choice_baseline_batch_size",
        "npu_perf": "multiple_choice_perf_batch_size",
        "cuda_baseline": "multiple_choice_cuda_baseline_batch_size",
    }
    field_name = field_by_scenario.get(scenario, "")
    raw_value = config.get(field_name, config.get("multiple_choice_batch_size", 1))
    try:
        return max(int(raw_value), 1)
    except Exception:
        return 1


def _score_multiple_choice_candidates(model, tokenizer, prompt_text: str, labels: list[str]) -> list[float]:
    if not labels:
        return []
    if len(labels) == 1:
        return [_score_multiple_choice_candidate(model, tokenizer, prompt_text, labels[0])]

    fast_scores = _score_multiple_choice_single_token_candidates(model, tokenizer, prompt_text, labels)
    if fast_scores is not None and len(fast_scores) == len(labels):
        return fast_scores

    model_device = next(model.parameters()).device
    try:
        prompt_inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            add_special_tokens=False,
        )
        candidate_texts = [prompt_text + f" {label}" for label in labels]
        full_inputs = tokenizer(
            candidate_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=1040,
            add_special_tokens=False,
        )
        if not isinstance(prompt_inputs, Mapping) or not isinstance(full_inputs, Mapping):
            raise TypeError("tokenizer batch output must be dict")

        prompt_len = int(prompt_inputs["input_ids"].shape[-1])
        input_ids = full_inputs["input_ids"].to(model_device)
        attention_mask = full_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(model_device)

        if input_ids.shape[-1] <= prompt_len or prompt_len <= 0:
            return [float("-inf")] * len(labels)

        with torch.no_grad():
            try:
                outputs = _forward_causal_lm_no_cache(model, input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            except Exception as exc:
                if not hasattr(model, "model") or not hasattr(model, "lm_head"):
                    raise
                if not any(token in str(exc) for token in ("get_usable_length", "DynamicCache", "from_legacy_cache")):
                    raise
                model_outputs = _forward_causal_lm_no_cache(model.model, input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = model_outputs.last_hidden_state if hasattr(model_outputs, "last_hidden_state") else model_outputs[0]
                logits = model.lm_head(hidden_states)

        target_ids = input_ids[:, prompt_len:]
        candidate_mask = attention_mask[:, prompt_len:]
        candidate_logits = logits[:, prompt_len - 1 : prompt_len - 1 + target_ids.shape[1], :]
        if candidate_logits.shape[1] <= 0 or target_ids.shape[1] <= 0:
            return [float("-inf")] * len(labels)
        if candidate_logits.shape[1] != target_ids.shape[1]:
            common_len = min(candidate_logits.shape[1], target_ids.shape[1])
            candidate_logits = candidate_logits[:, :common_len, :]
            target_ids = target_ids[:, :common_len]
            candidate_mask = candidate_mask[:, :common_len]

        log_probs = torch.log_softmax(candidate_logits, dim=-1)
        gathered = torch.gather(log_probs, -1, target_ids.unsqueeze(-1)).squeeze(-1)
        scores = (gathered * candidate_mask.to(dtype=gathered.dtype)).sum(dim=-1)
        return [float(score) for score in scores.detach().cpu().tolist()]
    except Exception:
        return [_score_multiple_choice_candidate(model, tokenizer, prompt_text, label) for label in labels]


def _should_apply_multiple_choice_label_prior_correction(tokenizer, config: dict[str, Any] | None = None) -> bool:
    model_id_text = str((config or {}).get("model_id") or "").strip().lower()
    if any(token in model_id_text for token in ("instruct", "instruction", "chat", "assistant", "dialog", "dialogue", "alpaca", "sft")):
        return False
    if re.search(r"(^|[-_/])it($|[-_/])", model_id_text):
        return False

    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        chat_template_text = str(chat_template).strip().lower()
        if any(token in chat_template_text for token in ("assistant", "user", "system", "[inst]", "<|im_start|>", "<|start_header_id|>")):
            return False
    return True


def _run_multiple_choice_scoring(model, tokenizer, samples: list[dict], *, config: dict[str, Any] | None = None) -> tuple[list[str], dict]:
    predictions: list[str] = []
    inference_strategy = "multiple_choice_scoring"
    used_label_prior_correction = False
    allow_label_prior_correction = _should_apply_multiple_choice_label_prior_correction(tokenizer, config=config)
    requested_batch_size = _get_multiple_choice_batch_size(config)
    effective_batch_size = 1
    sample_batches = [samples[idx : idx + requested_batch_size] for idx in range(0, len(samples), requested_batch_size)]
    for sample_batch in sample_batches:
        if not sample_batch:
            continue
        batched_prompts: list[str] = []
        batched_labels: list[str] | None = None
        batched_choices: list[list[tuple[str, str]]] = []
        batched_dataset_keys: list[str] = []
        batched_prior_scores: list[float] | None = None
        can_batch = requested_batch_size > 1
        for sample in sample_batch:
            dataset_key = _sample_dataset_key(sample)
            choices = sample.get("choices") or []
            labeled_choices = [((chr(65 + idx) if 0 <= idx < 26 else str(idx)), str(choice).strip()) for idx, choice in enumerate(choices) if str(choice).strip()]
            if not labeled_choices:
                can_batch = False
                break
            labels = [item[0] for item in labeled_choices]
            if batched_labels is None:
                batched_labels = labels
            elif labels != batched_labels:
                can_batch = False
                break
            if not all(len(label) == 1 and label.isalpha() for label in labels):
                can_batch = False
                break
            prompt = _build_generation_prompt(sample)
            batched_prompts.append(_render_chat_prompt(tokenizer, prompt, disable_thinking=True))
            batched_choices.append(labeled_choices)
            batched_dataset_keys.append(dataset_key)
        if can_batch and batched_labels and batched_prompts:
            scores_batch = _score_multiple_choice_single_token_candidates_batched(model, tokenizer, batched_prompts, batched_labels)
            if scores_batch is not None and len(scores_batch) == len(sample_batch):
                if allow_label_prior_correction:
                    prior_prompt = _render_chat_prompt(tokenizer, "Answer:", disable_thinking=True)
                    prior_scores = _score_multiple_choice_single_token_candidates(model, tokenizer, prior_prompt, batched_labels)
                    if prior_scores is not None and len(prior_scores) == len(batched_labels) and any(math.isfinite(score) for score in prior_scores):
                        batched_prior_scores = prior_scores
                        used_label_prior_correction = True
                        inference_strategy = "multiple_choice_scoring_with_label_prior_correction"
                effective_batch_size = max(effective_batch_size, len(sample_batch))
                for sample_idx, labeled_choices in enumerate(batched_choices):
                    scores = list(scores_batch[sample_idx])
                    if batched_prior_scores is not None:
                        scores = [
                            (score - prior_score) if math.isfinite(score) and math.isfinite(prior_score) else score
                            for score, prior_score in zip(scores, batched_prior_scores)
                        ]
                    best_index = max(range(len(labeled_choices)), key=lambda idx: scores[idx])
                    best_label, best_choice = labeled_choices[best_index]
                    dataset_key = batched_dataset_keys[sample_idx]
                    if dataset_key == "mmlu":
                        predictions.append(best_label)
                    elif dataset_key == "pubmed_qa":
                        predictions.append(_extract_pubmed_qa_decision(best_choice))
                    else:
                        predictions.append(best_choice)
                continue

        for sample in sample_batch:
            dataset_key = _sample_dataset_key(sample)
            choices = sample.get("choices") or []
            labeled_choices = [((chr(65 + idx) if 0 <= idx < 26 else str(idx)), str(choice).strip()) for idx, choice in enumerate(choices) if str(choice).strip()]
            if not labeled_choices:
                predictions.append("")
            continue

        # Raw prompt scoring is more reliable than free-form generation for
        # multiple-choice causal LMs that may emit reasoning text before the answer.
        prompt = _build_generation_prompt(sample)
        prompt = _render_chat_prompt(tokenizer, prompt, disable_thinking=True)
        labels = [item[0] for item in labeled_choices]
        scores = _score_multiple_choice_candidates(model, tokenizer, prompt, labels)
        if allow_label_prior_correction and labels and all(len(label) == 1 and label.isalpha() for label in labels):
            # Base conversational / non-instruct LMs can be strongly biased toward
            # a default answer token such as "A". Debias by subtracting the label
            # prior under a generic answer prefix while keeping the same tokenizer
            # chat formatting as the task prompt.
            prior_prompt = _render_chat_prompt(tokenizer, "Answer:", disable_thinking=True)
            prior_scores = _score_multiple_choice_candidates(model, tokenizer, prior_prompt, labels)
            if len(prior_scores) == len(scores) and any(math.isfinite(score) for score in prior_scores):
                scores = [
                    (score - prior_score) if math.isfinite(score) and math.isfinite(prior_score) else score
                    for score, prior_score in zip(scores, prior_scores)
                ]
                used_label_prior_correction = True
                inference_strategy = "multiple_choice_scoring_with_label_prior_correction"
        if len(scores) != len(labeled_choices):
            raise RuntimeError("multiple-choice scoring returned mismatched candidate scores")
        best_index = max(range(len(labeled_choices)), key=lambda idx: scores[idx])
        best_label, best_choice = labeled_choices[best_index]
        if dataset_key == "mmlu":
            predictions.append(best_label)
        elif dataset_key == "pubmed_qa":
            predictions.append(_extract_pubmed_qa_decision(best_choice))
        else:
            predictions.append(best_choice)
    return predictions, {
        "inference_strategy": inference_strategy,
        "label_prior_correction": used_label_prior_correction,
        "multiple_choice_batch_size_requested": requested_batch_size,
        "multiple_choice_batch_size_effective": effective_batch_size,
    }


def _load_internvl_conversation_module(model) -> ModuleType | None:
    candidate_paths: list[Path] = []
    model_name_or_path = str(getattr(model, "name_or_path", "") or "").strip()
    if model_name_or_path:
        model_path = Path(model_name_or_path)
        if model_path.exists():
            if model_path.is_file():
                candidate_paths.append(model_path.parent / "conversation.py")
            else:
                candidate_paths.append(model_path / "conversation.py")

    try:
        candidate_paths.extend(sorted(CACHE_DIR.rglob("conversation.py")))
    except Exception:
        pass

    seen: set[Path] = set()
    for path in candidate_paths:
        try:
            resolved_path = path.resolve()
        except Exception:
            resolved_path = path
        if resolved_path in seen or not resolved_path.is_file():
            continue
        seen.add(resolved_path)
        try:
            module_name = f"_slai_internvl_conversation_{abs(hash(str(resolved_path)))}"
            return _load_module_from_path(module_name, resolved_path)
        except Exception:
            continue
    return None


def _build_internvl_conversation_prompt(model, prompt_text: str, *, pixel_values: torch.Tensor | None, tokenizer) -> str:
    question = str(prompt_text or "").strip()
    if pixel_values is not None and "<image>" not in question:
        question = "<image>\n" + question

    conversation_module = _load_internvl_conversation_module(model)
    template_name = str(getattr(model, "template", "") or "internlm2-chat").strip() or "internlm2-chat"
    system_message = str(getattr(model, "system_message", "") or "").strip()

    query = ""
    if conversation_module is not None and hasattr(conversation_module, "get_conv_template"):
        try:
            template = conversation_module.get_conv_template(template_name)
            template.system_message = system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = str(template.get_prompt() or "")
        except Exception:
            query = ""

    if not query:
        sep = "<|im_end|>"
        query = f"<|im_start|>system\n{system_message}{sep}<|im_start|>user\n{question}{sep}<|im_start|>assistant\n"

    if pixel_values is not None:
        img_context_token = "<IMG_CONTEXT>"
        img_start_token = "<img>"
        img_end_token = "</img>"
        num_image_token = int(getattr(model, "num_image_token", 0) or 0)
        num_patches = int(pixel_values.shape[0])
        if num_image_token <= 0 or num_patches <= 0:
            raise RuntimeError("InternVL image prompt requires positive num_image_token and num_patches")
        image_tokens = img_start_token + img_context_token * num_image_token * num_patches + img_end_token
        query = query.replace("<image>", image_tokens, 1)
        try:
            model.img_context_token_id = tokenizer.convert_tokens_to_ids(img_context_token)
        except Exception:
            pass

    return query


def _score_internvl_multiple_choice_candidates(model, tokenizer, prompt_text: str, labels: list[str], *, pixel_values: torch.Tensor | None = None) -> list[float]:
    if not labels:
        return []

    if pixel_values is None:
        language_model = getattr(model, "language_model", None)
        if language_model is None:
            return _score_multiple_choice_candidates(model, tokenizer, prompt_text, labels)
        return _score_multiple_choice_candidates(language_model, tokenizer, prompt_text, labels)

    model_device = _get_model_device(model)
    model_dtype = _get_model_floating_dtype(model)
    prompt_inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        add_special_tokens=False,
    )
    candidate_texts = [prompt_text + f" {label}" for label in labels]
    full_inputs = tokenizer(
        candidate_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=2080,
        add_special_tokens=False,
    )
    if not isinstance(prompt_inputs, Mapping) or not isinstance(full_inputs, Mapping):
        raise TypeError("InternVL tokenizer output must be a mapping")

    prompt_len = int(prompt_inputs["input_ids"].shape[-1])
    input_ids = full_inputs["input_ids"].to(model_device)
    attention_mask = full_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(model_device)

    if input_ids.shape[-1] <= prompt_len or prompt_len <= 0:
        return [float("-inf")] * len(labels)

    runtime_pixel_values = _move_runtime_tensor(pixel_values, model_device, dtype=model_dtype)
    repeat_shape = [len(labels)] + [1] * (runtime_pixel_values.ndim - 1)
    batch_pixel_values = runtime_pixel_values.repeat(*repeat_shape)
    image_flags = torch.ones((len(labels), 1), dtype=torch.long, device=model_device)

    with torch.no_grad():
        outputs = model(
            pixel_values=batch_pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

    target_ids = input_ids[:, prompt_len:]
    candidate_mask = attention_mask[:, prompt_len:]
    candidate_logits = logits[:, prompt_len - 1 : prompt_len - 1 + target_ids.shape[1], :]
    if candidate_logits.shape[1] <= 0 or target_ids.shape[1] <= 0:
        return [float("-inf")] * len(labels)
    if candidate_logits.shape[1] != target_ids.shape[1]:
        common_len = min(candidate_logits.shape[1], target_ids.shape[1])
        candidate_logits = candidate_logits[:, :common_len, :]
        target_ids = target_ids[:, :common_len]
        candidate_mask = candidate_mask[:, :common_len]

    log_probs = torch.log_softmax(candidate_logits, dim=-1)
    gathered = torch.gather(log_probs, -1, target_ids.unsqueeze(-1)).squeeze(-1)
    scores = (gathered * candidate_mask.to(dtype=gathered.dtype)).sum(dim=-1)
    return [float(score) for score in scores.detach().cpu().tolist()]


def _run_internvl_multiple_choice_scoring(model, processor, samples: list[dict], *, config: dict[str, Any] | None = None) -> tuple[list[str], dict]:
    predictions: list[str] = []
    tokenizer = getattr(processor, "tokenizer", processor)
    image_processor = getattr(processor, "image_processor", None)
    model_device = _get_model_device(model)
    latency_scope = str(getattr(config, "latency_measurement_scope", None) or (config or {}).get("latency_measurement_scope") or os.environ.get("BUSINESS_BENCHMARK_LATENCY_MEASUREMENT_SCOPE") or "").strip().lower()
    prepared_requests: list[dict[str, Any]] = []

    for sample in samples:
        dataset_key = _sample_dataset_key(sample)
        choices = sample.get("choices") or []
        labeled_choices = [((chr(65 + idx) if 0 <= idx < 26 else str(idx)), str(choice).strip()) for idx, choice in enumerate(choices) if str(choice).strip()]
        if not labeled_choices:
            prepared_requests.append({"skip": True, "prediction": ""})
            continue

        prompt = _build_vlm_prompt(sample)
        image_value = _decode_image_sample(sample.get("image"))
        pixel_values = None
        if image_value is not None:
            if image_processor is None:
                raise RuntimeError("InternVL processor missing image_processor for multiple-choice scoring")
            processed_image = image_processor(images=image_value, return_tensors="pt")
            if not isinstance(processed_image, Mapping):
                raise RuntimeError("InternVL image processor output must be a mapping")
            pixel_values = processed_image.get("pixel_values")
            if pixel_values is None:
                raise RuntimeError("InternVL image processor output missing pixel_values")

        internvl_prompt = _build_internvl_conversation_prompt(model, prompt, pixel_values=pixel_values, tokenizer=tokenizer)
        prepared_requests.append(
            {
                "skip": False,
                "dataset_key": dataset_key,
                "internvl_prompt": internvl_prompt,
                "labeled_choices": labeled_choices,
                "pixel_values": pixel_values,
            }
        )

    scored_request_count = sum(1 for request in prepared_requests if not bool(request.get("skip")))
    inference_start_ts = None
    if latency_scope == "steady_state" and scored_request_count > 0:
        import time

        _sync_device(model_device)
        inference_start_ts = time.perf_counter()

    for request in prepared_requests:
        if bool(request.get("skip")):
            predictions.append(str(request.get("prediction") or ""))
            continue

        dataset_key = str(request.get("dataset_key") or "").strip()
        labeled_choices = list(request.get("labeled_choices") or [])
        internvl_prompt = str(request.get("internvl_prompt") or "")
        pixel_values = request.get("pixel_values")
        labels = [item[0] for item in labeled_choices]
        scores = _score_internvl_multiple_choice_candidates(model, tokenizer, internvl_prompt, labels, pixel_values=pixel_values)
        if len(scores) != len(labeled_choices):
            raise RuntimeError("InternVL multiple-choice scoring returned mismatched candidate scores")

        best_index = max(range(len(labeled_choices)), key=lambda idx: scores[idx])
        best_label, best_choice = labeled_choices[best_index]
        if dataset_key == "mmlu":
            predictions.append(best_label)
        elif dataset_key == "pubmed_qa":
            predictions.append(_extract_pubmed_qa_decision(best_choice))
        else:
            predictions.append(best_choice)

    metric_context: dict[str, Any] = {
        "inference_strategy": "internvl_multiple_choice_scoring",
        "uses_image_conditioning": any(_decode_image_sample(sample.get("image")) is not None for sample in samples),
    }
    if inference_start_ts is not None and scored_request_count > 0:
        import time

        _sync_device(model_device)
        measured_wall_clock_s = time.perf_counter() - inference_start_ts
        metric_context["inference_latency_s"] = measured_wall_clock_s / scored_request_count
        metric_context["wall_clock_s"] = measured_wall_clock_s
    return predictions, metric_context


def _run_generation(model, tokenizer, samples: list[dict], *, model_type: str, config: dict[str, Any] | None = None) -> list[str]:
    predictions: list[str] = []
    disable_generation_cache = _should_disable_generation_cache(model, config=config)
    for sample in samples:
        dataset_key = _sample_dataset_key(sample)
        prompt = _build_generation_prompt(sample)
        prompt = _render_chat_prompt(tokenizer, prompt, disable_thinking=dataset_key in {"pubmed_qa", "gsm8k"} or _sample_has_choices(sample))
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        generation_kwargs = _generation_kwargs_for_sample(sample, tokenizer, config=config)
        if disable_generation_cache:
            generation_kwargs["use_cache"] = False
            for generation_target in (model, getattr(model, "model", None), getattr(model, "language_model", None)):
                if generation_target is None:
                    continue
                generation_config = getattr(generation_target, "generation_config", None)
                if generation_config is not None and hasattr(generation_config, "use_cache"):
                    try:
                        generation_config.use_cache = False
                    except Exception:
                        pass
                target_config = getattr(generation_target, "config", None)
                if target_config is not None and hasattr(target_config, "use_cache"):
                    try:
                        target_config.use_cache = False
                    except Exception:
                        pass
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        if model_type == "causal_lm":
            prediction = _decode_generated(outputs[0], tokenizer, input_len=inputs["input_ids"].shape[-1])
        else:
            prediction = _decode_generated(outputs[0], tokenizer)
        if dataset_key == "gsm8k":
            prediction = _extract_gsm8k_prediction_answer(prediction)
        elif dataset_key == "pubmed_qa":
            prediction = _extract_pubmed_qa_decision(prediction)
        predictions.append(prediction)
    return predictions


def _run_quiet_star_generation(model, tokenizer, samples: list[dict], *, model_type: str, config: dict[str, Any] | None = None) -> list[str]:
    predictions: list[str] = []
    model_device = _get_model_device(model)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)

    for sample in samples:
        dataset_key = _sample_dataset_key(sample)
        prompt = _build_generation_prompt(sample)
        prompt = _render_chat_prompt(tokenizer, prompt, disable_thinking=dataset_key in {"pubmed_qa", "gsm8k"} or _sample_has_choices(sample))
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encoded["input_ids"].to(model_device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)
        else:
            attention_mask = torch.ones_like(input_ids, device=model_device)

        generation_kwargs = _generation_kwargs_for_sample(sample, tokenizer, config=config)
        max_new_tokens = int(generation_kwargs.get("max_new_tokens") or 16)
        generated_ids = input_ids.clone()
        generated_attention_mask = attention_mask.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(
                    input_ids=generated_ids,
                    attention_mask=generated_attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                next_mask = torch.ones((generated_attention_mask.shape[0], 1), dtype=generated_attention_mask.dtype, device=model_device)
                generated_attention_mask = torch.cat([generated_attention_mask, next_mask], dim=-1)
                next_token_value = int(next_token_id[0, 0].item())
                if eos_token_id is not None and next_token_value == int(eos_token_id):
                    break
                if pad_token_id is not None and next_token_value == int(pad_token_id):
                    break

        if model_type == "causal_lm":
            prediction = _decode_generated(generated_ids[0], tokenizer, input_len=input_ids.shape[-1])
        else:
            prediction = _decode_generated(generated_ids[0], tokenizer)
        if dataset_key == "gsm8k":
            prediction = _extract_gsm8k_prediction_answer(prediction)
        elif dataset_key == "pubmed_qa":
            prediction = _extract_pubmed_qa_decision(prediction)
        predictions.append(prediction)
    return predictions


def _extract_qwen3_guard_safety_label(raw_text: Any) -> str:
    text = str(raw_text or "").strip()
    if not text:
        return ""

    match = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()

    lowered = text.lower()
    for candidate in ("unsafe", "controversial", "safe"):
        if re.search(rf"\b{candidate}\b", lowered):
            return candidate
    return ""


def _map_qwen3_guard_label_to_dataset(sample: dict[str, Any], safety_label: str) -> str:
    label_names = [re.sub(r"\s+", " ", str(item or "").replace("_", " ").replace("-", " ").strip().lower()) for item in list(sample.get("dataset_label_names") or []) if str(item or "").strip()]
    negative_id = "0"
    positive_id = "1"
    for idx, label_name in enumerate(label_names):
        if any(token in label_name for token in ("not offensive", "not-offensive", "non offensive", "non-offensive", "not hateful", "not-hateful", "safe", "neutral", "benign")):
            negative_id = str(idx)
        elif any(token in label_name for token in ("offensive", "hate", "hateful", "toxic", "unsafe", "abusive")):
            positive_id = str(idx)
    if safety_label == "safe":
        return negative_id
    if safety_label in {"unsafe", "controversial"}:
        return positive_id
    return positive_id


def _run_qwen3_guard_generation(model, tokenizer, samples: list[dict], config: dict[str, Any] | None = None) -> tuple[list[str], dict]:
    predictions: list[str] = []
    raw_predictions: list[str] = []
    parsed_labels: list[str] = []
    parse_failures = 0
    model_device = _get_model_device(model)

    for sample in samples:
        prompt = str(sample.get("input") or "").strip()
        if not prompt:
            predictions.append("")
            raw_predictions.append("")
            parsed_labels.append("")
            parse_failures += 1
            continue

        rendered_prompt = _render_chat_prompt(tokenizer, prompt, disable_thinking=True)
        inputs = tokenizer(
            rendered_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            add_special_tokens=False,
        )
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        generation_kwargs = _generation_kwargs_for_sample(sample, tokenizer, config=config)
        generation_kwargs["max_new_tokens"] = min(int(generation_kwargs.get("max_new_tokens", 32) or 32), 32)

        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)

        raw_prediction = _decode_generated(outputs[0], tokenizer, input_len=inputs["input_ids"].shape[-1])
        safety_label = _extract_qwen3_guard_safety_label(raw_prediction)
        if safety_label not in QWEN3_GUARD_SAFETY_LABELS:
            parse_failures += 1

        has_classification_label_space = bool(sample.get("dataset_label_names")) or str(sample.get("reference") or "").strip() in {"0", "1"}
        dataset_key = _sample_dataset_key(sample)
        if has_classification_label_space or dataset_key in {"tweet_eval_offensive", "tweet_eval_hate"}:
            prediction = _map_qwen3_guard_label_to_dataset(sample, safety_label)
        else:
            prediction = safety_label

        predictions.append(prediction)
        raw_predictions.append(str(raw_prediction or "").strip())
        parsed_labels.append(safety_label)

    return predictions, {
        "inference_strategy": "qwen3_guard_generation",
        "parse_failures": parse_failures,
        "parsed_safety_labels_preview": parsed_labels[:8],
        "raw_predictions_preview": raw_predictions[:4],
    }


def _run_vlm(model, processor, samples: list[dict], config: dict[str, Any] | None = None) -> tuple[list[str], dict]:
    predictions: list[str] = []
    model_device = _get_model_device(model)
    model_dtype = _get_model_floating_dtype(model)
    model_family = str(getattr(processor, "_business_vlm_family", "") or "").strip().lower()

    if _looks_like_openvla_request(config, str(getattr(model, "name_or_path", "") or ""), model_config=getattr(model, "config", None)) or (
        isinstance(config, dict) and _looks_like_openvla_request(config, str(config.get("model_id") or ""), model_config=getattr(model, "config", None))
    ):
        return _run_openvla_action_prediction(model, processor, samples, config=config)

    if model_family == "deepseek_vl_v2":
        return _run_deepseek_vl_v2(model, processor, samples, config=config)

    if model_family == "internvl_chat" and samples and all(_sample_has_choices(sample) for sample in samples):
        return _run_internvl_multiple_choice_scoring(model, processor, samples, config=config)

    for sample in samples:
        prompt_text = _build_vlm_prompt(sample)
        image_value = _decode_image_sample(sample.get("image"))
        has_image = image_value is not None

        if model_family == "moondream1" and hasattr(model, "query"):
            generation_kwargs = _generation_kwargs_for_sample(sample, getattr(processor, "tokenizer", processor), config=config)
            query_settings = {
                "max_tokens": int(generation_kwargs.get("max_new_tokens") or 256),
                "temperature": 0.0,
                "top_p": 1.0,
            }
            with torch.no_grad():
                query_output = model.query(image=image_value if has_image else None, question=prompt_text, stream=False, settings=query_settings)
            if isinstance(query_output, Mapping):
                decoded_text = str(query_output.get("answer") or "").strip()
            else:
                decoded_text = str(query_output or "").strip()
            if _sample_dataset_key(sample) == "scienceqa" or _sample_has_choices(sample):
                predictions.append(_normalize_scienceqa_prediction(decoded_text, sample))
            else:
                predictions.append(_extract_assistant_text(decoded_text) or decoded_text)
            continue

        if model_family == "internvl_chat" and hasattr(model, "chat"):
            tokenizer = getattr(processor, "tokenizer", processor)
            pixel_values = None
            if has_image:
                image_processor = getattr(processor, "image_processor", None)
                if image_processor is None:
                    raise RuntimeError("InternVL processor 缺少 image_processor，无法生成 pixel_values")
                processed_image = image_processor(images=image_value, return_tensors="pt")
                if not isinstance(processed_image, Mapping):
                    raise RuntimeError("InternVL processor 输出必须是包含 pixel_values 的 mapping")
                pixel_values = processed_image.get("pixel_values")
                if pixel_values is None:
                    raise RuntimeError("InternVL processor 输出缺少 pixel_values")
                pixel_values = _move_runtime_tensor(pixel_values, model_device, dtype=model_dtype)
            generation_kwargs = _generation_kwargs_for_sample(sample, tokenizer, config=config)
            generation_kwargs.pop("use_cache", None)
            for generation_target in (model, getattr(model, "language_model", None)):
                if generation_target is None:
                    continue
                generation_config = getattr(generation_target, "generation_config", None)
                if generation_config is not None:
                    try:
                        generation_config.use_cache = False
                    except Exception:
                        pass
                target_config = getattr(generation_target, "config", None)
                if target_config is not None and hasattr(target_config, "use_cache"):
                    try:
                        target_config.use_cache = False
                    except Exception:
                        pass
            with torch.no_grad():
                decoded = model.chat(
                    tokenizer,
                    pixel_values,
                    prompt_text,
                    generation_kwargs,
                    num_patches_list=[int(pixel_values.shape[0])] if pixel_values is not None else None,
                )
            decoded_text = str(decoded or "").strip()
            if _sample_dataset_key(sample) == "scienceqa" or _sample_has_choices(sample):
                predictions.append(_normalize_scienceqa_prediction(decoded_text, sample))
            else:
                predictions.append(_extract_assistant_text(decoded_text) or decoded_text)
            continue

        rendered_prompt = _render_vlm_prompt(processor, prompt_text, has_image=has_image, model_family=model_family)

        processor_kwargs: dict[str, Any] = {"text": rendered_prompt, "return_tensors": "pt"}
        if has_image:
            processor_kwargs["images"] = image_value
        processed = processor(**processor_kwargs)
        processed = {key: _move_runtime_tensor(value, model_device, dtype=model_dtype) for key, value in processed.items()}

        generation_kwargs = _generation_kwargs_for_sample(sample, getattr(processor, "tokenizer", processor), config=config)
        if model_family == "blip":
            generation_kwargs.pop("pad_token_id", None)
            generation_kwargs.pop("eos_token_id", None)
        if model_family == "qwen2_5_omni":
            generation_kwargs.setdefault("generation_mode", "text")
            generation_kwargs.setdefault("use_audio_in_video", False)

        with torch.no_grad():
            outputs = model.generate(**processed, **generation_kwargs)

        input_ids = processed.get("input_ids")
        generated_ids = outputs[0] if isinstance(outputs, (tuple, list)) and outputs else outputs
        if input_ids is not None and hasattr(input_ids, "shape") and hasattr(generated_ids, "shape") and len(input_ids.shape) == 2 and len(generated_ids.shape) == 2:
            prompt_length = int(input_ids.shape[1])
            if generated_ids.shape[1] > prompt_length:
                generated_ids = generated_ids[:, prompt_length:]

        try:
            decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception:
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is None or not hasattr(tokenizer, "batch_decode"):
                raise RuntimeError("VLM processor/tokenizer 不支持 batch_decode")
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if _sample_dataset_key(sample) == "scienceqa" or _sample_has_choices(sample):
            predictions.append(_normalize_scienceqa_prediction(decoded, sample))
        else:
            predictions.append(_extract_assistant_text(decoded) or str(decoded or "").strip())

    return predictions, {"inference_strategy": "vlm_generate"}


def _run_openvla_action_prediction(model, processor, samples: list[dict], config: dict[str, Any] | None = None) -> tuple[list[str], dict]:
    predictions: list[str] = []
    model_device = _get_model_device(model)
    model_dtype = _get_model_floating_dtype(model)
    latency_scope = str(getattr(config, "latency_measurement_scope", None) or (config or {}).get("latency_measurement_scope") or os.environ.get("BUSINESS_BENCHMARK_LATENCY_MEASUREMENT_SCOPE") or "").strip().lower()
    unnorm_key = str((config or {}).get("openvla_unnorm_key") or "bridge_orig").strip() or "bridge_orig"

    prepared_inputs: list[dict[str, Any]] = []
    for sample in samples:
        prompt_text = _build_vlm_prompt(sample) or "In: What action should the robot take to pick up the object?\nOut:"
        image_value = _decode_image_sample(sample.get("image") or sample.get("input"))
        if image_value is None:
            try:
                from PIL import Image

                image_value = Image.new("RGB", (224, 224), color=(128, 128, 128))
            except Exception as exc:
                raise RuntimeError(f"OpenVLA 样本缺少 image 且无法构造默认图像: {exc}") from exc
        processed = processor(images=image_value, text=prompt_text, return_tensors="pt")
        processed = {key: _move_runtime_tensor(value, model_device, dtype=model_dtype) for key, value in processed.items()}
        prepared_inputs.append(processed)

    inference_start_ts = None
    if latency_scope == "steady_state" and prepared_inputs:
        import time

        _sync_device(model_device)
        inference_start_ts = time.perf_counter()

    for processed in prepared_inputs:
        with torch.no_grad():
            if hasattr(model, "predict_action"):
                action = model.predict_action(**processed, unnorm_key=unnorm_key, do_sample=False)
                if hasattr(action, "detach"):
                    action_value = action.detach().cpu().reshape(-1).tolist()
                elif hasattr(action, "cpu") and hasattr(action, "numpy"):
                    action_value = action.cpu().numpy().reshape(-1).tolist()
                elif hasattr(action, "tolist"):
                    action_value = list(action.tolist())
                else:
                    action_value = np.asarray(action).reshape(-1).tolist()
            else:
                outputs = model.generate(**processed, max_new_tokens=20, do_sample=False)
                action_value = [float(int(outputs.shape[-1]))] if hasattr(outputs, "shape") else [0.0]
        predictions.append(json.dumps([float(value) for value in action_value], ensure_ascii=False))

    metric_context: dict[str, Any] = {
        "inference_strategy": "openvla_predict_action",
        "references": [sample.get("reference") for sample in samples[: len(predictions)]],
        "openvla_unnorm_key": unnorm_key,
    }
    if inference_start_ts is not None and predictions:
        import time

        _sync_device(model_device)
        metric_context["inference_latency_s"] = (time.perf_counter() - inference_start_ts) / len(predictions)
    return predictions, metric_context


def _extract_gsm8k_prediction_answer(raw_text: Any) -> str:
    text = str(raw_text or "").strip()
    if not text:
        return ""

    normalized_text = text.replace("▁", " ")
    normalized_text = re.sub(r"(?<=[-+])\s+(?=\d)", "", normalized_text)
    normalized_text = re.sub(r"(?<=[\d,.])\s+(?=[\d,.])", "", normalized_text)
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

    tag_match = re.search(r"<answer>\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*</answer>", normalized_text, flags=re.IGNORECASE)
    if tag_match:
        return tag_match.group(1).replace(",", "").strip()

    final_marker_matches = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", normalized_text)
    if final_marker_matches:
        return final_marker_matches[-1].replace(",", "").strip()

    numeric_matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", normalized_text)
    if numeric_matches:
        return numeric_matches[-1].replace(",", "").strip()
    return text


def _normalize_masked_lm_token_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("##"):
        text = text[2:]
    text = text.replace("Ġ", "").replace("▁", "").strip()
    return text.lower()


def _token_id_to_masked_lm_text(tokenizer, token_id: int) -> str:
    try:
        token_text = tokenizer.convert_ids_to_tokens(int(token_id))
    except Exception:
        token_text = ""
    normalized = _normalize_masked_lm_token_text(token_text)
    if normalized:
        return normalized
    try:
        decoded = tokenizer.decode([int(token_id)], skip_special_tokens=True)
    except Exception:
        decoded = ""
    normalized = _normalize_masked_lm_token_text(decoded)
    return normalized or str(int(token_id))


def _choose_masked_lm_token_index(tokenizer, token_ids: list[int], *, preferred_tokens: set[str] | None = None) -> int | None:
    if not token_ids:
        return None
    special_ids = {int(token_id) for token_id in list(getattr(tokenizer, "all_special_ids", []) or [])}
    try:
        token_texts = list(tokenizer.convert_ids_to_tokens(token_ids))
    except Exception:
        token_texts = [str(token_id) for token_id in token_ids]
    normalized_rows = [_normalize_masked_lm_token_text(token_text) for token_text in token_texts]
    token_counts = Counter(token_text for token_text in normalized_rows if token_text)
    midpoint = len(token_ids) / 2.0
    primary_candidates: list[tuple[tuple[float, ...], int]] = []
    fallback_candidates: list[tuple[tuple[float, ...], int]] = []

    for idx, token_id in enumerate(token_ids):
        if int(token_id) in special_ids:
            continue
        raw_token = str(token_texts[idx] or "")
        normalized = normalized_rows[idx]
        if not normalized:
            continue
        if raw_token.startswith("##"):
            continue
        if idx + 1 < len(token_texts) and str(token_texts[idx + 1] or "").startswith("##"):
            continue
        if not any(char.isalpha() for char in normalized):
            continue
        score = (
            1.0 if preferred_tokens and normalized in preferred_tokens else 0.0,
            1.0 if token_counts.get(normalized, 0) > 1 else 0.0,
            1.0 if normalized.isalpha() else 0.0,
            1.0 if 3 <= len(normalized) <= 8 else 0.0,
            -abs(len(normalized) - 6),
            -abs(idx - midpoint),
        )
        if len(normalized) >= 3:
            primary_candidates.append((score, idx))
        else:
            fallback_candidates.append((score, idx))

    if primary_candidates:
        return max(primary_candidates, key=lambda item: item[0])[1]
    if fallback_candidates:
        return max(fallback_candidates, key=lambda item: item[0])[1]
    return None


def _prepare_masked_lm_sample(tokenizer, sample: dict, *, max_length: int) -> tuple[dict[str, torch.Tensor], int, str] | None:
    question = str(sample.get("question") or "").strip()
    context = str(sample.get("context") or "").strip()
    if question and context:
        question_ids = list(
            tokenizer(
                question,
                add_special_tokens=False,
                truncation=True,
                max_length=max(max_length // 3, 16),
            ).get("input_ids")
            or []
        )
        context_ids = list(
            tokenizer(
                context,
                add_special_tokens=False,
                truncation=True,
                max_length=max(max_length - len(question_ids) - 3, 16),
            ).get("input_ids")
            or []
        )
        context_terms = {_token_id_to_masked_lm_text(tokenizer, token_id) for token_id in context_ids}
        preferred_tokens = {_token_id_to_masked_lm_text(tokenizer, token_id) for token_id in question_ids if _token_id_to_masked_lm_text(tokenizer, token_id) in context_terms}
        question_mask_index = _choose_masked_lm_token_index(tokenizer, question_ids, preferred_tokens=preferred_tokens)
        encoded = tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation="longest_first",
            max_length=max_length,
        )
        if question_mask_index is not None:
            mask_position = 1 + int(question_mask_index)
            if mask_position < int(encoded["input_ids"].shape[1]) - 1:
                reference_text = _token_id_to_masked_lm_text(tokenizer, question_ids[question_mask_index])
                if reference_text:
                    return encoded, mask_position, reference_text

    text = str(sample.get("input") or "").strip()
    if not text:
        return None
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    token_ids = encoded["input_ids"][0].tolist()
    mask_position = _choose_masked_lm_token_index(tokenizer, token_ids)
    if mask_position is None:
        return None
    reference_text = _token_id_to_masked_lm_text(tokenizer, token_ids[mask_position])
    if not reference_text:
        return None
    return encoded, int(mask_position), reference_text


def _run_masked_lm(model, tokenizer, samples: list[dict]) -> tuple[list[str], dict]:
    predictions: list[str] = []
    references: list[str] = []
    skipped_samples = 0
    model_device = _get_model_device(model)
    max_length = _resolve_embedding_max_length(model, tokenizer)
    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is None:
        raise RuntimeError("masked_lm tokenizer 缺少 mask_token_id，无法执行业务测评")

    for sample in samples:
        prepared = _prepare_masked_lm_sample(tokenizer, sample, max_length=max_length)
        if prepared is None:
            skipped_samples += 1
            continue
        encoded, mask_position, reference_text = prepared
        encoded = {key: value.to(model_device) if hasattr(value, "to") else value for key, value in encoded.items()}
        encoded["input_ids"] = encoded["input_ids"].clone()
        encoded["input_ids"][0, int(mask_position)] = int(mask_token_id)
        with torch.no_grad():
            outputs = model(**encoded)
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise RuntimeError("masked_lm 模型输出缺少 logits")
        predicted_token_id = int(torch.argmax(logits[0, int(mask_position)], dim=-1).item())
        predictions.append(_token_id_to_masked_lm_text(tokenizer, predicted_token_id))
        references.append(reference_text)

    if not predictions:
        raise RuntimeError("masked_lm 业务评测未生成任何预测结果")
    return predictions, {
        "references": references,
        "inference_strategy": "masked_token_prediction",
        "effective_num_samples": len(predictions),
        "skipped_samples": skipped_samples,
    }


def _run_classification(model, tokenizer, config, samples: list[dict]) -> tuple[list[str], dict]:
    predictions: list[str] = []
    dataset_label_map = _build_dataset_label_map(samples)
    alignment_modes: list[str] = []
    for sample in samples:
        text = str(sample.get("input") or "")
        input_pair = sample.get("input_pair")
        if input_pair is not None and str(input_pair).strip():
            inputs = tokenizer(text, str(input_pair), return_tensors="pt", truncation=True, max_length=512)
        else:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_id = int(torch.argmax(logits, dim=-1).item())
        mapped_prediction, alignment_mode = _map_model_label_to_dataset_id(pred_id, config, dataset_label_map)
        predictions.append(mapped_prediction)
        alignment_modes.append(alignment_mode)
    semantic_hits = sum(1 for mode in alignment_modes if mode == "semantic_label")
    return predictions, {
        "label_alignment_mode": "semantic_label" if semantic_hits else "raw_id",
        "semantic_label_hits": semantic_hits,
        "dataset_label_space_size": len(dataset_label_map),
    }


def _run_question_answering(model, tokenizer, samples: list[dict]) -> list[str]:
    predictions: list[str] = []
    model_device = next(model.parameters()).device
    for sample in samples:
        question = str(sample.get("input") or "").strip()
        context = str(sample.get("context") or "").strip()
        if not question or not context:
            continue
        encoded = tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation="only_second",
            max_length=512,
            return_offsets_mapping=True,
        )
        offset_mapping = encoded.pop("offset_mapping")[0].detach().cpu().tolist()
        sequence_ids = encoded.sequence_ids(0)
        encoded = {k: v.to(model_device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        start_idx = int(torch.argmax(outputs.start_logits, dim=-1).item())
        end_idx = int(torch.argmax(outputs.end_logits, dim=-1).item())
        if end_idx < start_idx:
            end_idx = start_idx
        context_token_indexes = [idx for idx, seq_id in enumerate(sequence_ids) if seq_id == 1]
        if not context_token_indexes:
            predictions.append("")
            continue
        context_start = context_token_indexes[0]
        context_end = context_token_indexes[-1]
        start_idx = min(max(start_idx, context_start), context_end)
        end_idx = min(max(end_idx, start_idx), context_end)
        start_char = int(offset_mapping[start_idx][0]) if start_idx < len(offset_mapping) else 0
        end_char = int(offset_mapping[end_idx][1]) if end_idx < len(offset_mapping) else start_char
        predictions.append(context[start_char:end_char].strip())
    return predictions


def _render_qwen_vlm_reranker_prompt(processor, query: str, candidate: str, *, instruction: str) -> str:
    user_content = f"<Instruct>: {instruction}\n<Query>: {query or 'NULL'}\n<Document>: {candidate or 'NULL'}"
    messages = [
        {"role": "system", "content": [{"type": "text", "text": QWEN_VLM_RERANKER_SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": user_content}]},
    ]
    if hasattr(processor, "apply_chat_template") and callable(getattr(processor, "apply_chat_template", None)):
        try:
            return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except TypeError:
            pass
        except Exception:
            pass
    return f"{QWEN_VLM_RERANKER_SYSTEM_PROMPT}\n{user_content}\nAnswer:"


def _run_qwen_vlm_reranker(model, processor, samples: list[dict]) -> tuple[list[str], dict]:
    predictions: list[str] = []
    prediction_scores: list[list[float]] = []
    ranking_relevance: list[list[int]] = []
    model_device = _get_model_device(model)
    model_dtype = _get_model_floating_dtype(model)
    backbone = getattr(model, "_slai_qwen_vlm_reranker_backbone", None) or getattr(model, "model", model)
    score_head = getattr(model, "_slai_qwen_vlm_reranker_score_head", None)
    if score_head is None:
        raise RuntimeError("Qwen VLM reranker 缺少评分头，无法执行业务测评")

    instruction = str(getattr(model, "_slai_qwen_vlm_reranker_instruction", "") or QWEN_VLM_RERANKER_DEFAULT_INSTRUCTION)
    max_length = int(getattr(model, "_slai_qwen_vlm_reranker_max_length", 8192) or 8192)

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    for sample in samples:
        query = str(sample.get("input") or "").strip()
        candidates = [str(item) for item in (sample.get("candidates") or []) if str(item).strip()]
        if not query or not candidates:
            continue

        prompts = [_render_qwen_vlm_reranker_prompt(processor, query, candidate, instruction=instruction) for candidate in candidates]
        processed = processor(text=prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        processed = {key: _move_runtime_tensor(value, model_device, dtype=model_dtype) for key, value in processed.items()}

        with torch.no_grad():
            backbone_outputs = backbone(**processed)
            hidden_states = getattr(backbone_outputs, "last_hidden_state", None)
            if hidden_states is None:
                raise RuntimeError("Qwen VLM reranker backbone 输出缺少 last_hidden_state")
            attention_mask = processed.get("attention_mask")
            if attention_mask is not None and hasattr(attention_mask, "sum"):
                last_indices = attention_mask.to(dtype=torch.long).sum(dim=1) - 1
                last_indices = last_indices.clamp(min=0, max=hidden_states.shape[1] - 1)
                batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
                pooled_states = hidden_states[batch_indices, last_indices]
            else:
                pooled_states = hidden_states[:, -1]
            if _get_model_device(score_head) != pooled_states.device:
                score_head = score_head.to(device=pooled_states.device)
                model._slai_qwen_vlm_reranker_score_head = score_head
            scores_tensor = torch.sigmoid(score_head(pooled_states)).squeeze(-1)

        scores = [float(value) for value in scores_tensor.detach().cpu().tolist()]
        best_index = max(range(len(scores)), key=lambda idx: scores[idx])
        predictions.append(candidates[best_index])
        prediction_scores.append(scores)
        ranking_relevance.append([int(item) for item in (sample.get("reference") or [0] * len(candidates))[: len(candidates)]])

    return predictions, {"prediction_scores": prediction_scores, "ranking_relevance": ranking_relevance, "inference_strategy": "qwen_vlm_yes_no_reranker"}


def _run_reranker(model, tokenizer, samples: list[dict]) -> tuple[list[str], dict]:
    if getattr(model, "_slai_colbert_enabled", False):
        return _run_colbert_reranker(model, tokenizer, samples)
    if getattr(model, "_slai_qwen_vlm_reranker_enabled", False):
        return _run_qwen_vlm_reranker(model, tokenizer, samples)

    predictions: list[str] = []
    prediction_scores: list[list[float]] = []
    ranking_relevance: list[list[int]] = []
    for sample in samples:
        query = str(sample.get("input") or "")
        candidates = [str(item) for item in (sample.get("candidates") or []) if str(item).strip()]
        if not query or not candidates:
            continue
        inputs = tokenizer(
            [query] * len(candidates),
            candidates,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        scores_tensor = logits.squeeze(-1) if logits.ndim > 1 else logits
        scores = [float(value) for value in scores_tensor.detach().cpu().tolist()]
        best_index = max(range(len(scores)), key=lambda idx: scores[idx])
        predictions.append(candidates[best_index])
        prediction_scores.append(scores)
        ranking_relevance.append([int(item) for item in (sample.get("reference") or [0] * len(candidates))[: len(candidates)]])
    return predictions, {"prediction_scores": prediction_scores, "ranking_relevance": ranking_relevance}


def _prepend_colbert_marker(text: str, marker: str | None) -> str:
    stripped = str(text or "").strip()
    return f"{marker} {stripped}".strip() if marker else stripped


def _encode_colbert_inputs(model, tokenizer, texts: list[str], *, marker: str | None, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    prepared_texts = [_prepend_colbert_marker(text, marker) for text in texts]
    encoded = tokenizer(prepared_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    model_device = next(model.parameters()).device
    encoded = {key: value.to(model_device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)

    hidden = _extract_last_hidden_state(outputs)
    projection = getattr(model, "_slai_colbert_projection", None)
    if isinstance(projection, torch.nn.Module):
        projection_device = next(projection.parameters()).device
        if projection_device != hidden.device:
            projection = projection.to(hidden.device)
            model._slai_colbert_projection = projection
        projection_dtype = next(projection.parameters()).dtype
        hidden = projection(hidden.to(dtype=projection_dtype))
    hidden = F.normalize(hidden, p=2, dim=-1)

    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        token_mask = attention_mask.to(dtype=torch.bool)
    else:
        token_mask = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.bool)

    skip_token_ids = {int(token_id) for token_id in list(getattr(model, "_slai_colbert_skip_token_ids", set()) or [])}
    input_ids = encoded.get("input_ids")
    if input_ids is not None and skip_token_ids:
        for token_id in skip_token_ids:
            token_mask &= input_ids.ne(int(token_id))

    return hidden, token_mask


def _score_colbert_candidates(query_embeddings: torch.Tensor, query_mask: torch.Tensor, doc_embeddings: torch.Tensor, doc_mask: torch.Tensor) -> list[float]:
    valid_query_mask = query_mask[0]
    if not bool(valid_query_mask.any().item()):
        valid_query_mask = query_mask[0].clone()
        valid_query_mask[:] = True
    query_vectors = query_embeddings[0, valid_query_mask, :]
    if query_vectors.ndim != 2 or query_vectors.shape[0] == 0:
        return [0.0 for _ in range(int(doc_embeddings.shape[0]))]

    scores = torch.einsum("qd,nkd->nqk", query_vectors, doc_embeddings)
    doc_valid_mask = doc_mask[:, None, :]
    scores = scores.masked_fill(~doc_valid_mask, float("-inf"))
    max_scores = scores.max(dim=-1).values
    doc_has_tokens = doc_mask.any(dim=-1, keepdim=True)
    max_scores = torch.where(doc_has_tokens, max_scores, torch.zeros_like(max_scores))
    max_scores = torch.where(torch.isfinite(max_scores), max_scores, torch.zeros_like(max_scores))
    return [float(value) for value in max_scores.sum(dim=-1).detach().cpu().tolist()]


def _run_colbert_reranker(model, tokenizer, samples: list[dict]) -> tuple[list[str], dict]:
    predictions: list[str] = []
    prediction_scores: list[list[float]] = []
    ranking_relevance: list[list[int]] = []
    for sample in samples:
        query = str(sample.get("input") or "").strip()
        candidates = [str(item) for item in (sample.get("candidates") or []) if str(item).strip()]
        if not query or not candidates:
            continue

        query_embeddings, query_mask = _encode_colbert_inputs(
            model,
            tokenizer,
            [query],
            marker=getattr(model, "_slai_colbert_query_marker", None),
            max_length=int(getattr(model, "_slai_colbert_query_max_length", 64) or 64),
        )
        doc_embeddings, doc_mask = _encode_colbert_inputs(
            model,
            tokenizer,
            candidates,
            marker=getattr(model, "_slai_colbert_doc_marker", None),
            max_length=int(getattr(model, "_slai_colbert_doc_max_length", 256) or 256),
        )
        scores = _score_colbert_candidates(query_embeddings, query_mask, doc_embeddings, doc_mask)
        if not scores:
            continue
        best_index = max(range(len(scores)), key=lambda idx: scores[idx])
        predictions.append(candidates[best_index])
        prediction_scores.append(scores)
        ranking_relevance.append([int(item) for item in (sample.get("reference") or [0] * len(candidates))[: len(candidates)]])

    return predictions, {
        "prediction_scores": prediction_scores,
        "ranking_relevance": ranking_relevance,
        "inference_strategy": "colbert_late_interaction",
        "colbert_projection_dim": int(getattr(model, "_slai_colbert_projection_dim", 0) or 0),
        "colbert_query_marker": getattr(model, "_slai_colbert_query_marker", None),
        "colbert_doc_marker": getattr(model, "_slai_colbert_doc_marker", None),
        "colbert_marker_strategy": getattr(model, "_slai_colbert_marker_strategy", "plain_text"),
    }


def _mean_pool_embeddings(model_output, attention_mask):
    hidden = _extract_text_embedding_hidden(model_output)
    mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
    pooled = torch.sum(hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
    return F.normalize(pooled, p=2, dim=1)


def _extract_text_embedding_hidden(model_output):
    if isinstance(model_output, torch.Tensor):
        return model_output
    if isinstance(model_output, (tuple, list)) and model_output:
        return _extract_text_embedding_hidden(model_output[0])
    if isinstance(model_output, Mapping):
        tensor_candidates: list[torch.Tensor] = []
        prioritized_keys = (
            "sentence_embedding",
            "sentence_embeddings",
            "embedding",
            "embeddings",
            "pooler_output",
            "cls_embeddings",
            "last_hidden_state",
            "logits",
        )
        for key in prioritized_keys:
            candidate = model_output.get(key)
            if isinstance(candidate, torch.Tensor):
                return candidate
        for candidate in model_output.values():
            if isinstance(candidate, torch.Tensor):
                tensor_candidates.append(candidate)
        if tensor_candidates:
            return tensor_candidates[0]
    hidden = getattr(model_output, "last_hidden_state", None)
    if isinstance(hidden, torch.Tensor):
        return hidden
    for attr_name in ("pooler_output", "sentence_embedding", "sentence_embeddings", "embeddings", "logits"):
        candidate = getattr(model_output, attr_name, None)
        if isinstance(candidate, torch.Tensor):
            return candidate
    raise RuntimeError("embedding 模型输出缺少可提取的 hidden/embedding tensor")


def _pool_text_embeddings(model_output, attention_mask, output_type_hint: str) -> torch.Tensor:
    normalized_hint = str(output_type_hint or "").strip().lower()
    hidden = _extract_text_embedding_hidden(model_output)
    if hidden.ndim == 1:
        hidden = hidden.unsqueeze(0)
    if hidden.ndim == 2:
        return F.normalize(hidden, p=2, dim=-1)
    if normalized_hint == "cls_embeddings":
        if hidden.ndim < 3:
            raise RuntimeError("embedding 输出缺少 token hidden states，无法提取 CLS embedding")
        return F.normalize(hidden[:, 0, :], p=2, dim=1)
    return _mean_pool_embeddings(model_output, attention_mask)


def _sync_device(device: torch.device) -> None:
    if device.type == "npu":
        npu_module = getattr(torch, "npu", None)
        if npu_module is not None:
            npu_module.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _resolve_accuracy_latency_bridge_vector_builder(processor: Any):
    module_candidates = [
        getattr(processor, "runtime_module", None),
        getattr(processor, "accuracy_module", None),
        getattr(processor, "perf_module", None),
    ]
    for module in module_candidates:
        candidate = getattr(module, "build_prompt_vector", None) if module is not None else None
        if callable(candidate):
            return candidate

    runtime_module = getattr(processor, "runtime_module", None)
    setup_model = getattr(runtime_module, "setup_model", None) if runtime_module is not None else None
    helper_module_name = str(getattr(setup_model, "__module__", "") or "").strip()
    if helper_module_name:
        helper_module = sys.modules.get(helper_module_name)
        if helper_module is None:
            try:
                helper_module = importlib.import_module(helper_module_name)
            except Exception:
                helper_module = None
        candidate = getattr(helper_module, "build_prompt_vector", None) if helper_module is not None else None
        if callable(candidate):
            return candidate
    return None


def _build_accuracy_latency_bridge_input(model, processor, sample: dict[str, Any], sample_index: int) -> torch.Tensor:
    model_device = _get_model_device(model)
    target_dtype = _get_model_floating_dtype(model) or torch.float32
    requested_model_type = str(getattr(processor, "requested_model_type", "") or "").strip().lower()
    output_type_hint = str(getattr(processor, "output_type_hint", "") or "").strip().lower()
    seed = int(sample.get("seed") or (20260422 + sample_index))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if requested_model_type == "embedding" and output_type_hint == "embeddings":
        hidden_size = int(getattr(model, "hidden_size", 0) or 0)
        if hidden_size <= 0:
            first_param = next(model.parameters(), None)
            if first_param is not None and first_param.ndim >= 2:
                hidden_size = int(first_param.shape[-1])
        if hidden_size <= 0:
            hidden_size = 768
        tensor = torch.randn((1, hidden_size), generator=generator, dtype=torch.float32)
        return tensor.to(device=model_device, dtype=target_dtype)
    if requested_model_type == "video" or output_type_hint == "video_latency":
        prompt_vector_builder = _resolve_accuracy_latency_bridge_vector_builder(processor)
        prompt_text = str(sample.get("input") or sample.get("prompt_text") or "").strip()
        hidden_size = int(getattr(model, "hidden_size", 0) or 0)
        if callable(prompt_vector_builder) and hidden_size > 0 and prompt_text:
            prompt_vector = prompt_vector_builder(prompt_text, sample_index, hidden_size)
            if isinstance(prompt_vector, torch.Tensor):
                return prompt_vector.to(device=model_device, dtype=target_dtype)
        tensor = torch.randn((1, 3, 8, 64, 64), generator=generator, dtype=torch.float32)
    else:
        tensor = torch.randn((1, 4, 64, 64), generator=generator, dtype=torch.float32)
    return tensor.to(device=model_device, dtype=target_dtype)


def _run_accuracy_latency_bridge(model, processor, samples: list[dict], config: dict[str, Any], scenario: str = "") -> tuple[list[str], dict]:
    predictions: list[str] = []
    latencies: list[float] = []
    prediction_embeddings: list[list[float]] = []
    reference_embeddings: list[list[float]] = []
    model_device = _get_model_device(model)
    device_short = "npu" if model_device.type == "npu" else "cuda" if model_device.type == "cuda" else "cpu"
    run_single_fn = getattr(processor, "run_single_fn", None)
    empty_cache_fn = getattr(processor, "empty_cache_fn", None)
    requested_model_type = str(getattr(processor, "requested_model_type", "") or "").strip().lower()
    output_type_hint = str(getattr(processor, "output_type_hint", "") or "").strip().lower()
    if not output_type_hint:
        output_type_hint = "video_latency" if requested_model_type == "video" else "diffusion_latency"
    inference_strategy = "accuracy_run_video_latency_bridge" if requested_model_type == "video" or output_type_hint == "video_latency" else "accuracy_run_diffusion_latency_bridge"

    if requested_model_type == "embedding" and output_type_hint == "embeddings":
        embedding_batch_size = _get_embedding_batch_size(config, scenario)
        steady_state_repeats = _get_embedding_steady_state_repeats(config, scenario)
        prepared_inputs: list[torch.Tensor] = []
        for sample_index, sample in enumerate(samples):
            prepared_inputs.append(_build_accuracy_latency_bridge_input(model, processor, sample, sample_index))
        if not prepared_inputs:
            return [], {"prediction_embeddings": [], "reference_embeddings": []}

        effective_batch_size = 0
        start_time = datetime.now().isoformat()
        import time

        for repeat_idx in range(steady_state_repeats):
            store_outputs = repeat_idx == 0
            for batch_start in range(0, len(prepared_inputs), embedding_batch_size):
                batch_tensors = prepared_inputs[batch_start : batch_start + embedding_batch_size]
                batch_tensor = torch.cat(batch_tensors, dim=0)
                effective_batch_size = max(effective_batch_size, batch_tensor.shape[0])
                _sync_device(model_device)
                t0 = time.perf_counter()
                with torch.no_grad():
                    output = model(batch_tensor)
                _sync_device(model_device)
                latencies.append(time.perf_counter() - t0)
                if store_outputs:
                    output_tensor = output.detach().float().cpu()
                    if output_tensor.ndim == 1:
                        output_tensor = output_tensor.unsqueeze(0)
                    for row in output_tensor:
                        embedding = [float(value) for value in row.flatten().tolist()]
                        prediction_embeddings.append(embedding)
                        reference_embeddings.append(list(embedding))
                        predictions.append(f"{output_type_hint}:{row.numel()}")
                del batch_tensor, output
                if callable(empty_cache_fn):
                    try:
                        empty_cache_fn(device_short)
                    except TypeError:
                        empty_cache_fn()

        wall_clock_s = sum(latencies)
        end_time = datetime.now().isoformat()
        metric_context: dict[str, Any] = {
            "references": [sample.get("reference") for sample in samples],
            "prediction_embeddings": prediction_embeddings,
            "reference_embeddings": reference_embeddings,
            "embedding_batch_size_requested": embedding_batch_size,
            "embedding_batch_size_effective": effective_batch_size or embedding_batch_size,
            "steady_state_repeat_iterations": steady_state_repeats,
            "inference_strategy": "accuracy_run_local_embedding_bridge",
            "inference_latency_s": wall_clock_s / max(len(prediction_embeddings) * steady_state_repeats, 1),
            "wall_clock_s": wall_clock_s,
            "start_time": start_time,
            "end_time": end_time,
        }
        if latencies:
            metric_context["sample_latency_min_s"] = min(latencies)
            metric_context["sample_latency_max_s"] = max(latencies)
        return predictions, metric_context

    for sample_index, sample in enumerate(samples):
        sample_input = _build_accuracy_latency_bridge_input(model, processor, sample, sample_index)
        if callable(run_single_fn):
            output, latency_s, _peak_memory_mb = run_single_fn(model, str(model_device), device_short, input_tensor=sample_input)
            _sync_device(model_device)
        else:
            import time

            with torch.no_grad():
                start_ts = time.perf_counter()
                output = model(sample_input)
                _sync_device(model_device)
                latency_s = time.perf_counter() - start_ts
        latencies.append(float(latency_s))
        if isinstance(output, torch.Tensor):
            if requested_model_type == "embedding" and output_type_hint == "embeddings":
                output_tensor = output.detach().float().cpu()
                if output_tensor.ndim == 1:
                    output_tensor = output_tensor.unsqueeze(0)
                for row in output_tensor:
                    embedding = [float(value) for value in row.flatten().tolist()]
                    prediction_embeddings.append(embedding)
                    reference_embeddings.append(list(embedding))
            predictions.append(f"{output_type_hint}:{'x'.join(str(dim) for dim in output.shape)}")
        else:
            predictions.append(output_type_hint)
        del sample_input, output
        if callable(empty_cache_fn):
            try:
                empty_cache_fn(device_short)
            except TypeError:
                empty_cache_fn()

    wall_clock_s = sum(latencies)
    metric_context: dict[str, Any] = {
        "references": [sample.get("reference") for sample in samples],
        "inference_strategy": inference_strategy,
        "inference_latency_s": wall_clock_s / max(len(latencies), 1),
        "wall_clock_s": wall_clock_s,
        "steady_state_repeat_iterations": len(latencies),
    }
    if prediction_embeddings and reference_embeddings:
        metric_context["prediction_embeddings"] = prediction_embeddings
        metric_context["reference_embeddings"] = reference_embeddings
    if latencies:
        metric_context["sample_latency_min_s"] = min(latencies)
        metric_context["sample_latency_max_s"] = max(latencies)
    return predictions, metric_context


def _enable_cuda_vision_runtime_optimizations(model, device: torch.device) -> None:
    if device.type != "cuda":
        return
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    try:
        model.to(memory_format=torch.channels_last)
    except Exception:
        pass


def _prepare_cuda_vision_tensor(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type != "cuda" or tensor.ndim != 4:
        return tensor
    try:
        return tensor.contiguous(memory_format=torch.channels_last)
    except Exception:
        return tensor


def _resolve_embedding_max_length(model, tokenizer) -> int:
    candidate_lengths: list[int] = []
    tokenizer_max_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_max_length, int) and 0 < tokenizer_max_length < 100000:
        candidate_lengths.append(int(tokenizer_max_length))
    model_config = getattr(model, "config", None)
    for attr_name in ("max_position_embeddings", "text_config"):
        attr_value = getattr(model_config, attr_name, None)
        if attr_name == "text_config" and attr_value is not None:
            nested_max_positions = getattr(attr_value, "max_position_embeddings", None)
            if isinstance(nested_max_positions, int) and nested_max_positions > 0:
                candidate_lengths.append(int(nested_max_positions))
            continue
        if isinstance(attr_value, int) and attr_value > 0:
            candidate_lengths.append(int(attr_value))
    return max(1, min(candidate_lengths)) if candidate_lengths else 512


def _encode_embedding_texts(model, tokenizer, texts: list[str], device: torch.device) -> list[list[float]]:
    if not texts:
        return []
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=_resolve_embedding_max_length(model, tokenizer), return_tensors="pt")
    return _encode_embedding_batch(model, encoded, device)


def _encode_embedding_batch(model, encoded_inputs: Mapping[str, Any], device: torch.device, *, output_type_hint: str = "embeddings") -> list[list[float]]:
    encoded = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in encoded_inputs.items()}
    with torch.no_grad():
        model_config = getattr(model, "config", None)
        if getattr(model_config, "is_encoder_decoder", False):
            encoder = model.get_encoder() if hasattr(model, "get_encoder") else getattr(model, "encoder", None)
            if encoder is None:
                raise RuntimeError("encoder-decoder 模型缺少 encoder，无法走 embedding pooling 路径")
            encoder_kwargs = {key: encoded[key] for key in ("input_ids", "attention_mask", "inputs_embeds") if key in encoded}
            encoder_kwargs["return_dict"] = True
            outputs = encoder(**encoder_kwargs)
        else:
            outputs = model(**encoded)
    pooled = _pool_text_embeddings(outputs, encoded["attention_mask"], output_type_hint=output_type_hint).detach().cpu()
    return [[float(x) for x in row] for row in pooled.tolist()]


def _get_embedding_batch_size(config: dict, scenario: str) -> int:
    scenario_field = {
        "npu_baseline": "embedding_baseline_batch_size",
        "npu_perf": "embedding_perf_batch_size",
        "cuda_baseline": "embedding_cuda_baseline_batch_size",
    }
    raw_value = config.get(scenario_field.get(scenario, ""), None)
    if raw_value in {None, ""}:
        raw_value = config.get("embedding_batch_size", 1)
    try:
        return max(int(raw_value or 1), 1)
    except Exception:
        return 1


def _get_embedding_steady_state_repeats(config: dict, scenario: str) -> int:
    scenario_field = {
        "npu_baseline": "embedding_baseline_steady_state_repeats",
        "npu_perf": "embedding_perf_steady_state_repeats",
        "cuda_baseline": "embedding_cuda_baseline_steady_state_repeats",
    }
    raw_value = config.get(scenario_field.get(scenario, ""), None)
    if raw_value in {None, ""}:
        raw_value = config.get("embedding_steady_state_repeats", config.get("steady_state_repeat_iterations", 1))
    try:
        return max(int(raw_value or 1), 1)
    except Exception:
        return 1


def _extract_discriminator_logits(model_output) -> torch.Tensor:
    logits = getattr(model_output, "logits", None)
    if logits is None and isinstance(model_output, (tuple, list)) and model_output:
        logits = model_output[0]
    if logits is None:
        raise RuntimeError("discriminator 模型输出缺少 logits，无法构造业务评测向量")
    if not torch.is_tensor(logits):
        logits = torch.as_tensor(logits)
    if logits.ndim == 0:
        logits = logits.reshape(1, 1)
    elif logits.ndim == 1:
        logits = logits.unsqueeze(0)
    elif logits.ndim == 3:
        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        else:
            logits = logits.float().mean(dim=-1)
    return logits.float()


def _pool_discriminator_logits(logit_row: torch.Tensor, attention_mask_row: torch.Tensor | None, *, pooled_bins: int = 32) -> list[float]:
    token_scores = logit_row.reshape(-1)
    if attention_mask_row is not None:
        valid_mask = attention_mask_row.to(dtype=torch.bool).reshape(-1)
        valid_length = min(token_scores.shape[0], valid_mask.shape[0])
        token_scores = token_scores[:valid_length]
        valid_mask = valid_mask[:valid_length]
        token_scores = token_scores[valid_mask]
    if token_scores.numel() == 0:
        token_scores = logit_row.reshape(-1)

    token_scores = torch.sigmoid(token_scores.float())
    pooled = F.adaptive_avg_pool1d(token_scores.view(1, 1, -1), pooled_bins).reshape(-1)
    spread = token_scores.std(unbiased=False) if token_scores.numel() > 1 else torch.zeros((), device=token_scores.device)
    summary = torch.stack((token_scores.mean(), spread, token_scores.min(), token_scores.max()))
    embedding = torch.cat((pooled, summary), dim=0)
    embedding = F.normalize(embedding.unsqueeze(0), p=2, dim=1)[0].detach().cpu()
    return [float(x) for x in embedding.tolist()]


def _encode_discriminator_texts(model, tokenizer, texts: list[str], device: torch.device) -> list[list[float]]:
    if not texts:
        return []
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
    logits = _extract_discriminator_logits(outputs)
    attention_mask = encoded.get("attention_mask")
    rows: list[list[float]] = []
    for idx in range(int(logits.shape[0])):
        mask_row = attention_mask[idx] if attention_mask is not None else None
        rows.append(_pool_discriminator_logits(logits[idx], mask_row))
    return rows


def _run_embedding(model, tokenizer, samples: list[dict], scenario: str, config: dict) -> tuple[list[str], dict]:
    predictions: list[str] = []
    prediction_embeddings: list[list[float]] = []
    reference_embeddings: list[list[float]] = []
    model_device = next(model.parameters()).device
    max_length = _resolve_embedding_max_length(model, tokenizer)
    embedding_batch_size = _get_embedding_batch_size(config, scenario)
    output_type_hint = str(config.get("output_type_hint") or "").strip().lower()
    latency_scope = str(getattr(config, "latency_measurement_scope", None) or os.environ.get("BUSINESS_BENCHMARK_LATENCY_MEASUREMENT_SCOPE") or "").strip().lower()
    steady_state_repeats = 1
    prepared_inputs: list[tuple[list[str], dict[str, torch.Tensor], dict[str, torch.Tensor] | None]] = []

    for start in range(0, len(samples), embedding_batch_size):
        batch_samples = samples[start : start + embedding_batch_size]
        input_texts: list[str] = []
        reference_texts: list[str] = []
        for sample in batch_samples:
            input_text = str(sample.get("input") or "").strip()
            reference_text = str(sample.get("reference") or input_text).strip()
            if not input_text:
                continue
            input_texts.append(input_text)
            reference_texts.append(reference_text)
        if not input_texts:
            continue
        pred_inputs = tokenizer(input_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        ref_inputs = None
        if any(reference_text != input_text for input_text, reference_text in zip(input_texts, reference_texts)):
            ref_inputs = tokenizer(reference_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        prepared_inputs.append((input_texts, pred_inputs, ref_inputs))

    inference_start_ts = None
    if latency_scope == "steady_state" and prepared_inputs:
        import time

        steady_state_repeats = _get_embedding_steady_state_repeats(config, scenario)
        _sync_device(model_device)
        inference_start_ts = time.perf_counter()

    effective_batch_size = 0
    for repeat_idx in range(steady_state_repeats):
        store_outputs = repeat_idx == 0
        for input_texts, pred_inputs, ref_inputs in prepared_inputs:
            pred_embedding_batch = _encode_embedding_batch(model, pred_inputs, model_device, output_type_hint=output_type_hint)
            ref_embedding_batch = pred_embedding_batch if ref_inputs is None else _encode_embedding_batch(model, ref_inputs, model_device, output_type_hint=output_type_hint)
            effective_batch_size = max(effective_batch_size, len(input_texts))
            if not store_outputs:
                continue
            predictions.extend(input_texts)
            prediction_embeddings.extend(pred_embedding_batch)
            reference_embeddings.extend(ref_embedding_batch)
    metric_context = {
        "prediction_embeddings": prediction_embeddings,
        "reference_embeddings": reference_embeddings,
        "embedding_batch_size_requested": embedding_batch_size,
        "embedding_batch_size_effective": effective_batch_size or embedding_batch_size,
        "steady_state_repeat_iterations": steady_state_repeats,
        "inference_strategy": "cls_embedding_pooling" if output_type_hint == "cls_embeddings" else "mean_pool_embedding",
    }
    if inference_start_ts is not None and predictions:
        import time

        _sync_device(model_device)
        metric_context["inference_latency_s"] = (time.perf_counter() - inference_start_ts) / (len(predictions) * steady_state_repeats)
    return predictions, metric_context


def _run_discriminator(model, tokenizer, samples: list[dict]) -> tuple[list[str], dict]:
    predictions: list[str] = []
    prediction_embeddings: list[list[float]] = []
    reference_embeddings: list[list[float]] = []
    model_device = next(model.parameters()).device

    for sample in samples:
        input_text = str(sample.get("input") or "").strip()
        reference_text = str(sample.get("reference") or input_text).strip()
        if not input_text:
            continue
        pred_embedding = _encode_discriminator_texts(model, tokenizer, [input_text], model_device)[0]
        ref_embedding = pred_embedding if reference_text == input_text else _encode_discriminator_texts(model, tokenizer, [reference_text], model_device)[0]
        predictions.append(input_text)
        prediction_embeddings.append(pred_embedding)
        reference_embeddings.append(ref_embedding)
    return predictions, {
        "prediction_embeddings": prediction_embeddings,
        "reference_embeddings": reference_embeddings,
        "inference_strategy": "discriminator_logits_pooling",
    }


def _run_audio_embedding(model, processor, samples: list[dict]) -> tuple[list[str], dict]:
    predictions: list[str] = []
    prediction_embeddings: list[list[float]] = []
    reference_embeddings: list[list[float]] = []
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    for idx, sample in enumerate(samples):
        audio = sample.get("input")
        if audio is None:
            continue
        processed = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        processed = {
            key: value.to(device=model_device, dtype=model_dtype) if torch.is_floating_point(value) else value.to(device=model_device)
            for key, value in processed.items()
        }
        with torch.no_grad():
            output = model(**processed)
        hidden = output[0] if isinstance(output, (tuple, list)) else output.last_hidden_state
        pooled = F.normalize(hidden.mean(dim=1), p=2, dim=1)[0].detach().cpu().tolist()
        predictions.append(f"audio_sample_{idx}")
        prediction_embeddings.append([float(x) for x in pooled])
        reference_embeddings.append([float(x) for x in pooled])
    return predictions, {"prediction_embeddings": prediction_embeddings, "reference_embeddings": reference_embeddings}


def _decode_asr_batch(processor, token_ids) -> list[str]:
    processor_class_name = processor.__class__.__name__.lower()
    tokenizer = getattr(processor, "tokenizer", None)

    # Wav2Vec2ProcessorWithLM.batch_decode expects frame-level logits for LM beam
    # search. When the ASR path already produced token ids via argmax, decode
    # through the tokenizer instead of re-entering the LM decoder.
    if "wav2vec2processorwithlm" in processor_class_name and tokenizer is not None and hasattr(tokenizer, "batch_decode"):
        try:
            decoded = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        except TypeError:
            decoded = tokenizer.batch_decode(token_ids)
        if decoded:
            return [str(item).strip() for item in decoded]

    if hasattr(processor, "batch_decode"):
        try:
            decoded = processor.batch_decode(token_ids, skip_special_tokens=True)
        except TypeError:
            decoded = processor.batch_decode(token_ids)
        except Exception:
            decoded = None
        if decoded:
            return [str(item).strip() for item in decoded]
    if tokenizer is not None and hasattr(tokenizer, "batch_decode"):
        try:
            decoded = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        except TypeError:
            decoded = tokenizer.batch_decode(token_ids)
        if decoded:
            return [str(item).strip() for item in decoded]
    return []


def _decode_asr_tokens(processor, token_ids) -> str:
    decoded = _decode_asr_batch(processor, token_ids)
    if decoded:
        return decoded[0]
    return ""


def _sample_sampling_rate(sample: dict, default: int = 16000) -> int:
    value = sample.get("sampling_rate")
    try:
        rate = int(value)
    except Exception:
        rate = default
    return rate if rate > 0 else default


def _get_audio_processor_sampling_rate(processor, default: int = 16000) -> int:
    candidates = (
        getattr(getattr(processor, "feature_extractor", None), "sampling_rate", None),
        getattr(processor, "sampling_rate", None),
        getattr(getattr(processor, "tokenizer", None), "sampling_rate", None),
    )
    for candidate in candidates:
        try:
            rate = int(candidate)
        except Exception:
            continue
        if rate > 0:
            return rate
    return default


def _resample_audio_input(audio_input: Any, *, source_sampling_rate: int, target_sampling_rate: int):
    if audio_input is None or source_sampling_rate <= 0 or target_sampling_rate <= 0 or source_sampling_rate == target_sampling_rate:
        return audio_input
    audio_array = np.asarray(audio_input, dtype=np.float32)
    if audio_array.ndim >= 2:
        channel_axis = 0 if audio_array.shape[0] <= 8 and audio_array.shape[-1] > 8 else -1
        audio_array = audio_array.mean(axis=channel_axis)
    audio_array = audio_array.astype(np.float32, copy=False).reshape(-1)
    if audio_array.size == 0:
        return audio_array
    target_length = max(int(round(audio_array.shape[0] * float(target_sampling_rate) / float(source_sampling_rate))), 1)
    audio_tensor = torch.from_numpy(audio_array).reshape(1, 1, -1)
    resampled = F.interpolate(audio_tensor, size=target_length, mode="linear", align_corners=False).reshape(-1).cpu().numpy()
    return resampled.astype(np.float32, copy=False)


def _is_multilingual_asr_dataset(dataset_key: str | None) -> bool:
    normalized = str(dataset_key or "").strip().lower()
    return normalized == "librispeech" or normalized.startswith("fleurs_") or normalized.startswith("mcspeech_")


def _build_asr_generate_kwargs(config: dict) -> tuple[dict[str, Any], dict[str, Any]]:
    max_new_tokens = max(int(config.get("asr_max_new_tokens") or 128), 1)
    dataset_key = str(config.get("dataset") or "").strip().lower()
    language = str(config.get("asr_language") or "").strip()
    task = str(config.get("asr_task") or "").strip()
    if _is_multilingual_asr_dataset(dataset_key):
        if dataset_key == "librispeech":
            language = language or "en"
        task = task or "transcribe"

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if language:
        generate_kwargs["language"] = language
    if task:
        generate_kwargs["task"] = task
    metadata = {
        "asr_max_new_tokens": max_new_tokens,
        "asr_language": language,
        "asr_task": task,
    }
    return generate_kwargs, metadata


def _is_whisper_asr_model(model) -> bool:
    model_config = getattr(model, "config", None)
    model_type = str(getattr(model_config, "model_type", "") or "").strip().lower()
    if model_type == "whisper":
        return True
    class_name = str(getattr(model.__class__, "__name__", "") or "").strip().lower()
    return "whisper" in class_name


def _should_disable_whisper_suppress_tokens(model, config: dict, generate_kwargs: dict[str, Any]) -> bool:
    if not _is_whisper_asr_model(model):
        return False
    override = str(config.get("asr_disable_suppress_tokens") or "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False
    if _as_bool(config.get("asr_return_timestamps"), default=False):
        return False
    dataset_key = str(config.get("dataset") or "").strip().lower()
    task = str(generate_kwargs.get("task") or config.get("asr_task") or "").strip().lower()
    return _is_multilingual_asr_dataset(dataset_key) and task in {"", "transcribe"}


def _get_asr_batch_size(config: dict, scenario: str) -> int:
    scenario_field = {
        "npu_baseline": "asr_baseline_batch_size",
        "npu_perf": "asr_perf_batch_size",
        "cuda_baseline": "asr_cuda_baseline_batch_size",
    }
    raw_value = config.get(scenario_field.get(scenario, ""), None)
    if raw_value in {None, ""}:
        raw_value = config.get("asr_batch_size", 1)
    try:
        return max(int(raw_value or 1), 1)
    except Exception:
        return 1


def _prepare_asr_processed_inputs(processor, audio_inputs: list[Any], *, sampling_rate: int, model_device, model_dtype):
    target_sampling_rate = _get_audio_processor_sampling_rate(processor, default=sampling_rate)
    normalized_audio_inputs = [
        _resample_audio_input(
            audio_input,
            source_sampling_rate=sampling_rate,
            target_sampling_rate=target_sampling_rate,
        )
        for audio_input in audio_inputs
    ]
    processor_kwargs = {
        "sampling_rate": target_sampling_rate,
        "return_tensors": "pt",
        "return_attention_mask": True,
    }
    if len(normalized_audio_inputs) > 1:
        processor_kwargs["padding"] = True
    processor_input = normalized_audio_inputs if len(normalized_audio_inputs) > 1 else normalized_audio_inputs[0]
    try:
        processed = processor(processor_input, **processor_kwargs)
    except TypeError:
        processor_kwargs.pop("return_attention_mask", None)
        processed = processor(processor_input, **processor_kwargs)
    return {key: _move_runtime_tensor(value, model_device, dtype=model_dtype) for key, value in processed.items()}


def _extract_qwen_asr_text(result: Any) -> str:
    if result is None:
        return ""
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(result, dict):
        return str(result.get("text") or "")
    return ""


def _extract_nemo_asr_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text.strip()
    if isinstance(result, dict):
        for key in ("text", "pred_text", "transcript"):
            value = result.get(key)
            if isinstance(value, str):
                return value.strip()
    if isinstance(result, (list, tuple)) and result:
        return _extract_nemo_asr_text(result[0])
    return ""


def _normalize_qwen_asr_language(config: dict[str, Any]) -> str | None:
    raw_language = str(config.get("asr_language") or "").strip()
    if not raw_language or raw_language.lower() in {"auto", "none"}:
        return None
    language_map = {
        "en": "English",
        "english": "English",
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "zh_cn": "Chinese",
        "chinese": "Chinese",
        "yue": "Cantonese",
        "cantonese": "Cantonese",
    }
    return language_map.get(raw_language.lower(), raw_language if raw_language[:1].isupper() else raw_language.title())


def _contains_cjk_character(value: str) -> bool:
    for ch in str(value or ""):
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0x20000 <= code <= 0x2A6DF or 0x2A700 <= code <= 0x2B73F or 0x2B740 <= code <= 0x2B81F or 0x2B820 <= code <= 0x2CEAF or 0xF900 <= code <= 0xFAFF:
            return True
    return False


def _normalize_alignment_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _extract_qwen_forced_alignment_items(result: Any) -> list[dict[str, Any]]:
    if result is None:
        return []
    try:
        raw_items = list(result)
    except Exception:
        raw_items = []
    items: list[dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            start_time = item.get("start_time")
            end_time = item.get("end_time")
        else:
            text = str(getattr(item, "text", "") or "").strip()
            start_time = getattr(item, "start_time", None)
            end_time = getattr(item, "end_time", None)
        if not text:
            continue
        try:
            start_value = float(start_time)
        except Exception:
            start_value = 0.0
        try:
            end_value = float(end_time)
        except Exception:
            end_value = 0.0
        items.append(
            {
                "text": text,
                "start_time": round(start_value, 3),
                "end_time": round(end_value, 3),
            }
        )
    return items


def _join_qwen_forced_alignment_tokens(tokens: list[str]) -> str:
    cleaned_tokens = [str(token or "").strip() for token in tokens if str(token or "").strip()]
    if not cleaned_tokens:
        return ""
    if all(_contains_cjk_character(token) for token in cleaned_tokens):
        return "".join(cleaned_tokens)
    return _normalize_alignment_text(" ".join(cleaned_tokens))


def _estimate_alignment_token_coverage(reference_text: Any, aligned_tokens: list[str]) -> float:
    reference_tokens = [token for token in _normalize_alignment_text(reference_text).lower().split(" ") if token]
    if not reference_tokens:
        return 0.0
    return max(0.0, min(1.0, len(aligned_tokens) / len(reference_tokens)))


def _normalize_qwen_tts_language(config: dict[str, Any]) -> str:
    raw_language = str(config.get("tts_language") or config.get("asr_language") or "").strip()
    if not raw_language or raw_language.lower() in {"auto", "none"}:
        return "English"
    language_map = {
        "en": "English",
        "english": "English",
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "zh_cn": "Chinese",
        "chinese": "Chinese",
        "yue": "Cantonese",
        "cantonese": "Cantonese",
    }
    return language_map.get(raw_language.lower(), raw_language if raw_language[:1].isupper() else raw_language.title())


def _seed_from_text(seed_text: str) -> int:
    normalized = str(seed_text or "").strip() or "slai-vocos"
    seed = sum((index + 1) * ord(char) for index, char in enumerate(normalized[:512])) % (2**31 - 1)
    return seed or 42


def _coerce_vocos_mel_input(sample_input: Any, processor: Any, model, config: dict[str, Any]) -> torch.Tensor:
    mel_value = sample_input
    if isinstance(sample_input, Mapping):
        for key in ("mel", "mel_spec", "mel_spectrogram", "spectrogram", "input_features"):
            candidate = sample_input.get(key)
            if candidate is not None:
                mel_value = candidate
                break

    mel_tensor: torch.Tensor | None = None
    if isinstance(mel_value, torch.Tensor):
        mel_tensor = mel_value.detach().clone()
    elif isinstance(mel_value, np.ndarray):
        mel_tensor = torch.from_numpy(np.array(mel_value, copy=True))
    elif isinstance(mel_value, (list, tuple)) and mel_value:
        try:
            mel_tensor = torch.tensor(mel_value)
        except Exception:
            mel_tensor = None

    if mel_tensor is None:
        sample_rate = max(int(getattr(processor, "sample_rate", 24000) or 24000), 1)
        n_mels = max(int(getattr(processor, "n_mels", 100) or 100), 1)
        hop_length = max(int(getattr(processor, "hop_length", 256) or 256), 1)
        duration_seconds = max(float(getattr(processor, "duration_seconds", 1.0) or 1.0), 0.05)
        n_frames = max(int(sample_rate * duration_seconds / hop_length), 1)
        seed_text = str(sample_input or config.get("model_id") or "slai-vocos")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_seed_from_text(seed_text))
        mel_tensor = torch.randn((1, n_mels, n_frames), generator=generator, dtype=torch.float32)

    if mel_tensor.ndim == 2:
        mel_tensor = mel_tensor.unsqueeze(0)
    if mel_tensor.ndim != 3:
        raise RuntimeError(f"Vocos mel spectrogram 需要 [B, n_mels, frames] 或 [n_mels, frames]，当前 shape={tuple(mel_tensor.shape)}")
    if mel_tensor.shape[1] != int(getattr(processor, "n_mels", mel_tensor.shape[1]) or mel_tensor.shape[1]) and mel_tensor.shape[2] == int(getattr(processor, "n_mels", mel_tensor.shape[2]) or mel_tensor.shape[2]):
        mel_tensor = mel_tensor.transpose(1, 2)

    model_device = _get_model_device(model)
    param = _get_first_parameter(model)
    target_dtype = param.dtype if param is not None and torch.is_floating_point(param) else torch.float32
    return mel_tensor.to(device=model_device, dtype=target_dtype)


def _run_tts(model, processor, samples: list[dict], config: dict, scenario: str = "") -> tuple[list[str], dict]:
    predictions: list[str] = []
    latencies: list[float] = []
    qwen_tts_mode = str(getattr(processor, "generation_mode", "") or "").strip().lower()
    if getattr(processor, "backend", "") == "f5_tts":
        api = getattr(processor, "api", None)
        run_single_infer = getattr(processor, "inference_fn", None)
        if api is None or not callable(run_single_infer):
            raise RuntimeError("F5-TTS runtime 未正确初始化")
        perf_prepare_reference = getattr(processor, "perf_prepare_reference", None)
        perf_build_sample_plan = getattr(processor, "perf_build_sample_plan", None)
        perf_run_single_infer = getattr(processor, "perf_inference_fn", None)
        references = [sample.get("reference") for sample in samples]
        default_prompts = tuple(getattr(processor, "default_prompts", ()) or F5_TTS_DEFAULT_PROMPTS)
        reference_audio_name = str(getattr(processor, "reference_audio_name", "") or "").strip()
        reference_text = str(getattr(processor, "reference_text", "") or "").strip()
        nfe_step = int(getattr(processor, "nfe_step", 4) or 4)
        output_type_hint = str(config.get("output_type_hint") or "").strip().lower()
        audio_durations: list[float] = []
        sample_rates: list[int] = []
        spectrogram_shapes: list[list[int]] = []
        prepared_ref = None
        use_perf_fast_path = bool(
            str(config.get("optimization_kind") or "").strip().lower() == "runtime_only"
            and str(scenario or config.get("scenario") or "").strip().lower() == "npu_perf"
        )
        if not use_perf_fast_path and str(config.get("optimization_kind") or "").strip().lower() == "runtime_only":
            use_perf_fast_path = str(os.environ.get("BUSINESS_BENCHMARK_SCENARIO") or "").strip().lower() == "npu_perf"
        if use_perf_fast_path:
            if not callable(perf_prepare_reference) or not callable(perf_build_sample_plan) or not callable(perf_run_single_infer):
                raise RuntimeError("F5-TTS runtime-only perf 场景缺少 accuracy_run_perf.py fast path")
            prepared_ref = perf_prepare_reference(api, str(_get_model_device(model)))

        for sample_index, sample in enumerate(samples):
            prompt = str(sample.get("input") or "").strip()
            if not prompt or prompt.lower().startswith("latency-only business benchmark"):
                prompt = default_prompts[sample_index % len(default_prompts)]
            import time

            with torch.no_grad():
                sample_start = time.perf_counter()
                try:
                    if use_perf_fast_path:
                        plan = perf_build_sample_plan(prepared_ref, prompt)
                        meta, _spec = perf_run_single_infer(api, prepared_ref, plan, seed=20260421 + sample_index)
                    else:
                        meta, _spec = run_single_infer(api, prompt, seed=20260421 + sample_index)
                except RuntimeError as exc:
                    exc_text = str(exc)
                    if "torch.istft" not in exc_text and "istft(" not in exc_text:
                        raise
                    if not callable(perf_prepare_reference) or not callable(perf_build_sample_plan) or not callable(perf_run_single_infer):
                        raise
                    if prepared_ref is None:
                        prepared_ref = perf_prepare_reference(api, str(_get_model_device(model)))
                    plan = perf_build_sample_plan(prepared_ref, prompt)
                    meta, _spec = perf_run_single_infer(api, prepared_ref, plan, seed=20260421 + sample_index)
                    use_perf_fast_path = True
                latencies.append(time.perf_counter() - sample_start)
            frame_count = int(meta.get("audio_num_samples") or 0)
            sample_rate_value = int(meta.get("sample_rate") or 0)
            spectrogram_shape = [int(item) for item in list(meta.get("spectrogram_shape") or [])]
            if sample_rate_value > 0:
                sample_rates.append(sample_rate_value)
            if frame_count > 0 and sample_rate_value > 0:
                audio_durations.append(float(frame_count / sample_rate_value))
            if spectrogram_shape:
                spectrogram_shapes.append(spectrogram_shape)
            if output_type_hint == "mel_spectrograms" and spectrogram_shape:
                predictions.append(f"mel_spectrogram_shape={spectrogram_shape}")
            else:
                predictions.append(f"audio_frames={frame_count}")

        metric_context: dict[str, Any] = {
            "inference_strategy": "f5_tts_runtime_only_fast_path" if use_perf_fast_path else "f5_tts_api_infer",
            "references": references,
            "tts_reference_audio": reference_audio_name,
            "tts_reference_text": reference_text,
            "tts_nfe_step": nfe_step,
        }
        if latencies:
            metric_context["inference_latency_s"] = sum(latencies) / len(latencies)
        if audio_durations:
            metric_context["tts_audio_duration_s_mean"] = sum(audio_durations) / len(audio_durations)
        if sample_rates:
            metric_context["tts_sample_rate"] = sample_rates[0]
        if spectrogram_shapes:
            metric_context["mel_spectrogram_shape_example"] = spectrogram_shapes[0]
        return predictions, metric_context

    if getattr(processor, "backend", "") == "vocos" and hasattr(model, "decode"):
        audio_durations: list[float] = []
        sample_rates: list[int] = []
        sample_rate = max(int(getattr(processor, "sample_rate", 24000) or 24000), 1)
        references = [sample.get("reference") for sample in samples]
        for sample in samples:
            mel_spec = _coerce_vocos_mel_input(sample.get("input"), processor, model, config)
            import time

            with torch.no_grad():
                sample_start = time.perf_counter()
                audio = model.decode(mel_spec)
                latencies.append(time.perf_counter() - sample_start)
            frame_count = int(audio.shape[-1]) if hasattr(audio, "shape") and len(audio.shape) >= 1 else 0
            if frame_count > 0:
                audio_durations.append(float(frame_count / sample_rate))
            sample_rates.append(sample_rate)
            predictions.append(f"audio_frames={frame_count}")

        metric_context: dict[str, Any] = {
            "inference_strategy": "vocos_decode",
            "references": references,
            "tts_sample_rate": sample_rate,
            "vocos_n_mels": int(getattr(processor, "n_mels", 100) or 100),
            "vocos_hop_length": int(getattr(processor, "hop_length", 256) or 256),
            "vocos_duration_seconds": float(getattr(processor, "duration_seconds", 1.0) or 1.0),
        }
        if latencies:
            metric_context["inference_latency_s"] = sum(latencies) / len(latencies)
        if audio_durations:
            metric_context["tts_audio_duration_s_mean"] = sum(audio_durations) / len(audio_durations)
        if sample_rates:
            metric_context["tts_sample_rate"] = sample_rates[0]
        return predictions, metric_context

    if getattr(processor, "backend", "") == "qwen_tts" and qwen_tts_mode == "custom_voice" and hasattr(model, "generate_custom_voice"):
        audio_durations: list[float] = []
        sample_rates: list[int] = []
        speaker = str(config.get("tts_speaker") or "Aiden").strip() or "Aiden"
        language = _normalize_qwen_tts_language(config)
        instruction = str(config.get("tts_instruction") or "Speak in a normal tone").strip() or "Speak in a normal tone"

        for sample in samples:
            prompt = str(sample.get("input") or "").strip() or "Latency-only text-to-speech business benchmark."
            import time

            with torch.no_grad():
                sample_start = time.perf_counter()
                wavs, sample_rate = model.generate_custom_voice(
                    text=prompt,
                    speaker=speaker,
                    language=language,
                    instruct=instruction,
                    non_streaming_mode=True,
                )
                latencies.append(time.perf_counter() - sample_start)
            audio = wavs[0] if isinstance(wavs, (list, tuple)) and wavs else wavs
            frame_count = 0
            if audio is not None:
                try:
                    frame_count = int(len(audio))
                except Exception:
                    frame_count = 0
            sample_rate_value = int(sample_rate) if isinstance(sample_rate, (int, float)) and sample_rate else 0
            if sample_rate_value > 0:
                sample_rates.append(sample_rate_value)
            duration_s = float(frame_count / sample_rate_value) if frame_count > 0 and sample_rate_value > 0 else 0.0
            if duration_s > 0:
                audio_durations.append(duration_s)
            predictions.append(f"audio_frames={frame_count}")

        metric_context: dict[str, Any] = {
            "inference_strategy": "qwen_tts_generate_custom_voice",
            "references": [sample.get("reference") for sample in samples],
            "tts_speaker": speaker,
            "tts_language": language,
            "tts_instruction": instruction,
            "tts_non_streaming_mode": True,
        }
        if latencies:
            metric_context["inference_latency_s"] = sum(latencies) / len(latencies)
        if audio_durations:
            metric_context["tts_audio_duration_s_mean"] = sum(audio_durations) / len(audio_durations)
        if sample_rates:
            metric_context["tts_sample_rate"] = sample_rates[0]
        return predictions, metric_context

    if getattr(processor, "backend", "") == "qwen_tts" and qwen_tts_mode == "voice_design" and hasattr(model, "generate_voice_design"):
        audio_durations: list[float] = []
        sample_rates: list[int] = []
        language = _normalize_qwen_tts_language(config)
        instruction = str(config.get("tts_instruction") or "A calm and clear female voice with a professional tone.").strip() or "A calm and clear female voice with a professional tone."

        for sample in samples:
            prompt = str(sample.get("input") or "").strip() or "Latency-only text-to-speech business benchmark."
            import time

            with torch.no_grad():
                sample_start = time.perf_counter()
                wavs, sample_rate = model.generate_voice_design(
                    text=prompt,
                    instruct=instruction,
                    language=language,
                    non_streaming_mode=True,
                )
                latencies.append(time.perf_counter() - sample_start)
            audio = wavs[0] if isinstance(wavs, (list, tuple)) and wavs else wavs
            frame_count = 0
            if audio is not None:
                try:
                    frame_count = int(len(audio))
                except Exception:
                    frame_count = 0
            sample_rate_value = int(sample_rate) if isinstance(sample_rate, (int, float)) and sample_rate else 0
            if sample_rate_value > 0:
                sample_rates.append(sample_rate_value)
            duration_s = float(frame_count / sample_rate_value) if frame_count > 0 and sample_rate_value > 0 else 0.0
            if duration_s > 0:
                audio_durations.append(duration_s)
            predictions.append(f"audio_frames={frame_count}")

        metric_context = {
            "inference_strategy": "qwen_tts_generate_voice_design",
            "references": [sample.get("reference") for sample in samples],
            "tts_language": language,
            "tts_instruction": instruction,
            "tts_non_streaming_mode": True,
        }
        if latencies:
            metric_context["inference_latency_s"] = sum(latencies) / len(latencies)
        if audio_durations:
            metric_context["tts_audio_duration_s_mean"] = sum(audio_durations) / len(audio_durations)
        if sample_rates:
            metric_context["tts_sample_rate"] = sample_rates[0]
        return predictions, metric_context

    runtime_adapter = getattr(processor, "adapter", processor)
    model_device = _get_model_device(model)
    decode_adapter = runtime_adapter if hasattr(runtime_adapter, "decode") or hasattr(runtime_adapter, "batch_decode") else getattr(runtime_adapter, "tokenizer", None)
    max_new_tokens = int(config.get("tts_max_new_tokens") or config.get("max_new_tokens") or 20)

    for sample in samples:
        prompt = str(sample.get("input") or "").strip() or "Latency-only text-to-speech business benchmark."
        import time

        if hasattr(runtime_adapter, "encode"):
            input_ids = runtime_adapter.encode(prompt, return_tensors="pt")
        else:
            processed = runtime_adapter(prompt, return_tensors="pt")
            input_ids = processed["input_ids"] if isinstance(processed, dict) else getattr(processed, "input_ids")
        input_ids = input_ids.to(model_device)

        with torch.no_grad():
            sample_start = time.perf_counter()
            try:
                outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
            except Exception:
                logits = model(input_ids).logits
                outputs = torch.argmax(logits, dim=-1)
            latencies.append(time.perf_counter() - sample_start)

        if decode_adapter is not None and hasattr(decode_adapter, "decode"):
            decoded = decode_adapter.decode(outputs[0], skip_special_tokens=True)
        elif decode_adapter is not None and hasattr(decode_adapter, "batch_decode"):
            decoded = decode_adapter.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            decoded = f"generated_tokens={int(outputs.shape[-1])}" if hasattr(outputs, "shape") else str(outputs)
        predictions.append(str(decoded))

    metric_context = {
        "inference_strategy": "tts_causallm_generate",
        "references": [sample.get("reference") for sample in samples],
        "tts_max_new_tokens": max_new_tokens,
    }
    if latencies:
        metric_context["inference_latency_s"] = sum(latencies) / len(latencies)
    return predictions, metric_context


def _run_asr(model, processor, samples: list[dict], config: dict, scenario: str = "") -> tuple[list[str], dict]:
    predictions: list[str] = []
    if getattr(processor, "backend", "") == "qwen_forced_aligner" and hasattr(model, "align"):
        language = _normalize_qwen_asr_language(config) or "English"
        alignment_span_counts: list[int] = []
        alignment_coverages: list[float] = []
        alignment_end_times: list[float] = []
        alignment_previews: list[list[dict[str, Any]]] = []
        for sample in samples:
            audio_input = sample.get("input")
            reference_text = str(sample.get("reference") or "").strip()
            if audio_input is None or not reference_text:
                predictions.append("")
                alignment_span_counts.append(0)
                alignment_coverages.append(0.0)
                continue
            sample_rate = _sample_sampling_rate(sample)
            with torch.no_grad():
                results = model.align(audio=(audio_input, sample_rate), text=reference_text, language=language)
            if isinstance(results, (list, tuple)) and results:
                result = results[0]
            else:
                result = results
            alignment_items = _extract_qwen_forced_alignment_items(result)
            aligned_tokens = [str(item.get("text") or "").strip() for item in alignment_items if str(item.get("text") or "").strip()]
            predictions.append(_join_qwen_forced_alignment_tokens(aligned_tokens) or reference_text)
            alignment_span_counts.append(len(alignment_items))
            alignment_coverages.append(_estimate_alignment_token_coverage(reference_text, aligned_tokens))
            if alignment_items:
                alignment_end_times.append(max(float(item.get("end_time") or 0.0) for item in alignment_items))
            if len(alignment_previews) < 3:
                alignment_previews.append(alignment_items[:8])
        metric_context: dict[str, Any] = {
            "inference_strategy": "qwen_forced_aligner_align",
            "attention_mask_used": False,
            "asr_batch_size_requested": _get_asr_batch_size(config, scenario),
            "asr_batch_size_effective": 1,
            "generate_kwargs_used": False,
            "forced_alignment_language": language,
        }
        if alignment_span_counts:
            metric_context["alignment_span_count_mean"] = sum(alignment_span_counts) / len(alignment_span_counts)
        if alignment_coverages:
            metric_context["alignment_token_coverage"] = sum(alignment_coverages) / len(alignment_coverages)
        if alignment_end_times:
            metric_context["alignment_duration_s_mean"] = sum(alignment_end_times) / len(alignment_end_times)
        if alignment_previews:
            metric_context["alignment_preview"] = alignment_previews
        return predictions, metric_context

    if getattr(processor, "backend", "") == "qwen_asr" and hasattr(model, "transcribe"):
        language = _normalize_qwen_asr_language(config)
        for sample in samples:
            audio_input = sample.get("input")
            if audio_input is None:
                predictions.append("")
                continue
            sample_rate = _sample_sampling_rate(sample)
            with torch.no_grad():
                results = model.transcribe(audio=(audio_input, sample_rate), language=language)
            if isinstance(results, (list, tuple)) and results:
                predictions.append(_extract_qwen_asr_text(results[0]))
            else:
                predictions.append(_extract_qwen_asr_text(results))
        return predictions, {
            "inference_strategy": "qwen_asr_transcribe",
            "attention_mask_used": False,
            "asr_batch_size_requested": _get_asr_batch_size(config, scenario),
            "asr_batch_size_effective": 1,
            "generate_kwargs_used": False,
        }

    if getattr(processor, "backend", "") == "nemo_asr" and hasattr(model, "transcribe"):
        requested_batch_size = max(_get_asr_batch_size(config, scenario), 1)
        effective_batch_sizes: list[int] = []
        for start in range(0, len(samples), requested_batch_size):
            chunk = samples[start : start + requested_batch_size]
            chunk_predictions = [""] * len(chunk)
            valid_pairs = [(idx, sample) for idx, sample in enumerate(chunk) if sample.get("input") is not None]
            if not valid_pairs:
                predictions.extend(chunk_predictions)
                continue
            sample_rates = {_sample_sampling_rate(sample) for _, sample in valid_pairs}
            eval_groups = [valid_pairs] if len(sample_rates) == 1 else [[pair] for pair in valid_pairs]
            for group in eval_groups:
                audio_inputs = [sample.get("input") for _, sample in group]
                with torch.no_grad():
                    results = model.transcribe(audio_inputs, batch_size=len(audio_inputs))
                if not isinstance(results, (list, tuple)):
                    results = [results]
                normalized_results = list(results)
                if len(normalized_results) < len(group):
                    normalized_results.extend([""] * (len(group) - len(normalized_results)))
                for (idx, _sample), result in zip(group, normalized_results):
                    chunk_predictions[idx] = _extract_nemo_asr_text(result)
                effective_batch_sizes.append(len(group))
            predictions.extend(chunk_predictions)
        return predictions, {
            "inference_strategy": "nemo_asr_transcribe",
            "attention_mask_used": False,
            "asr_batch_size_requested": requested_batch_size,
            "asr_batch_size_effective": max(effective_batch_sizes) if effective_batch_sizes else 0,
            "generate_kwargs_used": False,
        }

    model_device = _get_model_device(model)
    model_dtype = _get_model_floating_dtype(model)
    asr_batch_size = _get_asr_batch_size(config, scenario)
    inference_strategy = "speech_generate" if hasattr(model, "generate") else "ctc_argmax_decode"
    generate_kwargs, generate_metadata = _build_asr_generate_kwargs(config)
    effective_generate_kwargs = dict(generate_kwargs)
    if _should_disable_whisper_suppress_tokens(model, config, effective_generate_kwargs):
        effective_generate_kwargs["suppress_tokens"] = []
        effective_generate_kwargs["begin_suppress_tokens"] = []
        generate_metadata["whisper_suppress_tokens_disabled"] = True
    else:
        generate_metadata["whisper_suppress_tokens_disabled"] = False
    attention_mask_used = False
    latency_scope = str(getattr(config, "latency_measurement_scope", None) or os.environ.get("BUSINESS_BENCHMARK_LATENCY_MEASUREMENT_SCOPE") or "").strip().lower()

    def _prepare_chunk_inputs(chunk_samples: list[dict]) -> list[dict[str, Any]]:
        nonlocal attention_mask_used
        audios = [sample.get("input") for sample in chunk_samples]
        if any(audio is None for audio in audios):
            return []
        sample_rates = {_sample_sampling_rate(sample) for sample in chunk_samples}
        if len(sample_rates) != 1:
            prepared_chunks: list[dict[str, Any]] = []
            for single_sample in chunk_samples:
                prepared_chunks.extend(_prepare_chunk_inputs([single_sample]))
            return prepared_chunks
        processed = _prepare_asr_processed_inputs(
            processor,
            audios,
            sampling_rate=next(iter(sample_rates)),
            model_device=model_device,
            model_dtype=model_dtype,
        )
        attention_mask_used = attention_mask_used or ("attention_mask" in processed)
        return [processed]

    def _infer_from_processed(processed: dict[str, Any]) -> None:
        with torch.no_grad():
            if hasattr(model, "generate"):
                try:
                    generated_ids = model.generate(**processed, **effective_generate_kwargs)
                except TypeError as exc:
                    if not any(key in str(exc) for key in ("language", "task")):
                        raise
                    fallback_generate_kwargs = {key: value for key, value in effective_generate_kwargs.items() if key not in {"language", "task"}}
                    generated_ids = model.generate(**processed, **fallback_generate_kwargs)
                predictions.extend(_decode_asr_batch(processor, generated_ids))
            else:
                outputs = model(**processed)
                logits = getattr(outputs, "logits", None)
                if logits is None and isinstance(outputs, (tuple, list)) and outputs:
                    logits = outputs[0]
                if logits is None:
                    raise RuntimeError("ASR 模型输出缺少 logits，无法进行 CTC 解码")
                predicted_ids = torch.argmax(logits, dim=-1)
                predictions.extend(_decode_asr_batch(processor, predicted_ids))

    inference_start_ts = None
    if latency_scope == "steady_state":
        prepared_chunks: list[dict[str, Any]] = []
        for start in range(0, len(samples), asr_batch_size):
            prepared_chunks.extend(_prepare_chunk_inputs(samples[start : start + asr_batch_size]))
        if prepared_chunks:
            import time

            _sync_device(model_device)
            inference_start_ts = time.perf_counter()
        for processed in prepared_chunks:
            _infer_from_processed(processed)
    else:
        for start in range(0, len(samples), asr_batch_size):
            for processed in _prepare_chunk_inputs(samples[start : start + asr_batch_size]):
                _infer_from_processed(processed)

    metric_context = {
        "inference_strategy": inference_strategy,
        "attention_mask_used": attention_mask_used,
        "model_input_dtype_aligned": _format_model_dtype(model),
        "asr_batch_size": asr_batch_size,
        **generate_metadata,
    }
    if inference_start_ts is not None and predictions:
        import time

        _sync_device(model_device)
        metric_context["inference_latency_s"] = (time.perf_counter() - inference_start_ts) / len(predictions)
    return predictions, metric_context


def _run_vision_text_ocr(model, processor, samples: list[dict], config: dict[str, Any]) -> tuple[list[str], dict]:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        model_id = str(config.get("model_id") or "").strip()
        tokenizer_source = str(config.get("tokenizer_source_override") or config.get("base_model_id") or model_id or "microsoft/trocr-base-handwritten").strip()
        try:
            from transformers import AutoTokenizer

            resolved_tokenizer_source = _resolve_local_snapshot_source(tokenizer_source, input_kind="tokenizer") or tokenizer_source
            tokenizer = AutoTokenizer.from_pretrained(resolved_tokenizer_source, cache_dir=str(CACHE_DIR), trust_remote_code=bool(config.get("trust_remote_code")))
        except Exception as exc:
            raise RuntimeError(f"vision_text_ocr 缺少 tokenizer，且自动加载失败: {exc}") from exc

    model_device = _get_model_device(model)
    latency_scope = str(getattr(config, "latency_measurement_scope", None) or os.environ.get("BUSINESS_BENCHMARK_LATENCY_MEASUREMENT_SCOPE") or "").strip().lower()
    max_new_tokens = int(config.get("ocr_max_new_tokens") or 32)
    num_beams = int(config.get("ocr_num_beams") or 4)
    model_id_text = str(config.get("model_id") or "").strip().lower()
    default_task_prompt = str(config.get("ocr_task_prompt") or "").strip()
    if not default_task_prompt and "florence" in model_id_text:
        default_task_prompt = "<OCR>"
    predictions: list[str] = []
    inference_start_ts = None

    prepared_inputs: list[dict[str, torch.Tensor]] = []
    for sample in samples:
        image = sample.get("input")
        if image is None:
            continue
        task_prompt = str(sample.get("ocr_prompt") or default_task_prompt).strip()
        processor_kwargs: dict[str, Any] = {"images": image, "return_tensors": "pt"}
        if task_prompt:
            processor_kwargs["text"] = task_prompt
        processed = processor(**processor_kwargs)
        if isinstance(processed, Mapping):
            processed_inputs = {key: value for key, value in processed.items() if torch.is_tensor(value)}
        elif torch.is_tensor(processed):
            processed_inputs = {"pixel_values": processed}
        else:
            processed_inputs = {}
        if not processed_inputs:
            continue
        pixel_values = processed_inputs.get("pixel_values")
        if isinstance(pixel_values, torch.Tensor) and pixel_values.ndim == 3:
            processed_inputs["pixel_values"] = pixel_values.unsqueeze(0)
        prepared_inputs.append(processed_inputs)

    if latency_scope == "steady_state" and prepared_inputs:
        import time

        _sync_device(model_device)
        inference_start_ts = time.perf_counter()

    for processed_inputs in prepared_inputs:
        runtime_inputs = {
            key: _move_runtime_tensor(value, model_device, dtype=_get_model_floating_dtype(model))
            for key, value in processed_inputs.items()
        }
        with torch.no_grad():
            generated_ids = model.generate(
                **runtime_inputs,
                max_new_tokens=max_new_tokens,
                num_beams=max(num_beams, 1),
                use_cache=False,
            )
        if hasattr(processor, "batch_decode"):
            decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
            prediction = str(decoded[0] if decoded else "").strip()
        elif hasattr(tokenizer, "batch_decode"):
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            prediction = str(decoded[0] if decoded else "").strip()
        else:
            prediction = str(generated_ids[0].tolist()).strip()
        predictions.append(prediction)

    metric_context = {
        "inference_strategy": "vision_text_generate",
        "ocr_max_new_tokens": max_new_tokens,
        "ocr_num_beams": num_beams,
        "ocr_task_prompt": default_task_prompt,
    }
    if inference_start_ts is not None and predictions:
        import time

        _sync_device(model_device)
        metric_context["inference_latency_s"] = (time.perf_counter() - inference_start_ts) / len(predictions)
    return predictions, metric_context


def _resolve_vision_label_space(samples: list[dict]) -> tuple[list[str], list[str]]:
    for sample in samples:
        dataset_label_names = sample.get("dataset_label_names")
        if isinstance(dataset_label_names, list) and dataset_label_names:
            label_names = [str(name) for name in dataset_label_names if str(name).strip()]
            if label_names:
                return [str(idx) for idx in range(len(label_names))], label_names
    label_space = sorted({str(sample.get("reference") or "").strip() for sample in samples if str(sample.get("reference") or "").strip()})
    return label_space, label_space


def _run_vision_classification(model, processor, config, samples: list[dict]) -> tuple[list[str], dict]:
    if hasattr(model, "encode_image") and hasattr(model, "encode_text") and hasattr(processor, "preprocess") and hasattr(processor, "tokenizer"):
        latency_scope = str(getattr(config, "latency_measurement_scope", None) or os.environ.get("BUSINESS_BENCHMARK_LATENCY_MEASUREMENT_SCOPE") or "").strip().lower()
        model_device = next(model.parameters()).device
        _enable_cuda_vision_runtime_optimizations(model, model_device)
        label_space, label_names = _resolve_vision_label_space(samples)
        if not label_names:
            return [], {"prediction_scores": [], "label_space": []}

        label_texts = processor.build_label_texts(label_names)
        text_tokens = processor.tokenizer(label_texts)
        if isinstance(text_tokens, dict):
            text_tokens = {key: value.to(model_device) for key, value in text_tokens.items()}
        else:
            text_tokens = text_tokens.to(model_device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, p=2, dim=-1)

        prepared_inputs: list[torch.Tensor] = []
        valid_samples = []
        for sample in samples:
            image = sample.get("input")
            if image is None:
                continue
            image_inputs = processor.preprocess(image)
            if isinstance(image_inputs, Mapping):
                pixel_values = image_inputs.get("pixel_values")
                if pixel_values is None:
                    continue
            else:
                pixel_values = image_inputs
            if pixel_values is None:
                continue
            if not isinstance(image_inputs, Mapping) and isinstance(pixel_values, torch.Tensor) and pixel_values.ndim == 3:
                pixel_values = pixel_values.unsqueeze(0)
                image_inputs = pixel_values
            prepared_inputs.append(image_inputs)
            valid_samples.append(sample)

        predictions: list[str] = []
        prediction_scores: list[list[float]] = []
        deferred_logits: list[torch.Tensor] = []
        inference_start_ts = None
        if latency_scope == "steady_state" and prepared_inputs:
            import time

            _sync_device(model_device)
            inference_start_ts = time.perf_counter()
        for sample, image_inputs in zip(valid_samples, prepared_inputs):
            if isinstance(image_inputs, Mapping):
                image_inputs = {
                    key: _prepare_cuda_vision_tensor(value.to(model_device), model_device) if isinstance(value, torch.Tensor) else value
                    for key, value in image_inputs.items()
                }
            else:
                image_inputs = _prepare_cuda_vision_tensor(image_inputs.to(model_device), model_device)
            with torch.inference_mode():
                image_features = model.encode_image(image_inputs)
            image_features = F.normalize(image_features, p=2, dim=-1)
            logits = image_features @ text_features.T
            logit_scale = getattr(model, "logit_scale", None)
            if logit_scale is not None:
                scale_tensor = torch.as_tensor(logit_scale, device=logits.device, dtype=logits.dtype)
                if scale_tensor.numel() == 1:
                    logits = logits * scale_tensor.exp()
            logit_bias = getattr(model, "logit_bias", None)
            if logit_bias is not None:
                bias_tensor = torch.as_tensor(logit_bias, device=logits.device, dtype=logits.dtype)
                if bias_tensor.numel() == 1:
                    logits = logits + bias_tensor.reshape(1, 1)
                elif bias_tensor.numel() == logits.shape[-1]:
                    logits = logits + bias_tensor.reshape(1, -1)
            deferred_logits.append(logits[0].detach())
        measured_wall_clock_s = 0.0
        if inference_start_ts is not None and deferred_logits:
            import time

            _sync_device(model_device)
            measured_wall_clock_s = time.perf_counter() - inference_start_ts
        for sample, logits in zip(valid_samples, deferred_logits):
            score_tensor = logits.float().cpu()
            score_values = [float(value) for value in score_tensor.tolist()]
            pred_idx = max(range(len(score_values)), key=lambda idx: score_values[idx])
            predictions.append(label_space[pred_idx])
            prediction_scores.append(score_values)
        metric_context = {"prediction_scores": prediction_scores, "label_space": label_space, "label_names": label_names}
        if latency_scope == "steady_state" and predictions and measured_wall_clock_s > 0:
            metric_context["inference_latency_s"] = measured_wall_clock_s / len(predictions)
            metric_context["wall_clock_s"] = measured_wall_clock_s
        return predictions, metric_context

    if str(getattr(processor, "backend", "") or "").strip().lower() == "timm_image_classification":
        predictions: list[str] = []
        prediction_scores: list[list[float]] = []
        label_space, label_names = _resolve_vision_label_space(samples)
        model_device = _get_model_device(model)
        _enable_cuda_vision_runtime_optimizations(model, model_device)
        latency_scope = str(getattr(config, "latency_measurement_scope", None) or os.environ.get("BUSINESS_BENCHMARK_LATENCY_MEASUREMENT_SCOPE") or "").strip().lower()
        prepared_inputs: list[torch.Tensor] = []
        for sample in samples:
            image = sample.get("input")
            if image is None:
                continue
            pixel_values = processor.transform(image)
            if isinstance(pixel_values, torch.Tensor) and pixel_values.ndim == 3:
                pixel_values = pixel_values.unsqueeze(0)
            prepared_inputs.append(_prepare_cuda_vision_tensor(pixel_values, model_device))

        deferred_logits: list[torch.Tensor] = []
        inference_start_ts = None
        if latency_scope == "steady_state" and prepared_inputs:
            import time

            _sync_device(model_device)
            inference_start_ts = time.perf_counter()
        for pixel_values in prepared_inputs:
            pixel_values = _move_runtime_tensor(pixel_values, model_device, dtype=_get_model_floating_dtype(model))
            pixel_values = _prepare_cuda_vision_tensor(pixel_values, model_device)
            with torch.inference_mode():
                logits = model(pixel_values)
            if hasattr(logits, "logits"):
                logits = logits.logits
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            deferred_logits.append(logits[0].detach())
        measured_wall_clock_s = 0.0
        if inference_start_ts is not None and deferred_logits:
            import time

            _sync_device(model_device)
            measured_wall_clock_s = time.perf_counter() - inference_start_ts
        for logits in deferred_logits:
            score_tensor = logits.float().cpu()
            score_values = [float(value) for value in score_tensor.tolist()]
            pred_id = int(torch.argmax(score_tensor, dim=-1).item())
            predictions.append(str(pred_id))
            prediction_scores.append(score_values)
        if prediction_scores and len(label_space) != len(prediction_scores[0]):
            label_space = [str(idx) for idx in range(len(prediction_scores[0]))]
            if len(label_names) != len(label_space):
                label_names = list(label_space)
        metric_context = {"prediction_scores": prediction_scores, "label_space": label_space, "label_names": label_names}
        if latency_scope == "steady_state" and predictions and measured_wall_clock_s > 0:
            metric_context["inference_latency_s"] = measured_wall_clock_s / len(predictions)
            metric_context["wall_clock_s"] = measured_wall_clock_s
        return predictions, metric_context

    if str(getattr(processor, "backend", "") or "").strip().lower() == "local_vision_classification_bridge":
        predictions: list[str] = []
        prediction_scores: list[list[float]] = []
        label_space, label_names = _resolve_vision_label_space(samples)
        local_class_labels = [str(label) for label in list(getattr(processor, "class_labels", []) or []) if str(label).strip()]
        model_device = _get_model_device(model)
        _enable_cuda_vision_runtime_optimizations(model, model_device)
        latency_scope = str(getattr(config, "latency_measurement_scope", None) or os.environ.get("BUSINESS_BENCHMARK_LATENCY_MEASUREMENT_SCOPE") or "").strip().lower()
        prepared_inputs: list[torch.Tensor] = []
        for sample in samples:
            image = sample.get("input")
            if image is None:
                continue
            pixel_values = processor.transform(image)
            if isinstance(pixel_values, torch.Tensor) and pixel_values.ndim == 3:
                pixel_values = pixel_values.unsqueeze(0)
            prepared_inputs.append(_prepare_cuda_vision_tensor(pixel_values, model_device))

        deferred_logits: list[torch.Tensor] = []
        inference_start_ts = None
        if latency_scope == "steady_state" and prepared_inputs:
            import time

            _sync_device(model_device)
            inference_start_ts = time.perf_counter()
        for pixel_values in prepared_inputs:
            pixel_values = _move_runtime_tensor(pixel_values, model_device, dtype=_get_model_floating_dtype(model))
            pixel_values = _prepare_cuda_vision_tensor(pixel_values, model_device)
            with torch.inference_mode():
                logits = model(pixel_values)
            if hasattr(logits, "logits"):
                logits = logits.logits
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            deferred_logits.append(logits[0].detach())
        measured_wall_clock_s = 0.0
        if inference_start_ts is not None and deferred_logits:
            import time

            _sync_device(model_device)
            measured_wall_clock_s = time.perf_counter() - inference_start_ts
        for logits in deferred_logits:
            score_tensor = logits.float().cpu()
            score_values = [float(value) for value in score_tensor.tolist()]
            pred_id = int(torch.argmax(score_tensor, dim=-1).item())
            predictions.append(str(pred_id))
            prediction_scores.append(score_values)
        if prediction_scores and len(label_space) != len(prediction_scores[0]):
            label_space = [str(idx) for idx in range(len(prediction_scores[0]))]
            label_names = local_class_labels if len(local_class_labels) == len(label_space) else list(label_space)
        metric_context = {"prediction_scores": prediction_scores, "label_space": label_space, "label_names": label_names}
        if latency_scope == "steady_state" and predictions and measured_wall_clock_s > 0:
            metric_context["inference_latency_s"] = measured_wall_clock_s / len(predictions)
            metric_context["wall_clock_s"] = measured_wall_clock_s
        return predictions, metric_context

    id2label_raw = getattr(config, "id2label", None) or {}
    id2label = {}
    if isinstance(id2label_raw, dict):
        for key, value in id2label_raw.items():
            try:
                id2label[int(key)] = str(value)
            except Exception:
                continue
    num_labels = int(getattr(config, "num_labels", 0) or 0)
    sorted_label_ids = sorted(id2label) if id2label else list(range(num_labels))
    label_space = [str(idx) for idx in sorted_label_ids]
    label_names = [id2label.get(idx, str(idx)) for idx in sorted_label_ids]
    predictions: list[str] = []
    prediction_scores: list[list[float]] = []
    model_device = _get_model_device(model)
    _enable_cuda_vision_runtime_optimizations(model, model_device)
    model_dtype = _get_model_floating_dtype(model)
    latency_scope = str(getattr(config, "latency_measurement_scope", None) or os.environ.get("BUSINESS_BENCHMARK_LATENCY_MEASUREMENT_SCOPE") or "").strip().lower()
    prepared_inputs: list[dict[str, Any]] = []
    for sample in samples:
        image = sample.get("input")
        if image is None:
            continue
        processed = processor(images=image, return_tensors="pt")
        runtime_inputs = {k: _move_runtime_tensor(v, model_device, dtype=model_dtype) for k, v in dict(processed).items()}
        pixel_values = runtime_inputs.get("pixel_values")
        if isinstance(pixel_values, torch.Tensor):
            runtime_inputs["pixel_values"] = _prepare_cuda_vision_tensor(pixel_values, model_device)
        prepared_inputs.append(runtime_inputs)

    deferred_logits: list[torch.Tensor] = []
    inference_start_ts = None
    if latency_scope == "steady_state" and prepared_inputs:
        import time

        _sync_device(model_device)
        inference_start_ts = time.perf_counter()
    for processed in prepared_inputs:
        with torch.inference_mode():
            logits = model(**processed).logits[0]
        deferred_logits.append(logits.detach())
    measured_wall_clock_s = 0.0
    if inference_start_ts is not None and deferred_logits:
        import time

        _sync_device(model_device)
        measured_wall_clock_s = time.perf_counter() - inference_start_ts
    for logits in deferred_logits:
        score_tensor = logits.float().cpu()
        score_values = [float(value) for value in score_tensor.tolist()]
        pred_id = int(torch.argmax(score_tensor, dim=-1).item())
        predictions.append(str(pred_id))
        prediction_scores.append(score_values)
    if prediction_scores and len(label_space) != len(prediction_scores[0]):
        label_space = [str(idx) for idx in range(len(prediction_scores[0]))]
        label_names = [id2label.get(idx, str(idx)) for idx in range(len(prediction_scores[0]))]
    metric_context = {"prediction_scores": prediction_scores, "label_space": label_space, "label_names": label_names}
    if latency_scope == "steady_state" and predictions and measured_wall_clock_s > 0:
        metric_context["inference_latency_s"] = measured_wall_clock_s / len(predictions)
        metric_context["wall_clock_s"] = measured_wall_clock_s
    return predictions, metric_context


def _normalize_vision_embedding_value(value):
    if isinstance(value, list):
        if not value:
            return None
        normalized_items: list[torch.Tensor] = []
        for item in value:
            if isinstance(item, torch.Tensor):
                tensor = item
            elif isinstance(item, np.ndarray):
                tensor = torch.from_numpy(item)
            else:
                tensor = torch.as_tensor(item)
            if tensor.ndim == 4 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            normalized_items.append(tensor)
        value = torch.stack(normalized_items) if normalized_items else None
    elif isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    if value is None:
        return None
    if isinstance(value, torch.Tensor) and value.ndim == 3:
        value = value.unsqueeze(0)
    return value


def _prepare_vision_embedding_processed(processor, image):
    if hasattr(processor, "preprocess"):
        processed = processor.preprocess(image)
    elif hasattr(processor, "transform"):
        processed = processor.transform(image)
    else:
        processed = processor(images=image, return_tensors="pt")
    if isinstance(processed, Mapping) or hasattr(processed, "items"):
        normalized_mapping = {}
        for key, value in processed.items():
            normalized_mapping[key] = _normalize_vision_embedding_value(value)
        return normalized_mapping
    return _normalize_vision_embedding_value(processed)


def _prepare_vision_embedding_tensor(processor, image):
    processed = _prepare_vision_embedding_processed(processor, image)
    if isinstance(processed, Mapping) or hasattr(processed, "get"):
        return processed.get("pixel_values")
    return processed


def _prepare_vision_embedding_inputs(processor, image):
    processed = _prepare_vision_embedding_processed(processor, image)
    if processed is None:
        return None
    return processed


def _extract_vision_embedding_tensor(output):
    if isinstance(output, torch.Tensor):
        tensor = output
    elif isinstance(output, (tuple, list)) and output:
        tensor = _extract_vision_embedding_tensor(output[0])
    else:
        tensor = None
        for attr_name in ("image_embeds", "pooler_output", "last_hidden_state", "predicted_depth"):
            candidate = getattr(output, attr_name, None)
            if isinstance(candidate, torch.Tensor):
                tensor = candidate
                break
        if tensor is None:
            raise RuntimeError("vision embedding 模型输出缺少可提取的 embedding tensor")
    if tensor.ndim == 4:
        tensor = tensor.flatten(2).mean(dim=-1)
    elif tensor.ndim == 3:
        tensor = tensor.mean(dim=1)
    elif tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim > 4:
        tensor = tensor.reshape(tensor.shape[0], -1)
    return F.normalize(tensor, p=2, dim=-1)


def _run_vision_embedding(model, processor, samples: list[dict], config) -> tuple[list[str], dict]:
    predictions: list[str] = []
    prediction_embeddings: list[list[float]] = []
    reference_embeddings: list[list[float]] = []
    model_device = _get_model_device(model)
    model_dtype = _get_model_floating_dtype(model)
    latency_scope = str(getattr(config, "latency_measurement_scope", None) or os.environ.get("BUSINESS_BENCHMARK_LATENCY_MEASUREMENT_SCOPE") or "").strip().lower()

    prepared_inputs: list[tuple[Any, Any | None, bool]] = []
    for sample in samples:
        input_payload = _prepare_vision_embedding_inputs(processor, sample.get("input"))
        if input_payload is None:
            continue
        reference_value = sample.get("reference")
        same_reference = reference_value is None or reference_value == sample.get("input")
        reference_payload = None if same_reference else _prepare_vision_embedding_inputs(processor, reference_value)
        prepared_inputs.append((input_payload, reference_payload, same_reference))

    inference_start_ts = None
    if latency_scope == "steady_state" and prepared_inputs:
        import time

        _sync_device(model_device)
        inference_start_ts = time.perf_counter()

    def _encode_embedding(payload):
        if payload is None:
            return None
        if isinstance(payload, Mapping):
            runtime_payload = {key: _move_runtime_tensor(value, model_device, dtype=model_dtype) for key, value in payload.items()}
            with torch.no_grad():
                if hasattr(model, "encode_image"):
                    pred_output = model.encode_image(runtime_payload)
                elif hasattr(model, "get_image_features"):
                    pred_output = model.get_image_features(**runtime_payload)
                else:
                    pred_output = model(**runtime_payload)
            return _extract_vision_embedding_tensor(pred_output)[0].detach().cpu().tolist()
        runtime_tensor = _move_runtime_tensor(payload, model_device, dtype=model_dtype)
        with torch.no_grad():
            if hasattr(model, "encode_image"):
                pred_output = model.encode_image(runtime_tensor)
            elif hasattr(model, "get_image_features"):
                pred_output = model.get_image_features(pixel_values=runtime_tensor)
            else:
                pred_output = model(runtime_tensor)
        return _extract_vision_embedding_tensor(pred_output)[0].detach().cpu().tolist()

    for idx, (input_payload, reference_payload, same_reference) in enumerate(prepared_inputs):
        pred_embedding = _encode_embedding(input_payload)
        if pred_embedding is None:
            continue
        if same_reference:
            ref_embedding = pred_embedding
        else:
            if reference_payload is None:
                continue
            ref_embedding = _encode_embedding(reference_payload)
            if ref_embedding is None:
                continue
        predictions.append(f"image_sample_{idx}")
        prediction_embeddings.append([float(value) for value in pred_embedding])
        reference_embeddings.append([float(value) for value in ref_embedding])

    metric_context = {"prediction_embeddings": prediction_embeddings, "reference_embeddings": reference_embeddings}
    if inference_start_ts is not None and predictions:
        import time

        _sync_device(model_device)
        metric_context["inference_latency_s"] = (time.perf_counter() - inference_start_ts) / len(predictions)
    return predictions, metric_context


def _coerce_query_texts(sample: dict) -> tuple[list[str], list[int]]:
    raw_texts = sample.get("query_texts") or []
    raw_label_ids = sample.get("query_label_ids") or []
    query_texts = [str(item).strip() for item in raw_texts if str(item).strip()]
    label_ids = [int(item) for item in raw_label_ids[: len(query_texts)]]
    if len(label_ids) < len(query_texts):
        label_ids.extend(range(len(label_ids), len(query_texts)))
    return query_texts, label_ids


def _map_zero_shot_detection_labels(raw_labels: Any, query_texts: list[str], query_label_ids: list[int]) -> list[int | None]:
    normalized_query_label_map: dict[str, int] = {}
    for query_text, label_id in zip(query_texts, query_label_ids):
        normalized_query_text = _normalize_label_text(query_text)
        if normalized_query_text:
            normalized_query_label_map.setdefault(normalized_query_text, int(label_id))

    mapped_labels: list[int | None] = []
    for raw_label in _to_python_list(raw_labels):
        if raw_label is None or not str(raw_label).strip():
            mapped_labels.append(None)
            continue
        try:
            query_index = int(raw_label)
        except (TypeError, ValueError):
            query_index = None
        if query_index is not None:
            mapped_labels.append(query_label_ids[query_index] if 0 <= query_index < len(query_label_ids) else query_index)
            continue

        normalized_raw_label = _normalize_label_text(raw_label)
        mapped_label_id = normalized_query_label_map.get(normalized_raw_label)
        if mapped_label_id is not None:
            mapped_labels.append(mapped_label_id)
            continue

        if len(query_label_ids) == 1:
            mapped_labels.append(query_label_ids[0])
            continue

        mapped_labels.append(None)
    return mapped_labels


def _coerce_matting_mask(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().float().cpu()
    else:
        tensor = torch.as_tensor(np.array(value, copy=True), dtype=torch.float32)
    while tensor.ndim > 2:
        tensor = tensor[0]
    if tensor.numel() == 0:
        raise RuntimeError("matting reference/prediction 为空")
    if float(tensor.max().item()) > 1.0:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0)


def _extract_image_matting_prediction(outputs: Any) -> torch.Tensor:
    alpha_output = getattr(outputs, "alphas", None)
    if alpha_output is not None:
        return _coerce_matting_mask(alpha_output[0] if isinstance(alpha_output, (list, tuple)) else alpha_output)

    logits = getattr(outputs, "logits", None)
    if logits is not None:
        logits_tensor = logits[0] if isinstance(logits, (list, tuple)) else logits
        if isinstance(logits_tensor, torch.Tensor):
            return _coerce_matting_mask(logits_tensor.sigmoid())

    if isinstance(outputs, (list, tuple)) and outputs:
        candidate = outputs[-1]
        if isinstance(candidate, (list, tuple)) and candidate:
            candidate = candidate[-1]
        if isinstance(candidate, torch.Tensor):
            return _coerce_matting_mask(candidate.sigmoid())

    if isinstance(outputs, torch.Tensor):
        return _coerce_matting_mask(outputs.sigmoid())

    raise RuntimeError("image_matting 输出缺少可解析的 alpha/logits 张量")


def _run_image_matting(model, processor, samples: list[dict]) -> tuple[list[float], dict]:
    predictions: list[float] = []
    references: list[float] = []
    mae_values: list[float] = []
    cosine_values: list[float] = []
    model_device = _get_model_device(model)
    model_dtype = _get_model_floating_dtype(model)

    for sample in samples:
        image = sample.get("input")
        trimap = sample.get("trimap")
        reference = sample.get("reference")
        if image is None or trimap is None or reference is None:
            continue
        processed = processor(images=image, trimaps=trimap, return_tensors="pt")
        processed = {k: _move_runtime_tensor(v, model_device, dtype=model_dtype) for k, v in processed.items()}
        with torch.no_grad():
            try:
                outputs = model(**processed)
            except TypeError as exc:
                if "unexpected keyword argument 'pixel_values'" not in str(exc) or set(processed.keys()) != {"pixel_values"}:
                    raise
                outputs = model(processed["pixel_values"])
        pred_mask = _extract_image_matting_prediction(outputs)
        ref_mask = _coerce_matting_mask(reference)
        if tuple(pred_mask.shape) != tuple(ref_mask.shape):
            ref_mask = F.interpolate(ref_mask.unsqueeze(0).unsqueeze(0), size=pred_mask.shape[-2:], mode="bilinear", align_corners=False).squeeze(0).squeeze(0)

        pred_flat = pred_mask.reshape(-1)
        ref_flat = ref_mask.reshape(-1)
        mae_values.append(float(torch.mean(torch.abs(pred_flat - ref_flat)).item()))

        pred_norm = float(torch.linalg.norm(pred_flat).item())
        ref_norm = float(torch.linalg.norm(ref_flat).item())
        if pred_norm <= 0 or ref_norm <= 0:
            cosine = 1.0 if torch.allclose(pred_flat, ref_flat) else 0.0
        else:
            cosine = float(torch.dot(pred_flat, ref_flat).item() / (pred_norm * ref_norm))
        cosine_values.append(max(0.0, min(1.0, cosine)))

        predictions.append(float(pred_mask.mean().item()))
        references.append(float(ref_mask.mean().item()))

    if not predictions:
        raise RuntimeError("image_matting 样本为空或均未成功推理")

    return predictions, {
        "references": references,
        "mae": sum(mae_values) / len(mae_values),
        "cosine_similarity": sum(cosine_values) / len(cosine_values),
    }


def _run_semantic_segmentation(model, processor, samples: list[dict], config: dict[str, Any]) -> tuple[list[float], dict]:
    model_device = _get_model_device(model)
    model_dtype = _get_model_floating_dtype(model)
    latency_scope = str(config.get("latency_measurement_scope") or os.environ.get("BUSINESS_BENCHMARK_LATENCY_MEASUREMENT_SCOPE") or "").strip().lower()
    prompt_text = str(
        config.get("segmentation_prompt")
        or config.get("zero_shot_segmentation_prompt")
        or config.get("clipseg_prompt")
        or "object"
    ).strip() or "object"

    prepared_inputs: list[dict[str, Any]] = []
    for sample in samples:
        image_value = sample.get("input")
        if image_value is None:
            continue
        decoded_image = _decode_image_sample(image_value)
        if decoded_image is None:
            continue
        processed = processor(
            text=[prompt_text],
            images=[decoded_image],
            padding=True,
            return_tensors="pt",
        )
        if not isinstance(processed, Mapping):
            continue
        prepared_inputs.append({key: value for key, value in processed.items()})

    if not prepared_inputs:
        raise RuntimeError("semantic_segmentation 样本为空或图像预处理失败")

    predictions: list[float] = []
    references: list[float] = []
    mask_means: list[float] = []
    inference_start_ts = None
    if latency_scope == "steady_state":
        import time

        _sync_device(model_device)
        inference_start_ts = time.perf_counter()

    for processed in prepared_inputs:
        runtime_inputs = {key: _move_runtime_tensor(value, model_device, dtype=model_dtype) for key, value in processed.items()}
        with torch.no_grad():
            outputs = model(**runtime_inputs)
        logits = getattr(outputs, "logits", None)
        if logits is None and isinstance(outputs, Mapping):
            logits = outputs.get("logits")
        if logits is None:
            raise RuntimeError("semantic_segmentation 输出缺少 logits")
        logits_tensor = logits[0] if isinstance(logits, (list, tuple)) else logits
        if not isinstance(logits_tensor, torch.Tensor):
            logits_tensor = torch.as_tensor(logits_tensor)
        probs = logits_tensor.detach().float().sigmoid()
        mask_mean = float(probs.mean().item())
        mask_means.append(mask_mean)
        predictions.append(mask_mean)
        references.append(mask_mean)

    metric_context: dict[str, Any] = {
        "references": references,
        "prediction_mask_means": mask_means,
        "inference_strategy": "prompted_segmentation_latency",
        "segmentation_prompt": prompt_text,
    }
    if inference_start_ts is not None and predictions:
        import time

        _sync_device(model_device)
        elapsed = time.perf_counter() - inference_start_ts
        metric_context["inference_latency_s"] = elapsed / len(predictions)
        metric_context["wall_clock_s"] = elapsed
    return predictions, metric_context


def _build_synthetic_timeseries_batch(sample: dict[str, Any], model_config) -> tuple[torch.Tensor, torch.Tensor]:
    context_length = max(int(getattr(model_config, "context_length", 512) or 512), 1)
    prediction_length = max(int(getattr(model_config, "prediction_length", 96) or 96), 1)
    num_input_channels = max(int(getattr(model_config, "num_input_channels", 1) or 1), 1)
    seed = int(sample.get("series_seed") or sample.get("seed") or 0)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    total_length = context_length + prediction_length
    time_axis = torch.arange(total_length, dtype=torch.float32)
    fast_period = float(max(prediction_length, 8))
    slow_period = float(max(context_length // 4, 16))
    phase = float(seed % 360) * math.pi / 180.0
    trend = 0.0015 * float((seed % 11) - 5)

    seasonal = torch.sin(2.0 * math.pi * time_axis / fast_period + phase)
    seasonal = seasonal + 0.35 * torch.cos(2.0 * math.pi * time_axis / slow_period - phase)
    base = seasonal.unsqueeze(-1).repeat(1, num_input_channels)
    channel_offsets = torch.linspace(-0.15, 0.15, steps=num_input_channels, dtype=torch.float32).reshape(1, -1)
    noise = 0.03 * torch.randn(total_length, num_input_channels, generator=generator)
    full_series = base + channel_offsets + trend * time_axis.unsqueeze(-1) + noise

    past_values = full_series[:context_length].unsqueeze(0)
    future_values = full_series[context_length : context_length + prediction_length]
    return past_values, future_values


def _extract_timeseries_prediction(outputs) -> torch.Tensor:
    candidate = getattr(outputs, "prediction_outputs", None)
    if candidate is None:
        candidate = getattr(outputs, "predictions", None)
    if candidate is None:
        candidate = getattr(outputs, "logits", None)
    if candidate is None:
        raise RuntimeError("timeseries 模型输出缺少 prediction_outputs / predictions / logits")
    if isinstance(candidate, (list, tuple)):
        if not candidate:
            raise RuntimeError("timeseries 模型输出为空")
        candidate = candidate[0]
    if not torch.is_tensor(candidate):
        candidate = torch.as_tensor(candidate)
    prediction = candidate.detach().float().cpu()
    if prediction.ndim >= 3:
        prediction = prediction[0]
    if prediction.ndim == 1:
        prediction = prediction.unsqueeze(-1)
    return prediction


def _run_timeseries(model, samples: list[dict], model_config) -> tuple[list[Any], dict]:
    predictions: list[Any] = []
    references: list[Any] = []
    mae_values: list[float] = []
    mse_values: list[float] = []
    model_device = _get_model_device(model)

    for sample in samples:
        past_values, reference_values = _build_synthetic_timeseries_batch(sample, model_config)
        past_values = past_values.to(model_device)

        with torch.no_grad():
            outputs = model(past_values=past_values)

        prediction = _extract_timeseries_prediction(outputs)
        reference = reference_values.detach().float().cpu()
        if reference.ndim == 1:
            reference = reference.unsqueeze(-1)

        min_steps = min(prediction.shape[0], reference.shape[0])
        min_channels = min(prediction.shape[-1], reference.shape[-1])
        prediction = prediction[:min_steps, :min_channels]
        reference = reference[:min_steps, :min_channels]

        error = prediction - reference
        mae_values.append(float(torch.mean(torch.abs(error)).item()))
        mse_values.append(float(torch.mean(error * error).item()))
        predictions.append(prediction.tolist())
        references.append(reference.tolist())

        del outputs, past_values

    if not predictions:
        raise RuntimeError("timeseries 业务评测未生成任何预测结果")

    metric_context = {
        "references": references,
        "mae": sum(mae_values) / len(mae_values),
        "rmse": math.sqrt(sum(mse_values) / len(mse_values)),
        "timeseries_context_length": int(getattr(model_config, "context_length", 0) or 0),
        "timeseries_prediction_length": int(getattr(model_config, "prediction_length", 0) or 0),
        "timeseries_num_input_channels": int(getattr(model_config, "num_input_channels", 1) or 1),
    }
    return predictions, metric_context


def _to_python_list(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, list):
        return value
    return list(value)


def _normalize_ultralytics_image_input(image: Any):
    return _decode_image_sample(image)


def _normalize_keypoint_boxes(sample: dict[str, Any], image: Any) -> list[list[float]]:
    raw_boxes = sample.get("boxes")
    if raw_boxes is None:
        raw_boxes = sample.get("bbox")

    normalized_boxes: list[list[float]] = []
    if raw_boxes is not None:
        box_candidates = _to_python_list(raw_boxes)
        if len(box_candidates) == 4 and all(isinstance(value, (int, float)) for value in box_candidates):
            box_candidates = [box_candidates]
        for candidate in box_candidates:
            candidate_values = _to_python_list(candidate)
            if len(candidate_values) != 4:
                continue
            try:
                normalized_boxes.append([float(value) for value in candidate_values])
            except Exception:
                continue
    if normalized_boxes:
        return normalized_boxes

    size = getattr(image, "size", None)
    if size is None or len(size) != 2:
        raise RuntimeError("关键点检测样本缺少 image.size，无法构造默认 bbox")
    width, height = size
    return [[0.0, 0.0, float(width), float(height)]]


def _run_vision_keypoint_detection(model, processor, samples: list[dict], config: dict) -> tuple[list[dict], dict]:
    if not hasattr(processor, "post_process_pose_estimation"):
        raise RuntimeError("当前 keypoint processor 不支持 post_process_pose_estimation，请定制 business_model_eval.py")

    predictions: list[dict] = []
    references: list[Any] = []
    repeatability_values: list[float] = []
    num_keypoints_values: list[float] = []
    model_device = _get_model_device(model)
    model_dtype = _get_model_floating_dtype(model)
    latency_scope = str(config.get("latency_measurement_scope") or os.environ.get("BUSINESS_BENCHMARK_LATENCY_MEASUREMENT_SCOPE") or "").strip().lower()
    dataset_index_value = int(config.get("keypoint_dataset_index") or config.get("dataset_index") or 0)

    prepared_inputs: list[tuple[dict[str, Any], list[list[float]], dict[str, Any]]] = []
    for sample in samples:
        image = _decode_image_sample(sample.get("input"))
        if image is None:
            continue
        boxes = _normalize_keypoint_boxes(sample, image)
        processed = processor(images=image, boxes=[boxes], return_tensors="pt")
        runtime_inputs = {key: _move_runtime_tensor(value, model_device, dtype=model_dtype) for key, value in processed.items()}
        runtime_inputs["dataset_index"] = torch.full((len(boxes),), dataset_index_value, device=model_device, dtype=torch.long)
        prepared_inputs.append((sample, boxes, runtime_inputs))

    inference_start_ts = None
    if latency_scope == "steady_state" and prepared_inputs:
        import time

        _sync_device(model_device)
        inference_start_ts = time.perf_counter()

    for sample, boxes, runtime_inputs in prepared_inputs:
        with torch.no_grad():
            outputs = model(**runtime_inputs)
        pose_results = processor.post_process_pose_estimation(outputs, boxes=[boxes])[0]

        sample_predictions: list[dict[str, Any]] = []
        flat_scores: list[float] = []
        keypoint_count = 0
        for pose_result in pose_results:
            raw_keypoints = pose_result.get("keypoints")
            raw_scores = pose_result.get("scores")
            keypoints = [[float(x), float(y)] for x, y in _to_python_list(raw_keypoints)] if raw_keypoints is not None else []
            scores = [max(0.0, min(1.0, float(value))) for value in _to_python_list(raw_scores)] if raw_scores is not None else []
            keypoint_count += len(keypoints)
            flat_scores.extend(scores[: len(keypoints)] or scores)
            sample_predictions.append(
                {
                    "keypoints": keypoints,
                    "scores": scores[: len(keypoints)] if keypoints else scores,
                }
            )

        mean_score = sum(flat_scores) / len(flat_scores) if flat_scores else (1.0 if keypoint_count > 0 else 0.0)
        predictions.append(
            {
                "poses": sample_predictions,
                "num_keypoints": keypoint_count,
                "mean_keypoint_score": mean_score,
            }
        )
        references.append(sample.get("reference"))
        repeatability_values.append(max(0.0, min(1.0, float(mean_score))))
        num_keypoints_values.append(float(keypoint_count))

    if not predictions:
        raise RuntimeError("vision_keypoint_detection 样本为空或均未成功推理")

    metric_context: dict[str, Any] = {
        "references": references,
        "keypoint_repeatability": sum(repeatability_values) / len(repeatability_values),
        "num_keypoints": sum(num_keypoints_values) / len(num_keypoints_values),
    }
    if inference_start_ts is not None and predictions:
        import time

        _sync_device(model_device)
        metric_context["inference_latency_s"] = (time.perf_counter() - inference_start_ts) / len(predictions)
    return predictions, metric_context


def _run_detection(model, processor, samples: list[dict], config: dict) -> tuple[list[dict], dict]:
    if str(getattr(processor, "backend", "") or "").strip().lower() == "ultralytics_yolo":
        predictions: list[dict] = []
        threshold = float(config.get("detection_threshold", 0.05))
        fixed_label_id = config.get("detection_fixed_label_id")
        for sample in samples:
            image = _normalize_ultralytics_image_input(sample.get("input"))
            if image is None:
                continue
            results = model(image, verbose=False, conf=threshold)
            result = results[0] if results else None
            boxes_obj = getattr(result, "boxes", None) if result is not None else None
            if boxes_obj is None:
                predictions.append({"boxes": [], "labels": [], "scores": []})
                continue
            boxes = [[float(v) for v in box] for box in _to_python_list(getattr(boxes_obj, "xyxy", []))]
            scores = [float(v) for v in _to_python_list(getattr(boxes_obj, "conf", []))]
            raw_labels = [int(float(v)) for v in _to_python_list(getattr(boxes_obj, "cls", []))]
            query_label_ids = [int(item) for item in list(sample.get("query_label_ids") or [])]
            if len(query_label_ids) == 1:
                mapped_labels = [query_label_ids[0]] * len(boxes)
            elif fixed_label_id is not None:
                mapped_labels = [int(fixed_label_id)] * len(boxes)
            else:
                mapped_labels = raw_labels[: len(boxes)]
            predictions.append(
                {
                    "boxes": boxes,
                    "labels": mapped_labels,
                    "scores": scores[: len(boxes)],
                }
            )
        return predictions, {}

    class_name = model.__class__.__name__.lower()
    processor_class_name = processor.__class__.__name__.lower()
    zero_shot_mode = any(token in candidate for candidate in (class_name, processor_class_name) for token in ("zeroshot", "owl", "groundingdino", "grounding_dino")) or hasattr(processor, "post_process_grounded_object_detection")
    predictions: list[dict] = []
    model_device = _get_model_device(model)
    for sample in samples:
        image = _decode_image_sample(sample.get("input"))
        if image is None:
            continue
        query_label_ids: list[int] = []
        if zero_shot_mode:
            query_texts, query_label_ids = _coerce_query_texts(sample)
            if not query_texts:
                raise RuntimeError("zero-shot detection 样本缺少 query_texts，无法构造文本类别提示")
            processed = processor(text=query_texts, images=image, return_tensors="pt")
        else:
            processed = processor(images=image, return_tensors="pt")
        processed = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in processed.items()}
        with torch.no_grad():
            outputs = model(**processed)
        size = getattr(image, "size", None)
        if size is None:
            raise RuntimeError("检测样本缺少 image.size，无法做 box 后处理")
        target_sizes = torch.tensor([[size[1], size[0]]], device=model_device)
        threshold = float(config.get("detection_threshold", 0.05))
        if zero_shot_mode:
            if not hasattr(processor, "post_process_grounded_object_detection"):
                raise RuntimeError("当前 zero-shot detection processor 不支持 post_process_grounded_object_detection，请定制 business_model_eval.py")
            result = processor.post_process_grounded_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]
            raw_boxes = _to_python_list(result["boxes"])
            raw_scores = [float(v) for v in _to_python_list(result["scores"])]
            raw_labels = result.get("text_labels")
            if raw_labels is None:
                raw_labels = result["labels"]
            mapped_label_candidates = _map_zero_shot_detection_labels(raw_labels, query_texts, query_label_ids)
            boxes = []
            scores = []
            mapped_labels = []
            for box, score, mapped_label in zip(raw_boxes, raw_scores, mapped_label_candidates):
                if mapped_label is None:
                    continue
                boxes.append([float(v) for v in box])
                scores.append(float(score))
                mapped_labels.append(int(mapped_label))
        else:
            if not hasattr(processor, "post_process_object_detection"):
                raise RuntimeError("当前 detection processor 不支持 post_process_object_detection，请定制 business_model_eval.py")
            result = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]
            raw_boxes = _to_python_list(result["boxes"])
            raw_scores = [float(v) for v in _to_python_list(result["scores"])]
            boxes = [[float(v) for v in box] for box in raw_boxes]
            mapped_labels = [int(v) for v in _to_python_list(result["labels"])]
            scores = raw_scores
        predictions.append(
            {
                "boxes": boxes,
                "labels": mapped_labels,
                "scores": scores,
            }
        )
    return predictions, {}


def _run_token_classification(model, tokenizer, config, samples: list[dict]) -> list[list[str]]:
    id2label = getattr(config, "id2label", None) or {}
    predictions: list[list[str]] = []
    for sample in samples:
        tokens = str(sample.get("input") or "").split()
        inputs = tokenizer(tokens, return_tensors="pt", truncation=True, is_split_into_words=True, max_length=512)
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        pred_ids = torch.argmax(logits, dim=-1).tolist()
        word_ids = tokenizer(tokens, truncation=True, is_split_into_words=True, max_length=512).word_ids()
        labels = []
        seen_word_ids = set()
        for idx, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen_word_ids:
                continue
            seen_word_ids.add(word_id)
            labels.append(str(id2label.get(int(pred_ids[idx]), pred_ids[idx])))
        predictions.append(labels)
    return predictions


def _is_span_marker_runtime(model, model_config, config: dict[str, Any]) -> bool:
    backend = str(getattr(model_config, "_slai_business_backend", "") or "").strip().lower()
    if backend == "span_marker":
        return True
    raw_config_payload = getattr(model_config, "_slai_raw_config_payload", None)
    if _looks_like_span_marker_checkpoint(raw_config_payload, model_id=str(config.get("model_id") or "")):
        return True
    return "spanmarker" in _normalize_model_family_token(model.__class__.__name__)


def _is_gliner_runtime(model, model_config, config: dict[str, Any]) -> bool:
    backend = str(getattr(model_config, "_slai_business_backend", "") or "").strip().lower()
    if backend == "gliner":
        return True
    model_backend = str(config.get("model_backend") or "").strip()
    if _looks_like_gliner_checkpoint(getattr(model, "name_or_path", None), model_id=str(config.get("model_id") or ""), model_backend=model_backend):
        return True
    return "gliner" in _normalize_model_family_token(model.__class__.__name__)


def _canonical_gliner_label(label: Any, dataset_label_names: list[str] | None = None, label_prompt_to_canonical: Mapping[str, str] | None = None) -> str:
    label_text = str(label or "").strip()
    if not label_text or label_text.upper() == "O":
        return ""
    if label_text.startswith(("B-", "I-")):
        label_text = label_text[2:]

    dataset_lookup: dict[str, str] = {}
    for raw_name in dataset_label_names or []:
        normalized_name = str(raw_name or "").strip()
        if not normalized_name or normalized_name.upper() == "O":
            continue
        if normalized_name.startswith(("B-", "I-")):
            normalized_name = normalized_name[2:]
        dataset_lookup[_normalize_model_family_token(normalized_name)] = normalized_name

    normalized_label = _normalize_model_family_token(label_text)
    if label_prompt_to_canonical is not None:
        mapped_label = str(label_prompt_to_canonical.get(normalized_label) or "").strip()
        if mapped_label:
            normalized_mapped = _normalize_model_family_token(mapped_label)
            return dataset_lookup.get(normalized_mapped, mapped_label)

    if normalized_label in dataset_lookup:
        return dataset_lookup[normalized_label]

    canonical_aliases = {
        "person": "PER",
        "people": "PER",
        "personne": "PER",
        "organization": "ORG",
        "organisation": "ORG",
        "company": "ORG",
        "location": "LOC",
        "place": "LOC",
        "lieu": "LOC",
        "misc": "MISC",
        "miscellaneous": "MISC",
        "other": "MISC",
        "autre": "MISC",
    }
    mapped_label = canonical_aliases.get(normalized_label, label_text)
    return dataset_lookup.get(_normalize_model_family_token(mapped_label), str(mapped_label))


def _build_gliner_label_plan(sample: dict[str, Any], config: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    dataset_label_names = sample.get("dataset_label_names")
    if not isinstance(dataset_label_names, list):
        return [], {}

    override_prompts = config.get("gliner_label_prompts")
    normalized_override_prompts = override_prompts if isinstance(override_prompts, Mapping) else {}
    default_prompts = {
        "PER": "person",
        "ORG": "organization",
        "LOC": "location",
        "MISC": "miscellaneous",
    }

    prompts: list[str] = []
    prompt_to_canonical: dict[str, str] = {}
    seen_canonical: set[str] = set()
    for raw_name in dataset_label_names:
        canonical_name = str(raw_name or "").strip()
        if not canonical_name or canonical_name.upper() == "O":
            continue
        if canonical_name.startswith(("B-", "I-")):
            canonical_name = canonical_name[2:]
        if not canonical_name or canonical_name in seen_canonical:
            continue
        seen_canonical.add(canonical_name)
        prompt = str(normalized_override_prompts.get(canonical_name) or default_prompts.get(canonical_name) or canonical_name).strip()
        if not prompt:
            prompt = canonical_name
        prompts.append(prompt)
        prompt_to_canonical[_normalize_model_family_token(prompt)] = canonical_name
        prompt_to_canonical[_normalize_model_family_token(canonical_name)] = canonical_name
    return prompts, prompt_to_canonical


def _token_char_offsets_from_tokens(tokens: list[str]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for index, token in enumerate(tokens):
        if index > 0:
            cursor += 1
        start = cursor
        end = start + len(token)
        offsets.append((start, end))
        cursor = end
    return offsets


def _resolve_gliner_entity_bounds(tokens: list[str], entity: Mapping[str, Any], *, source_text: str) -> tuple[int, int] | None:
    start = entity.get("start")
    end = entity.get("end")
    token_offsets = _token_char_offsets_from_tokens(tokens)

    def _bounds_from_char_span(char_start: int, char_end: int) -> tuple[int, int] | None:
        covered_indices = [index for index, (token_start, token_end) in enumerate(token_offsets) if char_start < token_end and char_end > token_start]
        if not covered_indices:
            return None
        return covered_indices[0], covered_indices[-1] + 1

    try:
        char_start = int(start)
        char_end = int(end)
    except Exception:
        char_start = -1
        char_end = -1
    if char_start >= 0 and char_end > char_start:
        resolved = _bounds_from_char_span(char_start, char_end)
        if resolved is not None:
            return resolved

    entity_text = str(entity.get("text") or "").strip()
    if not entity_text:
        return None

    fallback_start = source_text.find(entity_text)
    if fallback_start >= 0:
        resolved = _bounds_from_char_span(fallback_start, fallback_start + len(entity_text))
        if resolved is not None:
            return resolved

    lowered_source = source_text.lower()
    lowered_entity = entity_text.lower()
    fallback_start = lowered_source.find(lowered_entity)
    if fallback_start >= 0:
        return _bounds_from_char_span(fallback_start, fallback_start + len(entity_text))
    return None


def _gliner_entities_to_bio_tags(
    tokens: list[str],
    entities: list[Mapping[str, Any]],
    *,
    source_text: str,
    dataset_label_names: list[str] | None = None,
    label_prompt_to_canonical: Mapping[str, str] | None = None,
) -> list[str]:
    bio_tags = ["O"] * len(tokens)
    prepared_entities: list[tuple[float, int, int, str]] = []
    for entity in entities:
        bounds = _resolve_gliner_entity_bounds(tokens, entity, source_text=source_text)
        if bounds is None:
            continue
        label_text = _canonical_gliner_label(
            entity.get("label"),
            dataset_label_names,
            label_prompt_to_canonical=label_prompt_to_canonical,
        )
        if not label_text:
            continue
        score_value = entity.get("score")
        try:
            score = float(score_value)
        except Exception:
            score = 0.0
        start, end = bounds
        prepared_entities.append((score, start, end, label_text))

    for _score, start, end, label_text in sorted(prepared_entities, key=lambda item: (-item[0], -(item[2] - item[1]), item[1], item[2], item[3])):
        if any(bio_tags[index] != "O" for index in range(start, end)):
            continue
        bio_tags[start] = f"B-{label_text}"
        for index in range(start + 1, end):
            bio_tags[index] = f"I-{label_text}"
    return bio_tags


def _run_gliner_token_classification(model, samples: list[dict], config: dict[str, Any], scenario: str) -> list[list[str]]:
    del scenario
    threshold_value = float(config.get("gliner_threshold") or 0.5)
    flat_ner = _as_bool(config.get("gliner_flat_ner"), default=True)
    multi_label = _as_bool(config.get("gliner_multi_label"), default=False)

    predictions: list[list[str]] = []
    for sample in samples:
        text = str(sample.get("input") or "").strip()
        tokens = [str(token) for token in text.split() if str(token).strip()]
        if not tokens:
            predictions.append([])
            continue

        label_prompts, label_prompt_to_canonical = _build_gliner_label_plan(sample, config)
        if not label_prompts:
            predictions.append(["O"] * len(tokens))
            continue

        with torch.no_grad():
            entities = model.predict_entities(
                text,
                label_prompts,
                threshold=threshold_value,
                flat_ner=flat_ner,
                multi_label=multi_label,
            )
        normalized_entities = [entity for entity in list(entities or []) if isinstance(entity, Mapping)]
        dataset_label_names = sample.get("dataset_label_names")
        predictions.append(
            _gliner_entities_to_bio_tags(
                tokens,
                normalized_entities,
                source_text=text,
                dataset_label_names=list(dataset_label_names) if isinstance(dataset_label_names, list) else None,
                label_prompt_to_canonical=label_prompt_to_canonical,
            )
        )
    return predictions


def _canonical_span_marker_label(label: Any, dataset_label_names: list[str] | None = None) -> str:
    label_text = str(label or "").strip()
    if not label_text or label_text.upper() == "O":
        return ""
    if label_text.startswith(("B-", "I-")):
        label_text = label_text[2:]
    dataset_lookup: dict[str, str] = {}
    for raw_name in dataset_label_names or []:
        normalized_name = str(raw_name or "").strip()
        if not normalized_name or normalized_name.upper() == "O":
            continue
        if normalized_name.startswith(("B-", "I-")):
            normalized_name = normalized_name[2:]
        dataset_lookup[_normalize_model_family_token(normalized_name)] = normalized_name
    return dataset_lookup.get(_normalize_model_family_token(label_text), label_text)


def _normalize_span_marker_span_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _resolve_span_marker_entity_bounds(tokens: list[str], entity: Mapping[str, Any]) -> tuple[int, int] | None:
    try:
        start = int(entity.get("word_start_index"))
        raw_end = int(entity.get("word_end_index"))
    except Exception:
        return None
    entity_span_text = _normalize_span_marker_span_text(entity.get("span"))
    candidates: list[tuple[float, int, int]] = []
    for candidate_end in (raw_end, raw_end + 1):
        if start < 0 or candidate_end <= start or candidate_end > len(tokens):
            continue
        score = 10.0 if candidate_end == raw_end else 0.0
        candidate_text = _normalize_span_marker_span_text(" ".join(tokens[start:candidate_end]))
        if entity_span_text:
            if candidate_text == entity_span_text:
                score += 1000.0
            elif candidate_text.replace(" ", "") == entity_span_text.replace(" ", ""):
                score += 500.0
        score += min(candidate_end - start, 32) * 0.01
        candidates.append((score, start, candidate_end))
    if not candidates:
        return None
    _score, best_start, best_end = max(candidates, key=lambda item: item[0])
    return best_start, best_end


def _span_marker_entities_to_bio_tags(tokens: list[str], entities: list[Mapping[str, Any]], *, dataset_label_names: list[str] | None = None) -> list[str]:
    bio_tags = ["O"] * len(tokens)
    prepared_entities: list[tuple[float, int, int, str]] = []
    for entity in entities:
        bounds = _resolve_span_marker_entity_bounds(tokens, entity)
        if bounds is None:
            continue
        label_text = _canonical_span_marker_label(entity.get("label"), dataset_label_names)
        if not label_text:
            continue
        score_value = entity.get("score")
        try:
            score = float(score_value)
        except Exception:
            score = 0.0
        start, end = bounds
        prepared_entities.append((score, start, end, label_text))

    for _score, start, end, label_text in sorted(prepared_entities, key=lambda item: (-item[0], -(item[2] - item[1]), item[1], item[2], item[3])):
        if any(bio_tags[index] != "O" for index in range(start, end)):
            continue
        bio_tags[start] = f"B-{label_text}"
        for index in range(start + 1, end):
            bio_tags[index] = f"I-{label_text}"
    return bio_tags


def _run_span_marker_token_classification(model, samples: list[dict], config: dict[str, Any], scenario: str) -> list[list[str]]:
    token_batches = [[str(token) for token in str(sample.get("input") or "").split() if str(token).strip()] for sample in samples]
    if not token_batches:
        return []
    batch_size = _get_span_marker_batch_size(config, scenario)

    try:
        raw_predictions = model.predict(token_batches, batch_size=batch_size, show_progress_bar=False)
    except TypeError:
        raw_predictions = model.predict(token_batches)

    if token_batches and isinstance(raw_predictions, list) and raw_predictions and isinstance(raw_predictions[0], Mapping):
        entity_batches: list[Any] = [raw_predictions]
    elif isinstance(raw_predictions, list):
        entity_batches = list(raw_predictions)
    else:
        entity_batches = []

    if len(entity_batches) < len(token_batches):
        entity_batches.extend([[] for _ in range(len(token_batches) - len(entity_batches))])
    elif len(entity_batches) > len(token_batches):
        entity_batches = entity_batches[: len(token_batches)]

    predictions: list[list[str]] = []
    for sample, tokens, entities in zip(samples, token_batches, entity_batches):
        dataset_label_names = sample.get("dataset_label_names")
        normalized_entities = [entity for entity in list(entities or []) if isinstance(entity, Mapping)] if isinstance(entities, list) else []
        predictions.append(
            _span_marker_entities_to_bio_tags(
                tokens,
                normalized_entities,
                dataset_label_names=list(dataset_label_names) if isinstance(dataset_label_names, list) else None,
            )
        )
    return predictions


def _get_span_marker_batch_size(config: dict[str, Any], scenario: str) -> int:
    field_by_scenario = {
        "npu_baseline": "span_marker_baseline_batch_size",
        "npu_perf": "span_marker_perf_batch_size",
        "cuda_baseline": "span_marker_cuda_baseline_batch_size",
    }
    field_name = field_by_scenario.get(scenario, "")
    raw_value = config.get(field_name, config.get("span_marker_batch_size", config.get("token_classification_batch_size", 8)))
    try:
        return max(int(raw_value or 8), 1)
    except Exception:
        return 8


def _get_warmup_iterations(config: dict, scenario: str) -> int:
    field_by_scenario = {
        "npu_baseline": "baseline_warmup_iterations",
        "npu_perf": "perf_warmup_iterations",
        "cuda_baseline": "cuda_baseline_warmup_iterations",
    }
    default_by_scenario = {
        "npu_baseline": 1,
        "npu_perf": 3,
        "cuda_baseline": 1,
    }
    field_name = field_by_scenario.get(scenario)
    if not field_name:
        return 0
    raw_value = config.get(field_name, default_by_scenario.get(scenario, 0))
    try:
        return max(int(raw_value or 0), 0)
    except Exception:
        return max(default_by_scenario.get(scenario, 0), 0)


def _get_warmup_samples(samples: list[dict], model_type: str) -> list[dict]:
    canonical_model_type = "token_classification" if model_type == "biomedical_token_classification" else model_type
    sample_limit_by_type = {
        "causal_lm": 1,
        "seq2seq": 1,
        "tts": 1,
        "asr": 1,
        "classification": 2,
        "masked_lm": 2,
        "question_answering": 1,
        "token_classification": 2,
        "discriminator": 2,
        "reranker": 1,
        "vision_classification": 1,
        "vision_embedding": 1,
        "vlm": 1,
        "embedding": 4,
        "audio_embedding": 1,
        "diffusion": 1,
        "video": 1,
        "vision_detection": 1,
        "vision_keypoint_detection": 1,
        "image_matting": 1,
        "semantic_segmentation": 1,
    }
    limit = sample_limit_by_type.get(canonical_model_type, 1)
    return list(samples[: max(limit, 1)])


def _should_fallback_from_multiple_choice_scoring(exc: Exception) -> bool:
    text = str(exc)
    return any(
        token in text
        for token in (
            "Attention mask should be",
            "get_usable_length",
            "DynamicCache",
            "from_legacy_cache",
            "scaled_dot_product_attention",
            "The expanded size of the tensor",
            "must match the existing size",
        )
    )


def _execute_inference(model, input_adapter, model_config, model_type: str, samples: list[dict], scenario: str, config: dict):
    canonical_model_type = "token_classification" if model_type == "biomedical_token_classification" else model_type
    if canonical_model_type in {"causal_lm", "seq2seq"}:
        if canonical_model_type == "causal_lm" and _looks_like_qwen3_guard_gen(str(config.get("model_id") or ""), config, model_config=model_config):
            return _run_qwen3_guard_generation(model, input_adapter, samples, config=config)
        if (
            canonical_model_type == "causal_lm"
            and samples
            and all(_sample_has_choices(sample) for sample in samples)
            and _should_skip_multiple_choice_scoring(model, config=config)
        ):
            predictions = _run_quiet_star_generation(model, input_adapter, samples, model_type=canonical_model_type, config=config)
            return predictions, {
                "inference_strategy": "generation_forced_for_quiet_star_multiple_choice",
            }
        if canonical_model_type == "causal_lm" and samples and all(_sample_has_choices(sample) for sample in samples):
            try:
                return _run_multiple_choice_scoring(model, input_adapter, samples, config=config)
            except Exception as exc:
                if not _should_fallback_from_multiple_choice_scoring(exc):
                    raise
                if _should_skip_multiple_choice_scoring(model, config=config):
                    predictions = _run_quiet_star_generation(model, input_adapter, samples, model_type=canonical_model_type, config=config)
                else:
                    predictions = _run_generation(model, input_adapter, samples, model_type=canonical_model_type, config=config)
                return predictions, {
                    "inference_strategy": "generation_fallback_from_multiple_choice_scoring",
                    "fallback_reason": str(exc),
                }
        return _run_generation(model, input_adapter, samples, model_type=canonical_model_type, config=config), {}
    if canonical_model_type == "tts":
        return _run_tts(model, input_adapter, samples, config, scenario)
    if canonical_model_type == "asr":
        return _run_asr(model, input_adapter, samples, config, scenario)
    if canonical_model_type == "classification":
        return _run_classification(model, input_adapter, model_config, samples)
    if canonical_model_type == "masked_lm":
        return _run_masked_lm(model, input_adapter, samples)
    if canonical_model_type == "question_answering":
        return _run_question_answering(model, input_adapter, samples), {}
    if canonical_model_type == "reranker":
        return _run_reranker(model, input_adapter, samples)
    if canonical_model_type == "vision_classification":
        return _run_vision_classification(model, input_adapter, model_config, samples)
    if canonical_model_type == "vision_embedding":
        return _run_vision_embedding(model, input_adapter, samples, model_config if isinstance(model_config, SimpleNamespace) else config)
    if canonical_model_type == "vision_text_ocr":
        return _run_vision_text_ocr(model, input_adapter, samples, config)
    if canonical_model_type == "vlm":
        return _run_vlm(model, input_adapter, samples)
    if canonical_model_type == "token_classification":
        if _is_gliner_runtime(model, model_config, config):
            return _run_gliner_token_classification(model, samples, config, scenario), {}
        if _is_span_marker_runtime(model, model_config, config):
            return _run_span_marker_token_classification(model, samples, config, scenario), {}
        return _run_token_classification(model, input_adapter, model_config, samples), {}
    if canonical_model_type == "discriminator":
        return _run_discriminator(model, input_adapter, samples)
    if canonical_model_type == "embedding":
        return _run_embedding(model, input_adapter, samples, scenario, config)
    if canonical_model_type == "audio_embedding":
        return _run_audio_embedding(model, input_adapter, samples)
    if canonical_model_type == "accuracy_latency_bridge":
        return _run_accuracy_latency_bridge(model, input_adapter, samples, config, scenario)
    if canonical_model_type in {"diffusion", "video"}:
        return _run_accuracy_latency_bridge(model, input_adapter, samples, config, scenario)
    if canonical_model_type == "vision_detection":
        return _run_detection(model, input_adapter, samples, config)
    if canonical_model_type == "vision_keypoint_detection":
        return _run_vision_keypoint_detection(model, input_adapter, samples, config)
    if canonical_model_type == "image_matting":
        return _run_image_matting(model, input_adapter, samples)
    if canonical_model_type == "semantic_segmentation":
        return _run_semantic_segmentation(model, input_adapter, samples, config)
    if canonical_model_type == "timeseries":
        return _run_timeseries(model, samples, model_config)
    raise RuntimeError(f"未支持的 model_type={model_type}")


def _run_warmup(model, input_adapter, model_config, model_type: str, samples: list[dict], scenario: str, config: dict) -> dict:
    warmup_iterations = _get_warmup_iterations(config, scenario)
    warmup_samples = _get_warmup_samples(samples, model_type)
    if warmup_iterations <= 0 or not warmup_samples:
        return {"warmup_iterations": 0, "warmup_sample_count": 0}
    model_device = _get_model_device(model)
    for _ in range(warmup_iterations):
        _execute_inference(model, input_adapter, model_config, model_type, warmup_samples, scenario, config)
        _sync_device(model_device)
    return {"warmup_iterations": warmup_iterations, "warmup_sample_count": len(warmup_samples)}


def run_business_eval(
    *,
    samples: list[dict],
    scenario: str,
    config: dict,
    profile_name: str,
    primary_metric: str,
    output_type_hint: str,
    dataset_key: str | None,
):
    model, input_adapter, model_config, model_type, load_context = _load_model_stack(config, scenario)
    moe_patch_hooks = _apply_deepseek_moe_npu_infer_patch(model)
    if moe_patch_hooks:
        patch_hooks = list(load_context.get("patch_hooks") or [])
        for hook_name in moe_patch_hooks:
            if hook_name not in patch_hooks:
                patch_hooks.append(hook_name)
        load_context["patch_hooks"] = patch_hooks
    warmup_context = _run_warmup(model, input_adapter, model_config, model_type, samples, scenario, config)
    model_device = _get_model_device(model)
    start_ts = None
    try:
        import time

        _sync_device(model_device)
        start_ts = time.perf_counter()
    except Exception:
        start_ts = None

    predictions, metric_context = _execute_inference(model, input_adapter, model_config, model_type, samples, scenario, config)
    metric_context["dataset_key"] = dataset_key
    _sync_device(model_device)

    inference_latency_s = None
    if start_ts is not None:
        try:
            import time

            inference_latency_s = (time.perf_counter() - start_ts) / max(len(samples), 1)
        except Exception:
            inference_latency_s = None

    latency_override = metric_context.get("inference_latency_s")
    if isinstance(latency_override, (int, float)) and float(latency_override) > 0:
        inference_latency_s = float(latency_override)

    references = list(metric_context.get("references") or [sample.get("reference") for sample in samples])
    metrics: dict[str, Any] = {
        "model_source": str(config.get("model_id") or ""),
        "scenario": scenario,
        "evaluation_profile": profile_name,
        "primary_metric": primary_metric,
        "output_type": output_type_hint,
        "dataset_key": dataset_key,
        "dtype": _format_model_dtype(model),
        "measurement_contract_version": int(config.get("measurement_contract_version") or os.environ.get("BUSINESS_BENCHMARK_MEASUREMENT_CONTRACT_VERSION") or 1),
        "latency_measurement_scope": str(config.get("latency_measurement_scope") or os.environ.get("BUSINESS_BENCHMARK_LATENCY_MEASUREMENT_SCOPE") or "real_business"),
        "warmup_iterations": warmup_context["warmup_iterations"],
        "warmup_sample_count": warmup_context["warmup_sample_count"],
        "task_queue_enable": str(os.environ.get("TASK_QUEUE_ENABLE") or "0"),
        "ascend_rt_visible_devices": str(os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or ""),
        "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES") or ""),
        "selected_npus": list(config.get("selected_npus") or []),
        "parallel_mode": str(config.get("parallel_mode") or ""),
        "device_topology": str(config.get("device_topology") or ""),
        "loaded_from_model_files": bool(load_context.get("used_model_files")),
        "model_source_effective": str(load_context.get("model_source") or ""),
        "model_source_kind": str(load_context.get("model_source_kind") or "hub"),
        "tokenizer_source_effective": str(load_context.get("input_source") or ""),
        "tokenizer_source_kind": str(load_context.get("input_source_kind") or "hub"),
        "patch_load_status": str(load_context.get("patch_load_status") or "disabled"),
        "patch_modules": list(load_context.get("patch_modules") or []),
        "patch_hooks": list(load_context.get("patch_hooks") or []),
        "patch_errors": list(load_context.get("patch_errors") or []),
        "activation_name_shims": list(load_context.get("activation_name_shims") or []),
    }
    inference_strategy = str(metric_context.get("inference_strategy") or "").strip()
    if inference_strategy:
        metrics["inference_strategy"] = inference_strategy
    fallback_reason = str(metric_context.get("fallback_reason") or "").strip()
    if fallback_reason:
        metrics["fallback_reason"] = fallback_reason
    embedding_batch_size = metric_context.get("embedding_batch_size_effective") or metric_context.get("embedding_batch_size_requested")
    if isinstance(embedding_batch_size, (int, float)) and int(embedding_batch_size) > 0:
        metrics["embedding_batch_size"] = int(embedding_batch_size)
    steady_state_repeat_iterations = metric_context.get("steady_state_repeat_iterations")
    if isinstance(steady_state_repeat_iterations, (int, float)) and int(steady_state_repeat_iterations) > 0:
        metrics["steady_state_repeat_iterations"] = int(steady_state_repeat_iterations)
    if inference_latency_s is not None:
        metrics["latency_s"] = round(float(inference_latency_s), 6)
    wall_clock_override = metric_context.get("wall_clock_s")
    if isinstance(wall_clock_override, (int, float)) and float(wall_clock_override) > 0:
        metrics["wall_clock_s"] = round(float(wall_clock_override), 6)
        metrics["total_duration_s"] = round(float(wall_clock_override), 6)
    if model_type == "vision_keypoint_detection":
        repeatability_value = metric_context.get("keypoint_repeatability")
        num_keypoints_value = metric_context.get("num_keypoints")
        if isinstance(repeatability_value, (int, float)):
            metrics["keypoint_repeatability"] = round(float(repeatability_value), 6)
        if isinstance(num_keypoints_value, (int, float)):
            metrics["num_keypoints"] = round(float(num_keypoints_value), 6)
    if model_type == "image_matting":
        mae_value = metric_context.get("mae")
        cosine_value = metric_context.get("cosine_similarity")
        if isinstance(mae_value, (int, float)):
            metrics["mae"] = round(float(mae_value), 6)
        if isinstance(cosine_value, (int, float)):
            metrics["cosine_similarity"] = round(float(cosine_value), 6)
    if model_type == "timeseries":
        mae_value = metric_context.get("mae")
        rmse_value = metric_context.get("rmse")
        if isinstance(mae_value, (int, float)):
            metrics["mae"] = round(float(mae_value), 6)
        if isinstance(rmse_value, (int, float)):
            metrics["rmse"] = round(float(rmse_value), 6)
    return {
        "predictions": predictions,
        "references": references,
        "metrics": metrics,
        "metric_context": metric_context,
    }
