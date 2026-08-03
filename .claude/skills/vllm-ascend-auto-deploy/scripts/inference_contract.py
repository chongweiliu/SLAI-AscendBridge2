#!/usr/bin/env python3
"""Build endpoint-aware smoke-test contracts for supported model tasks."""

from __future__ import annotations


def build_contract(
    task_type: str,
    model_name: str,
    *,
    validation_asset_url: str | None = None,
) -> dict:
    if task_type == "embedding":
        return {
            "endpoint": "/v1/embeddings",
            "mode": "embedding",
            "payload": {
                "model": model_name,
                "input": ["The capital of China is Beijing.", "Gravity attracts bodies."],
            },
        }
    if task_type == "rerank":
        return {
            "endpoint": "/v1/rerank",
            "mode": "rerank",
            "payload": {
                "model": model_name,
                "query": "What is the capital of China?",
                "documents": [
                    "The capital of China is Beijing.",
                    "Gravity attracts bodies.",
                ],
            },
        }
    if task_type in {"multimodal_chat", "asr"}:
        if not validation_asset_url:
            raise ValueError(
                f"{task_type} requires validation_asset_url for a real smoke test"
            )
        media_type = "audio_url" if task_type == "asr" else "image_url"
        prompt = "Transcribe this audio." if task_type == "asr" else "Describe this image."
        return {
            "endpoint": "/v1/chat/completions",
            "mode": "text",
            "payload": {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": media_type, media_type: {"url": validation_asset_url}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": 32,
            },
        }
    if task_type == "reward":
        return {
            "endpoint": "/v1/reward",
            "mode": "reward",
            "payload": {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful math assistant."},
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "2+2 equals 4."},
                ],
            },
        }
    return {
        "endpoint": "/v1/chat/completions",
        "mode": "exact_text",
        "expected_exact": "4",
        "payload": {
            "model": model_name,
            "messages": [{"role": "user", "content": "Reply with exactly: 4"}],
            "temperature": 0,
            "max_tokens": 128,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
