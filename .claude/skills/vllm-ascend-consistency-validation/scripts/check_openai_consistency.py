#!/usr/bin/env python3
"""Compare two OpenAI-compatible chat endpoints with a deterministic request."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def request(url: str, model: str, prompt: str, timeout: float, api_key: str | None = None) -> dict:
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "top_p": 1, "max_tokens": 64, "stream": False}
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return {"status": response.status, "body": json.loads(response.read().decode())}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        return {"status": None, "error": str(error)}


def extract(result: dict) -> tuple[str | None, object]:
    body = result.get("body", {})
    choices = body.get("choices", []) if isinstance(body, dict) else []
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    if not isinstance(message, dict):
        message = {}
    return message.get("content"), body.get("usage") if isinstance(body, dict) else None


def compare(baseline: dict, candidate: dict, allow_text_difference: bool = False) -> dict[str, object]:
    baseline_text, baseline_usage = extract(baseline)
    candidate_text, candidate_usage = extract(candidate)
    passed = baseline.get("status") == 200 and candidate.get("status") == 200 and baseline_text is not None and candidate_text is not None
    if not allow_text_difference:
        passed = passed and baseline_text == candidate_text
    return {"passed": passed, "baseline": {"status": baseline.get("status"), "text": baseline_text, "usage": baseline_usage}, "candidate": {"status": candidate.get("status"), "text": candidate_text, "usage": candidate_usage}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--allow-text-difference", action="store_true")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="environment variable containing the bearer token")
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env)
    baseline = request(args.baseline, args.model, args.prompt, args.timeout, api_key)
    candidate = request(args.candidate, args.model, args.prompt, args.timeout, api_key)
    report = compare(baseline, candidate, args.allow_text_difference)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        print("consistency validation failed", file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
