"""Standalone script to retrieve model metadata from HuggingFace Hub or GitHub.
Outputs JSON to stdout and logs to stderr.
"""

import argparse
import base64
import json
import logging
import os
import sys
from typing import Any, Dict, List, cast

import requests
from huggingface_hub import HfApi, ModelInfo

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


class ModelInfoRetriever:
    """Retrieves model metadata."""

    def __init__(self):
        self.hf_api = HfApi()
        self.hf_mirror = os.getenv("HF_ENDPOINT", "https://huggingface.co")

    def retrieve(self, model_id: str, source: str) -> Dict[str, Any]:
        """Unified retrieval method."""
        if source == "huggingface":
            return self._retrieve_huggingface(model_id)
        elif source == "github":
            return self._retrieve_github(model_id)
        else:
            raise ValueError(f"Unsupported source: {source}")

    def _retrieve_huggingface(self, model_id: str) -> Dict[str, Any]:
        """Retrieve from HuggingFace Hub."""
        logger.info(f"Retrieving metadata for {model_id} from HuggingFace...")

        try:
            model_info: ModelInfo = self.hf_api.model_info(model_id)

            model_data = {
                "id": model_info.id,
                "source": "huggingface",
                "tags": model_info.tags or [],
                "siblings": [sibling.rfilename for sibling in (model_info.siblings or [])],
                "transformers_info": {},
                "model_type": "Custom",
                "dependencies": ["transformers>=4.36.0", "accelerate>=0.26.0"],
            }

            # Extract Transformers Info
            if model_info.transformers_info:
                transformers_info = model_info.transformers_info
                model_data["transformers_info"] = {
                    "auto_model": transformers_info.get("auto_model"),
                    "processor": transformers_info.get("processor"),
                }

                auto_model = transformers_info.get("auto_model", "")
                if "CausalLM" in auto_model:
                    model_data["model_type"] = "CausalLM"
                elif "SequenceClassification" in auto_model:
                    model_data["model_type"] = "Classification"
                elif auto_model:
                    model_data["model_type"] = "Base"

            # Try to fetch config.json
            try:
                config_url = f"{self.hf_mirror}/{model_id}/resolve/main/config.json"
                response = requests.get(config_url, timeout=10)
                if response.status_code == 200:
                    model_data["config"] = response.json()
            except Exception as e:
                logger.warning(f"Could not fetch config.json: {e}")

            # Estimate size
            model_data["size"] = self._estimate_size(cast(List[Any], model_info.siblings or []))

            logger.info("Metadata retrieval successful.")
            return model_data

        except Exception as e:
            logger.error(f"Failed to retrieve metadata: {e}")
            raise

    def _retrieve_github(self, repo_path: str) -> Dict[str, Any]:
        """Retrieve from GitHub."""
        logger.info(f"Retrieving metadata for {repo_path} from GitHub...")

        try:
            owner, repo = repo_path.split("/", 1)
            repo_url = f"https://api.github.com/repos/{owner}/{repo}"
            response = requests.get(repo_url, timeout=10)
            response.raise_for_status()
            repo_data = response.json()

            model_data = {
                "id": repo_path,
                "source": "github",
                "model_type": "Custom",
                "description": repo_data.get("description", ""),
                "dependencies": [],
                "transformers_info": {},
            }

            # Try to fetch requirements.txt
            try:
                req_url = f"https://api.github.com/repos/{owner}/{repo}/contents/requirements.txt"
                req_response = requests.get(req_url, timeout=10)
                if req_response.status_code == 200:
                    req_data = req_response.json()
                    if "content" in req_data:
                        content = base64.b64decode(req_data["content"]).decode("utf-8")
                        model_data["dependencies"] = self._parse_requirements(content)
            except Exception:
                pass

            logger.info("Metadata retrieval successful.")
            return model_data

        except Exception as e:
            logger.error(f"Failed to retrieve metadata: {e}")
            raise

    def _estimate_size(self, siblings: List[Any]) -> str:
        """Estimate model size."""
        total_size = 0
        for sibling in siblings:
            if sibling.size:
                total_size += sibling.size

        if total_size == 0:
            return "Unknown"

        for unit in ["B", "KB", "MB", "GB"]:
            if total_size < 1024:
                return f"{total_size:.2f}{unit}"
            total_size /= 1024
        return f"{total_size:.2f}TB"

    def _parse_requirements(self, content: str) -> List[str]:
        """Parse requirements.txt content."""
        dependencies = []
        for line in content.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                if "#" in line:
                    line = line.split("#")[0].strip()
                dependencies.append(line)
        return dependencies


def main():
    parser = argparse.ArgumentParser(description="Get model metadata as JSON")
    parser.add_argument("model_id", help="Model ID (e.g. microsoft/biogpt)")
    parser.add_argument("--source", choices=["huggingface", "github"], default="huggingface", help="Model source")

    args = parser.parse_args()

    try:
        retriever = ModelInfoRetriever()
        model_data = retriever.retrieve(args.model_id, args.source)
        print(json.dumps(model_data, indent=2, ensure_ascii=False))
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
