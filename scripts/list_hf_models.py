"""List Hugging Face model IDs by sort order (e.g. downloads, trending).
Outputs one model_id per line to stdout for piping to get_model_info + register_model.
"""

import argparse
import sys

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="List HF model IDs by downloads/trending for crawler discovery.")
    parser.add_argument(
        "--sort",
        choices=["downloads", "trending", "created"],
        default="downloads",
        help="Sort order (default: downloads for hot models)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max number of model IDs to output (default: 20)",
    )
    parser.add_argument(
        "--library",
        type=str,
        default="",
        help="Filter by library, e.g. transformers (optional)",
    )
    args = parser.parse_args()

    try:
        api = HfApi()
        kwargs = {"sort": args.sort, "limit": args.limit}
        if args.library:
            kwargs["library"] = args.library
        for model in api.list_models(**kwargs):
            print(model.id)
    except Exception as e:
        sys.stderr.write(f"[list_hf_models] Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
