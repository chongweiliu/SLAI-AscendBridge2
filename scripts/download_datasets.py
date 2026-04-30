#!/usr/bin/env python3
"""
下载 board 关联 adaptations 使用的测试数据集到 datasets/。

仅下载 validation/test（不拉 train），保存到 datasets/{name}/，
用 load_from_disk("datasets/{name}") 加载。
依赖: pip install datasets huggingface_hub
Token: 设置 HF_TOKEN 环境变量，或写入项目根 .env 文件（HF_TOKEN=xxx）
镜像: 设置 HF_ENDPOINT 或使用 --mirror，如 HF_ENDPOINT=https://hf-mirror.com

用法:
  uv run python scripts/download_datasets.py              # 下载全部（8 线程并行下载文件）
  uv run python scripts/download_datasets.py --jobs 16   # 16 线程并行下载文件
  uv run python scripts/download_datasets.py --list      # 列出可下载的数据集
  uv run python scripts/download_datasets.py --mirror    # 使用国内镜像加速
  uv run python scripts/download_datasets.py sst2 squad_v2 conll2003 science_ie  # 指定数据集
"""

import argparse
import io
import os
import shutil
import sys
import tarfile
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, cast

from dataset_mapping import get_business_benchmark_profile, get_dataset_candidates_for_model


def _safe_print(*args, **kwargs):
    print(*args, **kwargs)


# 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATASETS_DIR = _PROJECT_ROOT / "datasets"


def _load_dotenv():
    """从 .env 加载环境变量（不依赖 python-dotenv）"""
    env_path = _PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                os.environ.setdefault(k.strip(), v)


# 数据集映射: (hf_path, config, split)
# 注：已移除 scripts/不存在/格式错误：birdbench, spider, ncbi_disease, bc5cdr, blurb, scierc, cmmlu, ade20k, common_voice, indonlg
DATASET_MAP = {
    # === 已有 ===
    "sst2": ("glue", "sst2", "validation"),
    "glue_sst2": ("glue", "sst2", "validation"),
    "tweet_eval_sentiment": ("tweet_eval", "sentiment", "test"),
    "tweet_eval_emotion": ("tweet_eval", "emotion", "test"),
    "tweet_eval_offensive": ("tweet_eval", "offensive", "test"),
    "tweet_eval_hate": ("tweet_eval", "hate", "test"),
    "squad_v2": ("squad_v2", None, "validation"),
    "mmlu": ("cais/mmlu", "all", "test"),
    "imagenet": ("ILSVRC/imagenet-1k", "default", "validation"),
    "ms_marco": ("microsoft/ms_marco", "v2.1", "validation"),
    "conll2003": ("hgissbkh/conll2003-en", None, "validation"),
    "science_ie": ("DFKI-SLT/science_ie", None, "test"),
    "ncbi_disease": ("ncbi/ncbi_disease", None, "validation"),
    "bionlp2004": ("bionlp2004", None, "test"),
    "imdb": ("imdb", None, "test"),
    "ag_news": ("ag_news", None, "test"),
    # === 生物医学与科学 ===
    "pubmed_qa": ("qiaojin/PubMedQA", "pqa_labeled", "train"),  # 仅 train
    "chemprot": ("bigbio/chemprot", "chemprot_shared_task_eval_source", "test"),
    # === 通用 NLP / LLM ===
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "glue_mnli": ("glue", "mnli", "validation_matched"),
    "glue_qnli": ("glue", "qnli", "validation"),
    "cnn_dailymail": ("cnn_dailymail", "3.0.0", "validation"),
    "xsum": ("xsum", None, "test"),
    "samsum": ("samsum", None, "test"),
    "scienceqa": ("derek-thomas/ScienceQA", None, "test"),
    # === 计算机视觉 ===
    "coco": ("detection-datasets/coco", "default", "val"),
    "pubtables_detection_1500": ("ucsahin/pubtables-detection-1500-samples", None, "train"),
    "cifar100": ("cifar100", None, "test"),
    "fairface": ("nateraw/fairface", None, "validation"),
    "synthetic_keypoints": ("synthetic_keypoints", None, "validation"),
    "synthetic_matting": ("synthetic_matting", None, "validation"),
    "synthetic_protein": ("synthetic_protein", None, "validation"),
    # === 语音 ASR ===
    "librispeech": ("openslr/librispeech_asr", "clean", "test"),
    "fleurs_ar_eg": ("google/fleurs", "ar_eg", "test"),
    "fleurs_cmn_hans_cn": ("google/fleurs", "cmn_hans_cn", "test"),
    "fleurs_el_gr": ("google/fleurs", "el_gr", "test"),
    "fleurs_hu_hu": ("google/fleurs", "hu_hu", "test"),
    "fleurs_ja_jp": ("google/fleurs", "ja_jp", "test"),
    "fleurs_nl_nl": ("google/fleurs", "nl_nl", "test"),
    "fleurs_pl_pl": ("google/fleurs", "pl_pl", "test"),
    "fleurs_pt_br": ("google/fleurs", "pt_br", "test"),
    "fleurs_ro_ro": ("google/fleurs", "ro_ro", "test"),
    "fleurs_ru_ru": ("google/fleurs", "ru_ru", "test"),
    "fleurs_ur_pk": ("google/fleurs", "ur_pk", "test"),
    "fleurs_vi_vn": ("google/fleurs", "vi_vn", "test"),
    "mcspeech_pl": ("openslr/mcspeech", "pl", "train"),
    "ceval": ("ceval/ceval-exam", "computer_network", "val"),
    # === 语言模型常用 ===
    "wikitext": ("wikitext", "wikitext-2-raw-v1", "test"),
}

HF_MIRROR = "https://hf-mirror.com"
HF_OFFICIAL_ENDPOINT = "https://huggingface.co"

# 无 org 的 path 对应的 hub repo_id（部分数据集需 datasets/ 前缀）
REPO_ID_MAP = {
    "glue": "datasets/glue",
    "squad_v2": "datasets/squad_v2",
    "imdb": "datasets/imdb",
    "ag_news": "datasets/ag_news",
    "cifar100": "datasets/cifar100",
    "tweet_eval": "datasets/tweet_eval",
}

FAIRFACE_AGE_LABELS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
FAIRFACE_GENDER_LABELS = ["Female", "Male"]
FAIRFACE_RACE_LABELS = ["Black", "East Asian", "Indian", "Latino_Hispanic", "Middle Eastern", "Southeast Asian", "White"]
MCSPEECH_PL_MIRRORS = (
    "https://openslr.elda.org/resources/142/mcspeech.tar.gz",
    "https://openslr.trmal.net/resources/142/mcspeech.tar.gz",
    "https://openslr.magicdatatech.com/resources/142/mcspeech.tar.gz",
)


def _repo_id(hf_path: str) -> str:
    """解析 hub repo_id"""
    if "/" in hf_path:
        return hf_path
    return REPO_ID_MAP.get(hf_path, hf_path)


def _normalize_hf_endpoint(value: Optional[str]) -> str:
    return str(value or "").strip().rstrip("/")


def _is_official_hf_endpoint(value: Optional[str]) -> bool:
    normalized = _normalize_hf_endpoint(value)
    return normalized in {"", _normalize_hf_endpoint(HF_OFFICIAL_ENDPOINT), "https://www.huggingface.co"}


@contextmanager
def _temporary_hf_endpoint(value: Optional[str]):
    previous = os.environ.get("HF_ENDPOINT")
    try:
        if value is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = value
        yield
    finally:
        if previous is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = previous


def _coerce_class_label_index(value: object, names: list[str]) -> int:
    if isinstance(value, bool):
        raise ValueError(f"非法 ClassLabel 值: {value!r}")
    if isinstance(value, int):
        if 0 <= value < len(names):
            return value
        raise ValueError(f"ClassLabel 超出范围: {value}")
    if isinstance(value, float):
        return _coerce_class_label_index(int(value), names)
    text = str(value or "").strip()
    if not text:
        raise ValueError("空 ClassLabel")
    if text in names:
        return names.index(text)
    raise ValueError(f"未知 ClassLabel: {text}")


def _download_fairface_dataset(out_dir: Path, cache_dir: Path, token: Optional[str] = None) -> None:
    from datasets import Dataset, Features
    from datasets.features import ClassLabel, Value
    from huggingface_hub import hf_hub_download
    import pickle

    val_path = hf_hub_download(
        repo_id="nateraw/fairface",
        filename="val.pt",
        repo_type="dataset",
        cache_dir=str(cache_dir / "hub"),
        token=token,
    )
    with Path(val_path).open("rb") as handle:
        examples = pickle.load(handle)

    columns = {
        "img_bytes": [],
        "age": [],
        "gender": [],
        "race": [],
    }
    for example in examples:
        if not isinstance(example, dict):
            continue
        image_bytes = example.get("img_bytes")
        if not isinstance(image_bytes, (bytes, bytearray)):
            continue
        columns["img_bytes"].append(bytes(image_bytes))
        columns["age"].append(_coerce_class_label_index(example.get("age"), FAIRFACE_AGE_LABELS))
        columns["gender"].append(_coerce_class_label_index(example.get("gender"), FAIRFACE_GENDER_LABELS))
        columns["race"].append(_coerce_class_label_index(example.get("race"), FAIRFACE_RACE_LABELS))

    features = Features(
        {
            "img_bytes": Value("binary"),
            "age": ClassLabel(names=FAIRFACE_AGE_LABELS),
            "gender": ClassLabel(names=FAIRFACE_GENDER_LABELS),
            "race": ClassLabel(names=FAIRFACE_RACE_LABELS),
        }
    )
    dataset = Dataset.from_dict(columns, features=features)
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    dataset.save_to_disk(str(out_dir))


def _download_mcspeech_pl_dataset(out_dir: Path) -> None:
    import csv
    import gzip
    import subprocess

    from datasets import Dataset, Features
    from datasets.features import Value

    curl_path = shutil.which("curl")
    if not curl_path:
        raise RuntimeError("mcspeech_pl download requires curl")

    sample_limit_text = str(os.environ.get("SLAI_MCSPEECH_MAX_SAMPLES") or "64").strip()
    try:
        sample_limit = max(64, int(sample_limit_text))
    except Exception:
        sample_limit = 64

    chunk_size_mb_text = str(os.environ.get("SLAI_MCSPEECH_RANGE_CHUNK_MB") or "8").strip()
    max_download_mb_text = str(os.environ.get("SLAI_MCSPEECH_MAX_DOWNLOAD_MB") or "128").strip()
    try:
        chunk_size = max(2, int(chunk_size_mb_text)) * 1024 * 1024
    except Exception:
        chunk_size = 8 * 1024 * 1024
    try:
        max_download_bytes = max(32, int(max_download_mb_text)) * 1024 * 1024
    except Exception:
        max_download_bytes = 128 * 1024 * 1024

    partial_dir = out_dir.parent / "hub" / "mcspeech_pl"
    partial_dir.mkdir(parents=True, exist_ok=True)
    partial_path = partial_dir / "mcspeech.tar.gz.partial"

    def _extract_rows_from_partial_archive(archive_path: Path) -> list[dict[str, object]]:
        collected: list[dict[str, object]] = []
        transcripts: dict[str, str] = {}

        def _read_member_payload(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
            padded_size = ((member.size + 511) // 512) * 512
            payload = archive.fileobj.read(padded_size)
            return payload[: member.size]

        try:
            with gzip.open(archive_path, "rb") as compressed_stream:
                with tarfile.open(fileobj=compressed_stream, mode="r|") as archive:
                    for member in archive:
                        if not member.isfile():
                            continue
                        if member.name == "mcspeech/transcripts.tsv":
                            payload = _read_member_payload(archive, member)
                            reader = csv.DictReader(io.StringIO(payload.decode("utf-8")), delimiter="\t")
                            transcripts = {
                                str(row.get("id") or "").strip(): str(row.get("transcript") or "").strip()
                                for row in reader
                                if str(row.get("id") or "").strip() and str(row.get("transcript") or "").strip()
                            }
                            continue
                        if not member.name.startswith("mcspeech/wavs/") or not transcripts:
                            continue

                        sample_id = Path(member.name).stem
                        transcript = transcripts.get(sample_id)
                        if not transcript:
                            _read_member_payload(archive, member)
                            continue

                        audio_bytes = _read_member_payload(archive, member)
                        relative_name = f"{sample_id}.wav"
                        collected.append(
                            {
                                "id": sample_id,
                                "path": relative_name,
                                "audio": {"path": relative_name, "bytes": audio_bytes},
                                "transcription": transcript,
                                "raw_transcription": transcript,
                            }
                        )
                        if len(collected) >= sample_limit:
                            break
        except EOFError:
            pass
        return collected

    def _stream_rows(source_url: str) -> list[dict[str, object]]:
        partial_path.unlink(missing_ok=True)
        last_collected: list[dict[str, object]] = []
        while True:
            start = partial_path.stat().st_size if partial_path.exists() else 0
            if start >= max_download_bytes:
                break
            end = start + chunk_size - 1
            with partial_path.open("ab") as handle:
                subprocess.run(
                    [
                        curl_path,
                        "--fail",
                        "--location",
                        "--retry",
                        "5",
                        "--retry-delay",
                        "2",
                        "--connect-timeout",
                        "15",
                        "--max-time",
                        "0",
                        "--silent",
                        "--show-error",
                        "--range",
                        f"{start}-{end}",
                        source_url,
                    ],
                    check=True,
                    stdout=handle,
                )

            last_collected = _extract_rows_from_partial_archive(partial_path)
            if len(last_collected) >= sample_limit:
                return last_collected

        raise RuntimeError(f"mcspeech_pl only collected {len(last_collected)} samples within {max_download_bytes / 1024 / 1024:.0f} MiB")

    last_error: Exception | None = None
    for source_url in MCSPEECH_PL_MIRRORS:
        try:
            rows = _stream_rows(source_url)
            features = Features(
                {
                    "id": Value("string"),
                    "path": Value("string"),
                    "audio": {
                        "path": Value("string"),
                        "bytes": Value("binary"),
                    },
                    "transcription": Value("string"),
                    "raw_transcription": Value("string"),
                }
            )
            dataset = Dataset.from_list(rows, features=features)
            if out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)
            dataset.save_to_disk(str(out_dir))
            if os.environ.get("SLAI_KEEP_MCSPEECH_PARTIAL", "0") != "1":
                partial_path.unlink(missing_ok=True)
            return
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(f"mcspeech_pl download failed: {last_error}")


def _download_fleurs_dataset(out_dir: Path, cache_dir: Path, config: str, split: str, token: Optional[str] = None) -> None:
    import json
    import math
    import subprocess
    import time
    from urllib import request
    from urllib.parse import quote

    from datasets import Dataset, Features
    from datasets.features import Value

    repo_id = "google/fleurs"
    hub_cache = cache_dir / "hub"
    curl_path = shutil.which("curl")

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

    def _download_with_retries(filename: str) -> str:
        last_error: Exception | None = None
        download_timeout_text = str(os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT") or "120").strip()
        try:
            download_timeout = max(30, int(download_timeout_text))
        except Exception:
            download_timeout = 120

        cache_path = hub_cache / "direct_fleurs" / filename
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return str(cache_path)

        configured_endpoint = _normalize_hf_endpoint(os.environ.get("HF_ENDPOINT")) or _normalize_hf_endpoint(HF_OFFICIAL_ENDPOINT)
        official_endpoint = _normalize_hf_endpoint(HF_OFFICIAL_ENDPOINT)
        endpoint_plan = [configured_endpoint]
        if configured_endpoint != official_endpoint:
            endpoint_plan.append(official_endpoint)

        relative_url = f"/datasets/{repo_id}/resolve/main/{filename}"
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        headers = {"User-Agent": "slai-download-datasets/1.0"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        curl_path = shutil.which("curl")

        for endpoint_index, endpoint in enumerate(endpoint_plan, start=1):
            endpoint_label = "official" if endpoint == official_endpoint else "current"
            url = f"{endpoint}{relative_url}"
            if curl_path:
                cmd = [
                    curl_path,
                    "--fail",
                    "--location",
                    "--retry",
                    "5",
                    "--retry-delay",
                    "2",
                    "--connect-timeout",
                    "15",
                    "--max-time",
                    "0",
                    "--continue-at",
                    "-",
                    "--output",
                    str(tmp_path),
                    url,
                ]
                if token:
                    cmd[1:1] = ["--header", f"Authorization: Bearer {token}"]
                try:
                    subprocess.run(cmd, check=True)
                    tmp_path.replace(cache_path)
                    return str(cache_path)
                except subprocess.CalledProcessError as exc:
                    last_error = exc
                    if endpoint_index < len(endpoint_plan):
                        _safe_print(f"    [retry] fleurs download failed: {filename} via {endpoint_label} endpoint (curl): {exc}")
                        _safe_print(f"    [fallback] fleurs download switching HF endpoint to {HF_OFFICIAL_ENDPOINT} for {filename}")
                        continue
            for attempt in range(1, 4):
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                    req = request.Request(url, headers=headers)
                    with request.urlopen(req, timeout=download_timeout) as response, tmp_path.open("wb") as handle:
                        shutil.copyfileobj(response, handle, length=1024 * 1024)
                    tmp_path.replace(cache_path)
                    return str(cache_path)
                except Exception as exc:
                    last_error = exc
                    if tmp_path.exists():
                        tmp_path.unlink()
                    if attempt >= 3:
                        break
                    _safe_print(f"    [retry] fleurs download failed: {filename} via {endpoint_label} endpoint (attempt {attempt}/3): {exc}")
                    time.sleep(min(5 * attempt, 15))
            if endpoint_index < len(endpoint_plan):
                _safe_print(f"    [fallback] fleurs download switching HF endpoint to {HF_OFFICIAL_ENDPOINT} for {filename}")
        raise RuntimeError(f"下载 FLEURS 文件失败: {filename}: {last_error}")

    def _parse_rows_by_filename(lines: list[str]) -> dict[str, dict[str, object]]:
        rows_by_filename: dict[str, dict[str, object]] = {}
        for line in lines:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 7:
                continue
            sample_id, filename, raw_transcription, transcription, _unused, num_samples, gender = parts
            try:
                parsed_sample_id = int(sample_id)
            except Exception:
                continue
            try:
                parsed_num_samples = int(num_samples)
            except Exception:
                parsed_num_samples = 0
            rows_by_filename[filename] = {
                "id": parsed_sample_id,
                "num_samples": parsed_num_samples,
                "raw_transcription": raw_transcription,
                "transcription": transcription,
                "gender": gender,
                "lang_id": config,
                "language": config,
                "lang_group_id": "",
            }
        return rows_by_filename

    def _rows_from_rows_by_filename(rows_by_filename: dict[str, dict[str, object]], archive_path: Path, *, audio_prefix: Optional[str] = None) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                if audio_prefix and not member.name.startswith(audio_prefix):
                    continue
                audio_filename = Path(member.name).name
                if audio_filename not in rows_by_filename:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                row = dict(rows_by_filename[audio_filename])
                row["path"] = audio_filename
                row["audio"] = {"path": audio_filename, "bytes": extracted.read()}
                rows.append(row)
        return rows

    def _legacy_archive_metadata(config_name: str) -> tuple[str, int]:
        if not curl_path:
            raise RuntimeError("legacy FLEURS GCS fallback requires curl")
        archive_name = f"FLEURS102/{config_name}.tar.gz"
        meta_url = f"https://storage.googleapis.com/storage/v1/b/xtreme_translations/o/{quote(archive_name, safe='')}?fields=name,size"
        result = subprocess.run(
            [curl_path, "--fail", "--location", "--silent", "--show-error", meta_url],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        return str(payload["name"]), int(payload["size"])

    def _download_legacy_archive(config_name: str) -> Path:
        legacy_cache_dir = hub_cache / "legacy_fleurs"
        legacy_cache_dir.mkdir(parents=True, exist_ok=True)
        legacy_archive_path = legacy_cache_dir / f"{config_name}.tar.gz"
        archive_name, archive_size = _legacy_archive_metadata(config_name)
        legacy_url = f"https://storage.googleapis.com/xtreme_translations/{archive_name}"
        if legacy_archive_path.exists() and legacy_archive_path.stat().st_size == archive_size:
            return legacy_archive_path

        workers_text = str(os.environ.get("SLAI_FLEURS_GCS_WORKERS") or "1").strip()
        try:
            workers = max(1, int(workers_text))
        except Exception:
            workers = 1
        _safe_print(
            f"    [fallback] downloading legacy FLEURS archive via GCS: {archive_name} ({archive_size / 1024 / 1024:.1f} MiB, workers={workers})"
        )

        temp_path = legacy_archive_path.with_suffix(legacy_archive_path.suffix + ".tmp")
        parts_dir = legacy_cache_dir / f".{config_name}.parts"
        shutil.rmtree(parts_dir, ignore_errors=True)
        parts_dir.mkdir(parents=True, exist_ok=True)
        try:
            if temp_path.exists() and temp_path.stat().st_size == archive_size:
                temp_path.replace(legacy_archive_path)
                return legacy_archive_path

            if workers <= 1:
                subprocess.run(
                    [
                        curl_path,
                        "--fail",
                        "--location",
                        "--retry",
                        "5",
                        "--retry-delay",
                        "2",
                        "--connect-timeout",
                        "15",
                        "--max-time",
                        "0",
                        "--continue-at",
                        "-",
                        "--output",
                        str(temp_path),
                        legacy_url,
                    ],
                    check=True,
                )
            else:
                chunk_size = max(16 * 1024 * 1024, math.ceil(archive_size / workers))

                def _download_chunk(index: int) -> Path:
                    start = index * chunk_size
                    end = min(archive_size - 1, ((index + 1) * chunk_size) - 1)
                    part_path = parts_dir / f"{index:04d}.part"
                    if start > end:
                        part_path.write_bytes(b"")
                        return part_path
                    subprocess.run(
                        [
                            curl_path,
                            "--fail",
                            "--location",
                            "--retry",
                            "5",
                            "--retry-delay",
                            "2",
                            "--connect-timeout",
                            "15",
                            "--silent",
                            "--show-error",
                            "--output",
                            str(part_path),
                            "--range",
                            f"{start}-{end}",
                            legacy_url,
                        ],
                        check=True,
                    )
                    return part_path

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(_download_chunk, index) for index in range(workers)]
                    for future in as_completed(futures):
                        future.result()

                with temp_path.open("wb") as merged:
                    for index in range(workers):
                        part_path = parts_dir / f"{index:04d}.part"
                        with part_path.open("rb") as part_file:
                            shutil.copyfileobj(part_file, merged, length=1024 * 1024)

            temp_path.replace(legacy_archive_path)
        finally:
            if workers > 1 and temp_path.exists() and not legacy_archive_path.exists():
                temp_path.unlink(missing_ok=True)
            shutil.rmtree(parts_dir, ignore_errors=True)
        return legacy_archive_path

    def _rows_from_legacy_archive(config_name: str, split_name: str) -> tuple[list[dict[str, object]], Path]:
        legacy_archive_path = _download_legacy_archive(config_name)
        split_tsv_path = f"{config_name}/{split_name}.tsv"
        audio_prefix = f"{config_name}/audio/{split_name}/"

        with tarfile.open(legacy_archive_path, "r:gz") as archive:
            split_member = archive.getmember(split_tsv_path)
            extracted = archive.extractfile(split_member)
            if extracted is None:
                raise RuntimeError(f"legacy FLEURS archive missing {split_tsv_path}")
            text_lines = io.TextIOWrapper(extracted, encoding="utf-8").read().splitlines()

        rows_by_filename = _parse_rows_by_filename(text_lines)
        rows = _rows_from_rows_by_filename(rows_by_filename, legacy_archive_path, audio_prefix=audio_prefix)
        if not rows:
            raise RuntimeError(f"legacy FLEURS archive missing audio members for {audio_prefix}")
        return rows, legacy_archive_path

    legacy_archive_path: Optional[Path] = None
    try:
        metadata_path = Path(_download_with_retries(f"data/{config}/{split}.tsv"))
        archive_path = Path(_download_with_retries(f"data/{config}/audio/{split}.tar.gz"))
        rows_by_filename = _parse_rows_by_filename(metadata_path.read_text(encoding="utf-8").splitlines())
        rows = _rows_from_rows_by_filename(rows_by_filename, archive_path)
        if not rows:
            raise RuntimeError(f"下载的 FLEURS 音频包为空: {archive_path}")
    except Exception as exc:
        _safe_print(f"    [fallback] HF-hosted FLEURS download unavailable for {config}: {exc}")
        rows, legacy_archive_path = _rows_from_legacy_archive(config, split)

    rows.sort(key=lambda item: int(item["id"]))
    features = Features(
        {
            "id": Value("int32"),
            "num_samples": Value("int32"),
            "path": Value("string"),
            "audio": {
                "path": Value("string"),
                "bytes": Value("binary"),
            },
            "transcription": Value("string"),
            "raw_transcription": Value("string"),
            "gender": Value("string"),
            "lang_id": Value("string"),
            "language": Value("string"),
            "lang_group_id": Value("string"),
        }
    )
    dataset = Dataset.from_list(rows, features=features)
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    dataset.save_to_disk(str(out_dir))
    if legacy_archive_path is not None and os.environ.get("SLAI_KEEP_FLEURS_GCS_ARCHIVE", "0") != "1":
        legacy_archive_path.unlink(missing_ok=True)


def _render_synthetic_keypoint_png(seed: int, *, height: int = 120, width: int = 160) -> bytes:
    from PIL import Image
    import numpy as np

    rng = np.random.default_rng(seed)
    image = np.zeros((height, width), dtype=np.uint8)

    for _ in range(3):
        x1 = int(rng.integers(10, max(width - 30, 11)))
        y1 = int(rng.integers(10, max(height - 30, 11)))
        rect_w = int(rng.integers(10, 30))
        rect_h = int(rng.integers(10, 30))
        intensity = int(rng.integers(100, 255))
        image[y1 : y1 + rect_h, x1 : x1 + rect_w] = intensity

    corner_size = 15
    image[10 : 10 + corner_size, 10 : 10 + corner_size] = 255
    image[10 : 10 + corner_size, width - corner_size - 10 : width - 10] = 200
    image[height - corner_size - 10 : height - 10, 10 : 10 + corner_size] = 180
    image[height - corner_size - 10 : height - 10, width - corner_size - 10 : width - 10] = 220

    buffer = io.BytesIO()
    Image.fromarray(image, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def _download_synthetic_keypoints_dataset(out_dir: Path) -> None:
    from datasets import Dataset, Features, Value

    columns = {
        "img_bytes": [],
        "sample_id": [],
        "seed": [],
    }
    for sample_id in range(64):
        seed = 42 + sample_id
        columns["img_bytes"].append(_render_synthetic_keypoint_png(seed))
        columns["sample_id"].append(sample_id)
        columns["seed"].append(seed)

    features = Features(
        {
            "img_bytes": Value("binary"),
            "sample_id": Value("int32"),
            "seed": Value("int32"),
        }
    )
    dataset = Dataset.from_dict(columns, features=features)
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    dataset.save_to_disk(str(out_dir))


def _render_synthetic_matting_triplet(seed: int, *, height: int = 224, width: int = 224) -> tuple[bytes, bytes, bytes]:
    from PIL import Image
    import numpy as np

    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)

    center_y = int(height * (0.35 + 0.3 * rng.random()))
    center_x = int(width * (0.35 + 0.3 * rng.random()))
    radius_y = int(height * (0.18 + 0.1 * rng.random()))
    radius_x = int(width * (0.18 + 0.1 * rng.random()))
    feather = max(int(min(height, width) * 0.08), 6)

    yy, xx = np.ogrid[:height, :width]
    ellipse_distance = ((yy - center_y) / max(radius_y, 1)) ** 2 + ((xx - center_x) / max(radius_x, 1)) ** 2
    alpha = np.clip(1.0 - (ellipse_distance - 1.0) / max(feather / max(radius_x, radius_y, 1), 1e-3), 0.0, 1.0).astype(np.float32)

    trimap = np.full((height, width), 128, dtype=np.uint8)
    trimap[alpha >= 0.95] = 255
    trimap[alpha <= 0.05] = 0

    alpha_u8 = np.clip(np.round(alpha * 255.0), 0, 255).astype(np.uint8)

    def _encode_png(array, mode: str) -> bytes:
        buffer = io.BytesIO()
        Image.fromarray(array, mode=mode).save(buffer, format="PNG")
        return buffer.getvalue()

    return _encode_png(image, "RGB"), _encode_png(trimap, "L"), _encode_png(alpha_u8, "L")


def _download_synthetic_matting_dataset(out_dir: Path) -> None:
    from datasets import Dataset, Features, Value

    columns = {
        "image_bytes": [],
        "trimap_bytes": [],
        "alpha_bytes": [],
        "sample_id": [],
        "seed": [],
    }
    for sample_id in range(64):
        seed = 2048 + sample_id
        image_bytes, trimap_bytes, alpha_bytes = _render_synthetic_matting_triplet(seed)
        columns["image_bytes"].append(image_bytes)
        columns["trimap_bytes"].append(trimap_bytes)
        columns["alpha_bytes"].append(alpha_bytes)
        columns["sample_id"].append(sample_id)
        columns["seed"].append(seed)

    features = Features(
        {
            "image_bytes": Value("binary"),
            "trimap_bytes": Value("binary"),
            "alpha_bytes": Value("binary"),
            "sample_id": Value("int32"),
            "seed": Value("int32"),
        }
    )
    dataset = Dataset.from_dict(columns, features=features)
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    dataset.save_to_disk(str(out_dir))


def _download_synthetic_protein_dataset(out_dir: Path) -> None:
    from datasets import Dataset, Features, Value
    import random

    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    def _random_sequence(rng: random.Random, length: int = 64) -> str:
        return "".join(rng.choices(amino_acids, k=length))

    def _mutate_once(rng: random.Random, sequence: str) -> str:
        chars = list(sequence)
        if not chars:
            return sequence
        index = rng.randrange(len(chars))
        original = chars[index]
        choices = [acid for acid in amino_acids if acid != original]
        chars[index] = rng.choice(choices)
        return "".join(chars)

    columns = {
        "sequence": [],
        "reference_sequence": [],
        "sample_id": [],
        "seed": [],
    }
    for sample_id in range(64):
        seed = 8192 + sample_id
        rng = random.Random(seed)
        sequence = _random_sequence(rng)
        columns["sequence"].append(sequence)
        columns["reference_sequence"].append(_mutate_once(rng, sequence))
        columns["sample_id"].append(sample_id)
        columns["seed"].append(seed)

    features = Features(
        {
            "sequence": Value("string"),
            "reference_sequence": Value("string"),
            "sample_id": Value("int32"),
            "seed": Value("int32"),
        }
    )
    dataset = Dataset.from_dict(columns, features=features)
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    dataset.save_to_disk(str(out_dir))


def _download_ncbi_disease_dataset(out_dir: Path) -> None:
    from datasets import Dataset, Features
    from datasets.features import ClassLabel, Sequence, Value
    import ssl
    import subprocess
    import tempfile
    import time
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    # 第四阶段当前只消费 validation split，避免在弱网环境中额外下载 train/test。
    split_urls = {
        "validation": [
            "https://cdn.jsdelivr.net/gh/spyysalo/ncbi-disease@master/conll/devel.tsv",
            "https://raw.githubusercontent.com/spyysalo/ncbi-disease/master/conll/devel.tsv",
            "https://media.githubusercontent.com/media/spyysalo/ncbi-disease/master/conll/devel.tsv",
            "https://github.com/spyysalo/ncbi-disease/raw/master/conll/devel.tsv",
        ],
    }

    columns = {
        "id": [],
        "tokens": [],
        "ner_tags": [],
        "split": [],
    }

    def _flush(tokens: list[str], ner_tags: list[str], split_name: str) -> None:
        if not tokens:
            return
        columns["id"].append(str(len(columns["id"])))
        columns["tokens"].append(list(tokens))
        columns["ner_tags"].append(list(ner_tags))
        columns["split"].append(split_name)

    def _create_ssl_context() -> ssl.SSLContext:
        try:
            import certifi

            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            return ssl._create_unverified_context()

    def _fetch_text(urls: list[str]) -> str:
        errors: list[str] = []
        headers = {"User-Agent": "slai-ascendbridge2/1.0"}
        curl_path = shutil.which("curl")
        for url in urls:
            if curl_path:
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(prefix="ncbi_disease_", suffix=".tsv", delete=False) as temp_file:
                        temp_path = temp_file.name
                    _safe_print(f"    [fetch] ncbi_disease via curl: {url}")
                    result = subprocess.run(
                        [
                            curl_path,
                            "--fail",
                            "--location",
                            "--retry",
                            "3",
                            "--retry-delay",
                            "2",
                            "--connect-timeout",
                            "10",
                            "--max-time",
                            "600",
                            "--progress-bar",
                            "--output",
                            temp_path,
                            url,
                        ],
                        check=True,
                    )
                    _ = result
                    return Path(temp_path).read_text(encoding="utf-8")
                except subprocess.CalledProcessError as exc:
                    message = str(exc).strip()
                    _safe_print(f"    [retry] ncbi_disease curl failed: {url}: {message}")
                    errors.append(f"{url} (curl): {message}")
                finally:
                    if temp_path and Path(temp_path).exists():
                        Path(temp_path).unlink(missing_ok=True)
            for attempt in range(1, 3):
                try:
                    request = Request(url, headers=headers)
                    with urlopen(request, timeout=25, context=_create_ssl_context()) as response:
                        return response.read().decode("utf-8")
                except (HTTPError, URLError, TimeoutError, ConnectionError, OSError) as exc:
                    _safe_print(f"    [retry] ncbi_disease fetch failed: {url} (attempt {attempt}/2): {exc}")
                    errors.append(f"{url} (attempt {attempt}/2): {exc}")
                    time.sleep(min(2 * attempt, 6))
        raise RuntimeError("; ".join(errors[-6:]) or "unknown ncbi_disease download error")

    for split_name, urls in split_urls.items():
        text = _fetch_text(urls)
        tokens: list[str] = []
        ner_tags: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                _flush(tokens, ner_tags, split_name)
                tokens = []
                ner_tags = []
                continue
            parts = raw_line.split("\t")
            if len(parts) < 2:
                continue
            tokens.append(str(parts[0]).strip())
            ner_tags.append(str(parts[1]).strip())
        _flush(tokens, ner_tags, split_name)

    features = Features(
        {
            "id": Value("string"),
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=["O", "B-Disease", "I-Disease"])),
            "split": Value("string"),
        }
    )
    dataset = Dataset.from_dict(columns, features=features)
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    dataset.save_to_disk(str(out_dir))


def _download_bionlp2004_dataset(out_dir: Path) -> None:
    from datasets import Dataset, Features
    from datasets.features import ClassLabel, Sequence, Value
    import ssl
    import subprocess
    import tempfile
    import time
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    split_urls = {
        "test": [
            "https://cdn.jsdelivr.net/gh/cambridgeltl/MTL-Bioinformatics-2016@master/data/JNLPBA/test.tsv",
            "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/JNLPBA/test.tsv",
            "https://media.githubusercontent.com/media/cambridgeltl/MTL-Bioinformatics-2016/master/data/JNLPBA/test.tsv",
            "https://github.com/cambridgeltl/MTL-Bioinformatics-2016/raw/master/data/JNLPBA/test.tsv",
        ],
    }
    label_names = [
        "O",
        "B-protein",
        "I-protein",
        "B-cell_type",
        "I-cell_type",
        "B-cell_line",
        "I-cell_line",
        "B-DNA",
        "I-DNA",
        "B-RNA",
        "I-RNA",
    ]

    columns = {
        "id": [],
        "tokens": [],
        "ner_tags": [],
        "split": [],
    }

    def _flush(tokens: list[str], ner_tags: list[str], split_name: str) -> None:
        if not tokens:
            return
        columns["id"].append(str(len(columns["id"])))
        columns["tokens"].append(list(tokens))
        columns["ner_tags"].append(list(ner_tags))
        columns["split"].append(split_name)

    def _create_ssl_context() -> ssl.SSLContext:
        try:
            import certifi

            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            return ssl._create_unverified_context()

    def _fetch_text(urls: list[str]) -> str:
        errors: list[str] = []
        headers = {"User-Agent": "slai-ascendbridge2/1.0"}
        curl_path = shutil.which("curl")
        for url in urls:
            if curl_path:
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(prefix="bionlp2004_", suffix=".tsv", delete=False) as temp_file:
                        temp_path = temp_file.name
                    _safe_print(f"    [fetch] bionlp2004 via curl: {url}")
                    result = subprocess.run(
                        [
                            curl_path,
                            "--fail",
                            "--location",
                            "--retry",
                            "3",
                            "--retry-delay",
                            "2",
                            "--connect-timeout",
                            "10",
                            "--max-time",
                            "600",
                            "--progress-bar",
                            "--output",
                            temp_path,
                            url,
                        ],
                        check=True,
                    )
                    _ = result
                    return Path(temp_path).read_text(encoding="utf-8")
                except subprocess.CalledProcessError as exc:
                    message = str(exc).strip()
                    _safe_print(f"    [retry] bionlp2004 curl failed: {url}: {message}")
                    errors.append(f"{url} (curl): {message}")
                finally:
                    if temp_path and Path(temp_path).exists():
                        Path(temp_path).unlink(missing_ok=True)
            for attempt in range(1, 3):
                try:
                    request = Request(url, headers=headers)
                    with urlopen(request, timeout=25, context=_create_ssl_context()) as response:
                        return response.read().decode("utf-8")
                except (HTTPError, URLError, TimeoutError, ConnectionError, OSError) as exc:
                    _safe_print(f"    [retry] bionlp2004 fetch failed: {url} (attempt {attempt}/2): {exc}")
                    errors.append(f"{url} (attempt {attempt}/2): {exc}")
                    time.sleep(min(2 * attempt, 6))
        raise RuntimeError("; ".join(errors[-6:]) or "unknown bionlp2004 download error")

    for split_name, urls in split_urls.items():
        text = _fetch_text(urls)
        tokens: list[str] = []
        ner_tags: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                _flush(tokens, ner_tags, split_name)
                tokens = []
                ner_tags = []
                continue
            parts = raw_line.split("\t")
            if len(parts) < 2:
                continue
            tokens.append(str(parts[0]).strip())
            ner_tags.append(str(parts[1]).strip())
        _flush(tokens, ner_tags, split_name)

    features = Features(
        {
            "id": Value("string"),
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=label_names)),
            "split": Value("string"),
        }
    )
    dataset = Dataset.from_dict(columns, features=features)
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    dataset.save_to_disk(str(out_dir))


def normalize_dataset_key(key: str) -> str:
    return key.lower().replace("-", "_").strip()


def get_dataset_output_dir_name(key: str) -> str:
    normalized = normalize_dataset_key(key)
    if normalized not in DATASET_MAP:
        raise KeyError(f"未知数据集: {key}")
    hf_path, config, _ = DATASET_MAP[normalized]
    base = hf_path.replace("/", "___")
    if config:
        base = f"{base}___{config}"
    return base


def get_dataset_disk_path(key: str, cache_dir: Optional[Path] = None) -> Path:
    return (cache_dir or _DATASETS_DIR) / get_dataset_output_dir_name(key)


def _get_split_files(
    hf_path: str,
    split: str,
    config: Optional[str],
    repo_type: str = "dataset",
    token: Optional[str] = None,
) -> List[str]:
    """列出目标 split 的 parquet 文件路径，排除 train"""
    from huggingface_hub import HfApi, list_repo_files

    repo_id = _repo_id(hf_path)
    files: List[str] = []
    if config:
        try:
            api = HfApi()
            files = [str(getattr(item, "path", "") or "") for item in api.list_repo_tree(repo_id, path_in_repo=config, repo_type=repo_type, recursive=True, token=token)]
        except Exception:
            files = []
    if not files:
        files = list_repo_files(repo_id, repo_type=repo_type, token=token)
    # 只要目标 split，显式排除 train
    candidates = [f for f in files if f.endswith(".parquet") and split in f and "/train" not in f and "train-" not in f.split("/")[-1]]
    if config:
        config_candidates = [f for f in candidates if f.startswith(f"{config}/") or f"/{config}/" in f]
        if config_candidates:
            return sorted(config_candidates)
    return sorted(candidates)


def download_one(
    key: str,
    cache_dir: Path,
    jobs: int = 8,
    index: Optional[tuple] = None,
    skip_existing: bool = True,
) -> str:
    """下载单个数据集，多线程并行下载文件。返回 'ok' | 'skip' | 'fail'"""
    if key not in DATASET_MAP:
        _safe_print(f"  [skip] 未知数据集: {key}")
        return "skip"

    idx_str = f" [{index[0]}/{index[1]}]" if index else ""
    hf_path, config, split = DATASET_MAP[key]
    token = os.environ.get("HF_TOKEN")

    def _out_dir() -> Path:
        base = hf_path.replace("/", "___")
        if config:
            base = f"{base}___{config}"
        return cache_dir / base

    out_dir = _out_dir()
    if skip_existing and out_dir.exists():
        _safe_print(f"  [skip]{idx_str} {key} (已存在: {out_dir.name})")
        return "skip"

    if key == "fairface":
        try:
            _safe_print(f"  [downloading]{idx_str} {key} (nateraw/fairface, split=validation)...")
            _download_fairface_dataset(out_dir, cache_dir, token=token)
            _safe_print(f"  [ok] {key}")
            return "ok"
        except Exception as e:
            _safe_print(f"  [fail]{idx_str} {key}: {e}")
            return "fail"

    if key == "synthetic_keypoints":
        try:
            _safe_print(f"  [building]{idx_str} {key} (synthetic repeatability dataset)...")
            _download_synthetic_keypoints_dataset(out_dir)
            _safe_print(f"  [ok] {key}")
            return "ok"
        except Exception as e:
            _safe_print(f"  [fail]{idx_str} {key}: {e}")
            return "fail"

    if key == "synthetic_matting":
        try:
            _safe_print(f"  [building]{idx_str} {key} (synthetic alpha-matting dataset)...")
            _download_synthetic_matting_dataset(out_dir)
            _safe_print(f"  [ok] {key}")
            return "ok"
        except Exception as e:
            _safe_print(f"  [fail]{idx_str} {key}: {e}")
            return "fail"

    if key == "synthetic_protein":
        try:
            _safe_print(f"  [building]{idx_str} {key} (synthetic protein similarity dataset)...")
            _download_synthetic_protein_dataset(out_dir)
            _safe_print(f"  [ok] {key}")
            return "ok"
        except Exception as e:
            _safe_print(f"  [fail]{idx_str} {key}: {e}")
            return "fail"

    if key == "ncbi_disease":
        try:
            _safe_print(f"  [downloading]{idx_str} {key} (ncbi disease validation split)...")
            _download_ncbi_disease_dataset(out_dir)
            _safe_print(f"  [ok] {key}")
            return "ok"
        except Exception as e:
            _safe_print(f"  [fail]{idx_str} {key}: {e}")
            return "fail"

    if key == "bionlp2004":
        try:
            _safe_print(f"  [downloading]{idx_str} {key} (BioNLP2004/JNLPBA test split)...")
            _download_bionlp2004_dataset(out_dir)
            _safe_print(f"  [ok] {key}")
            return "ok"
        except Exception as e:
            _safe_print(f"  [fail]{idx_str} {key}: {e}")
            return "fail"

    if key == "mcspeech_pl":
        try:
            _safe_print(f"  [downloading]{idx_str} {key} (OpenSLR 142 stream-extract)...")
            _download_mcspeech_pl_dataset(out_dir)
            _safe_print(f"  [ok] {key}")
            return "ok"
        except Exception as e:
            _safe_print(f"  [fail]{idx_str} {key}: {e}")
            return "fail"

    if key.startswith("fleurs_") and config:
        try:
            _safe_print(f"  [downloading]{idx_str} {key} (google/fleurs/{config}, split={split})...")
            _download_fleurs_dataset(out_dir, cache_dir, config, split, token=token)
            _safe_print(f"  [ok] {key}")
            return "ok"
        except Exception as e:
            _safe_print(f"  [fail]{idx_str} {key}: {e}")
            return "fail"

    try:
        from datasets import Dataset, concatenate_datasets, load_dataset

        try:
            split_files = _get_split_files(hf_path, split, config, token=token)
        except Exception:
            split_files = []

        if split_files:
            try:
                # 并行下载 parquet 到 hub 缓存
                from huggingface_hub import hf_hub_download

                repo_id = _repo_id(hf_path)
                hub_cache = str(cache_dir / "hub")

                def _dl(f: str) -> str:
                    return hf_hub_download(
                        repo_id=repo_id,
                        filename=f,
                        repo_type="dataset",
                        cache_dir=hub_cache,
                        token=token,
                    )

                _safe_print(f"  [downloading]{idx_str} {key} ({len(split_files)} 个文件, {jobs} 线程)...")
                paths = []
                with ThreadPoolExecutor(max_workers=jobs) as ex:
                    for future in as_completed({ex.submit(_dl, f): f for f in split_files}):
                        paths.append(future.result())

                # 仅目标 split，不拉 train
                order = {Path(p).name: i for i, p in enumerate(split_files)}
                paths.sort(key=lambda x: order.get(Path(x).name, 999))
                # 共享盘/NFS 上 `load_dataset("parquet", ...)` 会偶发 filelock 释放失败，
                # 这里直接从 parquet 构建 Dataset，避免 builder/cache 锁路径抖动。
                parquet_datasets = [Dataset.from_parquet(path) for path in paths]
                out_ds = parquet_datasets[0] if len(parquet_datasets) == 1 else concatenate_datasets(parquet_datasets)
                out_dir = _out_dir()
                if out_dir.exists():
                    shutil.rmtree(out_dir, ignore_errors=True)
                out_ds.save_to_disk(str(out_dir))
                # 清理 parquet 缓存，避免重复占用
                parquet_cache = cache_dir / "parquet"
                if parquet_cache.exists():
                    shutil.rmtree(parquet_cache, ignore_errors=True)
                _safe_print(f"  [ok] {key}")
                return "ok"
            except Exception as e:
                _safe_print(f"  [fallback]{idx_str} {key}: 并行下载失败 ({e})，改用 load_dataset...")

        # 回退：load_dataset 直接下载（返回单 split 的 Dataset）
        _safe_print(f"  [downloading]{idx_str} {key} ({hf_path}, split={split})...")
        if config:
            ds = load_dataset(hf_path, name=config, split=split, cache_dir=str(cache_dir))
        else:
            ds = load_dataset(hf_path, split=split, cache_dir=str(cache_dir))
        out_ds = ds[split] if split in ds else ds
        out_dir = _out_dir()
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        cast(Dataset, out_ds).save_to_disk(str(out_dir))
        _safe_print(f"  [ok] {key}")
        return "ok"
    except Exception as e:
        _safe_print(f"  [fail]{idx_str} {key}: {e}")
        return "fail"


def ensure_dataset(
    key: str,
    cache_dir: Optional[Path] = None,
    *,
    jobs: int = 8,
    skip_existing: bool = True,
) -> Path:
    base_dir = cache_dir or _DATASETS_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    normalized = normalize_dataset_key(key)
    status = download_one(normalized, base_dir, jobs=jobs, skip_existing=skip_existing)
    if status not in {"ok", "skip"}:
        raise RuntimeError(f"下载数据集失败: {normalized}")
    return get_dataset_disk_path(normalized, cache_dir=base_dir)


def ensure_business_dataset_for_model(
    model_id: str,
    model_class: str = "",
    cache_dir: Optional[Path] = None,
    *,
    jobs: int = 8,
    skip_existing: bool = True,
) -> dict:
    profile = get_business_benchmark_profile(model_id, model_class)
    dataset_key = profile.get("dataset_key")
    if not dataset_key:
        profile["dataset_path"] = None
        profile["download_status"] = "not_required"
        return profile
    dataset_path = get_dataset_disk_path(str(dataset_key), cache_dir=cache_dir or _DATASETS_DIR)
    existed = dataset_path.exists()
    if not existed or not skip_existing:
        dataset_path = ensure_dataset(str(dataset_key), cache_dir=cache_dir, jobs=jobs, skip_existing=skip_existing)
    profile["dataset_key"] = normalize_dataset_key(str(dataset_key))
    profile["dataset_path"] = str(dataset_path)
    profile["download_status"] = "skip" if existed and skip_existing else "ok"
    return profile


def ensure_candidate_datasets_for_model(
    model_id: str,
    model_class: str = "",
    cache_dir: Optional[Path] = None,
    *,
    jobs: int = 8,
    skip_existing: bool = True,
) -> dict:
    candidate_keys = get_dataset_candidates_for_model(model_id, model_class)
    base_dir = cache_dir or _DATASETS_DIR
    downloaded: list[dict[str, str]] = []
    for dataset_key in candidate_keys:
        normalized = normalize_dataset_key(dataset_key)
        if normalized not in DATASET_MAP:
            downloaded.append({"dataset_key": normalized, "status": "unsupported"})
            continue
        dataset_path = get_dataset_disk_path(normalized, cache_dir=base_dir)
        existed = dataset_path.exists()
        if not existed or not skip_existing:
            dataset_path = ensure_dataset(normalized, cache_dir=base_dir, jobs=jobs, skip_existing=skip_existing)
        downloaded.append(
            {
                "dataset_key": normalized,
                "status": "skip" if existed and skip_existing else "ok",
                "dataset_path": str(dataset_path),
            }
        )
    return {
        "model_id": model_id,
        "model_class": model_class or None,
        "dataset_candidates": candidate_keys,
        "downloads": downloaded,
    }


def main():
    _load_dotenv()

    # 确保显示下载进度（默认 WARNING 可能抑制 tqdm）
    os.environ.setdefault("DATASETS_VERBOSITY", "info")

    parser = argparse.ArgumentParser(description="下载测试数据集到 datasets/")
    parser.add_argument(
        "datasets",
        nargs="*",
        help="要下载的数据集 key，不指定则下载全部",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出可下载的数据集",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="使用国内镜像 (hf-mirror.com) 加速下载",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=8,
        metavar="N",
        help="单数据集内并行下载文件线程数（默认 8）",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="不跳过已存在的数据集，强制重新下载",
    )
    parser.add_argument("--model-id", help="按模型自动推断业务测评数据集")
    parser.add_argument("--model-class", default="", help="AutoModel 类名（配合 --model-id 使用）")
    parser.add_argument("--business-profile", action="store_true", help="配合 --model-id 使用，自动选择第四阶段业务数据集并确保已下载")
    parser.add_argument("--candidate-datasets", action="store_true", help="配合 --model-id 使用，下载该模型的候选数据集列表")
    args = parser.parse_args()

    if args.mirror:
        os.environ["HF_ENDPOINT"] = HF_MIRROR
        print("使用镜像: " + HF_MIRROR)

    cache_dir = _DATASETS_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "hub")

    if args.list:
        print("可下载的测试数据集:")
        for k, (path, config, split) in DATASET_MAP.items():
            cfg = f", config={config}" if config else ""
            print(f"  {k}: {path}{cfg} (split={split})")
        return 0

    skip_existing = not args.no_skip_existing
    jobs = max(1, args.jobs)

    if args.business_profile:
        if not args.model_id:
            print("错误: --business-profile 必须配合 --model-id 使用", file=sys.stderr)
            return 2
        profile = ensure_business_dataset_for_model(
            args.model_id,
            args.model_class,
            cache_dir=cache_dir,
            jobs=jobs,
            skip_existing=skip_existing,
        )
        import json

        print(json.dumps(profile, ensure_ascii=False, indent=2))
        return 0

    if args.candidate_datasets:
        if not args.model_id:
            print("错误: --candidate-datasets 必须配合 --model-id 使用", file=sys.stderr)
            return 2
        result = ensure_candidate_datasets_for_model(
            args.model_id,
            args.model_class,
            cache_dir=cache_dir,
            jobs=jobs,
            skip_existing=skip_existing,
        )
        import json

        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    to_download = args.datasets if args.datasets else list(DATASET_MAP.keys())
    seen = set()
    ordered = []
    for k in to_download:
        k_lower = k.lower().replace("-", "_")
        if k_lower in seen:
            continue
        if k_lower in DATASET_MAP:
            seen.add(k_lower)
            ordered.append(k_lower)
        else:
            matched = next((dk for dk in DATASET_MAP if dk == k_lower or dk.endswith("_" + k_lower)), None)
            if matched and matched not in seen:
                seen.add(matched)
                ordered.append(matched)
            else:
                print(f"  [skip] 未知数据集: {k}")

    print(f"缓存目录: {cache_dir}")
    print(f"待下载: {ordered} ({len(ordered)} 个)")
    if skip_existing:
        print("已存在的数据集将跳过（使用 --no-skip-existing 强制重新下载）")
    print("保存到 datasets/{name}/（仅 val/test）")
    print(f"单数据集内 {jobs} 线程并行下载文件")
    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN: 未设置（gated 数据集如 ImageNet 可能失败）")
    else:
        print("HF_TOKEN: 已设置")
    print("-" * 50)

    total = len(ordered)
    ok, skip, fail = 0, 0, 0
    for i, key in enumerate(ordered, 1):
        status = download_one(
            key,
            cache_dir,
            jobs=jobs,
            index=(i, total),
            skip_existing=skip_existing,
        )
        if status == "ok":
            ok += 1
        elif status == "skip":
            skip += 1
        else:
            fail += 1

    print("-" * 50)
    parts = [f"{ok} 成功"]
    if skip:
        parts.append(f"{skip} 跳过")
    if fail:
        parts.append(f"{fail} 失败")
    print(f"完成: {', '.join(parts)}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
