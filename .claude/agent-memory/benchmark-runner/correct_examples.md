# accuracy_run.py 正确示例（代码片段）

（从 MEMORY.md 索引展开的细节文档）

## CACHE_DIR

```python
from pathlib import Path

CACHE_DIR = (Path(__file__).resolve().parent / "models").as_posix()
```

## setup_model 与 use_pretrained 分支

```python
def setup_model(use_pretrained: bool, device, cache_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=cache_dir)
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=cache_dir)

    if use_pretrained:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype="auto", cache_dir=cache_dir)
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    return model, tokenizer
```

## 数据集加载（优先 load_from_disk）

```python
from datasets import load_from_disk

DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"

def load_benchmark_texts() -> tuple[list[str], str]:
    wikitext_path = DATASET_DIR / "wikitext___wikitext-2-raw-v1"
    if wikitext_path.exists():
        ds = load_from_disk(str(wikitext_path))
        texts = sorted([s["text"] for s in ds if s.get("text", "").strip()])
        return texts, "wikitext"
    # fallback
    return ["Hello, benchmark."], "builtin"
```

## 输出文件命名（mode/dtype 必须动态）

```python
def get_dtype_str(dtype: torch.dtype) -> str:
    """Convert torch dtype to short string (fp32, fp16, bf16)."""
    dtype_map = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}
    return dtype_map.get(dtype, "fp32")

adapt_dir = Path(__file__).resolve().parent  # 与 CACHE_DIR 同目录
device_short = device.replace(":", "_")  # npu:0 -> npu_0
dataset_name = "wikitext"  # 与实际加载一致
mode_str = "pretrained" if args.use_pretrained else "config"
# dtype_str 必须根据模型实际加载的 dtype，禁止按设备推断
dtype_str = get_dtype_str(next(model.parameters()).dtype)  # 有 --dtype 时可用 args.dtype
trace_path = adapt_dir / f"trace_{device_short}_{dtype_str}_{mode_str}_{dataset_name}.json"
outputs_path = adapt_dir / f"outputs_{device_short}_{dtype_str}_{mode_str}_{dataset_name}.pt"
```

## setup_model 与 shrink 严禁

- **严禁** `shrink_config_for_dry_run` 等任何 shrink 函数
- DRY RUN 模式直接 `from_config(config)`，不修改 config

## 强制检查命令

```bash
uv run python benchmark/scripts/check_accuracy_run.py              # 检查全部，违规时 exit 1
uv run python benchmark/scripts/check_accuracy_run.py --warn-only  # 仅输出，不失败
uv run python benchmark/scripts/check_accuracy_run.py --adapt xxx  # 仅检查指定 adaptation
```
