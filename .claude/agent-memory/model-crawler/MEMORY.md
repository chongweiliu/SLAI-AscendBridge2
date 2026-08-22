# Model-Crawler Agent Memory

## 状态摘要

**首次启动时间**: 2026-02-20
**最近活动**: 2026-08-22 抓取 HF 热榜 20 个主流模型（清板后重建）
**当前团队**: session-862c3d3c
**状态**: 等待 team-lead 发送抓取指令

## 最近完成任务

### 2026-08-22: HF 热榜 20 模型抓取（board.db 清空后重建）

- **指令**: count=20, source=huggingface（downloads+trending 双榜各取 60 候选）
- **结果**: 注册恰好 20 个模型，全部 adaptation_status=pending，全部非 gated，全部 <=30B
- **筛选要点**: 排除全部 GGUF/MLX/NVFP4/CoreML 量化与单文件模型（含 Comfy-Org/stable-diffusion-v1-5-archive，其标签为 diffusion-single-file 非 diffusers 格式）；排除 >100B（DeepSeek-V4/Kimi-K3/MiniMax-H3）；排除 Uncensored/abliterated 类；文生图仅保留 1 个（SDXL）
- **模型列表**: Qwen3-0.6B, Qwen3-8B, Qwen2.5-7B-Instruct, Qwen2.5-1.5B-Instruct, gpt2, Qwen2.5-VL-7B-Instruct, Qwen3.5-9B, Qwen3.8-27B, t5-small, bert-base-uncased, distilbert-base-uncased, roberta-base, all-MiniLM-L6-v2, bge-m3, ms-marco-MiniLM-L6-v2, mobilenetv3_small_100.lamb_in1k, clip-vit-base-patch32, whisper-large-v3-turbo, clap-htsat-fused, stable-diffusion-xl-base-1.0
- **经验**: 新版 HF API 不再接受 `sort=trending`，需用 `sort='trendingScore'`；`list_models(library=...)` 参数已移除，改用 `filter='diffusers'`

### 2026-02-21: Qwen3-OMNI 模型抓取

- **指令**: count=20, source=huggingface, collection=Qwen3-omni
- **结果**: 注册 20 个模型
- **模型列表**: Qwen2.5-Omni-7B, Qwen3.5-397B-A17B-FP8, Qwen3-Coder-Next-FP8, Qwen3-4B-GGUF, Qwen3-32B-AWQ, Qwen3-8B-GGUF, Qwen3-1.7B-GGUF, Qwen3-Coder-30B-A3B-Instruct-FP8, Qwen3-VL-8B-Instruct-GGUF, Qwen3-VL-30B-A3B-Instruct-FP8, Qwen3-VL-8B-Instruct-FP8, Qwen3-VL-2B-Instruct-GGUF, Qwen3-4B-FP8, Qwen3-14B-GGUF, Qwen3-14B-AWQ, Qwen3-Embedding-4B-GGUF, Qwen3-235B-A22B-Thinking-2507-FP8, Qwen3-30B-A3B-Thinking-2507-FP8, Qwen3-4B-Instruct-2507-FP8, Qwen3-VL-32B-Instruct-FP8

## 核心脚本

| 脚本 | 路径 | 用途 |
|------|------|------|
| board_ops.py | `$PROJECT_ROOT/scripts/board_ops.py` | 看板操作（register_model, heartbeat, list_adaptation_tasks） |
| list_hf_models.py | `$PROJECT_ROOT/scripts/list_hf_models.py` | 从 HuggingFace 获取模型列表 |
| get_model_info.py | `$PROJECT_ROOT/scripts/get_model_info.py` | 获取模型元数据 |

## 常用命令

```bash
# 心跳
$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/board_ops.py heartbeat --id "model-crawler" --status "active" --task "Crawling..."

# 注册模型
$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/board_ops.py register_model \
  --model_id "org/model-name" \
  --source "huggingface" \
  --url "https://huggingface.co/org/model-name" \
  --description "模型描述" \
  --status "pending"

# 查看已入库模型
$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/board_ops.py list_adaptation_tasks

# 获取模型列表
$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/list_hf_models.py --sort downloads --limit 100
```

## 工作流程

1. 启动后立即执行心跳（已完成）
2. 等待 team-lead 发送抓取指令（`action=crawl\ncount=N\nsource=huggingface`）
3. 解析指令后执行抓取
4. 逐个获取元数据并注册
5. 完成后发送 `result=crawl_done` 给 team-lead

## 注意事项

- **网络（2026-08-22 起强制）**: 本机 huggingface.co 直连不通（会挂死），所有 HF 调用必须 `export HF_ENDPOINT=https://hf-mirror.com` **且** `export HF_HUB_DISABLE_XET=1`（xet 后端会连 huggingface.co 导致超时）；Python 中显式 `HfApi(endpoint='https://hf-mirror.com')`；list_hf_models.py 用系统 python3 跑，跑前也要 export 这两个变量
- **HF API 兼容性**: 新版 huggingface_hub 不再接受 `sort=trending`（用 `sort='trendingScore'`）；`list_models(library=...)` 已移除（用 `filter='diffusers'` 等）
- PROJECT_ROOT 必须由当前仓库根目录动态解析，不要写死绝对路径
- board.db 路径: `$PROJECT_ROOT/board.db`（在项目根目录，不是 data 子目录）
- 表名: `models`（不是 adaptation_tasks）
- 发送消息给 team-lead 时使用 `recipient="team-lead"`（连字符）
- 抓取数量必须严格等于指令中的 count

## 从特定组织抓取模型的方法

```python
from huggingface_hub import HfApi
api = HfApi()
# 获取特定组织的所有模型
models = list(api.list_models(author='Qwen', limit=100))
```

## 去重方法

```python
import os
import sqlite3
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
conn = sqlite3.connect(str(project_root / "board.db"))
cursor = conn.cursor()
cursor.execute('SELECT model_id FROM models')
existing = set(row[0] for row in cursor.fetchall())
conn.close()
```

## 并行处理技巧

- 可以并行调用 `get_model_info.py` 和 `register_model` 提高效率
- 使用 `head -30` 限制 get_model_info 输出，避免输出过大

## ⚠️ nopua skill — 遇到困境必须调用

**nopua 不会自动触发**，需要主动 `Skill("nopua")`。

**触发条件**：同一 action 失败 2+ 次 / 陷入等待循环 / 被动等待而不改变策略。

**正确用法**：1. 停止当前循环 2. 查询 board.db 获取真实状态 3. 根据状态决定下一步 4. 写教训到 MEMORY。

**反面教训**：model-crawler 若 HF API 超时/失败，应立即查 board.db 确认已注册数量，而非反复重试抓取。
