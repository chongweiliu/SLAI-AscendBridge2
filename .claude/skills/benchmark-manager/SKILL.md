---
name: benchmark-manager
description: 管理 benchmark 运行与产出文件。支持 list、run、artifacts、pack、unpack、clean 命令。触发词："benchmark 管理"、"打包产出"、"benchmark 运行"。
---

# Benchmark Manager Skill

管理 benchmark 运行、产出文件的打包/解包。

## 与其他 benchmark skill 的区别

| Skill | 功能 |
|-------|------|
| `benchmark-script` | 生成并执行 `accuracy_run.py` |
| `benchmark-manager` | 管理运行、打包/解包产出文件 |
| `benchmark-analysis` | 聚合报告、对比输出、分析 trace |

---

## 命令

### list - 列出模型

从 `board.db` 列出指定 benchmark_status 的模型。

```bash
# 列出 benchmark_status=completed 的模型（默认）
uv run python benchmark/scripts/benchmark_manager.py list

# 列出指定状态的模型
uv run python benchmark/scripts/benchmark_manager.py list --status in_progress
uv run python benchmark/scripts/benchmark_manager.py list --status pending
```

**参数**:

- `--status`: 过滤 benchmark_status (completed/pending/in_progress)

---

### run - 运行 benchmark

批量或单个运行 benchmark。

```bash
# 运行所有 completed 模型的 benchmark (使用 NPU)
uv run python benchmark/scripts/benchmark_manager.py run

# 使用 CUDA 硬件
uv run python benchmark/scripts/benchmark_manager.py run --hardware cuda

# 运行指定模型 (Tier2 预训练权重)
uv run python benchmark/scripts/benchmark_manager.py run \
  --model Qwen/Qwen2-0.5B \
  --hardware npu \
  --pretrained \
  --max-samples 100
```

**参数**:

- `--model`: 指定模型 ID，不指定则运行所有 completed 的模型
- `--hardware`: 硬件类型 (cuda/npu)，默认 npu
- `--pretrained`: 使用预训练权重 (Tier2)
- `--max-samples`: 最大样本数，默认 250

**产出**:

- `outputs_*.pt`: 模型输出
- `benchmark_metrics_*.json`: 性能指标
- `trace_*.json`: Profiler trace
- `benchmark_runs_*.log`: 运行日志

---

### artifacts - 列出产出文件

列出 adaptation 目录中的产出文件。

```bash
# 列出所有 completed 模型的产出
uv run python benchmark/scripts/benchmark_manager.py artifacts

# 列出指定模型的产出
uv run python benchmark/scripts/benchmark_manager.py artifacts Qwen/Qwen2-0.5B
```

**参数**:

- `model_id`: 模型 ID，不指定则列出所有

---

### pack - 打包产出文件

将产出文件打包为 zip。

```bash
# 打包所有 completed 模型的产出
uv run python benchmark/scripts/benchmark_manager.py pack

# 指定输出文件名
uv run python benchmark/scripts/benchmark_manager.py pack --output my_benchmark.zip

# 打包指定模型
uv run python benchmark/scripts/benchmark_manager.py pack --model Qwen/Qwen2-0.5B
```

**参数**:

- `--output`: 输出 zip 文件路径
- `--model`: 指定模型 ID

**产出**:

- 包含 `manifest.json` 和所有产出文件

---

### unpack - 解包还原产出

将 zip 文件解包到 `adaptations/` 目录。

```bash
uv run python benchmark/scripts/benchmark_manager.py unpack --input benchmark_outputs_20260227.zip
```

**参数**:

- `--input`: 输入 zip 文件路径（必需）

---

### clean - 清空产出文件

删除 adaptation 目录中的产出文件。

```bash
# 清空所有 completed 模型的产出
uv run python benchmark/scripts/benchmark_manager.py clean

# 清空指定模型的产出
uv run python benchmark/scripts/benchmark_manager.py clean --model Qwen/Qwen2-0.5B
```

**参数**:

- `--model`: 指定模型 ID，不指定则清空所有

**删除的文件模式**:

- `outputs_*.pt`
- `benchmark_metrics_*.json`
- `trace_*.json`

---

## 常用工作流

### 1. 运行并打包

```bash
# 1. 运行 benchmark
uv run python benchmark/scripts/benchmark_manager.py run --hardware npu

# 2. 查看产出
uv run python benchmark/scripts/benchmark_manager.py artifacts

# 3. 打包
uv run python benchmark/scripts/benchmark_manager.py pack --output benchmark_npu.zip
```

### 2. 迁移产出

```bash
# 源机器：打包
uv run python benchmark/scripts/benchmark_manager.py pack

# 目标机器：解包
uv run python benchmark/scripts/benchmark_manager.py unpack --input benchmark_outputs_*.zip
```

### 3. 重新运行

```bash
# 1. 清空旧产出
uv run python benchmark/scripts/benchmark_manager.py clean --model Qwen/Qwen2-0.5B

# 2. 重新运行
uv run python benchmark/scripts/benchmark_manager.py run --model Qwen/Qwen2-0.5B --pretrained
```
