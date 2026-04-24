# Benchmark 可执行流水线

本目录提供 Benchmark 四维度的脚本与聚合能力，与 **benchmark-runner** Agent 及按适配生成的 `accuracy_run.py` 配合使用。

## 四维度说明

| 维度 | 说明 | 数据来源 |
|------|------|----------|
| **1. 工程与迁移成本** | 处理时间等 | board.db `started_at` / `last_updated`，由 `aggregate_reports.py` 计算 |
| **2. 正确性与精度对齐** | Logits 导出、可选对比 | `adaptations/{id}/accuracy_run.py` 产出 `logits_*.pt`；可选 `compare_logits.py` |
| **3. 硬件性能表现** | 延迟、显存峰值 | `accuracy_run.py` 写入当前适配目录的 `benchmark_metrics_*.json` |
| **4. 算子与底层支持度** | Trace、Raw Profiling、CPU/NPU 算子统计 | `accuracy_run.py` **必须**导出 `trace_*.json`；可选保留 `torch_npu.profiler` 原始目录；统一由 `benchmark_tool.py` 解析 |

## 目录结构

- **scripts/benchmark_tool.py**：统一入口，支持 `aggregate`、`compare`、`trace`、`profiling`。
- **scripts/benchmark_manager.py**：批量运行 benchmark、检查产物、管理产出文件。
- **reports/**：单模型或汇总报告输出（可 .gitignore）。
- **figures/**：matplotlib/seaborn 作图输出（可 .gitignore）。

## 使用方式

1. **benchmark-runner** 对需评测的模型在 `adaptations/{safe_name}/` 下生成并运行 `accuracy_run.py`，产出 `outputs_*.pt`、`benchmark_metrics_*.json`、`trace_*.json`；如开启 `torch_npu.profiler`，也可额外保留 raw profiling 目录。
2. 使用 `uv run python benchmark/scripts/benchmark_tool.py compare ...` 对比 CUDA/NPU 两份 `.pt`。
3. 使用 `uv run python benchmark/scripts/benchmark_tool.py trace ...` 解析 `trace_*.json`，得到 fallback 和 CPU/NPU 算子统计。
4. 使用 `uv run python benchmark/scripts/benchmark_tool.py profiling ...` 直接解析 raw profiling 目录，输出 `api_statistic.csv` / `step_trace_time.csv` 等的文本或 JSON 摘要。
5. 使用 `uv run python benchmark/scripts/benchmark_tool.py aggregate ...` 汇总 benchmark 指标，生成报告与图表。

依赖与运行方式见各脚本内注释或 `uv run python benchmark/scripts/xxx.py`（需在项目根执行）。
