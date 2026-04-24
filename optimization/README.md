# NPU Optimization 可执行流水线

本目录提供 NPU 优化（torch_npu 融合算子替换）的脚本与测评工具，与 **npu-optimizer** Agent 及 `accuracy_run_perf.py` 配合使用。

## 产出说明

| 类型 | 说明 | 数据来源 |
|------|------|----------|
| **性能指标** | 延迟、显存峰值 | `benchmark_metrics_*_perf.json` |
| **优化前后对比** | 提速比、延迟/显存降低百分比 | `benchmark_metrics_*_perf.json`（若存在 baseline 则自动合并） |
| **精度对比** | Logits 余弦相似度、PPL 相对差异 | `outputs_*_perf.pt`，通过 compare 对比 CUDA/NPU |
| **Trace 分析** | 算子统计、fallback 比例 | `trace_*_perf.json` |

## 目录结构

- **scripts/check_accuracy_run_perf.py**：检查 accuracy_run_perf.py 核心结构（强制，board_ops 与 CI 均会校验）
- **scripts/optimization_tool.py**：聚合 perf 指标、对比 perf 产出
- **scripts/optimization_manager.py**：列出优化任务、运行、产出管理（artifacts、pack、unpack、clean）
- **reports/**：汇总报告输出

## 使用方式

1. **npu-optimizer** 对 benchmark 完成的模型创建 model_files、accuracy_run_perf.py，产出 `benchmark_metrics_*_perf.json`、`outputs_*_perf.pt`、`trace_*_perf.json`
2. 可选：运行 `optimization_tool.py aggregate` 聚合 perf 指标
3. 可选：运行 `optimization_tool.py compare` 对比 CUDA/NPU 产出
4. 运行 `optimization_manager.py list` 查看 optimization_status

依赖与运行方式见各脚本内注释或 `uv run python optimization/scripts/xxx.py`（需在项目根执行）。
