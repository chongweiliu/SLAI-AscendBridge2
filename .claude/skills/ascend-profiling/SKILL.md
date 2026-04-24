---
name: agent-profiling
description: 华为昇腾 NPU 性能分析（Profiling）工具使用指南。包含 L0/L1/L2 三级 profiling 采集代码模板、三种解析方案（CSV / SQLite cluster.db / Chrome trace JSON）、auto-tuning 闭环流程。触发词：profiling、性能分析、算子耗时、NPU 性能诊断、torch_npu profiler、cluster.db、auto-tuning。
---

# Agent Profiling Skill

本 skill 指导在华为昇腾 NPU 上进行性能分析（Profiling），包括采集代码模板、三种解析方案（CSV / SQLite / JSON）、auto-tuning 闭环流程。

## 1. Profiling 采集代码

使用 `torch_npu.profiler` 进行 NPU 性能数据采集。按采集深度分为三个级别：

### L0 级别：非膨胀数据

性能影响最小，采集耗时最少，采集信息最少。适合快速定位大粒度瓶颈。

```python
import torch_npu

prof = torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.NPU],
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./Profiling_L0"),
    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1)
)
prof.start()

# ===== 实际执行推理、训练的代码 =====
# model(input)
# ===== 采集区域结束 =====

prof.step()
prof.stop()
```

### L1 级别：分析算子耗时（🌟 Agent 闭环迭代推荐）

性能影响适中，采集耗时适中，采集携带算子信息。适合定位具体算子瓶颈。

**Agent 自动调优场景强烈建议使用 L1**：L0 颗粒度太粗无法定位算子，L2 开启内存和调用栈会导致严重膨胀（Profiling 本身的开销会扭曲真实耗时，让 Agent 产生误判）。

```python
import torch_npu

prof_res_path = "./Profiling_L1"
exp_cfg = torch_npu.profiler._ExperimentalConfig(
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    l2_cache=False
)
prof = torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU
    ],
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_res_path),
    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
    experimental_config=exp_cfg,
    with_stack=False,
    record_shapes=True,
    profile_memory=False,
)
prof.start()

# ===== 实际执行推理、训练的代码 =====
# model(input)
# ===== 采集区域结束 =====

prof.step()
prof.stop()
```

### L2 级别：带有调用栈信息

性能影响最大，采集耗时最多，采集信息最详细。适合深度分析算子调用链路。⚠️ Agent 闭环不建议使用，膨胀过大会扭曲真实耗时。

```python
import torch_npu

prof_res_path = "./Profiling_L2"
exp_cfg = torch_npu.profiler._ExperimentalConfig(
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    l2_cache=False
)
prof = torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU
    ],
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_res_path),
    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
    experimental_config=exp_cfg,
    with_stack=True,
    record_shapes=True,
    profile_memory=True,
)
prof.start()

# ===== 实际执行推理、训练的代码 =====
# model(input)
# ===== 采集区域结束 =====

prof.step()
prof.stop()
```

### 级别选择指南

| 级别 | 性能影响 | 采集耗时 | 信息详细度 | 适用场景 |
|------|---------|---------|-----------|---------|
| L0 | 最小 | 最少 | 基础 NPU 活动 | 快速粗筛、判断是否存在 NPU 瓶颈 |
| **L1** | **适中** | **适中** | **算子耗时 + shape** | **🌟 Agent 闭环迭代、定位具体算子瓶颈** |
| L2 | 最大 | 最多 | 算子耗时 + shape + 调用栈 + 内存 | 深度分析算子调用链路、内存瓶颈（⚠️ 膨胀大，Agent 易误判） |

### schedule 参数说明

```python
schedule=torch_npu.profiler.schedule(
    wait=0,      # 跳过的 step 数
    warmup=0,    # warmup step 数（不计入采集）
    active=1,    # 采集的 step 数
    repeat=1     # 重复次数
)
```

- `wait + warmup + active` 为一个周期，`repeat` 控制周期数
- 典型配置：`wait=1, warmup=1, active=3, repeat=1`（跳过第 1 步，预热 1 步，采集 3 步）

## 2. Profiling 结果解析（三种方案）

Profiling 采集后，所有数据均以文件形式保存在采集目录中（CSV、SQLite、JSON）。Agent 可以通过以下三种方式直接解析，无需安装任何 GUI 工具。

### 方案一：CSV 解析（⚡ 最快，适合快速扫描）

`torch_npu.profiler` 采集目录中会直接生成 CSV 文件，仓库内置 CLI 可直接解析：

```bash
# 单目录分析
uv run python benchmark/scripts/benchmark_tool.py profiling Profiling_L1
uv run python benchmark/scripts/benchmark_tool.py profiling Profiling_L1 -j  # JSON 输出

# 按 adaptation 扫描
uv run python benchmark/scripts/benchmark_tool.py profiling --adaptation lightricks_ltx_2_3
uv run python benchmark/scripts/benchmark_tool.py profiling --all --output benchmark/reports/profiling_report.json
```

**CLI 解析的核心输入文件**：

| 文件 | 说明 | 优先级 |
|------|------|--------|
| `api_statistic.csv` | API 级耗时统计 | 核心 |
| `step_trace_time.csv` | step 级总耗时、计算耗时、通信耗时 | 核心 |
| `op_statistic.csv` | operator 级热点 | 辅助 |
| `kernel_details.csv` | kernel 级热点与 shape | 辅助 |
| `trace_view.json` | 细粒度事件（Chrome trace 格式） | 辅助 |

### 方案二：SQLite cluster.db 解析（🌟 最详细，适合 Agent 深度分析）

Profiling 目录下会生成 `cluster.db`（或 `summary.db`），这是 NPU profiling 最完整的数据源，包含所有算子耗时、调用次数、shape 等信息。Agent 可以用内置的 `sqlite3` 直接提取瓶颈算子。

```bash
# 使用 --deep 标志启用 cluster.db 解析
uv run python benchmark/scripts/benchmark_tool.py profiling Profiling_L1 --deep
uv run python benchmark/scripts/benchmark_tool.py profiling --adaptation lightricks_ltx_2_3 --deep -j
```

`--deep` 会自动：
1. 在 profiling 目录（及一级子目录）中搜索 `cluster.db` / `summary.db`
2. 探测 DB schema（自动适配不同 CANN 版本的表名/列名）
3. 提取 Top K NPU 算子（按总耗时排序）

**Agent 手动解析 cluster.db 示例**：

```python
import sqlite3

# CANN 8.x 实际 DB 路径（嵌套在 ASCEND_PROFILER_OUTPUT 中）
db_path = "./Profiling_L1/xxx_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler.db"
# CANN 7.x 路径：db_path = "./Profiling_L1/xxx_ascend_pt/cluster.db"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 探测可用表
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print(f"Tables: {tables}")

# CANN 8.x: 使用 STRING_IDS 关联获取算子名（nanosecond 精度）
# CANN 7.x: 使用 NpuOperator 表（微秒精度，直接文本列名）
# benchmark_tool.py --deep 会自动探测并选择正确的查询方式

conn.close()
```

**注意事项**：
- DB 文件名因 CANN 版本而异：CANN 7.x 通常为 `cluster.db`/`summary.db`，CANN 8.x 通常为 `analysis.db`/`ascend_pytorch_profiler.db`；`--deep` 会自动搜索所有已知文件名
- DB schema 因 CANN 版本而异：CANN 8.x 使用 `PYTORCH_API`/`CANN_API` 表 + `STRING_IDS` 关联（nanosecond 精度），CANN 7.x 使用 `NpuOperator` 表（直接文本列名，微秒精度）
- 耗时单位可能是 μs 或 ms，`benchmark_tool.py --deep` 会自动判断并转换
- 目录结构：CANN 8.3 实际数据位于 `ASCEND_PROFILER_OUTPUT/` 子目录中（嵌套 3 层），CLI 会自动发现
- `benchmark_tool.py profiling --adaptation` 会自动发现所有 profiling 子目录（含 `ASCEND_PROFILER_OUTPUT`），优先返回包含 CSV 文件的目录

### 方案三：Chrome Trace JSON 解析（适合事件级分析）

Profiling 目录下可能包含 `trace_view.json`（Chrome trace 格式），包含所有 Timeline 事件。

```python
import json

trace_file = "./Profiling_L1/xxx_ascend_pt/trace_view.json"

with open(trace_file, "r") as f:
    trace_data = json.load(f)

# 筛选 NPU 算子执行事件
npu_events = [
    e for e in trace_data.get("traceEvents", [])
    if e.get("cat") == "NpuOp" and "dur" in e
]

# 按耗时排序
top_events = sorted(npu_events, key=lambda x: x["dur"], reverse=True)[:5]
for event in top_events:
    print(f"算子: {event['name']}, 耗时: {event['dur']} μs")
```

### 方案对比

| 方案 | 数据源 | 速度 | 详细度 | Agent 适用场景 |
|------|--------|------|--------|---------------|
| **CSV** | `api_statistic.csv` 等 | ⚡ 最快 | 中 | 快速扫描 Top API / Top operator |
| **SQLite** | `cluster.db` | 中 | 🌟 最全 | 深度算子分析、完整 NPU 算子耗时数据 |
| **JSON** | `trace_view.json` | 慢（文件大） | 最细（事件级） | 分析具体算子调用时序 |

**推荐组合**：先用 CSV 快速扫描，再用 `--deep` SQLite 深挖瓶颈算子。

## 3. Auto-Tuning 闭环流程

Agent 可以利用上述工具实现自动性能调优闭环：

```
┌─────────────────────────────────────────────────┐
│  1. 采集 L1 profiling                           │
│     accuracy_run.py --profile-level L1           │
├─────────────────────────────────────────────────┤
│  2. 解析瓶颈算子                                 │
│     benchmark_tool.py profiling --deep -j        │
├─────────────────────────────────────────────────┤
│  3. Agent 分析结果                                │
│     - 识别 Top 耗时算子 (如 MatMulV3)             │
│     - 搜索 torch_npu 亲和算子替换方案             │
│     - 修改 model_files/ 或 npu_patches.py         │
├─────────────────────────────────────────────────┤
│  4. 验证优化效果                                 │
│     accuracy_run_perf.py run --profile-level L1  │
│     benchmark_tool.py profiling --adaptation X --deep -j │
├─────────────────────────────────────────────────┤
│  5. 对比前后耗时                                 │
│     确认瓶颈算子耗时下降 → 进入下一轮             │
│     确认无进一步优化空间 → 结束                   │
└─────────────────────────────────────────────────┘
```

### Agent 闭环注意事项

1. **始终使用 L1**：L0 无法定位算子，L2 膨胀过大会让 Agent 误判
2. **对比同一 input**：确保 profiling 前后使用相同输入（固定 seed、相同 batch size）
3. **关注总耗时而非单算子**：某算子耗时下降但总耗时不变，说明瓶颈已转移
4. **cluster.db 是金标准**：`--deep` 解析的数据最完整，是算子级别的权威数据源

## 4. 参考文档

- 采集代码参考：[PyTorch 训练/在线推理场景性能分析 - 昇腾社区](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/80RC1alpha003/devguide/appdevg/aclpythondevg/aclpythondevg_0000.html)
- 接口及参数说明：[性能数据采集和自动解析 - CANN 商用版 8.2.RC1 - 昇腾社区](https://www.hiascend.com/document/detail/en/CANN800RC1alphaX/inferapplicationdev/appdevg/aclpythondevg/aclpythondevg_0005.html)
- 算子问题定位：[使用 Profiling 工具定位算子问题 - CSDN 博客](https://blog.csdn.net/weixin_44786530/article/details/136360530)

## 5. 与 accuracy_run.py 的集成

在已有的 `accuracy_run.py` 或 `accuracy_run_perf.py` 中集成 profiling 时，使用 `--profile-level` 参数：

```bash
# accuracy_run.py（baseline）
uv run python accuracy_run.py --profile-level L1

# accuracy_run_perf.py（NPU 优化版）
uv run python accuracy_run_perf.py run --profile-level L1
```

采集后 Profiling 数据输出到 `profiling/{device}_{dtype}_{mode}_{dataset}_{level}/` 目录，
实际数据位于 `ASCEND_PROFILER_OUTPUT/` 子目录中（CANN 8.3 目录结构），
CLI 会自动发现并解析：

```bash
# CSV 快速扫描
uv run python benchmark/scripts/benchmark_tool.py profiling --adaptation lightricks_ltx_2_3

# SQLite 深度分析
uv run python benchmark/scripts/benchmark_tool.py profiling --adaptation lightricks_ltx_2_3 --deep -j
```

**手动集成示例**（不使用 `--profile-level` 时）：

```python
import torch_npu

# L1 级别示例：采集单次推理的算子耗时
prof = torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./profiling_result"),
    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
    experimental_config=torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False
    ),
    record_shapes=True,
    profile_memory=False,
)
prof.start()

# 执行推理
with torch.no_grad():
    output = model(input)

prof.step()
prof.stop()
print("Profiling 结果已保存到 ./profiling_result 目录，可用 benchmark_tool.py profiling --deep 解析。")
```
