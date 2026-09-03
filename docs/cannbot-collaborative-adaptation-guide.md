# CANNBot协同适配使用指导

> CANNBot 不内置、不全局常驻。只有 `team-lead` 明确触发算子缺口协同时，
> 才运行 `scripts/sync_cannbot.sh --print-path` 检查并下载上游 `master` 最新版。
> 下载目录位于本项目内部且被 Git 忽略的 `.cache/cannbot/`，不会下载到用户目录或项目外。普通 Claude Code 会话始终以
> SLAI-AscendBridge2 为项目主体，不加载 CANNBot 的根级身份文件或启动 hook。

cannbot 协同适配是 SLAI-AscendBridge2 的算子缺口补齐机制：当模型有 CUDA-only 算子缺口（NPU 原生/GitCode CANN 社区/Ascend 社区都没有）时，由 cannbot-adapter agent 走 cannbot 4 角色流程生成 Ascend C 算子补齐，让模型在 NPU 跑通/提速。

---

## 一、什么模型适合触发

**判断标准**：模型有 CUDA-only 算子缺口，且 NPU 原生/GitCode CANN 社区/Ascend 社区都没有现成替代。

| 模型类型 | 典型算子缺口 | 适合度 |
|---------|-------------|--------|
| 3D 生成（image-to-3D 等）| sparse conv / hashmap / QEF / 光栅化 / block-sparse attn | 高 ✅ |
| 点云/稀疏视觉 | sparse conv / scatter_reduce / 自定义采样 | 高 ✅ |
| SSM 类 | SSM 算子 / selective_state_update / causal_conv1d | 高 ✅ |
| 自定义 diffusion（triton kernel）| 自定义 sampler / 融合 kernel | 中 |
| 标准 transformer LLM（Llama/Qwen）| 几乎无（attention/MLP NPU 原生）| 不需要 ❌ |
| 标准 CNN | conv/bn 原生 | 不需要 ❌ |

**缺口信号**：适配时出现 `not supported on NPU backend` / `fallback to CPU` / `aten::xxx MISSING` / CUDA 扩展包装不了 → 算子缺口型。

---

## 二、怎么触发

启动 team-lead 后，给这样的提示词：

```
适配 {model_id}，走 adapt→benchmark→optimize→business 全流程。

该模型可能有 CUDA-only 算子缺口，遇缺口时走 CANNBot 协同适配：
- adaptation 阶段：adapter 遇算子无法跑通 → 报 blocked_by_operator_gap
  → 你派 cannbot-adapter 走三段式判定+4 角色生成 Ascend C 算子
- optimization 阶段：npu-optimizer 发现 CPU 回退算子是性能瓶颈 → 报 perf 缺口
  → 派 cannbot-adapter 生成算子提速

cannbot-adapter 流程见 `.claude/agents/cannbot-adapter.md`，方法论见
`.claude/agent-memory/team-lead/cannbot_adaptation_methodology.md`。先获取算子参考源码（CUDA kernel + 纯 torch golden）
再走 4 角色。
```

### 提示词要素

1. **指明算子缺口型**——让 team-lead 知道可能触发 cannbot
2. **两个触发入口都写**——adaptation 硬缺口 + optimization 性能瓶颈（optimization 是主路径，board_ops 状态机 completed 不可回退强制走这）
3. **引用 cannbot-adapter.md + methodology.md**——team-lead 自动按流程走
4. **强调先获取参考源码**——cannbot 4 角色的基础

---

## 三、流程概览

```
model-crawler 注册模型 → team-lead 分配 adaptation 给 adapter
  ↓
adapter 适配遇算子缺口 → 报 blocked_by_operator_gap
  ↓ （或 optimization 阶段 npu-optimizer 报 perf 缺口）
team-lead 派 cannbot-adapter：
  1. 三段式判定（标准 PyTorch 原生 NPU / torch_npu 单一原生接口 → [CANN](https://gitcode.com/cann) / [Ascend](https://gitcode.com/Ascend) 全仓搜索 → cannbot，实测留证到 TRIAGE.md）
  2. 获取参考源码（CUDA kernel + 纯 torch golden + 模型调用点契约）
  3. cannbot 4 角色：Architect → DesignReviewer → Developer → Reviewer
     （产出 .so + DESIGN/PLAN/WALKTHROUGH/REVIEW.md）
  4. cannbot_ops.py 集成（env 开关 + fallback + adaptation-local 路径）
  5. 回报 operator_gap_fixed
  ↓
adapter 继续 / npu-optimizer 集成测速 → benchmark → optimization → business
```

---

## 四、关键规则（必读，血泪换的）

1. **三段式优先级**：标准 PyTorch 原生 NPU > torch_npu 单一原生接口 > [CANN](https://gitcode.com/cann) / [Ascend](https://gitcode.com/Ascend) 现成实现 > cannbot 新建。若 torch_npu 没有单一、公开、语义等价的接口，就判定为没有；禁止组合多个 torch_npu 接口冒充一个原生接口。
2. **社区搜索必须完整且禁止下载仓库**：使用 `scripts/search_operator_communities.py` 全量分页枚举两个组织的公开仓库，并对每个 namespace/关键词完整分页调用 GitCode 服务端代码搜索。只有报告 `complete=true`、`downloaded_repositories=false`、服务端未截断且覆盖数等于枚举数时才能下“未找到”结论；分页或接口失败必须导致 `search_incomplete`。
3. **先获取参考源码**：Architect 前必做。CUDA kernel（设计蓝本）+ 纯 torch golden（精度对照）+ 模型调用点（形状/dtype/是否非连续）。
4. **cannbot 算子集成必须 `.contiguous()`**：算子按裸指针读不遵守 torch strides，来自 torch.split/transpose/view 的非连续张量必须 `.contiguous()`。
5. **Reviewer 必须含真实权重 fuzz**：加载 pretrained 跑 50+ 样本 op vs golden，覆盖实际模型配置，并把样本数、dtype/shape、误差指标、非连续输入、stream 和连续调用结果写入 `acceptance.json`。
6. **stream 一致性**：`at::full(-1)` 等用 torch_npu 默认流，与算子 aclStream 不同步致非确定性。用 aclStream 上的 memsetAsync + synchronize。
7. **MIX 模式陷阱**：`__mix__(1,2)` 融合 Cube+Vector kernel 与自定义 program 调度冲突，AIC 不被驱动。优先非融合双 kernel（纯 AIC `__aicore__` + 纯 AIV `__vector__`）。
8. **编译用 adaptation .venv python**：`cmake -DPython3_EXECUTABLE=<adaptation>/.venv/bin/python`，否则 dlopen 崩 `std::length_error`。
9. **FP32 归约精度**：树形归约 vs aten 顺序累加有结合律差异，MERE<1.22e-4 / MARE<1.22e-3，不强求 atol<1e-5。

---

## 五、监控与验收

- **看文件时间戳判活**：cannbot-adapter 嵌套 spawn 子 agent 时父 agent 心跳停更（阻塞等待），必须看子 agent 产出文件时间戳（DESIGN.md / .asc / .so 的 mtime）判断是否真卡，不能用心跳。
- **4 角色完整流程耗时 3-4 小时**（每角色 spawn + 工作 30min-1h），正常节奏，不是卡。
- **Reviewer 嵌套 spawn 在 GLM-5.2 下可能卡**：前 3 角色可靠出 .so，Reviewer 可能卡。建议 team-lead 顶层 spawn Reviewer（不嵌套）。
- **speedup_ratio ≥3x 需独立 baseline 工件** + comparison_method=independent_baseline_artifact + 非 self_baseline + steady_state latencies 正数。
- **board_ops 写库 + 回读 DB 核验**。
- **结构化验收**：每个自定义算子必须生成并通过 [`operator-acceptance-contract.md`](operator-acceptance-contract.md) 定义的 `acceptance.json` 门禁。

---

## 七、资产位置

| 文件 | 用途 |
|------|------|
| `.claude/agents/cannbot-adapter.md` | cannbot-adapter agent 定义（三段式+4角色+集成+工具链坑+获取参考源码+边界）|
| `.claude/agents/team-lead.md` §0.10 | team-lead 协调流程（adapter 报缺口→派 cannbot-adapter→回报）|
| `.claude/workflows/ascend-cannbot-pipeline.js` | cannbot 协同 workflow 脚本（9 阶段，参考）|
| `.claude/agent-memory/team-lead/cannbot_adaptation_methodology.md` | 核心方法论+所有教训 |
| `.claude/agent-memory/team-lead/ascend_cannbot_pipeline_workflow.md` | workflow 经验+踩坑 |
| `.claude/agent-memory/team-lead/cannbot_mix_mode_pitfall.md` | MIX 模式陷阱 |
| `.claude/agent-memory/team-lead/cannbot_dav2201_viable.md` | dav-2201 硬件支持 |

---

## 八、不适合的场景

- 标准 LLM/CNN：无算子缺口，cannbot 无用武之地，走普通 adapter→benchmark→optimization 即可
- 算子 NPU 原生有（matmul→bmm）：三段式第一段解决，不到 cannbot
- 算子 GitCode CANN 社区或 Ascend 社区有：三段式第二段解决

**一句话**：给 team-lead"模型可能有算子缺口，遇缺口走 cannbot 协同"的提示词，选算子缺口型模型（3D/稀疏/SSM/自定义 diffusion），team-lead 自动按 cannbot-adapter.md + methodology.md 走流程。
