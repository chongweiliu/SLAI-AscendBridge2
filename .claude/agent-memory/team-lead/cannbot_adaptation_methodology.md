---
name: cannbot-adaptation-methodology
description: cannbot + SLAI-AscendBridge2 协同适配方法论——通用层（任何算子缺口型模型）vs 2D-to-3D 专属层，含触发条件与验证边界
metadata:
  type: feedback
---

# cannbot 协同适配方法论

## 四步基础方法论（用户指定，算子缺口型模型通用）

适配需要自定义算子的模型时，严格按以下四步执行：

1. **先获取源码，严格按源码流程推理**：不臆造 pipeline，克隆官方 GitHub 源码，按其 README/demo 跑通 CPU/CUDA 基线后再迁 NPU。
2. **完整获取权重**：从 HuggingFace 拉全量 checkpoint，不偷懒只取部分。
3. **算子替换优先级（三段式）**：
   - 优先用 torch_npu 原生算子
   - 缺失算子先到 **gitcode CANN 社区**找替代算子
   - 都没有，再描述清楚功能与需求，用 **cannbot 生成 Ascend C 算子**
4. **结果对齐**：最终 NPU 推理输出必须与 GitHub 项目页展示的效果对比一致（质量/形状/纹理不漂移）。

**Why:** Mamba + Direct3D-S2 已用此方法论跑通。cannbot 是最后手段而非首选。Direct3D-S2 实战教训：cannbot 不要做 NPU 已有算子的重复造轮子（matmul 用原生 bmm 即可），只用在 NPU 真正缺失的算子（scatter_reduce、自定义 block 聚合）。

## 关键分层判断：通用层 vs 2D-to-3D 专属层（2026-07-21 沉淀）

cannbot 协同适配的资产要拆两层看，**不可整体照搬**。

### ✅ 通用层（任何算子缺口型模型都有效）

1. **三段式算子优先级**本身（torch_npu 原生 > gitcode CANN 社区 > cannbot 新建）。
2. **cannbot 工具链流程**：env-check → Architect(DESIGN+PLAN) → Design Reviewer(WALKTHROUGH) → Developer(代码+编译) → Reviewer(100 分制) → 修复循环。
3. **cannbot 工具链的坑**（跨模型复用，已用血泪换）：
   - 编译必须用 adaptation `.venv` 的 python，否则 dlopen 崩 `std::length_error: vector::reserve`
   - FP32 归约精度标准：MERE<1.22e-4 / MARE<1.22e-3（树形归约 vs aten 顺序累加的结合律差异，不强求 atol<1e-5）
   - 跨流水线 EnQue/DeQue 屏障：Cast(PIPE_V) 与 DataCopyPad(PIPE_MTE3) 之间必须有屏障
   - MIX 模式 `__mix__(1,2)` 的 AIC 驱动陷阱 → 优先非融合双 kernel（纯 AIC + 纯 AIV），见 [[cannbot-mix-mode-pitfall]]
   - `.so` 构建坑：cpp 的 `<<<>>>` device launch 语法要改 extern C++ 调用，CMakeLists 把 `op_kernel/*.asc` 加入 ops target
   - op 签名以 `op_extension/register.cpp` 的 `m.def` 为准，不是 `*_torch.cpp`
4. **`cannbot_ops.py` 中央加载器模式**：env 开关 + per-op 覆盖 + 失败只禁用自己 + 调用点日志 + `Path(__file__)` adaptation-local 路径解析（禁硬编码 repo-root 路径）。
5. **workflow 骨架**（9 阶段：Preflight→Adaptation→OperatorGap→CannbotDev→Benchmark→Optimization→BusinessBenchmark→Sync），阶段划分与模型无关。OperatorGap 报告"无缺口"则 CannbotDev 跳过。
6. **方法论通则**："先源码基线再迁 NPU"、"结果对齐官方"、"先定位第一个坏的前级，不靠后级掩盖"。

### ❌ 2D-to-3D 专属层（不可照搬到非 3D 模型）

1. **DINO exact GELU + chunk FP32 Linear**：DINOv3 是 3D 生成 pipeline 图像编码器专属。
2. **dense attention FP32 累加**：针对稀疏 DiT 数值漂移；标准 LLM 用 NPU 原生 SDPA 即可。
3. **PBR 材质升级 / 顶点色 sRGB→linear / 4 材质组分桶**：3D mesh 导出专属。
4. **accent 投影 / 几何保留（禁 meshfix/smooth）**：image-to-3D 后处理专属。
5. **bit-exact 严格度**：3D 生成对数值极敏感（顶点位置越过 mesh 提取阈值→拓扑变化）。LLM 对微小数值差不敏感（logits argmax 几乎不变），不需要这么苛刻。
6. **视觉验证契约（vision agent + probe + 8 gates）**：只对**有视觉输出**的模型有效。

## cannbot 价值触发条件（核心判断）

**模型必须有"CUDA-only 算子缺口且 NPU 原生/社区都没有"才能让 cannbot 产生价值。没缺口就没价值。**

| 模型类型 | 算子缺口 | cannbot 价值 |
|---------|---------|-------------|
| 3D 生成（Direct3D-S2/cadpalette） | 稀疏卷积/hashmap/QEF/光栅化/block-sparse attn | 高，已验证 |
| 点云/稀疏视觉 | sparse conv / scatter_reduce / 自定义采样 | 高，未验证但同类 |
| Mamba/SSM 类 | selective_scan（NPU 可能不支持） | 中高，潜在缺口，**待验证** |
| 某些扩散模型 | 自定义 sampler / triton kernel | 中，看具体算子 |
| 标准 transformer LLM（Llama/Qwen/GPT） | 几乎无 | 低，cannbot 无用武之地 |
| 标准 CNN（分类/检测） | conv/bn 原生支持 | 低 |
| 表格/科学计算模型 | 无特殊算子 | 低 |

## 实操迁移规则

- **下一个 3D 生成 / 稀疏视觉 / 点云模型**：直接复用整套（三段式 + cannbot_ops 模板 + workflow 骨架 + 工具链坑）。前端数值对齐思路可参考（前端数值对齐→后处理不掩盖前级），具体改什么看源码。
- **Mamba/SSM 等有特殊算子的非视觉模型**：复用通用层，但视觉验证契约换成对应模态（logits perplexity / 下游任务指标），bit-exact 严格度可放宽。
- **标准 LLM/CNN**：不需要 cannbot 协同适配，走普通 adapter→benchmark→optimization 流程即可。
- **生成式模型（图像/视频/3D/音频）**：都该有"禁止 text-only 模型凭文件名断言质量 + 多证据 + probe"契约，评测器按模态换。

## 真实场景触发路径（2026-07-21 沉淀，重要）

cannbot-adapter 的自然触发点是个架构关键点。**不能只依赖 adapter 在 adaptation 阶段报缺口**——因为 adaptation 的 dry-run 只验证"能跑通"不验证性能，"能跑通但慢"的算子（如 selective_scan 纯 torch 循环、scatter_reduce CPU 回退）会被 adapter 用纯 torch/CPU fallback 绕过 dry-run，completed，永远不报缺口，cannbot-adapter 不被触发。

真实场景下无人工提示词干预，cannbot-adapter 的正确触发路径：

- **adaptation 阶段触发（仅限"跑不通"的算子）**：算子在 NPU 完全无法执行（报错 not supported 且无任何 torch 改写能跑通）→ adapter 报 blocked_by_operator_gap。这是硬缺口，少数情况。
- **optimization 阶段触发（"能跑但慢"的算子，主路径）**：npu-optimizer 发现性能瓶颈是某 CPU 回退算子（日志 `fallback to CPU` / aten 算子回退 / 极慢的热点）→ 报 perf_blocked_by_operator_gap → cannbot-adapter 生成 Ascend C 算子 → npu-optimizer 重测 perf → 真实提速。segment_reduce 等 cannbot 算子实质就是为性能而生，这是最自然的触发点。

**结论**：cannbot-adapter 需要两个触发入口——adapter 的 adaptation 硬缺口 + npu-optimizer 的 optimization 性能瓶颈。当前 cannbot-adapter.md 只定义了前者，需补后者。验证时若用人工提示词强制 adapter 报缺口，能验证 cannbot 技术可行性，但不反映真实触发；真实场景要靠 optimization 阶段 npu-optimizer 上报才能自然触发。

**board_ops 状态机限制（2026-07-21 Mamba 验证实证）**：`update_adaptation_status` 拒绝 completed→in_progress 回退（`adaptation_status is already completed and cannot be changed`）。这意味着"adaptation 先 completed（用 fallback 跑通 dry-run），后发现需 cannbot 重做 selective_scan"的场景无法通过回退 adaptation 状态实现。后果：一旦 adapter 用 fallback 过了 dry-run 写库 completed，cannbot 算子只能以 optimization 阶段性能优化的方式介入（model_files 集成 + perf 测速），不能回头重做 adaptation。这强制印证了"optimization 触发是 cannbot 在真实场景的主路径"——不是设计选择，是状态机约束。若要支持 adaptation 回退重做，需 board_ops 增加 completed→in_progress 回退接口（带审计）。

**嵌套 spawn 心跳监控盲区（2026-07-21 实证·重要）**：cannbot-adapter 走 4 角色流程时 spawn 子 subagent（ops-direct-invoke:ascendc-kernel-architect 等），**父 agent 阻塞等待子 agent 返回期间不更新自己的心跳**。team-lead 看父 agent 心跳停更（40+ 分钟）容易误判"卡死"，实际子 agent 在持续产出文件。诊断方法：**看子 agent 产出文件时间戳**（operators/<op>/docs/DESIGN.md、op_kernel/*.asc、build/*.so 的 mtime），文件在更新 = 在工作，别用心跳判卡死。cannbot 4 角色完整流程耗时 3-4 小时（每角色 spawn + 工作 30min-1h），是正常节奏。**严禁**因父 agent 心跳停更就 spawn 新 cannbot-adapter 接手——会破坏正在跑的 4 角色流程（差点犯错，被用户打断纠正）。正确做法：耐心等子 agent 产出，或 SendMessage 询问（但父 agent 阻塞期间不处理 inbox，回复会延迟到子 agent 返回后）。

**cannbot 4 角色 Reviewer 阶段卡死（2026-07-21 实证）**：两个 cannbot-adapter（Mamba selective_scan / Direct3D-S2 block_attention_score）都成功走完前 3 角色（Architect→DesignReviewer→Developer），.so 编译生成成功（Direct3D-S2 的可加载且注册到 torch.ops.npu.block_attention_score；Mamba 的可加载但 selective_scan 未注册，注册名待查）。但**两个都在 .so 生成后卡在 Reviewer（第 4 角色）spawn**——文件时间戳停更 2h+，无 REVIEW.md，无 operator_gap_fixed 回报。根因推断：Reviewer subagent spawn 后卡住（可能 GLM-5.2 嵌套 spawn 在第 4 层失败，或 Reviewer 独立构建验证耗时超限）。结论：cannbot 4 角色流程在 GLM-5.2 下前 3 角色可靠出 .so，Reviewer 阶段不可靠。后续若需 Reviewer 验收，考虑 team-lead 直接 spawn Reviewer（顶层，不嵌套）或省略 Reviewer 以 .so 加载+注册+test_torch 精度对照作为验收。

**Reviewer synthetic test 盲区（2026-07-22 实证·重要）**：cannbot 算子的 test_torch.py 用 synthetic 输入（正态分布 scale_a=0.5 等），**测不到真实权重下的边界情况**，导致 Reviewer PASS 但实际不可用：
- Mamba selective_scan：Reviewer PASS 90/100（synthetic 13/13），但 npu-optimizer 集成真实 mamba-130m 权重时暴露——discrete_A=exp(A·dt) 有 37% subnormal 值，bf16 kernel cosine 仅 0.41（精度悬崖）；fp16 跨调用 vector-core 状态污染致间歇 garbage。
- Direct3D-S2 block_attention_score：Reviewer 自构造扩展用例发现 FAIL（num_share=4 UB 崩溃 + isBlockAggr=false 实际配置不可用），但 Developer 的 test 全用 num_share=2/isBlockAggr=true 安全路径。
**结论**：Reviewer 验收必须含**真实权重 fuzz**（加载 pretrained 跑 50+ 样本 op vs golden，覆盖实际模型配置），不能只 synthetic test_torch.py。Developer 的 test 套件常选安全路径，与生产路径脱节。cannbot-adapter.md §1.2 的 Reviewer 职责应明确要求真实权重端到端验证。两个算子都因此返工 Developer。

**cannbot 算子集成 .contiguous() 通用规则（2026-07-22 Mamba 实证·重要）**：cannbot Ascend C 算子**按裸指针读输入、不遵守 torch strides**。集成 patch 必须对所有 op 输入 `.contiguous()`，尤其来自 `torch.split`/`transpose`/`view` 的张量。Mamba selective_scan 的真正根因（3 次误诊后定位）：transformers `slow_forward` 里 `C = torch.split(...)[2]` 是非连续 view，直接喂 op → 读错位内存 → cosine 0.12（被误判为"bf16 精度悬崖"/"stream(true) 取错流"）。修复：patch 对所有输入 `.contiguous()` → cosine=1.0 bit-exact，4.03x 提速。**隔离测试盲区**：`.detach().cpu().to(DEVICE)` 跨设备拷贝会静默使张量连续，掩盖此问题；真实 pipeline 计算图里张量保持非连续才暴露。所有 cannbot 算子集成都应遵守此规则。

## 已验证实战案例（截至 2026-07-21）

全部集中在 3D 生成：Mamba（selective_scan_ssm）、Direct3D-S2（block_attention_score）、cadpalette（conv_none/FlexGEMM）。**非 3D 算子缺口型模型尚未有 cannbot 实战验证**——这是当前方法论的空白区，下一个值得验证的方向是 Mamba/SSM 的 selective_scan。

相关 memory: [[cannbot-segment-reduce-success]]、[[cannbot-mix-mode-pitfall]]、[[direct3d-s2-cannbot-required]]、[[cannbot-dav2201-viable]]、[[ascend-cannbot-pipeline-workflow]]。
