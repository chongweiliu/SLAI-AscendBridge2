---
name: ascend-cannbot-pipeline-workflow
description: TRELLIS 式 cannbot 全流程 workflow 脚本的位置、用途、codex 验证过的关键模式与踩坑
metadata:
  type: project
---

`.claude/workflows/ascend-cannbot-pipeline.js` 是为「需要 cannbot 补齐 Ascend C 算子才能跑通/跑好的模型」（如 TRELLIS 这类带稀疏卷积/哈希表/QEF/光栅化的 3D 生成模型）固化的 9 阶段确定性 workflow。把现有适配工具链与 CANNBot 算子生成工具串成一条 pipeline，扩大可适配模型范围。

**Why:** TRELLIS v1/v2 的 cannbot 全流程经验只散落在 README/operator_gap_report/CANNBOT_LONGTERM_PLAN 里，没固化成可执行契约。用户要求用 Claude Code Workflow 功能落地，且只复用已有 agent（adapter/model-crawler/benchmark-runner/npu-optimizer/business-benchmark/team-lead）+ cannbot 自带 4 角色，不新增 agent。

**How to apply:**
- 调用：`Workflow({scriptPath: ".claude/workflows/ascend-cannbot-pipeline.js", args: {model_id, upstream_repo, replay?: true}})`。replay=true 为只读验证模式（只跑 check_* gate + update_status，不重建文件）。
- 9 阶段：Preflight(model-crawler+adapter) → Adaptation(adapter) → OperatorGap(adapter) → CannbotDev(cannbot 4角色 pipeline + adapter 集成) → Benchmark(benchmark-runner) → Optimization(npu-optimizer) → BusinessBenchmark(business-benchmark) → Sync(team-lead)。
- 脚本内 `CODEX_PATTERNS` 常量已注入所有 stage prompt，含 codex 在 TRELLIS.2-4B 验证过的可复用模式。

**codex 验证的关键模式（已固化，勿丢）：**
1. custom-repo 模型必须手写 business_benchmark_config.json 固定画像，**禁止 manager run-npu/print-remote-command 自动生成**（会覆盖 custom business_run.py + 把画像改成 image_matting 之类错误值）。自己写 business_run.py 复用 accuracy_run.py loader，不写 business_eval.py。
2. business_run.py env bootstrap self re-exec：sentinel guard（非 ASCEND_HOME_PATH 判断，host 默认指向 CANN 8.2.RC1 坏版本），钉死 ASCEND_RT_VISIBLE_DEVICES。
3. 算子缺口优先级：纯 torch bit-exact > 社区 cann-recipes > C++ CppExtension 降级（排除 .cu）> cannbot 新算子。cannbot 是最后手段。
4. CUDA 扩展降级：setup.py→setup_cpu.py，CUDAExtension→CppExtension，排除 .cu，ext.cpp→ext_cpu.cpp，CUDA-only 算子用 numpy/torch shim。
5. business_metrics 必备字段：evaluation_profile/primary_metric/ttft_ms=null(非 token-streaming 业务)/model_source_kind/throughput_qps/num_samples>50(用 52)。
6. 远端 CUDA 必须 scp NPU baseline outputs.pt 作跨设备 ref，否则 cosine=0.0。

**实踩坑（replay 验证发现）：canonical business_metrics_*.json 曾被 2 样本 smoke run 覆盖（num_samples=2 vs 真实 52），导致 check_business_benchmark_run.py 不过。business_run.py 落盘前必须校验 num_samples==config.max_samples，smoke run 写独立后缀文件。**

**独立复现验证发现（2026-07-17，隔离目录 adaptations/microsoft_trellis_2_4b_wfval/）：**
- workflow 能忠实复现 codex 推理链路：跑通、52 样本、perf vs baseline 自洽 cosine=1.0、peak_memory 一致。
- **但输出不是 bit 可复现**：独立跑的 baseline vs codex baseline mean cosine 0.968，52 样本顶点/面数 0/52 匹配。每个 session 内部自洽（baseline==perf），跨 session 不一致。根因是 cannbot mesh 提取（hashmap_3d + qef_solve_3x3）在 NPU 异步调度（TASK_QUEUE_ENABLE=1）下浮点归约非结合律 → 顶点位置微差越过 mesh 提取阈值 → 拓扑变化。**这是 cannbot+NPU 固有非确定性，非 workflow bug，codex 跨 session 也复现不出 bit 一致。**
- latency/speedup 不可比：共享卡争用噪声主导（codex 0.9766 vs 独立 1.3766 都不反映真实性能）。codex 用 best_of_runs（4 轮取 min）缓解，单轮不可信。
- 独立复现时入口脚本（business_run.py/accuracy_run.py）必须**复制**不能 symlink——它们用 `Path(__file__).resolve().parent` 作 SCRIPT_DIR，.resolve() 解符号链接会把输出写回原目录覆盖 codex 工件。数据资产（models/.venv/TRELLIS.2/npu_patches/operators）可 symlink。cannbot_ops.py 用绝对 _REPO_ROOT 加载 .so，跨目录可用。segment_reduce cannbot 是 adaptation-local（operators/segment_reduce/），隔离时也要 symlink operators/ 否则该算子回退 aten。

详见 [[custom-repo-business-benchmark-traps]]。相关记忆：.claude/agent-memory/business-benchmark/trellis_2_4b_3d_generation.md、custom_repo_trellis_cadpalette.md。
