---
name: ascend-cannbot-pipeline-workflow
description: 通用 cannbot 全流程 workflow 脚本的位置、用途、codex 验证过的关键模式与踩坑
metadata:
  type: project
---

`.claude/workflows/ascend-cannbot-pipeline.js` 是为「需要 cannbot 补齐 Ascend C 算子才能跑通/跑好的模型」（如 3D 生成/SSM 这类带稀疏卷积/哈希表/QEF/光栅化的 3D 生成模型）固化的 9 阶段确定性 workflow。把现有适配工具链与 CANNBot 算子生成工具串成一条 pipeline，扩大可适配模型范围。

**Why:** 算子缺口型模型 的 cannbot 全流程经验只散落在 README/operator_gap_report/CANNBOT_LONGTERM_PLAN 里，没固化成可执行契约。用户要求用 Claude Code Workflow 功能落地，且只复用已有 agent（adapter/model-crawler/benchmark-runner/npu-optimizer/business-benchmark/team-lead）+ cannbot 自带 4 角色，不新增 agent。

**How to apply:**
- 调用：`Workflow({scriptPath: ".claude/workflows/ascend-cannbot-pipeline.js", args: {model_id, upstream_repo, replay?: true}})`。replay=true 为只读验证模式（只跑 check_* gate + update_status，不重建文件）。
- 9 阶段：Preflight(model-crawler+adapter) → Adaptation(adapter) → OperatorGap(adapter) → CannbotDev(cannbot 4角色 pipeline + adapter 集成) → Benchmark(benchmark-runner) → Optimization(npu-optimizer) → BusinessBenchmark(business-benchmark) → Sync(team-lead)。
- 脚本内 `CODEX_PATTERNS` 常量已注入所有 stage prompt，含 cannbot 协同适配 验证过的可复用模式。

**codex 验证的关键模式（已固化，勿丢）：**
1. custom-repo 模型必须手写 business_benchmark_config.json 固定画像，**禁止 manager run-npu/print-remote-command 自动生成**（会覆盖 custom business_run.py + 把画像改成 image_matting 之类错误值）。自己写 business_run.py 复用 accuracy_run.py loader，不写 business_eval.py。
2. business_run.py env bootstrap self re-exec：sentinel guard（非 ASCEND_HOME_PATH 判断，host 默认指向 CANN 8.2.RC1 坏版本），钉死 ASCEND_RT_VISIBLE_DEVICES。
3. 算子缺口优先级：纯 torch bit-exact > GitCode CANN 社区/Ascend 社区现成算子 > C++ CppExtension 降级（排除 .cu）> cannbot 新算子。cannbot 是最后手段。
4. CUDA 扩展降级：setup.py→setup_cpu.py，CUDAExtension→CppExtension，排除 .cu，ext.cpp→ext_cpu.cpp，CUDA-only 算子用 numpy/torch shim。
5. business_metrics 必备字段：evaluation_profile/primary_metric/ttft_ms=null(非 token-streaming 业务)/model_source_kind/throughput_qps/num_samples>50(用 52)。
6. 远端 CUDA 必须 scp NPU baseline outputs.pt 作跨设备 ref，否则 cosine=0.0。

**实踩坑（replay 验证发现）：canonical business_metrics_*.json 曾被 2 样本 smoke run 覆盖（num_samples=2 vs 真实 52），导致 check_business_benchmark_run.py 不过。business_run.py 落盘前必须校验 num_samples==config.max_samples，smoke run 写独立后缀文件。**

