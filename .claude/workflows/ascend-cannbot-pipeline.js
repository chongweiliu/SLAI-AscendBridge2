// ascend-cannbot-pipeline.js
// 将现有模型适配工具链与 cannbot Ascend C 算子生成工具串成一条确定性 pipeline，
// 扩大可适配模型范围（专治算子缺口型模型：带稀疏卷积/哈希表/QEF/光栅化的 3D 生成、SSM 算子、block-sparse attention 等）。
//
// 编排原则：
// - 只用项目已有 agent（model-crawler/adapter/benchmark-runner/npu-optimizer/business-benchmark/team-lead）
//   + cannbot 自带 4 角色（优先参考 cannbot/cannbot-skills/plugins-official/ops-direct-invoke/agents/），不引入新 agent。
// - 每阶段与 board.db 绑定：agent 内部走 board_ops.py 的 assign_*_task / update_*_status 受控接口。
// - 副作用（写库、文件生成）全在 subagent 内；workflow 自身只编排、传上下文、过 gate、分支。
// - Stage 2 命中算子缺口且 decision=ascend_c 时，Stage 3 用 pipeline() 逐算子跑 cannbot 4 步子流程。

export const meta = {
  name: 'ascend-cannbot-pipeline',
  description: '算子缺口型模型全流程：适配→算子缺口分析→cannbot算子开发→评测→优化→业务测评→同步',
  whenToUse: '需要用 cannbot 补齐 Ascend C 算子才能跑通/跑好的算子缺口型模型（3D 生成/SSM/稀疏视觉等）',
  phases: [
    { title: 'Preflight' },
    { title: 'Adaptation' },
    { title: 'OperatorGap' },
    { title: 'CannbotDev' },
    { title: 'Benchmark' },
    { title: 'Optimization' },
    { title: 'BusinessBenchmark' },
    { title: 'Sync' },
  ],
}

// ───────────────────────────── schemas ─────────────────────────────

// replay 模式：每阶段只跑 check_* gate + update_*_status，不重建文件、不重跑 cannbot/CUDA。
// 用于对一个已 completed 的模型回放验证 workflow 的 gate 链与 schema 是否能接受真实工件。
// 防御性解析 args：兼容 object / JSON string / undefined。
const _rawArgs = typeof args === 'string' ? args : args
const _args = (function () {
  if (!_rawArgs) return {}
  if (typeof _rawArgs === 'object') return _rawArgs
  try { return JSON.parse(_rawArgs) } catch (e) { return {} }
})()
const REPLAY = !!_args.replay

// ───────────────────────────── cannbot 协同适配可复用模式 ─────────────────────────────
// 以下模式来自算子缺口型模型（3D 生成/SSM/稀疏视觉）的 cannbot 协同适配实践，固化进 workflow 作为各阶段 agent 的指引。
const CODEX_PATTERNS = [
  '【cannbot 协同适配可复用模式（算子缺口型 custom-repo 模型通用）】',
  '',
  '■ custom-repo 模型识别与处理：',
  '- 非 standard transformers（无 modeling_*.py / 无 setup.py）→ custom-repo。克隆源码到 adaptation_path/{repo}/（删嵌套 .git），sys.path.insert 注入模型包，不 uv add。',
  '- model_source_kind=custom_repo_*_checkpoint，tokenizer_source_kind=not_applicable_custom_repo（custom-repo 通常无 tokenizer）。',
  '- 画像必须手写 business_benchmark_config.json 固定（dataset/evaluation_profile/primary_metric）。dataset_mapping.py --business-profile 会把无标准信号的 custom-repo 误判成 causal_lm+mmlu 或 image_matting（实测被误判过）。',
  '',
  '■ 绕过 manager 自动生成（关键陷阱）：',
  '- custom-repo 模型禁止用 business_benchmark_manager.py 的 run-npu / print-remote-command 自动生成——它们会用通用模板覆盖 custom business_run.py，并把 config 画像改成错误值（实测曾被改成 image_matting）。',
  '- 自己写 business_run.py（复用 accuracy_run.py loader），不写 business_eval.py/business_model_eval.py（避免被当通用 evaluator 调用）。直接 .venv/bin/python business_run.py --scenario {npu_baseline|npu_perf|cuda_baseline} 执行。',
  '- 若误跑了 manager print-remote-command，立即检查 business_run.py 是否被覆盖，从备份恢复。config 里写 custom_repo_note 记录此风险。',
  '',
  '■ business_run.py 模板（codex 验证结构）：',
  '- env bootstrap self re-exec：顶部 sentinel guard（_PHASE4_ENV_READY），未设置则 source setup_env.sh && export sentinel=1 && exec python 自身。用 sentinel 判断而非 ASCEND_HOME_PATH（host 默认指向 8.2.RC1 坏版本）。钉死 ASCEND_RT_VISIBLE_DEVICES=<空闲卡>。',
  '- env defaults（import torch 前）：SPARSE_CONV_BACKEND=none、SPARSE_ATTN_BACKEND=torch_sdpa、ATTN_BACKEND=sdpa、TASK_QUEUE_ENABLE=1、HF_ENDPOINT=https://hf-mirror.com；HF_HOME/TRANSFORMERS_CACHE/TORCH_HOME 全部钉死在 adaptation_path/models/ 下（绝不污染项目根 models/）。',
  '- 两段 patch：apply_all()（repo import 前：stub + device shim + conv + attn + o_voxel）+ apply_lazy_patches()（repo import 后：model paths + image extractor + mesh + cannbot_ops.load_all() 预加载）。',
  '- 复用 accuracy_run.py loader：MODEL_PATH/DATASET_NAME/get_device/load_pipeline/load_benchmark_images/mesh_to_stats/run_step2/run_warmup/reset_peak_memory/empty_cache 全部复用，不重复造轮子。',
  '- 三 scenario：npu_baseline(warmup=0) / npu_perf(warmup=3) / cuda_baseline(warmup=0)。steady_state 计时：每样本 torch.npu.synchronize() 包裹 pipeline.run，wall_clock_s 只含 timed pass（不含 warmup）。每 8 样本 empty_cache。',
  '- 质量计算：npu_baseline 自比 cosine=1.0；npu_perf/cuda_baseline 读 baseline business_outputs_*.pt 的 mesh_statistics 逐样本比（vertices_sample[:512] 展平 cosine + match_rate=n_vertices&n_faces 精确匹配）。',
  '- 落盘命名：business_outputs_{device_short}_{dtype}_pretrained_{dataset}{suffix}.pt + business_metrics_*.json；suffix: npu_baseline=_baseline / npu_perf=_perf / cuda_baseline=_baseline。',
  '',
  '■ business_metrics 必备字段（gate 通过关键）：',
  '- evaluation_profile + primary_metric（与 config 一致，缺则 check 拦截）。',
  '- quality_metric_name/quality_metric_value + cosine_similarity/match_rate；throughput_metric_name/throughput_metric_value + throughput_qps。',
  '- ttft_ms=null, tpot_ms=null（非 token-streaming 业务如 3D 生成/图像一律 null，board_ops 要求 ttft_ms<=latency_s*1000，3D 首样本含初始化会被误拦）。',
  '- model_source_kind/tokenizer_source_kind（contract>=2 非空）；selected_npu/device_topology/parallel_mode=single_device/measurement_contract_version=3/latency_measurement_scope=steady_state/optimization_kind/loaded_from_model_files/scenario_command/benchmark_run_id。',
  '- num_samples > 50（check MIN_BUSINESS_SAMPLE_LOWER_BOUND=50，<=50 报错，用 52）。',
  '',
  '■ 算子缺口方法论（优先级，写进 operator_gap_report.md 每个缺口）：',
  '- 每缺口列 (a) GitCode CANN recipe (b) Ascend 社区替代算子 (c) cannbot Ascend C 新算子 (d) 采用方案 + 理由。',
  '- 优先级：纯 torch bit-exact（如 conv_none、attention_torch_sdpa）> GitCode CANN 社区/Ascend 社区现成算子（如 Hunyuan3D render_npu、gaussian_splatting meta_gauss_render）> C++ CppExtension 降级（排除 .cu，aarch64 可编译）> cannbot 新算子。',
  '- cannbot 新算子只在前三者都不适用时才开发。先用 ascendc-env-check skill 核 NPU arch 在 cannbot 支持矩阵内（如 dav-2201/arch22）。',
  '- 关键 NPU 小算子坑：scatter_reduce(module)→Tensor.scatter_reduce_(fp32)；coords.max/bincount int32→long（aclnnMaxDim rejects int32）；torchvision Normalize in-place→manual out-of-place（aclnnInplaceCopy fails）。',
  '',
  '■ cannbot 集成模式（npu_patches/cannbot_ops.py）：',
  '- 总开关 CANNBOT_OPS（默认1）+ 逐算子 CANNBOT_{NAME}（默认继承 _DEFAULT_ENABLED）。torch.ops.load_library(so) + hasattr(torch.ops.npu, op) 验证注册。',
  '- idempotent + sticky：_LOADED[name]=True 标记尝试不重试；_AVAILABLE[name] 仅 loaded AND registered 时 True。失败只禁该算子，不影响其他。',
  '- load_all() 在 apply_lazy_patches() 末尾预加载，把 ~5s/so 冷加载移出推理关键路径。',
  '- 算子 .so 位置：多 adaptation 共享放仓库根 operators/{op}/build/*.so；仅本模型放 adaptation 内 operators/。',
  '- 已知 pitfall：vector-core exception 507035 → 禁标量 GetValue/SetValue relay，改批量 ReduceSum Pattern::Reduce::AR；UB 192KB 限制下大 Co 走 gather-scatter fallback。kernel 实测慢于 torch fallback 则默认关（如 SPCONV_SKIP_CANNBOT=1）但保留 kernel+设计文档。',
  '',
  '■ CUDA 扩展降级 CppExtension（aarch64 无 nvcc）：',
  '- 复制 setup.py→setup_cpu.py，CUDAExtension→CppExtension，排除所有 .cu，ext.cpp→ext_cpu.cpp（CPU-only dispatch），extra_compile_args cxx: ["-O3","-std=c++17"]。',
  '- 保留纯 C++ 源（flexible_dual_grid/volumetic_attr/svo/filter_*），产 aarch64 原生 _C.so。',
  '- CUDA-only 算子（hash/serialize/z_order/rasterize）用纯 numpy/torch shim 补齐（如 Morton 位操作、searchsorted int64 key 查找）。',
  '- 优先 C++ CppExtension（精度对齐），fallback 纯 Python 重实现。',
  '',
  '■ 优化口径（runtime_only 边际收益）：',
  '- accuracy_run.py(baseline) 与 accuracy_run_perf.py(perf) loader/images/mesh_to_stats 完全相同（直接可比）。perf 版无 Step1 profiler，run_perf 显式 run_warmup 后 timed pass，t0 前额外 synchronize。',
  '- code patch 先试（如 segment_reduce→scatter_add），更慢或数值漂移则 revert（实测 0.92x 已 revert）。退 runtime_only（warmup+TQE）。',
  '- 共享卡噪声 → perf 多轮 best_of_runs（取 min latency over N runs），记 measurement_note。NPU 16 卡满载时 phase4 比 phase3 升 40-50% latency 属真实环境，npu_speedup_ratio<1.0 但 >0.90 gate 且 cosine=1.0 即真实业务结果。',
  '- optimization_notes: code_modified=false + code_change_attempts>=2 + 注明"模型代码无更改"时允许 runtime_only speedup_ratio=1.0。',
  '',
  '■ 远端 CUDA 对齐：',
  '- scp NPU baseline business_outputs_*.pt 到远端作跨设备质量参考（cuda_baseline 要读它算 cosine，不拷则 cosine=0.0 compared_samples=0）。',
  '- 远端 CUDA_VISIBLE_DEVICES=0 .venv/bin/python business_run.py --scenario cuda_baseline --max-samples 52 --use-pretrained。scp 回收 metrics+outputs（md5 校验），summarize 重建 business_summary.json。',
  '- SSH 不通降级 print-remote-command 命令模板 + 写 wait_cuda 释放 owner。',
  '',
  '■ 防工件污染（实踩坑）：',
  '- canonical business_metrics_*.json 曾被一次 2 样本 smoke run 覆盖（num_samples=2 vs 真实 52），导致 check 不过。business_run.py 落盘前必须校验 num_samples==config.max_samples，smoke/aligned run 写独立后缀文件（如 __2sample_aligned），绝不覆盖 canonical。',
  '- wait_cuda 前置 gate：check_business_benchmark_run.py --wait-cuda-npu-only（npu_speedup_ratio>=0.9、质量非全 0、num_samples>50）。',
  '',
  '■ 记忆沉淀：每个 custom-repo cannbot 模型适配完成后，写 .claude/agent-memory/business-benchmark/<model>.md（frontmatter + Why + How to apply + 实跑结果 + 相关 memory 链接），覆盖画像漂移、manager 覆盖、env bootstrap、wall_clock 口径、gate 坑、NPU 噪声等可复用陷阱。',
].join('\n')

const REGISTER_SCHEMA = {
  type: 'object',
  properties: {
    model_id: { type: 'string' },
    adaptation_path: { type: 'string', description: 'adaptations/{sanitized}，/ → _' },
    registered: { type: 'boolean' },
    already_existed: { type: 'boolean' },
  },
  required: ['model_id', 'adaptation_path', 'registered'],
  additionalProperties: false,
}

const ENV_SCHEMA = {
  type: 'object',
  properties: {
    env_ok: { type: 'boolean' },
    device: { type: 'string', description: '如 Ascend910_9382' },
    visible_devices: { type: 'string' },
    cann_version: { type: 'string' },
    torch_npu_version: { type: 'string' },
    failure_reason: { type: 'string' },
  },
  required: ['env_ok', 'device'],
  additionalProperties: false,
}

const ADAPTATION_SCHEMA = {
  type: 'object',
  properties: {
    status: { type: 'string', enum: ['completed', 'skipped', 'needs_authorization', 'not_applicable'] },
    adaptation_path: { type: 'string' },
    notes: { type: 'string' },
    dry_run_passed: { type: 'boolean' },
    failure_reason: { type: 'string' },
  },
  required: ['status', 'adaptation_path', 'notes'],
  additionalProperties: false,
}

const GAP_SCHEMA = {
  type: 'object',
  properties: {
    has_gap: { type: 'boolean' },
    report_path: { type: 'string' },
    candidates: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          op_name: { type: 'string' },
          math: { type: 'string' },
          gap_type: { type: 'string', enum: ['precision', 'performance', 'missing'] },
          decision: { type: 'string', enum: ['ascend_c', 'shim', 'skip'] },
          rationale: { type: 'string' },
          ub_limit: { type: 'string' },
          precision_req: { type: 'string' },
          location: { type: 'string', enum: ['repo_root', 'adaptation_local'] },
        },
        required: ['op_name', 'math', 'gap_type', 'decision', 'rationale'],
        additionalProperties: false,
      },
    },
  },
  required: ['has_gap', 'candidates'],
  additionalProperties: false,
}

const ARCH_SCHEMA = {
  type: 'object',
  properties: {
    op_name: { type: 'string' },
    design_path: { type: 'string' },
    plan_path: { type: 'string' },
    success: { type: 'boolean' },
    notes: { type: 'string' },
  },
  required: ['op_name', 'success'],
  additionalProperties: false,
}

const WALK_SCHEMA = {
  type: 'object',
  properties: {
    op_name: { type: 'string' },
    walkthrough_path: { type: 'string' },
    success: { type: 'boolean' },
    blocking_issues: { type: 'integer' },
    notes: { type: 'string' },
  },
  required: ['op_name', 'success'],
  additionalProperties: false,
}

const DEV_SCHEMA = {
  type: 'object',
  properties: {
    op_name: { type: 'string' },
    so_path: { type: 'string' },
    torch_op_name: { type: 'string' },
    tests_pass: { type: 'boolean' },
    max_diff: { type: 'string' },
    success: { type: 'boolean' },
    notes: { type: 'string' },
  },
  required: ['op_name', 'so_path', 'tests_pass', 'success'],
  additionalProperties: false,
}

const CANNBOT_REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    op_name: { type: 'string' },
    review_path: { type: 'string' },
    score: { type: 'integer' },
    precision_verified: { type: 'boolean' },
    success: { type: 'boolean' },
    notes: { type: 'string' },
  },
  required: ['op_name', 'score', 'precision_verified', 'success'],
  additionalProperties: false,
}

const INTEGRATION_SCHEMA = {
  type: 'object',
  properties: {
    ops_integrated: { type: 'array', items: { type: 'string' } },
    loader_path: { type: 'string', description: 'npu_patches/cannbot_ops.py' },
    e2e_stable: { type: 'boolean' },
    e2e_calls: { type: 'integer' },
    success: { type: 'boolean' },
    notes: { type: 'string' },
  },
  required: ['ops_integrated', 'loader_path', 'e2e_stable', 'success'],
  additionalProperties: false,
}

const BENCHMARK_SCHEMA = {
  type: 'object',
  properties: {
    status: { type: 'string', enum: ['completed', 'skipped', 'pending'] },
    num_samples: { type: 'integer' },
    artifacts: { type: 'array', items: { type: 'string' } },
    cosine_similarity: { type: 'string' },
    failure_reason: { type: 'string' },
  },
  required: ['status', 'num_samples'],
  additionalProperties: false,
}

const OPT_SCHEMA = {
  type: 'object',
  properties: {
    status: { type: 'string', enum: ['completed', 'skipped', 'pending', 'not_applicable'] },
    speedup_ratio: { type: 'number' },
    optimization_kind: { type: 'string', enum: ['code_patch', 'runtime_only'] },
    notes_path: { type: 'string' },
    failure_reason: { type: 'string' },
  },
  required: ['status', 'optimization_kind'],
  additionalProperties: false,
}

const BUSINESS_SCHEMA = {
  type: 'object',
  properties: {
    status: { type: 'string', enum: ['completed', 'skipped', 'pending', 'wait_cuda', 'not_applicable'] },
    npu_speedup_ratio: { type: 'number' },
    vs_cuda_latency_ratio: { type: 'number' },
    summary_path: { type: 'string' },
    failure_reason: { type: 'string' },
  },
  required: ['status'],
  additionalProperties: false,
}

const SYNC_SCHEMA = {
  type: 'object',
  properties: {
    committed: { type: 'boolean' },
    commit_sha: { type: 'string' },
    human_review_status: { type: 'string' },
    notes: { type: 'string' },
  },
  required: ['committed', 'human_review_status'],
  additionalProperties: false,
}

// ───────────────────────────── helpers ─────────────────────────────

const BOARD = '$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/board_ops.py'

function commonPreamble(ctx) {
  const parts = [
    '【项目根目录】先执行 export PROJECT_ROOT=$(git rev-parse --show-toplevel) 确认。',
    '【board_ops】统一用 ' + BOARD + ' <subcommand>；不确定参数时先 --help。',
    '【缓存边界】模型缓存仅允许 ' + ctx.adaptation_path + '/models；严禁项目根 models/。',
    '【镜像】export HF_ENDPOINT=https://hf-mirror.com；export TASK_QUEUE_ENABLE=1。',
    '【当前上下文】model_id=' + ctx.model_id + (ctx.upstream_repo ? '  upstream_repo=' + ctx.upstream_repo : '') +
      (ctx.adaptation_path ? '  adaptation_path=' + ctx.adaptation_path : '') +
      (ctx.device ? '  device=' + ctx.device : ''),
    '',
    CODEX_PATTERNS,
  ]
  return parts.join('\n')
}

// ───────────────────────────── Stage 0: Preflight ─────────────────────────────

phase('Preflight')

const modelId = _args.model_id
const upstreamRepo = _args.upstream_repo
if (!modelId) throw new Error('args.model_id 必填（如 {model_id}）')
if (!upstreamRepo) throw new Error('args.upstream_repo 必填（上游 git 仓库）')

log('Preflight: 注册模型 + 环境校验 model_id=' + modelId + (REPLAY ? ' [REPLAY]' : ''))

const regPrompt = [
  commonPreamble({ model_id: modelId, upstream_repo: upstreamRepo }),
  '',
  '你是 model-crawler。任务：注册模型并推导 adaptation_path。',
  REPLAY
    ? '【REPLAY】模型已入库。只读确认：list_adaptation_tasks 找到该 model_id；推导现有 adaptation_path（读 adaptations/ 目录已存在的 sanitized 子目录）。不注册、不修改任何文件。'
    : '1. 若 board.db 未入库：' + BOARD + ' register_model --model_id "' + modelId + '" --url "' + upstreamRepo + '" --source huggingface（按 --help 补全参数）。\n2. 若已入库则跳过。\n3. 推导 adaptation_path：参考现有 adaptations/ 目录命名（model_id 的 / → _，如 {model_id} → {sanitized_model_id}）。\n4. 通过 list_adaptation_tasks 确认该 model_id 存在且 adaptation_status=pending。',
  '返回 schema：registered / already_existed / adaptation_path / model_id。',
].join('\n')

const reg = await agent(regPrompt, {
  agentType: 'model-crawler',
  schema: REGISTER_SCHEMA,
  phase: 'Preflight',
  label: 'register',
})

const envPrompt = [
  commonPreamble({ model_id: modelId, adaptation_path: reg.adaptation_path }),
  '',
  '你是 adapter。任务：NPU 环境前置校验（复用 .claude/skills/ascend-adaptation 与 uv-env-setup）。',
  REPLAY ? '【REPLAY】只读确认环境可用即可（npu-smi info 选卡、确认 torch_npu 可 import）。不修改环境配置文件。' : '',
  '要求：',
  '- CANN 8.5.0（不得 8.2.RC1，会 aclnn 561103）；torch 2.9.0 + torch-npu 2.9.0 + torchvision 0.24.0；numpy<2（CANN TBE 用了已移除的 np.float_）；Python 3.11。',
  '- LD_LIBRARY_PATH 剥离 nnal/atb（ATB 插件会 abort torch_npu import）。',
  '- npu-smi info 选空闲/低占用单卡 → export ASCEND_RT_VISIBLE_DEVICES=<该卡>（不写死 0 号）。',
  '- export TASK_QUEUE_ENABLE=1；export HF_ENDPOINT=https://hf-mirror.com。',
  '返回 schema：env_ok / device / visible_devices / cann_version / torch_npu_version / failure_reason。',
].join('\n')

const env = await agent(envPrompt, {
  agentType: 'adapter',
  schema: ENV_SCHEMA,
  phase: 'Preflight',
  label: 'env-check',
})

if (!env.env_ok) {
  log('Preflight 失败：环境不满足，终止。reason=' + (env.failure_reason || ''))
  return { model_id: modelId, stage: 'Preflight', status: 'env_failed', env }
}
log('Preflight 通过：device=' + env.device + ' visible=' + env.visible_devices)

const ctx = {
  model_id: modelId,
  upstream_repo: upstreamRepo,
  adaptation_path: reg.adaptation_path,
  device: env.device,
  visible_devices: env.visible_devices,
}

// ───────────────────────────── Stage 1: Adaptation ─────────────────────────────

phase('Adaptation')
log('Adaptation: 克隆源码 + npu_patches + demo.py + healing')

const adaptPrompt = [
  commonPreamble(ctx),
  '',
  '你是 adapter。任务：完成模型适配（参考 .claude/skills/ascend-adaptation；diffusers pipeline 再参考 ascend-diffusers-adaptation）。',
  REPLAY
    ? [
        '【REPLAY】该 adaptation 已有 codex 产出的工件。你的任务是【验证而非重建】：',
        '1. 领任务：' + BOARD + ' assign_adaptation_task --agent_id adapter-1。',
        '2. 只读检查：adaptation/scripts/check_adaptation.py --adapt ' + ctx.adaptation_path + '（或按 check_adaptation.py 实际用法）；读 output.txt 确认含 [Run] Output: / [Success] 且无 "Falling back to simpler validation"。',
        '3. 不得修改/删除/重建任何现有文件。',
        '4. 若 gate 通过：' + BOARD + ' update_adaptation_status --model_id "' + modelId + '" --adaptation_status completed --adaptation_path ' + ctx.adaptation_path + ' --adaptation_notes "replay: codex 工件通过 check_adaptation"。',
        '5. 若 gate 失败：status=pending，failure_reason 记录原因，不修文件。',
      ].join('\n')
    : [
        '1. 领任务：' + BOARD + ' assign_adaptation_task --agent_id adapter-1。从输出解析 adaptation_path=... 原样使用。',
        '2. 克隆 ' + upstreamRepo + ' 进 adaptation_path/{repo}/，删除嵌套 .git/（.gitignore 加 **/{repo}/.git）。',
        '3. 写 pyproject.toml（含 ascend extra）、demo.py（支持 --dry-run）。',
        '4. 【custom-repo 检测】若非 standard transformers（无 modeling_*.py / 无 setup.py），按 CODEX_PATTERNS 的 custom-repo 处理：sys.path.insert 注入包，不 uv add；model_source_kind=custom_repo_*_checkpoint。',
        '5. 建 npu_patches/，优先复用现成 patch（参考已有 adaptation 的 npu_patches/）：conv_none、attention_torch_sdpa、o_voxel_npu、ovoxel_runtime、scatter_reduce→double scatter_add_、aclnnInplaceCopy 规避（torchvision Normalize 改手动 out-of-place）、.cuda()→input device、coords.long()、spatial_shape int64。',
        '6. 【CUDA 扩展降级】若模型含 CUDA C++ 扩展（如 o-voxel）且本机无 nvcc/aarch64，按 CODEX_PATTERNS 写 setup_cpu.py（CppExtension，排除 .cu，ext_cpu.cpp），CUDA-only 算子用纯 numpy/torch shim 补齐。',
        '7. healing loop：uv run python demo.py → 抓 Traceback → 自修，直到通过。',
        '8. gate：adaptation/scripts/check_adaptation.py 通过；output.txt 必须含 [Run] Output: / [Success]，不得含 "Falling back to simpler validation"。',
        '9. 完成后：' + BOARD + ' update_adaptation_status --model_id "' + modelId + '" --adaptation_status completed --adaptation_path <path> --adaptation_notes "<notes>"。',
        '边界：模型缓存仅 adaptation_path/models/，严禁项目根 models/。',
      ].join('\n'),
  '返回 schema：status / adaptation_path / notes / dry_run_passed / failure_reason。',
].join('\n')

const adapt = await agent(adaptPrompt, {
  agentType: 'adapter',
  schema: ADAPTATION_SCHEMA,
  phase: 'Adaptation',
  label: 'adapt',
})

ctx.adaptation_path = adapt.adaptation_path || ctx.adaptation_path

if (adapt.status !== 'completed') {
  log('Adaptation 非 completed：status=' + adapt.status + '，落库收口，终止主流程。')
  return { model_id: modelId, stage: 'Adaptation', status: adapt.status, adapt }
}
log('Adaptation 完成：path=' + adapt.adaptation_path)

// ───────────────────────────── Stage 2: OperatorGap ─────────────────────────────

phase('OperatorGap')
log('OperatorGap: trace/profile 找 CPU fallback + 性能热点')

const gapPrompt = [
  commonPreamble(ctx),
  '',
  '你是 adapter。任务：算子缺口分析（参考 .claude/skills/ascend-profiling）。',
  REPLAY
    ? [
        '【REPLAY】不重跑 profile。只读现有工件重建候选清单：',
        '1. 读 ' + ctx.adaptation_path + '/operator_gap_report.md（若存在）。',
        '2. 读 ' + ctx.adaptation_path + '/npu_patches/cannbot_ops.py 与仓库根 operators/ 下已被集成的算子目录，反推 decision=ascend_c 的算子清单（按模型实际缺口，如 hashmap_3d / submanifold_conv3d / qef_solve_3x3 / uv_rasterize_interp / sparse_grid_sample_3d / SSM 算子 / 稀疏注意力算子 等）。',
        '3. 不修改任何文件。',
      ].join('\n')
    : [
        '1. 跑 trace / demo 或 accuracy_run 带 --profile-level L1，定位 CPU fallback 算子与性能热点。',
        '2. 产 ' + ctx.adaptation_path + '/operator_gap_report.md：每个缺口写明 功能、输入/输出/语义、缺口类型（precision/performance/missing）、四条方案 (a)GitCode CANN recipe (b)Ascend 社区替代算子 (c)cannbot Ascend C 新算子 (d)采用方案 + 理由、UB/精度约束。',
        '3. 【优先级，严格按序】纯 torch bit-exact（conv_none/attention_torch_sdpa）> GitCode CANN 社区/Ascend 社区现成算子（Hunyuan3D render_npu / gaussian_splatting meta_gauss_render）> C++ CppExtension 降级（排除 .cu）> cannbot 新算子。cannbot 只在前三者都不适用时才 decision=ascend_c。先用 ascendc-env-check skill 核 NPU arch 在 cannbot 支持矩阵内。',
        '4. 仅列真正值得 cannbot 补齐的算子（精度关键如 sparse conv fp32 累加；性能关键如 hashmap 邻居查找、QEF、UV rasterize、sparse grid_sample）。python shim 可接受则 decision=shim；无价值则 skip。',
        '5. 对 decision=ascend_c 的算子给出 location：多 adaptation 共享 → repo_root（仓库根 operators/）；仅本模型 → adaptation_local。',
      ].join('\n'),
  '返回 schema：has_gap / report_path / candidates[]。',
].join('\n')

const gap = await agent(gapPrompt, {
  agentType: 'adapter',
  schema: GAP_SCHEMA,
  phase: 'OperatorGap',
  label: 'operator-gap',
})
log('OperatorGap: has_gap=' + gap.has_gap + ' candidates=' + gap.candidates.length)

const ascendOps = gap.candidates.filter(function (c) { return c.decision === 'ascend_c' })

// ───────────────────────────── Stage 3: CannbotDev ─────────────────────────────

let integration = null
if (gap.has_gap && ascendOps.length > 0) {
  phase('CannbotDev')
  if (REPLAY) {
    log('CannbotDev [REPLAY]: 跳过 4 步开发，只验证现有 cannbot 算子 + 集成 loader')
    const integPrompt = [
      commonPreamble(ctx),
      '',
      '你是 adapter。任务【REPLAY 验证】：确认 codex 已开发的 cannbot 算子与集成 loader 仍可用。',
      '1. 只读检查 ' + ctx.adaptation_path + '/npu_patches/cannbot_ops.py 存在且能 import（不修改）。',
      '2. 只读检查仓库根 operators/ 下各算子 .so 存在：' + ascendOps.map(function (o) { return o.op_name }).join(' / '),
      '3. 不重新开发、不重新编译、不修改文件。',
      '4. 若都存在且 loader 可加载：success=true，ops_integrated=算子清单，e2e_stable=true（沿用历史验证）。',
      '返回 schema：ops_integrated / loader_path / e2e_stable / e2e_calls / success / notes。',
    ].join('\n')
    integration = await agent(integPrompt, {
      agentType: 'adapter',
      schema: INTEGRATION_SCHEMA,
      phase: 'CannbotDev',
      label: 'integrate-verify',
    })
    log('CannbotDev [REPLAY] 验证：success=' + integration.success + ' ops=' + integration.ops_integrated.length)
  } else {
    log('CannbotDev: 对 ' + ascendOps.length + ' 个算子跑 cannbot 4 步子流程（pipeline 无 barrier）')

  function operatorsDirFor(op) {
    return op.location === 'adaptation_local'
      ? ctx.adaptation_path + '/operators/' + op.op_name
      : '$PROJECT_ROOT/operators/' + op.op_name
  }

  function archPrompt(op) {
    return [
      commonPreamble(ctx),
      '',
      '你是 ascendc-kernel-architect。任务：为算子 ' + op.op_name + ' 做架构设计。',
      '工作目录：' + operatorsDirFor(op),
      '算子数学定义：' + op.math,
      '缺口类型：' + op.gap_type + '；精度要求：' + (op.precision_req || '对齐 torch 参考实现') + '；UB 限制：' + (op.ub_limit || '默认 192KB') + '。',
      '背景：该算子是模型 ' + modelId + ' 在 Ascend NPU 上的 CPU fallback / 性能瓶颈，需要补齐为 Ascend C 直调算子，注册为 torch.ops.npu.' + op.op_name + '。',
      '产出：DESIGN.md + PLAN.md（写入工作目录）。遵循 CANNBot 工作流规范。',
      '返回 schema：op_name / design_path / plan_path / success / notes。',
    ].join('\n')
  }

  function walkPrompt(_prev, op) {
    return [
      commonPreamble(ctx),
      '',
      '你是 ascendc-kernel-design-reviewer。任务：独立审查 ' + op.op_name + ' 的设计（不参与开发）。',
      '工作目录：' + operatorsDirFor(op) + '。读取架构师产出的 DESIGN.md / PLAN.md。',
      '从可实现性角度产出 WALKTHROUGH.md 质疑清单。',
      '返回 schema：op_name / walkthrough_path / success / blocking_issues / notes。',
    ].join('\n')
  }

  function devPrompt(_prev, op) {
    return [
      commonPreamble(ctx),
      '',
      '你是 ascendc-kernel-developer。任务：实现 ' + op.op_name + '。',
      '工作目录：' + operatorsDirFor(op) + '。读取 DESIGN.md / PLAN.md / WALKTHROUGH.md。',
      '产出 op_kernel/*.asc、op_host/*.asc、op_extension/*.cpp（register.cpp 注册 torch.ops.npu.' + op.op_name + '），编译出 .so，写测试用例。',
      '【pitfall（实测教训）】507035 vector-core exception：禁用标量 GetValue/SetValue relay，改批量 ReduceSum Pattern::Reduce::AR；UB 192KB 限制下大 Co 走 gather-scatter fallback。',
      '验证：独立用例 max_diff 达标（fp32 累加），tests_pass=true。',
      '返回 schema：op_name / so_path / torch_op_name / tests_pass / max_diff / success / notes。',
    ].join('\n')
  }

  function reviewPrompt(_prev, op) {
    return [
      commonPreamble(ctx),
      '',
      '你是 ascendc-kernel-reviewer。任务：独立审查 ' + op.op_name + ' 的代码与精度。',
      '工作目录：' + operatorsDirFor(op) + '。读取已实现代码与测试。',
      '独立构建验证、100 分制代码质量评估、精度验证，产出 REVIEW.md。',
      '返回 schema：op_name / review_path / score / precision_verified / success / notes。',
    ].join('\n')
  }

  const cannbotResults = await pipeline(
    ascendOps,
    function (op) {
      return agent(archPrompt(op), { agentType: 'ascendc-kernel-architect', schema: ARCH_SCHEMA, phase: 'CannbotDev', label: 'arch:' + op.op_name })
    },
    function (_prev, op) {
      return agent(walkPrompt(_prev, op), { agentType: 'ascendc-kernel-design-reviewer', schema: WALK_SCHEMA, phase: 'CannbotDev', label: 'walk:' + op.op_name })
    },
    function (_prev, op) {
      return agent(devPrompt(_prev, op), { agentType: 'ascendc-kernel-developer', schema: DEV_SCHEMA, phase: 'CannbotDev', label: 'dev:' + op.op_name })
    },
    function (_prev, op) {
      return agent(reviewPrompt(_prev, op), { agentType: 'ascendc-kernel-reviewer', schema: CANNBOT_REVIEW_SCHEMA, phase: 'CannbotDev', label: 'rev:' + op.op_name })
    }
  )

  const okOps = cannbotResults.filter(Boolean).filter(function (r) { return r.success && r.precision_verified })
  log('CannbotDev: ' + okOps.length + '/' + ascendOps.length + ' 算子通过设计与精度审查')

  // 集成阶段
  const integPrompt = [
    commonPreamble(ctx),
    '',
    '你是 adapter。任务：把 cannbot 产出的算子集成进 ' + ctx.adaptation_path + '/npu_patches/cannbot_ops.py。',
    '已通过算子（torch_op_name）：' + okOps.map(function (o) { return o.op_name }).join(', '),
    '1. 写 cannbot_ops.py：CANNBOT_OPS 总开关 + 逐算子 CANNBOT_{OP} 覆盖，默认开启；加载各 .so，注册 torch.ops.npu.{op}。',
    '2. e2e 稳定性验证：在真实 pipeline 中多次调用（如 24 层 × 50 样本连续调用，含大 N/Cin/Co 配置），e2e_stable=true。',
    '3. 【诚实降级（v1 教训）】若某 kernel 实测慢于 torch fallback，默认关闭该算子（如 SPCONV_SKIP_CANNBOT=1），但保留 kernel + 设计文档。',
    '4. 把 .so 路径、torch_op_name、enabled_by_default 记入 notes。',
    '返回 schema：ops_integrated / loader_path / e2e_stable / e2e_calls / success / notes。',
  ].join('\n')

  integration = await agent(integPrompt, {
    agentType: 'adapter',
    schema: INTEGRATION_SCHEMA,
    phase: 'CannbotDev',
    label: 'integrate',
  })
  log('CannbotDev 集成：e2e_stable=' + integration.e2e_stable + ' ops=' + integration.ops_integrated.length)
  } // end non-REPLAY cannbot dev
} else {
  log('OperatorGap 无 ascend_c 算子，跳过 CannbotDev，直进 Benchmark')
}

// ───────────────────────────── Stage 4: Benchmark ─────────────────────────────

phase('Benchmark')
log('Benchmark: accuracy_run.py + outputs/metrics/trace')

const benchPrompt = [
  commonPreamble(ctx),
  '',
  '你是 benchmark-runner。任务：精度/性能/trace 评测（参考 .claude/skills/benchmark-script、benchmark-manager、dataset-mapping）。',
  REPLAY
    ? [
        '【REPLAY】不重跑 accuracy_run。只读验证现有工件：',
        '1. 领任务：' + BOARD + ' assign_benchmark_task --agent_id benchmark-runner-1。',
        '2. 只读检查：benchmark/scripts/check_accuracy_run.py --adapt ' + ctx.adaptation_path + ' 通过；读 benchmark_metrics_*.json 确认 num_samples>=50。',
        '3. 不得修改/重跑任何文件。',
        '4. 若通过：' + BOARD + ' update_benchmark_status --model_id "' + modelId + '" --benchmark_status completed --notes "replay: codex 工件通过 check_accuracy_run"。',
      ].join('\n')
    : [
        '1. 领任务：' + BOARD + ' assign_benchmark_task --agent_id benchmark-runner-1。',
        '2. 生成并运行 accuracy_run.py（--use-pretrained 加载真实权重；--cpu 兜底）。产出 outputs_*.pt、benchmark_metrics_*.json、trace_*.json。',
        '3. 要求 num_samples>=50；不足则先用 scripts/dataset_mapping.py --model-id ' + modelId + ' --candidates 补候选数据集并重测。',
        '4. 若 cannbot 已集成（' + (integration ? '是' : '否') + '），确保 accuracy_run 能加载 npu_patches/cannbot_ops.py 的 patch。',
        '5. gate：benchmark/scripts/check_accuracy_run.py --adapt ' + ctx.adaptation_path + ' 通过。',
        '6. 完成后：' + BOARD + ' update_benchmark_status --model_id "' + modelId + '" --benchmark_status completed --notes "<notes>"。',
      ].join('\n'),
  '返回 schema：status / num_samples / artifacts / cosine_similarity / failure_reason。',
].join('\n')

const bench = await agent(benchPrompt, {
  agentType: 'benchmark-runner',
  schema: BENCHMARK_SCHEMA,
  phase: 'Benchmark',
  label: 'benchmark',
})
log('Benchmark: status=' + bench.status + ' num_samples=' + bench.num_samples)

if (bench.status !== 'completed') {
  log('Benchmark 非 completed，终止主流程。')
  return { model_id: modelId, stage: 'Benchmark', status: bench.status, bench }
}

// ───────────────────────────── Stage 5: Optimization ─────────────────────────────

phase('Optimization')
log('Optimization: accuracy_run_perf.py + optimization_notes.json')

const optPrompt = [
  commonPreamble(ctx),
  '',
  '你是 npu-optimizer。任务：NPU 推理优化（参考 .claude/skills/torch-npu-optimization；diffusers 参考 ascend-diffusers-optimization）。',
  REPLAY
    ? [
        '【REPLAY】不重跑 perf。只读验证现有工件：',
        '1. 领任务：' + BOARD + ' assign_optimization_task --agent_id npu-optimizer-1。',
        '2. 只读检查：optimization/scripts/check_accuracy_run_perf.py --adapt ' + ctx.adaptation_path + ' 与 optimization/scripts/check_optimization_notes.py --adapt ' + ctx.adaptation_path + ' 通过。',
        '3. 读 ' + ctx.adaptation_path + '/optimization_notes.json 取 best_result.speedup_ratio 与 optimization_kind。不修改文件。',
        '4. 若通过：' + BOARD + ' update_optimization_status --model_id "' + modelId + '" --optimization_status completed --notes "$(cat ' + ctx.adaptation_path + '/optimization_notes.json)"。',
      ].join('\n')
    : [
        '1. 领任务：' + BOARD + ' assign_optimization_task --agent_id npu-optimizer-1。',
        '2. 先执行 benchmark/scripts/check_accuracy_run.py --adapt ' + ctx.adaptation_path + ' 核对 baseline 脚本；不通过先修 accuracy_run.py。',
        '3. 写 accuracy_run_perf.py。先试 code patch（如 segment_reduce→scatter_add）；若更慢则 revert（实测 0.92× 已 revert）。',
        '4. 退 runtime_only（warmup + TASK_QUEUE_ENABLE=1）。',
        '5. 产 optimization_notes.json：',
        '   - code_patch 要求 best_result.speedup_ratio>1.0；',
        '   - runtime_only 仅在 code_modified=false && code_change_attempts>=2 && 注明"模型代码无更改"时允许 speedup_ratio=1.0；',
        '   - speedup_ratio>=3x 须 comparison_method=independent_baseline_artifact + 有效 comparison_scope + 非空 validation_note + 正数 steady_state_baseline/perf_latency_s。',
        '6. 共享卡噪声 → perf 多轮 best-of-runs（取 min latency）。',
        '7. gate：optimization/scripts/check_accuracy_run_perf.py --adapt 与 optimization/scripts/check_optimization_notes.py --adapt 通过；同步 benchmark_metrics*.json 口径。',
        '8. 完成后：' + BOARD + ' update_optimization_status --model_id "' + modelId + '" --optimization_status completed --notes "$(cat ' + ctx.adaptation_path + '/optimization_notes.json)"。',
      ].join('\n'),
  '返回 schema：status / speedup_ratio / optimization_kind / notes_path / failure_reason。',
].join('\n')

const opt = await agent(optPrompt, {
  agentType: 'npu-optimizer',
  schema: OPT_SCHEMA,
  phase: 'Optimization',
  label: 'optimize',
})
log('Optimization: status=' + opt.status + ' kind=' + opt.optimization_kind + ' speedup=' + opt.speedup_ratio)

if (opt.status !== 'completed') {
  log('Optimization 非 completed，终止主流程。')
  return { model_id: modelId, stage: 'Optimization', status: opt.status, opt }
}

// ───────────────────────────── Stage 6: BusinessBenchmark ─────────────────────────────

phase('BusinessBenchmark')
log('BusinessBenchmark: NPU baseline/perf + CUDA baseline + business_summary.json')

const bizPrompt = [
  commonPreamble(ctx),
  '',
  '你是 business-benchmark。任务：第四阶段业务测评（参考 business_benchmark/scripts/*）。',
  REPLAY
    ? [
        '【REPLAY】不重跑 NPU/CUDA。只读验证现有工件：',
        '1. 领任务：' + BOARD + ' assign_business_benchmark_task --agent_id business-benchmark-1。',
        '2. 只读检查：business_benchmark/scripts/check_business_benchmark_run.py --adapt ' + ctx.adaptation_path + ' 通过。',
        '3. 读 ' + ctx.adaptation_path + '/business_summary.json 取 npu_speedup_ratio / vs_cuda_latency_ratio。不修改文件、不跑远端 CUDA。',
        '4. 若通过：' + BOARD + ' update_business_benchmark_status --model_id "' + modelId + '" --business_benchmark_status completed --notes "$(cat ' + ctx.adaptation_path + '/business_summary.json)"。',
      ].join('\n')
    : [
        '1. 领任务：' + BOARD + ' assign_business_benchmark_task --agent_id business-benchmark-1。',
        '2. 【custom-repo 必须】手写 business_benchmark_config.json 固定画像；自己写 business_run.py（复用 accuracy_run.py loader），禁止用 manager run-npu/print-remote-command 自动生成（会覆盖 business_run.py+画像）。直接 .venv/bin/python business_run.py --scenario {npu_baseline|npu_perf|cuda_baseline} 执行。business_run.py 结构按 CODEX_PATTERNS 的模板（env bootstrap self re-exec + 两段 patch + 复用 loader + 三 scenario + steady_state 计时 + mesh_statistics 质量）。',
        '3. 本机 NPU 一律 uv run --extra ascend ...（或 uv 不在 PATH 时 .venv/bin/python 等价）；远端 CUDA 优先 business_benchmark_manager.py run-remote-cuda（SSH），失败降级 print-remote-command / wait_cuda。远端 CUDA 必须先 scp NPU baseline business_outputs_*.pt 作跨设备质量 ref（否则 cosine=0.0）。',
        '4. npu_perf 必须继承 model_files/ patch + TASK_QUEUE_ENABLE=1；model_files/ 仅补丁模块无 config.json 不得静默退化为未打 patch baseline。',
        '5. business_metrics_*.json 必备字段见 CODEX_PATTERNS：evaluation_profile/primary_metric/ttft_ms=null/tpot_ms=null/model_source_kind/throughput_qps/num_samples>50（用 52）。',
        '6. 【防工件污染】business_run.py 落盘前校验 num_samples==config.max_samples；smoke/aligned 小样本 run 写独立后缀（如 __2sample_aligned），绝不覆盖 canonical business_metrics_*.json。历史 completed 补跑：旧正式工件先改名 __prev_rule_refresh_<ts> 备份。',
        '7. wait_cuda 前置 gate：business_benchmark/scripts/check_business_benchmark_run.py --adapt ' + ctx.adaptation_path + ' --wait-cuda-npu-only（npu_speedup_ratio>=0.9、质量非全 0、num_samples>50）。',
        '8. 最终 gate：check_business_benchmark_run.py --adapt ' + ctx.adaptation_path + '；business_summary.json 须含正数 npu_speedup_ratio / vs_cuda_latency_ratio、measurement_contract_version、运行时证据（python_executable/package_versions/scenario_command）。',
        '9. 完成后：' + BOARD + ' update_business_benchmark_status --model_id "' + modelId + '" --business_benchmark_status completed --notes "$(cat ' + ctx.adaptation_path + '/business_summary.json)"。',
      ].join('\n'),
  '返回 schema：status / npu_speedup_ratio / vs_cuda_latency_ratio / summary_path / failure_reason。',
].join('\n')

const biz = await agent(bizPrompt, {
  agentType: 'business-benchmark',
  schema: BUSINESS_SCHEMA,
  phase: 'BusinessBenchmark',
  label: 'business',
})
log('BusinessBenchmark: status=' + biz.status + ' npu_speedup=' + biz.npu_speedup_ratio + ' vs_cuda=' + biz.vs_cuda_latency_ratio)

if (biz.status !== 'completed') {
  log('BusinessBenchmark 非 completed（status=' + biz.status + '），终止主流程。')
  return { model_id: modelId, stage: 'BusinessBenchmark', status: biz.status, biz }
}

// ───────────────────────────── Stage 7: Sync ─────────────────────────────

phase('Sync')
log('Sync: human_review_status + git commit')

const syncPrompt = [
  commonPreamble(ctx),
  '',
  '你是 team-lead。任务：收口与同步。',
  REPLAY
    ? [
        '【REPLAY】不 git commit、不改文件。只做状态收口：',
        '1. ' + BOARD + ' update_human_review_status --model_id "' + modelId + '" --human_review_status pending（须 business_benchmark_status=completed）。',
        '2. committed=false，commit_sha 留空，notes 注明 "replay 验证完成，未提交"。',
      ].join('\n')
    : [
        '1. ' + BOARD + ' update_human_review_status --model_id "' + modelId + '" --human_review_status pending（须 business_benchmark_status=completed）。',
        '2. git commit：只纳入 adaptation 下的源码、NPU 修改、cannbot operator 源码、构建/验证脚本和必要文档；禁止提交 output/、build/、kernel_meta/、.so、缓存、临时日志和 wfval 预览。只提交当前模型相关 adaptation。',
        '3. 提交信息注明模型、cannbot 算子清单、各阶段 gate 通过情况。',
      ].join('\n'),
  '返回 schema：committed / commit_sha / human_review_status / notes。',
].join('\n')

const sync = await agent(syncPrompt, {
  agentType: 'team-lead',
  schema: SYNC_SCHEMA,
  phase: 'Sync',
  label: 'sync',
})
log('Sync 完成：committed=' + sync.committed + ' sha=' + sync.commit_sha)

// ───────────────────────────── 汇总 ─────────────────────────────

return {
  model_id: modelId,
  adaptation_path: ctx.adaptation_path,
  device: ctx.device,
  stages: {
    preflight: { env_ok: env.env_ok, device: env.device },
    adaptation: { status: adapt.status, path: adapt.adaptation_path },
    operator_gap: { has_gap: gap.has_gap, candidates: gap.candidates.length, ascend_c: ascendOps.length },
    cannbot_dev: integration
      ? { integrated: integration.ops_integrated, e2e_stable: integration.e2e_stable }
      : { skipped: true },
    benchmark: { status: bench.status, num_samples: bench.num_samples },
    optimization: { status: opt.status, kind: opt.optimization_kind, speedup_ratio: opt.speedup_ratio },
    business_benchmark: { status: biz.status, npu_speedup_ratio: biz.npu_speedup_ratio, vs_cuda_latency_ratio: biz.vs_cuda_latency_ratio },
    sync: { committed: sync.committed, commit_sha: sync.commit_sha, human_review_status: sync.human_review_status },
  },
}
