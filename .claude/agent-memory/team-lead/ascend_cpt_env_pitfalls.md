---
name: ascend-cpt-env-pitfalls
description: 昇腾 CPT（继续预训练）环境踩坑：950 系列（950PR/950DT）新芯片 torch_npu 不支持需升级、容器内存限制致 torch.load 大 ckpt OOM、torch2.12 后 torchvision/torchaudio ABI 错配
metadata:
  type: project
---

# 昇腾 CPT 环境踩坑（2026-08-24，Ascend950PR + Qwen3.5-4B CPT 实跑）

**背景**：在 Ascend950PR（128GB HBM，CANN 9.0.0-beta.2）上跑 Qwen3.5-4B 继续预训练 100 步，连环遇到 5 个环境/工程问题。其中 3 个是普遍问题，已沉淀到 [[ascend-torch-cpt]] skill 的 pitfalls.md（#42-45）与脚本模板。此文件记要点供后续协调 CPT 任务时快速判断。

## 1. 新芯片 soc：torch_npu 不支持 Ascend950 系列（950PR/950DT）→ 升级 torch+torch_npu（普遍）
- **症状**：`torch.npu.set_device(0)` 报 `Unsupported soc version: Ascend950PR 9579` 或 `Ascend950DT xxxx`；`device_count()`/`get_device_name()` 能返回（不触发 lazy_init）但真正初始化/算子即崩。
- **根因**：镜像预装 torch_npu 2.7.1.post2.dev 太旧，soc 映射表无 950 系列。**950PR 与 950DT 两款都是 950 系列 2026 新芯片**，CANN 9.0.0-beta.2 已支持（`platform_config` 下有 `Ascend950PR_*.ini` 与 `Ascend950DT_*.ini`），但旧 torch_npu 都不识别——两款同根因、同解法。
- **解法**：`pip install torch==2.12.0 torch_npu==2.12.0`（华为 mirror）。torch_npu wheel 精确 pin torch，会一并升级。升级后 matmul 正常、HBM free 131.8GB。**950PR 已实测可行**；**950DT 同系列同 CANN beta，预期同方案，待 950DT 实机验证** set_device+matmul。
- **判定**：device_count >0 ≠ 可用；必须 set_device + 真实算子跑通。env_probe 用 `npu-smi info -t board -i 0` 的 Chip Name 匹配 `Ascend950(PR|DT)` 即 950 系列。
- **Why**：950 系列是 2026 新芯片，旧 torch_npu 必不识别。未来新型号同理。
- **How to apply**：env_probe 阶段必实测 set_device+matmul，不能只看 device_count；遇 Unsupported soc（950PR 或 950DT）直接升级 torch_npu。

## 2. torch2.12 后 torchaudio/torchvision ABI 错配 → 首选装正式版，stub 仅 fallback（普遍，skill 原 #39 只覆盖 torchaudio，漏 torchvision）
- **症状**：`from transformers.models.qwen3_5 import Qwen3_5ForCausalLM` 报 `torchaudio undefined symbol torch_library_impl` 或 `torchvision::nms does not exist`。导入链：多模态文本头 → modeling_utils → loss_utils → image_transforms → image_utils → torchvision.io / audio_utils → torchaudio。
- **根因**：升级 torch 后系统旧 torchvision/torchaudio 按旧 torch 编译，符号错配。**torchvision/torchaudio 是 PyTorch 官方库，不是昇腾的库，CANN 社区不发布"昇腾版"。**
- **解法（首选正式库，已实测可行）**：装匹配 torch 的正式版——`pip install torchvision==0.27.1`（+cu130，匹配 torch 2.12.1+cu130）+ `torchaudio==2.11.0`（华为 mirror 最新，实测与 torch 2.12.1 import 兼容）。装完 `from transformers.models.qwen3_5 import Qwen3_5ForCausalLM` 不崩、模型可实例化，**无需打桩**。版本对应：torch 2.12↔torchvision 0.27.x；torchaudio 版本号与 torch 主版本对齐（2.12.0↔torch 2.12，华为 mirror 暂无 2.12，2.11.0 实测兼容）。
- **解法（fallback 打桩）**：仅当匹配正式版不可得（torch 太新无配套 release / 离线无 mirror）才用 stub：`sys.meta_path` 插 `_StubFinder` 同时拦 torchaudio+torchvision，torchvision 需带 `__getattr__` sentinel 的 `_StubModule`。
- **Why**：打桩是升级 torch 后没同步升 torchvision/torchaudio 的权宜之计；正确做法是装匹配正式版。模板 `try: import; except: stub` 防御式——正式版可用 no-op，崩了才兜底。
- **How to apply**：env_probe 验证 `from transformers.models.<mm> import XxxForCausalLM` 不崩才算依赖就绪；崩了先装匹配正式版，装不到再 stub。

## 3. 容器内存限制 + torch.load 大 ckpt 到 CPU → OOM SIGKILL(137) 静默死（普遍）
- **症状**：eval 时 `torch.load(ckpt, map_location='cpu')` 16.8GB + fp32 模型 16GB = 32GB+，进程被 SIGKILL(137)，**无 traceback**（日志突然断在加载行）。系统 `free -g` 显示 717GB available 误导。
- **根因**：容器（K8s/Docker）cgroup 内存 limit 远小于宿主机视图，且 `/sys/fs/cgroup/memory.max` 可能不在标准路径查不到。
- **解法**：建 fp32 CPU 模型 + `torch.load(ckpt, map_location='npu:0')` 把 state 放 NPU（131GB free）+ `load_state_dict` 跨设备 in-place copy，CPU 峰值仅模型本身 16GB。
- **判定**：exit 137 + 无 traceback + 大 ckpt 加载 = 强烈暗示容器 OOM。
- **Why**：容器化 NPU 环境普遍有 cgroup 内存限制；torch.load 反序列化峰值翻倍（state + 模型 + 复制）。
- **How to apply**：eval/加载大 ckpt 一律 `map_location='npu:0'`，别 `map_location='cpu'`。torch.save 不受影响（流式写盘峰值低）。

## 附：meta+assign+to_empty 路线两个陷阱（#44 的错误尝试，已记 #45）
- `to_empty(device)` 会把 assign 赋入的真实权重重置为未初始化（模型变随机，NLL=ln(vocab)）。
- meta 建模型跳过 RoPE `inv_freq` 等 non-persistent buffer 初始化 → 前向崩 `Cannot copy out of meta tensor`。
- 解法：大模型 eval 加载别走 meta，用上面 #3 的"正常建模型 + NPU 加载 + 跨设备 copy"。

## 实跑结果（Qwen3.5-4B，100 步 CPT）
- 训练：loss 3.17→1.58，first5=3.04→last5=1.56，275.92s（~2.4s/step 稳态），NpuFusedAdamW + SDPA + 梯度检查点 + fp32+bf16 autocast。
- 评估（10 条 held-out）：PPL 44.08→6.21，acc 0.39→0.63，F1 0.059→0.34，recall 0.156→0.347（全部改善）。
- 多模态 remap：Qwen3.5-4B tie=True，strip `model.language_model.`→`model.`，丢 visual/mtp，miss=1（lm_head 正常），调 tie_weights()。4.206B 参数。
- 工作目录：`training-ws/Qwen3.5-4B-cpt/`，产物含 cpt_model_state.pt(16.8GB)、eval_results.json、loss_curve.png（公网 https://d.uguu.se/CzfzYgFw.png）。

## 已沉淀位置
- pitfalls.md #42（soc 升级）、#43（torchvision stub）、#44（容器 OOM→map_location=npu）、#45（meta 路线陷阱）
- cpt_train.py.tmpl / eval_cpt.py.tmpl：stub 扩展含 torchvision + sentinel _StubModule
- eval_cpt.py.tmpl：cpt ckpt `map_location='cpu'` → `map_location=device`（避免 OOM）
- SKILL.md 阶段1：加新芯片 soc 升级提示
