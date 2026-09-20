---
name: mattersim-npu-cpt-pitfalls
description: MatterSim NPU CPT 实战：官方代码 3 处 NPU patch（batch_to_dict 静默降级/threebody 缺文件 shim/mp_api 懒加载）、shift-only scaling 策略、high_level_water 数据在 ModelScope 镜像、MLIP 评估禁 no_grad
metadata:
  type: project
---

# MatterSim 昇腾 NPU CPT 实战（2026-09-02, 8×910, 288s 训练完成）

> **2026-09-02 更新**：本文件核心经验已固化到 ascend-torch-cpt skill（pitfalls #91–#94 + SKILL.md MLIP 范式 + robust_download sha256 支持，MR !45），以 skill 为准；此记忆保留会话上下文与 MatterSim 专属细节（3 处 patch 位置、shim 写法、官方协议参数）。

## 任务结论
high_level_water.xyz（10 帧液态水，全电子口径）80% CPT MatterSim-v1.0.0-1M：held-out 能量 132→1.4 meV/atom（94×）、力 0.471→0.055 eV/Å（8.5×），train≈held 无过拟合，推理零开销。工作区 `training-ws/mattersim-cpt/`。

## 可迁移经验

### 1. MatterSim 官方代码的 3 处 NPU patch（src/mattersim/ 内）
- **`potential.py batch_to_dict` 静默降级**：`torch.cuda.is_available()=False` 且无 mps 时把 device="npu" 改成 "cpu"（只考虑 cuda/cpu/mps 三后端，pitfalls #40 家族新变体）。症状：`F.linear` 报 device mismatch 而模型明明 .to('npu') 了。解法：补 `import torch_npu; torch.npu.is_available()` 探测。
- **`datasets/utils/threebody_indices.py` 上游缺失**：converter 顶部 import 的 cython 版不在 GitHub 仓库，但官方 torch 等价版 `compute_threebody_indices_torch` 存在且签名一致——写 numpy↔torch shim 转发即可（`from .threebody_indices_torch import compute_threebody_torch`）。
- **`utils/atoms_utils.py` 顶部 `from mp_api.client import MPRester`**：装 mp-api 会拉 scipy≥1.15（make_splrep）与旧 scipy 冲突；训练链路根本不用 MP 下载——改 try/except 懒加载。
- 官方其余部分（M3Gnet 前向/e3nn 球谐/scatter_add_/autograd.grad 力微分/EMA）在 NPU 上**原生兼容**，0 patch。

### 2. MLIP（机器学习力场）CPT 的评估与 scaling 原则
- **评估函数绝不可 @torch.no_grad()**：力 = -dE/dx 需要 autograd 图，no_grad 下 `autograd.grad` 报 "does not have a grad_fn"（官方 predict_properties 本就不用 no_grad，自己包装反而翻车）。
- **re_normalize 全量重拟合会破坏力基线**：scale_key='per_species_forces_rms' 重设 scale 后 F=-scale·d(raw)/dx 整体偏移（base 力 0.448→1.515）。**正确姿势：只重拟合 shift**（`shift_key='per_species_energy_mean_linear_reg'` + `scale_key=None` + `init_scale=原模型normalizer.scale`）——shift 吸收能量基准差（d(shift)/dx=0 不影响力），力基线不动，base/CPT 指标直接可比。
- **能量基准差是物理事实不是 bug**：不同泛函/口径（全电子 vs PAW-PBE）能量差可达 ~150 eV/atom，先查 per-atom 能量量级再判断"模型不准"；力不受能量平移影响，是跨口径最可靠的对比指标。
- 官方 finetune 协议：HuberLoss(delta=0.01) × [E/atom + F×1.0]、Adam(lr=2e-4, eps=1e-7) + StepLR(10,0.95) + clip1.0 + EMA(0.99)——lr 2e-4 对域适应 CPT 稳定（400ep 无发散）。

### 3. 数据/权重获取
- **`high_level_water.xyz` 在 ModelScope `OneScience/Mattersim` 镜像的 `data/` 目录**，GitHub microsoft/mattersim 仓库反而没有（只有 data/benchmarks/*）；官方 `finetune_config.yaml`（也在 ModelScope scripts/）注明它与 mattersim-v1.0.0-1M.pth 是官方微调示例组合。**找官方示例数据先查 ModelScope 镜像全目录**。
- ModelScope API（`/api/v1/models/<ns>/<name>/repo/files?Root=<dir>` 递归列目录）给 **sha256** 不给 md5；robust_download 的 get 模式 md5 参数传 sha256 会误报校验失败（模板改进点：支持 sha256）。
- codeload 大 tarball（~24MB+，含 docs gif）当日连续两次截断；`api.github.com/git/trees?recursive=1` 列清单 + raw 逐文件 fetch（重试循环）更稳。空 `__init__.py`（0 字节）会被 `-s` 判空误报"缺失"。

### 4. 通用细节
- 小数据 MLIP CPT：10 帧 × 192 原子 = 1536 原子（力样本充足）足以支撑有效 CPT（力 8.5×改进）；早停以 held-out 力 MAE 为准。
- mattersim 依赖安装：torch/numpy 双 pin（`pip install ... "torch==2.10.0" "numpy==1.26.4"`）防止 torch_geometric/pymatgen/torchmetrics 连锁升级破坏 torch_npu ABI；不装官方全量依赖（atomate2/phonopy/azure/torch-sim-atomistic 非训练必需）。
- `Potential.from_checkpoint(device='npu')` 原生可用（torch.load map_location + M3Gnet(device=).to()）；单卡训练绕开官方 DDP：is_distributed=False 路径完整存在。

相关：[[proteinbert-keras2torch-cpt]]（同为非文本范式 CPT + 权重迁移）、[[hf-mirror-download-technique]]
