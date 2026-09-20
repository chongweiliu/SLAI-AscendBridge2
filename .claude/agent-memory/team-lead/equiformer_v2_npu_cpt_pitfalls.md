---
name: equiformer-v2-npu-cpt-pitfalls
description: EquiformerV2/fairchem v1 NPU CPT 实战：fairchem 1.10 与 torch 2.10 兼容、OCPTrainer 全栈 NPU 死锁（pin_memory 吃满 HBM）、能量 per-element 参考口径（extxyz raw vs 模型去参考差 500 eV，最小二乘反解 ref 表）、ase 读 xz O(n²)、EMA 污染 best ckpt、等变模型 batch 超线性显存
metadata:
  type: project
---

# EquiformerV2 (fairchem v1) 昇腾 NPU CPT 实战（2026-09-02, 单卡 910）

## 任务结论
OC20 200K 子集 80%（16k 帧）CPT EquiformerV2-31M（All+MD 官方 ckpt）：held-out 能量 50.4→8.8 eV（5.7×，eval10 十帧 8/10 大幅改善）；力指标口径分歧（原子加权 2.5× 改善 / 帧级 2× 退化——能量残差主导优化的发现）；推理零开销。工作区 `training-ws/equiformer_v2-cpt/`。

## 可迁移经验

### 1. fairchem v1 在新 torch 上的兼容（一串坑）
- **fairchem v2 与 v1 模型断代**：EquiformerV2×OC20 ckpt 属 v1（fairchem-core==1.10.0），v2 不兼容；1.10 pin `torch~=2.4.0` → `--no-deps` 装包 + 按 import 链手动补（wandb/tensorboard/submitit/lmdb/orjson/torchtnt）。
- **trainer 名断代**：ckpt config 的 `equiformerv2_forces` 在 1.10 注册表不存在 → 强制 `'ocp'`。
- **setup_imports 无差别 import 全部模型**：dimenet 要 torch_sparse、gemnet 要 torch_scatter.utils → 精确注册需要的 3 个模块 + torch_scatter/torch_sparse 最小 shim（EquiformerV2 主模型其实不用这俩）。
- **LRScheduler 自身 bug**：显式 `scheduler` 键时不生成 lr_lambda → scheduler: Null 恒定 lr。
- **BalancedBatchSampler** 要 LMDB metadata，ase 数据集没有 → load_balancing: None。
- **`data.energy = energy` 裸 python float**：NPU aclnnL1Loss 不支持 float64 → patch `torch.tensor(energy, dtype=torch.float32)`。

### 2. OCPTrainer 全栈 NPU 死锁（与 MatterSim 路线相反的选择）
- **症状**：trainer.train() 启动后 HBM 涨满 64GB、AICore 归零、进程静默挂死（无 traceback 无 plog）；chip 残留 45GB 显存不释放（进程死了驱动没回收）→ 换卡。
- **根因之一**：base_trainer 硬编码 `pin_memory=True`（预取 batch pin 进 HBM）；patch 条件化后仍有其它路径死锁（未完全根因化）。
- **务实解法**：**MINREPRO 最小复现证明模型直训仅 0.5GB@batch4** → 放弃 trainer 栈，自写轻量循环（复用官方模型/collater/loss 语义）。**教训：官方训练栈在非官方硬件上整体不可用时，拆出"模型+collater+loss 语义"自管循环往往 30 行解决**（MatterSim 同款路线）。
- **等变模型 batch 超线性显存**：batch 4=0.5GB 但 batch 16/32 在 grid 变换 einsum（`bai,zic->zbac`）处 OOM@59GB——batch 只能按官方单卡值（8）。
- **变长 batch 评估**：OC20 结构 36-87 原子，评估 batch 4 + 每批 `torch.npu.empty_cache()`（碎片化）；`aten::scatter_reduce.two_out` NPU 不支持回退 CPU（功能正常仅慢）。

### 3. 能量口径：per-element 参考能（本任务最大根因，通用方法论）
- **症状**：base 模型能量误差 ∝ 原子数（168~540 eV），官方 OCPCalculator 同样错（排除自身代码）；CPU=NPU 数值一致（排除硬件）。
- **根因**：All+MD ckpt 输出"去 per-element 参考"能量，extxyz 标签是 raw DFT，差 Σ N_el×ref_el。
- **定位法**：①逐帧打印 pred/true 发现 err/natoms 随元素组成波动 ②**最小二乘反解**：`E_raw − E_pred = A·ref`（A=每帧元素计数矩阵）——60 帧解 50 元素，残差 368→1.99 eV 即证实。
- **修复**：训练 target 减 Σref、评估预测加回（统一 raw 口径）。**更精 ref 表（≥500 帧拟合/官方 fit_references 脚本）可把残差压到 0.2 eV 级**。
- **通用**：跨口径（去参考 vs raw）能量差可达数百 eV；误差 ∝ 原子数是 per-element 偏移的指纹。

### 4. 数据/工程坑
- **ase 读 xz 多帧 O(n²)**：xz 不可 seek，逐帧定位反复重新解压（rchar 11.4GB 读 1.5GB 数据，40 分钟读不完 4 文件）→ 先 `xz -dc` 解压再读（5000 帧 3s，100×）。
- **ase_read_multi 双 bug**：get_atoms 每次访问 read 整个文件（16k 帧 O(n²) 必死）；index_file 模式 `ids = [...]` 覆盖不 extend（多文件只剩最后一个）→ 注册内存数据集（继承 AseAtomsDataset，一次读入直接索引）。
- **EMA 污染 final ckpt**：final 用 `ema.average_parameters()` 保存的权重（decay 0.999 × 2000 步含 25% 早期权重）比 step2000 当前权重差 5×（F 0.0215→0.107）→ **评估用 best（非 EMA）ckpt，EMA 权重单独存**。
- **力指标口径敏感**：原子加权（batch mean）vs 帧级（frame mean）聚合对同一模型可差 5×；OC20 官方 metrics 族是 free-atom/原子加权口径。报告 MLIP 力指标必须写明聚合口径。
- **NPU patch 必须防御式**：`getattr(torch, 'npu', None) is not None and torch.npu.is_available()`——否则无 torch_npu 的 CPU 进程（官方 calculator cpu=True）直接崩。
- **覆盖运行中脚本文件** → bash 逐行读取错位进程死；pkill 自匹配（三犯）按 PID 杀。

相关：[[mattersim-npu-cpt-pitfalls]]（同为 MLIP 范式，MatterSim 走成 Potential API 复用路线，fairchem 走成自管循环路线——两条路线的取舍见各文件）、[[proteinbert-keras2torch-cpt]]
