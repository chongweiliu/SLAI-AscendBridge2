---
name: powerflownet-npu-cpt-pitfalls
description: PowerFlowNet NPU CPT 实战：官方 main 与历史 ckpt 代码断代（git 历史对齐法）、surfdrive 限流封禁/PROPFIND zip 错位/urllib hang→curl 后端、Nextcloud 尾段截断、同形 bool 索引 1-D 展平
metadata:
  type: project
---

# PowerFlowNet 昇腾 NPU CPT 实战（2026-09-02, 单卡 910, 训练 3.3min）

## 任务结论
case118v2 四件套 80%（40,000 样本）CPT 官方 9288 ckpt（MaskEmbdMultiMPN 0.357M）：held-out 池 10,000 条 Masked_L2 0.02201→**0.02110（+4.13%）**，eval10 逐条 7/10 改善，base val 0.0219 与官方 0.0220 完美锚定，推理零开销。对比三原则③典型场景（同分布短程 CPT 小幅增益，如实报告）。工作区 `training-ws/powerflownet-cpt/`。

## 可迁移经验

### 1. 官方 main 与历史 ckpt 代码断代（科研包通病，#99 延伸）
- **症状**：load_state_dict 形状 mismatch（mask_embd 129×6 vs 模型 129×4）——main 分支 train.py `assert node_in_dim==4` 与 ckpt save_logs `nfeature_dim=6` 矛盾。
- **解法（git 历史对齐法）**：save_logs 时间戳（2023-06-27）→ GitHub commits API `?until=2023-07-01` 取历史 commit（bc0392a438）→ raw.githubusercontent 拉该代全套源码（数据类+模型+loss）。历史版协议：x 16 列（4 one-hot + 6 值 + 6 mask）、pred_mask=**x 值≠y 的逐元素不等处**（非 bus_type_mask）、edge from/to 从 1 起（内部 -1）、forward 从 x[:, -n:] 取 mask（不依赖 data.prediction_mask——collate 不保留）。
- **How to apply**：任何科研包 ckpt 加载 mismatch，先查 ckpt 内时间戳→拉同代代码，别用 main。

### 2. Nextcloud/surfdrive 大文件下载五坑
- **限流封禁**：48 路持续 ~30min 后服务器完全零响应（连单流都断），冷却 5min 恢复——**并发 ≤24 路**或分批冷却。
- **PROPFIND zip 错位**：href 顺序 ≠ size 顺序，名字-size 配对错位（x/y size 互换返工一次）——**逐文件 Depth:0 查询**。
- **urllib 多线程 hang**：同 URL curl 正常但 urllib 线程池全 hang——**curl 子进程后端**最稳。
- **尾段截断**：28MB 模型固定在 29.5MB 处截断（多来源复现）——服务器侧损坏，换文件。
- **WebDAV 认证**：公开分享 token 即用户名（`-u token:`），PROPFIND 列目录/GET 下载。
- 断点续传分块写入：**r+b + seek(have) + truncate(want)** 精确写（'ab' 追加在重试时重复叠加致块超额）。

### 3. PyTorch 语义与包解析
- **同形 bool 索引是 1-D 展平**：`(N,6)[(N,6)bool] → (K,)` 不是 (K,6)——按列聚合用 `(diff*mask).sum(0)/mask.sum(0)`；官方 masked_select 1-D 对 1-D 是对的。
- **regular 优先 namespace**：PYTHONPATH 里带 __init__.py 的 src/ 会赢过 sys.path.insert(0) 的无 __init__ 目录——给历史版目录补 __init__.py。
- torch 2.10 weights_only 拒载 PyG collated Data（#81 家族）→ weights_only=False。

### 4. 电网潮流 GNN 范式要点
- 输入按母线类型遮挡（slack 给 VmVa 求 PQ / gen 给 Vm 求 VaPQ / load 给 PQ 求 VmVa）——pred_mask 表达"哪些量未知"；KIT 数据 type 编码 1=gen 2=load（slack 特殊层在模型中已注释不用）。
- 历史版 normalize：x[:,4:] 与 y 同 mean/std（6 维物理量）；eval 报物理量 MAE 需 denormalize。
- 0.357M 小 GNN 在 910 上 BS=128 仅 15.4ms/batch（每样本 0.12ms）、训练 13s/epoch——超小模型训练成本可忽略，评估成本反而主导。

相关：[[equiformer-v2-npu-cpt-pitfalls]]、[[mattersim-npu-cpt-pitfalls]]（三案例组成 MLIP/科研包 CPT 全谱）
