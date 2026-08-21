# 断点续训（Resume）

长 CPT 中断后从断点继续，避免从头重训。需保存/恢复 4 样东西：**模型权重 + 优化器状态(m,v) + step 计数 + lr schedule 进度 + DistributedSampler epoch**。

## 何时存（断点续训开关默认关闭）
- **断点续训开关 `RESUME=1` 默认关闭**（`RESUME=0`）。只有明确要断点续训时才开启周期保存。
- 开启后**按时间基准**周期保存（不是固定步数）：保存间隔 `= max(15 分钟, 预估总时长/5)`，保证训练期间**最多 5 次**、**每次间隔 ≥15 分钟**。预估总时长 `= NUM_STEPS × EST_STEP_S`（`EST_STEP_S` 预估每步秒数，默认 10s）。
- 训练正常结束时**总是**存一次最终 ckpt（与开关无关，供评估与后续续训）。

## 周期保存规则（实现）
```python
RESUME = os.environ.get("RESUME", "0") == "1"          # 总开关，默认关闭
EST_STEP_S = float(os.environ.get("EST_STEP_S", "10")) # 预估每步秒数
CKPT_MIN_INTERVAL_S = float(os.environ.get("CKPT_MIN_INTERVAL_S", "900"))  # 最小间隔 15min
CKPT_MAX_SAVES = int(os.environ.get("CKPT_MAX_SAVES", "5"))                # 最多保存次数
_save_interval_s = max(CKPT_MIN_INTERVAL_S, NUM_STEPS * EST_STEP_S / CKPT_MAX_SAVES)
# 循环里（时间基准，非固定步数；最后一步交给结束保存，避免重复）：
last_save_t = t0
for step in range(...):
    ...
    if RESUME and (time.time() - last_save_t >= _save_interval_s) and step < NUM_STEPS - 1:
        _save_ckpt(step + 1); last_save_t = time.time()
```

## 存什么
```python
# rank0 存（DDP/FSDP2 都适用；FSDP2 权重先 full_tensor 聚合）
ckpt = {
    "step": step + 1,                 # 下次从这步开始
    "model": <full state_dict>,       # model.module / full_tensor 聚合
    "optimizer": optim.state_dict(),  # 含 m,v
    "lr": lr,                         # 当前 lr（或靠 step 重算）
    "rng": torch.get_rng_state(),     # 可选，复现性
}
torch.save(ckpt, os.path.join(OUT, "ckpt_latest.pt"))
```

## 载什么
```python
if os.path.exists(ckpt_path) and RESUME:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # NPU optimizer state 需 weights_only=False（见 pitfalls #21，否则默认 weights_only=True 拒载）
    model.load_state_dict(ck["model"], strict=False)
    optim.load_state_dict(ck["optimizer"])
    start_step = ck["step"]
    # lr schedule 靠 start_step 重算（lr_at(start_step)）
    if rank == 0: print(f"[resume] from step {start_step}", flush=True)
else:
    start_step = 0
```
注意：`optim.load_state_dict` 后需把优化器状态搬到正确 device（DDP/FSDP2 通常自动处理，否则手动 `.to(device)`）。

## sampler epoch
DistributedSampler 用 `set_epoch(epoch)` 控制 shuffle。resume 时传**已跑过的 epoch 数**（或用 start_step 推导），保证续跑数据顺序不重复。
```python
epoch_done = start_step // (nb * 1)   # 简化：按已跑 step 推
sampler.set_epoch(epoch_done)
```

## 关键约束（昇腾/PyTorch）
- **FSDP2**：模型权重存 `full_tensor()` 聚合全量（见 pitfalls #20）；优化器状态 `optim.state_dict()` 可直接存（已是各 rank 本地分片，load 后一致）。resume 时 FSDP2 重新 fully_shard 后 load 优化器 state。
- **DDP**：`model.module.state_dict()` rank0 存；优化器各 rank 相同（梯度 all-reduce 平均），直接存/载。
- **NpuFusedAdamW**：optim.state_dict() 可存；resume 时同样 load。
- **lr schedule**：用 step 重算 `lr_at(step)` 比存 lr 值更稳（避免 schedule 与 step 不同步）。

## 与评估 ckpt 的区别
- 评估用 ckpt（`cpt_model_state.pt`）：只存模型权重（bf16），供阶段 8 评估。
- resume ckpt（`ckpt_latest.pt`）：存模型+优化器+step（fp32），供中断续训。
- 训练结束**两者都存**：评估 ckpt 供评估，resume ckpt 供用户后续想继续训。
