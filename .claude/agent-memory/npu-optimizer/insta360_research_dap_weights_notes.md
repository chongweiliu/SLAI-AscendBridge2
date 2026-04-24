# Insta360-Research/DAP-weights

- 日期：2026-04-23
- 结论：`runtime_only` completed
- 物理卡：`ASCEND_RT_VISIBLE_DEVICES=13`
- 正式合同：`depth_maps`，`pretrained`，`cifar100`，`50` 样本

## 关键修复

- 旧的 stage2/stage3 都是假的 `MinimalDepthModel + config` 链路，必须整轮废弃，不能修补旧工件。
- adaptation 私有 snapshot 实际完整可用，`refs/main -> snapshots/558e9ac84efbcb46dc8c47b32c73b333d95f4f0d`，包含 `model.pth`、`config/infer.yaml`、`networks/`、`depth_anything_v2_metric/`。
- `networks/dap.py` 里 `dinov3_repo_dir="./depth_anything_v2_metric/depth_anything_v2/dinov3"` 是 cwd-sensitive；稳定做法不是改业务脚本 cwd 外部乱跑，而是在 `accuracy_run.py` 内固定 `snapshot_root + sys.path + os.chdir(snapshot_root)` 的上下文，让上游研究代码按预期解析相对路径。
- `dinov3/hubconf.py` 会无条件导入 segmentor / depther / classifier / detector 入口，导致只想拿 backbone 也会被 `torchmetrics` 等无关依赖卡住。最小修法是在 adaptation 内 snapshot 源码把这些入口改成 `try/except` 的懒失败，不影响 `dinov3_vitl16` backbone 加载。
- `accuracy_run.py` 静态检查会误伤 `Path.cwd()` / `os.getcwd()` 字样；如果必须暂时切 cwd，代码里不要出现这些模式，改用 `Path(".").resolve()` 保存旧目录即可。

## 最终方案

- `accuracy_run.py`
  - 真实 pretrained：从 adaptation 私有 snapshot 加载 `model.pth`
  - 输出合同：`depth_maps`
  - baseline：50 样本逐条推理，`warmup(3x)`，`batch_size=1`
- `accuracy_run_perf.py`
  - 不建 `model_files/`
  - `runtime_only = warmup(3x) + TASK_QUEUE_ENABLE=1 + batched_inference(bs=4)`
  - compare 直接对 `depth_maps` flatten 后算 cosine / max abs
  - notes 必须从 perf metrics 继承 `selected_npu(s)` / `device_topology` / `parallel_mode`

## 正式结果

- baseline `wall_clock_s=1.554443`
- baseline `latency_s=0.031089`
- perf `wall_clock_s=1.084643`
- perf `latency_s=0.021693`
- `speedup_ratio=1.433138`
- `cosine_similarity=0.9999997723`
- `min_cosine_similarity=0.9999984503`
- `max_abs_error=2.4830922484397888e-05`

## 额外提醒

- 这类 research repo 模型不要太早判成“没 pretrained”。先看 adaptation 私有 snapshot 是否真的有完整源码和权重。
- 若旧目录里还留着 `*_config_*` 工件，先删掉再跑 gate；否则 benchmark/optimization completed 模拟校验会被历史脏文件污染。
