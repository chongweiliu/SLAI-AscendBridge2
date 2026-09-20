---
name: shared-storage-multi-session-interference
description: /mnt/model 共享存储上多个 Claude 会话会互相移动/重组彼此的下载目录
metadata:
  type: project
---

/mnt/model/jiyg/ 是共享存储，同一台机器上常并行运行多个 `claude --dangerously-skip-...` 会话（2026-08-26 实测 ps aux 见 pts/1,3,4,6,8,9,10 共 7 个会话）。

**Why:** 下载 conflux 数据集到 `/mnt/model/jiyg/training-data/data/000/` 时，约 11:14 另一个会话把整个 `data/000` 连同 worker 脚本移进了新目录 `/mnt/model/jiyg/training-data/conflux-chest-ct/`，导致我的 xargs worker 因原路径失效而中断（521/1000 时被打断）。

**How to apply:**
- 在 `/mnt/model/jiyg/` 下工作时，优先使用**带项目/任务前缀的独立子目录**（如 `training-data/conflux-chest-ct/`）作为工作根，不要直接把文件铺在 `training-data/` 一级目录，以免被其他会话"整理"。
- 下载脚本检查中断后，先 `find /mnt/model/jiyg -name <自己的文件>` 全盘定位——文件可能被移动而非删除。
- 长时间后台下载会受其他会话干扰，worker 脚本应基于"按相对路径 + 可重定位 BASE"设计，移动后 `sed` 改 BASE 即可续传。
