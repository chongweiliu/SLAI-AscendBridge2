# 调度平台部署

不得假定平台是 KTP。平台接入、资源域和 manifest 契约必须由用户在当前对话提供或确认。

## 最小必填信息

只向用户收集平台无法安全推断的契约：

1. 平台、CLI/API、连接/context、认证方式。
2. queue/project/namespace，以及可采用的 job schema/profile。
3. 整节点/共享粒度、NPU 计量单位、`allocation_npu_per_node`。非 PD 还需
   节点数；PD 直接由用户给出的 `P + D` 得到节点数。
4. schema/profile 不能提供时才询问镜像和模型挂载。

授权接入后自动读取并验证 schema/profile 中的 CPU、内存、最长运行时间、
host network、rank 交换和服务暴露默认值；只有缺失或与实际资源冲突时才
追问。不要让用户重复填写 CLI 可以可靠读取的字段。

授权接入后只读验证认证、队列、资源、镜像和 volume。自动输出完整配置，
并在末尾询问“是否执行部署？”。确认前禁止真实提交；确认后生成无 token
manifest，先 dry-run，再提交并解析 job ID。Pending 使用有界等待；gang
不可调度连续复现两次后停止任务并清理。

## KTP

- 单机用 `vcjob`，多机用 `acjob`。
- 多机整节点申请值与 vLLM 实际运行 NPU 数分开记录。
- 多机 ACJob 默认采用整组准入：`min_available` 必须等于所有 task 的
  `replicas` 总和，每个 task 的 `min_member` 必须等于该 task 的
  `replicas`。提交前运行
  `python scripts/validate_ktp_manifest.py job.yaml --require-gang`。除非用户明确
  要求弹性/部分准入且运行时本身支持缩容，否则禁止让 2P2D 等固定 world
  任务占用部分节点长期等待。
- 某些 KTP CLI 版本会读取 task 的 `min_member`，却在提交合并阶段把 YAML 的
  `min_available` 覆盖成 0。固定 world 任务的 dry-run 和真实提交都显式追加
  `--min-available TOTAL_REPLICAS`，把 dry-run 保存后再运行
  `validate_ktp_manifest.py ... --dry-run-output FILE`；只有渲染结果中的
  `minavailable` 与 manifest 一致才允许真实提交。
- manifest 显式设置经平台确认的 `host_network`、`enable_ssh`、volume、queue 和服务暴露。
- 执行 `ktp submit -f job.yaml --dry-run` 后再 `ktp submit -f job.yaml`。
- 用 `ktp get`、`ktp pods` 和 `ktp logs` 验证全部 Pod。
- host-network 多机任务失败后优先停止并用冻结 manifest 重新直接提交；不要假定 `ktp restart` 会保留全部网络字段。
- ACJob 可能单独重启或重新调度某个 Pod。不得使用永久存在的 `rankN_ip` 文件作为唯一屏障；这会把旧 IP/旧进程与新 Pod 混成一个 DP/HCCL 组。
- 对共享可写挂载上的多机任务，用 `scripts/stable_cluster_supervisor.py` 包裹每个节点的角色入口。为每次冻结提交使用唯一 `--state-dir`，要求全部 rank 的 session/IP 连续稳定后才启动。运行中任一 rank session/IP 改变时立即退出；心跳或共享存储短暂不可用时用 `--unavailable-grace-seconds` 有限容忍，超过宽限期才退出，避免权重预取产生的 I/O 抖动触发误重启。保持有界的 `--child-tail-lines`，让 supervisor 在角色退出后把末尾输出持久化为 `events/rank-*-child-tail.log`；结合 `rank-*.jsonl` 判断首发 `role_exited` 与其他 rank 因 `generation_changed` 协调终止产生的次生错误。
- 同一 Pod 启动多个本地 DP 服务时，使用 `scripts/launch_online_dp.py`。任一实例异常退出必须终止同节点兄弟实例并原样传播非零退出码；若启动器只等待子进程而不检查退出码，平台会把实际失败错误标成 `Succeeded`，禁止提交。
- 角色入口同时启动服务和 Proxy 时使用 fail-fast 监督：任一子进程退出就终止其余子进程并让角色入口退出。禁止裸 `cmd1 & cmd2 & wait` 长时间掩盖单个服务失败。
- 入口可以保持 `set -euo pipefail`，但昇腾/CANN/ATB 等厂商 `set_env.sh` 常不兼容 nounset；仅在 `source` 这些已知脚本前后使用 `set +u`/`set -u`，并在任何 `source` 之前输出启动标记，避免静默重启。
- 严格 shell 下不得无保护调用镜像中的可选探测工具。网络接口探测先执行 `command -v ip`；探测命令或管道失败必须显式容忍并回退到已确认的 NIC。服务入口优先使用预检确认的 `vllm` 绝对路径，再查询 PATH；两者都不存在时明确报错退出，不能留下含糊的 127。
- 失败或结束时确认 `Active Pods: 0`；失败任务执行 `ktp stop JOB_ID`。

集群域名和私网地址加入 `NO_PROXY/no_proxy`，但不要输出 token 或未过滤环境。
