---
name: vllm-ascend-auto-deploy
description: 交互式部署 vLLM-Ascend 推理服务。用户要求“帮我部署某模型，权重位于某路径”，或要在本机、SSH 服务器、KTP/其他调度平台上进行单机、多机或 Prefill/Decode 分离部署时使用。负责逐步收集缺失信息、按模型配置规划安全并行拓扑、生成无凭据部署包、预检和提交，并用 OpenAI 兼容 API 完成真实推理验收。
---

# vLLM-Ascend 自动部署

把自然语言请求转换为已验证的运行中服务。只询问缺失字段；不要重复询问用户已提供的模型名和权重路径。

## 对话硬门禁

每次回复只处理第一个未完成 gate，提问后立即结束。gate 未完成时，不调用工具、不读取历史任务或平台配置、不写文件、不预检或提交。

提问必须给出选项清单，不得只抛一个裸问句。每条问题用 `数字编号 + 选项` 的 Markdown 列表形式呈现候选项，选项内括注简短含义，末尾仍只保留一个问句收束，不输出问候、解释、核对过程或下一步预告。固定门禁措辞如下：

- 缺部署规模时只能输出：

  ```text
  请确认部署规模：
  1. 单机部署（在单台机器上运行，无跨节点通信）
  2. 多机部署（跨节点张量并行或数据并行）
  ```

- 单机缺目标时只能输出：

  ```text
  请选择部署目标：
  1. 本机（在当前机器直接启动）
  2. SSH（通过 SSH 部署到指定远程服务器）
  3. 调度平台（提交到 KTP/Slurm/Kubernetes 等调度系统）
  ```

- 多机缺目标时只能输出：

  ```text
  请选择部署目标：
  1. SSH（通过 SSH 部署到多台远程服务器）
  2. 调度平台（提交到 KTP/Slurm/Kubernetes 等调度系统）
  ```

- 多机未确认 PD 时只能输出：

  ```text
  是否启用 PD（Prefill/Decode）分离？
  1. 启用 PD 分离（Prefill 与 Decode 分角色独立部署）
  2. 不启用 PD 分离（统一以单一角色运行）
  ```

- 启用 PD 但未给 P/D 数量时只能输出 `请提供几 P 几 D（例如 2P2D）？`；此 gate 为自由数值输入，无需选项列表，但仍不得附加解释。

输出前在内部逐字核对，不输出核对过程。除上述收束问句外，不得对裸问句做任何扩展。

运行参数默认自动规划，不把 Prefill/Decode 拓扑、作用域、版本、连接器、Direct KV、端口、Proxy、共享路径、网卡或 rendezvous 作为固定问卷。先通过模型配置、镜像、平台 profile 和预检推断；只有缺失或冲突会阻止成功部署时才追问，且一次最多三个。

1. 缺 `model_name`/`model_id`：只补齐缺失项。拿到 `model_id` 后**先进入"模型来源解析（自闭环）"子流程**（见下节），不直接向用户索取绝对路径。
2. 缺 `deployment_mode`：按上节固定措辞输出“单机部署 / 多机部署”选项清单，不得解释或追加问题。
3. 缺 `target`：按上节固定措辞输出目标选项清单（单机三选一、多机二选一）。不得索取节点、账号或平台字段。
4. 多机 PD 门禁：按上节固定措辞输出“启用 / 不启用 PD 分离”选项清单，然后结束；用户启用后只再问一次“几 P 几 D”。`P + D` 作为默认节点总数。
5. 按目标 reference 收集不可安全推断的接入和资源契约。SSH 必须有主机、用户名、认证方式；调度平台必须有或明确授权读取 platform/context、queue/project/namespace、节点数、申请粒度/NPU 单位、每节点申请值、镜像和模型挂载。
6. 接入信息足够后自动读取模型配置、预检环境并规划所有运行参数。只有预检歧义或硬约束冲突时才补问。
7. 输出完整配置摘要，末尾只问 `是否执行部署？`。用户确认前禁止真实提交、启动服务或在 SSH 目标执行变更；确认后不再逐项确认自动参数。

一次最多问三个同批次问题。`multi_node + local` 无效，禁止创造“当前机 + SSH 其他节点”等第四种 target。用户明确说 KTP/Slurm/Kubernetes 时可视为 `target=scheduler` 和对应 `platform` 已知。调度平台其余信息必须由用户在当前对话提供或明确确认；旧任务、CLI context 和环境只能在授权接入后做只读核验，不能替用户选择。

## 模型来源解析（自闭环）

项目要具备完全自闭环能力：`adaptations/` 是模型权重与代码的唯一落地处。**任何部署拿到 `model_id` 后，必须先在 `adaptations/` 查找，命中即复用，未命中再给选项**，不得默认索取外部绝对路径。

1. 调 `scripts/adaptation_utils.model_id_to_adaptation_path(model_id)` 推导 `adaptations/{safe_name}`；检查目录是否存在 `models/` 子目录且含 `config.json` 与至少一个权重分片（`.safetensors`/`.bin`/`.pt` 等）。
2. **命中** → 记录 `model_path = adaptations/{safe_name}/models`、`model_path_source = adaptation_local`、`config_path`、`config_sha256`，跳到预检，不再问用户。
3. **未命中** → 输出选项清单（遵循"追问给选项"规则，不得只抛裸问句）：

   ```text
   在 adaptations/ 下未找到该模型。请选择：
   1. 提供权重绝对路径（直接用外部路径部署，不入库 adaptations）
   2. 自动下载并准备（在 adaptations/{safe_name}/ 下创建 demo.py + pyproject.toml + 下载权重到 models/，可复用于适配/评测/优化全链路）
   3. 跳过本次部署
   ```

4. 选 1 → 追问绝对路径，`model_path_source = external_absolute`，跳过下载。
5. 选 2 → **deployer 内化部分 adapter 能力**：执行 `scripts/prepare_adaptation.py --model-id {model_id}`。该脚本复用 `adaptation_utils` / `run_completed_adaptations.download_model_snapshot` / `get_model_info` / `demo.py.j2` 产出满足 DoD 最小子集的骨架（demo.py + pyproject.toml + README.md + .status.json + output.txt + 下载权重到 `models/`），并 best-effort 登记到 board.db 供后续 adapter/benchmark/optimization 链路复用。完成后 `model_path = adaptations/{safe_name}/models`、`model_path_source = adaptation_local`。deployer 内化的是目录准备 + 权重下载 + 骨架渲染，**不替代**完整 adapter 流程（不做精度评测/优化/业务测评）。
6. 选 3 → 终止。

`model_path` 允许两种形式：`adaptations/{safe_name}/models`（自闭环）或外部绝对路径。`validate_deploy_request.py` 与 `plan_pd_topology.py` 的校验/精确匹配均接受这两种形式；profile 精确匹配时 `--config` 指向 `adaptations/{safe_name}/models/.../config.json`，`--model-path` 指向 `adaptations/{safe_name}/models`，`--image-ref` 按下节"镜像来源"解析。

## 镜像来源（自闭环）

项目根 `images/` 目录承载离线镜像 tar，命名 `vllm-ascend-<version>-<variant>.tar`（附 `.sha256`）。`image_ref` 默认**先查 `images/` 下是否有匹配版本+变体的 tar**：

- 命中 → 配置摘要标注 `image_source = local_tar`、`image_local_tar = images/<file>.tar`；裸机/SSH 节点部署前先 `docker load -i <project_root>/images/<file>.tar`；KTP/scheduler 提示操作员将该 tar push 到平台 registry（或 manifest 仍写 image 名，依赖平台已预置同名镜像）。
- 未命中 → `image_source = remote_registry`，`image_ref` 取远端 registry 全限定名，按原流程预检版本。

profile match 块的 `image_ref` 允许取本地镜像名（如 `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3`）；`plan_pd_topology.py` 的字符串精确相等校验语义不变，只需保证 match 块 `image_ref`/`model_path`/`config_sha256` 三字段同源（均来自自闭环目录或均来自外部值）。

## 部署知识库

本 Skill 自带 `knowledge-base/`（抓取自 vllm-project/vllm-ascend 官方文档 main 分支），deployer 在规划与预检阶段必须按需检索：

- **某模型启动命令 / 推荐 TP/DP/EP / 量化**：先查 `knowledge-base/models/<Model>.md`（41 个模型教程，索引 `knowledge-base/models/INDEX.md`），命中即取 `vllm serve` 原文命令与拓扑，禁止凭记忆编命令。
- **环境变量 / 参数语义**：`knowledge-base/03-configuration.md`（来自 `vllm_ascend/envs.py` 全量 env vars 表）。
- **PD 分离**：必查 `knowledge-base/09-pd-disaggregation.md`（mooncake 连接器配置、`kv-transfer-config`、端口、1P1D/2P1D 拓扑样例）。
- **硬件/版本兼容 / 支持模型与特性矩阵**：`knowledge-base/05-support-matrix.md` + `01-installation.md`。
- **特性是否可用**：`knowledge-base/07-features-core.md` + `08-parallelism.md`。
- 知识库为快照，引用命令时与目标镜像实际版本交叉核对（镜像内 `vllm-ascend` 版本以 `pip show` 为准）。

## 安全默认值



- 本机：当前可用环境，端口 8000。
- TP/DP/EP：读取 `config.json` 自动规划；MoE 开 EP，Dense 关 EP；默认 TP 不超过 KV heads。
- 非 PD 多机：每节点一个 DP rank，运行 NPU 可小于申请 NPU。
- 多机执行后端：默认且优先使用 vLLM 原生 `mp`。只有用户在当前提示词中显式要求 Ray，才允许选择 `ray` 并写入 `ray_explicitly_requested=true`。知识库示例、模型教程、已安装 Ray、现有 Ray 集群或 profile 均不能代替用户授权。若目标版本不支持所需原生 MP 拓扑，停止并报告版本/参数缺口；不得静默或自动回退 Ray。
- PD：用户只输入几 P 几 D。先在本 Skill 的 `profiles/` 和当前项目 `adaptations/` 中查找同模型架构、权重量化、镜像、vLLM-Ascend 版本、平台资源规格及 xPyD 均匹配，并且有真实推理通过证据的配置。命中时必须复用完整拓扑和关键运行参数，不能只继承模型参数后重新计算 TP/DP。没有精确命中时，每个 P/D 默认是单节点 `independent_instances`，调用 `scripts/plan_pd_topology.py --config CONFIG --prefill-count P --decode-count D --allocation-npu-per-node N --vllm-ascend-version VERSION` 自动计算其余全部参数。版本从镜像检查，连接器按兼容矩阵选择，`use_ascend_direct=true`，prefix caching 关闭。
- 性能优化：推测解码/MTP、动态 EPLB、图模式和特定预取策略只有在同模型制品、同镜像、同拓扑的真实推理证据中通过时才可默认启用；通用计算或候选 profile 默认采用保守值。失败证据必须保留且优先于未验证模板，禁止为了追求吞吐把已知失败优化重新带入部署。
- 网络与服务：安全端口自动探测；Proxy 共置首个 Prefill；共享挂载、HCCL 网卡和运行时地址通过预检验证。探测失败或有多个冲突候选时才询问。
- 用户明确提供的值始终覆盖默认值；性能最优 P:D 比例仍需工作负载和 SLO。

## 关键资源模型

始终区分：

- `allocation_npu_per_node`：调度 manifest 每节点申请量；整节点平台可能是 16。
- `runtime_npu_per_node`：实际加入 vLLM world 的 NPU 数，通常等于每节点 TP。
- `world_size = tensor_parallel_size * data_parallel_size`。
- 每节点一个 DP rank 时，`data_parallel_size = node_count`，`runtime_npu_per_node = tensor_parallel_size`。

禁止要求 `TP × DP = node_count × allocation_npu_per_node`。调度申请量可大于运行量，但必须明确报告资源闲置。

PD 分离时先显式区分角色的并行作用域：

- `global_group`：`role_TP × role_DP = role_node_count × role_runtime_npu_per_node`
- `independent_instances`：`instance_count × role_TP × role_DP_per_instance = role_node_count × role_runtime_npu_per_node`

不能用普通多机的总 TP/DP 公式代替角色公式，也不能把多个已验证的独立实例默认为一个跨节点全局组。运行 `scripts/plan_pd_topology.py` 做确定性计算；P:D 比例若由性能目标决定，必须询问典型输入/输出长度、并发量和 TTFT/TPOT 目标，不能只凭模型参数猜测。

## 安全 TP 规划

读取模型 `config.json`；Qwen/VLM/MoE 优先读取嵌套 `text_config`。运行：

```bash
python scripts/plan_topology.py CONFIG.json --node-count N --runtime-cap-per-node M
```

默认选择同时整除 `num_attention_heads` 和 `num_key_value_heads` 的最大 TP，且不超过单节点运行上限。只有已有真实推理证据时才允许 `TP > num_key_value_heads` 的 KV replication。

不要把“服务启动和 `/v1/models` 200”当作拓扑可用。TP 错误可能只在首次 forward 暴露；必须发送真实生成请求。

## 执行流程

1. gate 完成后，仅按目标读取一个 reference：本机读 [references/local.md](references/local.md)，SSH 读 [references/ssh.md](references/ssh.md)，调度平台读 [references/scheduler.md](references/scheduler.md)。
2. 将非敏感字段写入 `deploy-request.json`，运行：

   ```bash
   python scripts/validate_deploy_request.py deploy-request.json
   ```

3. 检查权重、模型配置、NPU、镜像/环境、端口、网络和资源。`model_path` 优先取"模型来源解析"得到的 `adaptations/{safe_name}/models`（自闭环），其次外部绝对路径。需要匹配已有适配模板、兼容配置或定位用户
   指定制品时允许遍历项目 `adaptations/`。避免无目的递归扫描整个
   `/models` 或共享权重根目录。
   镜像按"镜像来源（自闭环）"解析：先查 `images/` 下匹配 tar，命中标 `image_source=local_tar`，否则取远端 registry。
   若 `profiles/` 中存在候选，必须运行
   `plan_pd_topology.py --validated-profile PROFILE --config CONFIG ... --image-ref IMAGE --model-path MODEL_PATH`
   做精确兼容校验，其中包括权重绝对路径和 `config.json` SHA-256。任何模型架构、镜像、版本、模型制品、资源规格或 xPyD 不匹配都
   视为未命中并回退通用计算；禁止“近似套用”。配置摘要必须注明计划来源是
   `validated_profile` 还是 `generic_calculation`。
4. 展示最终配置：申请资源、实际运行资源、TP/DP/EP、多机执行后端、PD 角色与作用域、版本、连接器/Direct、节点、镜像、挂载、网络、全部端口和命令摘要；末尾询问“是否执行部署？”。多机默认显示 `distributed_executor_backend=mp`；只有当前提示词显式要求 Ray 时才可显示 `ray`，并注明其授权来源。
5. 只有用户明确确认后才生成部署文件并执行 Shell 语法检查和平台 dry-run。**每次部署必须渲染** `deploy-config.yaml`（按 `templates/deploy-config.yaml.j2` 填充 deploy-request 关键字段）、`run/*.sh` 角色脚本、以及 `deploy-baremetal.sh`（按 `templates/deploy-baremetal.sh.j2`）；`target=scheduler` 时额外渲染 `ktp-<topology>.yaml` manifest 与 `deploy-ktp.sh`（按 `templates/deploy-ktp.sh.j2`）。对生成的 `*.sh` 跑 `bash -n` 语法检查，`deploy-config.yaml` 跑 `python -c "import yaml;yaml.safe_load(open(...))"` 解析校验。KTP 固定 world 的多机
   ACJob 在 dry-run 前还必须执行
   `python scripts/validate_ktp_manifest.py job.yaml --require-gang`，确保
   `min_available` 覆盖全部 replicas 且各 task 的 `min_member=replicas`。
   KTP dry-run/真实提交显式传 `--min-available TOTAL_REPLICAS`，并用验证器的
   `--dry-run-output` 检查 CLI 实际渲染值，防止配置被 CLI 默认值静默覆盖。
   冻结前运行 `python scripts/validate_frozen_artifact.py DEPLOY_DIR --write`，
   提交前再不带 `--write` 检查精确文件集合和哈希；不能只用
   `sha256sum -c`，因为它不会发现额外的旧日志或编译缓存。
6. 多机调度任务若允许单 Pod 独立重启或换节点，必须用 `scripts/stable_cluster_supervisor.py` 包裹每个角色入口；共享状态目录必须按冻结任务唯一命名。session/IP 变化立即重建，瞬时存储/心跳不可用按 `--unavailable-grace-seconds` 有限容忍；保留非零的 `--child-tail-lines`，让容器重启前把角色末尾输出写入共享 `events/rank-*-child-tail.log`。验收时同时检查事件和子进程末尾日志，区分首发根因与协调终止产生的次生异常。同一节点有多个本地 DP 实例时使用 `scripts/launch_online_dp.py`，不得使用忽略子进程 `exitcode` 的启动器。用户确认执行后提交部署；轮询进程/Pod。
7. 所有节点 `/v1/models` 就绪后发送确定性最小推理请求，并用 `scripts/validate_inference_result.py` 对解析后的答案做严格断言。
8. 返回 endpoint、PID/job ID、日志位置、停止命令、`deploy-baremetal.sh`/`deploy-ktp.sh` 路径（一键部署/重提入口）。失败时只停止本次创建的资源。

PD 验收还必须确认 Prefill、Decode、Proxy 三类进程均健康，Proxy 真实推理通过语义断言，KV transfer 日志显示传输成功且未静默 fallback。`use_ascend_direct=true` 失败时不得自动关闭；换干净节点复测仍失败则报告阻塞。只有用户明确接受降级时才能用 `false` 做独立故障隔离，且不能用降级结果替代 direct 成功标准。

## 产物

在用户指定目录或 `vllm-ascend-deploy/{normalized_model}/` 下生成：

```text
deploy-request.json          # 部署请求（非敏感字段）
deploy-plan.json             # 部署计划摘要
deploy-config.yaml           # 【必出】配置 yaml（TP/DP/EP/端口/超时/image_source/model_path 可读视图）
run/                         # 角色启动脚本（preflight_check.sh / run_role.sh / run_prefill_*.sh / run_decode_*.sh / run_proxy_*.sh / start_*.sh）
deploy-baremetal.sh          # 【必出】裸机一键部署（load tar -> preflight -> start -> /v1/models -> 语义断言）
ktp-<topology>.yaml          # 【target=scheduler 必出】KTP manifest
deploy-ktp.sh                # 【target=scheduler 必出】KTP 一键部署（validate gang -> 冻结 -> apply --min-available -> 轮询 -> 语义断言）
logs/
README.md
```

所有产物复用 `templates/` 下 j2 模板渲染；冻结部署目录只读，详见下文。

Shell 使用 `set -euo pipefail`。对 `ip` 等镜像可能不提供的探测命令，必须先用 `command -v` 判断，并让探测管道显式 `|| true` 后采用已确认的 NIC 回退值；否则可选探测会在服务启动前把角色以 127 退出。`vllm` 入口优先使用已验证的绝对路径，不存在时再安全查询 PATH，仍不存在则输出明确错误并退出。不要在运行中的共享脚本上原地修改；先停止任务，原子替换冻结脚本，再重新直接提交。

多机入口不得只靠“等待 `rankN_ip` 文件存在”。旧文件会让不同启动代次的节点错误组网。生成的角色脚本必须在同一进程组内运行；任一角色退出时终止同 Pod 的其他角色，让 supervisor 能触发整组重新汇合。

冻结部署目录是只读输入，不作为 vLLM 工作目录。为每个 Pod/rank 创建独立的
本地运行目录，把 `kernel_meta`、编译缓存和临时日志写到该目录；脚本和
`RUN_TEMPLATE` 使用冻结目录中的绝对路径。验收后再次运行
`validate_frozen_artifact.py`，任何额外文件或哈希变化都必须清理或判为污染。

大模型跨节点 DP/EP 加载时，冷热缓存可能让不同节点的权重加载时长相差超过 PyTorch 默认的 30 分钟。`VLLM_ENGINE_READY_TIMEOUT_S` 和 engine core handshake timeout 不控制进程组 collective。必须按最坏加载偏差设置 vLLM 的 `--distributed-timeout-seconds`；若当前 vLLM-Ascend Worker 未把该配置传给公共分布式初始化，先用 `scripts/patch_vllm_distributed_timeout.py` 做严格、幂等且版本锚点可验证的兼容修复。补丁锚点不匹配时必须失败，不得静默继续。

HCCL communicator 建连还有独立的 `HCCL_CONNECT_TIMEOUT`，已经进入
`process_weights_after_loading` 的快节点可能在慢节点仍加载权重时等待建连；
`--distributed-timeout-seconds` 不覆盖这个阶段。跨节点大模型必须让
`HCCL_CONNECT_TIMEOUT` 同样覆盖最坏加载偏差，并让 `HCCL_EXEC_TIMEOUT`
覆盖后续长 collective；三类超时分别记录在配置摘要，禁止只增大其中一个。
`--distributed-timeout-seconds` 必须大于 `HCCL_EXEC_TIMEOUT`，为 HCCL
plog 和协调退出留出诊断窗口，避免二者相等触发 watchdog 告警。

## 凭据

密码、token 和私钥只在连接前请求并仅保存在会话中。禁止写入 JSON、manifest、脚本、日志、Git、shell history 或命令行参数；禁止回显或运行无过滤的 `env`。首次 SSH 必须确认 host key 指纹，禁止关闭 host key 校验。

## 成功标准

以下全部满足才能报告成功：

- 所有目标进程/Pod 正常，且无异常重启；
- `/v1/models` 返回目标模型；
- `/v1/chat/completions` 或 `/v1/completions` 返回 HTTP 200 和非空内容；
- 至少一个答案唯一、可机器断言的确定性提示通过语义校验；
- 返回 endpoint、日志和停止方法；
- 已交付 `deploy-config.yaml` + `deploy-baremetal.sh`（`target=scheduler` 时另含 `ktp-<topology>.yaml` + `deploy-ktp.sh`），且 `*.sh` 跑过 `bash -n` 语法检查、`deploy-config.yaml` 可被 `yaml.safe_load` 解析；
- 失败任务已停止，平台无本次遗留资源；
- 请求文件和日志不含凭据。

固定使用答案唯一的短提示，关闭模型 thinking 或提高 token 上限，并对解析后的 `choices[0].message.content` 做精确或锚定正则断言；不要只 grep `"content"`。任何语义断言失败均将部署整体判为失败，即使 HTTP、健康检查和 KV transfer 均成功。
