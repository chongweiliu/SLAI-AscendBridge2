# 调度平台部署

外部版本仅支持原生 Kubernetes、CCE 和 ACK。平台接入、资源域和 manifest
契约必须由用户在当前对话提供或确认。

## 最小必填信息

只向用户收集平台无法安全推断的契约：

1. 平台、CLI/API、连接/context、认证方式。
2. Kubernetes、CCE、ACK 提供 kube context、namespace、NPU 扩展资源键和 PVC。
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

## 原生 Kubernetes / CCE / ACK

原生路径使用标准 `StatefulSet + headless Service + API Service`，可由标准
`kubectl` 提交到 Kubernetes、CCE 或 ACK。

最小平台契约：

- `kube_context`（空值表示当前 context）和 `namespace`；
- 平台设备插件注册的 NPU 扩展资源键，例如由管理员确认的
  `vendor.example/npu`，禁止硬编码某个云的猜测值；
- 已存在的模型 PVC：`claim_name`、容器内 `mount_path`、可选 `sub_path`；
- 镜像、CPU、内存、每 Pod 申请 NPU 数、Service 类型；
- 私有镜像只记录 `imagePullSecret` 名称，不把 registry 凭据写入请求或清单。

执行：

1. 运行 `scripts/render_kubernetes_artifacts.py deploy-request.json --output-dir DIR`。
2. 运行 `scripts/validate_kubernetes_manifest.py DIR/kubernetes.yaml --request DIR/deploy-request.json`。
3. 对生成的 `deploy-kubernetes.sh` 执行 `bash -n`。
4. 一键脚本先执行 client/server dry-run，再 apply、等待 StatefulSet、临时
   port-forward，并用真实最小推理完成语义断言。
5. 用 `deploy-kubernetes.sh status|logs|delete` 查询、取日志和清理。

当前原生 Kubernetes v1 支持单机、非 PD 的多机原生 `mp`，以及基于 Mooncake
的多机 PD 分离。非 PD 多机时 TP 在单 Pod/节点内、DP 跨 Pod；
`data_parallel_size` 必须能被节点数整除，每节点运行 NPU 数等于 `TP * local_DP`。
PD 模式为每个 P/D 实例生成独立 StatefulSet、headless/API Service、KV transfer
端口和一个 Proxy Deployment/Service；API 请求必须经 Proxy 发送。所有角色共享
相同模型制品并使用同一套版本/连接器契约。

当前原生 Kubernetes v1 仍明确拒绝 Ray。不得因集群安装了 Ray 而自动切换；
只有用户显式要求 Ray 时才进入后续专用实现，而不是复用本清单。PD 分离可以
渲染和提交，但不能把 YAML 生成成功当作部署成功：必须等待 P/D/Proxy 健康，
确认 KV transfer 成功且没有静默 fallback，再通过 Proxy 做真实推理验收。
CCE/ACK 是否可直接使用同一清单，取决于目标集群已安装兼容的昇腾设备插件、
网络组件、存储类/PVC 和镜像访问配置；渲染器不写入云账号或临时凭据。
