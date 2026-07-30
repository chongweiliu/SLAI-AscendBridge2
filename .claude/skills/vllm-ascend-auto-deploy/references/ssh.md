# SSH 部署

## 最小必填信息

1. 每节点主机/IP和用户名；SSH 端口默认 22，用户明确指定时覆盖。
2. 认证方式。SSH Agent、密钥文件路径或交互密码均可；密钥内容和密码不得
   写入产物。密码模式在执行时建立临时 ControlMaster，只提示一次并由后续
   SSH/SCP 复用，结束时关闭连接和删除 socket。
3. 逐节点确认的 SHA256 host-key 指纹。先向可信管理员或控制台核对，再写入
   `nodes[].host_key_sha256`；禁止把首次扫描结果自动当作可信值。
4. 远端运行方式（host/container）、模型绝对路径和每节点 NPU device IDs。
   多机模型路径必须在每个节点上存在并指向同一制品。

其余自动处理：首个节点默认 Master；权重默认所有节点使用用户给出的同一
绝对路径并在预检验证；运行 NPU、TP/DP/EP 从模型配置和可见 NPU 计算；
网卡根据路由和可达地址探测；rendezvous 与服务端口自动选取并检查占用；
优先使用远端现有兼容环境/镜像。只有路径不可读、环境不存在、网卡歧义或
端口无法分配时才追问。

多机至少两个同构节点。先确认 host key 指纹并写入专用临时
`known_hosts`，禁止关闭校验。对所有节点先做只读检查；任一节点失败则不启动。

校验软件、镜像、模型文件、NPU 和时间同步。先启动 Worker，再启动 Master；记录每个 PID/容器 ID，失败时逆序停止。健康检查从调用方和 Master 各执行一次。

## 正式支持边界

SSH v1 支持：

- 单节点 TP；
- 非 PD 多节点原生 `mp`，TP 在节点内、DP 跨节点；
- host 上的 vLLM 可执行文件，或节点已预置镜像的 Docker；
- SSH 直接进入已运行容器时，可显式启用 `inherit_pid1_environment` 和
  `source_user_bashrc`，在内存中恢复容器运行环境，不把环境内容写入产物；
- 每节点独立指定 `device_ids`、通信 IP/网卡；
- SSH Agent、密钥或一次性交互密码认证；
- 通过 SSH 隧道执行 `/v1/models` 和真实最小推理验收。

SSH v1 明确不支持 PD 分离、自动 Ray、自动复制大模型权重、把密码或私钥写入
部署包。缺少这些能力时必须报告不支持，不能生成看似可用的脚本。

## 产物与执行

运行：

```bash
python scripts/render_ssh_artifacts.py deploy-request.json --output-dir DIR
bash -n DIR/deploy-ssh.sh
bash -n DIR/remote-node.sh
```

产物包括：

- `deploy-ssh.sh`：host-key 验证、上传、全节点预检、网络发现、启动和验收；
- `remote-node.sh`：按 deployment ID 管理单个 rank；
- `deploy-request.json`：不含密码、token 或私钥；
- `artifact-sha256.txt`：冻结文件集合与哈希；
- 自包含的冻结校验器和语义验收器。

`deploy-ssh.sh start` 在任何远端变更前验证冻结产物和全部 host-key。启动中途
失败、SSH 回包丢失或语义验收失败时，按本次 deployment ID 反向停止所有
rank。禁止 `pkill -f vllm` 等可能杀死其他服务的清理方式。
