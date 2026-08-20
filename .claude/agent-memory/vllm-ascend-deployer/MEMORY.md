# vLLM-Ascend Deployer Memory

## 精华规则

- 拓扑规划前必须先识别硬件代际。裸机/SSH 以只读 PCI/NPU 预检为准：
  `0xD802=A2`、`0xD803=A3`；调度平台必须提供并核验代际契约。
- 命中官方模型教程时，优先使用“当前硬件代际 + 当前部署模式”对应的完整
  recipe，包括 TP/DP/EP 和该场景环境变量。禁止把 A2、A3 等章节中的 TP
  压平成一个列表后取最大值。
- 只有官方教程没有当前硬件 recipe 时，才回退 `config.json` 的头数整除与
  资源上限计算；健康检查不能替代首次真实 forward。
- HCCL 算法、buffer、FlashComm/MC2 等通信参数具有模型、硬件、拓扑和版本
  作用域，不应作为跨 recipe 的无条件公共默认值。

详见 [hardware-official-recipes.md](hardware-official-recipes.md)。
