# 故障模式与隔离顺序

| 现象 | 先观察 | 最小隔离 | 不应直接做 |
|---|---|---|---|
| `torch_npu`/ACL 动态库加载失败 | Python、torch/torch-npu、CANN 路径与版本 | 容器内 import 和最小 NPU op | 直接重装驱动 |
| `Unsupported model`/registry 错误 | `architectures`、`model_type`、task、vLLM 版本 | dummy load 或缩小模型 config | 改近似模型名绕过 |
| 权重缺失/shape mismatch | index、safetensors 哈希、config、tokenizer vocab | 单卡真实权重加载 | 修改 shape 丢弃权重 |
| 首次 forward 才失败 | dtype、TP、KV heads、量化 kernel、图模式 | 单卡 eager、固定短 prompt | 仅看 `/v1/models` |
| OOM | 权重/激活/KV、max length、batch tokens、并发 | 单变量降低长度或并发并记录峰值 | 同时改 dtype、TP、batch |
| ACLGraph 编译/回放失败 | 首个失败算子、shape、动态图分支 | 同参数 eager 基线 | 把 eager 成功称为图模式修复 |
| HCCL timeout/hang | 最早 rank 日志、IP/网卡/端口、world size | all-rank 初始化或短 forward | 只增大 timeout |
| PD 无输出 | P/D/Proxy、KV role、connector、direct 日志 | 非 PD 基线，再做 1P1D | 静默关闭 direct |
| API 200 但答案错误 | chat template、tokenizer、dtype、采样参数、权重哈希 | 固定请求与基线服务对比 | 只断言非空字符串 |
| 性能下降 | TTFT、TPOT、吞吐、P95/P99、计算/通信时间 | 同负载 A/B 单变量实验 | 用总耗时估算 TTFT |

## 日志顺序

1. 按节点、rank 和进程保存原始日志，统一时间戳。
2. 找第一条异常前后的完整上下文；后续 connection reset、barrier 退出和清理异常通常是次生结果。
3. 建立单卡 eager 基线，再逐步恢复真实 dtype、权重、TP、图模式、多机和 PD；记录问题在哪一步重现。
4. workaround 与 root cause 分开报告。降低长度或关闭图模式只证明隔离有效，不证明根因已修复。
