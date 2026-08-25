# HCCL 集合通信测试

## 何时运行

- 新集群/新网络首次上线：先做小消息 AllReduce/AllGather 正确性。
- vLLM 多机首个 forward 卡住：在不占用生产作业设备的维护窗口做快速测试。
- 用户要求通信性能基线：扩大消息范围并重复测量，保留拓扑和环境。

`hccl_test`、MPI 版本和构建参数必须以目标 CANN/硬件官方文档为准。不要在部署过程中自动
安装 MPI、编译工具或配置 SSH 免密；这些都是独立的环境变更。

## 典型流程

1. 每节点确认 NPU health、驱动/CANN 和 `hccl_test` 制品一致。
2. 构造明确的 hostfile/rank table，卡数总和与进程数一致；特殊超节点拓扑按官方约束排列。
3. 快速正确性测试从小消息范围开始，例如已构建二进制支持时：

   ```bash
   mpirun -f hostfile -n ${WORLD_SIZE} ./bin/all_reduce_test \
     -p ${NPU_PER_NODE} -b 8K -e 64M -f 2 -d fp32 -o sum
   ```

4. 性能基线才扩大到代表性消息范围，同时测 AllReduce、AllGather；MoE/EP 按需补
   AlltoAll/AlltoAllV。
5. 保存每种消息大小的正确性、算法带宽、总线带宽、最慢 rank、错误码和重复次数。

任一数据校验错误、rank 缺失、timeout 或进程非零退出均为失败。带宽与同硬件、同拓扑的
已验证基线比较，报告中位数和尾部；低带宽先排查链路、NUMA/PCIe、网卡和交换网络。
