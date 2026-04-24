# Negative Applicability: Z-Image Pattern

这个案例的价值不在“怎么 patch”，而在“什么时候不要再 patch”。

## 典型信号

- 框架本身已经对某些算子做了 NPU 友好实现
- 原始前向结构并不适合融合算子接口
- 为了调用融合算子，反而要额外做 `cat`、reshape 或搬运

## 结果判断

若出现以下情况，应停止堆更多融合算子：

- patch 前后速度差异落在测量误差范围
- 加 patch 后没有端到端提速
- 单个算子理论更快，但被额外张量拼接开销抵消

## 实务建议

- 把这类模型优先归为“runtime-only 或 backend-only 候选”
- 不要因为有可用 API 就默认必须接入
- 把精力转向更高层级的瓶颈：offload、量化、图模式或数据路径
