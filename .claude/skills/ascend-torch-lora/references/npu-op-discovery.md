# NPU 算子发现方法论：报"不可行"之前的三步排查法

> 2026-09-01 实战教训（Qwen3.6-35B-A3B MoE LoRA 提速）：曾断言 "`grouped_mm` 依赖 CUDA 的
> `torch._grouped_mm`，NPU 无此内核"——**这个结论是错的**。系统排查后发现 NPU 不仅有自己的
> grouped GEMM（`npu_grouped_matmul`，已内置 torch_npu），官方还有配套桥接（torchtitan-npu）
> 和完整 MoE 算子族。**在昇腾上说"某能力不存在/不支持"之前，必须走完下面三步。**

## 核心原则

**先查本机 → 再搜官方仓 → 最后看文档案例。三步走完才允许下"不可行"结论，且结论要写明排查范围。**

PyTorch/CUDA 生态的函数名（如 `torch._grouped_mm`、`F.scaled_dot_product_attention`）在昇腾上
**几乎都有对应物**，只是名字不同、藏在不同的层（torch_npu 插件 / CANN aclnn / 独立算子库）。
"NPU 没有这个内核"绝大多数时候只是"没找到入口"。

## 第一步：盘点本机 CANN 与 torch_npu 的注册接口

```bash
# 1a. CANN 版本
cat /usr/local/Ascend/cann-*/version.info 2>/dev/null; ls /usr/local/Ascend/ascend-toolkit/

# 1b. CANN 的 aclnn 算子全集（头文件即接口面, ~1000+ 个）
ls /usr/local/Ascend/cann-9.0.0/aarch64-linux/include/aclnnop/ | grep -iE "grouped|moe|你的关键词"
#    例如: aclnn_grouped_matmul*.h (26个变体), aclnn_moe_*.h (36个, 含 10 个 _grad 反向!)

# 1c. torch_npu 已注册到 Python 层的算子（能直接 torch_npu.xxx() 调用的）
grep -oE "torch_npu\.[a-z_0-9]+" $(python -c "import torch_npu,os;print(os.path.dirname(torch_npu.__file__))")/utils/custom_ops.py | sort -u | grep -iE "grouped|moe"

# 1d. 二进制级确认（schema 是否带 autograd/_backward 变体）
strings <torch_npu路径>/lib/libtorch_npu.so | grep -E "你要查的算子名" | sort -u
#    看输出: 有 schema 无 Autograd 注册 = 前向可用反向需自包(见第三步)

# 1e. 算子的完整文档（参数/shape/dtype 约束/硬件支持/调用示例）
python -c "
import torch_npu, re
s = open(torch_npu.__file__.replace('__init__.py','') + '_op_plugin_docs.py').read()
i = s.find('\"npu_算子名\"')
print(s[i:i+3000])"
```

**实测要点**：找到候选算子后写最小用例验证（前向 + 反向各一次），别只看文档就下结论。
torch_npu 算子常见坑：入参要包 `List[Tensor]`（如 `npu_grouped_matmul([x],[w],...)`）、
group_list 是**前缀和累计值**而非每组数量、某些模式必须指定 `split_item`（如 group_type=0 须 2/3）。

## 第二步：搜索 https://gitcode.com/cann 官方组织（组织页列表不全，要顺藤摸瓜）

**关键仓库地图**（按用途）：

| 仓库 | 内容 | 何时查 |
|---|---|---|
| **cann/ops-transformer** | transformer 大模型算子库源码：`gmm/`(grouped_matmul 9 变体)、`moe/`(全套路由+分发+回收，**含 8 个 _grad**)、`mc2/`(融合通信 MoE)。源码随 CANN 版本分支发布（如 9.0.0 分支） | 找训练/推理算子的实现与反向 |
| **cann/torchtitan-npu** | 官方 NPU 训练框架（**组织页列表看不到，从其它仓库 README 引用发现**）。含把 `npu_grouped_matmul` 注册为 `aten::_grouped_mm` NPU 后端的桥接代码（`ops/_grouped_mm.py`，反向由 PyTorch 核心公式自动提供）、TileLang 写的 MoE 反向内核 | 找"官方怎么把 CANN 算子接进 PyTorch 训练" |
| **cann/catlass** | 昇腾版 CUTLASS 模板库（grouped GEMM slice_m/slice_k/MoE 量化模板 + v2.0 Python DSL） | 要自研算子（含自定义反向）时 |
| **cann/cann-recipes-train** | 训练配方（DeepSeekV3 预训练、Qwen3-30B-A3B MoE SFT/EP 并行、verl RL） | 找端到端参考实现 |
| cann/ops-math / ops-nn / ops-cv / ops-gnn | 分类算子库 | 找基础算子 |
| cann/cann-recipes-infer | 推理配方 | 推理场景 |

**搜索技巧**：组织页 HTML 里的仓库列表**不完整**（torchtitan-npu 就不在列表里）——要从
cann-recipes-*/README、算子文档的引用链接顺藤摸瓜；直接 `git clone --depth 1` 下来 grep 最快；
搜 grad/backward 关键词判断训练支持（`find . -iname "*grad*"`）。

## 第三步：结合文档与参考案例落地

- **官方算子无反向时的标准补法**（grouped GEMM 案例）：MoE grouped matmul 的反向在数学上
  可由前向组合——`dx = grouped_matmul(dy, W)`（原生布局直用）、`dW = grouped_matmul(xᵀ, dy, group_type=2)`。
  两种落地：(a) 照抄 torchtitan-npu 的 `torch.library.impl("aten::_grouped_mm", "PrivateUse1")`
  桥接（反向免费，实测 dx 逐位一致）；(b) 自写 ~15 行 autograd.Function 直调（LoRA 冻结权重只需 dx，更简）。
- **参考实现优先级**：torchtitan-npu（训练框架）> ops-transformer 的 tests/ 与 torch_extension/ >
  cann-recipes-train 配方 > CATLASS examples。
- ** transformers 集成**：transformers 5.16 的 MoE 有 `experts_implementation` 机制
  （`config._experts_implementation = "batched_mm"/"grouped_mm"/...`），但内置实现在 NPU 上不可用
  （batched_mm 物化 60GB 权重、grouped_mm 找 CUDA 内核）——正确做法是 monkey-patch 专家模块的
  forward（见 moe-optimization.md），或按本文件方法桥接 aten 算子。

## 排查结论的输出格式（防自我误导）

> ❌ 错误示范："NPU 不支持 grouped GEMM"
> ✅ 正确示范："CANN 9.0.0 有 aclnn_grouped_matmul(26 变体)但无反向注册（libtorch_npu.so 二进制
> 确认）；torchtitan-npu 提供 aten::_grouped_mm 桥接使反向可用（本机实测 dx 逐位一致）；
> 短序列场景下分发开销使其不敌稠密补丁（17.4 vs 13.5s/步）"

**结论必须包含**：查了哪些接口面（头文件/二进制/Python 注册）、哪些仓库、实测数据、适用边界。
