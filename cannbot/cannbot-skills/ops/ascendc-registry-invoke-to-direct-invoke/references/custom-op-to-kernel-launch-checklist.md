# 自定义算子转 `<<<>>>` kernel 直调检查清单

这个清单用于 **AscendC 自定义算子转 `<<<>>>` kernel 直调改造** 的静态核验。

适用场景：
- 把现有算子源树（kernel + tiling）迁到目标目录
- 删除相对 include，提取最小依赖闭包
- 判断当前结果是否只是"kernel 依赖解耦完成"，还是已经"可直接 launch / 独立编译"

**使用本清单前，先确认核心原则**：
1. kernel 实现代码零修改（只允许 include 替换、命名空间包装、TilingData 类型来源替换）
2. tiling 计算逻辑零修改（只允许平台接口替换、框架胶水删除、输入参数形态调整）
3. 只改"注册框架胶水"（入口、TilingData 宏定义、host tiling 接口）
4. 默认全量迁移，不做选择性裁剪——只有用户明确授权才裁剪指定字段/路径/variant

如果清单的某一项检查和这些原则冲突，以原则为准。

---

## 1. 任务边界确认

- [ ] 这是 **自定义算子转 `<<<>>>` kernel 直调改造**，不是 host / graph / 注册集成
- [ ] 目标目录已明确（由用户指定，不预设布局）
- [ ] 是否需要同时处理入口文件（命名可能是 `*_apt.cpp` / `*_invocation.cpp` / 其他）
- [ ] 是否要求"可独立编译 / 可直接 launch"，还是只要求"kernel 依赖解耦"

### 1.1 目标仓约定调研

开工前在目标仓库里 `Glob` + `Read` 2–3 个同类 story / sample（只看组织结构，不看算法），归纳：
- [ ] 目录布局（include/src/cmake 分层、kernel/tiling/host 放哪里）
- [ ] 文件命名约定（`.h` / `.hpp` 后缀、`_kernel` / `_entry` / `_tilingdata` 等后缀规则）
- [ ] CMakeLists.txt 骨架（target_include_directories 写法、编译选项组织）
- [ ] include 路径风格（相对路径 vs 逻辑路径）
- [ ] 顶层命名空间选择
- [ ] 测试 / README 目录的组织方式

本算子的目录结构和文件命名 **跟这些邻居约定对齐**，不要发明新规矩。如果同类 story 之间就不一致，选影响面最大或最新维护的那个参考。

---

## 2. 源文件盘点

列出：
- [ ] kernel 源目录下全部 `.h` / `.cpp`
- [ ] 入口文件（通常在父目录或 kernel 目录根）
- [ ] 依赖汇聚点（通常是 `*_common.h` 等）
- [ ] tiling 源（通常在 `op_host/` 或类似位置）
- [ ] 已被"从上游模板改造过的半本地化版本"文件（优先复用，不重建）

---

## 3. 相对 include 盘点

列出所有 `#include "../..."` / `#include "../../..."`。对每条至少确认：
- [ ] 被哪些目标文件引用
- [ ] 当前目标文件真正用了哪些符号
- [ ] 这些符号的**真实**定义位置
- [ ] 是否存在"表面来自 A，实际定义在 B"的中转依赖

不要接受"include 了就整包搬"。

---

## 4. 原则 1 验证：kernel 实现代码零修改

对目标目录下所有 kernel 实现文件（包括从上游复制 / 保留的实现头），逐文件对比原文件：

- [ ] 除了 include 指令、命名空间包装、TilingData 类型来源以外，没有其他改动
- [ ] 没有合并或拆开成员函数
- [ ] 没有重命名字段、参数、局部变量
- [ ] 没有"顺手"优化循环、计算顺序、分支写法
- [ ] 没有把类成员改成自由函数（或反过来）
- [ ] 没有删除看起来"没用"的成员或方法

**判断标准**：若从原工程同步新版本 kernel，能否直接覆盖 kernel 实现文件而不影响其他地方？能就通过。

---

## 5. 原则 2 验证：tiling 计算逻辑零修改

对比目标 host tiling 逻辑和原 `op_tiling` 代码：

- [ ] 对齐 / 非对齐分支独立保留（没有合并成"统一"公式）
- [ ] 分子 / 分母 / 上下取整按原样（如原本是 `AlignUp(baseColLen, ubMinBlockLen)` 就保持，原本不是就不要加）
- [ ] 边界判断、clamp、溢出检查保留
- [ ] 乘除法计算顺序保留
- [ ] 只做了接口替换（`gert::TilingContext*` → `PlatformAscendCManager`）、框架胶水删除、输入参数形态调整——没有数学改动

重点关注的典型公式（按算子而定）：
- [ ] `baseRowLen` / `baseColLen` / `tileLength` 等切块参数
- [ ] `is32BAligned` 等关键分支分叉
- [ ] `blockDim` 推导

---

## 6. 符号分类与最小闭包

把要搬的外部符号分成四类，逐类核验：

### 6.1 平台常量
- [ ] 能否直接改成本地 `constexpr`
- [ ] 是否保留最小 `platform` namespace 子集更清晰

### 6.2 基础常量 / traits
- [ ] 是否只搬需要的一小段
- [ ] 是否误把整份 `*_base.h` 复制

### 6.3 common 层工具
- [ ] 命名空间归属是否真实（最易出现"表面归属 ≠ 真实定义"）
- [ ] 迁移后 `using` 是否需要修正

### 6.4 算法 helper
- [ ] 是否只搬当前 kernel 真正使用的函数
- [ ] 是否已补齐它们的直接闭包
- [ ] 是否误把整份上游 `*_common.h` / `*_regbase_common.h` 拖进来

---

## 7. 收口结构

默认推荐：
- [ ] 保留原 kernel 实现文件主体不动
- [ ] 新增 `<op>_local_deps.h`，装 SDK include + 外部依赖闭包
- [ ] `local_deps.h` 只装外部依赖，不混入当前 kernel 主逻辑
- [ ] 文件职责清晰：kernel 实现 / 依赖闭包 / 入口 / tiling / host driver 分离

仅在依赖极少时考虑把外部内容直接塞回原头。

---

## 8. 命名空间与死引用

- [ ] `using` 的符号归属真实
- [ ] 无错误中转归属（例如表面从 `A` 引，真实在 `B`）
- [ ] 无未使用 `using`
- [ ] 无未使用 include
- [ ] 无只为历史遗留保留的空壳依赖

典型死引用模式：`using XXX::SomeHelper` 只声明不使用。

---

## 9. 入口文件检查

如果同时处理了入口文件：

- [ ] 入口最终放在独立头文件（如 `<op>_entry.h` / `<op>_apt.h`），不和 kernel 实现头混在一起
- [ ] 去掉了 `extern "C"`
- [ ] 去掉了 `GET_TILING_DATA` / `GET_TILING_DATA_WITH_STRUCT`
- [ ] 去掉了 `TILING_KEY_IS` 运行时分发
- [ ] 按 tilingKey 的实际划分维度拆成独立入口函数（按 dtype、计算模式或其他维度，因算子而异）
- [ ] `GM_ADDR tiling` 改成 `const XxxTilingData tilingData`（by value）
- [ ] 如果原始代码有 `KERNEL_TASK_TYPE_DEFAULT` 则保留；没有不要主动添加
- [ ] workspace 参数如果保留，是否加 `(void)workspace;` 抑制未使用警告
- [ ] 入口头避免 `using namespace`，改用 `::AscendC::...` 这类显式限定
- [ ] 入口函数只做 3 件事：构造 `TPipe` → 构造 kernel 类对象 → `Init` + `Process`

关于外部宏依赖，结论必须明确二选一：
- [ ] **kernel 已本地化，但入口仍依赖外部编译环境（`DTYPE_*` 等）**
- [ ] **入口也已本地化，可独立编译**

---

## 10. 静态 include 清理

在目标文件集合中 grep：
- [ ] `#include "../`
- [ ] `#include "../../`

目标：无匹配。

---

## 11. TilingData struct 替换

如果把 `BEGIN_TILING_DATA_DEF` 替换成 plain struct：
- [ ] 机械转换为普通 C++ struct（不需要 `#pragma pack`）
- [ ] **默认保留所有字段**（"全量迁移优先"原则）；如果裁剪了字段，必须对应用户明确授权
- [ ] 字段名、字段类型、字段顺序与原 `TILING_DATA_FIELD_DEF` 一致
- [ ] 放在独立 `<op>_tilingdata.h`，仅依赖 `<cstdint>`，kernel 和 host 共用

---

## 12. Host tiling 提取

如果提取了 host tiling 逻辑：

### 12.1 接口替换（功能等价）
- [ ] `gert::TilingContext*` 已替换为 `PlatformAscendCManager::GetInstance()`
- [ ] `OP_LOGE` / `OP_LOGI` 替换为 `throw` 或 `printf`
- [ ] `ge::DataType` 替换为自定义 enum
- [ ] `IMPL_OP_OPTILING` / `REGISTER_TILING_DATA_CLASS` 已删除

### 12.2 数学逻辑保真（原则 2 的具体检查）
- [ ] 对齐/非对齐分支独立保留
- [ ] 关键公式逐条和上游比对过，而不是"凭感觉等价"
- [ ] **默认保留所有 variant 路径**（"全量迁移优先"原则）；如有裁剪（如只要 fullload），必须对应用户明确授权
- [ ] 所有保留的路径内部逻辑一行不动
- [ ] 返回值包含 `blockDim` + `tiling` struct

---

## 13. Host 驱动检查

如果创建了 standalone host 驱动，先确认样板来源：
- [ ] 目标仓有同类 story / sample 参考 → 对齐邻居写法
- [ ] 目标仓无参考 → 基于 `references/host-driver-template.md`

通用检查项：
- [ ] 数据生成自包含（C++ 确定性生成，不依赖 Python）
- [ ] golden 计算用 float 精度完成
- [ ] 测试了所有目标 dtype
- [ ] `GM_ADDR` 贯穿 host / launch / kernel 三侧
- [ ] `aclrtMalloc` 按 `reinterpret_cast<void**>(&devicePtr)` 形式使用
- [ ] kernel launch 形如 `xxx<<<blockDim, 0, stream>>>(inputDevice, ..., tilingData)`

---

## 14. Host launch ABI 一致性

如果 kernel 入口参数是 `GM_ADDR`：
- [ ] host 侧 device buffer 变量声明为 `GM_ADDR`
- [ ] `aclrtMalloc` 按 `reinterpret_cast<void**>(&devicePtr)` 形式写入这个 `GM_ADDR` 变量
- [ ] kernel launch 时直接传 `devicePtr`
- [ ] 不存在 `reinterpret_cast<GM_ADDR>(voidPtr)` 这种调用点强转

---

## 15. `-xasc` 双 pass 编译兼容（有条件）

触发条件：`Grep` kernel 实现头中是否出现 `MicroAPI::` / `Reg::` / `__VEC_SCOPE__` / `RegTensor` / `MaskReg`。无则跳过。

- [ ] 所有使用 MicroAPI/Reg 的实现头已加 `#if !defined(__NPU_HOST__)` 保护
- [ ] `local_deps.h` / `tilingdata.h` 等纯类型定义头**未**加 guard（host/device 都需要可见）
- [ ] `__global__` 入口函数使用"body 内 guard"模式（函数签名在 guard 外，实现在 `#if !defined(__NPU_HOST__)` 内）
- [ ] 遇到 bf16 mangling 问题时，`__global__` 入口改成显式命名的非模板函数
- [ ] host 侧数据类型（如 `SampleBFloat16`）与 kernel 模板参数（如 `bfloat16_t`）严格分离

---

## 16. 自包含性

**运行时 / 编译依赖层面**：
- [ ] 适配结果只依赖 CANN SDK + ACL 运行时 + 标准 C++ 库
- [ ] 不依赖目标仓库中其他 sample 的代码文件（没有 include 或链接其他 sample 的 `.h` / `.cpp` / 库）
- [ ] 目标目录内所有需要的代码都闭环

**算法逻辑层面**：
- [ ] kernel 计算 / tiling 数学 / 数据搬运逻辑都来自原算子源码，没有从其他 sample 复制算法代码
- [ ] host 驱动的算法无关骨架（ACL init、malloc、类型转换、结果比对）来源清晰：目标仓有邻居 → 对齐邻居；无邻居 → 基于 `references/host-driver-template.md`

**工程约定层面（目标：对齐邻居而非独立）**：
- [ ] 目录布局、文件命名、CMake 写法与目标仓邻居同类 story 对齐
- [ ] include 路径风格、顶层命名空间选择跟随目标仓惯例

注意："运行时不依赖其他 sample" 和 "工程约定向其他 sample 对齐" 是两件不冲突的事——前者禁止代码级依赖，后者鼓励风格一致。

---

## 17. 输入校验（可选）

如果做了输入校验：
- [ ] 拆成独立 `check_inputs()`，不混进 Meta / shape inference
- [ ] 只做静态可推导的校验（attr / rank / dtype / 直接可推导的跨张量关系）
- [ ] 不做依赖张量数值才能判定的校验（避免 D2H 同步）

---

## 18. 结果汇报模板

完成后至少输出：

### 18.1 目标文件集合
落地了哪些 `.h` / `.cpp`

### 18.2 删除的相对 include
精确到文件和 include 语句

### 18.3 本地化依赖清单
按原来源头文件分组列出搬运符号

### 18.4 `local_deps.h` 职责
装了哪类依赖

### 18.5 剩余外部假设
哪些宏 / tiling / dtype 仍依赖外部环境

### 18.6 静态验证结论
`..` include / 错误命名空间 `using` / 本地闭包是否通过

### 18.7 原则合规声明
- kernel 实现代码零修改（除允许的 3 类修改外）？
- tiling 计算逻辑零修改（除允许的 3 类修改外）？
- 全量迁移：当前算子源码整文件搬，没有未经授权的字段/路径/variant 裁剪？

---

## 一句话验收标准

**kernel 代码零修改；tiling 数学零修改；只改框架胶水；默认全量迁移；相对依赖清零；外部依赖最小闭包；自包含可编译；边界说明清楚。**
