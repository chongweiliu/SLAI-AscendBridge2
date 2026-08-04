# Host 驱动代码模板（fallback 参考）

**何时读本文件**：只有当目标仓库里**没有**可参考的 standalone story / sample host 驱动惯例时才读。

**何时不该读**：
- 只做 kernel 依赖解耦，不涉及 host driver → 完全不需要
- 目标仓库已有 1+ 个同类 story，有现成的 ACL init 封装、main 骨架、golden 比对模式 → 按目标仓惯例写，不要用本模板（会造成风格漂移）
- 集成到 PyTorch extension / aclnn wrapper / CI sample 体系 → host 形态完全不同，不适用

---

## 模板定位

本模板是**最小可运行**的 standalone host 驱动骨架，包含：
- ACL init / finalize
- BFloat16 host 侧辅助（`FloatToBf16Bits` / `Bf16BitsToFloat` / `SampleBFloat16`）
- 类型转换 trait（`FromFloat` / `ToFloat` / `GetTolerance` / `GetDtypeName`）
- 错误检查（`CheckAcl`）
- 结果比对（`CompareBuffer`）
- `main` 骨架

替换 `/* OPERATOR-SPECIFIC */` 标注部分即可上手。

---

## 完整模板

```cpp
#include "acl/acl.h"
#include "acl/acl_rt.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "kernel_operator.h"
/* OPERATOR-SPECIFIC: include kernel 入口头和 host tiling 头 */
// #include "xxx_apt.h"
// #include "xxx_host_tiling.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {

/* OPERATOR-SPECIFIC: 问题规模常量 */
// constexpr size_t kRowLen = 128;
// constexpr size_t kColLen = 256;

constexpr int kMaxErrorElemNum = 10;
constexpr float kFloatTolerance = 1e-4f;
constexpr float kHalfTolerance = 5e-2f;
constexpr float kBFloat16Tolerance = 5e-2f;

// ---- BFloat16 host 侧辅助（AscendC bfloat16_t 仅 device 可用）----
using SampleHalf = half;
struct SampleBFloat16 {
    uint16_t bits;
};

uint16_t FloatToBf16Bits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1U;
    const uint32_t roundingBias = 0x7FFFU + lsb;
    return static_cast<uint16_t>((bits + roundingBias) >> 16);
}

float Bf16BitsToFloat(uint16_t bits)
{
    const uint32_t value = static_cast<uint32_t>(bits) << 16;
    float result = 0.0f;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

// ---- 类型转换 trait ----
template <typename T>
T FromFloat(float value) { return static_cast<T>(value); }

template <>
SampleBFloat16 FromFloat<SampleBFloat16>(float value) { return SampleBFloat16{FloatToBf16Bits(value)}; }

template <typename T>
float ToFloat(T value) { return static_cast<float>(value); }

template <>
float ToFloat<SampleBFloat16>(SampleBFloat16 value) { return Bf16BitsToFloat(value.bits); }

template <typename T> float GetTolerance();
template <> float GetTolerance<float>()           { return kFloatTolerance; }
template <> float GetTolerance<SampleHalf>()      { return kHalfTolerance; }
template <> float GetTolerance<SampleBFloat16>()  { return kBFloat16Tolerance; }

template <typename T> const char *GetDtypeName();
template <> const char *GetDtypeName<float>()           { return "float32"; }
template <> const char *GetDtypeName<SampleHalf>()      { return "float16"; }
template <> const char *GetDtypeName<SampleBFloat16>()  { return "bfloat16"; }

// ---- ACL 基础 ----
void CheckAcl(aclError ret, const char *msg)
{
    if (ret != ACL_SUCCESS) {
        std::ostringstream oss;
        oss << msg << " failed. aclError=" << ret;
        throw std::runtime_error(oss.str());
    }
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

// ---- 结果比对 ----
template <typename T>
size_t CompareBuffer(const std::string &name, const std::vector<T> &actual,
    const std::vector<T> &expected, float atol)
{
    if (actual.size() != expected.size()) {
        throw std::runtime_error(name + " size mismatch");
    }
    size_t mismatchCount = 0;
    float maxAbsErr = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float act = ToFloat(actual[i]);
        const float exp = ToFloat(expected[i]);
        const float err = std::fabs(act - exp);
        maxAbsErr = std::max(maxAbsErr, err);
        if (err > atol) {
            if (mismatchCount < static_cast<size_t>(kMaxErrorElemNum)) {
                std::cout << name << " mismatch[" << i << "]: expected=" << exp
                          << ", actual=" << act << ", abs_err=" << err << std::endl;
            }
            ++mismatchCount;
        }
    }
    std::cout << name << ": total=" << actual.size() << ", mismatch=" << mismatchCount
              << ", max_abs_err=" << maxAbsErr << std::endl;
    return mismatchCount;
}

/* OPERATOR-SPECIFIC: 数据生成 —— 确定性，不依赖 Python/numpy */
// template <typename T>
// void BuildInput(std::vector<T> &data, size_t rowLen, size_t colLen) { ... }

/* OPERATOR-SPECIFIC: Golden 计算 —— 用 float 精度算完再转回目标类型 */
// template <typename T>
// void ComputeReference(const std::vector<T> &input, std::vector<T> &output) { ... }

/* OPERATOR-SPECIFIC: Kernel launch 分发 —— 按 tilingKey 选择入口 */
// template <typename T>
// void LaunchKernel(GM_ADDR inputDevice, GM_ADDR outputDevice, GM_ADDR workspaceDevice,
//     const XxxLaunchConfig &launchConfig, aclrtStream stream)
// {
//     if constexpr (std::is_same_v<T, float>) {
//         xxx_fp32<<<launchConfig.blockDim, 0, stream>>>(inputDevice, outputDevice, workspaceDevice, launchConfig.tiling);
//     } else if constexpr (std::is_same_v<T, SampleHalf>) {
//         xxx_fp16<<<launchConfig.blockDim, 0, stream>>>(inputDevice, outputDevice, workspaceDevice, launchConfig.tiling);
//     } else { ... }
// }

/* OPERATOR-SPECIFIC: 单次测试流程 */
template <typename T>
void RunOneCase(aclrtStream stream)
{
    /* 1. 计算 size */
    // const size_t inputSize = ...;
    // const size_t outputSize = ...;

    /* 2. 生成输入 + golden */
    // std::vector<T> inputData;
    // BuildInput(inputData, ...);
    // std::vector<T> outputGolden;
    // ComputeReference(inputData, outputGolden);
    // std::vector<T> outputActual(outputElemNum, FromFloat<T>(0.0f));

    /* 3. 计算 tiling + launch config */
    // auto launchConfig = XxxHost::CalcLaunchConfig(...);

    /* 4. 设备内存分配 —— GM_ADDR 贯穿 */
    GM_ADDR inputDevice = nullptr;
    GM_ADDR outputDevice = nullptr;
    GM_ADDR workspaceDevice = nullptr;

    // CheckAcl(aclrtMalloc((void **)&inputDevice, inputSize, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc input");
    // CheckAcl(aclrtMalloc((void **)&outputDevice, outputSize, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc output");
    // CheckAcl(aclrtMalloc((void **)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc workspace");

    try {
        /* 5. H2D 拷贝 */
        // CheckAcl(aclrtMemcpy(inputDevice, inputSize, inputData.data(), inputSize, ACL_MEMCPY_HOST_TO_DEVICE), "H2D input");

        /* 6. Launch + Sync */
        // LaunchKernel<T>(inputDevice, outputDevice, workspaceDevice, launchConfig, stream);
        // CheckAcl(aclrtSynchronizeStream(stream), "sync");

        /* 7. D2H 拷贝 + 验证 */
        // CheckAcl(aclrtMemcpy(outputActual.data(), outputSize, outputDevice, outputSize, ACL_MEMCPY_DEVICE_TO_HOST), "D2H output");
        // size_t mismatch = CompareBuffer(GetDtypeName<T>(), outputActual, outputGolden, GetTolerance<T>());
        // if (mismatch != 0) { throw std::runtime_error("result check failed"); }
        // std::cout << GetDtypeName<T>() << " run succeeded" << std::endl;
    } catch (...) {
        if (workspaceDevice != nullptr) { aclrtFree(workspaceDevice); }
        if (outputDevice != nullptr) { aclrtFree(outputDevice); }
        if (inputDevice != nullptr) { aclrtFree(inputDevice); }
        throw;
    }

    CheckAcl(aclrtFree(workspaceDevice), "free workspace");
    CheckAcl(aclrtFree(outputDevice), "free output");
    CheckAcl(aclrtFree(inputDevice), "free input");
}

void RunSample(aclrtStream stream)
{
    /* OPERATOR-SPECIFIC: 按需测试各种 dtype */
    RunOneCase<SampleHalf>(stream);
    RunOneCase<float>(stream);
    RunOneCase<SampleBFloat16>(stream);
}
} // namespace

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    try {
        RunSample(stream);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        ret = 1;
    }

    Finalize(deviceId, stream);
    return ret;
}
```

---

## 模板使用说明

### 需要替换的部分

| 替换点 | 说明 |
|--------|------|
| include 头 | 实际的入口头和 host tiling 头 |
| 问题规模常量 | 合理的测试 shape |
| `BuildInput` | 确定性数据生成（线性递推 + 取模），不依赖 Python |
| `ComputeReference` | 用 float 精度计算 golden，再转回目标类型 |
| `LaunchKernel` | 按 tilingKey 分发到各入口函数 |
| `RunOneCase` | 填入实际 size 计算、tiling 调用、内存拷贝 |

### 直接复用的部分

- ACL init / finalize
- BFloat16 host 辅助（`FloatToBf16Bits` / `Bf16BitsToFloat` / `SampleBFloat16`）
- 类型转换 trait（`FromFloat` / `ToFloat` / `GetTolerance` / `GetDtypeName`）
- `CheckAcl` 错误检查
- `CompareBuffer` 结果比对
- `main` 骨架

### 注意事项

- `GM_ADDR` 要贯穿 host / launch / kernel 三侧，不要用 `void*` 中转
- host 侧数据类型（`SampleBFloat16`）仅用于 host 侧数据生成和 golden，launch 时应与 kernel 模板参数（`bfloat16_t`）严格分离
- CMakeLists.txt 属于"工程约定"，按目标仓既有 cmake 体系对齐，不由本模板提供
