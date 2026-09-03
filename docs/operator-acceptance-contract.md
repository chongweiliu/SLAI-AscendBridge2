# 自定义算子结构化验收契约

每个自定义算子必须在 `adaptations/<model>/operators/<operator>/acceptance.json` 保存验收证据。布尔值只能来自实际命令结果，不能由 Agent 根据文字描述推断。

```json
{
  "contract_version": 1,
  "operator": "example_op",
  "search": {
    "torch_npu_native_interface_found": false,
    "torch_npu_native_evidence": "已检查的 torch_npu 版本、API 和源码位置",
    "torch_npu_composed_implementation_used": false,
    "community_existing_implementation_found": false,
    "community_candidates_reviewed": true,
    "community_report": "operators/example_op/community_search.json"
  },
  "reference": {
    "implementation_source": "operators/example_op/reference/upstream.cu",
    "golden_implementation": "operators/example_op/scripts/golden.py"
  },
  "build": {
    "shared_library": "operators/example_op/build/libexample_op.so",
    "registered_op": "torch.ops.npu.example_op",
    "load_passed": true,
    "registration_passed": true
  },
  "validation": {
    "pretrained_weights": true,
    "sample_count": 50,
    "golden_passed": true,
    "dtype_coverage_passed": true,
    "shape_coverage_passed": true,
    "non_contiguous_passed": true,
    "stream_consistency_passed": true,
    "thresholds_passed": true,
    "dtypes": ["float32"],
    "shapes": ["[2, 16, 128]", "[1, 1, 128]"],
    "repeat_calls": 50,
    "metrics": {
      "max_abs_error": 0.00001,
      "mere": 0.00001,
      "mare": 0.0001
    }
  },
  "integration": {
    "enabled": true,
    "fallback_used": false,
    "invocation_count": 50
  }
}
```

社区报告必须由下面的全量搜索命令生成：

```bash
python scripts/search_operator_communities.py \
  --operator example_op \
  --query aten::example_op \
  --query example_op \
  --query "example reduction" \
  --output adaptations/<model>/operators/example_op/community_search.json
```

该命令只使用 GitCode 服务端 API，不得 clone、fetch 或缓存任何仓库源码。每次先重新分页枚举 [Ascend](https://gitcode.com/Ascend) 和 [CANN](https://gitcode.com/cann) 的全部公开仓库，再对两个 namespace 与每个关键词完整翻页执行代码搜索。任一枚举/搜索分页失败、数量不一致或服务端标记截断，都会令报告 `complete=false`，命令退出码为 1。

最终验收：

```bash
python scripts/check_operator_acceptance.py --adapt adaptations/<model>
```

检查失败时不得进入 benchmark，也不得回报 `operator_gap_fixed`。
