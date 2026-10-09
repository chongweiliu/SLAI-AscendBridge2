# Ascend compatibility profiles

These profiles are the version contract used by preflight. They are starting
points, not a promise that one environment works on every device. Always
record the detected hardware and the versions reported by the running
interpreter.

| Profile | Hardware target | CANN | PyTorch | torch-npu | Python | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| `ascend-910-legacy` | 910 deployments using the legacy workflow | 8.5.x | 2.9.x | 2.9.x | 3.10–3.12 | Existing workflow and legacy validation |
| `ascend-910c-current` | 910C/910B current path | 9.0.0 | 2.10.0 | 2.10.0 | 3.10–3.12 | Current adaptation template and 910C machine validation |
| `ascend-950-preview` | Ascend 950 series | 9.0.0-beta.2 | 2.12.x | 2.12.x | Match the published wheel | Hardware validation required |
| `cuda` | NVIDIA CUDA validation | N/A | CUDA wheel selected by the adaptation | N/A | Match the PyTorch wheel | Adaptation-specific |

The `ascend-910-legacy` row preserves the versions used by the existing
CANNBot workflow. It is not a global requirement. The `ascend-910c-current`
row is the default documented by the current `uv-env-setup` template. The 950
row remains a preview until the target machine has been tested.

Before installing, preflight must identify the device generation and loaded
CANN toolkit, inspect the selected `torch-npu` wheel metadata and its exact
PyTorch requirement, verify Python support, and return `env_ok=false` when no
profile covers the observed combination.

Do not resolve a mismatch by independently upgrading only PyTorch or only
`torch-npu`. Re-select a profile or document and validate a new pair first.
