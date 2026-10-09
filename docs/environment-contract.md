# Environment contract

SLAI-AscendBridge2 has two intentionally separate Python environments. Keeping
this boundary explicit prevents a framework utility environment from being
mistaken for the environment that loads a model.

## Root framework environment

The repository root is the framework environment. Create it with:

```bash
uv sync
uv run python scripts/board_ops.py init
```

Use it for board operations, checkers, dashboards, packaging helpers, and
other repository tooling. The root `pyproject.toml` does not define `ascend`
or `cuda` optional extras. Therefore `uv sync --extra ascend` from the root is
not a supported model setup command.

The root environment may contain framework-level dependencies useful for
inspection. It is not the source of truth for a model's PyTorch, CANN, or
`torch-npu` ABI.

## Adaptation environment

Every `adaptations/{name}` directory is an independent model environment. Its
`pyproject.toml` and `uv.lock` are the source of truth for model execution.
Run model code and stage-specific benchmarks from that directory:

```bash
cd adaptations/{name}
uv sync --locked --extra ascend
uv run --no-sync --extra ascend python demo.py --smoke-test

# CUDA validation, when the adaptation defines the CUDA extra
uv sync --locked --extra cuda
uv run --no-sync --extra cuda python accuracy_run.py
```

The `ascend` and `cuda` extras are mutually exclusive in the adaptation
template. Do not copy the root `.venv` into an adaptation or invoke model code
with the root `.venv/bin/python`; doing so bypasses the adaptation lockfile and
can silently mix incompatible PyTorch and `torch-npu` builds.

Repository checkers can run from the root because they inspect an adaptation
path. The command that imports the model must still run in the adaptation
environment. `--no-sync` is appropriate after the selected extra has already
been synchronized.

## Version selection

There is no single CANN/PyTorch version valid for every Ascend generation.
Select a compatibility profile before synchronization and record the selected
profile, actual versions, and hardware in the adaptation README or run notes.
See `.claude/skills/uv-env-setup/references/compatibility-profiles.md`.
`torch` and `torch-npu` must use the exact pair required by the selected
`torch-npu` wheel; a broad lower bound is not an ABI contract.

If a machine does not match a documented profile, stop preflight with the
actual CANN, Python, PyTorch, and `torch-npu` versions and add a validation
record before attempting model adaptation.
