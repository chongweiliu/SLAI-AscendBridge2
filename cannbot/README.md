# CANNBot

This directory keeps the upstream CANNBot skills repository as a separate open-source asset used by the SLAI-AscendBridge2 collaborative operator adaptation flow. It is intentionally not installed into this project as first-party `.claude/agents` or `.claude/skills` content.

## Source

- GitCode: https://gitcode.com/cann/cannbot-skills
- Branch: `master`
- Commit: `51507611e119c06acdea9b981db1fae6c61b7da0`
- Last commit: `5150761 2026-07-29 白盒用例设计SKILL V2.4优化`

## Layout

- `cannbot-skills/plugins-official/ops-direct-invoke/`: CANNBot four-role direct-invoke operator workflow.
- `cannbot-skills/ops/`: Ascend C skills used by the CANNBot roles.
- `cannbot-skills/infra/`: GitCode / infrastructure helper skills from the upstream repository.

When CANNBot collaborative adaptation is needed, use the material in this directory as the source of truth. Do not copy it into `.claude/agents` or `.claude/skills` unless explicitly requested.
