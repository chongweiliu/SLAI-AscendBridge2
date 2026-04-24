# Adaptation

第一阶段模块，负责 adaptation 产物检查、已完成 adaptation 的 demo 批量运行，以及产物打包。

常用命令：

```bash
uv run python adaptation/scripts/check_adaptation.py --adapt <name>
uv run python adaptation/scripts/adaptation_manager.py list --status completed
uv run python adaptation/scripts/adaptation_manager.py run --download-only
uv run python adaptation/scripts/adaptation_manager.py pack
```
