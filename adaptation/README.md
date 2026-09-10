# Adaptation

第一阶段模块，负责 adaptation 产物检查、已完成 adaptation 的 demo 批量运行，以及产物打包。

常用命令：

```bash
uv run python adaptation/scripts/check_adaptation.py --adapt <name>
uv run python adaptation/scripts/verify_environment.py --adapt <name>
uv run python adaptation/scripts/adaptation_manager.py list --status completed
uv run python adaptation/scripts/adaptation_manager.py run --download-only
uv run python adaptation/scripts/adaptation_manager.py pack
```

`verify_environment.py` 在目标 adaptation 的 `.validation/<run_id>/` 中复制交付文件并创建独立 uv 环境，严格按 `uv.lock` 和对应的 `ascend`/`cuda` extra 安装依赖，然后执行 `demo.py --smoke-test`。成功后删除临时目录并保留 `environment_validation.json` 与 `environment_validation.log`；失败时保留现场。该验证是写入 `adaptation_status=completed` 的强制门禁。
