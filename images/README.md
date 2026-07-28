# vLLM-Ascend 离线镜像目录

本目录承载项目自闭环所需的 vLLM-Ascend Docker 镜像离线 tar 包，使部署脱离外部 registry 也能复现。

## 命名规范

```
vllm-ascend-<version>-<variant>.tar        # 镜像 tar
vllm-ascend-<version>-<variant>.tar.sha256 # sha256 校验
```

- `<version>`：vLLM-Ascend 发布 tag，如 `v0.23.0rc1`、`v0.18.0`
- `<variant>`：硬件变体，如 `a3`（Atlas 800I A3 / A3 Training）、`a2`（默认 Atlas 800I A2）、`310p`、`a5`、`openEuler` 系列

## 当前清单

| 文件 | 来源 tag | 变体 | 说明 |
|------|---------|------|------|
| `vllm-ascend-v0.23.0rc1-a3.tar` | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` | A3 | 示例版本，按需生成（见下） |

> 镜像 tar 不纳入 Git。请在能访问镜像仓库的环境中运行
> `fetch-image.sh`，或按下方命令手动生成。

## 生成 tar

```bash
# 推荐：crane 直接拉成 docker-load 兼容 tar（绕开 daemon，单步）
bash images/fetch-image.sh v0.23.0rc1 a3
# 或回退 docker pull + docker save
```

`fetch-image.sh` 优先使用 `PATH` 中的 crane，也可通过 `CRANE_BIN`
指定命令或可执行文件路径；crane 不可用或拉取失败时回退到
docker pull + docker save。crane 可从
https://github.com/google/go-containerregistry/releases 获取。

### 加载镜像（裸机/SSH 节点）

```bash
sha256sum -c images/vllm-ascend-v0.23.0rc1-a3.tar.sha256      # 完整性校验
docker load -i images/vllm-ascend-v0.23.0rc1-a3.tar          # 加载到本机 daemon
docker images | grep vllm-ascend                              # 确认
```

加载后镜像保持原始 name:tag（`quay.io/ascend/vllm-ascend:v0.23.0rc1-a3`），可直接 `docker run`。

### deployer 引用

`vllm-ascend-auto-deploy` Skill 的"镜像来源（自闭环）"流程会先查本目录：

- 命中 → 配置摘要标 `image_source=local_tar`、`image_local_tar=images/<file>.tar`，裸机一键脚本 `deploy-baremetal.sh` 自动 `docker load`。
- 未命中 → 取远端 registry 全限定名，`image_source=remote_registry`。
- KTP/scheduler：manifest 的 `image:` 仍写镜像名；若平台 registry 未预置，操作员需先将本 tar `docker load` 后 `docker tag` + `docker push` 到平台 registry。

## 新增镜像

```bash
docker pull quay.io/ascend/vllm-ascend:<tag>
docker save -o images/vllm-ascend-<version>-<variant>.tar quay.io/ascend/vllm-ascend:<tag>
sha256sum images/vllm-ascend-<version>-<variant>.tar > images/vllm-ascend-<version>-<variant>.tar.sha256
```

## Git 策略

- `.tar` / `.tar.sha256` **不入库**（`.gitignore` 忽略），体积过大。
- 本 `README.md` 入库，作为目录索引与命名约定来源。
- 镜像 tar 由运维在每台目标机器或共享存储上准备；自闭环指"项目携带可复现的镜像获取与加载流程"，不要求把 16GB 二进制纳入 git。
