# Claude Code + Agent Teams 多智能体协作环境搭建指南

> 适用于中国大陆服务器环境（无法直接访问 claude.ai），使用国产大模型 API 驱动 Claude Code。

---

## 目录

1. [环境要求](#1-环境要求)
2. [安装 Node.js](#2-安装-nodejs)
3. [安装 Claude Code](#3-安装-claude-code)
4. [配置第三方 API（国产大模型）](#4-配置第三方-api国产大模型) — MiniMax / DeepSeek / GLM / 通义千问 / Kimi / 豆包 / 腾讯混元
5. [开启 Agent Teams 多智能体协作](#5-开启-agent-teams-多智能体协作)
6. [验证安装](#6-验证安装)
7. [使用 Agent Teams](#7-使用-agent-teams)
8. [自定义 Subagent 定义](#8-自定义-subagent-定义)
9. [常见问题](#9-常见问题)

---

## 1. 环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Linux（Ubuntu、openEuler、CentOS 等均可） |
| 架构 | x86_64 或 aarch64 |
| Node.js | >= 18（推荐 22.x LTS） |
| 网络 | 能访问 `nodejs.org`、`registry.npmjs.org` 及所选厂商的 API 域名 |
| Claude Code | >= 2.1.32（Agent Teams 最低版本要求） |

> **注意**：中国大陆服务器无法访问 `claude.ai`，官方安装脚本 `curl -fsSL https://claude.ai/install.sh | bash` 会返回 "App unavailable in region"。本指南使用 npm 方式安装。

---

## 2. 安装 Node.js

系统仓库通常自带的 Node.js 版本太旧（如 12.x），需要手动安装 22.x LTS。

### 2.1 下载官方二进制包

根据你的 CPU 架构选择对应版本：

```bash
# aarch64（ARM64）
curl -fsSL https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-arm64.tar.xz -o /tmp/node-v22.tar.xz

# x86_64（AMD64）
curl -fsSL https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-x64.tar.xz -o /tmp/node-v22.tar.xz
```

### 2.2 解压并安装

```bash
cd /tmp
tar -xf node-v22.tar.xz
cp -r node-v22.14.0-linux-*/bin /usr/local/
cp -r node-v22.14.0-linux-*/include /usr/local/
cp -r node-v22.14.0-linux-*/lib /usr/local/
cp -r node-v22.14.0-linux-*/share /usr/local/
rm -rf /tmp/node-v22*
```

### 2.3 验证

```bash
node --version   # 应输出 v22.14.0
npm --version    # 应输出 10.9.2
```

---

## 3. 安装 Claude Code

### 3.1 通过 npm 全局安装

```bash
npm install -g @anthropic-ai/claude-code
```

安装过程约 30-60 秒，取决于网络速度。

### 3.2 验证安装

```bash
claude --version
# 输出示例：2.1.79 (Claude Code)

which claude
# 输出：/usr/local/bin/claude
```

### 3.3 安装位置说明

| 文件 | 路径 |
|------|------|
| 可执行文件 | `/usr/local/bin/claude`（符号链接） |
| 实际代码 | `/usr/local/lib/node_modules/@anthropic-ai/claude-code/` |
| 配置目录 | `~/.claude/` |
| 用户配置 | `~/.claude.json` |

### 3.4 后续升级

npm 安装方式不会自动更新，需手动升级：

```bash
npm update -g @anthropic-ai/claude-code
```

---

## 4. 配置第三方 API（国产大模型）

由于中国大陆无法直接使用 Anthropic 官方 API，我们通过国产大模型厂商提供的 **Anthropic 兼容接口**来驱动 Claude Code。

以下是目前支持接入 Claude Code 的主流国产大模型，任选其一配置即可。

---

### 4.0 配置原理

Claude Code 通过三个核心环境变量连接后端模型：

| 变量 | 作用 |
|------|------|
| `ANTHROPIC_BASE_URL` | API 服务地址（各厂商不同） |
| `ANTHROPIC_AUTH_TOKEN` | API 密钥 |
| `ANTHROPIC_MODEL` | 默认使用的模型名称 |

所有厂商的配置方式完全相同，只是这三个值不同。

**配置优先级**（从高到低）：

1. 环境变量（终端 `export`）
2. 项目级 `.claude/settings.local.json`
3. 项目级 `.claude/settings.json`
4. 用户级 `~/.claude/settings.json`

> **重要**：如果终端中已有 `ANTHROPIC_AUTH_TOKEN` 或 `ANTHROPIC_BASE_URL` 环境变量，会覆盖配置文件。切换模型前务必先 `unset`：
>
> ```bash
> unset ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL ANTHROPIC_MODEL
> ```

---

### 4.1 厂商总览

> **最后更新：2026 年 3 月**。模型迭代很快，请以各厂商官网为准。

| 厂商 | 最新旗舰模型 | Base URL | API Key 获取地址 | 特点 |
|------|-------------|----------|-----------------|------|
| **MiniMax** | MiniMax-M2.5 | `https://api.minimaxi.com/anthropic` | [platform.minimaxi.com](https://platform.minimaxi.com/) | SWE-Bench 80.2%，性价比极高 |
| **DeepSeek** | deepseek-chat (V3.2) / deepseek-reasoner | `https://api.deepseek.com/anthropic` | [platform.deepseek.com](https://platform.deepseek.com/api_keys) | 价格最低，推理能力强 |
| **智谱 GLM** | glm-5 / glm-4.7 | `https://open.bigmodel.cn/api/anthropic` | [bigmodel.cn](https://bigmodel.cn/) | 745B 旗舰，智能路由 |
| **通义千问** | qwen3-coder-plus (480B) | `https://dashscope.aliyuncs.com/compatible-mode/anthropic` | [百炼控制台](https://bailian.console.aliyun.com/) | 256K 上下文，可扩展至 1M |
| **Kimi** | kimi-k2.5 | `https://api.moonshot.cn/anthropic` | [platform.moonshot.cn](https://platform.moonshot.cn/console/api-keys) | 1T 参数 MoE，多模态原生 |
| **豆包** | doubao-seed-1.6 / doubao-seed-code | `https://ark.cn-beijing.volces.com/api/compatible` | [火山引擎控制台](https://console.volcengine.com/ark) | 字节旗下，编程场景优化 |
| **腾讯混元** | hunyuan-2.0-thinking (HY 3.0 预计 4 月) | `https://api.hunyuan.cloud.tencent.com/anthropic` | [腾讯云控制台](https://console.cloud.tencent.com/hunyuan) | 腾讯云生态，MoE 406B |

---

### 4.2 各厂商详细配置

以下每个配置都是完整的 `~/.claude/settings.json` 文件内容，**复制后只需替换 API Key**。

#### A. MiniMax

> 官方文档：https://platform.minimaxi.com/docs/coding-plan/claude-code

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.minimaxi.com/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "你的API_Key",
    "API_TIMEOUT_MS": "3000000",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "ANTHROPIC_MODEL": "MiniMax-M2.5",
    "ANTHROPIC_SMALL_FAST_MODEL": "MiniMax-M2.5",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "MiniMax-M2.5",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "MiniMax-M2.5",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "MiniMax-M2.5"
  }
}
```

**可用模型**（2026.03）：

| 模型名 | 参数量 | 说明 |
|--------|--------|------|
| `MiniMax-M2.5` | 230B MoE | 最新旗舰，SWE-Bench 80.2%，Multi-SWE-Bench 第一 |
| `MiniMax-M2.5-Lightning` | 同上 | 高速版，100 tokens/s，成本仅 $1/小时 |
| `MiniMax-M2.7` | — | 上一代标准版 |
| `MiniMax-M2.7-highspeed` | — | 上一代高速版 |

---

#### B. DeepSeek

> API Key 获取：https://platform.deepseek.com/api_keys

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "你的API_Key",
    "API_TIMEOUT_MS": "3000000",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "ANTHROPIC_MODEL": "deepseek-chat",
    "ANTHROPIC_SMALL_FAST_MODEL": "deepseek-chat",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "deepseek-chat",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "deepseek-reasoner",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "deepseek-chat"
  }
}
```

**可用模型**（2026.03）：

| 模型名 | 当前版本 | 说明 |
|--------|---------|------|
| `deepseek-chat` | V3.2 | 通用对话 + 工具调用，128K 上下文，性价比最高 |
| `deepseek-reasoner` | V3.2 | 深度推理（R1），复杂任务 + 编程更强 |

> **推荐策略**：Opus 用 `deepseek-reasoner`（复杂推理），Sonnet/Haiku 用 `deepseek-chat`（快速响应）。
> DeepSeek 会自动将 `deepseek-chat` 和 `deepseek-reasoner` 指向最新版本，无需手动改版本号。

---

#### C. 智谱 GLM

> API Key 获取：https://bigmodel.cn/
>
> 编码套餐：[GLM Coding Plan](https://codingplan.org/plans/zhipu)（Lite ¥49/月起）

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://open.bigmodel.cn/api/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "你的API_Key",
    "API_TIMEOUT_MS": "3000000",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "ANTHROPIC_MODEL": "glm-5",
    "ANTHROPIC_SMALL_FAST_MODEL": "glm-4.5-air",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-4.7",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "glm-5",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "glm-4.5-air"
  }
}
```

**可用模型**（2026.03）：

| 模型名 | 参数量 | 说明 |
|--------|--------|------|
| `glm-5` | 745B MoE (44B 激活) | 2026 最新旗舰，200K 上下文，对标 Claude Opus |
| `glm-4.7` | 355B MoE (32B 激活) | 编码 SOTA（SWE-Bench 73.8%），MIT 开源 |
| `glm-4.5-air` | — | 轻量版，速度快、成本低 |

> **推荐策略**：Opus 用 `glm-5`（最强推理），Sonnet 用 `glm-4.7`（编程主力），Haiku 用 `glm-4.5-air`（快速响应）。
> GLM-5 在 Pro/Max 套餐中消耗 3 倍额度，按需使用。

---

#### D. 通义千问 Qwen

> API Key 获取：https://bailian.console.aliyun.com/
>
> 参考文档：[让 Claude Code 使用 Qwen3-Coder](https://qwenlm.github.io/zh/blog/qwen3-coder/)

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "你的API_Key",
    "API_TIMEOUT_MS": "3000000",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "ANTHROPIC_MODEL": "qwen3-coder-plus",
    "ANTHROPIC_SMALL_FAST_MODEL": "qwen3-coder-plus",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "qwen3-coder-plus",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "qwen3-coder-plus",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen3-coder-plus"
  }
}
```

**可用模型**（2026.03）：

| 模型名 | 参数量 | 说明 |
|--------|--------|------|
| `qwen3-coder-plus` | 480B MoE (35B 激活) | 编程专项优化，256K 上下文，可扩展至 1M |
| `qwen-plus` | — | 通用版 |

> Qwen3.5 系列（35B/122B）已于 2026.03 开源，API 名称可能随后更新，请关注百炼控制台。

---

#### E. Kimi（月之暗面）

> API Key 获取：https://platform.moonshot.cn/console/api-keys

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.moonshot.cn/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "你的API_Key",
    "API_TIMEOUT_MS": "3000000",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "ANTHROPIC_MODEL": "kimi-k2.5",
    "ANTHROPIC_SMALL_FAST_MODEL": "kimi-k2.5",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "kimi-k2.5",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "kimi-k2.5",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "kimi-k2.5"
  }
}
```

**可用模型**（2026.03）：

| 模型名 | 参数量 | 说明 |
|--------|--------|------|
| `kimi-k2.5` | 1T MoE (32B 激活) | 最新旗舰，256K 上下文，原生多模态，支持 100 子智能体并行 |
| `kimi-k2-thinking` | — | 深度思考版 |
| `kimi-k2-turbo-preview` | — | 上一代高速版 |

> **海外用户**可使用国际域名：`https://api.moonshot.ai/anthropic`

---

#### F. 豆包（火山引擎）

> API Key 获取：https://console.volcengine.com/ark
>
> 参考文档：阮一峰《国产大模型接入 Claude Code 教程》

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://ark.cn-beijing.volces.com/api/compatible",
    "ANTHROPIC_AUTH_TOKEN": "你的API_Key",
    "API_TIMEOUT_MS": "3000000",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "ANTHROPIC_MODEL": "doubao-seed-code-preview-latest",
    "ANTHROPIC_SMALL_FAST_MODEL": "doubao-seed-1.6-flash",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "doubao-seed-code-preview-latest",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "doubao-seed-1.6-thinking",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "doubao-seed-1.6-flash"
  }
}
```

**可用模型**（2026.03）：

| 模型名 | 说明 |
|--------|------|
| `doubao-seed-code-preview-latest` | 编程专项优化，256K 上下文，原生兼容 Anthropic API |
| `doubao-seed-1.6` | 全能综合型，首个支持 256K 的思考模型 |
| `doubao-seed-1.6-thinking` | 深度思考强化版，代码 + 数学 + 逻辑推理增强 |
| `doubao-seed-1.6-flash` | 极速版，响应延迟仅 10ms |

> **推荐策略**：编程任务用 `doubao-seed-code-preview-latest`，复杂推理用 `doubao-seed-1.6-thinking`，快速任务用 `doubao-seed-1.6-flash`。

---

#### G. 腾讯混元

> API Key 获取：https://console.cloud.tencent.com/hunyuan
>
> 兼容接口文档：https://cloud.tencent.com/document/product/1729/127293

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.hunyuan.cloud.tencent.com/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "你的API_Key",
    "API_TIMEOUT_MS": "3000000",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "ANTHROPIC_MODEL": "hunyuan-2.0-thinking-20251109",
    "ANTHROPIC_SMALL_FAST_MODEL": "hunyuan-2.0-instruct-20251111",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "hunyuan-2.0-thinking-20251109",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "hunyuan-2.0-thinking-20251109",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "hunyuan-2.0-instruct-20251111"
  }
}
```

**可用模型**（2026.03）：

| 模型名 | 参数量 | 说明 |
|--------|--------|------|
| `hunyuan-2.0-thinking-20251109` | 406B MoE (32B 激活) | 深度推理，数理顶尖 |
| `hunyuan-2.0-instruct-20251111` | 同上 | 指令跟随，快速响应 |

> **HY 3.0** 预计 2026 年 4 月推出，推理和 Agent 能力将有重大升级。

---

### 4.3 通用配置步骤（所有厂商通用）

无论选择哪个厂商，配置步骤都是相同的：

#### 步骤一：获取 API Key

到对应厂商的平台注册账号，创建 API Key。

#### 步骤二：写入 settings.json

```bash
mkdir -p ~/.claude
# 将上面对应厂商的 JSON 内容写入
cat > ~/.claude/settings.json << 'EOF'
{
  这里粘贴对应厂商的完整 JSON 配置
}
EOF
```

#### 步骤三：写入 .claude.json（跳过首次引导）

```bash
cat > ~/.claude.json << 'EOF'
{
  "hasCompletedOnboarding": true
}
EOF
```

#### 步骤四：清除可能冲突的环境变量

```bash
unset ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL ANTHROPIC_MODEL
```

#### 步骤五：启动验证

```bash
cd /你的项目目录
claude
```

### 4.4 多厂商快速切换

如果你有多个厂商的 API Key，可以通过 shell 函数实现一键切换：

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc

claude-minimax() {
  export ANTHROPIC_BASE_URL="https://api.minimaxi.com/anthropic"
  export ANTHROPIC_AUTH_TOKEN="你的MiniMax_Key"
  export ANTHROPIC_MODEL="MiniMax-M2.5"
  claude "$@"
}

claude-deepseek() {
  export ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic"
  export ANTHROPIC_AUTH_TOKEN="你的DeepSeek_Key"
  export ANTHROPIC_MODEL="deepseek-chat"
  claude "$@"
}

claude-glm() {
  export ANTHROPIC_BASE_URL="https://open.bigmodel.cn/api/anthropic"
  export ANTHROPIC_AUTH_TOKEN="你的GLM_Key"
  export ANTHROPIC_MODEL="glm-5"
  claude "$@"
}

claude-qwen() {
  export ANTHROPIC_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/anthropic"
  export ANTHROPIC_AUTH_TOKEN="你的Qwen_Key"
  export ANTHROPIC_MODEL="qwen3-coder-plus"
  claude "$@"
}

claude-kimi() {
  export ANTHROPIC_BASE_URL="https://api.moonshot.cn/anthropic"
  export ANTHROPIC_AUTH_TOKEN="你的Kimi_Key"
  export ANTHROPIC_MODEL="kimi-k2.5"
  claude "$@"
}

claude-doubao() {
  export ANTHROPIC_BASE_URL="https://ark.cn-beijing.volces.com/api/compatible"
  export ANTHROPIC_AUTH_TOKEN="你的豆包_Key"
  export ANTHROPIC_MODEL="doubao-seed-code-preview-latest"
  claude "$@"
}

claude-hunyuan() {
  export ANTHROPIC_BASE_URL="https://api.hunyuan.cloud.tencent.com/anthropic"
  export ANTHROPIC_AUTH_TOKEN="你的混元_Key"
  export ANTHROPIC_MODEL="hunyuan-2.0-thinking-20251109"
  claude "$@"
}
```

使用方式：

```bash
claude-deepseek          # 用 DeepSeek 启动
claude-glm               # 用 GLM 启动
claude-minimax           # 用 MiniMax 启动
```

### 4.5 常见 Base URL 错误

Claude Code 会在 `ANTHROPIC_BASE_URL` 后自动追加 `/v1/messages`，因此：

```
正确：https://api.deepseek.com/anthropic
      → 实际请求：https://api.deepseek.com/anthropic/v1/messages

错误：https://api.deepseek.com/anthropic/v1
      → 实际请求：https://api.deepseek.com/anthropic/v1/v1/messages（路径重复！）
```

**不要**在 Base URL 末尾加 `/v1` 或 `/v1/messages`。

### 4.6 如何选择厂商

| 需求 | 推荐厂商 | 推荐模型 |
|------|---------|---------|
| 性价比最高 | DeepSeek | `deepseek-chat`（V3.2，价格最低） |
| 编程能力最强 | MiniMax / 通义千问 | `MiniMax-M2.5`（SWE-Bench 80.2%）/ `qwen3-coder-plus` |
| 深度推理 | 智谱 GLM / DeepSeek | `glm-5`（745B）/ `deepseek-reasoner` |
| 超长上下文（>200K） | Kimi / 通义千问 | `kimi-k2.5`（256K）/ `qwen3-coder-plus`（256K→1M） |
| 多模态（图像+代码） | Kimi | `kimi-k2.5`（原生多模态，无需额外视觉编码器） |
| 极速响应 | 豆包 | `doubao-seed-1.6-flash`（10ms 延迟） |
| 大厂云服务生态 | 通义千问 / 腾讯混元 | 阿里云 / 腾讯云原生集成 |
| 开源可自部署 | 智谱 GLM / 通义千问 | `glm-4.7`（MIT）/ Qwen3-Coder（开源） |
| 字节生态 | 豆包 | `doubao-seed-code-preview-latest` |

---

## 5. 开启 Agent Teams 多智能体协作

Agent Teams 是 Claude Code 的实验性功能，默认关闭，需要手动开启。

### 5.1 修改 settings.json

在 `~/.claude/settings.json` 的 `env` 部分添加：

```json
"CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
```

完整配置示例：

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.minimaxi.com/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "你的MiniMax_API_Key",
    "API_TIMEOUT_MS": "3000000",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "ANTHROPIC_MODEL": "MiniMax-M2.7-highspeed",
    "ANTHROPIC_SMALL_FAST_MODEL": "MiniMax-M2.7-highspeed",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "MiniMax-M2.7-highspeed",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "MiniMax-M2.7-highspeed",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "MiniMax-M2.7-highspeed",
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

### 5.2 或者通过环境变量（临时生效）

```bash
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
claude
```

---

## 6. 验证安装

运行以下命令确认所有组件就绪：

```bash
# 1. Claude Code 版本（需 >= 2.1.32）
claude --version

# 2. 配置文件完整性
cat ~/.claude/settings.json
cat ~/.claude.json

# 3. 启动 Claude Code
cd /你的项目目录
claude
```

启动后应该看到 Claude Code 的交互界面。首次启动时会提示信任当前文件夹（Trust This Folder），选择信任即可。

---

## 7. 使用 Agent Teams

### 7.1 两种协作模式对比

| 维度 | Subagents（子智能体） | Agent Teams（团队模式） |
|------|----------------------|------------------------|
| 开启方式 | 默认可用 | 需设置环境变量 |
| agent 关系 | 父子关系，只能向上汇报 | 平等关系，互相通信 |
| 上下文 | 共享父 agent 的上下文 | 各自独立上下文（各 1M token） |
| Token 消耗 | 较低 | 约 2 倍 |
| 适用场景 | 聚焦任务，只需返回结果 | 复杂协作，需要讨论和协调 |

### 7.2 启动 Agent Team

进入项目目录后启动 Claude Code：

```bash
cd /你的项目目录
claude
```

然后用自然语言描述你需要的团队：

```
创建一个 agent team：
- 一个 teammate 负责后端 API 开发
- 一个 teammate 负责前端 UI 实现
- 一个 teammate 负责编写测试用例
并行完成用户认证模块的开发。
```

Claude 会自动：
1. 创建团队并分配角色
2. 生成共享任务列表
3. 启动各 teammate 并行工作
4. 协调和汇总结果

### 7.3 与 Teammate 交互

| 操作 | 快捷键/命令 |
|------|------------|
| 切换到下一个 teammate | `Shift+Down` |
| 查看共享任务列表 | `Ctrl+T` |
| 给当前 teammate 发消息 | 直接输入文字后回车 |
| 要求 teammate 关闭 | 告诉 lead："请关闭 xxx teammate" |
| 清理整个团队 | 告诉 lead："Clean up the team" |

### 7.4 显示模式

```json
// ~/.claude/settings.json 中添加
{
  "teammateMode": "in-process"
}
```

| 模式 | 说明 | 要求 |
|------|------|------|
| `in-process` | 所有 teammate 在同一终端内运行 | 无额外依赖 |
| `tmux` | 每个 teammate 独立窗格 | 需安装 tmux |
| `auto`（默认） | 有 tmux 就分窗格，否则 in-process | — |

服务器环境推荐 `in-process` 模式。

### 7.5 使用示例

#### 并行代码审查

```
创建一个 agent team 审查 PR #42，3 个审查员分别关注：
- 安全性问题
- 性能影响
- 测试覆盖率
各自审查完后汇总发现。
```

#### 竞争假设调试

```
用户报告应用在发送一条消息后就退出了。
创建 5 个 teammate 分别调查不同的假设。
让它们互相讨论、反驳，像科学辩论一样，
最终把共识写入调查报告。
```

#### 以特定 agent 身份运行整个会话

```bash
# 使用项目中定义的 team-lead agent 启动
claude --agent team-lead
```

---

## 8. 自定义 Subagent 定义

### 8.1 文件位置

| 位置 | 作用域 | 优先级 |
|------|--------|--------|
| `.claude/agents/` | 当前项目 | 高 |
| `~/.claude/agents/` | 所有项目 | 低 |

### 8.2 文件格式

每个 agent 是一个 Markdown 文件，包含 YAML 头部 + 系统提示词：

```markdown
---
name: code-reviewer
description: 代码审查专家，负责审查代码质量和安全性
tools: Read, Grep, Glob, Bash
model: sonnet
memory: project
---

你是一个高级代码审查工程师。

审查清单：
- 代码可读性
- 安全漏洞
- 性能问题
- 测试覆盖

按优先级分类反馈：
- 严重问题（必须修复）
- 警告（建议修复）
- 建议（可考虑改进）
```

### 8.3 YAML 头部字段说明

| 字段 | 必须 | 说明 |
|------|------|------|
| `name` | 是 | 唯一标识符，小写字母和连字符 |
| `description` | 是 | 描述何时应该使用该 agent |
| `tools` | 否 | 可用工具列表（默认继承所有） |
| `model` | 否 | 使用的模型：`sonnet`、`opus`、`haiku`、`inherit` |
| `skills` | 否 | 预加载的 skill 列表 |
| `memory` | 否 | 持久记忆范围：`user`、`project`、`local` |
| `permissionMode` | 否 | 权限模式：`default`、`acceptEdits`、`bypassPermissions` |
| `maxTurns` | 否 | 最大对话轮次 |

### 8.4 通过命令行交互式创建

在 Claude Code 中运行：

```
/agents
```

选择 "Create new agent"，按提示完成创建。

---

## 9. 常见问题

### Q1: `claude: command not found`

**原因**：PATH 中没有 `/usr/local/bin`，或者在 Docker 容器内外环境不同。

**解决**：

```bash
# 确认 claude 的安装位置
ls -la /usr/local/bin/claude

# 如果存在但找不到命令，添加 PATH
export PATH="/usr/local/bin:$PATH"

# 写入 bashrc 持久生效
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc
```

### Q2: 官方安装脚本返回 "App unavailable in region"

**原因**：中国大陆 IP 无法访问 `claude.ai`。

**解决**：使用 npm 安装（本指南的方式）：

```bash
npm install -g @anthropic-ai/claude-code
```

### Q3: Docker 容器内安装后，宿主机找不到命令（反之亦然）

**原因**：Docker 容器和宿主机是隔离的环境，安装只在当前环境生效。

**解决**：在你实际使用 Claude Code 的环境中执行安装。如果需要同时在两边用，两边都要装。

### Q4: Agent Teams 不生效

**检查清单**：

```bash
# 1. 版本是否满足（>= 2.1.32）
claude --version

# 2. 配置是否正确
cat ~/.claude/settings.json | grep AGENT_TEAMS
# 应输出："CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"

# 3. 环境变量是否被覆盖
echo $CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS
```

### Q5: 第三方模型（MiniMax）兼容性

Agent Teams 内部依赖特定的工具调用格式（如 `TeamCreate`、`SendMessage`、`TaskList`）。第三方模型可能不完全支持这些内部工具。如果遇到问题：

- 先用 Subagents 模式测试基本功能
- Agent Teams 如果不工作，考虑切换到官方 Anthropic API

### Q6: 如何查看已安装的 agents

```bash
# 命令行查看
claude agents

# 或在 Claude Code 交互界面中
/agents
```

---

## 附录：一键安装脚本

将以下内容保存为 `setup-claude-code.sh`，修改前三行配置后执行：

```bash
#!/bin/bash
set -e

# ============================================================
# 修改以下三行配置（必填）
# ============================================================
API_KEY="替换为你的API_Key"
BASE_URL="https://api.minimaxi.com/anthropic"    # 见下方厂商列表
MODEL="MiniMax-M2.5"                              # 见下方厂商列表

# ============================================================
# 厂商参考（取消注释你选择的厂商即可）：
# ============================================================
# --- MiniMax ---
# BASE_URL="https://api.minimaxi.com/anthropic"
# MODEL="MiniMax-M2.5"            # 或 MiniMax-M2.5-Lightning

# --- DeepSeek ---
# BASE_URL="https://api.deepseek.com/anthropic"
# MODEL="deepseek-chat"           # (V3.2) 或 deepseek-reasoner

# --- 智谱 GLM ---
# BASE_URL="https://open.bigmodel.cn/api/anthropic"
# MODEL="glm-5"                   # 或 glm-4.7 / glm-4.5-air

# --- 通义千问 ---
# BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/anthropic"
# MODEL="qwen3-coder-plus"        # 480B MoE 编程专项

# --- Kimi ---
# BASE_URL="https://api.moonshot.cn/anthropic"
# MODEL="kimi-k2.5"               # 1T MoE 最新旗舰

# --- 豆包 ---
# BASE_URL="https://ark.cn-beijing.volces.com/api/compatible"
# MODEL="doubao-seed-code-preview-latest"  # 或 doubao-seed-1.6

# --- 腾讯混元 ---
# BASE_URL="https://api.hunyuan.cloud.tencent.com/anthropic"
# MODEL="hunyuan-2.0-thinking-20251109"    # HY 3.0 预计 2026.04
# ============================================================

echo "=== 检测系统架构 ==="
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    NODE_URL="https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-arm64.tar.xz"
elif [ "$ARCH" = "x86_64" ]; then
    NODE_URL="https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-x64.tar.xz"
else
    echo "不支持的架构: $ARCH"
    exit 1
fi

echo "=== 安装 Node.js 22 ($ARCH) ==="
if ! command -v node &>/dev/null || [[ $(node -v | cut -d. -f1 | tr -d v) -lt 18 ]]; then
    curl -fsSL "$NODE_URL" -o /tmp/node-v22.tar.xz
    cd /tmp && tar -xf node-v22.tar.xz
    cp -r node-v22.14.0-linux-*/bin /usr/local/
    cp -r node-v22.14.0-linux-*/include /usr/local/
    cp -r node-v22.14.0-linux-*/lib /usr/local/
    cp -r node-v22.14.0-linux-*/share /usr/local/
    rm -rf /tmp/node-v22*
    echo "Node.js $(node --version) 安装完成"
else
    echo "Node.js $(node --version) 已满足要求，跳过"
fi

echo "=== 安装 Claude Code ==="
npm install -g @anthropic-ai/claude-code
echo "Claude Code $(claude --version) 安装完成"

echo "=== 写入配置 ==="
mkdir -p ~/.claude

cat > ~/.claude/settings.json << EOF
{
  "env": {
    "ANTHROPIC_BASE_URL": "$BASE_URL",
    "ANTHROPIC_AUTH_TOKEN": "$API_KEY",
    "API_TIMEOUT_MS": "3000000",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "ANTHROPIC_MODEL": "$MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL": "$MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "$MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "$MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "$MODEL",
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
EOF

cat > ~/.claude.json << 'EOF'
{
  "hasCompletedOnboarding": true
}
EOF

echo ""
echo "=== 安装完成 ==="
echo "Claude Code: $(claude --version)"
echo "Node.js:     $(node --version)"
echo "Base URL:    $BASE_URL"
echo "模型:        $MODEL"
echo "Agent Teams: 已开启"
echo ""
echo "使用方式："
echo "  cd /你的项目目录"
echo "  claude"
```

使用方式：

```bash
chmod +x setup-claude-code.sh
# 编辑脚本前三行，填入你的 API Key、Base URL、模型名
./setup-claude-code.sh
```

---

> 本指南基于 Claude Code 2.1.79 + MiniMax API + openEuler/Ubuntu aarch64 环境验证通过。
> 
> 最后更新：2026-03-19
