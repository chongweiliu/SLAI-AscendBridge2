#!/bin/bash
# 安装或更新 Codex CLI（Linux、容器、macOS）。
# 用法: bash install_codex.sh [模型名称] [API_KEY] [BASE_URL]
# 无参数在终端中选择第三方提供方。配置使用 OpenAI Responses API。
set -euo pipefail

NODE_MIN_VERSION=22
NODE_INSTALL_VERSION="${NODE_INSTALL_VERSION:-22}"

log_info()    { echo "🔹 $*"; }
log_success() { echo "✅ $*"; }
log_error()   { echo "❌ $*" >&2; }

MODEL="${1:-${CODEX_MODEL:-}}"
API_KEY="${2-${CODEX_API_KEY:-}}"
BASE_URL="${3:-${CODEX_BASE_URL:-}}"
PROVIDER="${CODEX_PROVIDER:-}"

PROVIDERS=(
    "zhipu|智谱 GLM Coding Plan|glm-5.4|https://open.bigmodel.cn/api/v1"
    "bailian|阿里百炼 Token Plan|auto|https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    "deepseek|DeepSeek|deepseek-v4-pro|https://api.deepseek.com"
    "volcano|火山方舟 Coding Plan|ark-code-latest|https://ark.cn-beijing.volces.com/api/coding/v3"
    "minimax|MiniMax M Plan（中国）|MiniMax-M3.1-Flash|https://api.minimax.cn/v1"
    "moonshot|Kimi Code（中国）|kimi-for-coding|https://api.kimi.com/coding/v1"
    "stepfun|阶跃星辰 Step Plan|step-3.7-flash|https://api.stepfun.com/step_plan/v1"
    "custom|自定义 Responses 网关||"
)

select_provider_menu() {
    local row key name model url choice index=1
    echo "请选择模型提供方:"
    for row in "${PROVIDERS[@]}"; do
        IFS='|' read -r key name model url <<< "$row"
        printf '  [%s] %s\n' "$index" "$name"
        index=$((index+1))
    done
    echo "  [0] 退出"
    IFS= read -r -p "请选择 [1-${#PROVIDERS[@]}]: " choice
    [ "$choice" != 0 ] || exit 0
    if ! [[ "$choice" =~ ^[1-9]$ ]] || [ "$choice" -gt "${#PROVIDERS[@]}" ]; then
        log_error "无效选择: $choice"
        exit 1
    fi
    local selected="${PROVIDERS[$((choice-1))]}"
    PROVIDER="${selected%%|*}"
}

provider_for_model() {
    local normalized
    normalized=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
    case "$normalized" in
        glm-*|chatglm-*) echo zhipu ;;
        qwen*|qwq*) echo bailian ;;
        deepseek-*) echo deepseek ;;
        doubao-*|ark-code-*) echo volcano ;;
        minimax-*) echo minimax ;;
        kimi-*|k3|k3-*|k3.*|moonshot-*) echo moonshot ;;
        step-*) echo stepfun ;;
        *) echo custom ;;
    esac
}

resolve_provider() {
    local row key name default_model default_url
    PROVIDER=$(printf '%s' "$PROVIDER" | tr '[:upper:]' '[:lower:]')
    case "$PROVIDER" in
        glm) PROVIDER=zhipu ;;
        qwen|dashscope) PROVIDER=bailian ;;
        ark|doubao) PROVIDER=volcano ;;
        kimi) PROVIDER=moonshot ;;
        step) PROVIDER=stepfun ;;
    esac
    for row in "${PROVIDERS[@]}"; do
        IFS='|' read -r key name default_model default_url <<< "$row"
        if [ "$key" = "$PROVIDER" ]; then
            BASE_URL="$default_url"
            PROVIDER_NAME="$name"
            if [ -z "$MODEL" ] && [ -n "$default_model" ]; then
                if [ -t 0 ]; then
                    IFS= read -r -p "请输入模型名称 (默认 $default_model): " MODEL
                fi
                MODEL="${MODEL:-$default_model}"
            fi
            return 0
        fi
    done
    log_error "未知 CODEX_PROVIDER: $PROVIDER"
    return 1
}

if [ "$#" -gt 3 ]; then
    log_error "用法: bash install_codex.sh [模型名称] [API_KEY] [BASE_URL]"
    exit 1
fi

PROVIDER_NAME="自定义 Responses 网关"
CONFIGURE_PROVIDER=false
if [ -n "$MODEL$API_KEY$BASE_URL$PROVIDER" ] || [ -t 0 ]; then
    CONFIGURE_PROVIDER=true
    if [ -z "$BASE_URL" ]; then
        if [ -z "$PROVIDER" ]; then
            if [ -z "$MODEL" ] && [ -t 0 ]; then
                select_provider_menu
            else
                PROVIDER=$(provider_for_model "$MODEL")
            fi
        fi
        resolve_provider
        if [ -z "$BASE_URL" ] && [ -t 0 ]; then
            IFS= read -r -p "请输入第三方 OpenAI Responses BASE_URL: " BASE_URL
        fi
    fi

    BASE_URL="${BASE_URL%/}"
    case "$BASE_URL" in
        */chat/completions|*/messages)
            log_error "请填写支持 OpenAI Responses API 的 BASE_URL"
            exit 1 ;;
        http://?*|https://?*) BASE_URL="${BASE_URL%/responses}" ;;
        *) log_error "请通过第三个参数或 CODEX_BASE_URL 指定第三方 HTTP(S) 地址"; exit 1 ;;
    esac
    if [ -z "$MODEL" ] && [ -t 0 ]; then
        IFS= read -r -p "请输入模型名称: " MODEL
    fi
    [ -n "$MODEL" ] || { log_error "请指定模型名称"; exit 1; }
    if [ "$#" -ge 2 ] && [ -z "$API_KEY" ]; then
        API_KEY=sk-local
    fi
    if [ -z "$API_KEY" ] && [ -t 0 ]; then
        IFS= read -r -p "请输入 API_KEY（输入可见）: " API_KEY
    fi
    [ -n "$API_KEY" ] || { log_error "请传入 API_KEY 或设置 CODEX_API_KEY"; exit 1; }
    log_info "提供方: $PROVIDER_NAME | 模型: $MODEL | BASE_URL: $BASE_URL"
fi

check_nodejs() {
    if command -v node &>/dev/null && command -v npm &>/dev/null; then
        current_version=$(node -v | sed 's/v//')
        major_version=$(echo "$current_version" | cut -d. -f1)
        if [ "$major_version" -ge "$NODE_MIN_VERSION" ]; then
            log_success "Node.js is already installed: v$current_version"
            return 0
        fi
        log_info "Node.js v$current_version < $NODE_MIN_VERSION, upgrading..."
    fi

    log_info "Installing Node.js..."

    local node_os machine_arch node_arch node_full_ver node_tar node_archive checksums_file
    local dist_base index_data expected_checksum actual_checksum install_prefix version_dir
    case "$(uname -s)" in
        Linux) node_os=linux ;;
        Darwin) node_os=darwin ;;
        *) log_error "不支持的操作系统: $(uname -s)"; return 1 ;;
    esac
    machine_arch=$(uname -m)
    case "$machine_arch" in
        aarch64|arm64) node_arch="$node_os-arm64" ;;
        x86_64|amd64) node_arch="$node_os-x64" ;;
        *) log_error "Unsupported architecture: $machine_arch"; return 1 ;;
    esac

    # 可用 NODE_VERSION=v22.x.y 锁定版本；未指定时解析 Node 22 的最新补丁版本。
    if [ -n "${NODE_VERSION:-}" ]; then
        node_full_ver="$NODE_VERSION"
        [[ "$node_full_ver" == v* ]] || node_full_ver="v${node_full_ver}"
    else
        index_data=""
        for dist_base in "https://nodejs.org/dist" "https://npmmirror.com/mirrors/node"; do
            log_info "Resolving latest Node.js ${NODE_INSTALL_VERSION}.x from ${dist_base}..."
            if index_data=$(curl --fail --show-error --silent --location --retry 1 --connect-timeout 10 --max-time 30 "${dist_base}/index.tab"); then
                # 使用 here-string，避免 awk 提前退出时上游 printf 收到 SIGPIPE；
                # 在 set -euo pipefail 下，该 SIGPIPE 会让脚本以 141 静默退出。
                node_full_ver=$(awk -v prefix="v${NODE_INSTALL_VERSION}." 'NR > 1 && index($1, prefix) == 1 { print $1; exit }' <<< "$index_data")
                [ -n "$node_full_ver" ] && break
            fi
        done
        if [ -z "${node_full_ver:-}" ]; then
            log_error "Unable to resolve the latest Node.js ${NODE_INSTALL_VERSION}.x version. Set NODE_VERSION=v${NODE_INSTALL_VERSION}.x.y and retry."
            return 1
        fi
    fi

    # 使用 gzip 包，避免 minimal 容器缺少 xz 时静默退出。
    node_tar="node-${node_full_ver}-${node_arch}.tar.gz"
    node_archive=$(mktemp "/tmp/${node_tar}.XXXXXX")
    checksums_file=$(mktemp "/tmp/node-shasums.XXXXXX")

    dist_base=""
    local candidate_base
    for candidate_base in "https://nodejs.org/dist" "https://npmmirror.com/mirrors/node"; do
        log_info "Downloading Node.js ${node_full_ver} (${node_arch}) from ${candidate_base}..."
        if curl --fail --show-error --location --retry 1 --connect-timeout 10 --max-time 180 \
            "${candidate_base}/${node_full_ver}/${node_tar}" -o "$node_archive" &&
            curl --fail --show-error --silent --location --retry 1 --connect-timeout 10 --max-time 30 \
                "${candidate_base}/${node_full_ver}/SHASUMS256.txt" -o "$checksums_file"; then
            dist_base="$candidate_base"
            break
        fi
        log_error "Download failed from ${candidate_base}; trying the next mirror."
    done
    if [ -z "$dist_base" ]; then
        rm -f "$node_archive" "$checksums_file"
        log_error "Failed to download ${node_tar} from all configured mirrors."
        return 1
    fi

    log_info "Verifying Node.js archive checksum..."
    expected_checksum=$(awk -v filename="$node_tar" '$2 == filename { print $1; exit }' "$checksums_file")
    if command -v sha256sum >/dev/null 2>&1; then
        actual_checksum=$(sha256sum "$node_archive" | awk '{ print $1 }')
    elif command -v shasum >/dev/null 2>&1; then
        actual_checksum=$(shasum -a 256 "$node_archive" | awk '{ print $1 }')
    else
        rm -f "$node_archive" "$checksums_file"
        log_error "Neither sha256sum nor shasum is available; refusing to install an unverified archive."
        return 1
    fi
    if [ -z "$expected_checksum" ] || [ "$actual_checksum" != "$expected_checksum" ]; then
        rm -f "$node_archive" "$checksums_file"
        log_error "Checksum verification failed for ${node_tar}."
        return 1
    fi
    rm -f "$checksums_file"

    # root 或 /usr/local 可写时使用系统目录，否则完整安装到当前用户目录。
    if [ "$(id -u)" -eq 0 ] || [ -w /usr/local ]; then
        install_prefix="/usr/local"
        mkdir -p "$install_prefix"
        log_info "Installing Node.js into ${install_prefix}..."
        tar -xzf "$node_archive" -C "$install_prefix" --strip-components=1
        export PATH="/usr/local/bin:$PATH"
    else
        install_prefix="$HOME/.local"
        version_dir="${install_prefix}/lib/nodejs/node-${node_full_ver}-${node_arch}"
        log_info "/usr/local is not writable; installing Node.js into ${version_dir}..."
        mkdir -p "$version_dir" "${install_prefix}/bin"
        tar -xzf "$node_archive" -C "$version_dir" --strip-components=1
        local executable
        for executable in node npm npx corepack; do
            if [ -e "${version_dir}/bin/${executable}" ]; then
                ln -sfn "${version_dir}/bin/${executable}" "${install_prefix}/bin/${executable}"
            fi
        done
        export PATH="${install_prefix}/bin:$PATH"
        touch "$HOME/.bashrc"
        grep -Fq 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" || \
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    fi
    rm -f "$node_archive"

    command -v node >/dev/null 2>&1 || { log_error "Node.js installation completed but node is not on PATH."; return 1; }
    command -v npm >/dev/null 2>&1 || { log_error "Node.js installation completed but npm is not on PATH."; return 1; }
    node -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 22 ? 0 : 1)' ||
        { log_error "需要 Node.js 22 或更新版本"; return 1; }
    log_success "Node.js installed: $(node -v)"
    log_success "npm version: $(npm -v)"
}

npm_install() {
    if ! npm install --no-audit --no-fund --fetch-retries=1 --fetch-timeout=60000 "$@"; then
        log_info "默认 npm 源安装失败，改用 npmmirror..."
        npm install --no-audit --no-fund --fetch-retries=1 --fetch-timeout=60000 \
            --registry=https://registry.npmmirror.com "$@"
    fi
}

add_bin_to_path() {
    local bin_dir=$1 rc path_line
    export PATH="$bin_dir:$PATH"
    path_line="export PATH=$(printf '%q' "$bin_dir"):\$PATH"
    for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
        if ! grep -Fxq "$path_line" "$rc" 2>/dev/null; then
            printf '\n%s\n' "$path_line" >> "$rc"
        fi
    done
}

check_nodejs
if [ "$(id -u)" -eq 0 ]; then
    npm_prefix=/usr/local
else
    npm_prefix="$HOME/.local"
fi
npm_install --global --prefix "$npm_prefix" --include=optional @openai/codex@latest
"$npm_prefix/bin/codex" --version
add_bin_to_path "$npm_prefix/bin"

if $CONFIGURE_PROVIDER; then
    INSTALL_MODEL="$MODEL" INSTALL_BASE_URL="$BASE_URL" INSTALL_API_KEY="$API_KEY" \
    INSTALL_PROVIDER_NAME="$PROVIDER_NAME" node <<'NODE'
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const home = process.env.CODEX_HOME || path.join(os.homedir(), ".codex");
const profile = path.join(home, "slai-ascendbridge.config.toml");
const marker = "# Managed by SLAI-AscendBridge2 install_codex";
const tomlString = (value) => JSON.stringify(value).replace(/\x7f/g, "\\u007f");
const lines = [
  marker,
  "model = " + tomlString(process.env.INSTALL_MODEL),
  'model_provider = "slai-ascendbridge"',
  'forced_login_method = "api"',
  "",
  "[model_providers.slai-ascendbridge]",
  "name = " + tomlString(process.env.INSTALL_PROVIDER_NAME),
  "base_url = " + tomlString(process.env.INSTALL_BASE_URL),
  'wire_api = "responses"',
  "experimental_bearer_token = " + tomlString(process.env.INSTALL_API_KEY),
  "",
];
fs.mkdirSync(home, { recursive: true, mode: 0o700 });
if (fs.existsSync(profile) && !fs.readFileSync(profile, "utf8").startsWith(marker + "\n")) {
  throw Error("已有同名 Codex 配置，请先将其改名: " + profile);
}
const tmp = profile + "." + require("node:crypto").randomUUID();
try {
  fs.writeFileSync(tmp, lines.join("\n"), { mode: 0o600, flag: "wx" });
  fs.renameSync(tmp, profile);
} finally {
  fs.rmSync(tmp, { force: true });
}
NODE
    log_success "安装完成。重新打开终端后运行: codex --profile slai-ascendbridge"
else
    log_success "安装完成。重新打开终端后运行: codex"
fi
