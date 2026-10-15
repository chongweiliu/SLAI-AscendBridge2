[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Model,

    [Parameter(Position = 1)]
    [string]$ApiKey,

    [Parameter(Position = 2)]
    [string]$BaseUrl,

    [string]$Provider = $env:CLAUDE_PROVIDER
)

# Claude Code Windows 原生一键安装脚本。
# 用法：powershell -ExecutionPolicy Bypass -File .\scripts\install_claude_code.ps1 [模型名称] [API_KEY] [BASE_URL]

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$NodeMinVersion = 22
$NodeInstallVersion = if ($env:NODE_INSTALL_VERSION) { $env:NODE_INSTALL_VERSION } else { "22" }
$ClaudePackage = "@anthropic-ai/claude-code"
$ApiTimeoutMs = "3000000"

function Write-Info([string]$Message) { Write-Host "[INFO] $Message" -ForegroundColor Cyan }
function Write-Success([string]$Message) { Write-Host "[OK] $Message" -ForegroundColor Green }
function Write-Failure([string]$Message) { Write-Host "[ERROR] $Message" -ForegroundColor Red }

$Providers = @(
    [pscustomobject]@{ Name = "智谱 AI (GLM)"; Key = "zhipu"; Model = "glm-4.6"; Url = "https://open.bigmodel.cn/api/anthropic" },
    [pscustomobject]@{ Name = "阿里百炼 (Qwen)"; Key = "bailian"; Model = "qwen3-coder-plus"; Url = "https://dashscope.aliyuncs.com/apps/anthropic" },
    [pscustomobject]@{ Name = "幻方 DeepSeek"; Key = "deepseek"; Model = "deepseek-v3"; Url = "https://api.deepseek.com/anthropic" },
    [pscustomobject]@{ Name = "火山方舟 (豆包)"; Key = "volcano"; Model = "doubao-seed-1-6-250615"; Url = "https://ark.cn-beijing.volces.com/api/coding" },
    [pscustomobject]@{ Name = "MiniMax (M1/abab)"; Key = "minimax"; Model = "MiniMax-M1"; Url = "https://api.minimaxi.com/anthropic" },
    [pscustomobject]@{ Name = "月之暗面 (Kimi)"; Key = "moonshot"; Model = "kimi-k2"; Url = "https://api.moonshot.cn/anthropic" },
    [pscustomobject]@{ Name = "阶跃星辰 (Step)"; Key = "stepfun"; Model = "step-2"; Url = "https://api.stepfun.com/step_plan" },
    [pscustomobject]@{ Name = "Anthropic 官方"; Key = "anthropic"; Model = "claude-sonnet-4-5"; Url = "https://api.anthropic.com" },
    [pscustomobject]@{ Name = "自定义 (手动填 URL)"; Key = "custom"; Model = ""; Url = "" }
)

function Get-ProviderUrl([string]$ProviderKey) {
    switch ($ProviderKey.ToLowerInvariant()) {
        { $_ -in @("anthropic", "claude") } { return "https://api.anthropic.com" }
        { $_ -in @("zhipu", "glm") } { return "https://open.bigmodel.cn/api/anthropic" }
        "deepseek" { return "https://api.deepseek.com/anthropic" }
        { $_ -in @("bailian", "dashscope", "qwen") } { return "https://dashscope.aliyuncs.com/apps/anthropic" }
        { $_ -in @("volcano", "ark", "doubao") } { return "https://ark.cn-beijing.volces.com/api/coding" }
        "minimax" { return "https://api.minimaxi.com/anthropic" }
        { $_ -in @("moonshot", "kimi") } { return "https://api.moonshot.cn/anthropic" }
        { $_ -in @("stepfun", "step") } { return "https://api.stepfun.com/step_plan" }
        default { return "" }
    }
}

function Get-ModelUrl([string]$ModelName) {
    $normalizedModel = $ModelName.ToLowerInvariant()
    if ($normalizedModel -match "^claude-") { return "https://api.anthropic.com" }
    if ($normalizedModel -match "^deepseek-") { return "https://api.deepseek.com/anthropic" }
    if ($normalizedModel -match "^(glm-|chatglm-)") { return "https://open.bigmodel.cn/api/anthropic" }
    if ($normalizedModel -match "^doubao-") { return "https://ark.cn-beijing.volces.com/api/coding" }
    if ($normalizedModel -match "^(qwen|qwq)") { return "https://dashscope.aliyuncs.com/apps/anthropic" }
    if ($normalizedModel -match "^(minimax-|abab)") { return "https://api.minimaxi.com/anthropic" }
    if ($normalizedModel -match "^(kimi-|moonshot-)") { return "https://api.moonshot.cn/anthropic" }
    if ($normalizedModel -match "^step-") { return "https://api.stepfun.com/step_plan" }
    return ""
}

function Read-RequiredValue([string]$Prompt, [string]$DefaultValue) {
    $value = Read-Host "$Prompt (默认 $DefaultValue)"
    if ([string]::IsNullOrWhiteSpace($value)) {
        if ([string]::IsNullOrWhiteSpace($DefaultValue)) { throw "$Prompt 不能为空" }
        return $DefaultValue
    }
    return $value
}

function Resolve-Configuration {
    if ([string]::IsNullOrWhiteSpace($script:Model) -and
        [string]::IsNullOrWhiteSpace($script:BaseUrl) -and
        [string]::IsNullOrWhiteSpace($script:Provider)) {
        Write-Host "请选择模型提供方:"
        for ($i = 0; $i -lt $Providers.Count; $i++) {
            Write-Host ("  [{0}] {1}" -f ($i + 1), $Providers[$i].Name)
        }
        Write-Host "  [0] 退出"
        $choiceText = Read-Host "请选择 [1-$($Providers.Count)]"
        $choice = 0
        if (-not [int]::TryParse($choiceText, [ref]$choice) -or $choice -lt 1 -or $choice -gt $Providers.Count) {
            throw "无效选择: $choiceText"
        }
        $selected = $Providers[$choice - 1]
        $script:Provider = $selected.Key
        $script:BaseUrl = if ($selected.Key -eq "custom") {
            Read-Host "请输入 API BASE_URL (如 http://192.168.1.100:8000/v1)"
        } else {
            $selected.Url
        }
        if ([string]::IsNullOrWhiteSpace($script:BaseUrl)) { throw "API 地址不能为空" }
        $script:Model = Read-RequiredValue "请输入模型名称" $selected.Model
        $script:ApiKey = Read-RequiredValue "请输入 API_KEY" "sk-local"
        return
    }

    if ([string]::IsNullOrWhiteSpace($script:Model)) {
        $script:Model = Read-RequiredValue "请输入模型名称" "deepseek-v4-pro"
    }
    if ([string]::IsNullOrWhiteSpace($script:ApiKey)) {
        $script:ApiKey = Read-RequiredValue "请输入 API_KEY" "sk-local"
    }
    if (-not [string]::IsNullOrWhiteSpace($script:BaseUrl)) { return }

    if (-not [string]::IsNullOrWhiteSpace($script:Provider)) {
        $script:BaseUrl = Get-ProviderUrl $script:Provider
        if ([string]::IsNullOrWhiteSpace($script:BaseUrl)) {
            throw "未知 CLAUDE_PROVIDER: $($script:Provider)"
        }
    } else {
        $script:BaseUrl = Get-ModelUrl $script:Model
    }
    if ([string]::IsNullOrWhiteSpace($script:BaseUrl)) {
        $script:BaseUrl = Read-Host "模型 '$($script:Model)' 未匹配已知平台，请输入 API BASE_URL"
        if ([string]::IsNullOrWhiteSpace($script:BaseUrl)) { throw "API 地址不能为空" }
    }
}

function Add-UserPath([string]$Directory) {
    $normalized = $Directory.TrimEnd("\")
    $currentEntries = @($env:Path -split ";" | Where-Object { $_ })
    if (-not ($currentEntries | Where-Object { $_.TrimEnd("\") -ieq $normalized })) {
        $env:Path = "$Directory;$($env:Path)"
    }

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $userEntries = @($userPath -split ";" | Where-Object { $_ })
    if (-not ($userEntries | Where-Object { $_.TrimEnd("\") -ieq $normalized })) {
        [Environment]::SetEnvironmentVariable("Path", ((@($Directory) + $userEntries) -join ";"), "User")
    }
}

function Get-LatestNodeVersion {
    if ($env:NODE_VERSION) {
        if ($env:NODE_VERSION.StartsWith("v")) { return $env:NODE_VERSION }
        return "v$($env:NODE_VERSION)"
    }

    foreach ($distBase in @("https://nodejs.org/dist", "https://npmmirror.com/mirrors/node")) {
        try {
            Write-Info "正在从 $distBase 查询 Node.js $NodeInstallVersion.x 最新版本..."
            $index = (Invoke-WebRequest -UseBasicParsing -Uri "$distBase/index.tab").Content
            $line = $index -split "`n" | Where-Object { $_ -match "^v$([regex]::Escape($NodeInstallVersion))\." } | Select-Object -First 1
            if ($line) { return ($line -split "`t")[0].Trim() }
        } catch {
            Write-Info "无法访问 $distBase，尝试下一个镜像。"
        }
    }
    throw "无法解析 Node.js $NodeInstallVersion.x 版本；可设置 NODE_VERSION=v$NodeInstallVersion.x.y 后重试。"
}

function Install-NodeJs {
    $nodeCommand = Get-Command node -ErrorAction SilentlyContinue
    if ($nodeCommand) {
        $currentVersion = (& node --version).Trim().TrimStart("v")
        $majorVersion = [int]($currentVersion.Split(".")[0])
        # PowerShell 可能优先解析到未签名的 npm.ps1；始终检查并调用 npm.cmd。
        if ($majorVersion -ge $NodeMinVersion -and (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
            Write-Success "Node.js 已安装: v$currentVersion"
            return
        }
        Write-Info "Node.js v$currentVersion 低于 v$NodeMinVersion，开始升级。"
    }

    $architecture = $env:PROCESSOR_ARCHITECTURE.ToUpperInvariant()
    $nodeArch = switch ($architecture) {
        "AMD64" { "win-x64" }
        "ARM64" { "win-arm64" }
        default { throw "不支持的 Windows 架构: $architecture" }
    }
    $nodeVersion = Get-LatestNodeVersion
    $archiveName = "node-$nodeVersion-$nodeArch.zip"
    $tempDir = Join-Path ([IO.Path]::GetTempPath()) ("claude-node-" + [guid]::NewGuid().ToString("N"))
    $archivePath = Join-Path $tempDir $archiveName
    $checksumsPath = Join-Path $tempDir "SHASUMS256.txt"
    New-Item -ItemType Directory -Path $tempDir | Out-Null

    try {
        $downloaded = $false
        foreach ($distBase in @("https://nodejs.org/dist", "https://npmmirror.com/mirrors/node")) {
            try {
                Write-Info "正在从 $distBase 下载 Node.js $nodeVersion ($nodeArch)..."
                Invoke-WebRequest -UseBasicParsing -Uri "$distBase/$nodeVersion/$archiveName" -OutFile $archivePath
                Invoke-WebRequest -UseBasicParsing -Uri "$distBase/$nodeVersion/SHASUMS256.txt" -OutFile $checksumsPath
                $downloaded = $true
                break
            } catch {
                Write-Info "从 $distBase 下载失败，尝试下一个镜像。"
            }
        }
        if (-not $downloaded) { throw "所有镜像均下载失败: $archiveName" }

        $checksumLine = Get-Content $checksumsPath | Where-Object { $_ -match "\s+$([regex]::Escape($archiveName))$" } | Select-Object -First 1
        if (-not $checksumLine) { throw "校验文件中找不到 $archiveName" }
        $expectedChecksum = ($checksumLine -split "\s+")[0].ToLowerInvariant()
        $actualChecksum = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant()
        if ($actualChecksum -ne $expectedChecksum) { throw "Node.js 安装包 SHA-256 校验失败" }

        $installRoot = Join-Path $env:LOCALAPPDATA "Programs\nodejs"
        $versionDir = Join-Path $installRoot "node-$nodeVersion-$nodeArch"
        New-Item -ItemType Directory -Force -Path $installRoot | Out-Null
        if (Test-Path $versionDir) { Remove-Item -LiteralPath $versionDir -Recurse -Force }
        Write-Info "正在安装 Node.js 到 $versionDir..."
        Expand-Archive -LiteralPath $archivePath -DestinationPath $installRoot -Force
        Add-UserPath $versionDir
    } finally {
        if (Test-Path $tempDir) { Remove-Item -LiteralPath $tempDir -Recurse -Force }
    }

    if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw "Node.js 安装完成，但 node 不在 PATH 中。" }
    if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) { throw "Node.js 安装完成，但 npm.cmd 不在 PATH 中。" }
    Write-Success "Node.js 已安装: $(& node --version)"
    $npmCommand = (Get-Command npm.cmd -ErrorAction Stop).Source
    Write-Success "npm 版本: $(& $npmCommand --version)"
}

function Install-ClaudeCode {
    $npmCommand = (Get-Command npm.cmd -ErrorAction Stop).Source
    $npmPrefix = Join-Path $env:LOCALAPPDATA "Programs\claude-code"
    New-Item -ItemType Directory -Force -Path $npmPrefix | Out-Null
    Add-UserPath $npmPrefix
    & $npmCommand config set prefix $npmPrefix

    if (Get-Command claude.cmd -ErrorAction SilentlyContinue) {
        Write-Info "正在更新 Claude Code..."
    } else {
        Write-Info "正在安装 Claude Code..."
    }
    & $npmCommand install -g "$ClaudePackage@latest"
    if ($LASTEXITCODE -ne 0) {
        Write-Info "默认 npm registry 安装失败，改用 npmmirror。"
        & $npmCommand install -g "$ClaudePackage@latest" --registry "https://registry.npmmirror.com"
    }
    if ($LASTEXITCODE -ne 0) { throw "Claude Code npm 安装失败" }
    $claudeCommand = (Get-Command claude.cmd -ErrorAction Stop).Source
    Write-Success "Claude Code 已就绪: $(& $claudeCommand --version)"
}

function Configure-ClaudeCode {
    $env:INSTALL_API_KEY = $script:ApiKey
    $env:INSTALL_PROVIDER_URL = $script:BaseUrl
    $env:INSTALL_MODEL = $script:Model
    $env:INSTALL_API_TIMEOUT_MS = $ApiTimeoutMs
    # 非 Anthropic 官方模型不在 Claude Code 内置模型目录里，它会把上下文窗口按 200k 兜底并在每次
    # 启动时告警。显式声明窗口即可消除告警；官方模型本身已知，不需要也不应该覆盖。
    $isKnownClaudeModel = ($script:Provider -in @("anthropic", "claude")) -or ($script:Model -match "^claude-")
    $env:INSTALL_MAX_CONTEXT_TOKENS = if ($isKnownClaudeModel) { "" } else { "1000000" }
    $configScriptPath = Join-Path ([IO.Path]::GetTempPath()) ("claude-config-" + [guid]::NewGuid().ToString("N") + ".js")
    try {
        # 先落盘成临时 .js 再执行：Windows PowerShell 5.1 调用原生 EXE 时会吞掉参数里的
        # 双引号，用 --eval 会把 ".claude.json" 传成 .claude.json 从而报语法错误。
        $configScript = @'
const os = require("os");
const fs = require("fs");
const path = require("path");
const homeDir = os.homedir();
const claudeJsonPath = path.join(homeDir, ".claude.json");
const claudeJson = fs.existsSync(claudeJsonPath)
    ? JSON.parse(fs.readFileSync(claudeJsonPath, "utf8"))
    : {};
fs.writeFileSync(claudeJsonPath, JSON.stringify({ ...claudeJson, hasCompletedOnboarding: true }, null, 2));

const settingsPath = path.join(homeDir, ".claude", "settings.json");
fs.mkdirSync(path.dirname(settingsPath), { recursive: true });
const settings = fs.existsSync(settingsPath)
    ? JSON.parse(fs.readFileSync(settingsPath, "utf8"))
    : {};
const env = {
    ...settings.env,
    ANTHROPIC_AUTH_TOKEN: process.env.INSTALL_API_KEY,
    ANTHROPIC_BASE_URL: process.env.INSTALL_PROVIDER_URL,
    ANTHROPIC_MODEL: process.env.INSTALL_MODEL,
    ANTHROPIC_DEFAULT_HAIKU_MODEL: process.env.INSTALL_MODEL,
    ANTHROPIC_DEFAULT_SONNET_MODEL: process.env.INSTALL_MODEL,
    ANTHROPIC_DEFAULT_OPUS_MODEL: process.env.INSTALL_MODEL,
    API_TIMEOUT_MS: process.env.INSTALL_API_TIMEOUT_MS,
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: 1,
    CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS: 1
};
// 自定义/第三方模型：声明真实上下文窗口，避免 Claude Code 按 200k 兜底并告警。
if (process.env.INSTALL_MAX_CONTEXT_TOKENS) {
    env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = process.env.INSTALL_MAX_CONTEXT_TOKENS;
}
fs.writeFileSync(settingsPath, JSON.stringify({ ...settings, env }, null, 2));
'@
        [IO.File]::WriteAllText($configScriptPath, $configScript, (New-Object Text.UTF8Encoding($false)))
        & node $configScriptPath
        if ($LASTEXITCODE -ne 0) { throw "写入 Claude Code 配置失败" }
    } finally {
        Remove-Item -LiteralPath $configScriptPath -Force -ErrorAction SilentlyContinue
        Remove-Item Env:INSTALL_API_KEY -ErrorAction SilentlyContinue
        Remove-Item Env:INSTALL_PROVIDER_URL -ErrorAction SilentlyContinue
        Remove-Item Env:INSTALL_MODEL -ErrorAction SilentlyContinue
        Remove-Item Env:INSTALL_API_TIMEOUT_MS -ErrorAction SilentlyContinue
        Remove-Item Env:INSTALL_MAX_CONTEXT_TOKENS -ErrorAction SilentlyContinue
    }
    Write-Success "Claude Code 配置完成，Agent Teams 已启用。"
}

try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Resolve-Configuration
    Write-Host ""
    Write-Host "========================================"
    Write-Host "  模型:    $Model"
    if ($Provider) { Write-Host "  平台:    $Provider" }
    Write-Host "  API URL: $BaseUrl"
    Write-Host "========================================"
    Write-Host ""

    Install-NodeJs
    Install-ClaudeCode
    Configure-ClaudeCode
    Write-Host ""
    Write-Success "安装完成！重新打开终端后运行: claude"
    # 安装脚本是子进程，改不了"已经开着"的那个终端的环境变量，只能在这里给出可直接粘贴的刷新命令。
    Write-Info "注意：当前已打开的终端不会自动加载新的 PATH。重开一个全新的终端窗口，"
    Write-Info "      或先在当前窗口执行下面这行刷新，即可直接用 claude:"
    Write-Host '      $env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User")' -ForegroundColor Yellow
} catch {
    Write-Failure $_.Exception.Message
    exit 1
}
