[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Model,

    [Parameter(Position = 1)]
    [string]$ApiKey = $env:ZCODE_API_KEY,

    [Parameter(Position = 2)]
    [string]$BaseUrl,

    [string]$Provider = $env:ZCODE_PROVIDER,

    [ValidateSet("openai-completions", "openai-responses", "anthropic-messages")]
    [string]$ApiProtocol = $env:ZCODE_API_PROTOCOL
)

# ZCode 官方 CLI Windows 一键构建、安装与推理服务配置脚本。
# 用法：powershell -ExecutionPolicy Bypass -File .\scripts\install_zcode.ps1 [模型名称] [API_KEY] [BASE_URL]

if ($PSBoundParameters.ContainsKey("ApiKey") -and -not $ApiKey) { $ApiKey = "sk-local" }

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$NodeInstallVersion = if ($env:NODE_INSTALL_VERSION) { $env:NODE_INSTALL_VERSION } else { "24" }
$ZCodeCliRepository = $env:ZCODE_CLI_REPOSITORY
$ZCodeCliRef = if ($env:ZCODE_CLI_REF) { $env:ZCODE_CLI_REF } else { "main" }
$ZCodeCliSourceDir = if ($env:ZCODE_CLI_SOURCE_DIR) { $env:ZCODE_CLI_SOURCE_DIR } else { Join-Path $env:LOCALAPPDATA "zcode-cli\source" }
$ZCodeCliInstallDir = if ($env:ZCODE_CLI_INSTALL_DIR) { $env:ZCODE_CLI_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA "Programs\zcode-cli" }
$ZCodeCliCommand = if ($env:ZCODE_CLI_COMMAND) { $env:ZCODE_CLI_COMMAND } else { "zcode" }
$ZCodeCliPnpmVersion = if ($env:ZCODE_CLI_PNPM_VERSION) { $env:ZCODE_CLI_PNPM_VERSION } else { "10.33.2" }

function Write-Info([string]$Message) { Write-Host "[INFO] $Message" -ForegroundColor Cyan }
function Write-Success([string]$Message) { Write-Host "[OK] $Message" -ForegroundColor Green }
function Write-Failure([string]$Message) { Write-Host "[ERROR] $Message" -ForegroundColor Red }

$Providers = @(
    [pscustomobject]@{ Name = "智谱 AI (GLM)"; Key = "zhipu"; Model = "glm-4.6" },
    [pscustomobject]@{ Name = "阿里百炼 (Qwen)"; Key = "bailian"; Model = "qwen3-coder-plus" },
    [pscustomobject]@{ Name = "幻方 DeepSeek"; Key = "deepseek"; Model = "deepseek-v3" },
    [pscustomobject]@{ Name = "火山方舟 (豆包)"; Key = "volcano"; Model = "doubao-seed-1-6-250615" },
    [pscustomobject]@{ Name = "MiniMax (M1/abab)"; Key = "minimax"; Model = "MiniMax-M1" },
    [pscustomobject]@{ Name = "月之暗面 (Kimi)"; Key = "moonshot"; Model = "kimi-k2" },
    [pscustomobject]@{ Name = "阶跃星辰 (Step)"; Key = "stepfun"; Model = "step-2" },
    [pscustomobject]@{ Name = "Anthropic 官方"; Key = "anthropic"; Model = "claude-sonnet-4-5" },
    [pscustomobject]@{ Name = "自定义 (手动填 URL)"; Key = "custom"; Model = "" }
)

function Get-ProviderUrl([string]$ProviderKey) {
    $useOpenAi = $script:ApiProtocol -in @("openai-completions", "openai-responses")
    switch ($ProviderKey.ToLowerInvariant()) {
        { $_ -in @("anthropic", "claude") } { return "https://api.anthropic.com" }
        { $_ -in @("zhipu", "glm") } { return $(if ($useOpenAi) { "https://open.bigmodel.cn/api/paas/v4" } else { "https://open.bigmodel.cn/api/anthropic" }) }
        "deepseek" { return $(if ($useOpenAi) { "https://api.deepseek.com/v1" } else { "https://api.deepseek.com/anthropic" }) }
        { $_ -in @("bailian", "dashscope", "qwen") } { return $(if ($useOpenAi) { "https://dashscope.aliyuncs.com/compatible-mode/v1" } else { "https://dashscope.aliyuncs.com/apps/anthropic" }) }
        { $_ -in @("volcano", "ark", "doubao") } { return $(if ($useOpenAi) { "https://ark.cn-beijing.volces.com/api/v3" } else { "https://ark.cn-beijing.volces.com/api/coding" }) }
        "minimax" { return $(if ($useOpenAi) { "https://api.minimaxi.com/v1" } else { "https://api.minimaxi.com/anthropic" }) }
        { $_ -in @("moonshot", "kimi") } { return $(if ($useOpenAi) { "https://api.moonshot.cn/v1" } else { "https://api.moonshot.cn/anthropic" }) }
        { $_ -in @("stepfun", "step") } { return $(if ($useOpenAi) { "https://api.stepfun.com/v1" } else { "https://api.stepfun.com/step_plan" }) }
        default { return "" }
    }
}

function Get-ModelUrl([string]$ModelName) {
    $normalizedModel = $ModelName.ToLowerInvariant()
    if ($normalizedModel -match "^claude-") { return (Get-ProviderUrl "anthropic") }
    if ($normalizedModel -match "^deepseek-") { return (Get-ProviderUrl "deepseek") }
    if ($normalizedModel -match "^(glm-|chatglm-)") { return (Get-ProviderUrl "zhipu") }
    if ($normalizedModel -match "^doubao-") { return (Get-ProviderUrl "volcano") }
    if ($normalizedModel -match "^(qwen|qwq)") { return (Get-ProviderUrl "bailian") }
    if ($normalizedModel -match "^(minimax-|abab)") { return (Get-ProviderUrl "minimax") }
    if ($normalizedModel -match "^(kimi-|moonshot-)") { return (Get-ProviderUrl "moonshot") }
    if ($normalizedModel -match "^step-") { return (Get-ProviderUrl "stepfun") }
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

function Read-ApiKey {
    $secret = Read-Host "请输入 API_KEY (默认 sk-local)" -AsSecureString
    $value = [Net.NetworkCredential]::new("", $secret).Password
    if ($value) { return $value }
    return "sk-local"
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
            Get-ProviderUrl $selected.Key
        }
        if ([string]::IsNullOrWhiteSpace($script:BaseUrl)) { throw "API 地址不能为空" }
        $script:Model = Read-RequiredValue "请输入模型名称" $selected.Model
        if (-not $script:ApiKey) { $script:ApiKey = Read-ApiKey }
        Resolve-ApiProtocol
        return
    }

    if ([string]::IsNullOrWhiteSpace($script:Model)) {
        $script:Model = Read-RequiredValue "请输入模型名称" "deepseek-v4-pro"
    }
    if ([string]::IsNullOrWhiteSpace($script:ApiKey)) {
        $script:ApiKey = Read-ApiKey
    }
    if ([string]::IsNullOrWhiteSpace($script:BaseUrl)) {
        if (-not [string]::IsNullOrWhiteSpace($script:Provider)) {
            $script:BaseUrl = Get-ProviderUrl $script:Provider
            if ([string]::IsNullOrWhiteSpace($script:BaseUrl)) {
                throw "未知 ZCODE_PROVIDER: $($script:Provider)"
            }
        } else {
            $script:BaseUrl = Get-ModelUrl $script:Model
        }
        if ([string]::IsNullOrWhiteSpace($script:BaseUrl)) {
            $script:BaseUrl = Read-Host "模型 '$($script:Model)' 未匹配已知平台，请输入 API BASE_URL"
            if ([string]::IsNullOrWhiteSpace($script:BaseUrl)) { throw "API 地址不能为空" }
        }
    }
    Resolve-ApiProtocol
}

function Resolve-ApiProtocol {
    if ([string]::IsNullOrWhiteSpace($script:ApiProtocol)) {
        $script:ApiProtocol = if ($script:BaseUrl -match "anthropic|^https://ark\.cn-beijing\.volces\.com/api/coding|^https://api\.stepfun\.com/step_plan") {
            "anthropic-messages"
        } else {
            "openai-completions"
        }
    }
    if ($script:ApiProtocol -notin @("openai-completions", "openai-responses", "anthropic-messages")) {
        throw "不支持的 API 协议: $script:ApiProtocol"
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
            $index = (Invoke-WebRequest -UseBasicParsing -TimeoutSec 60 -Uri "$distBase/index.tab").Content
            $line = $index -split "`n" | Where-Object { $_ -match "^v$([regex]::Escape($NodeInstallVersion))\." } | Select-Object -First 1
            if ($line) { return ($line -split "`t")[0].Trim() }
        } catch {
            Write-Info "无法访问 $distBase，尝试下一个镜像。"
        }
    }
    throw "无法解析 Node.js $NodeInstallVersion.x 版本；可设置 NODE_VERSION=v$NodeInstallVersion.x.y 后重试。"
}

function Install-NodeJs {
    $architecture = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
    $nodeArch = switch ($architecture) {
        "AMD64" { "win-x64" }
        "ARM64" { "win-arm64" }
        default { throw "不支持的 Windows 架构: $architecture" }
    }
    $installRoot = Join-Path $env:LOCALAPPDATA "zcode-cli\node"
    if (-not $env:NODE_VERSION) {
        $cached = Get-ChildItem -LiteralPath $installRoot -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match "^node-v$([regex]::Escape($NodeInstallVersion))\.\d+\.\d+-$nodeArch$" } |
            Sort-Object { [version]($_.Name -replace "^node-v|-$nodeArch$", "") } -Descending |
            Select-Object -First 1
        if ($cached -and (Test-Path (Join-Path $cached.FullName "node.exe")) -and (Test-Path (Join-Path $cached.FullName "npm.cmd"))) {
            $cachedVersion = [version]($cached.Name -replace "^node-v|-$nodeArch$", "")
            if ($cachedVersion -ge [version]"24.14.0") {
                $env:Path = "$($cached.FullName);$env:Path"
                Write-Info "使用缓存的官方 Node.js: $($cached.FullName)"
                return
            }
        }
    }
    $nodeVersion = Get-LatestNodeVersion
    $archiveName = "node-$nodeVersion-$nodeArch.zip"
    $tempDir = Join-Path ([IO.Path]::GetTempPath()) ("zcode-node-" + [guid]::NewGuid().ToString("N"))
    $archivePath = Join-Path $tempDir $archiveName
    $checksumsPath = Join-Path $tempDir "SHASUMS256.txt"
    New-Item -ItemType Directory -Path $tempDir | Out-Null

    try {
        $downloaded = $false
        foreach ($distBase in @("https://nodejs.org/dist", "https://npmmirror.com/mirrors/node")) {
            try {
                Write-Info "正在从 $distBase 下载 Node.js $nodeVersion ($nodeArch)..."
                Invoke-WebRequest -UseBasicParsing -TimeoutSec 60 -Uri "$distBase/$nodeVersion/$archiveName" -OutFile $archivePath
                Invoke-WebRequest -UseBasicParsing -TimeoutSec 60 -Uri "$distBase/$nodeVersion/SHASUMS256.txt" -OutFile $checksumsPath
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

        $versionDir = Join-Path $installRoot "node-$nodeVersion-$nodeArch"
        New-Item -ItemType Directory -Force -Path $installRoot | Out-Null
        if (Test-Path $versionDir) { Remove-Item -LiteralPath $versionDir -Recurse -Force }
        Write-Info "正在安装 Node.js 到 $versionDir..."
        Expand-Archive -LiteralPath $archivePath -DestinationPath $installRoot -Force
        $env:Path = "$versionDir;$env:Path"
    } finally {
        if (Test-Path $tempDir) { Remove-Item -LiteralPath $tempDir -Recurse -Force }
    }

    if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw "Node.js 安装完成，但 node 不在 PATH 中。" }
    if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) { throw "Node.js 安装完成，但 npm.cmd 不在 PATH 中。" }
    & node -e 'const [major, minor] = process.versions.node.split(".").map(Number); process.exit(major > 24 || major === 24 && minor >= 14 ? 0 : 1)'
    if ($LASTEXITCODE -ne 0) { throw "ZCode CLI 需要 Node.js 24.14 或更新版本" }
    Write-Success "Node.js 已安装: $(& node --version)"
    $npmCommand = (Get-Command npm.cmd -ErrorAction Stop).Source
    Write-Success "npm 版本: $(& $npmCommand --version)"
}

function Install-NpmPackages([string[]]$NpmArguments) {
    $npm = (Get-Command npm.cmd -ErrorAction Stop).Source
    & $npm install --no-audit --no-fund --fetch-retries=1 --fetch-timeout=60000 @NpmArguments
    if ($LASTEXITCODE -ne 0) {
        Write-Info "默认 npm 源安装失败，改用 npmmirror..."
        & $npm install --no-audit --no-fund --fetch-retries=1 --fetch-timeout=60000 --registry=https://registry.npmmirror.com @NpmArguments
    }
    if ($LASTEXITCODE -ne 0) { throw "npm 安装失败，退出码: $LASTEXITCODE" }
}

function Get-ZCodeSource {
    $gitCommand = (Get-Command git.exe -ErrorAction Stop).Source
    $gitOptions = @("-c", "http.lowSpeedLimit=512", "-c", "http.lowSpeedTime=20")
    $repositories = if ($ZCodeCliRepository) { @($ZCodeCliRepository) } else {
        @("https://gitcode.com/zai-org/ZCode.git", "https://github.com/zai-org/ZCode.git")
    }
    $oldPrompt = $env:GIT_TERMINAL_PROMPT
    $env:GIT_TERMINAL_PROMPT = "0"
    try {
        if (-not (Test-Path (Join-Path $ZCodeCliSourceDir ".git"))) {
            if (Test-Path $ZCodeCliSourceDir) { throw "$ZCodeCliSourceDir 已存在但不是 Git 仓库" }
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent ([IO.Path]::GetFullPath($ZCodeCliSourceDir))) | Out-Null
            foreach ($repo in $repositories) {
                Write-Info "从 $repo 获取 ZCode 源码 ($ZCodeCliRef)..."
                & $gitCommand @gitOptions clone --depth 1 --branch $ZCodeCliRef -- $repo $ZCodeCliSourceDir
                if ($LASTEXITCODE -eq 0) { return }
            }
        } else {
            $dirty = & $gitCommand -C $ZCodeCliSourceDir status --porcelain --untracked-files=no
            if ($LASTEXITCODE -ne 0 -or $dirty) { throw "无法使用 ZCode 源码缓存: $ZCodeCliSourceDir" }
            foreach ($repo in $repositories) {
                & $gitCommand @gitOptions -C $ZCodeCliSourceDir fetch --depth 1 -- $repo $ZCodeCliRef
                if ($LASTEXITCODE -ne 0) { continue }
                & $gitCommand -C $ZCodeCliSourceDir checkout --detach FETCH_HEAD
                if ($LASTEXITCODE -ne 0) { throw "切换 ZCode 版本失败" }
                return
            }
        }
    } finally {
        $env:GIT_TERMINAL_PROMPT = $oldPrompt
    }
    throw "无法获取 ZCode 源码。可设置 ZCODE_CLI_REPOSITORY 或 ZCODE_CLI_BINARY 后重试。"
}

function Build-ZCodeCli([string]$Target) {
    Get-ZCodeSource
    $pnpmRoot = Join-Path $env:LOCALAPPDATA "zcode-cli\build-tools"
    Install-NpmPackages @("--prefix", $pnpmRoot, "pnpm@$ZCodeCliPnpmVersion")
    $pnpm = Join-Path $pnpmRoot "node_modules\.bin\pnpm.cmd"
    $oldCorepack = $env:COREPACK_ENABLE_PROJECT_SPEC
    $oldPath = $env:Path
    $env:Path = "$(Split-Path -Parent $pnpm);$env:Path"
    $env:COREPACK_ENABLE_PROJECT_SPEC = "0"
    Push-Location -LiteralPath $ZCodeCliSourceDir
    try {
        Write-Info "正在安装 ZCode 构建依赖，初次构建可能需要数分钟..."
        $pnpmOptions = @("--ignore-scripts", "--frozen-lockfile", "--fetch-retries=1", "--fetch-timeout=60000")
        & $pnpm install @pnpmOptions
        if ($LASTEXITCODE -ne 0) { & $pnpm install @pnpmOptions --registry=https://registry.npmmirror.com }
        if ($LASTEXITCODE -ne 0) { throw "ZCode 依赖安装失败" }
        & $pnpm run build:sea:prepare
        if ($LASTEXITCODE -ne 0) { throw "ZCode CLI 构建失败" }
        $nodeBinary = & node -p 'process.execPath'
        & $pnpm --filter "@zcode/cli" sea --target $Target --node-binary "$Target=$nodeBinary"
        if ($LASTEXITCODE -ne 0) { throw "ZCode CLI 单文件打包失败" }
    } finally {
        Pop-Location
        $env:COREPACK_ENABLE_PROJECT_SPEC = $oldCorepack
        $env:Path = $oldPath
    }
}

function Install-ZCodeCli {
    $architecture = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
    $zcodeTarget = switch ($architecture) {
        "AMD64" { "win-x64" }
        "ARM64" { "win-arm64" }
        default { throw "不支持的 Windows 架构: $architecture" }
    }
    if ($ZCodeCliCommand -notmatch '^[A-Za-z0-9._-]+$' -or $ZCodeCliCommand -in @(".", "..")) {
        throw "ZCODE_CLI_COMMAND 只能包含字母、数字、点、下划线或连字符。"
    }
    $binaryPath = $env:ZCODE_CLI_BINARY
    if (-not $binaryPath) {
        Build-ZCodeCli $zcodeTarget
        $binaryPath = Join-Path $ZCodeCliSourceDir "apps\zcode-cli\packages\cli\dist\zcode-windows-$($zcodeTarget.Split('-')[1]).exe"
    }
    $binaryPath = (Resolve-Path -LiteralPath $binaryPath).Path
    & $binaryPath --version
    if ($LASTEXITCODE -ne 0) { throw "ZCode CLI 文件不可用: $binaryPath" }
    New-Item -ItemType Directory -Force -Path $ZCodeCliInstallDir | Out-Null
    $targetPath = Join-Path $ZCodeCliInstallDir "$ZCodeCliCommand.exe"
    if ([IO.Path]::GetFullPath($targetPath) -ine $binaryPath) {
        Copy-Item -LiteralPath $binaryPath -Destination $targetPath -Force
    }
    Add-UserPath ([IO.Path]::GetFullPath($ZCodeCliInstallDir))
    Write-Success "ZCode CLI 已就绪: $targetPath"
}

function Configure-ZCode {
    $routeId = if ($script:Provider) { $script:Provider.ToLowerInvariant() } else { "slai-custom" }
    $routeId = $routeId -replace "[^a-z0-9-]", "-"
    if ([string]::IsNullOrWhiteSpace($routeId)) { $routeId = "slai-custom" }
    $zcodeProtocol = switch ($script:ApiProtocol) {
        "openai-completions" { "openai-chat-completions" }
        "openai-responses" { "openai-responses" }
        "anthropic-messages" { "anthropic-messages" }
    }
    $configFile = if ($env:ZCODE_PERSONAL_PROVIDER_CONFIG_FILE) {
        $env:ZCODE_PERSONAL_PROVIDER_CONFIG_FILE
    } else {
        $dataRoot = if ($env:ZCODE_DATA_BASE_DIR) { $env:ZCODE_DATA_BASE_DIR } else { $HOME }
        Join-Path $dataRoot ".zcode\v2\provider_config.json"
    }
    $configFile = [IO.Path]::GetFullPath($configFile)
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $configFile) | Out-Null
    $env:INSTALL_API_KEY = $script:ApiKey
    $env:INSTALL_CONFIG_FILE = $configFile
    $env:INSTALL_ROUTE_ID = $routeId
    $env:INSTALL_BASE_URL = $script:BaseUrl
    $env:INSTALL_MODEL = $script:Model
    $env:INSTALL_PROTOCOL = $zcodeProtocol
    $configScriptPath = Join-Path ([IO.Path]::GetTempPath()) ("zcode-config-" + [guid]::NewGuid().ToString("N") + ".js")
    try {
        $configScript = @'
const fs = require("fs");
const path = require("path");
const filePath = process.env.INSTALL_CONFIG_FILE;
let doc = { schemaVersion: 1, config: {} };
if (fs.existsSync(filePath)) doc = JSON.parse(fs.readFileSync(filePath, "utf8").replace(/^\uFEFF/, ""));
doc.schemaVersion = 1;
doc.config ||= {};
doc.config.providerConfigRules ||= {};
doc.config.providerConfigRules.providerRules ||= [];
doc.config.modelConfigRules ||= { providerModelRules: [], manualProviderModelRules: [] };
const providerId = process.env.INSTALL_ROUTE_ID;
const rule = {
  providerId,
  providerName: providerId,
  config: {
    group: "standard-personal",
    access: { type: "api-key", apiKey: process.env.INSTALL_API_KEY },
    api: { type: process.env.INSTALL_PROTOCOL, baseUrl: process.env.INSTALL_BASE_URL },
    personalModelIds: [process.env.INSTALL_MODEL]
  }
};
const rules = doc.config.providerConfigRules.providerRules;
const index = rules.findIndex((item) => item && item.providerId === providerId);
if (index >= 0) rules[index] = rule; else rules.push(rule);
doc.config.defaultModelSelection = { providerId, modelId: process.env.INSTALL_MODEL };
const tmp = filePath + "." + require("node:crypto").randomUUID();
try {
  fs.writeFileSync(tmp, JSON.stringify(doc, null, 2) + "\n", { mode: 0o600, flag: "wx" });
  fs.renameSync(tmp, filePath);
} finally {
  fs.rmSync(tmp, { force: true });
}
'@
        [IO.File]::WriteAllText($configScriptPath, $configScript, (New-Object Text.UTF8Encoding($false)))
        & node $configScriptPath
        if ($LASTEXITCODE -ne 0) { throw "写入 ZCode 配置失败" }
    } finally {
        Remove-Item -LiteralPath $configScriptPath -Force -ErrorAction SilentlyContinue
        Remove-Item Env:INSTALL_API_KEY -ErrorAction SilentlyContinue
        Remove-Item Env:INSTALL_CONFIG_FILE -ErrorAction SilentlyContinue
        Remove-Item Env:INSTALL_ROUTE_ID -ErrorAction SilentlyContinue
        Remove-Item Env:INSTALL_BASE_URL -ErrorAction SilentlyContinue
        Remove-Item Env:INSTALL_MODEL -ErrorAction SilentlyContinue
        Remove-Item Env:INSTALL_PROTOCOL -ErrorAction SilentlyContinue
    }
    Write-Success "ZCode 配置完成: provider=$routeId model=$($script:Model) protocol=$zcodeProtocol"
}

try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Resolve-Configuration
    Write-Host ""
    Write-Host "========================================"
    Write-Host "  模型:    $Model"
    if ($Provider) { Write-Host "  平台:    $Provider" }
    Write-Host "  API URL: $BaseUrl"
    Write-Host "  协议:    $ApiProtocol"
    Write-Host "========================================"
    Write-Host ""

    $nodeVersion = if (Get-Command node -ErrorAction SilentlyContinue) { & node --version }
    if (-not $env:ZCODE_CLI_BINARY -or -not $nodeVersion -or [int]($nodeVersion.TrimStart("v").Split(".")[0]) -lt 16) {
        Install-NodeJs
    }
    Install-ZCodeCli
    Configure-ZCode
    Write-Host ""
    Write-Success "安装完成！重新打开终端后运行: $ZCodeCliCommand"
    # 安装脚本是子进程，改不了"已经开着"的那个终端的环境变量，只能在这里给出可直接粘贴的刷新命令。
    Write-Info "注意：当前已打开的终端不会自动加载新的 PATH。重开一个全新的终端窗口，"
    Write-Info "      或先在当前窗口执行下面这行刷新，即可直接用 $ZCodeCliCommand:"
    Write-Host '      $env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User")' -ForegroundColor Yellow
} catch {
    Write-Failure $_.Exception.Message
    exit 1
}
