[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$Model,
    [Parameter(Position = 1)][string]$ApiKey,
    [Parameter(Position = 2)][string]$BaseUrl,
    [string]$Provider,
    [ValidateSet("openai-completions", "openai-responses", "anthropic-messages")]
    [string]$ApiProtocol
)

# 安装可直接从 Windows 终端启动的 ZCode CLI。
if (-not $env:ZCODE_CLI_COMMAND) { $env:ZCODE_CLI_COMMAND = "zcode-cli" }
& (Join-Path $PSScriptRoot "install_zcode.ps1") @PSBoundParameters
