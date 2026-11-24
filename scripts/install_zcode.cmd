@echo off
setlocal
set "SLAI_INSTALLER_PS1=%~dp0install_zcode.ps1"
set "SLAI_INSTALLER_MODEL=%~1"
set "SLAI_INSTALLER_API_KEY=%~2"
set "SLAI_INSTALLER_BASE_URL=%~3"

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "$params = @{}; if ($env:SLAI_INSTALLER_MODEL) { $params.Model = $env:SLAI_INSTALLER_MODEL }; if ($env:SLAI_INSTALLER_API_KEY -or $env:SLAI_INSTALLER_BASE_URL) { $params.ApiKey = [string]$env:SLAI_INSTALLER_API_KEY }; if ($env:SLAI_INSTALLER_BASE_URL) { $params.BaseUrl = $env:SLAI_INSTALLER_BASE_URL }; . ([ScriptBlock]::Create((Get-Content -Raw -LiteralPath $env:SLAI_INSTALLER_PS1))) @params"
exit /b %errorlevel%
