@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_claude_code.ps1" %*
exit /b %errorlevel%
