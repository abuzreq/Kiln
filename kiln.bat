@echo off
REM Kiln launcher for Windows. First run installs everything; later runs just start.
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYLAUNCH=py -3"
) else (
    set "PYLAUNCH=python"
)

%PYLAUNCH% install.py --launch %*
endlocal
