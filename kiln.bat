@echo off
REM Kiln launcher for Windows. First run installs everything; later runs just start.
setlocal
cd /d "%~dp0"

REM --- Find a real Python -------------------------------------------------
REM The `py` launcher is the reliable one. A bare `python` may be the Microsoft
REM Store stub, which silently opens the Store instead of running anything.
set "PYLAUNCH="
where py >nul 2>nul
if %errorlevel%==0 (
    set "PYLAUNCH=py -3"
) else (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        if not defined PYLAUNCH (
            echo %%P | find /i "WindowsApps" >nul
            if errorlevel 1 set "PYLAUNCH=%%P"
        )
    )
)

if not defined PYLAUNCH (
    echo.
    echo   Kiln needs Python 3.10 or newer, and could not find it.
    echo.
    echo   Install it from https://www.python.org/downloads/
    echo   and tick "Add python.exe to PATH" in the installer.
    echo.
    echo   Note: the "python" that ships with the Microsoft Store is a stub
    echo   that cannot run Kiln. Use the python.org installer instead.
    echo.
    pause
    exit /b 1
)

%PYLAUNCH% install.py --launch %*
if errorlevel 1 (
    echo.
    echo   Kiln failed to start. The error above explains why.
    echo.
    echo   If it mentions a missing package, try a clean reinstall:
    echo       %PYLAUNCH% install.py --reinstall
    echo.
    pause
    exit /b 1
)

endlocal
