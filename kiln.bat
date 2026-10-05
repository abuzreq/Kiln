@echo off
REM Kiln launcher for Windows. First run installs everything; later runs just start.
setlocal
cd /d "%~dp0"

call :find_python
if not defined PYLAUNCH call :offer_python

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
goto :eof


REM --- Find a real Python -------------------------------------------------
REM The `py` launcher is the reliable one. A bare `python` may be the Microsoft
REM Store stub, which silently opens the Store instead of running anything.
REM A python.exe found on PATH is quoted: C:\Program Files\... has a space.
:find_python
set "PYLAUNCH="
where py >nul 2>nul
if %errorlevel%==0 (
    set "PYLAUNCH=py -3"
    goto :eof
)
for /f "delims=" %%P in ('where python 2^>nul') do (
    if not defined PYLAUNCH (
        echo %%P | find /i "WindowsApps" >nul
        if errorlevel 1 set PYLAUNCH="%%P"
    )
)
goto :eof


REM --- No Python: offer to install it with winget --------------------------
REM Windows 10 and 11 ship winget. The user scope needs no administrator. This
REM console's PATH predates the install, so the new launcher is found by path.
:offer_python
where winget >nul 2>nul
if errorlevel 1 goto :eof
echo.
echo   Kiln needs Python 3.10 or newer, and could not find it.
echo   It can install Python 3.12 for you now, from Microsoft's winget.
echo.
choice /c YN /m "  Install Python 3.12"
if errorlevel 2 goto :eof
winget install -e --id Python.Python.3.12 --scope user
set "LAUNCHER=%LOCALAPPDATA%\Programs\Python\Launcher\py.exe"
set "DIRECT=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if exist "%LAUNCHER%" (
    set PYLAUNCH="%LAUNCHER%" -3
) else if exist "%DIRECT%" (
    set PYLAUNCH="%DIRECT%"
)
goto :eof
