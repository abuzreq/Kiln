@echo off
REM Kiln, reachable from the other machines on this network (the lab).
REM
REM Same launcher as kiln.bat -- first run installs everything, later runs just
REM start -- with --lan added, which binds every network interface instead of
REM this machine only. The window still opens here; the address to use from
REM another machine is printed in this console as "On this network: ...".
REM
REM Anything you add is passed through, e.g.:
REM     start_lan.bat --no-window
REM     start_lan.bat --port 9000
setlocal

echo.
echo   Starting Kiln with network access enabled.
echo.
echo   Kiln has no password: anyone who can reach this machine can use it,
echo   including your datasets, models and files. Only do this on a network
echo   you trust.
echo.
echo   If another machine cannot connect, allow Python through Windows
echo   Defender Firewall for private networks -- the first run usually asks.
echo.

call "%~dp0kiln.bat" --lan %*

endlocal
