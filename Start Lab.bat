@echo off
REM ============================================================
REM PowerGuard - Start Jupyter Lab (NO TOKEN)
REM ============================================================

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo.
echo ============================================================
echo  PowerGuard - Jupyter Lab
echo ============================================================
echo.

REM Load .env
if exist ".env" (
    for /f "usebackq tokens=1,2 delims==" %%a in (".env") do (
        set "line=%%a"
        if not "!line:~0,1!"=="#" (
            if not "%%a"=="" (
                set "%%a=%%b"
            )
        )
    )
)

if not defined BASE_PORT set "BASE_PORT=8000"
set /a JUPYTER_PORT=%BASE_PORT% + 0

echo Configuration:
echo   JUPYTER_PORT = %JUPYTER_PORT%
echo.

echo [1/3] Checking Jupyter...
jupyter --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Jupyter is not installed!
    pause
    exit /b 1
)
echo    OK
echo.

echo [2/3] Opening Chrome...
start "" chrome "http://localhost:%JUPYTER_PORT%/lab"
timeout /t 2 /nobreak >nul

echo [3/3] Starting Jupyter Lab (no auth)...
echo   URL: http://localhost:%JUPYTER_PORT%/lab
echo   Press Ctrl+C to stop.
echo.

set JUPYTER_TOKEN_DISABLED=1

jupyter lab ^
    --port=%JUPYTER_PORT% ^
    --ServerApp.open_browser=False ^
    --LabApp.open_browser=False ^
    --ServerApp.token="" ^
    --ServerApp.password="" ^
    --ServerApp.allow_origin="*" ^
    --notebook-dir=%cd%\notebooks

endlocal