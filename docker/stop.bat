@echo off
REM ============================================================
REM PowerGuard - Stop Docker Stack
REM ============================================================

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo.
echo ============================================================
echo  PowerGuard - Stopping Stack
echo ============================================================
echo.

REM ============================================================
REM Auto-detect project name (same logic as start.bat)
REM ============================================================
for %%I in ("%~dp0..") do set "ROOT_DIR=%%~fI"
for %%I in ("!ROOT_DIR!") do set "PROJECT_NAME=%%~nxI"
for /f "usebackq delims=" %%a in (`powershell -NoProfile -Command "'!PROJECT_NAME!'.ToLower()"`) do set "PROJECT_NAME_LOWER=%%a"

REM Load .env from parent
if exist "..\.env" (
    for /f "usebackq tokens=1,2 delims==" %%a in ("..\.env") do (
        set "line=%%a"
        if not "!line:~0,1!"=="#" (
            if not "%%a"=="" (
                set "%%a=%%b"
            )
        )
    )
)

if not defined COMPOSE_PROJECT_NAME set "COMPOSE_PROJECT_NAME=!PROJECT_NAME_LOWER!"

echo Stopping containers for project: !COMPOSE_PROJECT_NAME!
echo.

docker compose -p !COMPOSE_PROJECT_NAME! down
if errorlevel 1 (
    echo    WARNING: Some containers may not have stopped properly.
)
echo.

echo Status:
docker compose -p !COMPOSE_PROJECT_NAME! ps
echo.

echo ============================================================
echo  Stack stopped
echo ============================================================
echo.

pause
endlocal