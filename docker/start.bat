@echo off
REM ============================================================
REM PowerGuard - Start Docker Stack
REM ============================================================

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo.
echo ============================================================
echo  PowerGuard - Starting Docker Stack
echo ============================================================
echo.

REM ============================================================
REM Auto-detect project name
REM ============================================================
for %%I in ("%~dp0..") do set "ROOT_DIR=%%~fI"
for %%I in ("!ROOT_DIR!") do set "PROJECT_NAME=%%~nxI"
for /f "usebackq delims=" %%a in (`powershell -NoProfile -Command "'!PROJECT_NAME!'.ToLower()"`) do set "PROJECT_NAME_LOWER=%%a"

REM ============================================================
REM Load .env from parent (root) — ако съществува
REM ============================================================
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

REM Default base port
if not defined BASE_PORT set "BASE_PORT=8000"

REM Compute ports
set /a JUPYTER_PORT=%BASE_PORT% + 0
set /a API_PORT=%BASE_PORT% + 1
set /a PROMETHEUS_PORT=%BASE_PORT% + 2
set /a GRAFANA_PORT=%BASE_PORT% + 3
set /a DASHBOARD_PORT=%BASE_PORT% + 4

REM Parse --no-cache
set "BUILD_ARGS="
if /i "%1"=="--no-cache" set "BUILD_ARGS=--no-cache"

echo Configuration:
echo   ROOT_DIR             = !ROOT_DIR!
echo   DETECTED_NAME        = !PROJECT_NAME!
echo   COMPOSE_PROJECT_NAME = !COMPOSE_PROJECT_NAME!   ^(auto^)
echo   BASE_PORT            = %BASE_PORT%
echo   JUPYTER_PORT         = %JUPYTER_PORT%       ^(Jupyter Lab^)
echo   API_PORT             = %API_PORT%       ^(FastAPI^)
echo   PROMETHEUS_PORT      = %PROMETHEUS_PORT%       ^(Prometheus^)
echo   GRAFANA_PORT         = %GRAFANA_PORT%       ^(Grafana^)
echo   DASHBOARD_PORT       = %DASHBOARD_PORT%       ^(Streamlit^)
if defined BUILD_ARGS echo   BUILD_ARGS           = %BUILD_ARGS%
echo.

REM ============================================================
REM [1/5] Check Docker
REM ============================================================
echo [1/5] Checking Docker...
docker --version >nul 2>&1
if errorlevel 1 (echo ERROR: Docker not installed! & pause & exit /b 1)
docker info >nul 2>&1
if errorlevel 1 (echo ERROR: Docker daemon not running! & pause & exit /b 1)
echo    OK: Docker is running
echo.

REM ============================================================
REM [2/5] Check image
REM ============================================================
echo [2/5] Checking image...
set "IMAGE_NAME=!COMPOSE_PROJECT_NAME!:latest"
docker images !IMAGE_NAME! --format "{{.Repository}}" 2>nul | findstr "!COMPOSE_PROJECT_NAME!" >nul 2>&1
if not errorlevel 1 (
    echo    OK: Image exists
    docker images !IMAGE_NAME! --format "      {{.Repository}}:{{.Tag}}  ({{.Size}})"
    echo.
) else (
    echo    Image NOT found. Building...
    docker build %BUILD_ARGS% -f Dockerfile -t !IMAGE_NAME! ..
    if errorlevel 1 (echo ERROR: Build failed! & pause & exit /b 1)
    echo    OK: Image built
    echo.
)

REM ============================================================
REM [3/5] Stop existing
REM ============================================================
echo [3/5] Stopping existing containers...
docker compose -p !COMPOSE_PROJECT_NAME! down 2>nul
echo    OK
echo.

REM ============================================================
REM [4/5] Start containers — EXPORT ENV VARS (БЕЗ кавички!)
REM ============================================================
echo [4/5] Starting containers...

REM Export: БЕЗ кавички, за да е environment variable за docker compose
set API_PORT=%API_PORT%
set PROMETHEUS_PORT=%PROMETHEUS_PORT%
set GRAFANA_PORT=%GRAFANA_PORT%
set DASHBOARD_PORT=%DASHBOARD_PORT%
set COMPOSE_PROJECT_NAME=%COMPOSE_PROJECT_NAME%
set BASE_PORT=%BASE_PORT%

docker compose -p !COMPOSE_PROJECT_NAME! up -d
if errorlevel 1 (echo ERROR: Failed to start! & pause & exit /b 1)
echo    OK: Containers started
echo.

REM ============================================================
REM [5/5] Wait for API
REM ============================================================
echo [5/5] Waiting for API...
set /a attempt=0
:wait_loop
set /a attempt+=1
if !attempt! gtr 60 (
    echo    WARNING: Timeout
    goto open_browser
)
curl -s "http://localhost:%API_PORT%/health" >nul 2>&1
if errorlevel 1 (
    echo    Waiting... !attempt!/60
    timeout /t 1 /nobreak >nul
    goto wait_loop
)
echo    OK: API is healthy!
echo.

:open_browser
echo Opening Chrome with 4 tabs...
start "" chrome "http://localhost:%API_PORT%/docs"
timeout /t 1 /nobreak >nul
start "" chrome "http://localhost:%DASHBOARD_PORT%"
timeout /t 1 /nobreak >nul
start "" chrome "http://localhost:%PROMETHEUS_PORT%/targets"
timeout /t 1 /nobreak >nul
start "" chrome "http://localhost:%GRAFANA_PORT%"

echo.
echo ============================================================
echo  Stack is running!
echo ============================================================
echo.
echo  Project:      !COMPOSE_PROJECT_NAME!
echo  Jupyter Lab:  http://localhost:%JUPYTER_PORT%/lab    ^(Start Lab.bat^)
echo  API:          http://localhost:%API_PORT%/docs
echo  Dashboard:    http://localhost:%DASHBOARD_PORT%
echo  Prometheus:   http://localhost:%PROMETHEUS_PORT%
echo  Grafana:      http://localhost:%GRAFANA_PORT%
echo.
echo Useful commands:
echo    docker compose -p !COMPOSE_PROJECT_NAME! ps             - status
echo    docker compose -p !COMPOSE_PROJECT_NAME! logs -f api    - logs
echo    docker compose -p !COMPOSE_PROJECT_NAME! down           - stop
echo    stop.bat                                                - stop
echo.
echo Press any key to exit...
pause >nul
endlocal