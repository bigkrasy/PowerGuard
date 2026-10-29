@echo off
REM ============================================================
REM FULL Docker Cleanup
REM Deletes ALL containers, images, volumes, networks, cache
REM ============================================================

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo.
echo ============================================================
echo  DOCKER FULL CLEANUP
echo ============================================================
echo.
echo WARNING: This will delete ALL Docker resources:
echo   - All containers (running + stopped)
echo   - All images (including official prometheus/grafana)
echo   - All volumes
echo   - All networks (except default)
echo   - Build cache
echo.
echo This affects EVERY project (GridForecast, PowerGuard, etc.)
echo.
echo You have 5 seconds to cancel (Ctrl+C)...
timeout /t 5 /nobreak

echo.
echo [1/6] Stopping all containers...
for /f "delims=" %%i in ('docker ps -aq 2^>nul') do (
    docker stop %%i >nul 2>&1
)
echo    OK

echo.
echo [2/6] Removing all containers...
for /f "delims=" %%i in ('docker ps -aq 2^>nul') do (
    docker rm -f %%i >nul 2>&1
)
echo    OK

echo.
echo [3/6] Removing all images...
for /f "delims=" %%i in ('docker images -q 2^>nul') do (
    docker rmi -f %%i >nul 2>&1
)
echo    OK

echo.
echo [4/6] Removing all volumes...
for /f "delims=" %%i in ('docker volume ls -q 2^>nul') do (
    docker volume rm -f %%i >nul 2>&1
)
echo    OK

echo.
echo [5/6] Removing all networks (except default)...
docker network prune -f >nul 2>&1
echo    OK

echo.
echo [6/6] Removing build cache...
docker builder prune -af >nul 2>&1
echo    OK

echo.
echo ============================================================
echo  FULL CLEANUP COMPLETE
echo ============================================================
echo.
echo  Remaining:
echo.
echo  Containers:
docker ps -a --format "  {{.Names}}" 2>nul
echo.
echo  Images:
docker images --format "  {{.Repository}}:{{.Tag}}" 2>nul
echo.
echo  Volumes:
docker volume ls --format "  {{.Name}}" 2>nul
echo.
echo  Networks:
docker network ls --format "  {{.Name}}" 2>nul
echo.
pause
endlocal