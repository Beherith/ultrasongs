@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Ultrastar Song Creator

:: Check for a real production build (BUILD_ID is written by next build)
if not exist ".next\BUILD_ID" (
    echo Compilando la aplicacion, esto tarda unos minutos...
    pnpm build
    if errorlevel 1 (
        echo ERROR: El build fallo. Revisa los errores arriba.
        pause
        exit /b 1
    )
)

:: Get local LAN IP (first IPv4 that starts with 192.168 or 10. or 172.)
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /R "IPv4.*192\." ^| findstr /v "192\.198"') do (
    set LAN_IP=%%a
    goto :found
)
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /R "IPv4.*10\." ') do (
    set LAN_IP=%%a
    goto :found
)
set LAN_IP=(desconocida)

:found
set LAN_IP=%LAN_IP: =%

echo.
echo  Ultrastar Song Creator
echo  Local:        http://localhost:3010
echo  Red local:    http://%LAN_IP%:3010
echo.
echo  Cerra esta ventana para detener la app.
echo.
pnpm start:all
