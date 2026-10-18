@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Could not find the virtual environment at "%CD%\.venv".
    echo Create it first, then run this file again.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python -m cli web

endlocal
