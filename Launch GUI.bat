@echo off
setlocal
cd /d "%~dp0"

REM Prefer parent project venv (Python 3.12, all ML deps), then local venv
set "PY="
if exist "..\.venv\Scripts\python.exe" set "PY=..\.venv\Scripts\python.exe"
if not defined PY if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

if not defined PY (
    echo No virtual environment found. Creating .venv with Python 3.12...
    py -3.12 -m venv .venv 2>nul
    if errorlevel 1 python -m venv .venv
    set "PY=.venv\Scripts\python.exe"
)

echo Installing / updating dependencies (includes Pillow for the GUI)...
"%PY%" -m pip install -q --upgrade pip
"%PY%" -m pip install -q -r requirements.txt
"%PY%" -m pip install -q Pillow customtkinter
"%PY%" -m pip install -q -e .

echo.
echo Starting PSO-TF-IDF GUI...
"%PY%" gui\app.py
if errorlevel 1 (
    echo.
    echo If you see "No module named PIL", run manually:
    echo   "%PY%" -m pip install Pillow customtkinter
    pause
)
endlocal
