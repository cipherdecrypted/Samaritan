@echo off
setlocal
cd /d "%~dp0"

set "PY="
py -3.12 -c "import sys" >nul 2>&1 && set "PY=py -3.12"
if not defined PY (
  python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo Python 3.12 is required: https://www.python.org/downloads/ or the Microsoft Store.
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating environment...
  %PY% -m venv .venv || exit /b 1
)

echo Installing packages...
".venv\Scripts\python.exe" -m pip --version >nul 2>&1 || ".venv\Scripts\python.exe" -m ensurepip --upgrade >nul || exit /b 1
".venv\Scripts\python.exe" -m pip install -q --disable-pip-version-check -r requirements.txt || exit /b 1

".venv\Scripts\python.exe" tools\setup_models.py %* || exit /b 1

echo.
echo Gotham is ready. Start it with Gotham.bat
endlocal
