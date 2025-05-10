@echo off
REM Setup script for Active Share Optimizer on Windows
REM Creates a Python 3.11 virtual environment, installs all dependencies, and bootstraps the CBC solver.

REM 1. Check for Python 3.11
where python3.11 >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo Python 3.11 is not installed.
  echo Please install Python 3.11 from https://www.python.org/downloads/ and check "Add Python to PATH".
  pause
  exit /b 1
)

echo Creating Python 3.11 virtual environment…
python3.11 -m venv activeshare_env_py311

echo Activating the environment…
call activeshare_env_py311\Scripts\activate

echo Upgrading pip & installing Python packages…
pip install --upgrade pip
pip install -r requirements.txt

echo Installing CBC solver…
REM Try Conda first
where conda >nul 2>&1
if %ERRORLEVEL% EQU 0 (
  echo • Installing via conda…
  conda install -c conda-forge coincbc -y
) else (
  REM Then winget
  where winget >nul 2>&1
  if %ERRORLEVEL% EQU 0 (
    echo • Installing via winget…
    winget install --exact --id coinor-cbc
  ) else (
    REM Then Chocolatey
    where choco >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
      echo • Installing via Chocolatey…
      choco install coinor-cbc -y
    ) else (
      echo Could not find conda, winget, or choco.
      echo Please install the CBC solver manually: https://github.com/coin-or/Cbc
      pause
    )
  )
)

echo Verifying installation…
python - <<PYCODE
import pulp
print("PuLP version:", pulp.__version__)
print("CBC available:", pulp.PULP_CBC_CMD().available())
PYCODE

if %ERRORLEVEL% EQU 0 (
  echo.
  echo ✅ Setup complete!
  echo To reactivate: activeshare_env_py311\Scripts\activate
  echo To run the app: python -m streamlit run app.py
) else (
  echo.
  echo ❌ There was an issue with the installation.
  echo Please check the errors above and try again.
  echo You can also try the conda-based method described in the README.
)

pause
