@echo off
echo Wallet Risk Scoring System
echo ================================

REM Activate virtual environment if it exists
if exist "wallet_risk_env\Scripts\activate.bat" (
    echo Activating virtual environment...
    call wallet_risk_env\Scripts\activate.bat
)

REM Check if main script exists
if not exist "src\main.py" (
    echo Main script not found. Please complete setup first.
    pause
    exit /b 1
)

echo Starting wallet risk scoring process...
python src\main.py

echo.
echo Process completed! Check results\ folder for output.
pause
