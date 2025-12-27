@echo off
echo ====================================
echo Football Analytics - Quick Setup
echo ====================================
echo.

echo [1/4] Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create venv
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo [4/4] Copying environment file...
copy .env.example .env

echo.
echo ====================================
echo ✓ Setup complete!
echo ====================================
echo.
echo Next steps:
echo   1. Run ETL: python src\etl\etl_pipeline.py
echo   2. Run ML: python src\ml\train_pipeline.py
echo   3. Launch dashboard: streamlit run src\dashboard\app.py
echo.
pause
