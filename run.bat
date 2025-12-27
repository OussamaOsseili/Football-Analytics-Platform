@echo off
echo Starting Football Analytics Platform...
echo.

call venv\Scripts\activate.bat

echo [1/2] Running ML Pipeline...
python src\ml\train_pipeline.py

echo.
echo [2/2] Launching Dashboard...
streamlit run src\dashboard\app.py

pause
