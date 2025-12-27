@echo off
echo =====================================
echo Football Analytics - API Server
echo =====================================
echo.

call venv\Scripts\activate.bat

echo Starting API on http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.

uvicorn src.api.main:app --reload --port 8000

pause
