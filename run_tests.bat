@echo off
echo Running Tests...
echo.

call venv\Scripts\activate.bat

echo [1/2] Unit Tests
python -m pytest tests/test_feature_engineering.py -v

echo.
echo [2/2] API Tests
python -m pytest tests/test_api.py -v

echo.
echo ===================================
echo Tests Complete!
echo ===================================
pause
