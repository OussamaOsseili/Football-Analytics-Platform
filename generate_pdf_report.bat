@echo off
echo ====================================
echo PDF Report Generator
echo ====================================
echo.

call venv\Scripts\activate.bat

echo Generating sample scouting PDF report...
python src\reports\pdf_report_generator.py

echo.
echo Report saved in reports/ folder
pause
