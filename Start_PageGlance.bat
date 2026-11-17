@echo off
title PageGlance Backend
cd /d "C:\Users\ansha\pageglance\backend"

echo ========================================
echo       Starting PageGlance Backend
echo ========================================
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Backend virtual environment not found.
    echo Check the backend folder and .venv.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

echo Starting FastAPI server...
echo API: http://127.0.0.1:8000
echo Health: http://127.0.0.1:8000/health
echo.
echo Keep this window open while using PageGlance.
echo Press CTRL+C to stop the backend.
echo.

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

echo.
echo PageGlance backend stopped.
pause
