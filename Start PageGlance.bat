
@echo off
title PageGlance Backend
cd /d "C:\Users\ansha\pageglance\backend"

echo Starting PageGlance backend...
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

echo.
echo PageGlance backend stopped.
pause
