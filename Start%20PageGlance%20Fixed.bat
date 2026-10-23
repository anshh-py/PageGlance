@echo off
setlocal
 title PageGlance Backend
cd /d "C:\Users\ansha\pageglance\backend"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Backend virtual environment was not found at:
    echo %CD%\.venv\Scripts\python.exe
    echo Check that PageGlance is installed at C:\Users\ansha\pageglance
    pause
    exit /b 1
)

echo Starting PageGlance backend on http://127.0.0.1:8765 ...
echo Swagger UI: http://127.0.0.1:8765/docs
echo Health check: http://127.0.0.1:8765/health
echo.
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8765

if errorlevel 1 (
    echo.
    echo PageGlance backend exited with an error. Read the message above.
)
echo.
echo PageGlance backend stopped.
pause
endlocal
