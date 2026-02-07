@echo off
echo ========================================
echo Starting GDELT Conflict Predictor API
echo ========================================
echo.

cd backend

:: Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
echo.

:: Upgrade pip and install build tools
echo Upgrading pip and setuptools...
python -m pip install --upgrade pip setuptools wheel
echo.

:: Install dependencies
echo Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt
echo.

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install dependencies
    echo.
    echo Try running manually:
    echo   cd prototype\backend
    echo   venv\Scripts\activate
    echo   pip install --upgrade pip setuptools wheel
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo ========================================
echo Dependencies installed successfully!
echo ========================================
echo.

:: Start the API
echo ========================================
echo Starting FastAPI server...
echo API will be available at: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo ========================================
echo.
python main.py

pause
