@echo off
echo ========================================
echo GDELT Conflict Predictor - Frontend
echo ========================================
echo.

:: Check if node_modules exists
if not exist node_modules (
    echo Installing dependencies...
    call npm install
    echo.
)

echo Starting Angular development server...
echo Frontend will be available at: http://localhost:4200
echo.
echo Make sure the backend is running on http://localhost:8000
echo.

call npm start

pause
