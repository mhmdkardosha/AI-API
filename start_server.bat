@echo off
REM ============================================================================
REM Unified Multi-Modal API Server - Windows Startup Script
REM ============================================================================

echo.
echo ========================================================================
echo    Unified Multi-Modal API Server - Startup Script
echo ========================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if Ollama is installed
ollama --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama is not installed or not in PATH
    echo Please install Ollama from https://ollama.ai/download
    pause
    exit /b 1
)

echo [OK] Python installed
echo [OK] Ollama installed
echo.

REM Install dependencies
echo Installing dependencies...
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo [OK] Dependencies installed
echo.

REM Check if models are available
echo Checking Ollama models...
ollama list | findstr /C:"gemma3:4b" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Model 'gemma3:4b' not found
    echo Pull it with: ollama pull gemma3:4b
    set MISSING_MODELS=1
)

ollama list | findstr /C:"embeddinggemma:300m" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Model 'embeddinggemma:300m' not found
    echo Pull it with: ollama pull embeddinggemma:300m
    set MISSING_MODELS=1
)

ollama list | findstr /C:"llava:7b" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Model 'llava:7b' not found
    echo Pull it with: ollama pull llava:7b
    set MISSING_MODELS=1
)

if defined MISSING_MODELS (
    echo.
    echo [WARNING] Some models are missing
    echo.
    echo Do you want to pull the missing models now? [Y/N]
    set /p PULL_MODELS=
    if /i "%PULL_MODELS%"=="Y" (
        echo.
        echo Pulling models... This may take several minutes...
        ollama pull gemma3:4b
        ollama pull embeddinggemma:300m
        ollama pull llava:7b
        echo.
        echo [OK] Models downloaded
    ) else (
        echo.
        echo [WARNING] Starting without all models
        echo Some features may not work correctly
        echo.
    )
)

echo.
echo ========================================================================
echo    Starting Unified Multi-Modal API Server
echo ========================================================================
echo.
echo Server will start on: http://localhost:8000
echo API Documentation:    http://localhost:8000/docs
echo Health Check:         http://localhost:8000/health
echo.
echo Press Ctrl+C to stop the server
echo.
echo ========================================================================
echo.

REM Start the server
python api_server_integrated.py

REM If server exits
echo.
echo ========================================================================
echo    Server Stopped
echo ========================================================================
echo.
pause
