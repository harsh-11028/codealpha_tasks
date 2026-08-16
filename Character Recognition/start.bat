@echo off
:: Windows Double-Click Launcher for AI Handwritten OCR Studio
:: Automatically switches to project root directory and executes one-command dev workflow

title Antigravity AI OCR Studio
cd /d "%~dp0"

echo =================================================================
echo 🦄 Starting Antigravity AI OCR Studio on Windows...
echo =================================================================

:: Check if Node/npm is installed
where npm >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Error: Node.js (npm) was not detected in PATH. Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

:: Ensure node_modules exists
if not exist "node_modules" (
    echo 📦 Initial run detected! Performing automated setup...
    call npm install
)

:: Launch developer suite
call npm run dev

pause
