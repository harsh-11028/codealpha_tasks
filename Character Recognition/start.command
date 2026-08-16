#!/bin/bash
# macOS Double-Click Launcher for AI Handwritten OCR Studio
# Automatically changes directory to project root and runs one-command workflow

cd "$(dirname "$0")"

echo "================================================================="
echo "🦄 Starting Antigravity AI OCR Studio on macOS..."
echo "================================================================="

# Verify npm / node is installed
if ! command -v npm &> /dev/null; then
    echo "❌ Error: Node.js (npm) was not detected. Please install Node.js from https://nodejs.org/"
    read -p "Press any key to close..."
    exit 1
fi

# Ensure root packages and automation scripts are configured
if [ ! -d "node_modules" ]; then
    echo "📦 Initial run detected! Performing automated setup..."
    npm install
fi

# Launch one-command developer workflow (FastAPI + React + Auto Browser)
npm run dev

read -p "Application shutdown complete. Press any key to exit..."
