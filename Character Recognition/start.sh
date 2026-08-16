#!/bin/bash
# Linux Desktop Launcher for AI Handwritten OCR Studio
# Automatically resolves project root directory and executes one-command dev workflow

cd "$(dirname "$0")"

echo "================================================================="
echo "🦄 Starting Antigravity AI OCR Studio on Linux..."
echo "================================================================="

# Check Node.js availability
if ! command -v npm &> /dev/null; then
    echo "❌ Error: Node.js (npm) was not detected. Please install via your distribution package manager or https://nodejs.org/"
    read -p "Press Enter to exit..."
    exit 1
fi

# Ensure initial node setup is completed
if [ ! -d "node_modules" ]; then
    echo "📦 Initial run detected! Performing automated setup..."
    npm install
fi

# Execute unified developer suite
npm run dev

read -p "Application terminated. Press Enter to exit..."
