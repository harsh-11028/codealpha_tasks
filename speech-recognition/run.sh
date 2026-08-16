#!/bin/bash
# run.sh - Orchestrator to run both backend and frontend

echo "🚀 Starting Speech Emotion Recognition System..."

# Function to kill all background processes on exit
cleanup() {
    echo "🛑 Shutting down..."
    kill $(jobs -p) 2>/dev/null
    exit
}
trap cleanup EXIT INT TERM

# Start Backend
echo "➔ Starting FastAPI Backend..."
cd backend
# Create a dummy trained model so the prediction engine loads successfully
mkdir -p saved_models
if [ ! -f "saved_models/best_model.pt" ]; then
    echo "   [Mocking best_model.pt for first run - prediction will run in fallback mode]"
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "   [Activating virtual environment...]"
    source venv/bin/activate
fi

uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Start Frontend
echo "➔ Starting Vite Frontend..."
cd ../frontend
npm run dev -- --port 5173 --host 127.0.0.1 &
FRONTEND_PID=$!

echo "✅ System is running!"
echo "   Backend API: http://127.0.0.1:8000"
echo "   Frontend UI: http://127.0.0.1:5173"
echo "   Press Ctrl+C to stop both servers."

wait
