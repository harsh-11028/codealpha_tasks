# backend/Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for audio processing (libsndfile for librosa/soundfile)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better cache utilization
COPY requirements.txt .

# Install Python dependencies
# We use --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY . .

# Set environment variables for production
ENV APP_ENV=production
ENV PYTHONPATH=/app
ENV BACKEND_HOST=0.0.0.0
ENV BACKEND_PORT=8001

# Create volume mount points for databases and models
RUN mkdir -p /app/data /app/saved_models

# Expose FastAPI port
EXPOSE 8001

# Run the FastAPI server using Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
