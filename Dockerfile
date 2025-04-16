# Use specific Python version for stability
FROM python:3.10-slim

# Add metadata
LABEL maintainer="GitHub-HF Dataset Creator"
LABEL description="Tool to create Hugging Face datasets from GitHub repositories"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libffi-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories for persistent data
RUN mkdir -p /data/cache /data/logs /data/config /data/dataset_metadata

# Set environment variables
ENV APP_DIR=/data \
    CACHE_DIR=/data/cache \
    LOG_DIR=/data/logs \
    CONFIG_DIR=/data/config \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure data directories have appropriate permissions
RUN chmod -R 755 /data

# Create non-root user for better security
RUN useradd -m appuser && \
    chown -R appuser:appuser /app /data

# Switch to non-root user
USER appuser

# Default command to run the application
ENTRYPOINT ["python", "main.py"]
