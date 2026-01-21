# Floppa Analyzer Dockerfile
# GPU-accelerated audio analysis service for EternalJukebox

FROM python:3.11-slim

# Prevent apt from hanging on interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

WORKDIR /app

# Install system dependencies (non-interactive)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Note: For GPU support, use nvidia/cuda base image and install torch with CUDA
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./

# Create directories for data
RUN mkdir -p /data/uploaded_analysis /data/audio

# Expose the API port
EXPOSE 6874

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:6874/docs || exit 1

# Run the API server
CMD ["python", "api.py"]
