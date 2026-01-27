# Hybrid Analyzer Dockerfile
# GPU-accelerated audio analysis service for EternalJukebox
# Supports Madmom, Essentia, nnAudio with CPU fallback

FROM python:3.11-slim

# Prevent apt from hanging on interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

WORKDIR /app

# =============================================================================
# INSTALL SYSTEM DEPENDENCIES
# =============================================================================
# Base dependencies + Essentia requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    # FFmpeg for audio decoding
    ffmpeg \
    curl \
    # Build tools for native packages
    build-essential \
    gcc \
    g++ \
    # Essentia dependencies
    libyaml-dev \
    libfftw3-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libsamplerate0-dev \
    libtag1-dev \
    libchromaprint-dev \
    # Additional audio libs
    libsndfile1 \
    libsndfile1-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# INSTALL PYTHON DEPENDENCIES
# =============================================================================
# Copy requirements first for better caching
COPY requirements.txt .

# Install build prerequisites first (required for madmom which needs Cython)
RUN pip install --no-cache-dir Cython numpy setuptools wheel

# Install Python dependencies
# Note: For GPU support, use nvidia/cuda base image and install torch with CUDA
# --no-build-isolation ensures madmom uses the pre-installed Cython
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

# Install Essentia (prefer binary wheel, fallback to source)
RUN pip install --only-binary=essentia essentia || pip install essentia || echo "Essentia installation failed, will use fallback"

# =============================================================================
# COPY APPLICATION CODE
# =============================================================================
COPY *.py ./
COPY *.json ./

# Create directories for data
RUN mkdir -p /data/uploaded_analysis /data/audio

# =============================================================================
# RUNTIME CONFIGURATION
# =============================================================================
# Expose the API port
EXPOSE 6874

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:6874/docs || exit 1

# Run the API server
CMD ["python", "api.py"]
