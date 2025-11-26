# =============================================================================
# ViewCreator Clipping Worker - Multi-stage Dockerfile
# =============================================================================
# Stage 1: Build stage - Install dependencies and download models
# Stage 2: Runtime stage - Minimal image for production
#
# This service provides:
# - Video detection (YOLO, MediaPipe, DeepSORT)
# - Full AI clipping pipeline (transcription, intelligence, rendering)
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Download YOLO model weights
RUN mkdir -p /models && \
    pip install --no-cache-dir ultralytics && \
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" && \
    cp ~/.config/ultralytics/yolov8n.pt /models/ || \
    curl -L -o /models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Labels
LABEL maintainer="ViewCreator"
LABEL description="AI-powered video clipping worker with transcription, intelligence, and rendering"
LABEL version="2.0.0"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    # App config
    APP_HOME=/app \
    TEMP_DIRECTORY=/tmp/clipping-worker \
    YOLO_MODEL_PATH=/app/models/yolov8n.pt \
    # Disable MediaPipe GPU (use CPU)
    MEDIAPIPE_DISABLE_GPU=1 \
    # yt-dlp config
    YTDLP_NO_UPDATE=1

WORKDIR $APP_HOME

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # FFmpeg for video processing
    ffmpeg \
    # OpenCV dependencies (libgl1 replaces deprecated libgl1-mesa-glx)
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    # For healthcheck
    curl \
    # Fonts for caption rendering (viral-style subtitles)
    fonts-dejavu-core \
    fonts-liberation \
    fontconfig \
    # yt-dlp dependencies
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    # Refresh font cache for caption rendering
    && fc-cache -fv

# Install yt-dlp binary (standalone, doesn't require Python dependencies)
RUN curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp \
    && chmod a+rx /usr/local/bin/yt-dlp

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy YOLO model weights
COPY --from=builder /models /app/models

# Copy application code
COPY app/ /app/app/

# Create temp directory and set permissions
RUN mkdir -p $TEMP_DIRECTORY && \
    chown -R appuser:appuser $APP_HOME && \
    chown -R appuser:appuser $TEMP_DIRECTORY

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

