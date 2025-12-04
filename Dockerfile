# =============================================================================
# ViewCreator Genesis - Multi-stage Dockerfile
# =============================================================================
# Stage 1: Build stage - Install dependencies
# Stage 2: Runtime stage - Minimal image for production
#
# Genesis is ViewCreator's media processing engine, providing:
# - Video detection (MediaPipe, DeepSORT)
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

# Create virtual environment - this ensures pip sees already-installed packages
# when resolving dependencies (unlike --prefix which causes path discovery issues)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

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
    # Virtual environment (copied from builder)
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    # App config
    APP_HOME=/app \
    TEMP_DIRECTORY=/tmp/genesis \
    # Disable MediaPipe GPU (use CPU)
    MEDIAPIPE_DISABLE_GPU=1 \
    # yt-dlp config
    YTDLP_NO_UPDATE=1 \
    # Parallel processing defaults
    MAX_WORKERS=4 \
    MAX_RENDER_WORKERS=3

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
    # For healthcheck and Node.js installation
    curl \
    gnupg \
    # Fonts for caption rendering (viral-style subtitles)
    fonts-dejavu-core \
    fonts-liberation \
    fontconfig \
    # yt-dlp dependencies
    ca-certificates \
    unzip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    # Refresh font cache for caption rendering
    && fc-cache -fv

# Install Node.js 20 LTS for yt-dlp YouTube extraction (required since late 2024)
# Using NodeSource for a proper Node.js installation that yt-dlp can detect
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && node --version \
    && which node

# Ensure 'node' command exists (symlink nodejs -> node if needed)
RUN if [ ! -f /usr/bin/node ] && [ -f /usr/bin/nodejs ]; then ln -s /usr/bin/nodejs /usr/bin/node; fi

# Verify node is executable
RUN chmod +x /usr/bin/node || true

# Install yt-dlp binary (standalone, doesn't require Python dependencies)
RUN curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp \
    && chmod a+rx /usr/local/bin/yt-dlp

# Install Deno (JS runtime for yt-dlp)
ENV DENO_INSTALL="/usr/local"
RUN curl -fsSL https://deno.land/install.sh | sh

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment from builder (contains all Python packages)
COPY --from=builder /opt/venv /opt/venv

# Create models directory for Haar cascades (bundled with OpenCV)
RUN mkdir -p /app/models

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

