# =============================================================================
# Epiclips - Multi-stage Dockerfile (Optimized)
# =============================================================================
# Optimizations:
# - Uses 'uv' for 10-100x faster pip installs
# - Better layer ordering for cache efficiency
# - Combines RUN commands where possible
# - Excludes test dependencies from production image
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install Python dependencies with uv (FAST)
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install uv (extremely fast Python package installer)
# https://github.com/astral-sh/uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create virtual environment
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv"

# Copy only requirements first (cache layer if code changes but deps don't)
COPY requirements.txt .

# Install Python dependencies with uv (10-100x faster than pip)
RUN uv pip install --no-cache -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Minimal production image
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Labels
LABEL maintainer="akshaynstack" \
      description="Epiclips - Open-source AI-powered video clipping" \
      version="1.0.0"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    APP_HOME=/app \
    TEMP_DIRECTORY=/tmp/genesis \
    MEDIAPIPE_DISABLE_GPU=1 \
    YTDLP_NO_UPDATE=1 \
    MAX_WORKERS=4 \
    MAX_RENDER_WORKERS=3

WORKDIR $APP_HOME

# Install runtime dependencies (combined into single layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # FFmpeg for video processing
    ffmpeg \
    # OpenCV dependencies
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    # For healthcheck and downloads
    curl \
    gnupg \
    # Fonts for caption rendering
    fonts-dejavu-core \
    fonts-liberation \
    fontconfig \
    # yt-dlp dependencies
    ca-certificates \
    unzip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    # Refresh font cache
    && fc-cache -fv

# Install Node.js 20 LTS (required for yt-dlp YouTube extraction)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && node --version

# Install yt-dlp binary and Deno in single layer
RUN curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp \
    && chmod a+rx /usr/local/bin/yt-dlp \
    && curl -fsSL https://deno.land/install.sh | DENO_INSTALL="/usr/local" sh

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create models directory
RUN mkdir -p /app/models

# Copy application code (this layer changes most often - keep at end)
COPY app/ /app/app/

# Set permissions
RUN mkdir -p $TEMP_DIRECTORY \
    && chown -R appuser:appuser $APP_HOME \
    && chown -R appuser:appuser $TEMP_DIRECTORY

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
