# =============================================================================
# Epiclips - Multi-stage Dockerfile (Lean Build)
# =============================================================================
# Optimizations:
# - Uses 'uv' for 10-100x faster pip installs
# - Removed Deno (not used in code)
# - CPU-only (MEDIAPIPE_DISABLE_GPU=1)
# - Minimal system dependencies
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install Python dependencies with uv (FAST)
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install uv (extremely fast Python package installer)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create virtual environment
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv"

# Copy requirements first (cache layer)
COPY requirements.txt .

# Install Python dependencies with uv (10-100x faster than pip)
RUN uv pip install --no-cache -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Minimal production image
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="akshaynstack" \
      description="Epiclips - Open-source AI-powered video clipping" \
      version="1.0.0"

# Environment variables (CPU-only mode)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    APP_HOME=/app \
    TEMP_DIRECTORY=/tmp/epiclips \
    # CPU-only mode - disable all GPU features
    MEDIAPIPE_DISABLE_GPU=1 \
    CUDA_VISIBLE_DEVICES="" \
    CT2_FORCE_CPU=1 \
    YTDLP_NO_UPDATE=1 \
    MAX_WORKERS=4 \
    MAX_RENDER_WORKERS=3

WORKDIR $APP_HOME

# Install runtime dependencies (single layer, minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # FFmpeg for video processing
    ffmpeg \
    # OpenCV dependencies (minimal set)
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    # For healthcheck
    curl \
    # Fonts for caption rendering
    fonts-dejavu-core \
    fontconfig \
    # yt-dlp dependencies
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -fv

# Install Node.js 20 LTS (required for yt-dlp YouTube extraction)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && node --version

# Install yt-dlp binary (standalone, no Deno needed)
RUN curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp \
    && chmod a+rx /usr/local/bin/yt-dlp

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create directories
RUN mkdir -p /app/models $TEMP_DIRECTORY

# Copy application code
COPY app/ /app/app/

# Set permissions
RUN chown -R appuser:appuser $APP_HOME $TEMP_DIRECTORY

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
