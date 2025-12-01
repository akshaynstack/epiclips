# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ViewCreator Genesis is the media processing engine for ViewCreator, a production-ready FastAPI microservice that transforms long-form videos into viral short-form clips using AI-powered analysis (Gemini), intelligent cropping (MediaPipe/YOLO), and automated caption generation (Whisper/FFmpeg).

## IMPORTANT: Streaming Context

**This codebase is being developed on stream with live viewers.** All output from Claude Code is visible to stream viewers.

**Security Requirements:**
- **NEVER display environment variable values in chat output** (API keys, secrets, tokens, database credentials, etc.)
- Reference environment variables by name only (e.g., "configure the GROQ_API_KEY variable")
- Be mindful that all responses are publicly visible
- Avoid exposing sensitive information such as:
  - API keys (Groq, OpenRouter, AWS)
  - S3 bucket names if they reveal internal structure
  - Private user data or PII
  - Internal URLs or private infrastructure details

**This should not affect the quality or thoroughness of work**, but outputs must be carefully considered to protect sensitive information while still being helpful and complete.

## Development Commands

```bash
# Local Setup (Python 3.10+)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start local server (hot reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Docker Development
docker-compose up --build
docker-compose up -d        # Detached
docker logs -f viewcreator-genesis

# Testing
pytest                      # Run all tests
pytest tests/test_pipeline.py # Run specific test file
pytest -v -s                # Verbose output with stdout

# Linting & Formatting (if configured)
# Currently relies on standard Python practices; ensure code is clean.
```

## Architecture

### Directory Structure
- `app/`: Main application code
  - `main.py`: FastAPI entry point
  - `routers/`: API endpoints (`ai_clipping`, `detection`, `health`)
  - `services/`: Business logic and core pipelines
    - `ai_clipping_pipeline.py`: Main orchestrator
    - `transcription_service.py`, `intelligence_planner.py`: AI integration
    - `detection_pipeline.py`, `rendering_service.py`: Media processing
  - `schemas/`: Pydantic request/response models
- `models/`: Local ML models (e.g., YOLO weights)
- `tests/`: Pytest suites

### Key Services & Pipeline Stages
1.  **VideoDownloader**: Handles YouTube (`yt-dlp`) and S3 downloads.
2.  **Transcription**: Uses Groq Whisper API for fast, word-level timestamps.
3.  **IntelligencePlanner**: Uses Gemini (via OpenRouter) to identify viral clips.
4.  **DetectionPipeline**:
    - Face Detection: Multi-tier (MediaPipe > YOLO > Haar).
    - Pose Estimation: MediaPipe.
    - Object Tracking: DeepSORT.
5.  **ContentRegionDetector**: Identifies screen shares vs. talking heads.
6.  **RenderingService**: FFmpeg-based cropping and composition (`focus_mode`, `split_mode`).
7.  **CaptionGenerator**: Creates `.ass` subtitles with word-level highlighting.
8.  **S3UploadService**: Manages artifact storage.

### Dependency Graph
`AIClippingPipeline` -> `VideoDownloader`, `TranscriptionService`, `IntelligencePlanner`
`DetectionPipeline` -> `FaceDetector`, `PoseEstimator`, `Tracker`
`RenderingService` -> `ContentRegionDetector`, `CaptionGenerator`

## Important Development Notes

### Environment Variables
Required variables (must be set in `.env` or environment):
- `GROQ_API_KEY`: For transcription.
- `OPENROUTER_API_KEY`: For AI planning.
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `S3_BUCKET`: For storage.
- `DEBUG`, `LOG_LEVEL`: Application tuning.

### System Dependencies
- **FFmpeg**: Required for all video processing and rendering.
- **yt-dlp**: Required for downloading YouTube videos.
- **Python 3.10+**: Recommended runtime.

### Common Gotchas
- **ML Model Loading**: Models (`yolov8n.pt`) are loaded on startup. Ensure `models/` directory exists or internet access is available for initial download.
- **Torch/CUDA**: The `Dockerfile` installs CPU-only PyTorch to keep image size down. Local dev might use CUDA if available, but ensure compatibility.
- **Async Processing**: The pipeline is heavily async. Use `await` correctly when calling services.
- **S3 & CDN**: Output URLs are transformed to CDN URLs in the API, but Genesis returns S3 URLs or pre-signed URLs depending on config.
