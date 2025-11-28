# AI Clipping Worker Migration Plan

## Overview

This document outlines the migration of the AI Clipping Agent from `viewcreator-api` (NestJS/TypeScript) to `viewcreator-genesis` (Python/FastAPI). The goal is to consolidate all video processing, transcription, intelligence planning, and rendering into a single Python microservice.

## Current Architecture (Before Migration)

### viewcreator-api (NestJS)
- `VideoIngestionService` - Downloads video via yt-dlp, extracts audio, generates transcripts
- `IntelligenceProcessorService` - Uses Gemini via OpenRouter to plan clips
- `ActionAwareCropService` - Orchestrates crop analysis (calls detection worker)
- `SmartCropService` - Legacy Gemini-based cropping (deprecated)
- `RenderingFactoryService` - FFmpeg-based rendering with captions
- `CaptionGeneratorService` - ASS subtitle generation
- `ClipStorageService` - S3 upload
- `DetectionWorkerClientService` - HTTP client to Python worker
- `AiClippingAgentProcessor` - BullMQ job processor

### viewcreator-genesis (Python)
- Face detection (YOLO)
- Pose estimation (MediaPipe)
- Object tracking (DeepSORT)

## Target Architecture (After Migration)

### viewcreator-genesis (Python) - FULL PIPELINE
```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AI Clipping Pipeline (Python)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   1. INGESTION                                                          │
│   ├── YouTube Download (yt-dlp)                                         │
│   ├── Audio Extraction (ffmpeg)                                         │
│   ├── Frame Extraction (ffmpeg)                                         │
│   └── Transcription (Groq Whisper - 216x realtime)                     │
│                                                                         │
│   2. INTELLIGENCE PLANNING                                              │
│   ├── Gemini 2.5 Pro via OpenRouter                                     │
│   └── Clip segment identification with virality scoring                 │
│                                                                         │
│   3. DETECTION & TRACKING                                               │
│   ├── YOLO Face Detection (existing)                                    │
│   ├── MediaPipe Pose Estimation (existing)                              │
│   └── DeepSORT Tracking (existing)                                      │
│                                                                         │
│   4. CROP ANALYSIS                                                      │
│   ├── Action-aware crop planning                                        │
│   ├── Physics-based smoothing (mass-spring-damper)                      │
│   └── Timeline generation for rendering                                 │
│                                                                         │
│   5. RENDERING                                                          │
│   ├── FFmpeg dynamic cropping                                           │
│   ├── ASS caption generation (word-by-word highlight)                   │
│   ├── Split-screen / focus mode rendering                               │
│   └── Final H.264 MP4 output                                           │
│                                                                         │
│   6. OUTPUT                                                             │
│   ├── S3 Upload                                                         │
│   └── Callback to NestJS API                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### viewcreator-api (NestJS) - THIN ORCHESTRATOR
- Job queue management (BullMQ)
- Credit deduction
- Job status tracking
- HTTP endpoint for initiating jobs
- Receives callbacks from Python worker

## New Directory Structure

```
viewcreator-genesis/
├── app/
│   ├── __init__.py
│   ├── main.py                          # FastAPI app entry point
│   ├── config.py                        # Configuration (updated)
│   │
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── health.py                    # Health checks (existing)
│   │   ├── detection.py                 # Detection endpoints (existing)
│   │   └── clipping.py                  # NEW: Full clipping pipeline
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── requests.py                  # Request DTOs (updated)
│   │   ├── responses.py                 # Response DTOs (updated)
│   │   └── clip_plan.py                 # NEW: Clip planning schemas
│   │
│   └── services/
│       ├── __init__.py
│       │
│       │ # EXISTING SERVICES
│       ├── detection_pipeline.py
│       ├── face_detector.py
│       ├── frame_extractor.py
│       ├── pose_estimator.py
│       ├── s3_client.py
│       ├── tracker.py
│       │
│       │ # NEW SERVICES
│       ├── video_downloader.py          # yt-dlp integration
│       ├── transcription_service.py     # Whisper transcription
│       ├── intelligence_planner.py      # Gemini clip planning
│       ├── action_aware_crop.py         # Crop analysis
│       ├── physics_smoother.py          # Mass-spring-damper
│       ├── caption_generator.py         # ASS subtitle generation
│       ├── rendering_service.py         # FFmpeg rendering
│       └── clipping_pipeline.py         # Full pipeline orchestrator
│
├── models/                              # ML model weights
├── tests/
├── Dockerfile                           # Updated with new deps
├── docker-compose.yml
├── requirements.txt                     # Updated
└── README.md                            # Updated
```

## API Endpoints

### New Clipping Endpoint

```
POST /clip
{
    "job_id": "uuid",
    "youtube_url": "https://www.youtube.com/watch?v=...",
    "requested_clip_count": 3,
    "preferred_language": "en",
    "owner_user_id": "user-123",
    "callback_url": "https://api.viewcreator.com/ai-clipping/callback",
    "options": {
        "max_video_duration_seconds": 7200,
        "frame_interval_seconds": 2.0,
        "caption_style": {
            "font_name": "Arial Black",
            "font_size": 72,
            "primary_color": "#FFFFFF",
            "highlight_color": "#FFD700",
            "position": "center",
            "word_by_word_highlight": true
        }
    }
}
```

### Response

```
{
    "job_id": "uuid",
    "status": "completed",
    "clips": [
        {
            "clip_index": 0,
            "start_time_ms": 15000,
            "end_time_ms": 45000,
            "duration_ms": 30000,
            "layout_type": "talking_head",
            "virality_score": 0.92,
            "summary": "Hook about AI revolution",
            "tags": ["ai", "technology", "future"],
            "s3_key": "users/user-123/clips/uuid-0.mp4",
            "public_url": "https://cdn.viewcreator.com/...",
            "file_size": 12345678
        }
    ],
    "insights": "This video contains compelling AI discussion with strong hooks",
    "processing_time_ms": 125000,
    "video_duration_ms": 600000
}
```

## Dependencies to Add

### requirements.txt additions

```
# Transcription (Groq Whisper)
groq==0.4.2                       # Groq API client

# AI/LLM
httpx>=0.26.0                     # HTTP client for OpenRouter

# Video Download
yt-dlp>=2024.1.0                  # YouTube downloader

# FFmpeg (already have opencv, add explicit ffmpeg-python)
ffmpeg-python>=0.2.0              # FFmpeg bindings

# Pydantic (already have, ensure v2)
pydantic>=2.0.0
pydantic-settings>=2.0.0
```

## Environment Variables

```bash
# OpenRouter (for Gemini)
OPENROUTER_API_KEY=your_key

# Groq (optional, faster transcription)
GROQ_API_KEY=your_key

# AWS S3
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
S3_BUCKET=viewcreator-media
AWS_REGION=us-east-1

# Processing Configuration
# Groq Whisper is the only supported transcription provider
TRANSCRIPTION_MODEL=whisper-large-v3-turbo
GEMINI_MODEL=google/gemini-2.5-flash
MAX_VIDEO_DURATION_SECONDS=7200
FRAME_INTERVAL_SECONDS=2.0
MAX_CONCURRENT_JOBS=2

# yt-dlp
YTDLP_PATH=/usr/local/bin/yt-dlp

# Output
WORKSPACE_ROOT=/tmp/ai-clipping-agent
```

## Migration Steps

### Phase 1: Core Services ✅
1. ✅ Create migration plan document
2. ✅ Add `video_downloader.py` - yt-dlp integration
3. ✅ Add `transcription_service.py` - Groq Whisper integration
4. ✅ Add `intelligence_planner.py` - Gemini via OpenRouter

### Phase 2: Rendering ✅
5. ✅ Add `caption_generator.py` - ASS subtitle generation (viral-style)
6. ✅ Add `rendering_service.py` - FFmpeg rendering with dynamic crop

### Phase 3: Integration ✅
7. ✅ Add `ai_clipping_pipeline.py` - Full orchestrator
8. ✅ Add `s3_upload_service.py` - S3 upload with metadata
9. ✅ Add `/ai-clipping` endpoint router

### Phase 4: Infrastructure ✅
10. ✅ Update `requirements.txt`
11. ✅ Update `Dockerfile` (yt-dlp, fonts, ffmpeg)
12. ✅ Update `config.py` (new settings)
13. ✅ Update `docker-compose.yml`
14. ✅ Update `README.md`

### Phase 5: API Integration (Pending)
15. ⏳ Update NestJS API to use new endpoint
16. ⏳ Add webhook callback handling
17. ⏳ Update job status tracking in API

## Key Implementation Notes

### Transcription Service
- Uses Groq Whisper (216x realtime speed, most cost-effective)
- Handle long audio via chunking (30min chunks)
- Return word-level timestamps for caption highlighting

### Intelligence Planner
- Use Gemini 2.5 Pro/Flash via OpenRouter
- Send transcript + sampled frames for multimodal analysis
- Return structured clip plans with timestamps, layouts, scores

### Action-Aware Crop
- Leverage existing YOLO/MediaPipe detection
- Build editorial frames from detections
- Apply physics smoothing for natural camera movement
- Generate primary/secondary timelines for focus/stack modes

### Rendering Service
- Dynamic FFmpeg crop expressions
- Support focus mode (single pan) and stack mode (split-screen)
- Burn in ASS captions with word-by-word highlighting
- Two-pass rendering for stack mode to avoid FFmpeg issues

### Caption Generator
- Generate ASS format subtitles
- Word-by-word highlighting (gold highlight on current word)
- Support configurable styling (font, color, position)
- Handle transcript segments and word-level timing

## Testing Strategy

1. **Unit Tests**: Each service independently
2. **Integration Tests**: Full pipeline with test videos
3. **Performance Tests**: Ensure processing stays under 5min for 10min videos
4. **E2E Tests**: API endpoint → S3 upload → callback

## Rollout Plan

1. **Dev Environment**: Deploy new worker, test with staging API
2. **Canary**: Route 10% of jobs to new worker
3. **Gradual Rollout**: Increase to 50%, then 100%
4. **Deprecation**: Remove NestJS clipping code after 2 weeks stable

## Monitoring

- CloudWatch metrics for processing time, success rate
- Logging for each pipeline stage
- Alerts on failure rate > 5%
- Memory/CPU monitoring for resource optimization

