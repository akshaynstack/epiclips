# ViewCreator Genesis - AI Clipping API Integration Guide

> **For AI Agents & Automated Systems**  
> Complete reference for integrating with the ViewCreator Genesis AI Clipping API.

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [API Overview](#api-overview)
3. [Authentication](#authentication)
4. [Core Endpoints](#core-endpoints)
5. [Request/Response Schemas](#requestresponse-schemas)
6. [Feature Reference](#feature-reference)
7. [Integration Patterns](#integration-patterns)
8. [Error Handling](#error-handling)
9. [Configuration Reference](#configuration-reference)
10. [Testing & Validation](#testing--validation)

---

## Quick Reference

### Base URL
```
Production: https://genesis.viewcreator.ai
Local:      http://localhost:8000
```

### Essential Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ai-clipping/jobs` | Submit a new clipping job |
| `GET` | `/ai-clipping/jobs/{job_id}` | Get job status/result |
| `GET` | `/ai-clipping/jobs` | List all jobs |
| `DELETE` | `/ai-clipping/jobs/{job_id}` | Cancel a job |
| `POST` | `/ai-clipping/metadata` | Get video metadata (for cost estimation) |
| `GET` | `/ai-clipping/caption-presets` | List caption styles |
| `GET` | `/ai-clipping/layout-presets` | List layout options |
| `GET` | `/health` | Health check |

### Minimal Job Submission
```json
POST /ai-clipping/jobs
{
  "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "max_clips": 5,
  "duration_ranges": ["medium"],
  "caption_preset": "viral_gold"
}
```

---

## API Overview

### What It Does
ViewCreator Genesis transforms long-form videos into viral short-form clips optimized for TikTok, YouTube Shorts, and Instagram Reels.

### Pipeline Stages
1. **Download** - Fetches video from YouTube, S3, or direct URL
2. **Transcribe** - Generates word-level transcript with timing
3. **Plan** - AI identifies viral-worthy segments using Gemini/OpenRouter
4. **Detect** - Analyzes faces, layouts, and content regions per frame
5. **Render** - Produces 9:16 vertical clips with captions

### Processing Time
- ~3-8 minutes per minute of source video
- Varies based on clip count and video complexity

---

## Authentication

### Header Authentication
```http
X-Genesis-API-Key: your-api-key-here
```

### Protected Endpoints (require auth)
- `POST /ai-clipping/jobs`
- `POST /ai-clipping/metadata`
- `DELETE /ai-clipping/jobs/{job_id}`

### Public Endpoints (no auth)
- `GET /ai-clipping/jobs`
- `GET /ai-clipping/jobs/{job_id}`
- `GET /ai-clipping/caption-presets`
- `GET /ai-clipping/layout-presets`
- `GET /health`

---

## Core Endpoints

### 1. Submit Job

**POST /ai-clipping/jobs**

Submit a video for AI clipping. Returns immediately with a job ID.

#### Request Body
```json
{
  "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "max_clips": 5,
  "auto_clip_count": true,
  "duration_ranges": ["short", "medium"],
  "target_platform": "tiktok",
  "include_captions": true,
  "caption_preset": "viral_gold",
  "start_time_seconds": 0,
  "end_time_seconds": 600,
  "callback_url": "https://your-server.com/webhook"
}
```

#### Response (202 Accepted)
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "accepted",
  "message": "Job queued for processing",
  "estimated_processing_minutes": 8
}
```

---

### 2. Get Job Status

**GET /ai-clipping/jobs/{job_id}**

Poll this endpoint to track progress and get results.

#### Response (Processing)
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress_percent": 45.5,
  "current_step": "Rendering clip 3/5",
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### Response (Completed)
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress_percent": 100,
  "output": {
    "source_video_title": "My Amazing Podcast",
    "source_video_duration_seconds": 1834.5,
    "total_clips": 5,
    "processing_time_seconds": 245.3,
    "clips": [
      {
        "clip_index": 1,
        "s3_url": "https://bucket.s3.amazonaws.com/clips/clip_01.mp4",
        "start_time_ms": 45000,
        "end_time_ms": 90000,
        "duration_ms": 45000,
        "virality_score": 0.85,
        "layout_type": "talking_head",
        "summary": "Speaker explains the key concept...",
        "tags": ["motivational", "insight", "quotable"]
      }
    ]
  }
}
```

#### Response (Failed)
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "failed",
  "error": "Video download failed: Video is private"
}
```

---

### 3. Get Video Metadata

**POST /ai-clipping/metadata**

Get video info before processing (useful for cost estimation).

#### Request
```json
{
  "video_url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

#### Response
```json
{
  "title": "My Amazing Podcast Episode",
  "duration_seconds": 1834.5,
  "duration_minutes": 30.58,
  "width": 1920,
  "height": 1080,
  "source_type": "youtube",
  "estimated_credits": 306
}
```

---

### 4. List Jobs

**GET /ai-clipping/jobs**

#### Query Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `status` | string | - | Filter by status: `processing`, `completed`, `failed` |
| `limit` | int | 20 | Max results (1-100) |

#### Response
```json
{
  "jobs": [
    {
      "job_id": "550e8400-...",
      "status": "completed",
      "progress_percent": 100,
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 42
}
```

---

## Request/Response Schemas

### Job Submit Request - Full Schema

```typescript
interface ClipJobSubmitRequest {
  // Video Source (one required)
  video_url?: string;      // YouTube, S3, or direct URL
  s3_key?: string;         // S3 key if using configured bucket

  // Clip Count
  max_clips?: number;      // 1-50, optional with auto_clip_count=true
  auto_clip_count?: boolean; // Default: true - scales with video duration

  // Duration Control
  duration_ranges?: ("short" | "medium" | "long")[];
  min_clip_duration_seconds?: number;  // Legacy, use duration_ranges
  max_clip_duration_seconds?: number;  // Legacy, use duration_ranges

  // Time Range Selection
  start_time_seconds?: number;  // Process from this time (0 = start)
  end_time_seconds?: number;    // Process until this time (null = end)

  // Output Settings
  target_platform?: "tiktok" | "youtube_shorts" | "instagram_reels";
  include_captions?: boolean;  // Default: true
  caption_preset?: string;     // See caption presets below
  layout_type?: "auto";        // Always "auto" (AI-powered detection)

  // Webhooks & Tracking
  callback_url?: string;       // Webhook for progress updates
  external_job_id?: string;    // Your system's job ID
  owner_user_id?: string;      // User ID for tracking
}
```

### Duration Ranges

| Range | Duration | Best For |
|-------|----------|----------|
| `short` | 15-30 seconds | Quick hooks, punchy moments |
| `medium` | 30-60 seconds | Standard viral clips |
| `long` | 60-120 seconds | In-depth segments |

### Caption Presets

| Preset ID | Name | Style |
|-----------|------|-------|
| `viral_gold` | Viral Gold | White text, gold highlight |
| `clean_white` | Clean White | White text, blue accent |
| `neon_pop` | Neon Pop | Cyan text, magenta highlight |
| `bold_boxed` | Bold Boxed | White text, red emphasis |
| `gradient_glow` | Gradient Glow | White text, green highlight |

### Job Status Values

| Status | Description |
|--------|-------------|
| `accepted` | Job received, queued |
| `processing` | Actively processing |
| `completed` | Finished successfully |
| `failed` | Error occurred |
| `cancelled` | User cancelled |

---

## Feature Reference

### 1. Auto Clip Count Scaling

When `auto_clip_count: true`, the system automatically determines optimal clip count based on video duration.

**Formula:**
```
suggested_clips = floor(video_duration_minutes × 0.5)
min_clips = 2
max_clips = min(suggested_clips, user_max_clips or 30)
```

**Examples:**
| Video Length | Auto Clips |
|--------------|------------|
| 5 minutes | 2 clips |
| 13 minutes | 6-7 clips |
| 30 minutes | 15 clips |
| 60 minutes | 30 clips |

### 2. Time Range Selection

Process only a specific portion of the video:

```json
{
  "video_url": "...",
  "start_time_seconds": 300,   // Start at 5:00
  "end_time_seconds": 900      // End at 15:00
}
```

**Validation:**
- `end_time_seconds` must be > `start_time_seconds`
- Minimum range: 60 seconds
- Clips will only be generated from within this range

### 3. Sentence Boundary Snapping

All clips automatically:
- **Start** at word boundaries (no mid-word cuts)
- **End** at sentence boundaries (no mid-sentence cuts)
- Include 150ms audio padding for smooth playback

### 4. Layout Detection

The AI automatically detects and switches between layouts:
- **Talking Head**: Face-focused dynamic crop for speakers
- **Split Screen**: Screen content top, face bottom (for tutorials)
- **Screen Share**: Full screen content when no face detected

Layout switching happens frame-by-frame within each clip.

---

## Integration Patterns

### Pattern 1: Simple Fire-and-Forget

```python
import requests

def submit_job(video_url: str) -> str:
    response = requests.post(
        "https://genesis.viewcreator.ai/ai-clipping/jobs",
        headers={"X-Genesis-API-Key": API_KEY},
        json={
            "video_url": video_url,
            "max_clips": 5,
            "duration_ranges": ["medium"],
            "caption_preset": "viral_gold",
            "callback_url": "https://your-server.com/genesis-webhook"
        }
    )
    return response.json()["job_id"]
```

### Pattern 2: Polling with Progress

```python
import time

def process_with_polling(video_url: str):
    # Submit
    job_id = submit_job(video_url)
    
    # Poll until complete
    while True:
        status = requests.get(
            f"https://genesis.viewcreator.ai/ai-clipping/jobs/{job_id}"
        ).json()
        
        if status["status"] == "completed":
            return status["output"]["clips"]
        elif status["status"] == "failed":
            raise Exception(status["error"])
        
        print(f"Progress: {status['progress_percent']}% - {status['current_step']}")
        time.sleep(5)
```

### Pattern 3: Webhook Integration

```python
from flask import Flask, request

app = Flask(__name__)

@app.route("/genesis-webhook", methods=["POST"])
def handle_webhook():
    data = request.json
    job_id = data["job_id"]
    status = data["status"]
    
    if status == "completed":
        clips = data["output"]["clips"]
        for clip in clips:
            download_clip(clip["s3_url"])
    elif status == "failed":
        log_error(data["error"])
    
    return {"received": True}
```

### Pattern 4: Batch Processing

```python
async def process_batch(video_urls: list[str]):
    # Submit all jobs
    job_ids = []
    for url in video_urls:
        job_id = submit_job(url)
        job_ids.append(job_id)
    
    # Poll all jobs
    results = {}
    pending = set(job_ids)
    
    while pending:
        for job_id in list(pending):
            status = get_status(job_id)
            if status["status"] in ("completed", "failed"):
                results[job_id] = status
                pending.remove(job_id)
        
        await asyncio.sleep(10)
    
    return results
```

---

## Error Handling

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 202 | Job accepted (async processing) |
| 400 | Bad request (invalid parameters) |
| 401 | Unauthorized (invalid/missing API key) |
| 404 | Job not found |
| 422 | Validation error |
| 429 | Rate limited |
| 500 | Server error |

### Common Errors

```json
// Invalid video URL
{
  "detail": "Invalid video URL format"
}

// Video unavailable
{
  "status": "failed",
  "error": "Video download failed: Video is unavailable"
}

// Time range validation
{
  "detail": "end_time_seconds (300) must be greater than start_time_seconds (600)"
}

// Missing required field
{
  "detail": "max_clips is required when auto_clip_count is false"
}
```

### Retry Strategy

```python
import time
from requests.exceptions import RequestException

def submit_with_retry(video_url: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{BASE_URL}/ai-clipping/jobs",
                headers={"X-Genesis-API-Key": API_KEY},
                json={"video_url": video_url},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt  # Exponential backoff
            time.sleep(wait)
```

---

## Configuration Reference

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GENESIS_API_KEY` | Yes | API authentication key |
| `OPENROUTER_API_KEY` | Yes | For AI clip planning (Gemini) |
| `GROQ_API_KEY` | Yes | For transcription (Whisper) |
| `AWS_ACCESS_KEY_ID` | Yes | S3 access |
| `AWS_SECRET_ACCESS_KEY` | Yes | S3 secret |
| `S3_BUCKET` | Yes | Output bucket name |
| `AWS_REGION` | No | Default: us-east-1 |

### Tunable Parameters (Backend Config)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clips_per_minute_ratio` | 0.5 | Clips per minute of video (0.5 = 1 clip per 2 min) |
| `min_clips` | 2 | Minimum clips regardless of video length |
| `max_clips_absolute` | 30 | Hard cap on clip count |
| `sentence_extension_max_seconds` | 5.0 | Max seconds to extend clip for sentence boundary |
| `audio_padding_ms` | 150 | Audio buffer at clip boundaries |

---

## Testing & Validation

### Test Script Usage

```bash
# Quick test (2 short clips)
python test_job.py --quick

# Standard test with auto scaling
python test_job.py

# Custom configuration
python test_job.py --video "URL" --max-clips 5 --duration medium

# Time range selection
python test_job.py --start-time 300 --end-time 600

# Full test suite
python test_job.py --full-suite

# Check API health
python test_job.py --health-only

# Analyze previous job
python test_job.py --analyze-only JOB_ID
```

### Validation Checklist

Before deploying integration:

- [ ] API key authentication works
- [ ] Job submission returns 202 with job_id
- [ ] Polling shows progress updates
- [ ] Completed jobs include S3 URLs
- [ ] S3 URLs are accessible
- [ ] Webhook receives callbacks (if configured)
- [ ] Error responses are handled gracefully
- [ ] Rate limiting is respected

### Expected Processing Times

| Video Length | Short Clips | Medium Clips | Long Clips |
|--------------|-------------|--------------|------------|
| 5 min | ~2 min | ~3 min | ~4 min |
| 15 min | ~5 min | ~7 min | ~10 min |
| 30 min | ~10 min | ~15 min | ~20 min |
| 60 min | ~20 min | ~30 min | ~40 min |

---

## Changelog

### v1.1.0 (December 2024)
- Increased clip count ratio: 0.2 → 0.5 (1 clip per 2 min instead of 5 min)
- Improved sentence boundary snapping with 150ms audio padding
- Added comprehensive test suite with multiple scenarios
- Enhanced logging for debugging

### v1.0.0 (Initial Release)
- Full AI clipping pipeline
- YouTube, S3, and direct URL support
- Multiple caption presets
- Auto layout detection
- Time range selection
- Webhook callbacks

---

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Use `--analyze-only JOB_ID` to inspect job details
3. Verify environment variables are set correctly
4. Check API health with `GET /health`
