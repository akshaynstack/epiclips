# ViewCreator Genesis API Usage Guide

Complete documentation for interacting with the ViewCreator Genesis FastAPI application.

## Table of Contents

- [Quick Start](#quick-start)
- [Authentication](#authentication)
- [Base URL](#base-url)
- [AI Clipping API](#ai-clipping-api)
  - [From YouTube Videos](#from-youtube-videos)
  - [From Direct Video URLs](#from-direct-video-urls)
  - [From S3 Storage](#from-s3-storage)
  - [Video Metadata (Cost Estimation)](#video-metadata-cost-estimation)
  - [Caption Presets](#caption-presets)
  - [Layout Presets](#layout-presets)
  - [Job Management](#job-management)
- [Detection API](#detection-api)
- [Health Endpoints](#health-endpoints)
- [Response Schemas](#response-schemas)
- [Error Handling](#error-handling)
- [Code Examples](#code-examples)
- [Rate Limits & Best Practices](#rate-limits--best-practices)

---

## Quick Start

python test_job.py --max-clips 3 --duration short --version _v16 --video "https://www.youtube.com/watch?v=w1wNajAY3Ho"

```bash
# 1. Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 2. Submit a YouTube clipping job
curl -X POST http://localhost:8000/ai-clipping/jobs \
  -H "Content-Type: application/json" \
  -H "X-Genesis-API-Key: your-api-key" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "max_clips": 3,
    "duration_ranges": ["short", "medium"],
    "include_captions": true,
    "caption_preset": "viral_gold"
  }'

# 3. Check job status
curl http://localhost:8000/ai-clipping/jobs/{job_id}
```

---

## Authentication

Genesis uses API key authentication via the `X-Genesis-API-Key` header.

### Configuration

Set the `GENESIS_API_KEY` environment variable to enable authentication:

```bash
export GENESIS_API_KEY=your-secure-api-key
```

### Making Authenticated Requests

```bash
curl -X POST http://localhost:8000/ai-clipping/jobs \
  -H "Content-Type: application/json" \
  -H "X-Genesis-API-Key: your-api-key" \
  -d '{"video_url": "..."}'
```

### Development Mode

If `GENESIS_API_KEY` is not configured, authentication is skipped (development mode).

**Protected Endpoints:**
- `POST /ai-clipping/jobs` - Submit clipping jobs
- `POST /ai-clipping/metadata` - Get video metadata
- `DELETE /ai-clipping/jobs/{job_id}` - Cancel jobs

**Public Endpoints (no auth required):**
- `GET /ai-clipping/jobs` - List jobs
- `GET /ai-clipping/jobs/{job_id}` - Get job status
- `GET /ai-clipping/caption-presets` - List caption presets
- `GET /ai-clipping/layout-presets` - List layout presets
- All `/health` endpoints
- All `/detect` endpoints

---

## Base URL

| Environment | Base URL |
|-------------|----------|
| Local Development | `http://localhost:8000` |
| Docker | `http://localhost:8000` |
| Production | `https://genesis.viewcreator.ai` |

**Interactive API Documentation:**
- Swagger UI: `{BASE_URL}/docs`
- ReDoc: `{BASE_URL}/redoc`

---

## AI Clipping API

The AI Clipping API transforms long-form videos into viral short-form clips.

### From YouTube Videos

**Endpoint:** `POST /ai-clipping/jobs`

```bash
curl -X POST http://localhost:8000/ai-clipping/jobs \
  -H "Content-Type: application/json" \
  -H "X-Genesis-API-Key: your-api-key" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "max_clips": 5,
    "duration_ranges": ["short", "medium"],
    "target_platform": "tiktok",
    "include_captions": true,
    "caption_preset": "viral_gold",
    "layout_type": "auto"
  }'
```

**Supported YouTube URL Formats:**
- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://youtube.com/watch?v=VIDEO_ID`
- `https://www.youtube.com/shorts/VIDEO_ID`

### From Direct Video URLs

**Endpoint:** `POST /ai-clipping/jobs`

```bash
curl -X POST http://localhost:8000/ai-clipping/jobs \
  -H "Content-Type: application/json" \
  -H "X-Genesis-API-Key: your-api-key" \
  -d '{
    "video_url": "https://example.com/videos/my-video.mp4",
    "max_clips": 3,
    "duration_ranges": ["medium"],
    "include_captions": true
  }'
```

**Supported Formats:** MP4, MOV, MKV, WEBM, AVI

### From S3 Storage

Genesis supports multiple S3 URL formats:

#### Using S3 URL

```bash
curl -X POST http://localhost:8000/ai-clipping/jobs \
  -H "Content-Type: application/json" \
  -H "X-Genesis-API-Key: your-api-key" \
  -d '{
    "video_url": "s3://my-bucket/videos/source.mp4",
    "max_clips": 5
  }'
```

#### Using S3 HTTPS URL

```bash
curl -X POST http://localhost:8000/ai-clipping/jobs \
  -H "Content-Type: application/json" \
  -H "X-Genesis-API-Key: your-api-key" \
  -d '{
    "video_url": "https://my-bucket.s3.us-east-1.amazonaws.com/videos/source.mp4",
    "max_clips": 5
  }'
```

#### Using S3 Key (Configured Bucket)

```bash
curl -X POST http://localhost:8000/ai-clipping/jobs \
  -H "Content-Type: application/json" \
  -H "X-Genesis-API-Key: your-api-key" \
  -d '{
    "s3_key": "users/user123/uploads/video.mp4",
    "max_clips": 5
  }'
```

### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `video_url` | string | Yes* | - | Video URL (YouTube, S3, or direct) |
| `s3_key` | string | Yes* | - | S3 key if using configured bucket |
| `max_clips` | int | No | 5 | Maximum clips to generate (1-20) |
| `duration_ranges` | array | No | - | Duration preferences: `"short"`, `"medium"`, `"long"` |
| `min_clip_duration_seconds` | int | No | 15 | Legacy: minimum duration (5-120) |
| `max_clip_duration_seconds` | int | No | 90 | Legacy: maximum duration (15-300) |
| `target_platform` | string | No | `"tiktok"` | Target: `"tiktok"`, `"youtube_shorts"`, `"instagram_reels"` |
| `include_captions` | bool | No | true | Include viral-style captions |
| `caption_preset` | string | No | - | Preset ID (takes precedence over custom style) |
| `caption_style` | object | No | - | Custom caption configuration |
| `layout_type` | string | No | `"auto"` | Layout: `"auto"`, `"split_screen"`, `"talking_head"` |
| `callback_url` | string | No | - | Webhook URL for progress updates |
| `external_job_id` | string | No | - | External job ID for tracking |
| `owner_user_id` | string | No | - | User ID for S3 key scoping |

*Either `video_url` or `s3_key` must be provided.

### Duration Ranges

| Range | Duration | Description |
|-------|----------|-------------|
| `short` | 15-30 seconds | Quick hooks, punchy content |
| `medium` | 30-60 seconds | Standard viral clips |
| `long` | 60-120 seconds | In-depth segments |

```json
{
  "duration_ranges": ["short", "medium"]
}
```

### Layout Types

| Type | Description |
|------|-------------|
| `auto` | AI-powered dynamic detection - switches layouts mid-clip |
| `split_screen` | Screen content top (50%), face bottom (50%) |
| `talking_head` | Dynamic face-focused crop following the speaker |

### Submit Response

```json
{
  "job_id": "e2ee893e-0bcd-47cb-9173-578388a96a78",
  "status": "accepted",
  "message": "Job queued for processing",
  "estimated_processing_minutes": 8
}
```

---

## Video Metadata (Cost Estimation)

Get video metadata before submitting a job to estimate processing costs.

**Endpoint:** `POST /ai-clipping/metadata`

```bash
curl -X POST http://localhost:8000/ai-clipping/metadata \
  -H "Content-Type: application/json" \
  -H "X-Genesis-API-Key: your-api-key" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=VIDEO_ID"
  }'
```

**Response:**

```json
{
  "title": "My Amazing Video",
  "duration_seconds": 1834.5,
  "duration_minutes": 30.58,
  "width": 1920,
  "height": 1080,
  "source_type": "youtube",
  "estimated_credits": 306
}
```

**Pricing:** 10 credits per minute of source video (minimum 10 credits).

---

## Caption Presets

### List Available Presets

**Endpoint:** `GET /ai-clipping/caption-presets`

```bash
curl http://localhost:8000/ai-clipping/caption-presets
```

**Response:**

```json
[
  {
    "id": "viral_gold",
    "name": "Viral Gold",
    "description": "Classic viral caption style with gold highlight",
    "preview_colors": {
      "primary": "#FFFFFF",
      "highlight": "#FFD700"
    }
  },
  {
    "id": "clean_white",
    "name": "Clean White",
    "description": "Clean, professional look with blue accent",
    "preview_colors": {
      "primary": "#FFFFFF",
      "highlight": "#3B82F6"
    }
  },
  {
    "id": "neon_pop",
    "name": "Neon Pop",
    "description": "Bold, eye-catching neon style",
    "preview_colors": {
      "primary": "#00FFFF",
      "highlight": "#FF00FF"
    }
  },
  {
    "id": "bold_boxed",
    "name": "Bold Boxed",
    "description": "High contrast red emphasis",
    "preview_colors": {
      "primary": "#FFFFFF",
      "highlight": "#EF4444"
    }
  },
  {
    "id": "gradient_glow",
    "name": "Gradient Glow",
    "description": "Fresh green highlight",
    "preview_colors": {
      "primary": "#FFFFFF",
      "highlight": "#22C55E"
    }
  }
]
```

### Using a Caption Preset

```json
{
  "video_url": "https://youtube.com/watch?v=...",
  "caption_preset": "viral_gold"
}
```

### Custom Caption Style

If `caption_preset` is not provided, you can use custom styling:

```json
{
  "video_url": "https://youtube.com/watch?v=...",
  "caption_style": {
    "font_family": "Arial Black",
    "font_size": 72,
    "primary_color": "FFFFFF",
    "highlight_color": "00FF00",
    "outline_color": "000000",
    "outline_width": 4,
    "position": "center",
    "alignment": "center",
    "words_per_group": 3
  }
}
```

---

## Layout Presets

### List Available Layouts

**Endpoint:** `GET /ai-clipping/layout-presets`

```bash
curl http://localhost:8000/ai-clipping/layout-presets
```

**Response:**

```json
[
  {
    "id": "split_screen",
    "name": "Split Screen",
    "description": "Screen content on top, face close-up on bottom (50/50 split)",
    "icon": "layout-split",
    "preview_layout": {
      "top": "screen",
      "bottom": "face"
    }
  },
  {
    "id": "talking_head",
    "name": "Talking Head",
    "description": "Dynamic face-focused crop that follows the speaker",
    "icon": "user",
    "preview_layout": {
      "type": "single",
      "focus": "face"
    }
  }
]
```

---

## Job Management

### Get Job Status

**Endpoint:** `GET /ai-clipping/jobs/{job_id}`

```bash
curl http://localhost:8000/ai-clipping/jobs/e2ee893e-0bcd-47cb-9173-578388a96a78
```

**Response (Processing):**

```json
{
  "job_id": "e2ee893e-0bcd-47cb-9173-578388a96a78",
  "status": "processing",
  "progress_percent": 45.5,
  "current_step": "Analyzing content with AI...",
  "clips_completed": 0,
  "total_clips": 5,
  "error": null,
  "output": null
}
```

**Response (Completed):**

```json
{
  "job_id": "e2ee893e-0bcd-47cb-9173-578388a96a78",
  "status": "completed",
  "progress_percent": 100,
  "current_step": "Completed",
  "clips_completed": 5,
  "total_clips": 5,
  "error": null,
  "output": {
    "job_id": "e2ee893e-0bcd-47cb-9173-578388a96a78",
    "source_video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "source_video_title": "My Video Title",
    "source_video_duration_seconds": 1834.5,
    "total_clips": 5,
    "clips": [
      {
        "clip_index": 0,
        "s3_url": "https://bucket.s3.amazonaws.com/clips/user/job/clip_0.mp4",
        "duration_ms": 45000,
        "start_time_ms": 134200,
        "end_time_ms": 179200,
        "virality_score": 0.92,
        "layout_type": "split_screen",
        "summary": "Key insight about the main topic...",
        "tags": ["technology", "AI", "tutorial"]
      }
    ],
    "transcript_url": "https://bucket.s3.amazonaws.com/clips/.../transcript.json",
    "plan_url": "https://bucket.s3.amazonaws.com/clips/.../plan.json",
    "processing_time_seconds": 182.5,
    "created_at": "2025-12-03T10:30:00.000Z"
  }
}
```

### Job Status Values

| Status | Description |
|--------|-------------|
| `pending` | Job queued, not yet started |
| `downloading` | Downloading source video |
| `transcribing` | Transcribing audio |
| `planning` | AI analyzing for viral segments |
| `detecting` | Running face/pose detection |
| `rendering` | Rendering clips |
| `uploading` | Uploading to S3 |
| `completed` | Successfully finished |
| `failed` | Job failed with error |

### List All Jobs

**Endpoint:** `GET /ai-clipping/jobs`

```bash
# List all jobs
curl http://localhost:8000/ai-clipping/jobs

# Filter by status
curl "http://localhost:8000/ai-clipping/jobs?status_filter=completed"

# Limit results
curl "http://localhost:8000/ai-clipping/jobs?limit=10"
```

### Cancel a Job

**Endpoint:** `DELETE /ai-clipping/jobs/{job_id}`

```bash
curl -X DELETE http://localhost:8000/ai-clipping/jobs/e2ee893e-0bcd-47cb-9173-578388a96a78 \
  -H "X-Genesis-API-Key: your-api-key"
```

Returns `204 No Content` on success.

---

## Detection API

The Detection API runs computer vision analysis on videos stored in S3.

### Run Detection

**Endpoint:** `POST /detect`

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "video_s3_key": "users/user123/videos/source.mp4",
    "frame_interval_seconds": 2.0,
    "detect_faces": true,
    "detect_poses": true,
    "start_time_ms": 0,
    "end_time_ms": 60000
  }'
```

### Detection Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `job_id` | string | Yes | - | Unique job identifier |
| `video_s3_key` | string | Yes | - | S3 key of the video |
| `frame_interval_seconds` | float | No | 2.0 | Interval between frames (0.5-10.0) |
| `detect_faces` | bool | No | true | Run MediaPipe face detection |
| `detect_poses` | bool | No | true | Run MediaPipe pose estimation |
| `callback_url` | string | No | - | URL to POST results when complete |
| `start_time_ms` | int | No | - | Start time in milliseconds |
| `end_time_ms` | int | No | - | End time in milliseconds |

### Detection Response

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "source_dimensions": {
    "width": 1920,
    "height": 1080
  },
  "frame_interval_ms": 2000,
  "frames": [
    {
      "index": 0,
      "timestamp_ms": 0,
      "faces": [
        {
          "track_id": 1,
          "bbox": {"x": 423, "y": 156, "width": 187, "height": 234},
          "confidence": 0.94
        }
      ],
      "poses": [
        {
          "track_id": 1,
          "keypoints": {
            "nose": [0.45, 0.22],
            "left_shoulder": [0.38, 0.35],
            "right_shoulder": [0.52, 0.35]
          },
          "confidence": 0.87,
          "gesture": null
        }
      ]
    }
  ],
  "tracks": [
    {
      "track_id": 1,
      "track_type": "face",
      "first_frame": 0,
      "last_frame": 29,
      "frame_count": 30,
      "avg_bbox": {"x": 450, "y": 160, "width": 190, "height": 240},
      "avg_confidence": 0.91
    }
  ],
  "summary": {
    "total_frames": 30,
    "faces_detected": 28,
    "poses_detected": 30,
    "unique_face_tracks": 1,
    "unique_pose_tracks": 1,
    "processing_time_ms": 4523
  }
}
```

### Async Detection with Callback

For long videos, use callback mode:

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "my-job-123",
    "video_s3_key": "videos/long-video.mp4",
    "callback_url": "https://my-server.com/webhooks/detection"
  }'
```

Returns immediately with `status: "queued"`. Results are POSTed to the callback URL.

### Queue Status

**Endpoint:** `GET /detect/status`

```bash
curl http://localhost:8000/detect/status
```

```json
{
  "max_concurrent_jobs": 2,
  "queued_jobs": 3,
  "active_jobs": 2,
  "jobs": {
    "job-1": "processing",
    "job-2": "processing",
    "job-3": "queued"
  }
}
```

---

## Health Endpoints

### Basic Health Check

**Endpoint:** `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### Readiness Check

**Endpoint:** `GET /health/ready`

```bash
curl http://localhost:8000/health/ready
```

```json
{
  "ready": true,
  "models_loaded": true,
  "face_detector": "ready",
  "pose_estimator": "ready"
}
```

### Model Status

**Endpoint:** `GET /health/models`

```bash
curl http://localhost:8000/health/models
```

```json
{
  "models": {
    "face_detector": {
      "loaded": true,
      "ready": true,
      "model_type": "MediaPipe + Haar Cascade"
    },
    "pose_estimator": {
      "loaded": true,
      "ready": true,
      "model_type": "MediaPipe Pose"
    }
  }
}
```

### AI Clipping Health

**Endpoint:** `GET /ai-clipping/health`

```bash
curl http://localhost:8000/ai-clipping/health
```

```json
{
  "status": "healthy",
  "services": {
    "detection_pipeline": "healthy",
    "video_downloader": "healthy",
    "transcription": "healthy",
    "intelligence_planner": "healthy",
    "rendering": "healthy",
    "s3_upload": "healthy"
  },
  "timestamp": "2025-12-03T10:30:00.000000"
}
```

---

## Response Schemas

### Clip Artifact

```typescript
interface ClipArtifact {
  clip_index: number;        // 0-based index
  s3_url: string;            // S3 URL to the clip
  duration_ms: number;       // Clip duration in milliseconds
  start_time_ms: number;     // Start time in source video
  end_time_ms: number;       // End time in source video
  virality_score: number;    // 0.0 to 1.0
  layout_type: string;       // "split_screen" or "talking_head"
  summary: string | null;    // Why this clip is viral-worthy
  tags: string[];            // Content categories
}
```

### Job Output

```typescript
interface JobOutput {
  job_id: string;
  source_video_url: string;
  source_video_title: string;
  source_video_duration_seconds: number;
  total_clips: number;
  clips: ClipArtifact[];
  transcript_url: string | null;
  plan_url: string | null;
  processing_time_seconds: number;
  created_at: string;        // ISO 8601 timestamp
}
```

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common HTTP Status Codes

| Status | Description |
|--------|-------------|
| `200 OK` | Request successful |
| `202 Accepted` | Job accepted for processing |
| `204 No Content` | Successful deletion |
| `400 Bad Request` | Invalid request parameters |
| `401 Unauthorized` | Missing or invalid API key |
| `404 Not Found` | Resource not found |
| `500 Internal Server Error` | Server error |
| `503 Service Unavailable` | Service not ready (models loading) |

### Error Examples

**Missing Required Field:**
```json
{
  "detail": "Either video_url or s3_key must be provided"
}
```

**Invalid Platform:**
```json
{
  "detail": "Invalid platform. Must be one of: ['tiktok', 'youtube_shorts', 'instagram_reels']"
}
```

**Invalid Caption Preset:**
```json
{
  "detail": "Unknown caption preset: invalid_preset. Available: viral_gold, clean_white, neon_pop, bold_boxed, gradient_glow"
}
```

**Job Not Found:**
```json
{
  "detail": "Job not found: invalid-job-id"
}
```

---

## Code Examples

### Python (requests)

```python
import requests
import time

BASE_URL = "http://localhost:8000"
API_KEY = "your-api-key"

headers = {
    "Content-Type": "application/json",
    "X-Genesis-API-Key": API_KEY
}

# Submit a job
response = requests.post(
    f"{BASE_URL}/ai-clipping/jobs",
    headers=headers,
    json={
        "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "max_clips": 3,
        "duration_ranges": ["short", "medium"],
        "caption_preset": "viral_gold",
        "include_captions": True
    }
)

job_data = response.json()
job_id = job_data["job_id"]
print(f"Job submitted: {job_id}")

# Poll for completion
while True:
    status_response = requests.get(f"{BASE_URL}/ai-clipping/jobs/{job_id}")
    status = status_response.json()
    
    print(f"Status: {status['status']} - {status['progress_percent']:.1f}%")
    
    if status["status"] == "completed":
        print(f"Completed! Generated {status['output']['total_clips']} clips")
        for clip in status["output"]["clips"]:
            print(f"  Clip {clip['clip_index']}: {clip['s3_url']}")
        break
    elif status["status"] == "failed":
        print(f"Failed: {status['error']}")
        break
    
    time.sleep(5)
```

### JavaScript (fetch)

```javascript
const BASE_URL = "http://localhost:8000";
const API_KEY = "your-api-key";

async function createClips(youtubeUrl) {
  // Submit job
  const submitResponse = await fetch(`${BASE_URL}/ai-clipping/jobs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Genesis-API-Key": API_KEY
    },
    body: JSON.stringify({
      video_url: youtubeUrl,
      max_clips: 5,
      duration_ranges: ["short", "medium"],
      caption_preset: "viral_gold",
      include_captions: true
    })
  });

  const { job_id } = await submitResponse.json();
  console.log(`Job submitted: ${job_id}`);

  // Poll for completion
  while (true) {
    const statusResponse = await fetch(`${BASE_URL}/ai-clipping/jobs/${job_id}`);
    const status = await statusResponse.json();

    console.log(`Status: ${status.status} - ${status.progress_percent.toFixed(1)}%`);

    if (status.status === "completed") {
      console.log(`Generated ${status.output.total_clips} clips:`);
      status.output.clips.forEach(clip => {
        console.log(`  Clip ${clip.clip_index}: ${clip.s3_url}`);
      });
      return status.output;
    }

    if (status.status === "failed") {
      throw new Error(status.error);
    }

    await new Promise(resolve => setTimeout(resolve, 5000));
  }
}

// Usage
createClips("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
  .then(output => console.log("Done!", output))
  .catch(err => console.error("Error:", err));
```

### TypeScript with Webhook

```typescript
interface ClipJob {
  job_id: string;
  status: string;
  message: string;
  estimated_processing_minutes: number;
}

interface ClipResult {
  job_id: string;
  status: string;
  output?: {
    clips: Array<{
      clip_index: number;
      s3_url: string;
      virality_score: number;
    }>;
  };
}

async function submitJobWithWebhook(
  videoUrl: string,
  callbackUrl: string
): Promise<ClipJob> {
  const response = await fetch("http://localhost:8000/ai-clipping/jobs", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Genesis-API-Key": process.env.GENESIS_API_KEY!
    },
    body: JSON.stringify({
      video_url: videoUrl,
      max_clips: 5,
      duration_ranges: ["medium"],
      caption_preset: "viral_gold",
      callback_url: callbackUrl,
      external_job_id: crypto.randomUUID(),
      owner_user_id: "user-123"
    })
  });

  return response.json();
}
```

---

## Rate Limits & Best Practices

### Concurrency

- Default: 2 concurrent jobs
- Jobs exceeding the limit are queued automatically
- Check queue status: `GET /detect/status`

### Best Practices

1. **Use Webhooks for Production**
   - Pass `callback_url` instead of polling
   - Reduces API calls and latency

2. **Choose Appropriate Duration Ranges**
   - `short` for hooks and teasers
   - `medium` for standard viral content
   - `long` for tutorials and explanations

3. **Use Caption Presets**
   - Consistent branding across clips
   - Tested for readability

4. **Handle Errors Gracefully**
   - Check for `failed` status
   - Implement exponential backoff for retries

5. **Optimize for Your Platform**
   - Set `target_platform` appropriately
   - Different platforms have different optimal lengths

### Performance Tips

| Video Length | Estimated Processing Time | Recommended max_clips |
|--------------|---------------------------|----------------------|
| < 10 min | 2-3 minutes | 3-5 |
| 10-30 min | 3-5 minutes | 5-10 |
| 30-60 min | 5-8 minutes | 5-10 |
| > 60 min | 8-15 minutes | 10-15 |

---

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `GENESIS_API_KEY` | No | API key for authentication (skip if not set) |
| `GROQ_API_KEY` | Yes | Groq API key for transcription |
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key for AI planning |
| `AWS_REGION` | No | AWS region (default: us-east-1) |
| `AWS_ACCESS_KEY_ID` | No* | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | No* | AWS credentials |
| `S3_BUCKET` | No | S3 bucket name (default: viewcreator-media) |
| `MAX_WORKERS` | No | Max concurrent jobs (default: 4, set to vCPU count) |
| `MAX_RENDER_WORKERS` | No | Max concurrent FFmpeg renders (default: 3) |

*AWS credentials are optional when using IAM roles (ECS, EC2).

---

## Performance Optimizations

### Parallel Processing

Genesis uses parallel processing throughout the pipeline for maximum speed:

| Stage | Parallelization | Controlled By |
|-------|-----------------|---------------|
| Detection | Parallel per-clip | Runs in thread pool to avoid blocking |
| Rendering | Parallel FFmpeg | `MAX_RENDER_WORKERS` (default: 3) |
| S3 Upload | Parallel uploads | All clips uploaded concurrently |

### AWS ECS Recommendations

For optimal performance on AWS ECS:

```bash
# For t3.xlarge (4 vCPU, 16GB RAM)
MAX_WORKERS=4
MAX_RENDER_WORKERS=3

# For t3.2xlarge (8 vCPU, 32GB RAM)
MAX_WORKERS=8
MAX_RENDER_WORKERS=4

# For c6i.xlarge (4 vCPU, 8GB RAM) - compute optimized
MAX_WORKERS=4
MAX_RENDER_WORKERS=3
```

### Face Detection

Face detection uses MediaPipe with Haar Cascade fallback:
- **Primary**: MediaPipe short-range (best for close faces)
- **Fallback 1**: MediaPipe full-range (better for small/distant faces)
- **Fallback 2**: Haar Cascade (classical CV fallback)

YOLO has been removed to reduce startup time and memory usage (~200MB savings).

---

## Support

- **API Documentation:** `http://localhost:8000/docs`
- **GitHub Issues:** [viewcreator-genesis/issues](https://github.com/view-creator/viewcreator-genesis/issues)
- **Email:** support@viewcreator.ai
