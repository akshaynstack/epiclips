# ViewCreator Clipping Worker

A production-ready FastAPI video processing microservice that transforms long-form YouTube videos into viral-optimized short-form clips using AI-powered analysis, intelligent cropping, and automated caption generation.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Pipeline Stages](#pipeline-stages)
- [Service Components](#service-components)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Integration](#integration)
- [Development](#development)

---

## Overview

The ViewCreator Clipping Worker is a specialized microservice that handles the computationally intensive work of:

1. **Downloading** videos from YouTube or S3
2. **Transcribing** audio with word-level timestamps using Whisper
3. **Analyzing** content with AI to identify viral-worthy segments
4. **Detecting** faces and poses for intelligent cropping
5. **Rendering** 9:16 portrait clips with dynamic face tracking and captions
6. **Uploading** finished clips to S3 with metadata

### Key Features

| Feature | Technology | Speed/Performance |
|---------|------------|-------------------|
| Video Download | yt-dlp | Up to 50 MB/s |
| Transcription | Groq Whisper | 216x realtime |
| AI Planning | Gemini 2.5 Flash | ~2s per analysis |
| Face Detection | YOLOv8n | 60+ FPS |
| Pose Estimation | MediaPipe | 30+ FPS |
| Object Tracking | DeepSORT | Persistent IDs |
| Video Rendering | FFmpeg H.264 | Hardware accel |
| Caption Style | ASS Subtitles | Word-by-word highlight |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      ViewCreator AI Clipping Architecture                        │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Creator UI      │────▶│  NestJS API      │────▶│  Clipping Worker │
│  (Next.js)       │     │  (viewcreator-   │     │  (FastAPI)       │
│                  │◀────│   api)           │◀────│                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                │                          │
                                │                          │
                                ▼                          ▼
                         ┌──────────────┐          ┌──────────────┐
                         │   Postgres   │          │     AWS      │
                         │   (Jobs DB)  │          │  S3 Bucket   │
                         └──────────────┘          └──────────────┘

Worker Internal Architecture:
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AI Clipping Pipeline                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
│  │   Video     │    │   Audio     │    │ Intelligence│    │  Detection  │       │
│  │  Downloader │───▶│ Transcriber │───▶│   Planner   │───▶│  Pipeline   │       │
│  │  (yt-dlp)   │    │  (Whisper)  │    │  (Gemini)   │    │ (YOLO/MP)   │       │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘       │
│                                                                   │              │
│                                                                   ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
│  │   S3       │◀───│   Rendering  │◀───│   Content   │◀───│   Object    │       │
│  │  Upload    │    │   Service    │    │   Region    │    │   Tracker   │       │
│  │  Service   │    │  (FFmpeg)    │    │   Detector  │    │  (DeepSORT) │       │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘       │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages

### Stage 1: Video Download (`VideoDownloaderService`)

Downloads the source video using yt-dlp with automatic format selection.

```
Input:  YouTube URL, S3 URL, or direct video URL
Output: Local video file + VideoMetadata
```

**Capabilities:**
- YouTube videos (any length up to 2 hours)
- S3 bucket URLs (s3:// or https://)
- Direct HTTP/HTTPS video URLs
- Automatic best quality selection (720p-1080p preferred)
- Metadata extraction (title, duration, dimensions)

**Code Flow:**
```python
# video_downloader.py
async def download(video_url: str, work_dir: str) -> DownloadResult:
    source_type = self._detect_source_type(video_url)  # youtube, s3, direct
    
    if source_type == "youtube":
        return await self._download_youtube(video_url, work_dir)
    elif source_type == "s3":
        return await self._download_s3(video_url, work_dir)
    else:
        return await self._download_direct(video_url, work_dir)
```

---

### Stage 2: Audio Transcription (`TranscriptionService`)

Extracts audio and transcribes using Groq's Whisper API (216x realtime speed).

```
Input:  Video file path
Output: TranscriptionResult (segments with word-level timestamps)
```

**Data Structures:**
```python
@dataclass
class TranscriptWord:
    word: str
    start_time_ms: int
    end_time_ms: int

@dataclass
class TranscriptSegment:
    start_time_ms: int
    end_time_ms: int
    text: str
    words: list[TranscriptWord]  # Word-level timing for captions

@dataclass
class TranscriptionResult:
    segments: list[TranscriptSegment]
    full_text: str
    language: Optional[str]
    duration_seconds: float
```

**Process:**
1. Extract audio from video using FFmpeg → MP3
2. Send to Groq Whisper API with `timestamp_granularities=["word", "segment"]`
3. Parse response into structured segments with word timings
4. Return for use in planning and caption generation

---

### Stage 3: Intelligence Planning (`IntelligencePlannerService`)

Uses Gemini via OpenRouter to analyze the transcript and identify viral-worthy segments.

```
Input:  TranscriptionResult + optional vision frames
Output: ClipPlanResponse (list of segments with virality scores)
```

**AI Prompt Strategy:**
```
You are AI-Clipping-Agent, a virality analyst that identifies the most engaging 
segments from long-form videos for short-form content.

Your task:
1. Analyze the provided transcript to understand the content
2. Identify moments with high viral potential:
   - Strong hooks (first 3 seconds are critical)
   - Emotional peaks
   - Surprising reveals
   - Compelling storytelling moments
3. Assign layout types:
   - "talking_head": Speaker facing camera, focus on face
   - "screen_share": Screen content with webcam overlay

Return exactly {clip_count} clips as JSON with:
- start_time, end_time (in seconds)
- virality_score (0.0 to 1.0)
- layout_type ("talking_head" or "screen_share")
- summary (why this clip is viral-worthy)
- tags (content categories)
```

**Duration Range Support:**
Users can select preferred clip durations:
- `short`: 15-30 seconds
- `medium`: 30-60 seconds  
- `long`: 60-120 seconds

The AI respects these preferences when selecting clip boundaries.

---

### Stage 4: Detection Pipeline (`DetectionPipeline`)

Runs computer vision analysis on the planned clip segments.

```
Input:  Video path + clip segments
Output: Detection frames with face/pose data per timestamp
```

**Components:**

| Component | Model | Purpose |
|-----------|-------|---------|
| `FaceDetector` | YOLOv8n | Bounding boxes + confidence |
| `PoseEstimator` | MediaPipe | 33 body keypoints |
| `ObjectTracker` | DeepSORT | Persistent track IDs |
| `FrameExtractor` | FFmpeg | Extract frames at intervals |

**Detection Flow:**
```python
async def process_local_video(
    video_path: str,
    start_time_ms: int,
    end_time_ms: int,
    frame_interval_seconds: float = 0.5,  # 2 FPS for detection
) -> dict:
    frames = extract_frames(video_path, start_time_ms, end_time_ms)
    
    for frame in frames:
        faces = face_detector.detect(frame)
        poses = pose_estimator.estimate(frame)
        
        # Track objects across frames for consistent IDs
        tracked_faces = tracker.update_faces(faces, frame)
        tracked_poses = tracker.update_poses(poses, frame)
        
        frame_detections.append({
            "timestamp_sec": timestamp,
            "faces": tracked_faces,
            "poses": tracked_poses,
        })
    
    return {"frames": frame_detections}
```

---

### Stage 5: Content Region Detection (`ContentRegionDetector`)

Intelligently analyzes frames to detect screen content vs webcam overlays.

```
Input:  Detection frames + face data
Output: FrameAnalysis (webcam position, screen content center, layout type)
```

**Detection Logic:**
```python
def analyze_frame(frame, face_detections) -> FrameAnalysis:
    # 1. Find primary face (largest/most confident)
    primary_face_center = find_best_face(face_detections)
    
    # 2. Detect webcam overlay (small face in corner)
    webcam_region = detect_webcam_overlay(face_detections)
    
    # 3. Calculate screen content center (avoiding webcam)
    screen_content_center = calculate_screen_center(webcam_region)
    
    # 4. Determine layout type
    is_screen_share = determine_layout_type(face_detections, webcam_region)
    
    return FrameAnalysis(
        primary_face_center=primary_face_center,
        webcam_region=webcam_region,
        screen_content_center=screen_content_center,
        is_screen_share_layout=is_screen_share,
    )
```

**Webcam Detection Heuristics:**
- Face in corner region (< 35% from edge)
- Face size < 15% of frame area
- Different texture than surrounding content

---

### Stage 6: Crop Timeline Building

Builds dynamic crop trajectories from detection data.

```
Input:  Detection frames + segment metadata
Output: CropTimeline (keyframes with center positions)
```

**Crop Modes:**

#### Talking Head Mode (Single Region)
```
┌────────────────────────────────┐
│     Original 16:9 Frame        │
│  ┌──────────────────────────┐  │
│  │   Source Video           │  │
│  │      ┌─────────┐         │  │
│  │      │  Face   │◀────────│──│── Dynamic tracking
│  │      └─────────┘         │  │
│  │                          │  │
│  └──────────────────────────┘  │
└────────────────────────────────┘
              │
              ▼
        ┌───────────┐
        │   9:16    │
        │  Output   │
        │  ┌─────┐  │
        │  │Face │  │  ◀── Face centered in upper 35%
        │  └─────┘  │
        │           │
        │           │
        └───────────┘
```

#### Screen Share Mode (Stacked Layout)
```
┌────────────────────────────────┐
│     Original 16:9 Frame        │
│  ┌──────────────────────────┐  │
│  │   Screen Content         │──│── Screen crop (avoid webcam)
│  │                          │  │
│  │               ┌───┐      │  │
│  │               │Web│◀─────│──│── Webcam overlay detected
│  │               │cam│      │  │
│  └──────────────────────────┘  │
└────────────────────────────────┘
              │
              ▼
        ┌───────────┐
        │   9:16    │
        │  ┌─────┐  │  ◀── Top 55%: Screen content
        │  │Screen│ │
        │  └─────┘  │
        │  ┌─────┐  │  ◀── Bottom 45%: Face tracking
        │  │ Face │ │
        │  └─────┘  │
        └───────────┘
```

**Smoothing Algorithm:**
```python
def build_crop_timeline(detection_frames, segment) -> CropTimeline:
    # 3-frame moving average for smooth camera movement
    smoothed_centers = apply_moving_average(detection_frames, window_size=3)
    
    keyframes = []
    for i, frame_data in enumerate(detection_frames):
        center_x, center_y = smoothed_centers[i]
        
        # For talking_head: frame face in upper 35% of output
        if segment.layout_type == "talking_head":
            center_y = adjust_for_face_framing(center_y, target_ratio=0.35)
        
        keyframes.append(CropKeyframe(
            timestamp_ms=frame_data["timestamp_ms"],
            center_x=center_x,
            center_y=center_y,
        ))
    
    return CropTimeline(keyframes=keyframes, ...)
```

---

### Stage 7: Caption Generation (`CaptionGeneratorService`)

Generates viral-style ASS subtitles with word-by-word highlighting.

```
Input:  TranscriptSegments + clip timing + style config
Output: .ass subtitle file
```

**Caption Style System:**
```python
@dataclass
class CaptionStyle:
    font_name: str = "Arial Black"
    font_size: int = 72
    primary_color: str = "#FFFFFF"      # White base text
    highlight_color: str = "#FFD700"    # Gold highlight
    outline_color: str = "#000000"      # Black outline
    outline_width: int = 4
    position: str = "center"            # top/center/bottom
    max_words_per_line: int = 4
    word_by_word_highlight: bool = True
    bold: bool = True
    uppercase: bool = True
```

**Word-by-Word Highlighting:**
```
Time: 0.0s  → "THIS is how you"        (THIS highlighted)
Time: 0.3s  → "This IS how you"        (IS highlighted)
Time: 0.5s  → "This is HOW you"        (HOW highlighted)
Time: 0.7s  → "This is how YOU"        (YOU highlighted)
```

---

### Stage 8: Video Rendering (`RenderingService`)

Renders final clips using FFmpeg with dynamic cropping and caption burning.

```
Input:  RenderRequest (video, timelines, captions)
Output: RenderResult (MP4 file path + metadata)
```

**Render Modes:**

| Mode | Description | Use Case |
|------|-------------|----------|
| `focus_mode` | Single dynamic crop following face | Talking head content |
| `stack_mode` | Split screen (55% screen + 45% face) | Screen share with speaker |
| `static_mode` | Center crop, no motion | Fallback when no faces |

**FFmpeg Command Structure (Focus Mode):**
```bash
ffmpeg -i input.mp4 \
  -ss 134.2 -t 63.2 \                           # Clip timing
  -vf "
    crop=w:h:x:y,                                # Dynamic crop expression
    scale=1080:1920:force_original_aspect_ratio=decrease,
    pad=1080:1920:(ow-iw)/2:(oh-ih)/2,
    ass=captions.ass                             # Burn captions
  " \
  -c:v libx264 -preset veryfast -crf 20 \       # H.264 encoding
  -c:a aac -b:a 128k \                          # AAC audio
  output.mp4
```

**Dynamic Crop Expression:**
```python
def build_crop_expression(timeline: CropTimeline, clip_start_ms: int) -> str:
    # Interpolate X position between keyframes
    x_expr = build_interpolation_expr(timeline.keyframes, "x", clip_start_ms)
    y_expr = build_interpolation_expr(timeline.keyframes, "y", clip_start_ms)
    
    return f"crop={timeline.window_width}:{timeline.window_height}:{x_expr}:{y_expr}"
```

---

### Stage 9: S3 Upload (`S3UploadService`)

Uploads rendered clips and metadata to AWS S3.

```
Input:  Clip files + job metadata
Output: JobOutput (S3 URLs for all artifacts)
```

**S3 Key Structure:**
```
viewcreator-media/
├── clips/
│   └── {user_id}/
│       └── {job_id}/
│           ├── clip_0.mp4
│           ├── clip_1.mp4
│           ├── transcript.json
│           └── plan.json
```

**Upload Process:**
```python
async def upload_job_artifacts(
    job_id: str,
    owner_user_id: str,
    clip_artifacts: list[ClipArtifact],
    transcript_json: dict,
    plan_json: dict,
) -> JobOutput:
    # Upload clips with proper content-type
    for clip in clip_artifacts:
        s3_url = await upload_file(
            file_path=clip.file_path,
            s3_key=f"clips/{owner_user_id}/{job_id}/clip_{clip.index}.mp4",
            content_type="video/mp4",
        )
        clip.s3_url = s3_url
    
    # Upload metadata
    transcript_url = await upload_json(transcript_json, f"...transcript.json")
    plan_url = await upload_json(plan_json, f"...plan.json")
    
    return JobOutput(clips=clip_artifacts, transcript_url=transcript_url, ...)
```

---

## Service Components

### Directory Structure

```
viewcreator-clipping-worker/
├── app/
│   ├── main.py                    # FastAPI application entry
│   ├── config.py                  # Pydantic Settings configuration
│   ├── routers/
│   │   ├── ai_clipping.py         # /ai-clipping endpoints
│   │   ├── detection.py           # /detect endpoints
│   │   └── health.py              # /health endpoints
│   ├── services/
│   │   ├── ai_clipping_pipeline.py    # Main orchestrator
│   │   ├── video_downloader.py        # yt-dlp integration
│   │   ├── transcription_service.py   # Groq Whisper
│   │   ├── intelligence_planner.py    # Gemini planning
│   │   ├── detection_pipeline.py      # Detection orchestrator
│   │   ├── face_detector.py           # YOLOv8 faces
│   │   ├── pose_estimator.py          # MediaPipe poses
│   │   ├── tracker.py                 # DeepSORT tracking
│   │   ├── frame_extractor.py         # FFmpeg frame extraction
│   │   ├── content_region_detector.py # Screen/webcam detection
│   │   ├── caption_generator.py       # ASS subtitle generation
│   │   ├── rendering_service.py       # FFmpeg rendering
│   │   ├── s3_upload_service.py       # S3 upload handling
│   │   └── s3_client.py               # Low-level S3 operations
│   └── schemas/
│       ├── requests.py                # Pydantic request models
│       └── responses.py               # Pydantic response models
├── models/                        # Pre-downloaded ML models
│   └── yolov8n.pt
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

### Service Dependency Graph

```
                    ┌─────────────────────┐
                    │ AIClippingPipeline  │  ◀── Main Orchestrator
                    └─────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌─────────────────┐
│VideoDownloader│   │TranscriptionSvc │   │IntelligencePlnr│
└───────────────┘   └─────────────────┘   └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │DetectionPipeline│
                    └─────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ FaceDetector  │   │ PoseEstimator   │   │  ObjectTracker  │
└───────────────┘   └─────────────────┘   └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ContentRegionDet │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │RenderingService │───▶ CaptionGenerator
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ S3UploadService │
                    └─────────────────┘
```

---

## API Reference

### Health Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Basic liveness check |
| `/health/ready` | GET | Readiness (models loaded) |
| `/health/models` | GET | Detailed model status |

### AI Clipping Endpoints

#### Submit Job

```http
POST /ai-clipping/jobs
Content-Type: application/json

{
  "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "max_clips": 5,
  "duration_ranges": ["short", "medium"],
  "target_platform": "tiktok",
  "include_captions": true,
  "caption_style": {
    "font_family": "Arial Black",
    "font_size": 72,
    "primary_color": "FFFFFF",
    "highlight_color": "FFD700",
    "position": "center"
  },
  "external_job_id": "uuid-from-api",
  "owner_user_id": "user-123"
}
```

**Response:**
```json
{
  "job_id": "e2ee893e-0bcd-47cb-9173-578388a96a78",
  "status": "accepted",
  "message": "Job queued for processing",
  "estimated_processing_minutes": 8
}
```

#### Get Job Status

```http
GET /ai-clipping/jobs/{job_id}
```

**Response (Processing):**
```json
{
  "job_id": "e2ee893e-...",
  "status": "rendering",
  "progress_percent": 75,
  "current_stage": "rendering",
  "current_message": "Rendering clip 2 of 5"
}
```

**Response (Completed):**
```json
{
  "job_id": "e2ee893e-...",
  "status": "completed",
  "progress_percent": 100,
  "output": {
    "job_id": "e2ee893e-...",
    "source_video_title": "Video Title Here",
    "total_clips": 5,
    "clips": [
      {
        "clip_index": 0,
        "s3_url": "https://cdn.viewcreator.ai/clips/user-123/e2ee893e/clip_0.mp4",
        "duration_ms": 45000,
        "start_time_ms": 134200,
        "end_time_ms": 179200,
        "virality_score": 0.92,
        "layout_type": "screen_share",
        "summary": "Key insight about Claude Opus 4.5 pricing...",
        "tags": ["AI", "Claude", "Pricing"]
      }
    ],
    "transcript_url": "https://cdn.viewcreator.ai/clips/.../transcript.json",
    "plan_url": "https://cdn.viewcreator.ai/clips/.../plan.json",
    "processing_time_seconds": 182.5
  }
}
```

#### List Jobs

```http
GET /ai-clipping/jobs?status=completed&limit=10
```

#### Cancel Job

```http
DELETE /ai-clipping/jobs/{job_id}
```

### Detection Endpoints

```http
POST /detect
Content-Type: application/json

{
  "job_id": "uuid",
  "video_s3_key": "users/user123/videos/source.mp4",
  "frame_interval_seconds": 2.0,
  "detect_faces": true,
  "detect_poses": true
}
```

---

## Configuration

### Environment Variables

```env
# ═══════════════════════════════════════════════════════════════
# CORE SETTINGS
# ═══════════════════════════════════════════════════════════════
DEBUG=false
LOG_LEVEL=INFO
MAX_CONCURRENT_JOBS=2

# ═══════════════════════════════════════════════════════════════
# AWS CONFIGURATION
# ═══════════════════════════════════════════════════════════════
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key          # Optional in ECS (use IAM roles)
AWS_SECRET_ACCESS_KEY=your-secret-key      # Optional in ECS (use IAM roles)
S3_BUCKET=viewcreator-media

# ═══════════════════════════════════════════════════════════════
# AI SERVICES
# ═══════════════════════════════════════════════════════════════

# Transcription (Groq Whisper - 216x realtime)
GROQ_API_KEY=gsk_...
TRANSCRIPTION_MODEL=whisper-large-v3-turbo

# Intelligence Planning (OpenRouter → Gemini)
OPENROUTER_API_KEY=sk-or-...
GEMINI_MODEL=google/gemini-2.5-flash

# ═══════════════════════════════════════════════════════════════
# VIDEO PROCESSING
# ═══════════════════════════════════════════════════════════════

# Download
YTDLP_PATH=/usr/local/bin/yt-dlp
MAX_DOWNLOAD_DURATION_SECONDS=7200

# Rendering
TARGET_OUTPUT_WIDTH=1080
TARGET_OUTPUT_HEIGHT=1920
FFMPEG_PRESET=veryfast
FFMPEG_CRF=20

# Stack Layout (screen share mode)
STACK_SCREEN_HEIGHT_RATIO=0.55
STACK_FACE_HEIGHT_RATIO=0.45

# ═══════════════════════════════════════════════════════════════
# DETECTION
# ═══════════════════════════════════════════════════════════════
FACE_CONFIDENCE_THRESHOLD=0.5
POSE_CONFIDENCE_THRESHOLD=0.5
FRAME_INTERVAL_SECONDS=2.0
```

### Full Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `false` | Enable debug logging |
| `LOG_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `MAX_CONCURRENT_JOBS` | `2` | Maximum parallel processing jobs |
| `AWS_REGION` | `us-east-1` | AWS region for S3 |
| `S3_BUCKET` | `viewcreator-media` | S3 bucket for video storage |
| `GROQ_API_KEY` | - | **Required** Groq API key |
| `OPENROUTER_API_KEY` | - | **Required** OpenRouter API key |
| `TRANSCRIPTION_MODEL` | `whisper-large-v3-turbo` | Whisper model variant |
| `GEMINI_MODEL` | `google/gemini-2.5-flash` | Gemini model for planning |
| `TARGET_OUTPUT_WIDTH` | `1080` | Output video width (px) |
| `TARGET_OUTPUT_HEIGHT` | `1920` | Output video height (px) |
| `FFMPEG_PRESET` | `veryfast` | Encoding speed preset |
| `FFMPEG_CRF` | `20` | Quality (lower = better, 18-23 recommended) |
| `FACE_CONFIDENCE_THRESHOLD` | `0.5` | Minimum face detection confidence |
| `MAX_DOWNLOAD_DURATION_SECONDS` | `7200` | Max video length (2 hours) |

---

## Deployment

### Docker Compose (Development)

```yaml
# docker-compose.yml
version: '3.8'

services:
  clipping-worker:
    build: .
    container_name: viewcreator-clipping-worker
    ports:
      - "8000:8000"
    environment:
      - AWS_REGION=us-east-1
      - S3_BUCKET=viewcreator-media-dev
      - GROQ_API_KEY=${GROQ_API_KEY}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
    volumes:
      - /tmp/clipping-worker:/tmp/clipping-worker
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
# Build and run
docker-compose up --build

# View logs
docker logs viewcreator-clipping-worker -f

# Rebuild after code changes
docker-compose up -d --build
```

### AWS ECS (Production)

```json
{
  "family": "clipping-worker",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "4096",
  "memory": "8192",
  "executionRoleArn": "arn:aws:iam::...:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::...:role/clipping-worker-task-role",
  "containerDefinitions": [
    {
      "name": "clipping-worker",
      "image": "your-ecr-repo/clipping-worker:latest",
      "portMappings": [
        { "containerPort": 8000, "protocol": "tcp" }
      ],
      "secrets": [
        {
          "name": "GROQ_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:...:secret:groq-api-key"
        },
        {
          "name": "OPENROUTER_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:...:secret:openrouter-api-key"
        }
      ],
      "environment": [
        { "name": "S3_BUCKET", "value": "viewcreator-media" },
        { "name": "AWS_REGION", "value": "us-east-1" }
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 10,
        "retries": 3,
        "startPeriod": 120
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/clipping-worker",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

**IAM Task Role Permissions:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::viewcreator-media/*"
    }
  ]
}
```

---

## Integration

### NestJS API Integration

The `viewcreator-api` orchestrates jobs and transforms URLs:

```typescript
// viewcreator-api/src/modules/ai-clipping-agent/services/clipping-worker-client.service.ts

@Injectable()
export class ClippingWorkerClientService {
  constructor(
    private readonly httpService: HttpService,
    private readonly configService: ConfigService,
  ) {}

  async submitJob(request: ClippingJobSubmitRequest): Promise<{ jobId: string }> {
    const workerUrl = this.configService.get('CLIPPING_WORKER_URL');
    
    const response = await this.httpService.post(
      `${workerUrl}/ai-clipping/jobs`,
      {
        video_url: request.video_url,
        max_clips: request.max_clips,
        duration_ranges: request.clip_duration_ranges,
        target_platform: request.target_platform,
        include_captions: request.include_captions,
        caption_style: request.caption_style,
        external_job_id: request.external_job_id,
        owner_user_id: request.owner_user_id,
      },
    ).toPromise();
    
    return { jobId: response.data.job_id };
  }

  async pollJobCompletion(jobId: string): Promise<ClippingJobResult> {
    const workerUrl = this.configService.get('CLIPPING_WORKER_URL');
    const response = await this.httpService.get(
      `${workerUrl}/ai-clipping/jobs/${jobId}`,
    ).toPromise();
    
    // Transform S3 URLs to CDN URLs
    if (response.data.output?.clips) {
      response.data.output.clips = response.data.output.clips.map(clip => ({
        ...clip,
        s3_url: this.transformToCdnUrl(clip.s3_url),
      }));
    }
    
    return response.data;
  }

  private transformToCdnUrl(s3Url: string): string {
    // Transform s3://bucket/key to https://cdn.viewcreator.ai/key
    const cdnDomain = this.configService.get('MEDIA_CDN_DOMAIN');
    return s3Url.replace(/^https:\/\/[^\/]+\//, `https://${cdnDomain}/`);
  }
}
```

### Frontend Integration

```typescript
// viewcreator-creator-ui/src/app/(dashboard)/ai-clipping-agent/page.tsx

const [durationRanges, setDurationRanges] = useState<string[]>(['medium']);

const handleSubmit = async () => {
  const response = await fetch('/api/ai-clipping-agent/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      youtubeUrl: videoUrl,
      requestedClipCount: clipCount,
      clipDurationRanges: durationRanges,
      targetPlatform: 'tiktok',
      includeCaptions: true,
    }),
  });
  
  const { jobId } = await response.json();
  // Poll for completion...
};

// Duration range selector UI
<div className="flex gap-2">
  {['short', 'medium', 'long'].map(range => (
    <Button
      key={range}
      variant={durationRanges.includes(range) ? 'default' : 'outline'}
      onClick={() => toggleDurationRange(range)}
    >
      {range === 'short' ? '15-30s' : range === 'medium' ? '30-60s' : '1-2min'}
    </Button>
  ))}
</div>
```

---

## Development

### Local Setup

```bash
# 1. Clone and setup
cd viewcreator-clipping-worker
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install system dependencies
# macOS:
brew install ffmpeg

# 4. Download yt-dlp
curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp
chmod +x /usr/local/bin/yt-dlp

# 5. Set environment variables
export GROQ_API_KEY=gsk_...
export OPENROUTER_API_KEY=sk-or-...

# 6. Run server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing

```bash
# Submit a test job
curl -X POST http://localhost:8000/ai-clipping/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "max_clips": 1,
    "duration_ranges": ["short"],
    "include_captions": true
  }'

# Check job status
curl http://localhost:8000/ai-clipping/jobs/{job_id}

# View API docs
open http://localhost:8000/docs
```

### Debugging

```bash
# View Docker logs
docker logs viewcreator-clipping-worker -f --tail=100

# Check specific stage
docker logs viewcreator-clipping-worker 2>&1 | grep "Transcription"
docker logs viewcreator-clipping-worker 2>&1 | grep "Detection"
docker logs viewcreator-clipping-worker 2>&1 | grep "Rendering"

# Rebuild after changes
docker-compose up -d --build
```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `GROQ_API_KEY not set` | Missing env var | Export GROQ_API_KEY |
| `yt-dlp not found` | Missing binary | Install yt-dlp |
| `FFmpeg not found` | Missing binary | Install ffmpeg |
| `S3 upload failed` | Missing credentials | Set AWS credentials or IAM role |
| `Detection returning 0 faces` | Low confidence threshold | Lower `FACE_CONFIDENCE_THRESHOLD` |
| `Black screen in output` | S3 URL not transformed | Check CDN URL transformation |

---

## Performance Benchmarks

| Stage | Time (30 min video) | Notes |
|-------|---------------------|-------|
| Download | 15-30s | Depends on network |
| Transcription | ~8s | 216x realtime (Groq) |
| Planning | 2-3s | Gemini Flash |
| Detection | 30-45s | 2 FPS sampling |
| Rendering | 10-15s per clip | FFmpeg veryfast |
| Upload | 5-10s per clip | S3 PUT |
| **Total** | **2-4 min** | For 5 clips |

---

## License

Proprietary - ViewCreator © 2025
