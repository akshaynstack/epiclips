# Clip Count Scaling & Auto-Optimization

This document explains the intelligent clip count system that automatically scales the number of clips based on video duration, prevents mid-sentence cutoffs, and optimizes resource usage.

---

## Quick Reference

| Mode | Request Example | Behavior |
|------|-----------------|----------|
| **Full Auto** | `{"video_url": "...", "auto_clip_count": true}` | Backend decides clip count based on video length (1 clip per 5 min) |
| **Auto with Limit** | `{"video_url": "...", "auto_clip_count": true, "max_clips": 10}` | Auto-scale up to 10 clips max |
| **Exact Count** | `{"video_url": "...", "auto_clip_count": false, "max_clips": 5}` | Generate exactly 5 clips (required) |

---

## API Usage

### Endpoint
```
POST /ai-clipping/jobs
```

### Request Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `video_url` | string | Yes* | - | Video URL (YouTube, S3, or direct) |
| `s3_key` | string | Yes* | - | S3 key (alternative to video_url) |
| `auto_clip_count` | boolean | No | `true` | Enable intelligent clip count scaling |
| `max_clips` | integer | Conditional | `null` | Maximum clips. **Required when `auto_clip_count=false`** |
| `duration_ranges` | array | No | `["medium"]` | Clip duration: `["short"]`, `["medium"]`, `["long"]`, or combinations |

*One of `video_url` or `s3_key` is required.

---

## Scaling Modes Explained

### Mode 1: Full Automatic (Recommended)

```json
{
  "video_url": "https://youtube.com/watch?v=example",
  "auto_clip_count": true,
  "duration_ranges": ["medium"]
}
```

**Behavior:**
- Backend calculates optimal clip count using formula: `clips = video_minutes × 0.2`
- Applies minimum of 2 clips, maximum of 30 clips (configurable)
- A 15-minute video → 3 clips
- A 60-minute video → 12 clips
- A 3-minute video → 2 clips (minimum)

**Best for:** Production use, hands-off processing

---

### Mode 2: Auto with Upper Limit

```json
{
  "video_url": "https://youtube.com/watch?v=example",
  "auto_clip_count": true,
  "max_clips": 10,
  "duration_ranges": ["short", "medium"]
}
```

**Behavior:**
- Backend calculates optimal count, but won't exceed your `max_clips`
- A 60-minute video would suggest 12 clips → capped to 10
- A 15-minute video would suggest 3 clips → stays at 3

**Best for:** Budget control, limiting processing time

---

### Mode 3: Exact Count (Manual)

```json
{
  "video_url": "https://youtube.com/watch?v=example",
  "auto_clip_count": false,
  "max_clips": 5,
  "duration_ranges": ["long"]
}
```

**Behavior:**
- Generates exactly the number specified (up to config maximum)
- `max_clips` field is **REQUIRED** in this mode
- Omitting `max_clips` when `auto_clip_count=false` returns validation error

**Best for:** Testing, specific requirements, A/B testing clip counts

---

## Scaling Formula

```
suggested_clips = floor(video_duration_minutes × clips_per_minute_ratio)

where:
  clips_per_minute_ratio = 0.2 (default, configurable)
  
final_clips = clamp(suggested_clips, min_clips, min(max_clips_absolute, user_max_clips))

where:
  min_clips = 2
  max_clips_absolute = 30 (system limit)
  user_max_clips = request.max_clips or max_clips_absolute
```

### Examples

| Video Duration | Suggested | With max_clips=5 | With max_clips=50 |
|---------------|-----------|------------------|-------------------|
| 5 minutes | 2 (min) | 2 | 2 |
| 15 minutes | 3 | 3 | 3 |
| 30 minutes | 6 | 5 (capped) | 6 |
| 60 minutes | 12 | 5 (capped) | 12 |
| 120 minutes | 24 | 5 (capped) | 24 |
| 180 minutes | 36 → 30 | 5 (capped) | 30 (system max) |

---

## Sentence Boundary Snapping

All clips are automatically extended to complete sentences, preventing mid-sentence cutoffs.

### How It Works

1. AI plans clip end time (e.g., 45.2 seconds)
2. System searches transcript for nearest sentence-ending punctuation (`.`, `!`, `?`)
3. If found within 5 seconds, extends clip to that boundary
4. If not found, keeps original end time

### Example

```
Original clip: 0:00 - 0:45.2 (cuts mid-sentence: "and that's why I think...")
Adjusted clip: 0:00 - 0:47.8 (complete: "and that's why I think it matters.")
```

### Log Indicator
```
Clip end time adjusted for sentence boundary: 45200ms -> 47800ms (+2600ms)
```

---

## Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `CLIP_SCALING_ENABLED` | `true` | Enable/disable auto-scaling globally |
| `CLIPS_PER_MINUTE_RATIO` | `0.2` | Clips per minute of video (0.2 = 1 clip per 5 min) |
| `MIN_CLIPS` | `2` | Minimum clips even for short videos |
| `MAX_CLIPS_ABSOLUTE` | `30` | System-wide maximum clips |
| `SENTENCE_SNAPPING_ENABLED` | `true` | Enable sentence boundary detection |
| `SENTENCE_EXTENSION_MAX_SECONDS` | `5.0` | Max seconds to extend for sentence completion |

---

## cURL Examples

### Full Auto Mode
```bash
curl -X POST "http://localhost:8000/ai-clipping/jobs" \
  -H "Content-Type: application/json" \
  -H "X-Genesis-API-Key: your-api-key" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=example",
    "auto_clip_count": true,
    "duration_ranges": ["medium"],
    "include_captions": true,
    "caption_preset": "viral_gold"
  }'
```

### Auto with 10 Clip Limit
```bash
curl -X POST "http://localhost:8000/ai-clipping/jobs" \
  -H "Content-Type: application/json" \
  -H "X-Genesis-API-Key: your-api-key" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=example",
    "auto_clip_count": true,
    "max_clips": 10,
    "duration_ranges": ["short", "medium"]
  }'
```

### Exact 5 Clips (Manual Mode)
```bash
curl -X POST "http://localhost:8000/ai-clipping/jobs" \
  -H "Content-Type: application/json" \
  -H "X-Genesis-API-Key: your-api-key" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=example",
    "auto_clip_count": false,
    "max_clips": 5,
    "duration_ranges": ["long"]
  }'
```

---

## Python SDK Examples

### Using test_job.py

```bash
# Full auto mode (default)
python test_job.py --duration medium

# Auto mode with 10 clip limit
python test_job.py --max-clips 10 --duration medium

# Exact 5 clips (disable auto)
python test_job.py --max-clips 5 --no-auto --duration short
```

### Programmatic Usage

```python
import requests

# Full auto mode
response = requests.post(
    "http://localhost:8000/ai-clipping/jobs",
    headers={"X-Genesis-API-Key": "your-key"},
    json={
        "video_url": "https://youtube.com/watch?v=example",
        "auto_clip_count": True,  # Let backend decide
        "duration_ranges": ["medium"]
    }
)

# Auto with limit
response = requests.post(
    "http://localhost:8000/ai-clipping/jobs",
    headers={"X-Genesis-API-Key": "your-key"},
    json={
        "video_url": "https://youtube.com/watch?v=example",
        "auto_clip_count": True,
        "max_clips": 8,  # Up to 8 clips
        "duration_ranges": ["short", "medium"]
    }
)

# Exact count
response = requests.post(
    "http://localhost:8000/ai-clipping/jobs",
    headers={"X-Genesis-API-Key": "your-key"},
    json={
        "video_url": "https://youtube.com/watch?v=example",
        "auto_clip_count": False,  # Manual mode
        "max_clips": 3,  # REQUIRED in this mode
        "duration_ranges": ["long"]
    }
)
```

---

## Response Example

```json
{
  "job_id": "3e9c09b5-0f7d-403b-b32b-31faa5384875",
  "status": "accepted",
  "message": "Job queued for processing",
  "estimated_processing_minutes": 13
}
```

---

## Validation Errors

### Missing max_clips in Manual Mode
```json
// Request
{
  "video_url": "...",
  "auto_clip_count": false
  // max_clips missing!
}

// Response: 422 Unprocessable Entity
{
  "detail": [
    {
      "msg": "Value error, max_clips is required when auto_clip_count is false",
      "type": "value_error"
    }
  ]
}
```

### max_clips Out of Range
```json
// Request
{
  "video_url": "...",
  "max_clips": 100  // exceeds limit
}

// Response: 422 Unprocessable Entity
{
  "detail": [
    {
      "msg": "ensure this value is less than or equal to 50",
      "type": "value_error.number.not_le"
    }
  ]
}
```

---

## Monitoring & Logs

Look for these log entries to verify clip scaling:

```log
# Auto-scaling calculation
Clip count scaling: 15.8 min video -> suggested 3 clips (ratio: 0.2), effective max: 5, final: 3

# Sentence boundary adjustment
Clip end time adjusted for sentence boundary: 419700ms -> 420053ms (+353ms)

# Manual mode (auto-scaling disabled)
Clip count (auto-scaling OFF): using max 5 -> final: 5
```

---

## Performance Considerations

| Clips | Estimated Time | Notes |
|-------|----------------|-------|
| 1-3 | 2-5 minutes | Quick turnaround |
| 4-6 | 5-10 minutes | Standard processing |
| 7-15 | 10-20 minutes | Moderate load |
| 16-30 | 20-40 minutes | Consider async/webhook |

**Tip:** Use `auto_clip_count: true` with a reasonable `max_clips` to balance quality and processing time.

---

## Related Documentation

- [API_USAGE.md](API_USAGE.md) - Complete API reference
- [TEST_CURL_REQUESTS.md](TEST_CURL_REQUESTS.md) - More cURL examples
