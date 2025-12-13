# Time Range Selection & Duration Enforcement

This document explains the time range selection feature that allows users to select a specific portion of a video for clip generation, and how clip duration ranges are strictly enforced.

---

## Overview

The time range selection feature allows users to:

1. **Get full video duration** via the `/metadata` endpoint
2. **Select a specific time range** (start to end) using a dual-slider UI
3. **Generate clips only from that selected portion** of the video
4. **Receive clips that strictly adhere** to the selected duration ranges

---

## Frontend Implementation: Recommended Flow

### Why Use Backend for Metadata?

✅ **Use the `/ai-clipping/metadata` endpoint** instead of frontend YouTube API calls:

| Approach | Pros | Cons |
|----------|------|------|
| **Backend `/metadata` (Recommended)** | No API keys in frontend, works for all sources (YouTube/S3/URLs), returns credit estimate | Adds one API call |
| Frontend YouTube API | Slightly faster | Requires API keys, only works for YouTube, no credit estimate |

### Complete Frontend Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  1. User enters YouTube URL                                     │
│     ↓                                                           │
│  2. Frontend calls POST /ai-clipping/metadata                   │
│     ↓                                                           │
│  3. Backend returns: duration, title, credits                   │
│     ↓                                                           │
│  4. Frontend renders slider (0 to duration_seconds)             │
│     ↓                                                           │
│  5. User adjusts start/end handles                              │
│     ↓                                                           │
│  6. Frontend submits job with start_time_seconds/end_time_seconds│
└─────────────────────────────────────────────────────────────────┘
```

### Dual-Handle Slider UI Concept

```
Video Timeline (0:00 - 45:00)
├─────────────────────────────────────────────────────────┤
        ▲                                         ▲
     Start                                       End
   (10:00)                                    (35:00)
        ├─────────────────────────────────────┤
              Selected Range: 25 minutes
```

### Step-by-Step Implementation

#### Step 1: Get Video Metadata (on URL input)

```javascript
// When user enters/pastes a YouTube URL
async function onVideoUrlChange(videoUrl) {
  setLoading(true);
  
  try {
    const response = await fetch('/ai-clipping/metadata', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY
      },
      body: JSON.stringify({ video_url: videoUrl })
    });
    
    if (!response.ok) {
      throw new Error('Failed to fetch video metadata');
    }
    
    const metadata = await response.json();
    
    // Store for slider configuration
    setVideoMetadata({
      title: metadata.title,
      durationSeconds: metadata.duration_seconds,
      durationMinutes: metadata.duration_minutes,
      estimatedCredits: metadata.estimated_credits
    });
    
    // Initialize slider with full range
    setTimeRange({
      start: 0,
      end: metadata.duration_seconds
    });
    
  } catch (error) {
    showError('Could not load video info. Please check the URL.');
  } finally {
    setLoading(false);
  }
}
```

#### Step 2: Render the Dual-Handle Slider

```jsx
// React example with a range slider component
function TimeRangeSlider({ metadata, timeRange, setTimeRange }) {
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };
  
  return (
    <div className="time-range-selector">
      <h3>{metadata.title}</h3>
      <p>Duration: {formatTime(metadata.durationSeconds)}</p>
      
      {/* Visual timeline bar */}
      <div className="timeline-container">
        <RangeSlider
          min={0}
          max={metadata.durationSeconds}
          step={1}
          value={[timeRange.start, timeRange.end]}
          onChange={([start, end]) => setTimeRange({ start, end })}
        />
        
        <div className="time-labels">
          <span>{formatTime(timeRange.start)}</span>
          <span className="selected-duration">
            Selected: {formatTime(timeRange.end - timeRange.start)}
          </span>
          <span>{formatTime(timeRange.end)}</span>
        </div>
      </div>
      
      {/* Show adjusted credit estimate */}
      <p className="credits-estimate">
        Estimated credits: {Math.ceil((timeRange.end - timeRange.start) / 60 * 10)}
        {timeRange.end - timeRange.start < metadata.durationSeconds && (
          <span className="savings">
            (Saving {metadata.estimatedCredits - Math.ceil((timeRange.end - timeRange.start) / 60 * 10)} credits)
          </span>
        )}
      </p>
    </div>
  );
}
```

#### Step 3: Submit Job with Time Range

```javascript
async function submitClippingJob() {
  const payload = {
    video_url: videoUrl,
    duration_ranges: selectedDurationRanges, // ['short', 'medium', 'long']
    auto_clip_count: true,
    
    // Only include if user modified the range (optimization)
    ...(timeRange.start > 0 && { start_time_seconds: timeRange.start }),
    ...(timeRange.end < videoMetadata.durationSeconds && { end_time_seconds: timeRange.end })
  };
  
  const response = await fetch('/ai-clipping/jobs', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY
    },
    body: JSON.stringify(payload)
  });
  
  const job = await response.json();
  // Start polling for job status...
}
```

---

## API Reference

### 1. Get Video Metadata (Call This First!)

**Endpoint:** `POST /ai-clipping/metadata`

Use this to get the full video duration before showing the slider.

**Request:**
```json
{
  "video_url": "https://youtube.com/watch?v=example"
}
```

**Response:**
```json
{
  "title": "My Long Video",
  "duration_seconds": 2700.0,
  "duration_minutes": 45.0,
  "width": 1920,
  "height": 1080,
  "source_type": "youtube",
  "estimated_credits": 450
}
```

---

### 2. Submit Job with Time Range

**Endpoint:** `POST /ai-clipping/jobs`

**New Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `start_time_seconds` | float | No | `0` | Start of time range (seconds from video start) |
| `end_time_seconds` | float | No | `null` (end of video) | End of time range (seconds from video start) |

**Example - Process Only Minutes 10-35:**
```json
{
  "video_url": "https://youtube.com/watch?v=example",
  "start_time_seconds": 600,
  "end_time_seconds": 2100,
  "duration_ranges": ["medium"],
  "auto_clip_count": true
}
```

**Example - Process Last 10 Minutes:**
```json
{
  "video_url": "https://youtube.com/watch?v=example",
  "start_time_seconds": 2100,
  "end_time_seconds": null,
  "duration_ranges": ["short", "medium"],
  "max_clips": 5
}
```

---

## Duration Range Enforcement

### The Problem

Previously, when users selected duration ranges like "medium" (30-60 seconds), clips could sometimes be generated outside these bounds (e.g., 25 seconds or 75 seconds).

### The Solution

The backend now **strictly enforces** duration ranges through:

1. **Clear AI Prompts**: The AI is explicitly instructed to generate clips within exact bounds
2. **Post-Processing Validation**: After AI returns clips, they are validated and adjusted
3. **Hard Limits**: Clips that can't be adjusted are filtered out

### Duration Range Definitions

| Range | Min Duration | Max Duration | Description |
|-------|--------------|--------------|-------------|
| `short` | 15 seconds | 30 seconds | Quick, punchy clips |
| `medium` | 30 seconds | 60 seconds | Standard viral format |
| `long` | 60 seconds | 120 seconds | In-depth content |

### How Enforcement Works

```
User Selects: ["medium"]
             ↓
AI Generates Clip: 25s → ADJUSTED to 30s (extended to sentence boundary)
AI Generates Clip: 45s → VALID ✓
AI Generates Clip: 75s → ADJUSTED to 60s (trimmed at sentence boundary)
AI Generates Clip: 12s → FILTERED OUT (too short even after extension)
```

### Validation Rules

1. **Clips too short** → Extended to nearest sentence boundary up to max duration
2. **Clips too long** → Trimmed at nearest sentence boundary down to max duration  
3. **Clips that can't fit** → Filtered out with warning logged
4. **Sentence boundaries** → Always preferred to prevent mid-sentence cutoffs

---

## Complete Workflow Example

### Step 1: User Opens Video Selection

Frontend fetches metadata to configure the UI:

```javascript
// Get video info for slider setup
const info = await getVideoMetadata(videoUrl);
console.log(`Video is ${info.duration_minutes} minutes long`);
```

### Step 2: User Adjusts Time Range

User moves sliders to select portion of interest:

```javascript
// User selects 5:00 to 25:00 of a 45-minute video
const startSeconds = 300;  // 5:00
const endSeconds = 1500;   // 25:00
```

### Step 3: User Selects Duration Preferences

User picks which clip lengths they want:

```javascript
const durationRanges = ['short', 'medium']; // 15-60 second clips
```

### Step 4: Submit Job

```javascript
const job = await submitJob({
  video_url: videoUrl,
  start_time_seconds: startSeconds,
  end_time_seconds: endSeconds,
  duration_ranges: durationRanges,
  auto_clip_count: true
});
```

### Step 5: Receive Results

All generated clips will:
- Start at or after `start_time_seconds` (5:00)
- End at or before `end_time_seconds` (25:00)  
- Be between 15-60 seconds long (short + medium ranges)
- End at sentence boundaries (no mid-sentence cutoffs)

---

## Response Format

The job output includes time range information:

```json
{
  "job_id": "abc123",
  "status": "completed",
  "output": {
    "source_video_duration_seconds": 2700.0,
    "processed_range": {
      "start_time_seconds": 300,
      "end_time_seconds": 1500,
      "processed_duration_seconds": 1200
    },
    "total_clips": 4,
    "clips": [
      {
        "clip_index": 1,
        "start_time_ms": 312000,
        "end_time_ms": 354000,
        "duration_ms": 42000,
        "virality_score": 0.92,
        "summary": "Key insight about productivity..."
      }
    ]
  }
}
```

---

## Error Handling

### Invalid Time Range

```json
{
  "detail": "end_time_seconds (300) must be greater than start_time_seconds (600)"
}
```

### Time Range Exceeds Video Duration

```json
{
  "detail": "end_time_seconds (3600) exceeds video duration (2700 seconds)"
}
```

### Time Range Too Short

```json
{
  "detail": "Selected time range (25 seconds) is too short. Minimum is 30 seconds for clip generation."
}
```

---

## Configuration

The following settings control time range and duration enforcement (in `config.py`):

```python
# Minimum selected range for processing
MIN_PROCESSING_RANGE_SECONDS = 60

# Duration range definitions
DURATION_RANGE_CONFIG = {
    "short": {"min": 15, "max": 30},
    "medium": {"min": 30, "max": 60},
    "long": {"min": 60, "max": 120},
}

# Sentence boundary snapping
SENTENCE_SNAPPING_ENABLED = True
SENTENCE_EXTENSION_MAX_SECONDS = 5  # Max extension to find sentence boundary
```

---

## Best Practices

1. **Always fetch metadata first** to properly configure the slider bounds
2. **Validate client-side** that end > start before submitting
3. **Show visual feedback** of the selected range duration
4. **Display estimated clip count** based on selected range and auto_clip_count setting
5. **Handle edge cases** like very short videos where time selection may not be needed

---

## Migration Notes

### From Previous API Versions

The `start_time_seconds` and `end_time_seconds` fields are **optional** and backward compatible:

- If omitted, the entire video is processed (existing behavior)
- If only `start_time_seconds` is provided, processes from that point to end
- If only `end_time_seconds` is provided, processes from beginning to that point

No changes required for existing integrations that process full videos.
