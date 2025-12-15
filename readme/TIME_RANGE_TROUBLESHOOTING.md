# Time Range Integration - Troubleshooting Guide

## Expected Request Format

Genesis expects **seconds as numbers**:

```json
{
  "video_url": "https://youtube.com/watch?v=...",
  "start_time_seconds": 300.0,
  "end_time_seconds": 900.0
}
```

---

## Field Flow Summary

| Layer | Input Field | Output Field | Unit |
|-------|-------------|--------------|------|
| **UI → API** | `startTimeSeconds`, `endTimeSeconds` | - | seconds |
| **API → Genesis** | `start_time_seconds`, `end_time_seconds` | - | seconds |
| **Genesis → API** | - | `start_time_ms`, `end_time_ms` (per clip) | milliseconds |
| **API Internal** | - | `startTimeMs`, `endTimeMs` | milliseconds |

---

## Common Issues

### ❌ Issue 1: Wrong Field Names

| Wrong | Correct |
|-------|---------|
| `startTime` | `start_time_seconds` |
| `endTime` | `end_time_seconds` |
| `start_time` | `start_time_seconds` |
| `end_time` | `end_time_seconds` |

### ❌ Issue 2: Milliseconds Instead of Seconds

| Wrong (ms) | Correct (seconds) |
|------------|-------------------|
| `300000` | `300.0` |
| `900000` | `900.0` |

**Conversion:** `seconds = milliseconds / 1000`

### ❌ Issue 3: String Instead of Number

| Wrong | Correct |
|-------|---------|
| `"300"` | `300.0` |
| `"900"` | `900.0` |

### ❌ Issue 4: Empty String Instead of Null

| Wrong | Correct |
|-------|---------|
| `""` | `null` or omit field |

---

## Quick Fix Checklist

```javascript
// In viewcreator-api, ensure:
const payload = {
  video_url: videoUrl,
  // ✅ Field names must be exactly these:
  start_time_seconds: startSeconds,  // NOT startTime, NOT start_time
  end_time_seconds: endSeconds,      // NOT endTime, NOT end_time
};

// ✅ Convert ms to seconds if needed:
start_time_seconds: startTimeMs / 1000,
end_time_seconds: endTimeMs / 1000,

// ✅ Omit if not set (don't send empty string):
...(startSeconds && { start_time_seconds: startSeconds }),
...(endSeconds && { end_time_seconds: endSeconds }),
```

---

## Test with curl

```bash
curl -X POST http://localhost:8000/ai-clipping/jobs \
  -H "Content-Type: application/json" \
  -H "X-Genesis-API-Key: your-key" \
  -d '{
    "video_url": "https://youtube.com/watch?v=example",
    "start_time_seconds": 300,
    "end_time_seconds": 900,
    "duration_ranges": ["medium"],
    "max_clips": 3
  }'
```
