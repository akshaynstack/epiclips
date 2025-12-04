# Test cURL Requests

Collection of tested cURL requests for the AI Clipping API.

---

## Test 1: N8n Hindi Video (3 clips, default duration)

```bash
curl -X POST http://localhost:8000/ai-clipping/jobs \
  -H "Content-Type: application/json" \
  -H "X-Genesis-API-Key: dev-api-key-12345" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=fPkGIdBVB3Y",
    "max_clips": 3,
    "include_captions": true,
    "caption_preset": "viral_gold"
  }'
```

**Result:** Job `ea3946d8-9c2f-4628-970d-caff417cecbe` - 2 clips generated in ~195s

---

## Test 2: Code Video (2 clips, default duration)

```bash
curl -X POST http://localhost:8000/ai-clipping/jobs \
  -H "Content-Type: application/json" \
  -H "X-Genesis-API-Key: dev-api-key-12345" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=dNviWH13dNY",
    "max_clips": 2,
    "include_captions": true,
    "caption_preset": "viral_gold"
  }'
```

**Result:** Job `ae8abae0-3d67-4268-b048-d525644f59ea` - 2 clips (9.1MB, 7.5MB) in ~267s

---

## Test 3: Vibe Coding with Claude (2 clips, short duration 15-30s)

```bash
curl -X POST http://localhost:8000/ai-clipping/jobs \
  -H "Content-Type: application/json" \
  -H "X-Genesis-API-Key: dev-api-key-12345" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=mWZGWP2yjn0",
    "max_clips": 2,
    "include_captions": true,
    "caption_preset": "viral_gold",
    "duration_ranges": ["short"]
  }'
```

**Result:** Job `5b2ef6f6-43ec-4707-92a5-a7511c0c0576` - 2 clips (4.5MB, 4.0MB) in ~75s

---

## Test 4: Work Harder Motivational (2 clips, short duration 15-30s)

```bash
curl -X POST http://localhost:8000/ai-clipping/jobs \
  -H "Content-Type: application/json" \
  -H "X-Genesis-API-Key: dev-api-key-12345" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=oWqbdZ-C7uQ",
    "max_clips": 2,
    "include_captions": true,
    "caption_preset": "viral_gold",
    "duration_ranges": ["short"]
  }'
```

**Result:** Job `38fe854b-646c-4dcf-801f-3b1472d1c8d3` - 2 clips (6.4MB, 7.9MB) in ~150s

---

## Check Job Status

```bash
curl http://localhost:8000/ai-clipping/jobs/{job_id}
```

---

## PowerShell Equivalents

For Windows PowerShell, use `Invoke-RestMethod`:

```powershell
# Submit job
$body = '{"video_url":"https://www.youtube.com/watch?v=VIDEO_ID","max_clips":2,"include_captions":true,"caption_preset":"viral_gold","duration_ranges":["short"]}'
Invoke-RestMethod -Uri "http://localhost:8000/ai-clipping/jobs" -Method POST -Headers @{"Content-Type"="application/json";"X-Genesis-API-Key"="dev-api-key-12345"} -Body $body

# Check status
Invoke-RestMethod -Uri "http://localhost:8000/ai-clipping/jobs/{job_id}" -Method GET
```

---

## Duration Ranges

| Range | Duration | Use Case |
|-------|----------|----------|
| `short` | 15-30 seconds | Quick hooks, TikTok |
| `medium` | 30-60 seconds | Standard viral clips |
| `long` | 60-120 seconds | In-depth content |

---

## Caption Presets

- `viral_gold` - Classic viral style with gold highlight
- `clean_white` - Professional look with blue accent
- `neon_pop` - Bold neon style
- `bold_boxed` - High contrast red emphasis
- `gradient_glow` - Fresh green highlight
