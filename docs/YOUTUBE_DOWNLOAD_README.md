# YouTube Video Download System

## Overview

This document describes the YouTube video download system implemented in viewcreator-genesis, which uses advanced browser impersonation techniques to bypass YouTube's sophisticated bot detection.

## The Problem: YouTube Bot Detection

YouTube employs multi-layered bot detection that examines:

1. **TLS Fingerprints**: Cipher suites, extensions, ALPN negotiation, and TLS version
2. **HTTP/2 Settings**: Initial window size, header table size, max concurrent streams
3. **HTTP/2 Frame Ordering**: The sequence in which HTTP/2 frames are sent
4. **User-Agent Strings**: Browser identification headers
5. **Behavioral Patterns**: Request timing, navigation patterns

Standard Python HTTP libraries (urllib, requests, httpx) have completely different TLS/HTTP2 fingerprints than real browsers, making them easily detectable. This results in "Sign in to confirm you're not a bot" errors.

## The Solution: rnet Browser Impersonation

We use the `rnet` library (Rust-based HTTP client with Python bindings) to impersonate real browsers at the TLS/HTTP2 fingerprint level. This makes requests indistinguishable from actual browser traffic.

### Supported Browser Impersonations

The system rotates through the following browser profiles:

| Browser | Version | Impersonation Key |
|---------|---------|-------------------|
| Chrome | 130 | `Impersonate.Chrome130` |
| Chrome | 129 | `Impersonate.Chrome129` |
| Chrome | 128 | `Impersonate.Chrome128` |
| Chrome | 127 | `Impersonate.Chrome127` |
| Chrome | 126 | `Impersonate.Chrome126` |
| Edge | 131 | `Impersonate.Edge131` |
| Edge | 134 | `Impersonate.Edge134` |
| Firefox | 136 | `Impersonate.Firefox136` |
| Firefox | 139 | `Impersonate.Firefox139` |
| Safari | 18.3 | `Impersonate.Safari18_3` |

## Architecture

### File Structure

```
app/services/
├── rnet_handler.py      # rnet browser impersonation handler
├── video_downloader.py  # Main video download service
└── ...
```

### Components

#### 1. RnetHttpHandler (`rnet_handler.py`)

The core browser impersonation handler that:
- Creates rnet clients with browser impersonation
- Rotates through browser profiles on each request
- Supports proxy configuration (SOCKS5, HTTP)
- Provides async HTTP methods (GET, POST, HEAD)

```python
class RnetHttpHandler:
    def __init__(self, proxy: Optional[str] = None):
        self.proxy = proxy
        self.impersonations = [
            Impersonate.Chrome130,
            Impersonate.Chrome129,
            # ... more browsers
        ]
```

#### 2. RnetUrllibHandler (`rnet_handler.py`)

A urllib-compatible handler that wraps RnetHttpHandler for use with yt-dlp:

```python
class RnetUrllibHandler(urllib.request.BaseHandler):
    def http_open(self, req):
        # Routes requests through rnet
    def https_open(self, req):
        # Routes HTTPS requests through rnet
```

#### 3. VideoDownloaderService (`video_downloader.py`)

The main service that orchestrates video downloads with:
- rnet integration for YouTube anti-bot bypass
- Three-tier fallback mechanism
- High-quality format selection (1080p priority)
- ffprobe-based metadata extraction

## Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `YTDLP_PROXY` | Proxy URL for YouTube requests | `socks5h://user:pass@host:port` |
| `MAX_DOWNLOAD_DURATION_SECONDS` | Maximum video duration to download | `600` |

### Docker Compose Configuration

```yaml
services:
  genesis:
    environment:
      - YTDLP_PROXY=${YTDLP_PROXY:-}
```

### Proxy Support

The system supports various proxy types:
- **SOCKS5**: `socks5://user:pass@host:port`
- **SOCKS5H** (DNS through proxy): `socks5h://user:pass@host:port`
- **HTTP**: `http://user:pass@host:port`
- **HTTPS**: `https://user:pass@host:port`

## Download Flow

### 1. Request Initiation

```
User Request → VideoDownloaderService.download_video()
                        ↓
              detect_source_type(url)
                        ↓
              _download_from_youtube()
```

### 2. Metadata Extraction

```
_get_video_info(url)
        ↓
_build_ytdlp_opts(use_rnet=True)
        ↓
create_rnet_ydl_opts(proxy)
        ↓
yt_dlp.extract_info() with rnet handlers
```

### 3. Video Download

```
_build_ytdlp_opts(download=True, use_rnet=True)
        ↓
yt_dlp.download() with rnet handlers
        ↓
ffprobe for actual metadata
        ↓
Return DownloadResult
```

## Fallback Mechanism

The system implements a three-tier fallback for resilience:

### Tier 1: rnet with Full Options
- Browser impersonation via rnet
- Proxy support
- Multiple player clients
- Comprehensive HTTP headers
- High-quality format selection

### Tier 2: Standard yt-dlp (No rnet)
- Falls back if rnet fails
- Retains proxy configuration
- Retains player clients and headers
- Same format selection

### Tier 3: Emergency Fallback
- Minimal options
- Basic format selection (`best[height>=720]/best`)
- Proxy only if configured

```python
def do_download():
    try:
        # Tier 1: rnet with full options
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        try:
            # Tier 2: No rnet, keep other options
            fallback_opts = self._build_ytdlp_opts(use_rnet=False)
            with yt_dlp.YoutubeDL(fallback_opts) as ydl:
                ydl.download([url])
        except Exception as e2:
            # Tier 3: Emergency minimal options
            emergency_opts = {"format": "best[height>=720]/best", ...}
            with yt_dlp.YoutubeDL(emergency_opts) as ydl:
                ydl.download([url])
```

## Format Selection

The system prioritizes high-quality video with the following format selector chain:

```python
format_selectors = [
    # 1. Best 1080p h264 video + AAC audio
    "bestvideo[height<=1080][vcodec^=avc]+bestaudio[acodec^=mp4a]/bestvideo[height<=1080]+bestaudio",

    # 2. Best 1080p video + best audio
    "bestvideo[height=1080]+bestaudio/bestvideo[height<=1080]+bestaudio",

    # 3. Best 720p video + best audio (fallback)
    "bestvideo[height=720]+bestaudio/bestvideo[height<=720]+bestaudio",

    # 4. Best combined format
    "best[height<=1080][vcodec^=avc]/best[height=1080]/best[height<=1080]",

    # 5. Best high quality available
    "best[height>=720]"
]
```

### Format Selection Priorities

1. **Codec Preference**: H.264/AVC preferred over VP9/AV1 for compatibility
2. **Resolution**: 1080p prioritized, 720p as fallback
3. **Audio**: AAC (mp4a) preferred for MP4 container compatibility
4. **Container**: MP4 output with FFmpeg post-processing

## yt-dlp Options

### Core Options

| Option | Value | Purpose |
|--------|-------|---------|
| `format` | Format selector chain | Quality selection |
| `merge_output_format` | `mp4` | Output container |
| `noplaylist` | `True` | Single video only |
| `quiet` | `True` | Suppress output |
| `no_warnings` | `True` | Suppress warnings |

### Anti-Detection Options

| Option | Value | Purpose |
|--------|-------|---------|
| `http_handler` | RnetUrllibHandler | Browser impersonation |
| `https_handler` | RnetUrllibHandler | HTTPS impersonation |
| `http_headers` | Browser-like headers | Request authenticity |
| `extractor_args` | Multiple player clients | Fallback sources |

### Player Clients

```python
"extractor_args": {
    "youtube": {
        "player_client": ["web_embedded", "web", "tv", "android", "ios"]
    }
}
```

### HTTP Headers

```python
"http_headers": {
    "User-Agent": "<rotated from UA list>",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}
```

## Metadata Extraction

### Pre-Download (yt-dlp)
- Title, uploader, upload date
- Description, thumbnail URL
- Duration (for max duration check)

### Post-Download (ffprobe)
- **Actual** video dimensions (width, height)
- **Actual** frame rate (fps)
- **Actual** duration
- Codec information

This two-stage approach ensures accurate metadata, as yt-dlp's pre-download info may not match the actual downloaded format.

## Verification

### Check rnet Availability

```bash
docker exec viewcreator-genesis python -c "
from app.services.rnet_handler import is_rnet_available
print('rnet available:', is_rnet_available())
"
```

### Check Proxy Configuration

```bash
docker exec viewcreator-genesis printenv | grep YTDLP_PROXY
```

### Test rnet Handler

```bash
docker exec viewcreator-genesis python -c "
from app.services.rnet_handler import create_rnet_ydl_opts
opts = create_rnet_ydl_opts()
print('Handler keys:', list(opts.keys()))
"
```

## Troubleshooting

### "Sign in to confirm you're not a bot" Error

1. **Check rnet availability**: Ensure rnet is installed and available
2. **Check proxy**: Verify YTDLP_PROXY is set and accessible
3. **Rotate IP**: The proxy IP may be flagged; try a different one
4. **Check logs**: Look for fallback tier activations

### Low Quality Downloads

1. **Format selector**: Verify format selector prioritizes 1080p
2. **Source availability**: Some videos may not have 1080p
3. **ffprobe check**: Verify actual downloaded resolution with ffprobe

### Proxy Not Working

1. **Environment variable**: Ensure YTDLP_PROXY is set in host environment
2. **Docker compose**: Verify YTDLP_PROXY is mapped in docker-compose.yml
3. **Container check**: Run `printenv | grep PROXY` inside container
4. **Proxy format**: Ensure correct format (socks5h://user:pass@host:port)

## Dependencies

### Python Packages

```
yt-dlp>=2024.1.0     # Video download library
rnet                  # Browser impersonation (Rust-based)
```

### System Dependencies

```
ffmpeg               # Video processing and format conversion
ffprobe              # Video metadata extraction
```

## Security Considerations

1. **Proxy credentials**: Never commit proxy credentials to version control
2. **Environment variables**: Use environment variables for sensitive configuration
3. **Rate limiting**: Respect YouTube's terms of service
4. **IP rotation**: Consider rotating proxies for heavy usage

## Performance

- **Browser rotation**: Distributes requests across different browser profiles
- **Concurrent fragments**: Downloads video fragments in parallel (`concurrent_fragments: 4`)
- **Retry logic**: Automatic retries with exponential backoff
- **Connection reuse**: HTTP/2 connection multiplexing through rnet

## Related Files

| File | Description |
|------|-------------|
| `app/services/video_downloader.py` | Main download service |
| `app/services/rnet_handler.py` | rnet browser impersonation |
| `app/config.py` | Configuration and settings |
| `requirements.txt` | Python dependencies |
| `docker-compose.yml` | Container configuration |
