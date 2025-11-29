# YouTube Proxy Configuration & Datacenter Blocking Fix

## Overview
**Date Implemented:** November 29, 2025
**Issue:** YouTube aggressively blocks AWS Datacenter IPs with the error: `Sign in to confirm you’re not a bot`.
**Solution:** Route `yt-dlp` traffic through **ISP Proxies** (Static Residential IPs) which appear as legitimate home/business connections.

## Configuration Guide

### 1. Environment Variable Format
The system uses the `YTDLP_PROXY` environment variable. The format depends on the protocol selected in your proxy provider dashboard.

#### Option A: SOCKS5 (Recommended)
Use this if your dashboard shows "SOCKS5" and a specific SOCKS port (e.g., 12324).
**Crucial:** Use `socks5h://` to perform DNS resolution at the proxy side. This hides your AWS DNS usage from YouTube.

**Format:**
```bash
YTDLP_PROXY=socks5h://username:password@host:port
```

**Example (based on IPRoyal):**
If IPRoyal gives you:
- Host: `92.113.40.159`
- Port: `12324`
- User: `user123`
- Pass: `pass456`

Set this:
```bash
YTDLP_PROXY=socks5h://user123:pass456@92.113.40.159:12324
```

#### Option B: HTTP (Standard)
Use this if your dashboard shows "HTTP/HTTPS" (usually port 12323).

**Format:**
```bash
YTDLP_PROXY=http://username:password@host:port
```

### 2. Codebase Integration
*   **File:** `app/services/video_downloader.py`
*   **Logic:** Automatically injects `--proxy` and anti-bot headers (`User-Agent`, `Referer`, `Sleep`) when `YTDLP_PROXY` is set.

## Troubleshooting

### "Sign in to confirm you’re not a bot"
1.  **Check Protocol:** Ensure you aren't using the HTTP port (12323) with `socks5://` scheme, or vice versa.
2.  **Use Remote DNS:** Always use `socks5h://` instead of `socks5://`.
3.  **Rotate IP:** If configured correctly and still blocked, the specific IP is likely flagged. Rotate to the next IP in your pool.

### IPRoyal Specifics
*   **HTTP Port:** Usually `12323`
*   **SOCKS5 Port:** Usually `12324`
*   **Copy List Format:** `HOST:PORT:USER:PASS` -> Requires rearranging to URL format.
