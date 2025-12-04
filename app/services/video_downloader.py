"""
Video Downloader Service - Downloads videos from YouTube or S3.

Uses yt-dlp Python library with rnet browser impersonation to avoid
YouTube's sophisticated bot detection. rnet impersonates real browsers
at the TLS/HTTP2 fingerprint level.
"""

import asyncio
import json
import logging
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Literal, Optional
from urllib.parse import urlparse

import boto3
import yt_dlp
from botocore.exceptions import ClientError

from app.config import get_settings
from app.services.rnet_handler import create_rnet_ydl_opts, is_rnet_available

logger = logging.getLogger(__name__)


# User-Agent rotation list for avoiding detection
UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
]


# Source types for videos
VideoSourceType = Literal["youtube", "s3", "direct_url"]


@dataclass
class VideoMetadata:
    """Metadata extracted from downloaded video."""
    
    title: str
    duration_seconds: float
    width: int
    height: int
    fps: float
    format_id: str
    extractor: str
    uploader: Optional[str] = None
    upload_date: Optional[str] = None
    description: Optional[str] = None
    thumbnail_url: Optional[str] = None
    source_type: VideoSourceType = "youtube"


@dataclass
class DownloadResult:
    """Result of video download operation."""
    
    video_path: str
    metadata: VideoMetadata
    file_size_bytes: int
    source_type: VideoSourceType


class VideoDownloaderService:
    """
    Service for downloading videos from various sources.
    
    Supported sources:
    - YouTube URLs (via yt-dlp)
    - S3 URLs or keys (via boto3)
    - Direct video URLs (via httpx)
    
    Features:
    - Downloads video in best quality up to 1080p
    - Extracts metadata (title, duration, dimensions)
    - Handles various URL formats
    - Configurable via environment variables
    """

    def __init__(self):
        self.settings = get_settings()
        self._s3_client: Optional[boto3.client] = None

        # Log yt-dlp version and rnet availability for diagnostics
        try:
            logger.info(f"VideoDownloaderService initialized with yt-dlp {yt_dlp.version.__version__}")
        except Exception:
            logger.info("VideoDownloaderService initialized with yt-dlp library")

        if is_rnet_available():
            logger.info("rnet browser impersonation is available - YouTube anti-bot bypass enabled")
        else:
            logger.warning("rnet not available - using standard HTTP (may be blocked by YouTube)")

    def _get_format_selector(self) -> str:
        """
        Returns a format selector that prioritizes 1080p quality.
        Uses multiple fallbacks to ensure we get the best available quality.
        """
        # Prioritize 1080p, then 720p, then best available
        # First try exact 1080p, then >=720p, then best available
        return (
            "bestvideo[height=1080]+bestaudio/bestvideo[height>=720]+bestaudio/"
            "bestvideo[height<=1080]+bestaudio/bestvideo+bestaudio/best"
        )

    def _build_ytdlp_opts(
        self,
        output_path: Optional[str] = None,
        download: bool = True,
        use_rnet: bool = True,
    ) -> dict:
        """
        Build yt-dlp options dictionary for downloading.

        Uses rnet browser impersonation when available for better YouTube bypass.

        Args:
            output_path: Optional output file path
            download: Whether these options are for downloading (vs just info extraction)
            use_rnet: Whether to use rnet browser impersonation (default True)

        Returns:
            Dictionary of yt-dlp options
        """
        # Start with rnet-enhanced options for browser impersonation
        if use_rnet and is_rnet_available():
            opts = create_rnet_ydl_opts(proxy=self.settings.ytdlp_proxy)
        else:
            opts = {}
            # Add proxy directly if not using rnet
            if self.settings.ytdlp_proxy:
                opts["proxy"] = self.settings.ytdlp_proxy

        # Add our custom options - keep it simple to avoid format issues
        opts.update({
            "format": self._get_format_selector(),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "http_headers": {
                "User-Agent": random.choice(UA_LIST),
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
            "nocheckcertificate": True,
            "geo_bypass": True,
        })

        if output_path:
            opts["outtmpl"] = output_path

        if download:
            opts["merge_output_format"] = "mp4"
            opts["postprocessors"] = [
                {
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': 'mp4',
                }
            ]
            opts["retries"] = 10
            opts["fragment_retries"] = 10
            opts["force_overwrites"] = True

        return opts

    @property
    def s3_client(self) -> boto3.client:
        """Lazy-initialize S3 client."""
        if self._s3_client is None:
            config = {
                "region_name": self.settings.aws_region,
            }
            if self.settings.aws_access_key_id and self.settings.aws_secret_access_key:
                config["aws_access_key_id"] = self.settings.aws_access_key_id
                config["aws_secret_access_key"] = self.settings.aws_secret_access_key
            
            self._s3_client = boto3.client("s3", **config)
        
        return self._s3_client

    def detect_source_type(self, url_or_key: str) -> VideoSourceType:
        """
        Detect the source type from URL or key.
        
        Args:
            url_or_key: URL or S3 key
            
        Returns:
            VideoSourceType
        """
        # S3 key (no protocol)
        if not url_or_key.startswith("http"):
            return "s3"
        
        parsed = urlparse(url_or_key)
        
        # S3 URL formats
        if parsed.hostname and (
            ".s3." in parsed.hostname or
            parsed.hostname.endswith(".amazonaws.com") or
            parsed.hostname == "s3.amazonaws.com"
        ):
            return "s3"
        
        # YouTube URLs
        youtube_patterns = [
            r"(youtube\.com|youtu\.be)",
            r"youtube\.com/watch",
            r"youtu\.be/",
        ]
        for pattern in youtube_patterns:
            if re.search(pattern, url_or_key, re.IGNORECASE):
                return "youtube"
        
        # Direct video URL
        return "direct_url"

    async def download_video(
        self,
        url: str,
        output_dir: str,
        output_filename: str = "source.mp4",
        max_duration_seconds: Optional[int] = None,
        s3_bucket: Optional[str] = None,
    ) -> DownloadResult:
        """
        Download a video from various sources.
        
        Args:
            url: Video URL (YouTube, S3, direct) or S3 key
            output_dir: Directory to save the video
            output_filename: Output filename (default: source.mp4)
            max_duration_seconds: Maximum duration to download
            s3_bucket: S3 bucket (required if url is an S3 key)
            
        Returns:
            DownloadResult with path and metadata
            
        Raises:
            VideoDownloadError: If download fails
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        
        source_type = self.detect_source_type(url)
        logger.info(f"Detected source type: {source_type} for URL: {url[:100]}...")
        
        if source_type == "s3":
            return await self._download_from_s3(url, output_path, s3_bucket)
        elif source_type == "youtube":
            return await self._download_from_youtube(url, output_path, output_dir, max_duration_seconds)
        else:
            return await self._download_direct_url(url, output_path)

    async def _download_from_youtube(
        self,
        url: str,
        output_path: str,
        output_dir: str,
        max_duration_seconds: Optional[int] = None,
    ) -> DownloadResult:
        """
        Download video from YouTube using yt-dlp Python library.

        Optimized for fast downloads with minimal retries:
        - Uses flexible format selectors that work reliably
        - iOS/web/android player clients (most compatible)
        - Skips problematic DASH/HLS formats
        """
        # First, get video metadata to check duration
        metadata = await self._get_video_info(url)

        max_duration = max_duration_seconds or self.settings.max_download_duration_seconds
        if metadata.duration_seconds > max_duration:
            raise VideoDownloadError(
                f"Video duration ({metadata.duration_seconds}s) exceeds maximum "
                f"allowed duration ({max_duration}s)"
            )

        logger.info(f"Downloading video from YouTube: {url}")
        logger.info(f"Using proxy: {self.settings.ytdlp_proxy or 'None'}")
        logger.info(f"rnet available: {is_rnet_available()}")

        # Format selectors prioritizing 1080p, then 720p, then best available
        format_selectors = [
            # Priority 1: Exact 1080p with audio
            "bestvideo[height=1080]+bestaudio/bestvideo[height=1080][ext=mp4]+bestaudio[ext=m4a]",
            # Priority 2: 720p or higher
            "bestvideo[height>=720]+bestaudio/bestvideo[height>=720][ext=mp4]+bestaudio[ext=m4a]",
            # Priority 3: Best available up to 1080p
            "bestvideo[height<=1080]+bestaudio/bestvideo+bestaudio/best",
            # Final fallback
            "best",
        ]

        # Run download in thread pool to not block event loop
        loop = asyncio.get_event_loop()

        def do_download():
            last_error = None
            
            for fmt_idx, format_selector in enumerate(format_selectors):
                try:
                    logger.info(f"Download attempt {fmt_idx + 1}/{len(format_selectors)} with format: {format_selector[:50]}...")
                    
                    # Build options - start without rnet for speed
                    if fmt_idx == 0:
                        ydl_opts = self._build_ytdlp_opts(output_path=output_path, download=True, use_rnet=True)
                    else:
                        ydl_opts = self._build_ytdlp_opts(output_path=output_path, download=True, use_rnet=False)
                    
                    ydl_opts["format"] = format_selector
                    
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([url])
                    
                    # If we get here, download succeeded
                    return
                    
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    
                    # Check if it's a format issue - try next selector immediately
                    if "Requested format" in error_str or "No video formats" in error_str:
                        logger.warning(f"Format not available, trying next: {error_str[:100]}")
                        continue
                    
                    # For other errors, also continue to next format
                    logger.warning(f"Download attempt {fmt_idx + 1} failed: {error_str[:200]}")
                    continue
            
            # All attempts failed
            raise last_error or VideoDownloadError("All download attempts failed")

        try:
            await loop.run_in_executor(None, do_download)
        except Exception as e:
            raise VideoDownloadError(f"Failed to download video: {e}")

        # Verify output exists
        if not os.path.isfile(output_path):
            # yt-dlp might have added extension
            possible_paths = [
                output_path,
                f"{output_path}.mp4",
                f"{output_path}.webm",
                f"{output_path}.mkv",
            ]
            for path in possible_paths:
                if os.path.isfile(path):
                    if path != output_path:
                        os.rename(path, output_path)
                    break
            else:
                raise VideoDownloadError(f"Download completed but output file not found: {output_path}")

        file_size = os.path.getsize(output_path)
        logger.info(f"Video downloaded: {output_path} ({file_size / 1024 / 1024:.1f} MB)")

        # CRITICAL: Get ACTUAL video metadata using ffprobe after download
        # This ensures we have the real dimensions of the downloaded file,
        # not the pre-download estimates from yt-dlp info
        actual_metadata = await self._get_video_metadata_ffprobe(output_path)

        # Log the actual downloaded resolution for debugging
        logger.info(
            f"Downloaded video quality: {actual_metadata.width}x{actual_metadata.height} "
            f"@ {actual_metadata.fps}fps ({file_size / 1024 / 1024:.1f} MB)"
        )
        
        # Warn if we got low quality (less than 720p)
        if actual_metadata.height < 720:
            logger.warning(
                f"WARNING: Downloaded video is only {actual_metadata.height}p! "
                f"Expected 720p or higher. Proxy: {self.settings.ytdlp_proxy or 'None'}"
            )

        # Preserve useful info from yt-dlp metadata (title, uploader, etc.)
        # but use actual dimensions from ffprobe
        actual_metadata.title = metadata.title
        actual_metadata.uploader = metadata.uploader
        actual_metadata.upload_date = metadata.upload_date
        actual_metadata.description = metadata.description
        actual_metadata.thumbnail_url = metadata.thumbnail_url
        actual_metadata.source_type = "youtube"

        logger.info(f"Actual video dimensions: {actual_metadata.width}x{actual_metadata.height} @ {actual_metadata.fps}fps")

        return DownloadResult(
            video_path=output_path,
            metadata=actual_metadata,
            file_size_bytes=file_size,
            source_type="youtube",
        )

    async def _download_from_s3(
        self,
        url_or_key: str,
        output_path: str,
        s3_bucket: Optional[str] = None,
    ) -> DownloadResult:
        """Download video from S3."""
        # Parse S3 URL or use key directly
        bucket, key = self._parse_s3_url(url_or_key, s3_bucket)
        
        logger.info(f"Downloading video from S3: s3://{bucket}/{key}")
        
        # Download file
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self.s3_client.download_file(bucket, key, output_path),
            )
        except ClientError as e:
            raise VideoDownloadError(f"Failed to download from S3: {e}")
        
        if not os.path.isfile(output_path):
            raise VideoDownloadError(f"S3 download completed but file not found: {output_path}")
        
        file_size = os.path.getsize(output_path)
        logger.info(f"S3 video downloaded: {output_path} ({file_size / 1024 / 1024:.1f} MB)")
        
        # Get video metadata using ffprobe
        metadata = await self._get_video_metadata_ffprobe(output_path)
        metadata.source_type = "s3"
        
        return DownloadResult(
            video_path=output_path,
            metadata=metadata,
            file_size_bytes=file_size,
            source_type="s3",
        )

    def _parse_s3_url(
        self,
        url_or_key: str,
        default_bucket: Optional[str] = None,
    ) -> tuple[str, str]:
        """
        Parse S3 URL or key into bucket and key.
        
        Supports formats:
        - s3://bucket/key
        - https://bucket.s3.region.amazonaws.com/key
        - https://s3.region.amazonaws.com/bucket/key
        - just-a-key (uses default bucket)
        """
        # Plain key
        if not url_or_key.startswith("http") and not url_or_key.startswith("s3://"):
            bucket = default_bucket or self.settings.s3_bucket
            return bucket, url_or_key
        
        # s3:// URL
        if url_or_key.startswith("s3://"):
            parts = url_or_key[5:].split("/", 1)
            if len(parts) != 2:
                raise VideoDownloadError(f"Invalid S3 URL: {url_or_key}")
            return parts[0], parts[1]
        
        # HTTP(S) URL
        parsed = urlparse(url_or_key)
        
        # Virtual-hosted style: bucket.s3.region.amazonaws.com/key
        if parsed.hostname and ".s3." in parsed.hostname:
            bucket = parsed.hostname.split(".s3.")[0]
            key = parsed.path.lstrip("/")
            return bucket, key
        
        # Path style: s3.region.amazonaws.com/bucket/key
        if parsed.hostname and parsed.hostname.startswith("s3."):
            path_parts = parsed.path.lstrip("/").split("/", 1)
            if len(path_parts) != 2:
                raise VideoDownloadError(f"Invalid S3 URL: {url_or_key}")
            return path_parts[0], path_parts[1]
        
        raise VideoDownloadError(f"Unable to parse S3 URL: {url_or_key}")

    async def _download_direct_url(
        self,
        url: str,
        output_path: str,
    ) -> DownloadResult:
        """Download video from a direct URL using httpx."""
        import httpx
        
        logger.info(f"Downloading video from direct URL: {url}")
        
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                
                with open(output_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)
        
        if not os.path.isfile(output_path):
            raise VideoDownloadError(f"Direct download completed but file not found: {output_path}")
        
        file_size = os.path.getsize(output_path)
        logger.info(f"Video downloaded: {output_path} ({file_size / 1024 / 1024:.1f} MB)")
        
        # Get video metadata using ffprobe
        metadata = await self._get_video_metadata_ffprobe(output_path)
        metadata.source_type = "direct_url"
        
        return DownloadResult(
            video_path=output_path,
            metadata=metadata,
            file_size_bytes=file_size,
            source_type="direct_url",
        )

    def _run_ffprobe_sync(self, video_path: str) -> tuple[int, bytes, bytes]:
        """Run ffprobe synchronously (for use with run_in_executor on Windows)."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.returncode, result.stdout, result.stderr

    async def _get_video_metadata_ffprobe(self, video_path: str) -> VideoMetadata:
        """Get video metadata using ffprobe."""
        # Use run_in_executor for Windows compatibility
        # asyncio.create_subprocess_exec doesn't work on Windows without ProactorEventLoop
        loop = asyncio.get_event_loop()
        returncode, stdout, stderr = await loop.run_in_executor(
            None, self._run_ffprobe_sync, video_path
        )
        
        if returncode != 0:
            logger.warning(f"ffprobe failed, using defaults: {stderr.decode()[:200]}")
            return VideoMetadata(
                title=os.path.basename(video_path),
                duration_seconds=0,
                width=1920,
                height=1080,
                fps=30,
                format_id="mp4",
                extractor="file",
            )
        
        try:
            info = json.loads(stdout.decode())
            
            # Find video stream
            video_stream = None
            for stream in info.get("streams", []):
                if stream.get("codec_type") == "video":
                    video_stream = stream
                    break
            
            format_info = info.get("format", {})
            
            # Extract FPS from r_frame_rate (e.g., "30/1" -> 30.0)
            fps = 30.0
            if video_stream and "r_frame_rate" in video_stream:
                fps_str = video_stream["r_frame_rate"]
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    fps = float(num) / float(den) if float(den) != 0 else 30.0
                else:
                    fps = float(fps_str)
            
            return VideoMetadata(
                title=format_info.get("filename", os.path.basename(video_path)),
                duration_seconds=float(format_info.get("duration", 0)),
                width=int(video_stream.get("width", 1920)) if video_stream else 1920,
                height=int(video_stream.get("height", 1080)) if video_stream else 1080,
                fps=fps,
                format_id=format_info.get("format_name", "mp4"),
                extractor="file",
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse ffprobe output: {e}")
            return VideoMetadata(
                title=os.path.basename(video_path),
                duration_seconds=0,
                width=1920,
                height=1080,
                fps=30,
                format_id="mp4",
                extractor="file",
            )

    async def _get_video_info(self, url: str) -> VideoMetadata:
        """
        Get video metadata without downloading using yt-dlp Python library.

        Uses minimal options for maximum compatibility - NO format specification.
        """
        logger.debug(f"Getting video info for: {url}")

        # Run in thread pool to not block event loop
        loop = asyncio.get_event_loop()

        def do_extract():
            # MINIMAL options - don't specify format or complex extractor args
            # This avoids "format not available" errors
            opts = {
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
                "noplaylist": True,
                "socket_timeout": 30,
                "nocheckcertificate": True,
            }
            
            # Add proxy if configured
            if self.settings.ytdlp_proxy:
                opts["proxy"] = self.settings.ytdlp_proxy
            
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(url, download=False)

        try:
            info = await loop.run_in_executor(None, do_extract)
        except Exception as e:
            raise VideoDownloadError(f"Failed to get video info: {e}")

        return VideoMetadata(
            title=info.get("title", "Unknown"),
            duration_seconds=float(info.get("duration", 0)),
            width=int(info.get("width", 1920)),
            height=int(info.get("height", 1080)),
            fps=float(info.get("fps", 30)),
            format_id=info.get("format_id", "unknown"),
            extractor=info.get("extractor", "unknown"),
            uploader=info.get("uploader"),
            upload_date=info.get("upload_date"),
            description=info.get("description"),
            thumbnail_url=info.get("thumbnail"),
        )

class VideoDownloadError(Exception):
    """Exception raised when video download fails."""
    pass

