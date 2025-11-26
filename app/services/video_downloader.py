"""
Video Downloader Service - Downloads videos from YouTube or S3.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

from app.config import get_settings

logger = logging.getLogger(__name__)


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
        self.ytdlp_path = self._resolve_ytdlp_path()
        self._s3_client: Optional[boto3.client] = None
        logger.info(f"VideoDownloaderService initialized with yt-dlp at: {self.ytdlp_path}")

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

    def _resolve_ytdlp_path(self) -> str:
        """Resolve the path to yt-dlp binary."""
        configured_path = self.settings.ytdlp_path
        
        # Check if configured path exists
        if os.path.isfile(configured_path):
            return configured_path
        
        # Check if it's in PATH
        which_result = shutil.which(configured_path)
        if which_result:
            return which_result
        
        # Fallback to 'yt-dlp' in PATH
        fallback = shutil.which("yt-dlp")
        if fallback:
            return fallback
        
        raise RuntimeError(
            f"yt-dlp not found. Checked: {configured_path}, PATH. "
            "Install with: pip install yt-dlp"
        )

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
        """Download video from YouTube using yt-dlp."""
        # First, get video metadata to check duration
        metadata = await self._get_video_info(url)
        
        max_duration = max_duration_seconds or self.settings.max_download_duration_seconds
        if metadata.duration_seconds > max_duration:
            raise VideoDownloadError(
                f"Video duration ({metadata.duration_seconds}s) exceeds maximum "
                f"allowed duration ({max_duration}s)"
            )
        
        # Build yt-dlp command
        args = self._build_download_args(url, output_path)
        
        logger.info(f"Downloading video from YouTube: {url}")
        logger.debug(f"yt-dlp args: {args}")
        
        # Run download
        await self._run_ytdlp(args, output_dir)
        
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
        
        metadata.source_type = "youtube"
        
        return DownloadResult(
            video_path=output_path,
            metadata=metadata,
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

    async def _get_video_metadata_ffprobe(self, video_path: str) -> VideoMetadata:
        """Get video metadata using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
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

    def _build_download_args(self, url: str, output_path: str) -> list[str]:
        """Build yt-dlp command arguments."""
        args = [
            self.ytdlp_path,
            "--no-playlist",
            "--no-progress",
            "--newline",
            "--ignore-config",
            "--force-overwrites",
            "--retries", "5",
            "--fragment-retries", "25",
            "--concurrent-fragments", "4",
            # Format selection: best video up to 1080p with best audio
            # Explicitly exclude AV1 codec (vcodec!*=av01) as it's not supported
            # on many platforms without hardware acceleration
            "--format", "bv*[ext=mp4][vcodec!*=av01][height<=1080]+ba[ext=m4a]/bv*[ext=mp4][height<=1080]+ba/b[vcodec!*=av01]/b",
            "--merge-output-format", "mp4",
            "--output", output_path,
        ]
        
        # Add cookies from browser if configured
        if self.settings.ytdlp_cookies_from_browser:
            args.extend(["--cookies-from-browser", self.settings.ytdlp_cookies_from_browser])
        
        # Add extra args from config
        extra_args = self.settings.get_ytdlp_extra_args()
        if extra_args:
            args.extend(extra_args)
        
        # Add URL last
        args.append(url)
        
        return args

    async def _get_video_info(self, url: str) -> VideoMetadata:
        """Get video metadata without downloading."""
        args = [
            self.ytdlp_path,
            "--no-download",
            "--dump-json",
            "--no-playlist",
            url,
        ]
        
        logger.debug(f"Getting video info for: {url}")
        
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode()[:500] if stderr else "Unknown error"
            raise VideoDownloadError(f"Failed to get video info: {error_msg}")
        
        try:
            info = json.loads(stdout.decode())
        except json.JSONDecodeError as e:
            raise VideoDownloadError(f"Failed to parse video info: {e}")
        
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

    async def _run_ytdlp(self, args: list[str], cwd: str) -> None:
        """Run yt-dlp command asynchronously."""
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        
        # Set timeout
        timeout = self.settings.max_download_duration_seconds + 300  # Extra 5 min buffer
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            raise VideoDownloadError(f"Download timed out after {timeout} seconds")
        
        if process.returncode != 0:
            error_msg = stderr.decode()[-1000:] if stderr else "Unknown error"
            raise VideoDownloadError(f"yt-dlp failed with code {process.returncode}: {error_msg}")
        
        # Log any warnings from stderr
        if stderr:
            stderr_text = stderr.decode()
            for line in stderr_text.split("\n"):
                line = line.strip()
                if line and "WARNING" in line:
                    logger.warning(f"yt-dlp: {line}")


class VideoDownloadError(Exception):
    """Exception raised when video download fails."""
    pass

