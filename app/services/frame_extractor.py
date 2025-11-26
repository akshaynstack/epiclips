"""
Frame extraction service using FFmpeg.
"""

import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFrame:
    """Represents an extracted frame with metadata."""

    index: int
    timestamp_ms: int
    file_path: str
    width: int
    height: int


@dataclass
class VideoMetadata:
    """Video file metadata."""

    duration_ms: int
    width: int
    height: int
    fps: float
    codec: str


class FrameExtractor:
    """
    Service for extracting frames from videos using FFmpeg.
    
    Extracts frames at configurable intervals for ML processing.
    """

    def __init__(self, temp_directory: Optional[str] = None):
        """
        Initialize frame extractor.
        
        Args:
            temp_directory: Directory for temporary frame files
        """
        self.temp_directory = temp_directory or tempfile.gettempdir()
        os.makedirs(self.temp_directory, exist_ok=True)

    def get_video_metadata(self, video_path: str) -> VideoMetadata:
        """
        Get metadata for a video file using ffprobe.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            VideoMetadata object with video properties
        """
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration,r_frame_rate,codec_name",
            "-show_entries", "format=duration",
            "-of", "csv=p=0:s=,",
            video_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout.strip()
            
            # Parse output - format: width,height,duration,fps,codec or similar
            lines = output.split("\n")
            
            # Try to extract values
            width = 1920
            height = 1080
            duration_s = 0.0
            fps = 30.0
            codec = "unknown"

            for line in lines:
                parts = line.split(",")
                for i, part in enumerate(parts):
                    part = part.strip()
                    if part.isdigit():
                        val = int(part)
                        if val > 100 and val <= 7680:  # Likely width or height
                            if width == 1920:
                                width = val
                            elif height == 1080:
                                height = val
                    elif "/" in part:  # FPS as fraction
                        try:
                            num, den = part.split("/")
                            fps = float(num) / float(den)
                        except (ValueError, ZeroDivisionError):
                            pass
                    elif "." in part:  # Duration as float
                        try:
                            duration_s = float(part)
                        except ValueError:
                            pass
                    elif part.isalpha():  # Codec name
                        codec = part

            # Fallback: use OpenCV if ffprobe parsing fails
            if duration_s == 0:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration_s = frame_count / fps if fps > 0 else 0
                    cap.release()

            return VideoMetadata(
                duration_ms=int(duration_s * 1000),
                width=width,
                height=height,
                fps=fps,
                codec=codec,
            )

        except subprocess.CalledProcessError as e:
            logger.warning(f"ffprobe failed, using OpenCV fallback: {e}")
            # Fallback to OpenCV
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_ms = int((frame_count / fps) * 1000) if fps > 0 else 0
            cap.release()

            return VideoMetadata(
                duration_ms=duration_ms,
                width=width,
                height=height,
                fps=fps,
                codec="unknown",
            )

    def extract_frames(
        self,
        video_path: str,
        interval_seconds: float = 2.0,
        output_directory: Optional[str] = None,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        max_dimension: int = 1280,
    ) -> Tuple[List[ExtractedFrame], VideoMetadata]:
        """
        Extract frames from a video at specified intervals.
        
        Args:
            video_path: Path to the video file
            interval_seconds: Interval between frame extractions
            output_directory: Directory to save frames (uses temp if not provided)
            start_time_ms: Optional start time in milliseconds
            end_time_ms: Optional end time in milliseconds
            max_dimension: Maximum dimension for extracted frames (for efficiency)
            
        Returns:
            Tuple of (list of ExtractedFrame objects, VideoMetadata)
        """
        # Get video metadata
        metadata = self.get_video_metadata(video_path)
        logger.info(
            f"Video: {metadata.width}x{metadata.height}, "
            f"{metadata.duration_ms}ms, {metadata.fps:.2f}fps"
        )

        # Create output directory
        if output_directory is None:
            output_directory = tempfile.mkdtemp(dir=self.temp_directory, prefix="frames_")
        os.makedirs(output_directory, exist_ok=True)

        # Calculate scale filter to limit resolution
        scale_filter = self._calculate_scale_filter(
            metadata.width, metadata.height, max_dimension
        )

        # Build FFmpeg command
        output_pattern = os.path.join(output_directory, "frame_%05d.jpg")
        
        cmd = ["ffmpeg", "-y", "-i", video_path]
        
        # Add time range if specified
        if start_time_ms is not None:
            cmd.extend(["-ss", f"{start_time_ms / 1000:.3f}"])
        if end_time_ms is not None:
            duration = end_time_ms - (start_time_ms or 0)
            cmd.extend(["-t", f"{duration / 1000:.3f}"])

        # Frame extraction filter
        fps_filter = f"fps=1/{interval_seconds}"
        if scale_filter:
            filter_complex = f"{fps_filter},{scale_filter}"
        else:
            filter_complex = fps_filter

        cmd.extend([
            "-vf", filter_complex,
            "-q:v", "3",  # JPEG quality (2-31, lower is better)
            output_pattern,
        ])

        logger.info(f"Extracting frames with command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.debug(f"FFmpeg stdout: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e.stderr}")
            raise RuntimeError(f"Frame extraction failed: {e.stderr}")

        # Collect extracted frames
        frames = []
        frame_files = sorted(
            [f for f in os.listdir(output_directory) if f.startswith("frame_")]
        )

        for i, filename in enumerate(frame_files):
            file_path = os.path.join(output_directory, filename)
            timestamp_ms = int(i * interval_seconds * 1000)
            if start_time_ms:
                timestamp_ms += start_time_ms

            # Get actual frame dimensions
            img = cv2.imread(file_path)
            if img is not None:
                h, w = img.shape[:2]
                frames.append(ExtractedFrame(
                    index=i,
                    timestamp_ms=timestamp_ms,
                    file_path=file_path,
                    width=w,
                    height=h,
                ))

        logger.info(f"Extracted {len(frames)} frames to {output_directory}")
        return frames, metadata

    def extract_frames_in_memory(
        self,
        video_path: str,
        interval_seconds: float = 2.0,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        max_dimension: int = 1280,
    ) -> Tuple[List[Tuple[int, np.ndarray]], VideoMetadata]:
        """
        Extract frames directly into memory (as numpy arrays).
        
        More efficient for short videos or when disk I/O is expensive.
        
        Args:
            video_path: Path to the video file
            interval_seconds: Interval between frame extractions
            start_time_ms: Optional start time
            end_time_ms: Optional end time
            max_dimension: Maximum dimension for frames
            
        Returns:
            Tuple of (list of (timestamp_ms, frame_array) tuples, VideoMetadata)
        """
        metadata = self.get_video_metadata(video_path)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = int(fps * interval_seconds)
        
        start_frame = 0
        if start_time_ms is not None:
            start_frame = int((start_time_ms / 1000) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        end_frame = None
        if end_time_ms is not None:
            end_frame = int((end_time_ms / 1000) * fps)

        frames = []
        current_frame = start_frame

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if end_frame and current_frame > end_frame:
                break

            if (current_frame - start_frame) % frame_interval == 0:
                # Resize if needed
                h, w = frame.shape[:2]
                if max(w, h) > max_dimension:
                    scale = max_dimension / max(w, h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h))

                timestamp_ms = int((current_frame / fps) * 1000)
                frames.append((timestamp_ms, frame))

            current_frame += 1

        cap.release()
        logger.info(f"Extracted {len(frames)} frames in memory")
        return frames, metadata

    def _calculate_scale_filter(
        self, width: int, height: int, max_dimension: int
    ) -> Optional[str]:
        """Calculate FFmpeg scale filter string."""
        if max(width, height) <= max_dimension:
            return None

        if width > height:
            return f"scale={max_dimension}:-2"
        else:
            return f"scale=-2:{max_dimension}"

    def cleanup_frames(self, frames: List[ExtractedFrame]) -> None:
        """
        Clean up extracted frame files.
        
        Args:
            frames: List of ExtractedFrame objects to clean up
        """
        directories = set()
        for frame in frames:
            try:
                if os.path.exists(frame.file_path):
                    os.remove(frame.file_path)
                    directories.add(os.path.dirname(frame.file_path))
            except OSError as e:
                logger.warning(f"Failed to remove frame file {frame.file_path}: {e}")

        # Try to remove empty directories
        for directory in directories:
            try:
                if os.path.exists(directory) and not os.listdir(directory):
                    os.rmdir(directory)
            except OSError:
                pass

