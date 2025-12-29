"""
Local Storage Service - Handles storing clips and artifacts to local filesystem.

Replaces S3UploadService for local-only deployments.
"""

import asyncio
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class UploadResult:
    """Result of storage operation."""
    
    file_url: str  # Local file path (for compatibility, named as URL)
    file_path: str  # Absolute file path
    file_size_bytes: int
    content_type: str
    
    # Alias for compatibility with code expecting s3_url
    @property
    def s3_url(self) -> str:
        return self.file_url


@dataclass
class ClipArtifact:
    """A clip artifact with metadata."""
    
    clip_index: int
    s3_url: str  # Keep name for compatibility, but stores local path
    duration_ms: int
    start_time_ms: int
    end_time_ms: int
    virality_score: float
    layout_type: str
    summary: Optional[str] = None
    tags: list[str] = None
    
    # Alias for local file access
    @property
    def file_path(self) -> str:
        return self.s3_url


@dataclass
class JobOutput:
    """Complete output of an AI clipping job."""

    job_id: str
    source_video_url: str
    source_video_title: str
    source_video_duration_seconds: float  # Source video duration for credit calculation
    total_clips: int
    clips: list[ClipArtifact]
    user_id: Optional[str] = None  # User ID (not used for local storage)
    transcript_url: Optional[str] = None
    plan_url: Optional[str] = None
    processing_time_seconds: float = 0
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()


class LocalStorageService:
    """
    Service for storing clips and artifacts to local filesystem.
    
    Features:
    - Copies rendered clips to output directory
    - Saves JSON artifacts (transcript, plan, manifest)
    - Organizes files by job_id
    
    Output structure:
        output/{job_id}/
        ├── clip_00.mp4
        ├── clip_01.mp4
        ├── transcript.json
        ├── plan.json
        └── manifest.json
    """

    def __init__(self):
        self.settings = get_settings()
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.settings.output_directory, exist_ok=True)
        logger.info(f"Local storage output directory: {self.settings.output_directory}")

    def _get_job_dir(self, job_id: str) -> str:
        """Get the output directory for a specific job."""
        job_dir = os.path.join(self.settings.output_directory, job_id)
        os.makedirs(job_dir, exist_ok=True)
        return job_dir

    async def upload_clip(
        self,
        local_path: str,
        job_id: str,
        clip_index: int,
        user_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> UploadResult:
        """
        Copy a rendered clip to the output directory.
        
        Args:
            local_path: Path to local clip file
            job_id: Job identifier
            clip_index: Clip index for naming
            user_id: User ID (not used for local storage)
            metadata: Optional metadata (saved alongside clip)
            
        Returns:
            UploadResult with local file path
        """
        if not os.path.isfile(local_path):
            raise LocalStorageError(f"File not found: {local_path}")
        
        job_dir = self._get_job_dir(job_id)
        filename = f"clip_{clip_index:02d}.mp4"
        output_path = os.path.join(job_dir, filename)
        
        logger.info(f"Copying clip to {output_path}")
        
        file_size = os.path.getsize(local_path)
        
        # Copy file (use thread pool for blocking IO)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: shutil.copy2(local_path, output_path),
        )
        
        # Save metadata if provided
        if metadata:
            metadata_path = os.path.join(job_dir, f"clip_{clip_index:02d}_metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Clip saved: {output_path} ({file_size / 1024 / 1024:.1f} MB)")
        
        return UploadResult(
            file_url=output_path,
            file_path=output_path,
            file_size_bytes=file_size,
            content_type="video/mp4",
        )

    async def save_file(self, local_path: str, relative_path: str) -> str:
        """
        Save a file to local storage with an arbitrary relative path.
        
        Args:
            local_path: Path to source file
            relative_path: Relative path in output directory
            
        Returns:
            Absolute path to saved file (as URL)
        """
        output_path = os.path.join(self.settings.output_directory, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: shutil.copy2(local_path, output_path),
        )
        
        return output_path

    async def upload_json_artifact(
        self,
        data: dict[str, Any],
        job_id: str,
        artifact_name: str,
        user_id: Optional[str] = None,
    ) -> UploadResult:
        """
        Save a JSON artifact (transcript, plan, etc.) to local storage.
        
        Args:
            data: Dictionary to serialize as JSON
            job_id: Job identifier
            artifact_name: Name of artifact (e.g., 'transcript', 'plan')
            user_id: User ID (not used for local storage)
            
        Returns:
            UploadResult with local file path
        """
        job_dir = self._get_job_dir(job_id)
        output_path = os.path.join(job_dir, f"{artifact_name}.json")
        
        json_content = json.dumps(data, indent=2, ensure_ascii=False)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: Path(output_path).write_text(json_content, encoding="utf-8"),
        )
        
        logger.info(f"Artifact saved: {output_path}")
        
        return UploadResult(
            file_url=output_path,
            file_path=output_path,
            file_size_bytes=len(json_content.encode("utf-8")),
            content_type="application/json",
        )

    async def upload_job_output(
        self,
        output: JobOutput,
    ) -> UploadResult:
        """
        Save the final job output manifest to local storage.
        
        Args:
            output: JobOutput with all clip details
            
        Returns:
            UploadResult for the manifest file
        """
        # Convert to dict, handling dataclass conversions
        output_dict = asdict(output)
        
        return await self.upload_json_artifact(
            data=output_dict,
            job_id=output.job_id,
            artifact_name="manifest",
            user_id=output.user_id,
        )

    def get_clip_path(self, job_id: str, clip_index: int) -> str:
        """Get the expected path for a clip file."""
        job_dir = self._get_job_dir(job_id)
        return os.path.join(job_dir, f"clip_{clip_index:02d}.mp4")

    def check_file_exists(self, file_path: str) -> bool:
        """Check if a file exists."""
        return os.path.isfile(file_path)


class LocalStorageError(Exception):
    """Exception raised when local storage operation fails."""
    pass
