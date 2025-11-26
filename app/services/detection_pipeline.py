"""
Detection pipeline that orchestrates all detection services.
"""

import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2

from app.config import get_settings
from app.schemas.responses import (
    BoundingBox,
    DetectionResponse,
    FaceDetection,
    FrameDetection,
    PoseDetection,
    ProcessingSummary,
    SourceDimensions,
    TrackSummary,
)
from app.services.face_detector import FaceDetector
from app.services.frame_extractor import FrameExtractor
from app.services.pose_estimator import PoseEstimator
from app.services.s3_client import S3Client
from app.services.tracker import ObjectTracker

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the detection pipeline."""

    frame_interval_seconds: float = 2.0
    detect_faces: bool = True
    detect_poses: bool = True
    start_time_ms: Optional[int] = None
    end_time_ms: Optional[int] = None
    max_frames: int = 1000  # Safety limit


class DetectionPipeline:
    """
    Orchestrates the full detection pipeline:
    1. Download video from S3
    2. Extract frames
    3. Run face detection
    4. Run pose estimation
    5. Track objects across frames
    6. Upload results to S3
    """

    def __init__(
        self,
        face_detector: FaceDetector,
        pose_estimator: PoseEstimator,
        s3_client: Optional[S3Client] = None,
    ):
        """
        Initialize detection pipeline.
        
        Args:
            face_detector: Face detection service
            pose_estimator: Pose estimation service
            s3_client: S3 client (optional, for downloading/uploading)
        """
        self.face_detector = face_detector
        self.pose_estimator = pose_estimator
        self.s3_client = s3_client or S3Client()
        self.frame_extractor = FrameExtractor()
        self.tracker = ObjectTracker()
        
        settings = get_settings()
        self.temp_directory = settings.temp_directory

    async def process_video(
        self,
        job_id: str,
        video_s3_key: str,
        config: PipelineConfig,
    ) -> DetectionResponse:
        """
        Process a video through the full detection pipeline.
        
        Args:
            job_id: Unique job identifier
            video_s3_key: S3 key of the source video
            config: Pipeline configuration
            
        Returns:
            DetectionResponse with all detection results
        """
        start_time = time.time()
        work_dir = None
        
        try:
            # Create working directory
            work_dir = tempfile.mkdtemp(
                dir=self.temp_directory,
                prefix=f"job_{job_id}_",
            )
            logger.info(f"[{job_id}] Working directory: {work_dir}")

            # Step 1: Download video from S3
            logger.info(f"[{job_id}] Downloading video from S3: {video_s3_key}")
            video_path = os.path.join(work_dir, "source.mp4")
            self.s3_client.download_video(video_s3_key, video_path)

            # Step 2: Extract frames
            logger.info(f"[{job_id}] Extracting frames at {config.frame_interval_seconds}s intervals")
            frames_dir = os.path.join(work_dir, "frames")
            extracted_frames, video_metadata = self.frame_extractor.extract_frames(
                video_path=video_path,
                interval_seconds=config.frame_interval_seconds,
                output_directory=frames_dir,
                start_time_ms=config.start_time_ms,
                end_time_ms=config.end_time_ms,
            )
            
            if len(extracted_frames) > config.max_frames:
                logger.warning(f"[{job_id}] Limiting to {config.max_frames} frames")
                extracted_frames = extracted_frames[:config.max_frames]

            logger.info(f"[{job_id}] Extracted {len(extracted_frames)} frames")

            # Step 3 & 4: Run detection on each frame
            frame_detections: List[FrameDetection] = []
            total_faces = 0
            total_poses = 0

            # Reset tracker for new video
            self.tracker.reset()

            for i, frame_info in enumerate(extracted_frames):
                logger.debug(f"[{job_id}] Processing frame {i + 1}/{len(extracted_frames)}")
                
                # Load frame
                frame = cv2.imread(frame_info.file_path)
                if frame is None:
                    logger.warning(f"[{job_id}] Failed to load frame: {frame_info.file_path}")
                    continue

                frame_result = FrameDetection(
                    index=frame_info.index,
                    timestamp_ms=frame_info.timestamp_ms,
                    faces=[],
                    poses=[],
                )

                # Face detection
                if config.detect_faces:
                    face_result = self.face_detector.detect_faces(
                        frame,
                        frame_info.index,
                        frame_info.timestamp_ms,
                    )
                    
                    # Track faces
                    if face_result.detections:
                        face_detections_for_tracking = [
                            (d.bbox, d.confidence) for d in face_result.detections
                        ]
                        tracked_faces = self.tracker.update_faces(
                            face_detections_for_tracking,
                            frame,
                            frame_info.index,
                        )
                        
                        for track_id, bbox, conf in tracked_faces:
                            frame_result.faces.append(FaceDetection(
                                track_id=track_id,
                                bbox=BoundingBox(
                                    x=bbox[0],
                                    y=bbox[1],
                                    width=bbox[2],
                                    height=bbox[3],
                                ),
                                confidence=conf,
                            ))
                        
                        total_faces += len(tracked_faces)

                # Pose detection
                if config.detect_poses:
                    pose_result = self.pose_estimator.estimate_pose(
                        frame,
                        frame_info.index,
                        frame_info.timestamp_ms,
                    )
                    
                    # Track poses
                    if pose_result.detections:
                        pose_detections_for_tracking = [
                            (d.bounding_box, d.confidence)
                            for d in pose_result.detections
                            if d.bounding_box is not None
                        ]
                        tracked_poses = self.tracker.update_poses(
                            pose_detections_for_tracking,
                            frame,
                            frame_info.index,
                        )
                        
                        for idx, (track_id, bbox, conf) in enumerate(tracked_poses):
                            if idx < len(pose_result.detections):
                                pose_data = pose_result.detections[idx]
                                frame_result.poses.append(PoseDetection(
                                    track_id=track_id,
                                    keypoints=pose_data.keypoints,
                                    confidence=conf,
                                    gesture=pose_data.gesture,
                                ))
                        
                        total_poses += len(tracked_poses)

                frame_detections.append(frame_result)

            # Step 5: Generate track summaries
            track_summaries = self._generate_track_summaries()

            # Step 6: Upload results to S3
            results_s3_key = self._generate_results_key(video_s3_key, job_id)
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            response = DetectionResponse(
                job_id=job_id,
                status="completed",
                source_dimensions=SourceDimensions(
                    width=video_metadata.width,
                    height=video_metadata.height,
                ),
                frame_interval_ms=int(config.frame_interval_seconds * 1000),
                frames=frame_detections,
                tracks=track_summaries,
                summary=ProcessingSummary(
                    total_frames=len(frame_detections),
                    faces_detected=total_faces,
                    poses_detected=total_poses,
                    unique_face_tracks=len([t for t in track_summaries if t.track_type == "face"]),
                    unique_pose_tracks=len([t for t in track_summaries if t.track_type == "pose"]),
                    processing_time_ms=processing_time_ms,
                ),
                results_s3_key=results_s3_key,
            )

            # Upload full results to S3
            results_dict = response.model_dump()
            self.s3_client.upload_json(results_dict, results_s3_key)
            
            logger.info(
                f"[{job_id}] Pipeline complete: "
                f"{len(frame_detections)} frames, "
                f"{total_faces} faces, "
                f"{total_poses} poses, "
                f"{processing_time_ms}ms"
            )

            return response

        except Exception as e:
            logger.error(f"[{job_id}] Pipeline failed: {e}", exc_info=True)
            return DetectionResponse(
                job_id=job_id,
                status="failed",
                source_dimensions=SourceDimensions(width=0, height=0),
                frame_interval_ms=int(config.frame_interval_seconds * 1000),
                frames=[],
                tracks=[],
                summary=ProcessingSummary(
                    total_frames=0,
                    faces_detected=0,
                    poses_detected=0,
                    unique_face_tracks=0,
                    unique_pose_tracks=0,
                    processing_time_ms=int((time.time() - start_time) * 1000),
                ),
                error=str(e),
            )

        finally:
            # Cleanup working directory
            if work_dir and os.path.exists(work_dir):
                try:
                    shutil.rmtree(work_dir)
                    logger.debug(f"[{job_id}] Cleaned up working directory")
                except Exception as e:
                    logger.warning(f"[{job_id}] Failed to cleanup: {e}")

    def process_video_sync(
        self,
        job_id: str,
        video_s3_key: str,
        config: PipelineConfig,
    ) -> DetectionResponse:
        """
        Synchronous version of process_video for non-async contexts.
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.process_video(job_id, video_s3_key, config)
        )

    def _generate_track_summaries(self) -> List[TrackSummary]:
        """Generate track summaries from tracker state."""
        summaries = []
        
        for track in self.tracker.get_all_tracks():
            avg_bbox = track.avg_bbox
            summaries.append(TrackSummary(
                track_id=track.track_id,
                track_type=track.track_type,
                first_frame=track.first_frame,
                last_frame=track.last_frame,
                frame_count=track.frame_count,
                avg_bbox=BoundingBox(
                    x=avg_bbox[0],
                    y=avg_bbox[1],
                    width=avg_bbox[2],
                    height=avg_bbox[3],
                ) if avg_bbox != (0, 0, 0, 0) else None,
                avg_confidence=track.avg_confidence,
            ))
        
        return summaries

    def _generate_results_key(self, video_s3_key: str, job_id: str) -> str:
        """Generate S3 key for results JSON."""
        # Extract user path from video key
        # e.g., "users/user123/videos/source.mp4" -> "users/user123/detection/job_id.json"
        parts = video_s3_key.split("/")
        if len(parts) >= 2:
            base_path = "/".join(parts[:-2])  # Remove "videos/filename"
            return f"{base_path}/detection/{job_id}.json"
        
        return f"detection/{job_id}.json"

