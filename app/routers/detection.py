"""
Detection API endpoints.
"""

import asyncio
import logging
from typing import Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from app.config import get_settings
from app.schemas.requests import DetectionRequest
from app.schemas.responses import DetectionResponse, ProcessingSummary, SourceDimensions
from app.services.detection_pipeline import DetectionPipeline, PipelineConfig
from app.services.s3_client import S3Client

logger = logging.getLogger(__name__)

router = APIRouter()

# Track active jobs for status endpoint
active_jobs: dict[str, str] = {}  # job_id -> status
queued_jobs: int = 0


def get_pipeline(request: Request) -> DetectionPipeline:
    """Get detection pipeline with services from app state."""
    face_detector = getattr(request.app.state, "face_detector", None)
    pose_estimator = getattr(request.app.state, "pose_estimator", None)
    
    if face_detector is None or pose_estimator is None:
        raise HTTPException(
            status_code=503,
            detail="Detection models not loaded. Service not ready.",
        )
    
    return DetectionPipeline(
        face_detector=face_detector,
        pose_estimator=pose_estimator,
    )


async def send_callback(callback_url: str, response: DetectionResponse) -> None:
    """Send detection results to callback URL."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                callback_url,
                json=response.model_dump(),
                timeout=30.0,
            )
        logger.info(f"Callback sent to {callback_url}")
    except Exception as e:
        logger.error(f"Failed to send callback to {callback_url}: {e}")


async def process_detection_job(
    request: DetectionRequest,
    pipeline: DetectionPipeline,
    semaphore: asyncio.Semaphore,
) -> DetectionResponse:
    """
    Process a detection job with concurrency control.
    
    Uses a semaphore to limit concurrent jobs and prevent resource exhaustion.
    Jobs that can't acquire the semaphore immediately will wait in queue.
    """
    global active_jobs, queued_jobs
    
    job_id = request.job_id
    
    # Mark as queued while waiting for semaphore
    queued_jobs += 1
    active_jobs[job_id] = "queued"
    logger.info(f"[{job_id}] Job queued (waiting for slot). Queue depth: {queued_jobs}")
    
    try:
        # Wait for a slot (blocks if max concurrent jobs reached)
        async with semaphore:
            queued_jobs -= 1
            active_jobs[job_id] = "processing"
            logger.info(f"[{job_id}] Job started processing")
            
            config = PipelineConfig(
                frame_interval_seconds=request.frame_interval_seconds,
                detect_faces=request.detect_faces,
                detect_poses=request.detect_poses,
                start_time_ms=request.start_time_ms,
                end_time_ms=request.end_time_ms,
            )
            
            result = await pipeline.process_video(
                job_id=job_id,
                video_s3_key=request.video_s3_key,
                config=config,
            )
            
            active_jobs[job_id] = "completed"
            return result
            
    except Exception as e:
        active_jobs[job_id] = "failed"
        raise
    finally:
        # Clean up job tracking after a delay
        async def cleanup():
            await asyncio.sleep(60)  # Keep status for 1 minute
            active_jobs.pop(job_id, None)
        asyncio.create_task(cleanup())


def get_semaphore(request: Request) -> asyncio.Semaphore:
    """Get job semaphore from app state."""
    semaphore = getattr(request.app.state, "job_semaphore", None)
    if semaphore is None:
        # Fallback: create a semaphore with default limit
        settings = get_settings()
        return asyncio.Semaphore(settings.max_concurrent_jobs)
    return semaphore


@router.get("/status")
async def get_queue_status() -> dict:
    """
    Get the current job queue status.
    
    Returns:
        Current queue depth, active jobs, and their statuses
    """
    settings = get_settings()
    return {
        "max_concurrent_jobs": settings.max_concurrent_jobs,
        "queued_jobs": queued_jobs,
        "active_jobs": len([s for s in active_jobs.values() if s == "processing"]),
        "jobs": active_jobs.copy(),
    }


@router.post("", response_model=DetectionResponse)
async def detect(
    request: DetectionRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
) -> DetectionResponse:
    """
    Run detection on a video.
    
    Jobs are processed with concurrency control (default: 2 concurrent jobs).
    If more jobs arrive than can be processed, they queue and wait.
    
    If callback_url is provided, returns immediately and sends results via callback.
    Otherwise, processes synchronously and returns results.
    
    Args:
        request: Detection request with video location and options
        background_tasks: FastAPI background tasks for async processing
        http_request: HTTP request for accessing app state
        
    Returns:
        DetectionResponse with detection results or processing status
    """
    logger.info(f"Detection request received: job_id={request.job_id}")
    
    pipeline = get_pipeline(http_request)
    semaphore = get_semaphore(http_request)
    
    # If callback URL provided, process in background
    if request.callback_url:
        async def background_process():
            result = await process_detection_job(request, pipeline, semaphore)
            await send_callback(request.callback_url, result)
        
        background_tasks.add_task(background_process)
        
        return DetectionResponse(
            job_id=request.job_id,
            status="queued",
            source_dimensions=SourceDimensions(width=0, height=0),
            frame_interval_ms=int(request.frame_interval_seconds * 1000),
            frames=[],
            tracks=[],
            summary=ProcessingSummary(
                total_frames=0,
                faces_detected=0,
                poses_detected=0,
                unique_face_tracks=0,
                unique_pose_tracks=0,
                processing_time_ms=0,
            ),
        )
    
    # Otherwise, process synchronously (with concurrency control)
    return await process_detection_job(request, pipeline, semaphore)


@router.post("/local", response_model=DetectionResponse)
async def detect_local(
    job_id: str,
    video_path: str,
    frame_interval_seconds: float = 2.0,
    detect_faces: bool = True,
    detect_poses: bool = True,
    http_request: Request = None,
) -> DetectionResponse:
    """
    Run detection on a local video file (for testing).
    
    Args:
        job_id: Unique job identifier
        video_path: Local path to the video file
        frame_interval_seconds: Interval between frame extractions
        detect_faces: Whether to run face detection
        detect_poses: Whether to run pose estimation
        
    Returns:
        DetectionResponse with detection results
    """
    import os
    
    if not os.path.exists(video_path):
        raise HTTPException(
            status_code=404,
            detail=f"Video file not found: {video_path}",
        )
    
    face_detector = getattr(http_request.app.state, "face_detector", None)
    pose_estimator = getattr(http_request.app.state, "pose_estimator", None)
    
    if face_detector is None or pose_estimator is None:
        raise HTTPException(
            status_code=503,
            detail="Detection models not loaded.",
        )
    
    # Create pipeline without S3
    from app.services.frame_extractor import FrameExtractor
    from app.services.tracker import ObjectTracker
    
    frame_extractor = FrameExtractor()
    tracker = ObjectTracker()
    
    config = PipelineConfig(
        frame_interval_seconds=frame_interval_seconds,
        detect_faces=detect_faces,
        detect_poses=detect_poses,
    )
    
    # Process locally (simplified pipeline)
    import time
    import cv2
    from app.schemas.responses import (
        BoundingBox,
        FaceDetection,
        FrameDetection,
        PoseDetection,
        TrackSummary,
    )
    
    start_time = time.time()
    
    # Extract frames
    frames, metadata = frame_extractor.extract_frames(
        video_path=video_path,
        interval_seconds=frame_interval_seconds,
    )
    
    frame_detections = []
    total_faces = 0
    total_poses = 0
    
    for frame_info in frames:
        frame = cv2.imread(frame_info.file_path)
        if frame is None:
            continue
        
        frame_result = FrameDetection(
            index=frame_info.index,
            timestamp_ms=frame_info.timestamp_ms,
            faces=[],
            poses=[],
        )
        
        if detect_faces:
            face_result = face_detector.detect_faces(
                frame, frame_info.index, frame_info.timestamp_ms
            )
            for det in face_result.detections:
                face_detections_for_tracking = [(det.bbox, det.confidence)]
                tracked = tracker.update_faces(face_detections_for_tracking, frame, frame_info.index)
                for track_id, bbox, conf in tracked:
                    frame_result.faces.append(FaceDetection(
                        track_id=track_id,
                        bbox=BoundingBox(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3]),
                        confidence=conf,
                    ))
                    total_faces += 1
        
        if detect_poses:
            pose_result = pose_estimator.estimate_pose(
                frame, frame_info.index, frame_info.timestamp_ms
            )
            for det in pose_result.detections:
                if det.bounding_box:
                    pose_detections_for_tracking = [(det.bounding_box, det.confidence)]
                    tracked = tracker.update_poses(pose_detections_for_tracking, frame, frame_info.index)
                    for track_id, bbox, conf in tracked:
                        frame_result.poses.append(PoseDetection(
                            track_id=track_id,
                            keypoints=det.keypoints,
                            confidence=conf,
                            gesture=det.gesture,
                        ))
                        total_poses += 1
        
        frame_detections.append(frame_result)
    
    # Generate track summaries
    track_summaries = []
    for track in tracker.get_all_tracks():
        avg_bbox = track.avg_bbox
        track_summaries.append(TrackSummary(
            track_id=track.track_id,
            track_type=track.track_type,
            first_frame=track.first_frame,
            last_frame=track.last_frame,
            frame_count=track.frame_count,
            avg_bbox=BoundingBox(x=avg_bbox[0], y=avg_bbox[1], width=avg_bbox[2], height=avg_bbox[3]),
            avg_confidence=track.avg_confidence,
        ))
    
    # Cleanup frames
    frame_extractor.cleanup_frames(frames)
    
    processing_time_ms = int((time.time() - start_time) * 1000)
    
    return DetectionResponse(
        job_id=job_id,
        status="completed",
        source_dimensions=SourceDimensions(width=metadata.width, height=metadata.height),
        frame_interval_ms=int(frame_interval_seconds * 1000),
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
    )

