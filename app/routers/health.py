"""
Health check endpoints for the clipping worker.
"""

from fastapi import APIRouter, Request

from app.schemas.responses import HealthResponse, ReadinessResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns 200 if the service is running.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
    )


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_check(request: Request):
    """
    Readiness check endpoint.
    
    Returns whether the service is ready to accept detection requests.
    Checks that ML models are loaded and ready.
    """
    face_detector = getattr(request.app.state, "face_detector", None)
    pose_estimator = getattr(request.app.state, "pose_estimator", None)

    face_ready = face_detector is not None and face_detector.is_ready()
    pose_ready = pose_estimator is not None and pose_estimator.is_ready()

    return ReadinessResponse(
        ready=face_ready and pose_ready,
        models_loaded=face_ready and pose_ready,
        face_detector="ready" if face_ready else "not_loaded",
        pose_estimator="ready" if pose_ready else "not_loaded",
    )


@router.get("/health/models")
async def model_status(request: Request):
    """
    Detailed model status endpoint.
    
    Returns information about loaded ML models.
    """
    face_detector = getattr(request.app.state, "face_detector", None)
    pose_estimator = getattr(request.app.state, "pose_estimator", None)

    return {
        "models": {
            "face_detector": {
                "loaded": face_detector is not None,
                "ready": face_detector.is_ready() if face_detector else False,
                "model_type": "YOLOv8" if face_detector else None,
            },
            "pose_estimator": {
                "loaded": pose_estimator is not None,
                "ready": pose_estimator.is_ready() if pose_estimator else False,
                "model_type": "MediaPipe Pose" if pose_estimator else None,
            },
        }
    }



