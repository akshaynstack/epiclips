# ViewCreator Clipping Worker

A FastAPI-based video detection service that uses YOLO for face detection, MediaPipe for pose estimation, and DeepSORT for object tracking across frames.

## Features

- **YOLO Face Detection**: Accurate face detection using YOLOv8
- **MediaPipe Pose Estimation**: Full body pose with 33 keypoints
- **DeepSORT Tracking**: Consistent object IDs across frames
- **S3 Integration**: Download videos and upload results to AWS S3
- **Docker Ready**: Production-ready containerization
- **ECS Compatible**: Health checks and graceful shutdown

## Quick Start

### Local Development

1. **Install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

2. **Run the server**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Test the API**:
   ```bash
   curl http://localhost:8000/health
   ```

### Docker

1. **Build and run**:
   ```bash
   docker-compose up --build
   ```

2. **With LocalStack (S3 emulation)**:
   ```bash
   docker-compose --profile localstack up --build
   ```

## API Endpoints

### Health Checks

- `GET /health` - Basic health check
- `GET /health/ready` - Readiness check (models loaded)
- `GET /health/models` - Detailed model status

### Detection

- `POST /detect` - Run detection on a video from S3

**Request:**
```json
{
    "job_id": "uuid",
    "video_s3_key": "users/user123/videos/source.mp4",
    "frame_interval_seconds": 2.0,
    "detect_faces": true,
    "detect_poses": true,
    "callback_url": null
}
```

**Response:**
```json
{
    "job_id": "uuid",
    "status": "completed",
    "source_dimensions": {"width": 1920, "height": 1080},
    "frame_interval_ms": 2000,
    "frames": [
        {
            "index": 0,
            "timestamp_ms": 0,
            "faces": [
                {
                    "track_id": 1,
                    "bbox": {"x": 423, "y": 156, "width": 187, "height": 234},
                    "confidence": 0.94
                }
            ],
            "poses": [...]
        }
    ],
    "tracks": [...],
    "summary": {
        "total_frames": 150,
        "faces_detected": 142,
        "poses_detected": 145,
        "unique_face_tracks": 1,
        "unique_pose_tracks": 1,
        "processing_time_ms": 4523
    }
}
```

## Configuration

### Setup (Local Development)

1. **Create a `.env` file** from the template:
   ```bash
   # Copy the example below into a new file called .env
   # NEVER commit .env to git (it's in .gitignore)
   ```

2. **Example `.env` file**:
   ```env
   # AWS Configuration (REQUIRED for S3 access)
   AWS_REGION=us-east-1
   AWS_ACCESS_KEY_ID=your-access-key-here
   AWS_SECRET_ACCESS_KEY=your-secret-key-here
   S3_BUCKET=viewcreator-media
   
   # Optional overrides
   LOG_LEVEL=INFO
   DEBUG=false
   FRAME_INTERVAL_SECONDS=2.0
   FACE_CONFIDENCE_THRESHOLD=0.5
   ```

### Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AWS_ACCESS_KEY_ID` | Yes* | - | AWS credentials for S3 |
| `AWS_SECRET_ACCESS_KEY` | Yes* | - | AWS credentials for S3 |
| `S3_BUCKET` | Yes | `viewcreator-media` | S3 bucket for videos |
| `AWS_REGION` | No | `us-east-1` | AWS region |
| `YOLO_MODEL_PATH` | No | `/app/models/yolov8n.pt` | Path to YOLO weights |
| `FRAME_INTERVAL_SECONDS` | No | `2.0` | Frame extraction interval |
| `FACE_CONFIDENCE_THRESHOLD` | No | `0.5` | Min confidence for faces |
| `POSE_CONFIDENCE_THRESHOLD` | No | `0.5` | Min confidence for poses |
| `TRACKING_MAX_AGE` | No | `30` | Frames before track deletion |
| `MAX_CONCURRENT_JOBS` | No | `2` | Parallel job limit |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `DEBUG` | No | `false` | Enable debug mode |

*\*In ECS, use IAM Task Roles instead of explicit credentials*

### How Configuration Works

This project uses **Pydantic Settings** (`app/config.py`), the Python best practice:

```python
# Pydantic automatically loads from environment variables
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    aws_region: str = "us-east-1"        # Loads from AWS_REGION
    s3_bucket: str = "viewcreator-media" # Loads from S3_BUCKET
    
    class Config:
        env_file = ".env"  # Also loads from .env file if present
```

**Priority order** (highest to lowest):
1. Environment variables set in shell/Docker
2. `.env` file
3. Default values in code

## ECS Deployment

### Task Definition

```json
{
    "family": "clipping-worker",
    "cpu": "2048",
    "memory": "4096",
    "containerDefinitions": [
        {
            "name": "clipping-worker",
            "image": "your-ecr-repo/clipping-worker:latest",
            "portMappings": [
                {
                    "containerPort": 8000,
                    "protocol": "tcp"
                }
            ],
            "healthCheck": {
                "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
                "interval": 30,
                "timeout": 10,
                "retries": 3,
                "startPeriod": 60
            },
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/clipping-worker",
                    "awslogs-region": "us-east-1",
                    "awslogs-stream-prefix": "ecs"
                }
            }
        }
    ]
}
```

## Integration with NestJS

The NestJS API calls this worker via HTTP:

```typescript
// In NestJS: detection-worker-client.service.ts
async detect(jobId: string, videoS3Key: string): Promise<DetectionResult> {
    const response = await this.httpService.post(
        `${this.workerUrl}/detect`,
        {
            job_id: jobId,
            video_s3_key: videoS3Key,
            frame_interval_seconds: 2.0,
            detect_faces: true,
            detect_poses: true,
        },
        { timeout: 300000 }  // 5 minute timeout
    ).toPromise();
    
    return response.data;
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Detection Pipeline                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   S3 Download ──▶ FFmpeg Extract ──▶ YOLO + MediaPipe          │
│        │              │                    │                    │
│        ▼              ▼                    ▼                    │
│   source.mp4    frames/*.jpg         detections[]               │
│                                            │                    │
│                                            ▼                    │
│                                     DeepSORT Track              │
│                                            │                    │
│                                            ▼                    │
│                                     S3 Upload JSON              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## License

Proprietary - ViewCreator

