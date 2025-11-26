"""
Configuration module using Pydantic Settings for environment variable management.
"""

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class CaptionStyle(BaseSettings):
    """Caption styling configuration."""
    
    font_name: str = "Arial Black"
    font_size: int = 72
    primary_color: str = "#FFFFFF"
    highlight_color: str = "#FFD700"  # Gold
    outline_color: str = "#000000"
    outline_width: int = 4
    shadow_color: str = "#000000"
    position: Literal["top", "center", "bottom"] = "center"
    max_words_per_line: int = 4
    word_by_word_highlight: bool = True
    alignment: Literal["left", "center", "right"] = "center"
    bold: bool = True
    uppercase: bool = True


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "viewcreator-clipping-worker"
    debug: bool = False
    log_level: str = "INFO"

    # AWS S3
    aws_region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    s3_bucket: str = "viewcreator-media"

    # Model paths
    yolo_model_path: str = "/app/models/yolov8n-face.pt"
    yolo_model_name: str = "yolov8n-face.pt"

    # Processing settings
    frame_interval_seconds: float = 2.0
    max_concurrent_jobs: int = 2
    temp_directory: str = "/tmp/clipping-worker"
    workspace_root: str = "/tmp/ai-clipping-agent"

    # Detection settings
    face_confidence_threshold: float = 0.5
    pose_confidence_threshold: float = 0.5
    tracking_max_age: int = 30  # Frames before a track is deleted

    # API settings
    api_timeout_seconds: int = 300
    max_video_duration_seconds: int = 7200  # 2 hours max

    # ============================================================
    # AI CLIPPING PIPELINE SETTINGS
    # ============================================================
    
    # yt-dlp Configuration
    ytdlp_path: str = Field(default="yt-dlp", description="Path to yt-dlp binary")
    ytdlp_cookies_from_browser: Optional[str] = None
    ytdlp_extra_args: str = ""  # Space-separated extra arguments
    max_download_duration_seconds: int = 7200

    # Transcription Configuration (Groq Whisper)
    groq_api_key: Optional[str] = None
    transcription_model: str = "whisper-large-v3-turbo"
    
    # Chunk duration for long audio transcription (Groq supports 30 min chunks)
    groq_chunk_duration_seconds: int = 1800

    # OpenRouter / LLM Configuration
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    gemini_model: str = "google/gemini-2.5-flash"
    qwen_model: str = "qwen/qwen-2.5-vl-72b-instruct"
    
    # Clip Planning Configuration
    max_suggested_clips: int = 5
    max_frames_per_vision_batch: int = 48
    
    # Rendering Configuration
    target_output_width: int = 1080
    target_output_height: int = 1920
    ffmpeg_preset: str = "veryfast"
    ffmpeg_crf: int = 20
    
    # Stack Layout Configuration (screen on top, face on bottom)
    stack_screen_height_ratio: float = 0.55  # Screen takes top 55%
    stack_face_height_ratio: float = 0.45    # Face takes bottom 45%
    
    # OpusClip-Style Layout (screen top, captions middle, face bottom)
    opusclip_screen_ratio: float = 0.45      # Screen content: 45%
    opusclip_caption_ratio: float = 0.12     # Caption band: 12%
    opusclip_face_ratio: float = 0.43        # Speaker face: 43%
    use_opusclip_layout: bool = True         # Enable OpusClip-style layout for screen_share
    
    # Camera Physics (mass-spring-damper smoothing)
    camera_mass: float = 1.0
    camera_stiffness: float = 150.0
    camera_damping: float = 20.0
    
    # Caption styling (can be overridden per-job)
    caption_font_name: str = "Arial Black"
    caption_font_size: int = 72
    caption_primary_color: str = "#FFFFFF"
    caption_highlight_color: str = "#FFD700"
    caption_position: Literal["top", "center", "bottom"] = "center"
    caption_max_words_per_line: int = 4
    caption_word_by_word_highlight: bool = True
    caption_uppercase: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def get_caption_style(self) -> CaptionStyle:
        """Build CaptionStyle from settings."""
        return CaptionStyle(
            font_name=self.caption_font_name,
            font_size=self.caption_font_size,
            primary_color=self.caption_primary_color,
            highlight_color=self.caption_highlight_color,
            position=self.caption_position,
            max_words_per_line=self.caption_max_words_per_line,
            word_by_word_highlight=self.caption_word_by_word_highlight,
            uppercase=self.caption_uppercase,
        )
    
    def get_ytdlp_extra_args(self) -> list[str]:
        """Parse yt-dlp extra arguments."""
        if not self.ytdlp_extra_args:
            return []
        return [arg.strip() for arg in self.ytdlp_extra_args.split(" ") if arg.strip()]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()



