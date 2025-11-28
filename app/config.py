"""
Configuration module using Pydantic Settings for environment variable management.

Only essential environment variables are exposed. All other settings are hardcoded
for consistency and simplicity.
"""

from functools import lru_cache
from typing import Literal, Optional

from pydantic_settings import BaseSettings


class CaptionStyle:
    """Caption styling configuration (hardcoded)."""

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
    """
    Application settings.

    Only essential configuration is loaded from environment variables.
    All processing/rendering settings are hardcoded for consistency.
    """

    # ============================================================
    # ENVIRONMENT VARIABLES (minimal set)
    # ============================================================

    # Application
    app_name: str = "viewcreator-genesis"
    debug: bool = False
    log_level: str = "INFO"

    # AWS S3
    aws_region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    s3_bucket: str = "viewcreator-media"

    # API Keys (required)
    groq_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None

    # Security - API authentication
    genesis_api_key: Optional[str] = None  # API key for authenticating incoming requests
    genesis_webhook_secret: Optional[str] = None  # Secret for signing outgoing webhooks

    # ============================================================
    # HARDCODED SETTINGS (not configurable via env vars)
    # ============================================================

    # Model paths
    @property
    def yolo_model_path(self) -> str:
        return "/app/models/yolov8n.pt"

    @property
    def yolo_model_name(self) -> str:
        return "yolov8n.pt"

    # Processing settings
    @property
    def frame_interval_seconds(self) -> float:
        return 2.0

    @property
    def max_concurrent_jobs(self) -> int:
        return 2

    @property
    def temp_directory(self) -> str:
        return "/tmp/genesis"

    @property
    def workspace_root(self) -> str:
        return "/tmp/ai-clipping-agent"

    # Detection settings
    @property
    def face_confidence_threshold(self) -> float:
        return 0.5

    @property
    def pose_confidence_threshold(self) -> float:
        return 0.5

    @property
    def tracking_max_age(self) -> int:
        return 30  # Frames before a track is deleted

    # API settings
    @property
    def api_timeout_seconds(self) -> int:
        return 300

    @property
    def max_video_duration_seconds(self) -> int:
        return 14400  # 4 hours max (credit-guarded in API)

    # yt-dlp Configuration
    @property
    def ytdlp_path(self) -> str:
        return "yt-dlp"

    @property
    def ytdlp_cookies_from_browser(self) -> Optional[str]:
        return None

    @property
    def max_download_duration_seconds(self) -> int:
        return 14400  # 4 hours max (credit-guarded in API)

    # Transcription Configuration (Groq Whisper)
    @property
    def transcription_model(self) -> str:
        return "whisper-large-v3-turbo"

    @property
    def groq_chunk_duration_seconds(self) -> int:
        return 1800  # 30 minutes

    # OpenRouter / LLM Configuration
    @property
    def openrouter_base_url(self) -> str:
        return "https://openrouter.ai/api/v1"

    @property
    def gemini_model(self) -> str:
        return "google/gemini-2.5-flash"

    # Clip Planning Configuration
    @property
    def max_suggested_clips(self) -> int:
        return 5

    @property
    def max_frames_per_vision_batch(self) -> int:
        return 48

    # Rendering Configuration
    @property
    def target_output_width(self) -> int:
        return 1080

    @property
    def target_output_height(self) -> int:
        return 1920

    @property
    def ffmpeg_preset(self) -> str:
        return "veryfast"

    @property
    def ffmpeg_crf(self) -> int:
        return 20

    # OpusClip-Style Layout (screen top, face bottom, captions overlaid)
    @property
    def opusclip_screen_ratio(self) -> float:
        return 0.50  # Screen content: 50% (top)

    @property
    def opusclip_face_ratio(self) -> float:
        return 0.50  # Speaker face: 50% (bottom)

    @property
    def use_opusclip_layout(self) -> bool:
        return True  # Enable OpusClip-style layout for screen_share

    # Legacy Stack Layout (kept for fallback)
    @property
    def stack_screen_height_ratio(self) -> float:
        return 0.55

    @property
    def stack_face_height_ratio(self) -> float:
        return 0.45

    # Camera Physics (mass-spring-damper smoothing)
    @property
    def camera_mass(self) -> float:
        return 1.0

    @property
    def camera_stiffness(self) -> float:
        return 150.0

    @property
    def camera_damping(self) -> float:
        return 20.0

    # Caption styling
    @property
    def caption_font_name(self) -> str:
        return "Arial Black"

    @property
    def caption_font_size(self) -> int:
        return 72

    @property
    def caption_primary_color(self) -> str:
        return "#FFFFFF"

    @property
    def caption_highlight_color(self) -> str:
        return "#FFD700"

    @property
    def caption_position(self) -> Literal["top", "center", "bottom"]:
        return "center"

    @property
    def caption_max_words_per_line(self) -> int:
        return 4

    @property
    def caption_word_by_word_highlight(self) -> bool:
        return True

    @property
    def caption_uppercase(self) -> bool:
        return True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def get_caption_style(self) -> CaptionStyle:
        """Build CaptionStyle from settings."""
        style = CaptionStyle()
        style.font_name = self.caption_font_name
        style.font_size = self.caption_font_size
        style.primary_color = self.caption_primary_color
        style.highlight_color = self.caption_highlight_color
        style.position = self.caption_position
        style.max_words_per_line = self.caption_max_words_per_line
        style.word_by_word_highlight = self.caption_word_by_word_highlight
        style.uppercase = self.caption_uppercase
        return style

    def get_ytdlp_extra_args(self) -> list[str]:
        """Parse yt-dlp extra arguments (none by default)."""
        return []


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
