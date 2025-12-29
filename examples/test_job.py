#!/usr/bin/env python3
"""
Epirium Genesis - Comprehensive Test Suite for AI Clipping API.

This script provides complete testing coverage for all API features:
- Clip count scaling (auto and manual)
- Duration ranges (short, medium, long, mixed)
- Time range selection (start/end time filtering)
- Sentence boundary snapping (no word cutoffs)
- Layout detection (auto, split_screen, talking_head)
- Caption presets (viral_gold, clean_white, neon_pop, etc.)
- S3 download integration

Usage Examples:
    # Quick test (3 short clips)
    python test_job.py --quick

    # Standard test (auto clip count, medium duration)
    python test_job.py

    # Full test suite (runs multiple scenarios)
    python test_job.py --full-suite

    # Custom video with specific settings
    python test_job.py --video "https://www.youtube.com/watch?v=VIDEO_ID" --max-clips 5 --duration medium

    # Time range selection (process only 5:00-15:00 of video)
    python test_job.py --start-time 300 --end-time 900

    # Disable auto scaling (get exact clip count)
    python test_job.py --max-clips 7 --no-auto

    # Different caption styles
    python test_job.py --caption-preset neon_pop

    # Analyze previous job logs
    python test_job.py --analyze-only JOB_ID
"""

import argparse
import json
import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from enum import Enum

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

class TestConfig:
    """Test configuration loaded from environment and defaults."""
    
    BASE_URL = os.getenv("GENESIS_API_URL", "http://localhost:8000")
    API_KEY = os.getenv("GENESIS_API_KEY", "your-api-key")
    
    # AWS Configuration for S3 downloads
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET = os.getenv("S3_BUCKET")
    
    # Default test video (13+ minutes with varied content)
    DEFAULT_VIDEO_URL = "https://www.youtube.com/watch?v=BSATK8sL4yw"
    
    # Output directories
    OUTPUT_DIR = Path("test_clips")
    LOGS_DIR = Path("logs")
    
    # Polling configuration
    POLL_INTERVAL = 5  # seconds
    MAX_WAIT_TIME = 3600  # 1 hour max


class DurationRange(Enum):
    """Duration range options."""
    SHORT = "short"      # 15-30 seconds
    MEDIUM = "medium"    # 30-60 seconds
    LONG = "long"        # 60-120 seconds


class CaptionPreset(Enum):
    """Available caption presets."""
    VIRAL_GOLD = "viral_gold"
    CLEAN_WHITE = "clean_white"
    NEON_POP = "neon_pop"
    BOLD_BOXED = "bold_boxed"
    GRADIENT_GLOW = "gradient_glow"


@dataclass
class TestScenario:
    """Test scenario configuration."""
    name: str
    description: str
    max_clips: Optional[int]
    auto_clip_count: bool
    duration_ranges: list[str]
    caption_preset: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None


# Predefined test scenarios
TEST_SCENARIOS = {
    "quick": TestScenario(
        name="Quick Test",
        description="Fast test with 3 short clips",
        max_clips=3,
        auto_clip_count=False,
        duration_ranges=["short"],
        caption_preset="viral_gold",
    ),
    "standard": TestScenario(
        name="Standard Test",
        description="Auto clip count with medium duration",
        max_clips=None,
        auto_clip_count=True,
        duration_ranges=["medium"],
        caption_preset="viral_gold",
    ),
    "full_duration": TestScenario(
        name="Full Duration Range",
        description="All duration ranges enabled",
        max_clips=10,
        auto_clip_count=True,
        duration_ranges=["short", "medium", "long"],
        caption_preset="viral_gold",
    ),
    "time_range": TestScenario(
        name="Time Range Selection",
        description="Process only middle portion (5:00-10:00)",
        max_clips=3,
        auto_clip_count=True,
        duration_ranges=["medium"],
        caption_preset="viral_gold",
        start_time=300.0,
        end_time=600.0,
    ),
    "max_clips": TestScenario(
        name="Max Clips Test",
        description="Test with high clip count limit",
        max_clips=15,
        auto_clip_count=True,
        duration_ranges=["short", "medium"],
        caption_preset="neon_pop",
    ),
}


# ============================================================================
# API Client
# ============================================================================

class GenesisAPIClient:
    """Client for interacting with Genesis AI Clipping API."""
    
    def __init__(self, base_url: str = None, api_key: str = None):
        self.base_url = base_url or TestConfig.BASE_URL
        self.api_key = api_key or TestConfig.API_KEY
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "X-Genesis-API-Key": self.api_key,
        })
    
    def health_check(self) -> dict:
        """Check API health status."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_caption_presets(self) -> list:
        """Get available caption presets."""
        response = self.session.get(f"{self.base_url}/ai-clipping/caption-presets")
        response.raise_for_status()
        return response.json()
    
    def get_layout_presets(self) -> list:
        """Get available layout presets."""
        response = self.session.get(f"{self.base_url}/ai-clipping/layout-presets")
        response.raise_for_status()
        return response.json()
    
    def get_video_metadata(self, video_url: str) -> dict:
        """Get video metadata before processing."""
        response = self.session.post(
            f"{self.base_url}/ai-clipping/metadata",
            json={"video_url": video_url}
        )
        response.raise_for_status()
        return response.json()
    
    def submit_job(
        self,
        video_url: str,
        max_clips: Optional[int] = None,
        auto_clip_count: bool = True,
        duration_ranges: list[str] = None,
        caption_preset: str = "viral_gold",
        start_time_seconds: Optional[float] = None,
        end_time_seconds: Optional[float] = None,
        include_captions: bool = True,
        callback_url: Optional[str] = None,
    ) -> dict:
        """Submit a new clipping job."""
        payload = {
            "video_url": video_url,
            "auto_clip_count": auto_clip_count,
            "duration_ranges": duration_ranges or ["medium"],
            "include_captions": include_captions,
            "caption_preset": caption_preset,
            "layout_type": "auto",
        }
        
        if max_clips is not None:
            payload["max_clips"] = max_clips
        
        if start_time_seconds is not None:
            payload["start_time_seconds"] = start_time_seconds
        
        if end_time_seconds is not None:
            payload["end_time_seconds"] = end_time_seconds
        
        if callback_url:
            payload["callback_url"] = callback_url
        
        response = self.session.post(
            f"{self.base_url}/ai-clipping/jobs",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_job_status(self, job_id: str) -> dict:
        """Get job status and progress."""
        response = self.session.get(f"{self.base_url}/ai-clipping/jobs/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def list_jobs(self, status: str = None, limit: int = 20) -> list:
        """List recent jobs."""
        params = {"limit": limit}
        if status:
            params["status"] = status
        response = self.session.get(
            f"{self.base_url}/ai-clipping/jobs",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def cancel_job(self, job_id: str) -> dict:
        """Cancel a running job."""
        response = self.session.delete(f"{self.base_url}/ai-clipping/jobs/{job_id}")
        response.raise_for_status()
        return response.json()


# ============================================================================
# Test Runner
# ============================================================================

class TestRunner:
    """Orchestrates test execution and result collection."""
    
    def __init__(self, client: GenesisAPIClient):
        self.client = client
        self.start_time = None
        self.results = []
    
    def print_header(self, title: str, char: str = "="):
        """Print formatted header."""
        line = char * 80
        print(f"\n{line}")
        print(f" {title}")
        print(f"{line}")
    
    def print_status(self, message: str, status: str = "INFO"):
        """Print status message with timestamp."""
        icons = {
            "INFO": "ℹ️ ",
            "SUCCESS": "✅",
            "ERROR": "❌",
            "WARNING": "⚠️ ",
            "PROGRESS": "⏳",
            "DOWNLOAD": "📥",
        }
        icon = icons.get(status, "  ")
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {icon} {message}")
    
    def submit_job(
        self,
        video_url: str,
        scenario: TestScenario = None,
        **kwargs
    ) -> Optional[str]:
        """Submit a job and return job_id."""
        self.print_header(f"SUBMITTING JOB: {scenario.name if scenario else 'Custom'}")
        
        # Use scenario settings or kwargs
        max_clips = kwargs.get("max_clips", scenario.max_clips if scenario else None)
        auto_clip_count = kwargs.get("auto_clip_count", scenario.auto_clip_count if scenario else True)
        duration_ranges = kwargs.get("duration_ranges", scenario.duration_ranges if scenario else ["medium"])
        caption_preset = kwargs.get("caption_preset", scenario.caption_preset if scenario else "viral_gold")
        start_time = kwargs.get("start_time", scenario.start_time if scenario else None)
        end_time = kwargs.get("end_time", scenario.end_time if scenario else None)
        
        # Print configuration
        self.print_status(f"Video: {video_url}")
        self.print_status(f"Auto clip count: {auto_clip_count}")
        if max_clips:
            self.print_status(f"Max clips: {max_clips}")
        self.print_status(f"Duration ranges: {duration_ranges}")
        self.print_status(f"Caption preset: {caption_preset}")
        if start_time is not None or end_time is not None:
            start_str = f"{start_time:.0f}s" if start_time else "0s"
            end_str = f"{end_time:.0f}s" if end_time else "end"
            self.print_status(f"Time range: {start_str} - {end_str}")
        
        try:
            result = self.client.submit_job(
                video_url=video_url,
                max_clips=max_clips,
                auto_clip_count=auto_clip_count,
                duration_ranges=duration_ranges,
                caption_preset=caption_preset,
                start_time_seconds=start_time,
                end_time_seconds=end_time,
            )
            
            job_id = result.get("job_id")
            self.print_status(f"Job submitted: {job_id}", "SUCCESS")
            self.print_status(f"Estimated time: {result.get('estimated_processing_minutes', 'unknown')} minutes")
            return job_id
            
        except requests.HTTPError as e:
            self.print_status(f"Failed to submit job: {e}", "ERROR")
            self.print_status(f"Response: {e.response.text if e.response else 'N/A'}", "ERROR")
            return None
        except Exception as e:
            self.print_status(f"Error: {e}", "ERROR")
            return None
    
    def poll_job(self, job_id: str, poll_interval: int = 5) -> Optional[dict]:
        """Poll job status until completion."""
        self.print_header(f"MONITORING JOB: {job_id}", "-")
        
        start_time = time.time()
        last_step = ""
        
        while True:
            try:
                status = self.client.get_job_status(job_id)
                progress = status.get("progress_percent", 0)
                current_step = status.get("current_step", "")
                job_status = status.get("status", "")
                
                # Show progress updates
                if current_step != last_step:
                    elapsed = time.time() - start_time
                    self.print_status(
                        f"[{progress:5.1f}%] [{elapsed:6.1f}s] {job_status}: {current_step}",
                        "PROGRESS"
                    )
                    last_step = current_step
                
                if job_status == "completed":
                    elapsed = time.time() - start_time
                    self.print_status(f"Job completed in {elapsed:.1f}s!", "SUCCESS")
                    return status.get("output")
                
                elif job_status == "failed":
                    self.print_status(f"Job failed: {status.get('error')}", "ERROR")
                    return None
                
                # Check timeout
                if time.time() - start_time > TestConfig.MAX_WAIT_TIME:
                    self.print_status("Job timed out", "ERROR")
                    return None
                
                time.sleep(poll_interval)
                
            except Exception as e:
                self.print_status(f"Error polling status: {e}", "ERROR")
                time.sleep(poll_interval)
    
    def download_clips_from_s3(self, output_data: dict, version_suffix: str = "") -> list:
        """Download clips from S3 to local directory."""
        if not TestConfig.AWS_ACCESS_KEY_ID or not TestConfig.AWS_SECRET_ACCESS_KEY:
            self.print_status("AWS credentials not found - skipping S3 download", "WARNING")
            return []
        
        TestConfig.OUTPUT_DIR.mkdir(exist_ok=True)
        
        clips = output_data.get("clips", [])
        if not clips:
            self.print_status("No clips found in output", "WARNING")
            return []
        
        self.print_header(f"DOWNLOADING {len(clips)} CLIPS FROM S3", "-")
        
        # Set AWS environment
        env = os.environ.copy()
        env["AWS_ACCESS_KEY_ID"] = TestConfig.AWS_ACCESS_KEY_ID
        env["AWS_SECRET_ACCESS_KEY"] = TestConfig.AWS_SECRET_ACCESS_KEY
        env["AWS_DEFAULT_REGION"] = TestConfig.AWS_REGION
        
        downloaded = []
        for clip in clips:
            s3_url = clip.get("s3_url", "")
            if not s3_url or not s3_url.startswith("http"):
                continue
            
            clip_index = clip.get("clip_index", 0)
            new_name = f"clip_{clip_index:02d}{version_suffix}.mp4"
            dest_path = TestConfig.OUTPUT_DIR / new_name
            
            # Convert HTTPS URL to s3:// URL
            s3_path = s3_url.replace(
                f"https://{TestConfig.S3_BUCKET}.s3.{TestConfig.AWS_REGION}.amazonaws.com/",
                f"s3://{TestConfig.S3_BUCKET}/"
            )
            
            try:
                result = subprocess.run(
                    ["aws", "s3", "cp", s3_path, str(dest_path)],
                    env=env,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    file_size = dest_path.stat().st_size / 1024 / 1024
                    self.print_status(f"Downloaded: {new_name} ({file_size:.1f} MB)", "DOWNLOAD")
                    downloaded.append(dest_path)
                else:
                    self.print_status(f"Failed: {new_name} - {result.stderr}", "ERROR")
                    
            except FileNotFoundError:
                self.print_status("AWS CLI not found - install with: pip install awscli", "ERROR")
                return downloaded
            except Exception as e:
                self.print_status(f"Error downloading {new_name}: {e}", "ERROR")
        
        self.print_status(f"Downloaded {len(downloaded)} clips to {TestConfig.OUTPUT_DIR.absolute()}", "SUCCESS")
        return downloaded
    
    def analyze_job_log(self, job_id: str):
        """Analyze job log for key metrics and issues."""
        log_files = list(TestConfig.LOGS_DIR.glob(f"job_{job_id}*.log"))
        
        if not log_files:
            self.print_status(f"No log file found for job {job_id}", "WARNING")
            return
        
        log_file = log_files[0]
        self.print_header(f"LOG ANALYSIS: {log_file.name}")
        
        # Metrics to track
        metrics = {
            "vlm_frames": [],
            "transitions": [],
            "segments": [],
            "clip_scaling": [],
            "sentence_boundary": [],
            "memory": [],
            "time_range": [],
            "duration_enforcement": [],
            "pose_status": None,
            "errors": [],
        }
        
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Categorize log lines
                if "VLM high-precision sampling:" in line:
                    metrics["vlm_frames"].append(line.strip())
                elif "Layout transition detected at frame" in line:
                    metrics["transitions"].append(line.strip())
                elif "Segment" in line and "rendered:" in line:
                    metrics["segments"].append(line.strip())
                elif "Clip count scaling:" in line:
                    metrics["clip_scaling"].append(line.strip())
                elif "sentence boundary" in line.lower():
                    metrics["sentence_boundary"].append(line.strip())
                elif "Memory [" in line or "GC [" in line:
                    metrics["memory"].append(line.strip())
                elif "Time range filter:" in line:
                    metrics["time_range"].append(line.strip())
                elif "Duration" in line and ("Extended" in line or "Trimmed" in line):
                    metrics["duration_enforcement"].append(line.strip())
                elif "Pose estimation DISABLED" in line:
                    metrics["pose_status"] = "DISABLED (optimized)"
                elif "ERROR" in line or "Exception" in line:
                    metrics["errors"].append(line.strip())
        
        # Print analysis
        self._print_metric("🏃 POSE ESTIMATION", [metrics["pose_status"]] if metrics["pose_status"] else [])
        self._print_metric("📐 CLIP COUNT SCALING", metrics["clip_scaling"])
        self._print_metric("⏱️  TIME RANGE SELECTION", metrics["time_range"])
        self._print_metric("📏 DURATION ENFORCEMENT", metrics["duration_enforcement"][:5])
        self._print_metric("✂️  SENTENCE BOUNDARY SNAPPING", metrics["sentence_boundary"][:5])
        self._print_metric("💾 MEMORY CHECKPOINTS", metrics["memory"][-6:])
        self._print_metric("🔍 VLM SAMPLING", metrics["vlm_frames"][:5])
        self._print_metric("🔄 LAYOUT TRANSITIONS", metrics["transitions"])
        self._print_metric("📹 SEGMENTS RENDERED", metrics["segments"])
        
        if metrics["errors"]:
            self._print_metric("❌ ERRORS", metrics["errors"][:10])
    
    def _print_metric(self, title: str, items: list):
        """Print metric section."""
        if items:
            print(f"\n{title}:")
            for item in items:
                # Extract message from log line
                if "INFO - " in item:
                    item = item.split("INFO - ")[-1]
                elif "ERROR - " in item:
                    item = item.split("ERROR - ")[-1]
                print(f"   {item}")
    
    def print_summary(self, output_data: dict):
        """Print detailed job summary."""
        self.print_header("JOB SUMMARY")
        
        print(f"   Source: {output_data.get('source_video_title', 'Unknown')}")
        print(f"   Duration: {output_data.get('source_video_duration_seconds', 0):.1f}s")
        print(f"   Clips Generated: {output_data.get('total_clips', 0)}")
        print(f"   Processing Time: {output_data.get('processing_time_seconds', 0):.1f}s")
        
        clips = output_data.get("clips", [])
        for clip in clips:
            clip_idx = clip.get('clip_index', 0)
            layout = clip.get('layout_type', 'unknown')
            duration_sec = clip.get('duration_ms', 0) / 1000
            start_sec = clip.get('start_time_ms', 0) / 1000
            end_sec = clip.get('end_time_ms', 0) / 1000
            virality = clip.get('virality_score', 0)
            summary = clip.get('summary', '')[:60]
            
            print(f"\n   Clip {clip_idx}:")
            print(f"      Time: {start_sec:.1f}s - {end_sec:.1f}s ({duration_sec:.1f}s)")
            print(f"      Layout: {layout}")
            print(f"      Virality: {virality:.2f}")
            print(f"      Summary: {summary}...")


# ============================================================================
# CLI Entry Point
# ============================================================================

def run_health_check(client: GenesisAPIClient) -> bool:
    """Run health check and print results."""
    print("\n🏥 Running health check...")
    try:
        health = client.health_check()
        print(f"   Status: {health.get('status', 'unknown')}")
        print(f"   Version: {health.get('version', 'unknown')}")
        return health.get('status') == 'healthy'
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return False


def run_full_suite(client: GenesisAPIClient, video_url: str):
    """Run full test suite with multiple scenarios."""
    runner = TestRunner(client)
    runner.print_header("FULL TEST SUITE")
    
    scenarios_to_run = ["quick", "time_range", "full_duration"]
    results = []
    
    for scenario_name in scenarios_to_run:
        scenario = TEST_SCENARIOS[scenario_name]
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario.name}")
        print(f"Description: {scenario.description}")
        print(f"{'='*80}")
        
        job_id = runner.submit_job(video_url, scenario)
        if not job_id:
            results.append((scenario_name, "FAILED", "Submit failed"))
            continue
        
        output = runner.poll_job(job_id)
        runner.analyze_job_log(job_id)
        
        if output:
            results.append((scenario_name, "PASSED", f"{output.get('total_clips', 0)} clips"))
            runner.print_summary(output)
        else:
            results.append((scenario_name, "FAILED", "Processing failed"))
    
    # Print suite summary
    runner.print_header("TEST SUITE RESULTS")
    for name, status, details in results:
        icon = "✅" if status == "PASSED" else "❌"
        print(f"   {icon} {name}: {status} - {details}")


def main():
    parser = argparse.ArgumentParser(
        description="Epirium Genesis - Comprehensive AI Clipping Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Scenarios:
  --quick           Run quick test (2 short clips)
  --standard        Run standard test (auto clips, medium)
  --full-suite      Run all test scenarios

Examples:
  python test_job.py                                     # Standard test
  python test_job.py --quick                             # Quick 2-clip test
  python test_job.py --video "URL" --max-clips 5         # Custom video
  python test_job.py --start-time 300 --end-time 600     # Time range (5:00-10:00)
  python test_job.py --max-clips 7 --no-auto             # Exact 7 clips
  python test_job.py --duration short medium             # Multiple durations
  python test_job.py --caption-preset neon_pop           # Different style
  python test_job.py --full-suite                        # Run all scenarios
  python test_job.py --analyze-only JOB_ID               # Analyze existing job log
  python test_job.py --health-only                       # Just check API health
  python test_job.py --list-presets                      # List caption/layout presets

Configuration (via environment or .env):
  GENESIS_API_URL      API base URL (default: http://localhost:8000)
  GENESIS_API_KEY      API key for authentication
  AWS_ACCESS_KEY_ID    AWS credentials for S3 download
  AWS_SECRET_ACCESS_KEY
  AWS_REGION           AWS region (default: us-east-1)
  S3_BUCKET            S3 bucket name
        """
    )
    
    # Video source
    parser.add_argument("--video", type=str, default=TestConfig.DEFAULT_VIDEO_URL,
                        help="Video URL to process (YouTube, S3, or direct)")
    
    # Clip count options
    parser.add_argument("--max-clips", type=int, default=None,
                        help="Maximum clips to generate (optional with auto mode)")
    parser.add_argument("--auto-clip-count", type=bool, default=True,
                        help="Auto-scale clip count based on video duration")
    parser.add_argument("--no-auto", action="store_true",
                        help="Disable auto clip count (use exact max-clips)")
    
    # Duration options
    parser.add_argument("--duration", type=str, nargs="+", default=["medium"],
                        choices=["short", "medium", "long"],
                        help="Duration ranges to generate")
    
    # Time range options
    parser.add_argument("--start-time", type=float, default=None,
                        help="Start time in seconds (e.g., 300 for 5:00)")
    parser.add_argument("--end-time", type=float, default=None,
                        help="End time in seconds (e.g., 600 for 10:00)")
    
    # Caption options
    parser.add_argument("--caption-preset", type=str, default="viral_gold",
                        choices=["viral_gold", "clean_white", "neon_pop", "bold_boxed", "gradient_glow"],
                        help="Caption preset to use")
    parser.add_argument("--no-captions", action="store_true",
                        help="Disable captions")
    
    # Output options
    parser.add_argument("--version", type=str, default="",
                        help="Version suffix for downloaded files (e.g., _v1)")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip downloading clips from S3")
    
    # Test scenario shortcuts
    parser.add_argument("--quick", action="store_true",
                        help="Run quick test (2 short clips)")
    parser.add_argument("--full-suite", action="store_true",
                        help="Run full test suite with multiple scenarios")
    
    # Utility options
    parser.add_argument("--analyze-only", type=str, metavar="JOB_ID",
                        help="Only analyze log for given job ID")
    parser.add_argument("--health-only", action="store_true",
                        help="Only run health check")
    parser.add_argument("--list-presets", action="store_true",
                        help="List available caption and layout presets")
    parser.add_argument("--list-jobs", action="store_true",
                        help="List recent jobs")
    
    args = parser.parse_args()
    
    # Initialize client
    client = GenesisAPIClient()
    runner = TestRunner(client)
    
    # Handle utility commands
    if args.health_only:
        success = run_health_check(client)
        sys.exit(0 if success else 1)
    
    if args.list_presets:
        runner.print_header("AVAILABLE PRESETS")
        try:
            print("\n📝 CAPTION PRESETS:")
            for preset in client.get_caption_presets():
                print(f"   • {preset['id']}: {preset['name']} - {preset['description']}")
            
            print("\n📐 LAYOUT PRESETS:")
            for preset in client.get_layout_presets():
                print(f"   • {preset['id']}: {preset['name']} - {preset['description']}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        return
    
    if args.list_jobs:
        runner.print_header("RECENT JOBS")
        try:
            jobs = client.list_jobs(limit=10)
            for job in jobs:
                status_icon = {"completed": "✅", "failed": "❌", "processing": "⏳"}.get(job['status'], "❓")
                print(f"   {status_icon} {job['job_id'][:8]}... | {job['status']} | {job.get('created_at', 'N/A')}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        return
    
    if args.analyze_only:
        runner.analyze_job_log(args.analyze_only)
        return
    
    # Run health check first
    if not run_health_check(client):
        print("\n⚠️  API may not be running. Continuing anyway...")
    
    # Handle test scenarios
    if args.full_suite:
        run_full_suite(client, args.video)
        return
    
    # Build scenario from args
    if args.quick:
        scenario = TEST_SCENARIOS["quick"]
    else:
        scenario = TestScenario(
            name="Custom Test",
            description="User-configured test",
            max_clips=args.max_clips,
            auto_clip_count=not args.no_auto,
            duration_ranges=args.duration,
            caption_preset=args.caption_preset,
            start_time=args.start_time,
            end_time=args.end_time,
        )
    
    # Submit and process job
    job_id = runner.submit_job(args.video, scenario)
    if not job_id:
        sys.exit(1)
    
    # Wait for completion
    output = runner.poll_job(job_id)
    
    # Analyze log
    runner.analyze_job_log(job_id)
    
    if not output:
        runner.print_status("Job failed - check logs for details", "ERROR")
        sys.exit(1)
    
    # Download clips
    if not args.no_download:
        version = args.version or f"_v{int(time.time()) % 1000}"
        runner.download_clips_from_s3(output, version)
    
    # Print summary
    runner.print_summary(output)
    
    runner.print_header("TEST COMPLETE", "=")
    print(f"\n💡 Tips:")
    print(f"   • Use --analyze-only {job_id} to re-analyze this job's logs")
    print(f"   • Check logs/ directory for detailed processing logs")
    print(f"   • Downloaded clips are in test_clips/ directory")


if __name__ == "__main__":
    main()
