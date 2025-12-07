#!/usr/bin/env python3
"""
Test script for ViewCreator Genesis AI Clipping API.

HIGH-PRECISION MODE: Tests 1 FPS VLM sampling with frame-accurate rendering.

Usage:
    python test_job.py                    # Submit job and wait for completion (auto-download from S3)
    python test_job.py --max-clips 3 --duration short  # Short clips for faster testing
    
Features:
- 1 FPS VLM sampling (vs old 5-7 frames total)
- Frame-accurate transition detection
- Precise FFmpeg seeking
- Auto-download from S3 using .env credentials
"""

import argparse
import json
import os
import time
import uuid
import requests
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Configuration
BASE_URL = "http://localhost:8000"
API_KEY = "your-api-key"
TEST_VIDEO_URL = "https://www.youtube.com/watch?v=QpgjzP6cBFA"  # Test video with layout transitions

# AWS Configuration from .env
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET")

# Output directory
OUTPUT_DIR = Path("test_clips")
LOGS_DIR = Path("logs")


def submit_job(max_clips: int = 3, duration_ranges: list = None):
    """Submit a clipping job with high-precision settings."""
    if duration_ranges is None:
        duration_ranges = ["short"]
    
    headers = {
        "Content-Type": "application/json",
        "X-Genesis-API-Key": API_KEY
    }
    
    payload = {
        "video_url": TEST_VIDEO_URL,
        "max_clips": max_clips,
        "duration_ranges": duration_ranges,
        "include_captions": True,
        "caption_preset": "viral_gold",
        "layout_type": "auto"  # High-precision auto-detection with 1 FPS VLM sampling
    }
    
    print(f"\n🚀 HIGH-PRECISION MODE: Submitting job")
    print(f"   Video: {TEST_VIDEO_URL}")
    print(f"   Max clips: {max_clips}, Duration ranges: {duration_ranges}")
    print(f"   VLM Sampling: 1 FPS (frame-accurate transition detection)")
    print(f"   FFmpeg: Frame-accurate seeking enabled")
    
    response = requests.post(
        f"{BASE_URL}/ai-clipping/jobs",
        headers=headers,
        json=payload
    )
    
    if response.status_code != 202:
        print(f"❌ Failed to submit job: {response.status_code}")
        print(response.text)
        return None
    
    job_data = response.json()
    job_id = job_data["job_id"]
    print(f"✅ Job submitted: {job_id}")
    
    return job_id


def poll_job_status(job_id: str, poll_interval: int = 5):
    """Poll job status until completion with detailed progress tracking."""
    print(f"\n⏳ Waiting for job {job_id} to complete...")
    print(f"   Monitoring: VLM sampling rate, transition detection, segment rendering")
    
    start_time = time.time()
    last_step = ""
    
    while True:
        response = requests.get(f"{BASE_URL}/ai-clipping/jobs/{job_id}")
        
        if response.status_code != 200:
            print(f"❌ Failed to get job status: {response.status_code}")
            return None
        
        status = response.json()
        progress = status.get("progress_percent", 0)
        current_step = status.get("current_step", "")
        job_status = status.get("status", "")
        
        # Show detailed progress
        if current_step != last_step:
            elapsed = time.time() - start_time
            print(f"   [{progress:5.1f}%] [{elapsed:6.1f}s] {job_status}: {current_step}")
            last_step = current_step
        
        if job_status == "completed":
            elapsed = time.time() - start_time
            print(f"\n✅ Job completed in {elapsed:.1f}s!")
            return status.get("output")
        elif job_status == "failed":
            print(f"\n❌ Job failed: {status.get('error')}")
            return None
        
        time.sleep(poll_interval)


def analyze_job_log(job_id: str):
    """Analyze job log for high-precision metrics."""
    log_files = list(LOGS_DIR.glob(f"job_{job_id}*.log"))
    
    if not log_files:
        print(f"\n⚠️  No log file found for job {job_id}")
        return
    
    log_file = log_files[0]
    print(f"\n📊 HIGH-PRECISION ANALYSIS: {log_file.name}")
    print("=" * 80)
    
    vlm_frames = []
    transitions = []
    segments = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Track VLM sampling
            if "VLM high-precision sampling:" in line:
                vlm_frames.append(line.strip())
            
            # Track transition detection
            if "Layout transition detected at frame" in line:
                transitions.append(line.strip())
            
            # Track segment rendering
            if "Segment" in line and "rendered:" in line and "boundary:" in line:
                segments.append(line.strip())
            
            # Track concatenation
            if "Segments concatenated successfully:" in line:
                print(f"\n✅ CONCATENATION: {line.split('INFO - ')[-1].strip()}")
    
    # Print VLM sampling stats
    if vlm_frames:
        print(f"\n🔍 VLM SAMPLING:")
        for frame_info in vlm_frames[:5]:  # Show first 5
            print(f"   {frame_info.split('INFO - ')[-1]}")
        if len(vlm_frames) > 5:
            print(f"   ... and {len(vlm_frames) - 5} more clips")
    
    # Print transitions
    if transitions:
        print(f"\n🔄 LAYOUT TRANSITIONS DETECTED:")
        for trans in transitions:
            print(f"   {trans.split('INFO - ')[-1]}")
    else:
        print(f"\n✓ No layout transitions detected (single-layout clips)")
    
    # Print segments
    if segments:
        print(f"\n📹 SEGMENT RENDERING:")
        for seg in segments:
            print(f"   {seg.split('INFO - ')[-1]}")
    
    print("=" * 80)


def copy_clips_from_temp(job_id: str, version_suffix: str = ""):
    """Copy rendered clips from temp directory to test_clips."""
    import shutil
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Look for clips in /tmp/genesis/{job_id}/clips/
    if os.name == 'nt':  # Windows
        temp_base = Path(os.environ.get('TEMP', 'C:/tmp')) / 'genesis'
    else:
        temp_base = Path('/tmp/genesis')
    
    job_dir = temp_base / job_id / 'clips'
    
    if not job_dir.exists():
        print(f"\n⚠️  Temp directory not found: {job_dir}")
        print("   Clips may have been cleaned up already")
        return
    
    clip_files = list(job_dir.glob("clip_*.mp4"))
    
    if not clip_files:
        print(f"\n⚠️  No clips found in {job_dir}")
        return
    
    print(f"\n📥 Copying {len(clip_files)} clips from temp directory...")
    
    for clip_file in sorted(clip_files):
        clip_index = clip_file.stem.split('_')[1]
        new_name = f"clip_{clip_index}{version_suffix}.mp4"
        dest_path = OUTPUT_DIR / new_name
        
        shutil.copy2(clip_file, dest_path)
        file_size = dest_path.stat().st_size / 1024 / 1024
        print(f"   ✅ Copied: {new_name} ({file_size:.1f} MB)")
    
    print(f"\n✅ Clips saved to: {OUTPUT_DIR.absolute()}")


def download_clips_from_s3(output_data: dict, version_suffix: str = ""):
    """Download clips from S3 using AWS CLI with credentials from .env."""
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        print("\n⚠️  AWS credentials not found in .env - skipping S3 download")
        return
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    clips = output_data.get("clips", [])
    if not clips:
        print("\n⚠️  No clips found in output")
        return
    
    print(f"\n📥 Downloading {len(clips)} clips from S3...")
    
    # Set AWS environment variables
    env = os.environ.copy()
    env["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    env["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
    env["AWS_DEFAULT_REGION"] = AWS_REGION
    
    for clip in clips:
        s3_url = clip.get("s3_url", "")
        if not s3_url or not s3_url.startswith("http"):
            continue
        
        clip_index = clip.get("clip_index", 0)
        new_name = f"clip_{clip_index:02d}{version_suffix}.mp4"
        dest_path = OUTPUT_DIR / new_name
        
        # Convert HTTPS URL to s3:// URL
        # https://bucket.s3.region.amazonaws.com/path -> s3://bucket/path
        s3_path = s3_url.replace(f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/", f"s3://{S3_BUCKET}/")
        
        print(f"   Downloading clip {clip_index}...")
        
        try:
            result = subprocess.run(
                ["aws", "s3", "cp", s3_path, str(dest_path)],
                env=env,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                file_size = dest_path.stat().st_size / 1024 / 1024
                print(f"   ✅ Downloaded: {new_name} ({file_size:.1f} MB)")
            else:
                print(f"   ❌ Failed to download clip {clip_index}: {result.stderr}")
        except FileNotFoundError:
            print(f"   ❌ AWS CLI not found - install with: pip install awscli")
            return
        except Exception as e:
            print(f"   ❌ Error downloading clip {clip_index}: {e}")
    
    print(f"\n✅ Clips saved to: {OUTPUT_DIR.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Test ViewCreator Genesis AI Clipping API (HIGH-PRECISION MODE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
High-Precision Features:
  - 1 FPS VLM sampling (every second vs old 5-7 frames total)
  - Frame-accurate transition detection (±1 second precision)
  - Frame-accurate FFmpeg seeking (microsecond precision)
  - Smooth segment concatenation (no frame gaps/overlaps)
  - Auto-download from S3 using .env credentials
  
Examples:
  python test_job.py
  python test_job.py --max-clips 5 --duration medium
  python test_job.py --version _test1
        """
    )
    parser.add_argument("--max-clips", type=int, default=1, help="Maximum clips to generate")
    parser.add_argument("--duration", type=str, default="short", help="Duration range: short, medium, long")
    parser.add_argument("--version", type=str, default="", help="Version suffix for downloaded files")
    parser.add_argument("--analyze-only", type=str, help="Only analyze log for given job ID")
    
    args = parser.parse_args()
    
    if args.analyze_only:
        analyze_job_log(args.analyze_only)
        return
    
    # Submit and process job
    duration_ranges = [args.duration]
    job_id = submit_job(args.max_clips, duration_ranges)
    
    if not job_id:
        return
    
    # Wait for completion
    output = poll_job_status(job_id)
    
    # Analyze log regardless of success/failure
    print("\n" + "=" * 80)
    analyze_job_log(job_id)
    
    if not output:
        print("\n⚠️  Job failed - check logs for details")
        return
    
    # Download clips from S3
    version = args.version or f"_v{int(time.time()) % 1000}"
    download_clips_from_s3(output, version)
    
    # Print detailed summary
    print("\n" + "=" * 80)
    print("📊 JOB SUMMARY (HIGH-PRECISION MODE)")
    print("=" * 80)
    print(f"   Source: {output.get('source_video_title', 'Unknown')}")
    print(f"   Duration: {output.get('source_video_duration_seconds', 0):.1f}s")
    print(f"   Clips Generated: {output.get('total_clips', 0)}")
    print(f"   Processing Time: {output.get('processing_time_seconds', 0):.1f}s")
    
    clips = output.get("clips", [])
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
    
    print("\n" + "=" * 80)
    print("✅ HIGH-PRECISION TEST COMPLETE")
    print("=" * 80)
    print(f"\n💡 Check logs for:")
    print(f"   - 'VLM high-precision sampling: X frames' (should be ~1 per second)")
    print(f"   - 'Layout transition detected at frame Xms' (if transitions exist)")
    print(f"   - 'Segment N rendered: ... boundary: Xms-Yms' (exact boundaries)")
    print(f"   - 'Segments concatenated successfully' (no gaps/overlaps)")


if __name__ == "__main__":
    main()
