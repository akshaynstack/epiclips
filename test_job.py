#!/usr/bin/env python3
"""
Test script for ViewCreator Genesis AI Clipping API.

Usage:
    python test_job.py                    # Submit job and wait for completion
    python test_job.py --download-only    # Download clips from last job
    python test_job.py --video /path/to/local/video.mp4  # Upload local file and process
"""

import argparse
import json
import os
import time
import uuid
import requests
import boto3
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
API_KEY = "your-api-key"
TEST_VIDEO_URL = "https://www.youtube.com/watch?v=qVW7uIQgTGQ"  # Claude Opus 4.5 video

# AWS Configuration (from .env)
AWS_ACCESS_KEY_ID = "AKIA4CRJFEIHMBK5EAGK"
AWS_SECRET_ACCESS_KEY = "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"
AWS_REGION = "us-east-1"
S3_BUCKET = "viewcreator-media-dev-830088749582-d3de3c10"

# Output directory
OUTPUT_DIR = Path("test_clips")


def is_local_file(path: str) -> bool:
    """Check if the path is a local file."""
    # Check if it's an absolute path or relative path that exists
    return os.path.exists(path) and os.path.isfile(path)


def upload_local_file_to_s3(local_path: str) -> str:
    """Upload a local file to S3 and return the S3 key."""
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    
    # Generate unique S3 key
    file_ext = Path(local_path).suffix
    s3_key = f"test-uploads/{uuid.uuid4()}{file_ext}"
    
    print(f"📤 Uploading local file to S3...")
    print(f"   Local: {local_path}")
    print(f"   S3: s3://{S3_BUCKET}/{s3_key}")
    
    file_size = os.path.getsize(local_path) / 1024 / 1024
    print(f"   Size: {file_size:.1f} MB")
    
    s3_client.upload_file(local_path, S3_BUCKET, s3_key)
    print(f"✅ Upload complete!")
    
    return s3_key


def submit_job(video_url: str = TEST_VIDEO_URL, max_clips: int = 3, duration_ranges: list = None, s3_key: str = None):
    """Submit a clipping job."""
    if duration_ranges is None:
        duration_ranges = ["short"]
    
    headers = {
        "Content-Type": "application/json",
        "X-Genesis-API-Key": API_KEY
    }
    
    payload = {
        "max_clips": max_clips,
        "duration_ranges": duration_ranges,
        "include_captions": True,
        "caption_preset": "viral_gold",
        "layout_type": "auto"
    }
    
    # Use s3_key if provided, otherwise use video_url
    if s3_key:
        payload["s3_key"] = s3_key
        print(f"\n🚀 Submitting job for S3 key: {s3_key}")
    else:
        payload["video_url"] = video_url
        print(f"\n🚀 Submitting job for: {video_url}")
    
    print(f"   Max clips: {max_clips}, Duration ranges: {duration_ranges}")
    
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
    """Poll job status until completion."""
    print(f"\n⏳ Waiting for job {job_id} to complete...")
    
    while True:
        response = requests.get(f"{BASE_URL}/ai-clipping/jobs/{job_id}")
        
        if response.status_code != 200:
            print(f"❌ Failed to get job status: {response.status_code}")
            return None
        
        status = response.json()
        progress = status.get("progress_percent", 0)
        current_step = status.get("current_step", "")
        job_status = status.get("status", "")
        
        print(f"   [{progress:5.1f}%] {job_status}: {current_step}")
        
        if job_status == "completed":
            print(f"\n✅ Job completed!")
            return status.get("output")
        elif job_status == "failed":
            print(f"\n❌ Job failed: {status.get('error')}")
            return None
        
        time.sleep(poll_interval)


def download_clips_from_output(output: dict, version_suffix: str = ""):
    """Download clips from job output."""
    if not output:
        print("❌ No output to download")
        return
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    clips = output.get("clips", [])
    print(f"\n📥 Downloading {len(clips)} clips...")
    
    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    
    for clip in clips:
        clip_index = clip.get("clip_index", 0)
        s3_url = clip.get("s3_url", "")
        
        if not s3_url:
            print(f"   ⚠️ Clip {clip_index} has no S3 URL")
            continue
        
        # Extract S3 key from URL
        # URL format: https://bucket.s3.region.amazonaws.com/key
        # or: https://bucket.s3.amazonaws.com/key
        try:
            if ".s3." in s3_url:
                s3_key = s3_url.split(".amazonaws.com/")[1]
            else:
                s3_key = s3_url.split(f"{S3_BUCKET}/")[1]
        except IndexError:
            print(f"   ⚠️ Could not parse S3 URL: {s3_url}")
            continue
        
        # Download
        local_filename = f"clip_{clip_index:02d}{version_suffix}.mp4"
        local_path = OUTPUT_DIR / local_filename
        
        print(f"   Downloading clip {clip_index} -> {local_path}")
        
        try:
            s3_client.download_file(S3_BUCKET, s3_key, str(local_path))
            file_size = local_path.stat().st_size / 1024 / 1024
            print(f"   ✅ Downloaded: {local_filename} ({file_size:.1f} MB)")
        except Exception as e:
            print(f"   ❌ Failed to download: {e}")
    
    print(f"\n✅ Clips saved to: {OUTPUT_DIR.absolute()}")


def download_from_urls(urls: list, version_suffix: str = "_latest"):
    """Download clips from S3 URLs directly."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print(f"\n📥 Downloading {len(urls)} clips...")
    
    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    
    for i, s3_url in enumerate(urls):
        # Extract S3 key from URL
        try:
            if ".s3." in s3_url:
                s3_key = s3_url.split(".amazonaws.com/")[1]
            else:
                print(f"   ⚠️ Could not parse S3 URL: {s3_url}")
                continue
        except IndexError:
            print(f"   ⚠️ Could not parse S3 URL: {s3_url}")
            continue
        
        # Download
        local_filename = f"clip_{i:02d}{version_suffix}.mp4"
        local_path = OUTPUT_DIR / local_filename
        
        print(f"   Downloading -> {local_path}")
        
        try:
            s3_client.download_file(S3_BUCKET, s3_key, str(local_path))
            file_size = local_path.stat().st_size / 1024 / 1024
            print(f"   ✅ Downloaded: {local_filename} ({file_size:.1f} MB)")
        except Exception as e:
            print(f"   ❌ Failed to download: {e}")
    
    print(f"\n✅ Clips saved to: {OUTPUT_DIR.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="Test ViewCreator Genesis AI Clipping API")
    parser.add_argument("--download-only", action="store_true", help="Only download clips from URLs")
    parser.add_argument("--urls", nargs="+", help="S3 URLs to download")
    parser.add_argument("--video", type=str, default=TEST_VIDEO_URL, help="YouTube video URL")
    parser.add_argument("--max-clips", type=int, default=3, help="Maximum clips to generate")
    parser.add_argument("--duration", type=str, default="short", help="Duration range: short, medium, long")
    parser.add_argument("--version", type=str, default="", help="Version suffix for downloaded files")
    
    args = parser.parse_args()
    
    if args.download_only:
        if args.urls:
            download_from_urls(args.urls, args.version or "_manual")
        else:
            print("❌ No URLs provided. Use --urls to specify S3 URLs.")
        return
    
    # Check if video is a local file
    s3_key = None
    video_url = args.video
    
    if is_local_file(args.video):
        # Upload local file to S3 first
        s3_key = upload_local_file_to_s3(args.video)
        video_url = None  # Will use s3_key instead
    
    # Submit and process job
    duration_ranges = [args.duration]
    job_id = submit_job(video_url, args.max_clips, duration_ranges, s3_key=s3_key)
    
    if not job_id:
        return
    
    # Wait for completion
    output = poll_job_status(job_id)
    
    if output:
        # Download clips
        version = args.version or f"_v{int(time.time()) % 1000}"
        download_clips_from_output(output, version)
        
        # Print summary
        print("\n📊 Job Summary:")
        print(f"   Source: {output.get('source_video_title', 'Unknown')}")
        print(f"   Duration: {output.get('source_video_duration_seconds', 0):.1f}s")
        print(f"   Clips: {output.get('total_clips', 0)}")
        print(f"   Processing time: {output.get('processing_time_seconds', 0):.1f}s")
        
        for clip in output.get("clips", []):
            print(f"\n   Clip {clip.get('clip_index', 0)}:")
            print(f"      Layout: {clip.get('layout_type', 'unknown')}")
            print(f"      Duration: {clip.get('duration_ms', 0) / 1000:.1f}s")
            print(f"      Virality: {clip.get('virality_score', 0):.2f}")
            print(f"      Summary: {clip.get('summary', '')[:80]}...")


if __name__ == "__main__":
    main()
