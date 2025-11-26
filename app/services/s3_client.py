"""
S3 client service for downloading videos and uploading detection results.
"""

import json
import logging
import os
import tempfile
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from app.config import get_settings

logger = logging.getLogger(__name__)


class S3Client:
    """
    Service for interacting with AWS S3.
    
    Handles downloading source videos and uploading detection results.
    """

    def __init__(
        self,
        bucket: Optional[str] = None,
        region: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
    ):
        """
        Initialize S3 client.
        
        Args:
            bucket: S3 bucket name (defaults to settings)
            region: AWS region (defaults to settings)
            access_key_id: AWS access key (defaults to settings/env)
            secret_access_key: AWS secret key (defaults to settings/env)
        """
        settings = get_settings()
        
        self.bucket = bucket or settings.s3_bucket
        self.region = region or settings.aws_region

        # Build client kwargs
        client_kwargs = {"region_name": self.region}
        
        if access_key_id and secret_access_key:
            client_kwargs["aws_access_key_id"] = access_key_id
            client_kwargs["aws_secret_access_key"] = secret_access_key
        elif settings.aws_access_key_id and settings.aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = settings.aws_access_key_id
            client_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key

        self._client = boto3.client("s3", **client_kwargs)
        logger.info(f"S3 client initialized for bucket: {self.bucket}")

    def download_video(self, s3_key: str, local_path: Optional[str] = None) -> str:
        """
        Download a video file from S3.
        
        Args:
            s3_key: S3 key of the video file
            local_path: Optional local path to save to (will create temp file if not provided)
            
        Returns:
            Local path where the video was saved
            
        Raises:
            ClientError: If download fails
        """
        if local_path is None:
            # Create a temp file with the same extension
            ext = os.path.splitext(s3_key)[1] or ".mp4"
            fd, local_path = tempfile.mkstemp(suffix=ext)
            os.close(fd)

        logger.info(f"Downloading s3://{self.bucket}/{s3_key} to {local_path}")
        
        try:
            self._client.download_file(self.bucket, s3_key, local_path)
            file_size = os.path.getsize(local_path)
            logger.info(f"Downloaded {file_size / 1024 / 1024:.2f} MB to {local_path}")
            return local_path
        except ClientError as e:
            logger.error(f"Failed to download from S3: {e}")
            # Clean up temp file if download failed
            if os.path.exists(local_path):
                os.remove(local_path)
            raise

    def upload_json(self, data: dict, s3_key: str) -> str:
        """
        Upload detection results JSON to S3.
        
        Args:
            data: Dictionary to serialize as JSON
            s3_key: S3 key to upload to
            
        Returns:
            S3 key where the file was uploaded
            
        Raises:
            ClientError: If upload fails
        """
        logger.info(f"Uploading JSON to s3://{self.bucket}/{s3_key}")
        
        try:
            json_bytes = json.dumps(data, indent=2).encode("utf-8")
            self._client.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=json_bytes,
                ContentType="application/json",
            )
            logger.info(f"Uploaded {len(json_bytes)} bytes to {s3_key}")
            return s3_key
        except ClientError as e:
            logger.error(f"Failed to upload to S3: {e}")
            raise

    def upload_file(self, local_path: str, s3_key: str, content_type: Optional[str] = None) -> str:
        """
        Upload a file to S3.
        
        Args:
            local_path: Local file path
            s3_key: S3 key to upload to
            content_type: Optional content type
            
        Returns:
            S3 key where the file was uploaded
        """
        logger.info(f"Uploading {local_path} to s3://{self.bucket}/{s3_key}")
        
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        try:
            self._client.upload_file(local_path, self.bucket, s3_key, ExtraArgs=extra_args or None)
            file_size = os.path.getsize(local_path)
            logger.info(f"Uploaded {file_size / 1024 / 1024:.2f} MB to {s3_key}")
            return s3_key
        except ClientError as e:
            logger.error(f"Failed to upload to S3: {e}")
            raise

    def file_exists(self, s3_key: str) -> bool:
        """
        Check if a file exists in S3.
        
        Args:
            s3_key: S3 key to check
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            self._client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError:
            return False

    def get_presigned_url(self, s3_key: str, expires_in: int = 3600) -> str:
        """
        Generate a presigned URL for a file.
        
        Args:
            s3_key: S3 key
            expires_in: URL expiration time in seconds (default 1 hour)
            
        Returns:
            Presigned URL
        """
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": s3_key},
            ExpiresIn=expires_in,
        )



