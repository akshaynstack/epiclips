"""
API Key Authentication for Genesis.

Provides FastAPI dependencies for validating API key authentication
on protected endpoints.
"""

import logging
from typing import Optional

from fastapi import Header, HTTPException, status

from app.config import get_settings

logger = logging.getLogger(__name__)


async def verify_api_key(
    x_genesis_api_key: Optional[str] = Header(None, alias="X-Genesis-API-Key"),
) -> None:
    """
    FastAPI dependency to verify the Genesis API key.

    If GENESIS_API_KEY is configured, requests must include a matching
    X-Genesis-API-Key header. If not configured, authentication is skipped
    (development mode).

    Args:
        x_genesis_api_key: The API key from the request header

    Raises:
        HTTPException: 401 if API key is missing or invalid
    """
    settings = get_settings()
    expected_key = settings.genesis_api_key

    # If no API key configured, skip validation (development mode)
    if not expected_key:
        logger.debug("GENESIS_API_KEY not configured, skipping authentication")
        return

    # API key is required
    if not x_genesis_api_key:
        logger.warning("Request missing X-Genesis-API-Key header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "X-Genesis-API-Key"},
        )

    # Validate the API key
    if x_genesis_api_key != expected_key:
        logger.warning("Invalid API key received")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "X-Genesis-API-Key"},
        )

    logger.debug("API key validated successfully")
