"""
Rnet HTTP Handler for yt-dlp with browser impersonation.

Uses the rnet library to impersonate real browsers at the TLS/HTTP2 fingerprint level,
which is necessary to bypass YouTube's sophisticated bot detection.

YouTube's detection looks at:
- TLS fingerprints (cipher suites, extensions, ALPN, etc.)
- HTTP/2 settings (initial window size, header table size, max concurrent streams)
- HTTP/2 frame ordering

A regular Python urllib/requests connection has a completely different fingerprint
than a real Chrome browser. The rnet library solves this by impersonating real browsers.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
import urllib.request
import urllib.error
import urllib.parse

logger = logging.getLogger(__name__)

try:
    import rnet
    from rnet.impersonate import Impersonate
    RNET_AVAILABLE = True
except ImportError:
    RNET_AVAILABLE = False
    logger.warning("rnet not available - browser impersonation disabled")


class RnetHttpHandler:
    """Custom HTTP handler for yt-dlp using rnet with browser impersonation"""

    def __init__(self, proxy: Optional[str] = None):
        """
        Initialize the rnet handler.

        Args:
            proxy: Optional proxy URL (e.g., "socks5://user:pass@host:port")
        """
        self.proxy = proxy

        # List of browser impersonations to rotate through
        self.impersonations = [
            Impersonate.Chrome130,
            Impersonate.Chrome129,
            Impersonate.Chrome128,
            Impersonate.Chrome127,
            Impersonate.Chrome126,
            Impersonate.Edge131,
            Impersonate.Edge134,
            Impersonate.Firefox136,
            Impersonate.Firefox139,
            Impersonate.Safari18_3,
        ]
        self.current_impersonation = 0

    def get_client(self):
        """Get a new rnet client with random browser impersonation"""
        # Rotate through impersonations
        impersonation = self.impersonations[self.current_impersonation]
        self.current_impersonation = (self.current_impersonation + 1) % len(self.impersonations)

        client_opts = {
            "impersonate": impersonation,
            "timeout": 30.0,
            "follow_redirects": True,
            "max_redirects": 5,
        }

        # Add proxy if configured
        if self.proxy:
            client_opts["proxy"] = self.proxy

        return rnet.Client(**client_opts)

    async def make_request(
        self,
        url: str,
        method: str = 'GET',
        headers: Optional[Dict[str, str]] = None,
        data: Optional[bytes] = None
    ):
        """Make an HTTP request using rnet"""
        client = self.get_client()

        try:
            if method.upper() == 'GET':
                response = await client.get(url, headers=headers)
            elif method.upper() == 'POST':
                response = await client.post(url, headers=headers, data=data)
            elif method.upper() == 'HEAD':
                response = await client.head(url, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")

            # Return response in format yt-dlp expects
            return RnetResponse(response)

        except Exception as e:
            raise RnetError(f"Request failed: {str(e)}")
        finally:
            # Clean up client
            await client.aclose()


class RnetResponse:
    """Wrapper for rnet response to match yt-dlp expectations"""

    def __init__(self, rnet_response):
        self._response = rnet_response
        self._text = None
        self._content = None

    @property
    def status_code(self):
        return self._response.status_code

    @property
    def headers(self):
        return dict(self._response.headers)

    async def text(self):
        if self._text is None:
            self._text = await self._response.text()
        return self._text

    async def content(self):
        if self._content is None:
            self._content = await self._response.bytes()
        return self._content

    def read(self):
        """Synchronous read for compatibility"""
        return asyncio.run(self.content())

    def info(self):
        """Return response info"""
        return {
            'status': self.status_code,
            'headers': self.headers,
            'url': str(self._response.url)
        }


class RnetError(Exception):
    """Custom error for rnet handler"""
    pass


class RnetUrllibHandler(urllib.request.BaseHandler):
    """urllib handler that uses rnet for requests"""

    def __init__(self, proxy: Optional[str] = None):
        self.rnet_handler = RnetHttpHandler(proxy=proxy)

    def http_open(self, req):
        """Handle HTTP requests using rnet"""
        try:
            # Run async request in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(
                    self.rnet_handler.make_request(
                        url=req.get_full_url(),
                        method=req.get_method(),
                        headers=dict(req.headers),
                        data=req.data
                    )
                )
                return RnetUrllibResponse(response)
            finally:
                loop.close()
        except Exception as e:
            raise urllib.error.URLError(str(e))

    def https_open(self, req):
        """Handle HTTPS requests using rnet"""
        return self.http_open(req)


class RnetUrllibResponse:
    """urllib response wrapper for rnet response"""

    def __init__(self, rnet_response):
        self._response = rnet_response
        self._content = None

    def read(self):
        """Read response content"""
        if self._content is None:
            self._content = self._response.read()
        return self._content

    def info(self):
        """Get response info"""
        return self._response.info()

    def geturl(self):
        """Get final URL"""
        return self._response.info().get('url', '')

    def getcode(self):
        """Get status code"""
        return self._response.status_code

    def close(self):
        """Close response"""
        pass


def create_rnet_ydl_opts(proxy: Optional[str] = None) -> dict:
    """
    Create yt-dlp options with rnet handler for browser impersonation.

    Args:
        proxy: Optional proxy URL (e.g., "socks5://user:pass@host:port")

    Returns:
        Dictionary of yt-dlp options
    """
    if not RNET_AVAILABLE:
        logger.warning("rnet not available, returning empty options")
        return {}

    handler = RnetUrllibHandler(proxy=proxy)

    return {
        # Use custom HTTP handler with browser impersonation
        'http_handler': handler,
        'https_handler': handler,

        # Additional options for better compatibility
        'no_check_certificate': True,
        'socket_timeout': 30,
        'retries': 3,
        'fragment_retries': 3,

        # Keep existing options
        'quiet': True,
        'no_warnings': True,
    }


def is_rnet_available() -> bool:
    """Check if rnet is available for browser impersonation"""
    return RNET_AVAILABLE
