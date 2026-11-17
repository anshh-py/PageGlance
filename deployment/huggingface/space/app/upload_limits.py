"""Bound multipart request bodies before Starlette spools uploaded files."""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

DEFAULT_MAX_UPLOAD_MB = 100
MAX_CONFIGURED_UPLOAD_MB = 512
BYTES_PER_MIB = 1024 * 1024


def get_max_upload_mb() -> int:
    """Return the bounded upload size in MiB from the repository environment."""
    env_file = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=env_file, override=False)
    raw_value = os.getenv("PAGEGLANCE_MAX_UPLOAD_MB", str(DEFAULT_MAX_UPLOAD_MB)).strip()
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning(
            "upload_limit_configuration_invalid reason=not_integer using_default_mb=%d",
            DEFAULT_MAX_UPLOAD_MB,
        )
        return DEFAULT_MAX_UPLOAD_MB
    if not 1 <= value <= MAX_CONFIGURED_UPLOAD_MB:
        logger.warning(
            "upload_limit_configuration_invalid reason=out_of_range allowed_mb=1-%d "
            "using_default_mb=%d",
            MAX_CONFIGURED_UPLOAD_MB,
            DEFAULT_MAX_UPLOAD_MB,
        )
        return DEFAULT_MAX_UPLOAD_MB
    return value


def get_max_upload_bytes() -> int:
    return get_max_upload_mb() * BYTES_PER_MIB


MAX_UPLOAD_BYTES = DEFAULT_MAX_UPLOAD_MB * BYTES_PER_MIB

# A single file may use the full allowance plus bounded multipart headers and boundary data.
MAX_MULTIPART_OVERHEAD_BYTES = 256 * 1024
MAX_UPLOAD_REQUEST_BYTES = MAX_UPLOAD_BYTES + MAX_MULTIPART_OVERHEAD_BYTES
_UPLOAD_PATH = "/documents/upload"


class UploadBodyLimitMiddleware:
    """Reject oversized upload requests before multipart parsing writes file parts."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != _UPLOAD_PATH
        ):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        max_request_bytes = get_max_upload_bytes() + MAX_MULTIPART_OVERHEAD_BYTES
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                if int(raw_length) > max_request_bytes:
                    await self._reject(send)
                    return
            except ValueError:
                # Continue with streaming enforcement for malformed/missing lengths.
                pass

        received_bytes = 0

        async def bounded_receive() -> Message:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > max_request_bytes:
                    raise _UploadBodyTooLarge
            return message

        try:
            await self.app(scope, bounded_receive, send)
        except _UploadBodyTooLarge:
            await self._reject(send)

    @staticmethod
    async def _reject(send: Send) -> None:
        body = json.dumps(
            {
                "detail": (
                    f"Upload request must be at most "
                    f"{get_max_upload_bytes() + MAX_MULTIPART_OVERHEAD_BYTES} bytes."
                )
            }
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class _UploadBodyTooLarge(Exception):
    """Internal control flow raised when streamed body bytes exceed the limit."""
