"""Bound multipart request bodies before Starlette spools uploaded files."""

import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.document_ingestion import MAX_UPLOAD_BYTES

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
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                if int(raw_length) > MAX_UPLOAD_REQUEST_BYTES:
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
                if received_bytes > MAX_UPLOAD_REQUEST_BYTES:
                    raise _UploadBodyTooLarge
            return message

        try:
            await self.app(scope, bounded_receive, send)
        except _UploadBodyTooLarge:
            await self._reject(send)

    @staticmethod
    async def _reject(send: Send) -> None:
        body = json.dumps(
            {"detail": f"Upload request must be at most {MAX_UPLOAD_REQUEST_BYTES} bytes."}
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
