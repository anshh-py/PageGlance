"""Safe server-side URL fetching with SSRF protection.

Used by the URL ingestion endpoint. The fetcher only ever follows http/https,
rejects hostnames and literal addresses that resolve to private, loopback, or
link-local networks, bounds the response body, and caps redirects.
"""

import asyncio
import ipaddress
import logging
import socket
import ssl
from collections.abc import Iterable
from urllib.parse import urlsplit

import httpcore
import httpx
from httpx._transports.default import AsyncResponseStream

logger = logging.getLogger(__name__)

MAX_FETCH_BYTES = 5 * 1024 * 1024  # 5 MiB default content cap
FETCH_TIMEOUT_SECONDS = 15.0
MAX_REDIRECTS = 5

ALLOWED_SCHEMES = frozenset({"http", "https"})


class FetchError(RuntimeError):
    """A predictable, user-safe fetch failure with an HTTP status code."""

    def __init__(self, code: str, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _is_disallowed_ip(address: str) -> bool:
    """Return True when a literal IP is private, loopback, or link-local."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        # Not a literal IP (hostname) — checked separately via DNS resolution.
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local


def validate_fetch_url(url: str):
    """Validate scheme and (for literal IPs) address safety before fetching.

    Hostname safety is resolved during ``fetch_url`` because a hostname may
    resolve to multiple addresses. Literal IPs are checked eagerly here.
    """
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise FetchError("invalid_url", "The URL is not valid.") from exc

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise FetchError(
            "unsupported_scheme", "Only http and https URLs are supported.", 400
        )
    if not parsed.hostname:
        raise FetchError("missing_host", "The URL must include a hostname.", 400)
    if _is_disallowed_ip(parsed.hostname):
        raise FetchError(
            "ssrf_blocked", "Private or local addresses are not allowed.", 400
        )
    return parsed


def _assert_public_resolution(hostname: str) -> None:
    """Resolve a hostname and reject it if any address is private/loopback/link-local."""
    try:
        infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise FetchError(
            "dns_resolution_failed", "Could not resolve the URL host.", 422
        ) from exc
    for info in infos:
        address = info[4][0]
        if _is_disallowed_ip(address):
            raise FetchError(
                "ssrf_blocked", "The URL resolves to a private or local address.", 400
            )


def _resolve_public_ip(hostname: str) -> str:
    """Resolve ``hostname`` once and return ONE validated public IP for connecting.

    This is the single DNS resolution the connection layer relies on: the TCP
    socket is opened to the returned address rather than re-resolving the
    hostname at connect time. Combining validation and the connection target in
    one lookup closes the DNS-rebinding (TOCTOU) window where a hostname could
    be validated as public and later resolve to a private address when the
    socket is actually opened.
    """
    try:
        infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise FetchError(
            "dns_resolution_failed", "Could not resolve the URL host.", 422
        ) from exc
    for info in infos:
        address = info[4][0]
        if not _is_disallowed_ip(address):
            return address
    raise FetchError(
        "ssrf_blocked", "The URL resolves to a private or local address.", 400
    )


class _PinnedAsyncNetworkStream(httpcore.AsyncNetworkStream):
    """An async stream that opens its socket to a pre-validated public IP.

    The original hostname is retained so TLS SNI/Host headers still target the
    real host while the TCP connection goes to the validated address. This is
    what prevents DNS rebinding: the address used for the connection is the
    exact address that was validated for public reachability.
    """

    def __init__(self, hostname: str, ip: str, port: int) -> None:
        self._hostname = hostname
        self._ip = ip
        self._port = port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._sock: socket.socket | None = None

    async def _ensure(self) -> tuple[asyncio.AbstractEventLoop, socket.socket]:
        if self._sock is None:
            loop = asyncio.get_running_loop()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setblocking(False)
            await loop.sock_connect(sock, (self._ip, self._port))
            self._loop = loop
            self._sock = sock
        assert self._loop is not None and self._sock is not None
        return self._loop, self._sock

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        loop, sock = await self._ensure()
        return await loop.sock_recv(sock, max_bytes)

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        loop, sock = await self._ensure()
        await loop.sock_sendall(sock, buffer)

    async def aclose(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> "_PinnedAsyncNetworkStream":
        loop, sock = await self._ensure()
        tls_sock = ssl_context.wrap_socket(
            sock,
            server_hostname=server_hostname or self._hostname,
            do_handshake_on_connect=False,
        )
        tls_sock.setblocking(False)
        await loop.sock_handshake(tls_sock)
        self._sock = tls_sock
        return self

    def get_extra_info(self, info: str):
        if self._sock is None:
            return None
        if info == "socket":
            return self._sock
        if info == "ssl_object":
            return getattr(self._sock, "_sslobj", None)
        return None


class _PinnedNetworkBackend(httpcore.AsyncNetworkBackend):
    """Network backend whose TCP connections are pinned to validated public IPs."""

    def __init__(self) -> None:
        self._resolver = _resolve_public_ip

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[tuple] | None = None,
    ) -> _PinnedAsyncNetworkStream:
        ip = self._resolver(host)
        return _PinnedAsyncNetworkStream(host, ip, port)

    async def connect_unix_socket(
        self, path: str, timeout: float | None = None
    ) -> _PinnedAsyncNetworkStream:
        raise NotImplementedError("Unix sockets are not supported.")

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)



async def _build_pinned_http_client(timeout: float) -> httpx.AsyncClient:
    """Build an httpx client whose TCP connections are pinned to validated public IPs."""
    pool = httpcore.AsyncConnectionPool(
        network_backend=_PinnedNetworkBackend(),
        http1=True,
        http2=False,
        retries=0,
    )

    class _PinnedTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            req = httpcore.Request(
                method=request.method,
                url=httpcore.URL(
                    scheme=request.url.raw_scheme,
                    host=request.url.raw_host,
                    port=request.url.port,
                    target=request.url.raw_path,
                ),
                headers=request.headers.raw,
                content=request.stream,
                extensions=request.extensions,
            )
            resp = await pool.handle_async_request(req)
            return httpx.Response(
                status_code=resp.status,
                headers=resp.headers,
                stream=AsyncResponseStream(resp.stream),
                extensions=resp.extensions,
            )

    return httpx.AsyncClient(
        transport=_PinnedTransport(),
        follow_redirects=True,
        timeout=httpx.Timeout(timeout),
        max_redirects=MAX_REDIRECTS,
        headers={"User-Agent": "PageGlance/0.1.0"},
    )


async def fetch_url(
    url: str,
    *,
    max_bytes: int = MAX_FETCH_BYTES,
    timeout: float = FETCH_TIMEOUT_SECONDS,
    http_client: httpx.AsyncClient | None = None,
) -> httpx.Response:
    """Fetch a validated URL with SSRF guards, bounded body size, and redirect caps."""
    validate_fetch_url(url)
    hostname = urlsplit(url).hostname or ""
    _assert_public_resolution(hostname)

    owns_client = http_client is None
    if owns_client:
        http_client = _build_pinned_http_client(timeout)
    try:
        response = await http_client.get(url)
    except httpx.HTTPError:
        logger.warning("fetch_http_error scheme=%s", urlsplit(url).scheme)
        raise FetchError("fetch_failed", "Could not fetch the URL.", 502) from None
    finally:
        if owns_client:
            await http_client.aclose()

    # Re-check the final address (defense against a redirect into a private network).
    try:
        final_parsed = validate_fetch_url(str(response.url))
    except FetchError:
        raise
    if final_parsed.hostname:
        _assert_public_resolution(final_parsed.hostname)

    if len(response.content) > max_bytes:
        raise FetchError(
            "content_too_large", f"Fetched content exceeds {max_bytes} bytes.", 413
        )
    return response
