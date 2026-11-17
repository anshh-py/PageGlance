import asyncio

import httpx
import pytest

from app import url_fetcher
from app.url_fetcher import FetchError, fetch_url, validate_fetch_url


def test_validate_fetch_url_rejects_non_http_schemes() -> None:
    with pytest.raises(FetchError) as excinfo:
        validate_fetch_url("ftp://example.com/file")
    assert excinfo.value.code == "unsupported_scheme"
    with pytest.raises(FetchError):
        validate_fetch_url("file:///etc/passwd")


def test_validate_fetch_url_rejects_missing_host() -> None:
    with pytest.raises(FetchError) as excinfo:
        validate_fetch_url("https:///path")
    assert excinfo.value.code == "missing_host"


def test_validate_fetch_url_rejects_literal_private_addresses() -> None:
    for url in (
        "http://127.0.0.1/admin",
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
    ):
        with pytest.raises(FetchError) as excinfo:
            validate_fetch_url(url)
        assert excinfo.value.code == "ssrf_blocked"


def test_validate_fetch_url_accepts_public_url() -> None:
    parsed = validate_fetch_url("https://example.com/article?q=1#sec")
    assert parsed.hostname == "example.com"
    assert parsed.scheme == "https"


def _run(coro):
    return asyncio.run(coro)


def test_fetch_url_blocks_redirect_into_private_network(monkeypatch) -> None:
    """M9.5: a redirect that lands on a private address is re-validated."""
    monkeypatch.setattr(url_fetcher, "_assert_public_resolution", lambda hostname: None)

    def handler(request):
        if request.url.path == "/":
            return httpx.Response(302, headers={"location": "http://10.0.0.5/admin"})
        return httpx.Response(200, text="ok")

    async def scenario():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
            return await fetch_url("https://example.com/", http_client=client)

    with pytest.raises(FetchError) as excinfo:
        _run(scenario())
    assert excinfo.value.code == "ssrf_blocked"


def test_fetch_url_returns_body_and_respects_size_limit(monkeypatch) -> None:
    # Keep the unit test fully offline; the DNS resolution branch is covered
    # by the SSRF tests above.
    monkeypatch.setattr(url_fetcher, "_assert_public_resolution", lambda hostname: None)

    async def handler(request):
        return httpx.Response(200, text="<html><body>hello</body></html>")

    async def scenario(max_bytes):
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
            return await fetch_url("https://example.com/", max_bytes=max_bytes, http_client=client)

    response = _run(scenario(1024 * 1024))
    assert response.status_code == 200
    assert b"hello" in response.content

    with pytest.raises(FetchError) as excinfo:
        _run(scenario(10))
    assert excinfo.value.code == "content_too_large"


def test_fetch_url_rejects_private_url_before_network() -> None:
    with pytest.raises(FetchError) as excinfo:
        _run(fetch_url("http://127.0.0.1/"))
    assert excinfo.value.code == "ssrf_blocked"


def test_resolve_public_ip_returns_single_validated_address(monkeypatch) -> None:
    import socket

    # Simulate a hostname resolving to a mixture of private and public addresses.
    monkeypatch.setattr(
        url_fetcher.socket,
        "getaddrinfo",
        lambda hostname, port, proto=socket.IPPROTO_TCP: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 80)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("92.184.216.34", 80)),
        ],
    )
    assert url_fetcher._resolve_public_ip("example.com") == "92.184.216.34"

    # All-private resolution must be rejected.
    monkeypatch.setattr(
        url_fetcher.socket,
        "getaddrinfo",
        lambda hostname, port, proto=socket.IPPROTO_TCP: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.10", 80)),
        ],
    )
    with pytest.raises(FetchError) as excinfo:
        url_fetcher._resolve_public_ip("example.com")
    assert excinfo.value.code == "ssrf_blocked"


def test_pinned_backend_pins_connection_to_validated_ip(monkeypatch) -> None:
    """The TCP connection target is the validated IP, not a re-resolution."""
    captured = {}

    monkeypatch.setattr(
        url_fetcher,
        "_resolve_public_ip",
        lambda hostname: "92.184.216.34",
    )
    monkeypatch.setattr(
        url_fetcher,
        "_PinnedAsyncNetworkStream",
        lambda hostname, ip, port: captured.update({"hostname": hostname, "ip": ip, "port": port})
        or object(),
    )

    async def scenario():
        backend = url_fetcher._PinnedNetworkBackend()
        await backend.connect_tcp("example.com", 443)

    _run(scenario())
    assert captured["hostname"] == "example.com"
    assert captured["ip"] == "92.184.216.34"
    assert captured["port"] == 443
