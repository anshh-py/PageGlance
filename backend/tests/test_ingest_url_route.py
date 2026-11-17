from fastapi.testclient import TestClient

from app import main
from app.page_indexing import PageIndexResult

client = TestClient(main.app)


class FakeResponse:
    url = "https://example.com/"
    status_code = 200
    content = b"<html><title>Example</title><body>Hello world</body></html>"


def test_ingest_url_route_indexes_fetched_page(monkeypatch) -> None:
    async def fake_fetch(url, **kwargs):
        assert url == "https://example.com/"
        return FakeResponse()

    monkeypatch.setattr(main, "fetch_url", fake_fetch)
    monkeypatch.setattr(
        main, "index_page_context", lambda page: PageIndexResult(True, "doc-url-1", 1)
    )

    response = client.post("/ingest/url", json={"url": "https://example.com/"})

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == "doc-url-1"
    assert body["chunk_count"] == 1
    assert body["indexed"] is True
    assert body["source_url"] == "https://example.com/"
    assert body["title"] == "Example"


def test_ingest_url_route_rejects_unsupported_scheme() -> None:
    response = client.post("/ingest/url", json={"url": "ftp://example.com/file"})
    assert response.status_code == 422


def test_ingest_url_route_rejects_blank_url() -> None:
    response = client.post("/ingest/url", json={"url": "   "})
    assert response.status_code == 422


def test_ingest_url_route_blocks_private_address(monkeypatch) -> None:
    """fetch_url raises before any network I/O for literal private addresses."""
    response = client.post("/ingest/url", json={"url": "http://127.0.0.1/admin"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Private or local addresses are not allowed."


def test_ingest_url_route_propagates_fetch_error(monkeypatch) -> None:
    from app.url_fetcher import FetchError

    async def failing_fetch(url, **kwargs):
        raise FetchError("fetch_failed", "Could not fetch the URL.", 502)

    monkeypatch.setattr(main, "fetch_url", failing_fetch)

    response = client.post("/ingest/url", json={"url": "https://example.com/"})
    assert response.status_code == 502
    assert response.json()["detail"] == "Could not fetch the URL."
