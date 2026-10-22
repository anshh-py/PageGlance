"""Conservative URL validation for persistent webpage identity."""

import pytest

from app.ingestion import ingest_text
from app.url_identity import normalize_page_url


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "HTTPS://EXAMPLE.TEST:443/article#section",
            "https://example.test/article#section",
        ),
        ("http://Example.test:80", "http://example.test/"),
        ("https://example.test/article?b=2&a=1", "https://example.test/article?b=2&a=1"),
    ],
)
def test_page_url_normalization_only_changes_safe_authority_details(
    source: str, expected: str
) -> None:
    assert normalize_page_url(source) == expected


@pytest.mark.parametrize(
    "source",
    ["", "file:///C:/document.pdf", "chrome://extensions", "https://user:pass@example.test/"],
)
def test_page_url_rejects_unsupported_schemes_and_credentials(source: str) -> None:
    with pytest.raises(ValueError):
        normalize_page_url(source)


def test_webpage_ingestion_uses_normalized_url_for_stable_identity() -> None:
    first = ingest_text(
        "Same page body", source_type="webpage", source_url="HTTPS://EXAMPLE.TEST:443/article"
    )
    second = ingest_text(
        "Same page body", source_type="webpage", source_url="https://example.test/article"
    )

    assert first.document.id == second.document.id
    assert first.document.source_url == "https://example.test/article"
