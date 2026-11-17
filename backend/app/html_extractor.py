"""Convert fetched HTML into plain text and a title without external dependencies.

Uses only the standard-library HTMLParser so the URL ingestion path adds no new
runtime dependency. Scripts, styles, and templates are skipped; block elements
are separated by newlines so the shared normalization/chunking pipeline can
produce meaningful passages.
"""

from html.parser import HTMLParser

MAX_PAGE_TEXT_CHARS = 100_000

_SKIP_TAGS = frozenset({"script", "style", "noscript", "template"})
_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "br",
        "li",
        "tr",
        "section",
        "article",
        "blockquote",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }
)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "title":
            self._in_title = True
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
        self.parts.append(data)


def extract_html_text(content: bytes, base_url: str = "") -> dict[str, str]:
    """Return a bounded ``{"title": ..., "text": ...}`` dict for a fetched page."""
    try:
        text = content.decode("utf-8", errors="replace")
    except Exception:
        text = content.decode("latin-1", errors="replace")

    parser = _TextExtractor()
    try:
        parser.feed(text)
        parser.close()
    except Exception:
        # Malformed HTML should not break ingestion; keep whatever was parsed.
        pass

    lines = [line.strip() for line in "".join(parser.parts).splitlines()]
    body = "\n".join(line for line in lines if line)
    title = parser.title.strip()
    return {
        "title": title or base_url,
        "text": body[:MAX_PAGE_TEXT_CHARS],
    }
