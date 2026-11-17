from app.html_extractor import extract_html_text


def test_extract_html_text_strips_tags_and_keeps_title() -> None:
    html = (
        b"<html><head><title>  Example Page </title></head>"
        b"<body><p>Hello <b>world</b>.</p></body></html>"
    )
    result = extract_html_text(html, base_url="https://example.com/")
    assert result["title"] == "Example Page"
    assert "Hello world." in result["text"]


def test_extract_html_text_skips_script_and_style() -> None:
    html = (
        b"<html><head><style>body{color:red}</style></head><body>"
        b"<script>window.alert('xss');</script><p>Visible text</p></body></html>"
    )
    result = extract_html_text(html)
    assert "Visible text" in result["text"]
    assert "alert" not in result["text"]
    assert "color:red" not in result["text"]


def test_extract_html_text_falls_back_to_base_url_title() -> None:
    result = extract_html_text(b"<html><body><p>No title here</p></body></html>", base_url="https://x.dev/")
    assert result["title"] == "https://x.dev/"


def test_extract_html_text_bounds_output_length() -> None:
    big = b"<p>" + b"a" * 200_000 + b"</p>"
    result = extract_html_text(big)
    assert len(result["text"]) <= 100_000
