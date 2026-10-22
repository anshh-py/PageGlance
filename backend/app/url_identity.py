"""Validation and conservative normalization for webpage source URLs."""

from urllib.parse import urlsplit, urlunsplit


def normalize_page_url(value: str) -> str:
    """Normalize only URL components that are case-insensitive or defaulted.

    Paths, query strings, and fragments remain intact because websites may use them
    to distinguish content (including client-side routed pages).
    """
    if not value or value != value.strip() or any(character.isspace() for character in value):
        raise ValueError("Page URL must not be blank or contain whitespace.")
    try:
        parsed = urlsplit(value)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        raise ValueError("Page URL is invalid.") from None

    if scheme not in {"http", "https"} or not hostname:
        raise ValueError("Page URL must use http or https and include a hostname.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Page URL must not contain credentials.")

    host = hostname.lower()
    if ":" in host:  # Preserve IPv6 authority syntax.
        host = f"[{host}]"
    if port is not None and port != (80 if scheme == "http" else 443):
        host = f"{host}:{port}"

    return urlunsplit((scheme, host, parsed.path or "/", parsed.query, parsed.fragment))
