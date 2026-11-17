"""Deterministic, conservative normalization for already-extracted text."""

import hashlib
import re

_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Normalize line endings and redundant blank lines while preserving paragraphs."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ").replace("\u2028", "\n").replace("\u2029", "\n\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = _EXCESS_BLANK_LINES.sub("\n\n", normalized)
    return normalized.strip()


def content_hash(text: str) -> str:
    """Return a stable SHA-256 digest for normalized UTF-8 text."""
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def exact_text_hash(text: str) -> str:
    """Hash exact text bytes for caches whose inputs must match byte-for-byte."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def has_duplicate_content(first: str, second: str) -> bool:
    """Compare normalized text using its stable content digest."""
    return content_hash(first) == content_hash(second)
