"""Select useful, bounded excerpts from long webpage text."""

import re

TOKEN_PATTERN = re.compile(r"[a-z0-9]{3,}", re.IGNORECASE)
PARAGRAPH_PATTERN = re.compile(r"\n\s*\n+")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
STOP_WORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "because",
    "being",
    "between",
    "but",
    "can",
    "could",
    "does",
    "for",
    "from",
    "have",
    "how",
    "into",
    "its",
    "more",
    "most",
    "not",
    "of",
    "our",
    "should",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "through",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "with",
    "would",
    "you",
    "your",
}
CLIPPED_MARKER = " [passage shortened to fit the context limit]"


def _passages(text: str) -> list[str]:
    blocks = [part.strip() for part in PARAGRAPH_PATTERN.split(text.strip()) if part.strip()]
    if len(blocks) == 1:
        lines = [line.strip() for line in blocks[0].splitlines() if line.strip()]
        if len(lines) > 1:
            blocks = lines

    passages: list[str] = []
    for block in blocks:
        if len(block) > 4000:
            passages.extend(part.strip() for part in SENTENCE_PATTERN.split(block) if part.strip())
        else:
            passages.append(block)
    return passages


def _fit_passage(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    available = max(1, limit - len(CLIPPED_MARKER))
    candidate = text[:available]
    sentence_end = max(candidate.rfind(". "), candidate.rfind("! "), candidate.rfind("? "))
    if sentence_end >= available // 2:
        candidate = candidate[: sentence_end + 1]
    else:
        word_end = candidate.rfind(" ")
        if word_end > available // 2:
            candidate = candidate[:word_end]
    return candidate.rstrip() + CLIPPED_MARKER


def select_page_context(question: str, page: dict[str, str], max_chars: int) -> dict[str, str]:
    """Keep the introduction and query-relevant passages within a character budget."""
    text = page["text"]
    if len(text) <= max_chars:
        return page

    passages = _passages(text)
    if not passages:
        return {**page, "text": ""}

    query_terms = {
        term.lower() for term in TOKEN_PATTERN.findall(question) if term.lower() not in STOP_WORDS
    }
    title_terms = {term.lower() for term in TOKEN_PATTERN.findall(page.get("title", ""))}

    def score(index: int) -> tuple[int, int, int]:
        terms = {term.lower() for term in TOKEN_PATTERN.findall(passages[index])}
        overlap = len(query_terms & terms)
        title_overlap = len(title_terms & terms)
        # Earlier page text breaks ties, retaining the article's useful flow.
        return overlap, title_overlap, -index

    ranked = sorted(range(1, len(passages)), key=score, reverse=True)
    selected: dict[int, str] = {}
    remaining = max_chars

    for index in [0, *ranked]:
        if index != 0 and score(index)[0] == 0:
            break
        passage = passages[index]
        separator = 2 if selected else 0
        available = remaining - separator
        if available <= len(CLIPPED_MARKER):
            continue
        if len(passage) > available:
            passage = _fit_passage(passage, available)
        selected[index] = passage
        remaining -= separator + len(passage)

    if not selected:
        selected[0] = _fit_passage(passages[0], max_chars)

    selected_text = "\n\n".join(selected[index] for index in sorted(selected))
    return {**page, "text": selected_text}
