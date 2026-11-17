"""Safe TXT/PDF extraction and persistence for uploaded user documents."""

import csv
import io
import json
import logging
import math
import os
import re
import subprocess
import tempfile
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import BinaryIO
from xml.etree import ElementTree as ET

from dotenv import load_dotenv
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.contracts import (
    BlockReference,
    ContentSource,
    ExtractedDocument,
)
from app.converters import ingest_extracted_document
from app.db import DatabaseConfigurationError
from app.documents import Document, DocumentSourceType
from app.embeddings import EmbeddingError, EmbeddingService
from app.normalization import normalize_text
from app.page_indexing import get_ingestion_session_factory
from app.repository import DocumentRepository
from app.upload_limits import get_max_upload_bytes, get_max_upload_mb

logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = get_max_upload_bytes()
MAX_EXTRACTED_TEXT_CHARS = 100_000
MAX_UPLOAD_CHUNKS = 128
MAX_PDF_PAGES = 200
MAX_OCR_PAGES = MAX_PDF_PAGES
MAX_OCR_PIXELS_PER_PAGE = 20_000_000
MAX_OCR_PAGE_SECONDS = 20
MAX_OCR_TOTAL_SECONDS = 900
OCR_BATCH_PAGES = 4
MAX_CONFIGURED_OCR_SECONDS = 3600
MAX_OCR_DPI = 150
MAX_FILENAME_CHARS = 255
MAX_CSV_ROWS = 1_000_000
MAX_CSV_COLUMNS = 2_000
MAX_CSV_TABLE_ROWS = 500
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 100_000
MAX_JSON_LINES_PER_BLOCK = 200
MAX_XML_DEPTH = 64
MAX_XML_NODES = 200_000
MAX_XML_ATTRS_PER_ELEMENT = 500
MAX_XML_LINES_PER_BLOCK = 200
MAX_IMAGE_MEGAPIXELS = 25
MAX_IMAGE_SIDE_PIXELS = 16_000
MAX_IMAGE_OCR_SECONDS = 40
MAX_IMAGE_UPSCALE_FACTOR = 2.0
MAX_IMAGE_MIN_SIDE_UPSCALE = 300
MAX_IMAGE_OCR_ATTEMPTS = 5
MAX_IMAGE_OCR_LAYOUT_RECORDS = 100
MAX_IMAGE_OCR_LAYOUT_METADATA_BYTES = 16 * 1024
MAX_SRT_SUBTITLES = 2000
MAX_SRT_TIMESTAMP_METADATA_BYTES = 16 * 1024
MAX_VTT_CUES = 2000
MAX_VTT_TIMESTAMP_METADATA_BYTES = 16 * 1024
# Raw audio/video binary extensions never contain ingestible transcript text.
_UNSUPPORTED_MEDIA_SUFFIXES = frozenset(
    {
        "mp3", "mp4", "wav", "mkv", "mov", "avi", "flac", "ogg", "ogv",
        "webm", "m4a", "m4v", "mpeg", "mpg", "aac", "wma", "flv", "3gp",
    }
)
DEFAULT_IMAGE_OCR_LANGUAGE = "eng"
MAX_IMAGE_OCR_LANGS = 4
DEFAULT_TESSERACT_COMMAND = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DOCX_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
DOCX_MIME_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
HTML_MIME_TYPE = "text/html"
IMAGE_MIME_TYPES = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "tiff": "image/tiff",
}
IMAGE_SUPPORTED_FORMATS = frozenset(IMAGE_MIME_TYPES)
HTML_TITLE_MAX = 200
HTML_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)

HTML_SKIP_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "head",
        "meta",
        "link",
        "svg",
        "canvas",
        "iframe",
        "object",
        "embed",
        "nav",
        "form",
        "button",
        "input",
        "select",
        "textarea",
    }
)
HTML_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "main",
        "aside",
        "header",
        "footer",
        "blockquote",
        "li",
        "tr",
        "ul",
        "ol",
        "table",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "pre",
        "br",
    }
)
HTML_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
HTML_TABLE_TAGS = frozenset({"table", "thead", "tbody", "tfoot", "tr", "td", "th"})
HTML_LIST_TAGS = frozenset({"ul", "ol", "li"})
HTML_HIDDEN_RE = re.compile(
    r"display\s*:\s*none|visibility\s*:\s*hidden|hidden", re.IGNORECASE
)
HTML_BOILERPLATE_RE = re.compile(
    r"(?:cookie|consent|gdpr|advert|\bad\b|banner|popup|modal)\b|ad-container|ad-holder",
    re.IGNORECASE,
)


def _configured_ocr_limit(name: str, default: int, maximum: int) -> int:
    """Read an integer OCR limit from the repository environment with hard bounds."""
    env_file = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    load_dotenv(dotenv_path=os.path.abspath(env_file), override=False)
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("pdf_ocr_configuration_invalid setting=%s reason=not_integer", name)
        return default
    if not 1 <= value <= maximum:
        logger.warning(
            "pdf_ocr_configuration_invalid setting=%s reason=out_of_range max=%d", name, maximum
        )
        return default
    return value


def get_max_ocr_pages() -> int:
    """Return the bounded number of scanned pages allowed in one PDF."""
    configured = _configured_ocr_limit("PAGEGLANCE_MAX_OCR_PAGES", MAX_OCR_PAGES, MAX_PDF_PAGES)
    return min(configured, MAX_PDF_PAGES)


def get_max_ocr_seconds() -> int:
    """Return the bounded wall-time budget for OCR in one PDF."""
    return _configured_ocr_limit(
        "PAGEGLANCE_MAX_OCR_SECONDS", MAX_OCR_TOTAL_SECONDS, MAX_CONFIGURED_OCR_SECONDS
    )


def _upload_size(content: bytes | BinaryIO) -> int:
    if isinstance(content, bytes):
        return len(content)
    position = content.tell()
    content.seek(0, os.SEEK_END)
    size = content.tell()
    content.seek(position)
    return size


def _read_upload_range(content: bytes | BinaryIO, start: int, length: int) -> bytes:
    if isinstance(content, bytes):
        return content[start : start + length]
    position = content.tell()
    content.seek(start)
    value = content.read(length)
    content.seek(position)
    return value


def _copy_upload_to_temp(content: bytes | BinaryIO) -> str:
    """Copy a PDF to a temporary file in bounded blocks for PyMuPDF OCR rendering."""
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            prefix="pageglance-ocr-", suffix=".pdf", delete=False
        ) as out:
            temp_path = out.name
            if isinstance(content, bytes):
                source: BinaryIO = io.BytesIO(content)
            else:
                source = content
            original_position = source.tell()
            source.seek(0)
            while block := source.read(1024 * 1024):
                out.write(block)
            source.seek(original_position)
        return temp_path
    except Exception:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        raise


class UploadValidationError(ValueError):
    """Safe client-side upload rejection with an HTTP status and stable code."""

    def __init__(self, code: str, detail: str, status_code: int = 422) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.status_code = status_code


class UploadProcessingError(RuntimeError):
    """Safe server-side persistence/provider failure."""

    def __init__(
        self,
        code: str = "document_ingestion_unavailable",
        detail: str = "Document ingestion is temporarily unavailable. Please try again.",
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.code = code


@dataclass(frozen=True)
class ExtractedUpload:
    """Normalized source text and bounded metadata extracted from an upload."""

    filename: str
    source_type: DocumentSourceType
    media_type: str
    text: str
    metadata: dict[str, object]
    page_ranges: tuple[tuple[int, int, int], ...] = ()
    structured_blocks: tuple[BlockReference, ...] = ()


@dataclass(frozen=True)
class UploadedDocument:
    """Persisted upload details returned by the HTTP layer."""

    document: Document
    chunk_count: int
    duplicate: bool
    page_count: int | None


def _safe_filename(value: str | None) -> str:
    """Remove path components and control characters without trusting client paths."""
    basename = (value or "").replace("\\", "/").rsplit("/", 1)[-1]
    safe = "".join(character for character in basename if ord(character) >= 32).strip()
    if not safe or len(safe) > MAX_FILENAME_CHARS:
        raise UploadValidationError("invalid_filename", "The uploaded filename is invalid.")
    return safe


def _check_text_size(text: str) -> None:
    if len(text) > MAX_EXTRACTED_TEXT_CHARS:
        raise UploadValidationError(
            "extracted_text_too_large",
            f"Extracted text must be at most {MAX_EXTRACTED_TEXT_CHARS} characters.",
            413,
        )


def _has_meaningful_pdf_text(text: str) -> bool:
    """Distinguish usable selectable text from whitespace/parser artifacts."""
    return sum(character.isalnum() for character in text) >= 3 and any(
        character.isalpha() for character in text
    )


def _load_ocr_tools():
    """Load local OCR dependencies only when a PDF page needs OCR."""
    try:
        import fitz
    except ImportError:
        raise UploadProcessingError(
            "pdf_renderer_unavailable",
            "Scanned PDF processing is unavailable because PyMuPDF is not installed.",
        ) from None
    try:
        import pytesseract
    except ImportError:
        raise UploadProcessingError(
            "pytesseract_unavailable",
            "Scanned PDF processing is unavailable because pytesseract is not installed.",
        ) from None

    configured_command = os.getenv("PAGEGLANCE_TESSERACT_CMD", "").strip()
    pytesseract.pytesseract.tesseract_cmd = configured_command or DEFAULT_TESSERACT_COMMAND
    return fitz, pytesseract


def _ocr_pdf_page(pdf_page, fitz, pytesseract, timeout_seconds: float) -> str:
    """Render one bounded-size PDF page and recognize text using local Tesseract."""
    try:
        from PIL import Image

        rect = pdf_page.rect
        area_points = max(float(rect.width) * float(rect.height), 1.0)
        scale = min(MAX_OCR_DPI / 72, math.sqrt(MAX_OCR_PIXELS_PER_PAGE / area_points))
        pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False, annots=False)
        if pixmap.width * pixmap.height > MAX_OCR_PIXELS_PER_PAGE:
            raise UploadValidationError(
                "pdf_ocr_page_too_large", "A PDF page exceeds the safe OCR image limit.", 413
            )
        with Image.open(io.BytesIO(pixmap.tobytes("png"))) as image:
            return pytesseract.image_to_string(image, timeout=timeout_seconds)
    except UploadValidationError:
        raise
    except ImportError:
        raise UploadProcessingError(
            "ocr_engine_unavailable",
            "Scanned PDF processing is unavailable because Pillow is not installed.",
        ) from None
    except getattr(pytesseract, "TesseractNotFoundError", ()):
        raise UploadProcessingError(
            "tesseract_executable_unavailable",
            "Tesseract OCR could not be started. Set PAGEGLANCE_TESSERACT_CMD to the "
            "executable path or ensure Tesseract is on PATH.",
        ) from None
    except subprocess.TimeoutExpired:
        raise UploadProcessingError(
            "ocr_timeout", "Local OCR took too long to process this PDF. Try a smaller file."
        ) from None
    except Exception:
        # Renderer/OCR exception details can include local paths or document data.
        raise UploadProcessingError(
            "pdf_ocr_failed", "Local OCR could not read a scanned PDF page."
        ) from None


def _docx_paragraph_text(paragraph: ET.Element) -> str:
    """Concatenate the text runs of one OOXML paragraph in document order."""
    return "".join(
        run.text for run in paragraph.iter(f"{{{DOCX_NS}}}t") if run.text
    )


def _docx_style_val(paragraph: ET.Element) -> str | None:
    """Return the paragraph style identifier (e.g. Heading1) if present."""
    properties = paragraph.find(f"{{{DOCX_NS}}}pPr")
    if properties is None:
        return None
    style = properties.find(f"{{{DOCX_NS}}}pStyle")
    if style is None:
        return None
    return style.get(f"{{{DOCX_NS}}}val")


def _docx_block_type(paragraph: ET.Element) -> str:
    """Map an OOXML paragraph to a structured block type."""
    style = (_docx_style_val(paragraph) or "").lower()
    if style.startswith("heading") or style.startswith("title"):
        return "heading"
    return "paragraph"


def _docx_cell_text(cell: ET.Element) -> str:
    """Extract a table cell's text from its constituent paragraphs."""
    parts = [
        _docx_paragraph_text(paragraph).strip()
        for paragraph in cell.findall(f"{{{DOCX_NS}}}p")
    ]
    return " ".join(part for part in parts if part)


def _docx_table_blocks(table: ET.Element) -> list[tuple[str, str]]:
    """Render a table as one structured block with readable row/cell order."""
    rows: list[str] = []
    for row in table.findall(f"{{{DOCX_NS}}}tr"):
        cells = [
            _docx_cell_text(cell)
            for cell in row.findall(f"{{{DOCX_NS}}}tc")
        ]
        row_text = " | ".join(cell for cell in cells if cell).strip()
        if row_text:
            rows.append(row_text)
    if not rows:
        return []
    return [("table", "\n".join(rows))]


def _docx_body_pieces(body: ET.Element) -> list[tuple[str, str]]:
    """Flatten a document body into ordered (block_type, text) pieces."""
    pieces: list[tuple[str, str]] = []
    for child in body:
        if child.tag == f"{{{DOCX_NS}}}p":
            text = _docx_paragraph_text(child).strip()
            if text:
                pieces.append((_docx_block_type(child), text))
        elif child.tag == f"{{{DOCX_NS}}}tbl":
            pieces.extend(_docx_table_blocks(child))
    return pieces


def _extract_docx(
    safe_name: str, content: bytes | BinaryIO, file_size: int
) -> ExtractedUpload:
    """Parse a DOCX (OOXML ZIP) into normalized text with structured blocks."""
    try:
        if isinstance(content, bytes):
            raw = content
        else:
            content.seek(0)
            raw = content.read(file_size)
    except Exception:
        raise UploadValidationError(
            "malformed_docx", "The DOCX file could not be read."
        ) from None

    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            if "word/document.xml" not in archive.namelist():
                raise UploadValidationError(
                    "malformed_docx",
                    "The DOCX file is missing its main document part.",
                )
            document_xml = archive.read("word/document.xml")
    except UploadValidationError:
        raise
    except (zipfile.BadZipFile, KeyError, OSError):
        raise UploadValidationError(
            "malformed_docx", "The DOCX file is not a valid compressed zip archive."
        ) from None

    try:
        document = ET.fromstring(document_xml)
    except ET.ParseError:
        raise UploadValidationError(
            "malformed_docx", "The DOCX document markup is malformed."
        ) from None

    body = document.find(f"{{{DOCX_NS}}}body")
    if body is None:
        raise UploadValidationError(
            "malformed_docx", "The DOCX file has no document body."
        )

    pieces = _docx_body_pieces(body)
    pieces = [(kind, normalize_text(text)) for kind, text in pieces]
    pieces = [(kind, text) for kind, text in pieces if text]
    if not pieces:
        raise UploadValidationError(
            "empty_docx", "No readable text was found in the DOCX file."
        )

    blocks: list[BlockReference] = []
    parts: list[str] = []
    cursor = 0
    for kind, text in pieces:
        if parts:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(text)
        cursor += len(text)
        blocks.append(
            BlockReference(block_type=kind, text=text, start_char=start, end_char=cursor)
        )

    combined = normalize_text("".join(parts))
    _check_text_size(combined)

    return ExtractedUpload(
        filename=safe_name,
        source_type="text",
        media_type=DOCX_MIME_TYPE,
        text=combined,
        metadata={
            "upload_filename": safe_name,
            "size_bytes": file_size,
            "format": "docx",
        },
        structured_blocks=tuple(blocks),
    )


def _html_has_block_ancestor(stack: tuple[str, ...]) -> bool:
    """Return whether the current tag is inside a block-level container."""
    return any(tag in HTML_BLOCK_TAGS for tag in stack)


def _html_is_boilerplate(tag: str, attrs: list[tuple[str, str]]) -> bool:
    """Detect obvious advertising/cookie/consent boilerplate from id/class."""
    if tag in {"nav", "aside"}:
        return tag == "aside"
    for key, value in attrs:
        if key in {"id", "class"} and HTML_BOILERPLATE_RE.search(value or ""):
            return True
    return False


def _html_is_hidden(tag: str, attrs: list[tuple[str, str]]) -> bool:
    """Detect elements hidden via aria-hidden, hidden attr, or CSS."""
    for key, value in attrs:
        if key == "aria-hidden" and value and value.lower() == "true":
            return True
        if key == "hidden":
            return True
        if key == "style" and value and HTML_HIDDEN_RE.search(value):
            return True
    return False


class _DocumentHtmlParser(HTMLParser):
    """Parse HTML into ordered (block_type, text) pieces with titles."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.pieces: list[tuple[str, str]] = []
        self.title = ""
        self._in_title = False
        self._skip_depth = 0
        self._hidden_depth = 0
        self._stack: list[str] = []
        self._paragraph_parts: list[str] = []
        self._paragraph_type = "paragraph"
        self._in_pre = False
        self._in_cell = False
        self._cell_parts: list[str] = []
        self._row_cells: list[str] = []
        self._table_rows: list[list[str]] = []
        self._table_depth = 0
        self._hidden_stack: list[bool] = []

    def _flush_paragraph(self) -> None:
        if not self._paragraph_parts:
            return
        text = "".join(self._paragraph_parts)
        text = normalize_text(text)
        if text:
            self.pieces.append((self._paragraph_type, text))
        self._paragraph_parts = []
        self._paragraph_type = "paragraph"

    def _flush_table(self) -> None:
        if not self._table_rows:
            return
        rows = [" | ".join(cells) for cells in self._table_rows if cells]
        rows = [row.strip() for row in rows if row.strip()]
        if rows:
            self.pieces.append(("table", normalize_text("\n".join(rows))))
        self._table_rows = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str]]) -> None:
        if tag == "title":
            self._in_title = True
            return
        if tag in HTML_VOID_TAGS:
            # Void elements have no end tag; never disturb depth counters.
            if tag == "br":
                self._flush_paragraph()
            return
        if tag in HTML_SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        hidden = _html_is_hidden(tag, attrs) or _html_is_boilerplate(tag, attrs)
        self._hidden_stack.append(hidden)
        if hidden:
            self._hidden_depth += 1
            return
        if self._hidden_depth:
            return

        self._stack.append(tag)

        if tag in HTML_TABLE_TAGS:
            if tag == "table":
                self._table_depth += 1
                self._flush_paragraph()
            elif tag == "tr":
                self._row_cells = []
            elif tag in {"td", "th"}:
                self._in_cell = True
                self._cell_parts = []
            return

        if tag in HTML_HEADING_TAGS:
            self._flush_paragraph()
            self._paragraph_type = "heading"
            return
        if tag == "li":
            self._flush_paragraph()
            self._paragraph_type = "list_item"
            return
        if tag == "pre":
            self._flush_paragraph()
            self._in_pre = True
            self._paragraph_type = "code"
            return
        if tag in {"p", "blockquote"} or tag == "br":
            self._flush_paragraph()
            return
        if tag in HTML_LIST_TAGS:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            return

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            return
        if tag in HTML_SKIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if self._hidden_stack:
            was_hidden = self._hidden_stack.pop()
            if was_hidden and self._hidden_depth:
                self._hidden_depth -= 1
            if was_hidden:
                return
        if self._hidden_depth:
            return

        if tag in HTML_TABLE_TAGS:
            if tag in {"td", "th"} and self._in_cell:
                cell = normalize_text("".join(self._cell_parts))
                if cell:
                    self._row_cells.append(cell)
                self._in_cell = False
                self._cell_parts = []
            elif tag == "tr":
                if self._row_cells:
                    self._table_rows.append(self._row_cells)
                self._row_cells = []
            elif tag == "table":
                self._flush_table()
            return

        if tag == "pre":
            self._flush_paragraph()
            self._in_pre = False
            self._paragraph_type = "paragraph"
            return
        if tag in HTML_HEADING_TAGS or tag == "li" or tag in {"p", "blockquote"}:
            self._flush_paragraph()
            self._paragraph_type = "paragraph"
            return

        if self._stack and self._stack[-1] == tag:
            self._stack.pop()
            if tag in HTML_BLOCK_TAGS and tag not in HTML_TABLE_TAGS:
                # Stay in the same paragraph unless a new block opens.
                pass

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if self._skip_depth or self._hidden_depth:
            return
        if self._in_cell and self._table_depth:
            self._cell_parts.append(data)
            return
        if self._in_pre:
            self._paragraph_parts.append(data)
            return
        self._paragraph_parts.append(data)


def _extract_html(safe_name: str, content: bytes | BinaryIO, file_size: int) -> ExtractedUpload:
    """Parse an HTML document into normalized text with structured blocks."""
    try:
        if isinstance(content, bytes):
            raw = content
        else:
            content.seek(0)
            raw = content.read(file_size)
    except Exception:
        raise UploadValidationError(
            "malformed_html", "The HTML file could not be read."
        ) from None

    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        raise UploadValidationError(
            "malformed_html", "The HTML file could not be decoded."
        ) from None

    parser = _DocumentHtmlParser()
    try:
        parser.feed(text)
        parser.close()
    except Exception:
        raise UploadValidationError(
            "malformed_html", "The HTML document is malformed."
        ) from None

    parser._flush_paragraph()
    parser._flush_table()
    pieces = parser.pieces
    if not pieces:
        raise UploadValidationError(
            "empty_html", "No readable text was found in the HTML file."
        )

    blocks: list[BlockReference] = []
    parts: list[str] = []
    cursor = 0
    for kind, text in pieces:
        if parts:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(text)
        cursor += len(text)
        blocks.append(
            BlockReference(block_type=kind, text=text, start_char=start, end_char=cursor)
        )

    combined = normalize_text("".join(parts))
    _check_text_size(combined)

    title = normalize_text(parser.title)[:HTML_TITLE_MAX]
    metadata: dict[str, object] = {
        "upload_filename": safe_name,
        "size_bytes": file_size,
        "format": "html",
    }
    if title:
        metadata["title"] = title

    return ExtractedUpload(
        filename=safe_name,
        source_type="text",
        media_type=HTML_MIME_TYPE,
        text=combined,
        metadata=metadata,
        structured_blocks=tuple(blocks),
    )


MARKDOWN_FENCE_RE = re.compile(r"^\s*(```|~~~)\s*[\w+.-]*\s*$")
MARKDOWN_HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")
MARKDOWN_UL_RE = re.compile(r"^\s*[-*+]\s+(.+?)\s*$")
MARKDOWN_OL_RE = re.compile(r"^\s*\d+\.\s+(.+?)\s*$")
MARKDOWN_QUOTE_RE = re.compile(r"^\s*>\s?(.+?)\s*$")
MARKDOWN_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*(?::?-{3,}\s*\|?\s*)+$")
MARKDOWN_FRONT_MATTER_RE = re.compile(r"^\s*([A-Za-z_-]+)\s*:\s*(.+?)\s*$")


def _markdown_inline(text: str) -> str:
    """Strip common inline Markdown syntax markers from a span of text."""
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text.strip()


def _markdown_table_cells(line: str) -> list[str]:
    """Split one pipe-table row into trimmed cell values."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _markdown_table_blocks(header_line: str, rows: list[str]) -> list[tuple[str, str]]:
    """Render a pipe table as one structured block in row/cell order."""
    table_rows = [_markdown_table_cells(header_line)]
    table_rows.extend(_markdown_table_cells(row) for row in rows)
    if not any(any(cell for cell in row) for row in table_rows):
        return []
    return [("table", "\n".join(" | ".join(row) for row in table_rows))]


def _markdown_body_pieces(
    text: str,
) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Flatten Markdown text into ordered (block_type, text) pieces.

    Returns the pieces plus a metadata dict parsed conservatively from optional
    YAML front matter (front matter is never silently discarded).
    """
    pieces: list[tuple[str, str]] = []
    metadata: dict[str, str] = {}
    lines = text.splitlines()

    if lines and lines[0].strip() == "---":
        end: int | None = None
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                end = index
                break
        if end is not None:
            front_matter = lines[1:end]
            lines = lines[end + 1 :]
            fm_text = "\n".join(front_matter)
            for line in front_matter:
                match = MARKDOWN_FRONT_MATTER_RE.match(line)
                if match and match.group(1).lower() == "title":
                    metadata["title"] = match.group(2).strip()
            if fm_text.strip():
                pieces.append(("paragraph", normalize_text(fm_text)))

    index = 0
    count = len(lines)
    while index < count:
        raw = lines[index]
        if not raw.strip():
            index += 1
            continue

        fence_match = MARKDOWN_FENCE_RE.match(raw)
        if fence_match:
            code_lines: list[str] = []
            index += 1
            while index < count and not MARKDOWN_FENCE_RE.match(lines[index]):
                code_lines.append(lines[index])
                index += 1
            if index < count:
                index += 1  # consume the closing fence
            code_text = "\n".join(code_lines)
            if code_text.strip():
                pieces.append(("code", code_text))
            continue

        heading_match = MARKDOWN_HEADING_RE.match(raw)
        if heading_match:
            pieces.append(
                ("heading", normalize_text(_markdown_inline(heading_match.group(2))))
            )
            index += 1
            continue

        if "|" in raw and index + 1 < count and MARKDOWN_TABLE_SEP_RE.match(
            lines[index + 1]
        ):
            header_line = raw
            cursor = index + 2
            rows: list[str] = []
            while (
                cursor < count
                and lines[cursor].strip()
                and "|" in lines[cursor]
                and not MARKDOWN_TABLE_SEP_RE.match(lines[cursor])
            ):
                rows.append(lines[cursor])
                cursor += 1
            pieces.extend(_markdown_table_blocks(header_line, rows))
            index = cursor
            continue

        quote_match = MARKDOWN_QUOTE_RE.match(raw)
        if quote_match:
            quote_lines: list[str] = []
            while index < count:
                qi = MARKDOWN_QUOTE_RE.match(lines[index])
                if qi is None:
                    break
                quote_lines.append(qi.group(1))
                index += 1
            quote_text = " ".join(_markdown_inline(line) for line in quote_lines)
            if quote_text.strip():
                pieces.append(("paragraph", normalize_text(quote_text)))
            continue

        unordered = MARKDOWN_UL_RE.match(raw)
        ordered = MARKDOWN_OL_RE.match(raw)
        if unordered or ordered:
            while index < count:
                item_ul = MARKDOWN_UL_RE.match(lines[index])
                item_ol = MARKDOWN_OL_RE.match(lines[index])
                if item_ul is None and item_ol is None:
                    break
                item_text = (item_ul or item_ol).group(1)
                if item_text.strip():
                    pieces.append(
                        ("list_item", normalize_text(_markdown_inline(item_text)))
                    )
                index += 1
            continue

        paragraph_lines: list[str] = []
        while index < count:
            line = lines[index]
            if not line.strip():
                break
            if (
                MARKDOWN_HEADING_RE.match(line)
                or MARKDOWN_FENCE_RE.match(line)
                or MARKDOWN_UL_RE.match(line)
                or MARKDOWN_OL_RE.match(line)
                or MARKDOWN_QUOTE_RE.match(line)
            ):
                break
            if "|" in line and index + 1 < count and MARKDOWN_TABLE_SEP_RE.match(
                lines[index + 1]
            ):
                break
            paragraph_lines.append(_markdown_inline(line))
            index += 1
        paragraph_text = " ".join(part for part in paragraph_lines if part.strip())
        if paragraph_text.strip():
            pieces.append(("paragraph", normalize_text(paragraph_text)))

    return pieces, metadata


def _extract_markdown(
    safe_name: str, content: bytes | BinaryIO, file_size: int
) -> ExtractedUpload:
    """Parse Markdown (`.md`/`.markdown`) into normalized text with structured blocks."""
    try:
        if isinstance(content, bytes):
            raw = content
        else:
            content.seek(0)
            raw = content.read(file_size)
    except Exception:
        raise UploadValidationError(
            "malformed_markdown", "The Markdown file could not be read."
        ) from None

    try:
        text = raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError:
        raise UploadValidationError(
            "invalid_text_encoding", "Markdown uploads must use UTF-8 encoding."
        ) from None

    pieces, front_matter = _markdown_body_pieces(text)
    if not pieces:
        raise UploadValidationError(
            "empty_markdown", "No readable text was found in the Markdown file."
        )

    blocks: list[BlockReference] = []
    parts: list[str] = []
    cursor = 0
    for kind, piece in pieces:
        if parts:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(piece)
        cursor += len(piece)
        blocks.append(
            BlockReference(
                block_type=kind, text=piece, start_char=start, end_char=cursor
            )
        )

    combined = normalize_text("".join(parts))
    if not combined:
        raise UploadValidationError(
            "empty_markdown", "No readable text was found in the Markdown file."
        )
    _check_text_size(combined)

    markdown_metadata: dict[str, object] = {
        "upload_filename": safe_name,
        "size_bytes": file_size,
        "format": "markdown",
    }
    markdown_metadata.update(front_matter)
    return ExtractedUpload(
        filename=safe_name,
        source_type="markdown",
        media_type="text/markdown",
        text=combined,
        metadata=markdown_metadata,
        structured_blocks=tuple(blocks),
    )


def _extract_csv(
    safe_name: str, content: bytes | BinaryIO, file_size: int
) -> ExtractedUpload:
    """Parse a CSV file into normalized text with structured table blocks.

    Uses the standard-library :mod:`csv` reader, which handles quoted fields,
    commas inside quoted values, escaped quotes and newlines inside quoted
    fields. Rows are rendered as bounded ``table`` blocks that preserve the
    header record, row boundaries and cell-to-column alignment.
    """
    try:
        if isinstance(content, bytes):
            raw = content
        else:
            content.seek(0)
            raw = content.read(file_size)
    except Exception:
        raise UploadValidationError(
            "malformed_csv", "The CSV file could not be read."
        ) from None

    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise UploadValidationError(
            "invalid_text_encoding", "CSV uploads must use UTF-8 encoding."
        ) from None

    rows: list[list[str]] = []
    try:
        for record in csv.reader(io.StringIO(text)):
            cells = [re.sub(r"\s+", " ", cell).strip() for cell in record]
            if len(cells) > MAX_CSV_COLUMNS:
                raise UploadValidationError(
                    "csv_column_limit_exceeded",
                    f"CSV files must have at most {MAX_CSV_COLUMNS} columns.",
                    413,
                )
            rows.append(cells)
            if len(rows) > MAX_CSV_ROWS:
                raise UploadValidationError(
                    "csv_row_limit_exceeded",
                    f"CSV files must have at most {MAX_CSV_ROWS} rows.",
                    413,
                )
    except UploadValidationError:
        raise
    except (csv.Error, StopIteration, ValueError):
        raise UploadValidationError(
            "malformed_csv", "The CSV file is malformed."
        ) from None

    if not any(any(cell for cell in row) for row in rows):
        raise UploadValidationError(
            "empty_csv", "No readable data was found in the CSV file."
        )

    # Deterministic column alignment: pad every row to the widest row so no
    # cell is silently dropped or misaligned, even for inconsistent widths.
    max_cols = max((len(row) for row in rows), default=0)
    aligned = [row + [""] * (max_cols - len(row)) for row in rows]
    header = aligned[0]

    blocks: list[BlockReference] = []
    parts: list[str] = []
    cursor = 0
    for start in range(0, len(aligned), MAX_CSV_TABLE_ROWS):
        chunk = aligned[start : start + MAX_CSV_TABLE_ROWS]
        lines = [" | ".join(row) for row in ((header,) + tuple(chunk))]
        piece = normalize_text("\n".join(lines))
        if not piece:
            continue
        if parts:
            parts.append("\n\n")
            cursor += 2
        block_start = cursor
        parts.append(piece)
        cursor += len(piece)
        blocks.append(
            BlockReference(
                block_type="table",
                text=piece,
                start_char=block_start,
                end_char=cursor,
            )
        )

    combined = normalize_text("".join(parts))
    if not combined:
        raise UploadValidationError(
            "empty_csv", "No readable data was found in the CSV file."
        )
    _check_text_size(combined)

    return ExtractedUpload(
        filename=safe_name,
        source_type="text",
        media_type="text/csv",
        text=combined,
        metadata={
            "upload_filename": safe_name,
            "size_bytes": file_size,
            "format": "csv",
        },
        structured_blocks=tuple(blocks),
    )


def _json_render(value: object, path: str, out: list[str], state: dict[str, int]) -> None:
    """Flatten a JSON value into deterministic ``path: value`` lines.

    ``state["nodes"]`` bounds the total number of emitted scalar values and
    ``state["depth"]`` tracks recursion depth so pathological structures fail
    with a controlled error instead of exhausting memory or the call stack.
    """
    if state["depth"] > MAX_JSON_DEPTH:
        raise UploadValidationError(
            "json_depth_limit_exceeded",
            f"JSON nesting must be at most {MAX_JSON_DEPTH} levels.",
            413,
        )
    state["depth"] += 1
    try:
        if isinstance(value, dict):
            for key, val in value.items():
                child = f"{path}.{key}" if path else str(key)
                _json_render(val, child, out, state)
            return
        if isinstance(value, list):
            for index, val in enumerate(value):
                child = f"{path}[{index}]" if path else f"[{index}]"
                _json_render(val, child, out, state)
            return
        if value is None:
            rendered = "null"
        elif value is True:
            rendered = "true"
        elif value is False:
            rendered = "false"
        elif isinstance(value, str):
            rendered = value
        elif isinstance(value, (int, float)):
            rendered = repr(value)
        else:
            rendered = str(value)
        state["nodes"] += 1
        if state["nodes"] > MAX_JSON_NODES:
            raise UploadValidationError(
                "json_size_limit_exceeded",
                f"JSON files must have at most {MAX_JSON_NODES} values.",
                413,
            )
        cleaned = re.sub(r"\s+", " ", rendered).strip()
        out.append(f"{path}: {cleaned}")
    finally:
        state["depth"] -= 1


def _extract_json(
    safe_name: str, content: bytes | BinaryIO, file_size: int
) -> ExtractedUpload:
    """Parse a JSON file into deterministic, readable path-based lines."""
    try:
        if isinstance(content, bytes):
            raw = content
        else:
            content.seek(0)
            raw = content.read(file_size)
    except Exception:
        raise UploadValidationError(
            "malformed_json", "The JSON file could not be read."
        ) from None

    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise UploadValidationError(
            "invalid_text_encoding", "JSON uploads must use UTF-8 encoding."
        ) from None

    if not text.strip():
        raise UploadValidationError(
            "malformed_json", "The JSON file is empty."
        )
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, ValueError, RecursionError):
        raise UploadValidationError(
            "malformed_json", "The JSON file is malformed or too deeply nested."
        ) from None

    lines: list[str] = []
    state: dict[str, int] = {"nodes": 0, "depth": 0}
    _json_render(value, "", lines, state)
    if not lines:
        raise UploadValidationError(
            "empty_json", "No readable values were found in the JSON file."
        )

    blocks: list[BlockReference] = []
    parts: list[str] = []
    cursor = 0
    for start in range(0, len(lines), MAX_JSON_LINES_PER_BLOCK):
        piece = normalize_text("\n".join(lines[start : start + MAX_JSON_LINES_PER_BLOCK]))
        if not piece:
            continue
        if parts:
            parts.append("\n\n")
            cursor += 2
        block_start = cursor
        parts.append(piece)
        cursor += len(piece)
        blocks.append(
            BlockReference(
                block_type="json",
                text=piece,
                start_char=block_start,
                end_char=cursor,
            )
        )

    combined = normalize_text("".join(parts))
    _check_text_size(combined)

    return ExtractedUpload(
        filename=safe_name,
        source_type="text",
        media_type="application/json",
        text=combined,
        metadata={
            "upload_filename": safe_name,
            "size_bytes": file_size,
            "format": "json",
        },
        structured_blocks=tuple(blocks),
    )


def _xml_local(tag: str) -> str:
    """Return the local-name part of an ElementTree tag, stripping any namespace URI."""
    return tag.rsplit("}", 1)[-1] if tag.startswith("{") else tag


def _xml_render(
    elem: ET.Element,
    path: str,
    out: list[str],
    state: dict[str, int],
) -> None:
    """Flatten an XML element into deterministic ``path`` lines.

    Repeated siblings get array-like indexes (``book[0]``), attributes are
    addressed as ``@name``, and meaningful mixed text / tail text is preserved
    in document order. ``state["depth"]`` bounds recursion and
    ``state["nodes"]`` bounds the total emitted output.
    """
    if state["depth"] > MAX_XML_DEPTH:
        raise UploadValidationError(
            "xml_depth_limit_exceeded",
            f"XML nesting must be at most {MAX_XML_DEPTH} levels.",
            413,
        )
    state["depth"] += 1
    try:
        for index, (attr_name, attr_value) in enumerate(elem.attrib.items()):
            if index >= MAX_XML_ATTRS_PER_ELEMENT:
                raise UploadValidationError(
                    "xml_attribute_limit_exceeded",
                    "XML elements must have a bounded number of attributes.",
                    413,
                )
            state["nodes"] += 1
            if state["nodes"] > MAX_XML_NODES:
                raise UploadValidationError(
                    "xml_size_limit_exceeded",
                    f"XML files must have at most {MAX_XML_NODES} values.",
                    413,
                )
            cleaned = re.sub(r"\s+", " ", attr_value).strip()
            out.append(f"{path}.@{_xml_local(attr_name)}: {cleaned}")

        text = elem.text or ""
        if text.strip():
            state["nodes"] += 1
            if state["nodes"] > MAX_XML_NODES:
                raise UploadValidationError(
                    "xml_size_limit_exceeded",
                    f"XML files must have at most {MAX_XML_NODES} values.",
                    413,
                )
            cleaned = re.sub(r"\s+", " ", text).strip()
            out.append(f"{path}: {cleaned}")

        children = list(elem)
        seen: dict[str, int] = {}
        for child in children:
            child_local = _xml_local(child.tag)
            index = seen.get(child_local, 0)
            seen[child_local] = index + 1
            child_path = f"{path}.{child_local}"
            if len([c for c in children if _xml_local(c.tag) == child_local]) > 1:
                child_path = f"{child_path}[{index}]"
            _xml_render(child, child_path, out, state)
            tail = child.tail or ""
            if tail.strip():
                state["nodes"] += 1
                if state["nodes"] > MAX_XML_NODES:
                    raise UploadValidationError(
                        "xml_size_limit_exceeded",
                        f"XML files must have at most {MAX_XML_NODES} values.",
                        413,
                    )
                cleaned = re.sub(r"\s+", " ", tail).strip()
                out.append(f"{path}: {cleaned}")
    finally:
        state["depth"] -= 1


def _extract_xml(
    safe_name: str, content: bytes | BinaryIO, file_size: int
) -> ExtractedUpload:
    """Parse an XML file into deterministic, bounded ``path: value`` lines.

    Uses the standard-library :class:`xml.etree.ElementTree`. DTD / entity
    declarations are rejected up front so neither internal entity expansion
    ("billion laughs") nor external entity resolution can run.
    """
    try:
        if isinstance(content, bytes):
            raw = content
        else:
            content.seek(0)
            raw = content.read(file_size)
    except Exception:
        raise UploadValidationError(
            "malformed_xml", "The XML file could not be read."
        ) from None

    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise UploadValidationError(
            "invalid_text_encoding", "XML uploads must use UTF-8 encoding."
        ) from None

    if not text.strip():
        raise UploadValidationError("empty_xml", "The XML file is empty.")

    if re.search(r"(?i)<!DOCTYPE|<!ENTITY", text):
        raise UploadValidationError(
            "xml_dtd_blocked",
            "XML files with DTD or entity declarations are not supported.",
            415,
        )

    try:
        root = ET.fromstring(text)
    except (ET.ParseError, RecursionError, ValueError):
        raise UploadValidationError(
            "malformed_xml", "The XML file is malformed or too deeply nested."
        ) from None

    lines: list[str] = []
    state: dict[str, int] = {"nodes": 0, "depth": 0}
    _xml_render(root, _xml_local(root.tag), lines, state)
    if not lines:
        raise UploadValidationError(
            "empty_xml", "No readable content was found in the XML file."
        )

    blocks: list[BlockReference] = []
    parts: list[str] = []
    cursor = 0
    for start in range(0, len(lines), MAX_XML_LINES_PER_BLOCK):
        piece = normalize_text("\n".join(lines[start : start + MAX_XML_LINES_PER_BLOCK]))
        if not piece:
            continue
        if parts:
            parts.append("\n\n")
            cursor += 2
        block_start = cursor
        parts.append(piece)
        cursor += len(piece)
        blocks.append(
            BlockReference(
                block_type="xml",
                text=piece,
                start_char=block_start,
                end_char=cursor,
            )
        )

    combined = normalize_text("".join(parts))
    _check_text_size(combined)

    return ExtractedUpload(
        filename=safe_name,
        source_type="text",
        media_type="application/xml",
        text=combined,
        metadata={
            "upload_filename": safe_name,
            "size_bytes": file_size,
            "format": "xml",
        },
        structured_blocks=tuple(blocks),
    )


def _load_pytesseract():
    """Load only pytesseract (used by image OCR) with the shared Tesseract config."""
    try:
        import pytesseract
    except ImportError:
        raise UploadProcessingError(
            "pytesseract_unavailable",
            "Image OCR requires the pytesseract Python package. Install backend dependencies.",
        ) from None
    configured_command = os.getenv("PAGEGLANCE_TESSERACT_CMD", "").strip()
    pytesseract.pytesseract.tesseract_cmd = configured_command or DEFAULT_TESSERACT_COMMAND
    return pytesseract


def _ocr_image(image, pytesseract, timeout_seconds: float) -> str:
    """Recognize text from a decoded image using local Tesseract."""
    lang = "+".join(_resolve_ocr_languages())
    try:
        return pytesseract.image_to_string(image, lang=lang, timeout=timeout_seconds)
    except getattr(pytesseract, "TesseractNotFoundError", ()):
        raise UploadProcessingError(
            "tesseract_executable_unavailable",
            "Tesseract OCR could not be started. Set PAGEGLANCE_TESSERACT_CMD to the "
            "executable path or ensure Tesseract is on PATH.",
        ) from None
    except getattr(pytesseract, "TesseractError", ()) as error:
        if "language" in str(error).lower():
            raise UploadProcessingError(
                "ocr_language_unavailable",
                f"Tesseract language data is unavailable for '{lang}'. Install the matching "
                "traineddata files or set PAGEGLANCE_TESSERACT_LANGUAGES.",
            ) from None
        raise UploadProcessingError(
            "image_ocr_failed", "Local OCR could not read this image."
        ) from None
    except subprocess.TimeoutExpired:
        raise UploadProcessingError(
            "ocr_timeout", "Local OCR took too long to process this image. Try a smaller file."
        ) from None
    except Exception:
        raise UploadProcessingError(
            "image_ocr_failed", "Local OCR could not read this image."
        ) from None


def _image_ocr_candidates(image):
    """Build a bounded set of preprocessing variants for OCR.

    The original normalized image is always the first (fallback) candidate.
    Remaining variants are mild, bounded transformations that preserve detail
    and stay within the configured pixel/dimension limits. EXIF orientation is
    applied by the caller before this function runs.
    """
    from PIL import Image, ImageEnhance

    variants = [image.convert("RGB")]  # original normalized fallback

    # Upscale small images within strict pixel/dimension limits first (most
    # impactful for low-resolution inputs) so it is not lost to the attempt cap.
    width, height = image.size
    min_side = min(width, height)
    if min_side < MAX_IMAGE_MIN_SIDE_UPSCALE:
        scale = min(
            MAX_IMAGE_UPSCALE_FACTOR,
            MAX_IMAGE_MIN_SIDE_UPSCALE / float(min_side),
        )
        new_width = int(width * scale)
        new_height = int(height * scale)
        if (
            new_width <= MAX_IMAGE_SIDE_PIXELS
            and new_height <= MAX_IMAGE_SIDE_PIXELS
            and new_width * new_height <= MAX_IMAGE_MEGAPIXELS * 1_000_000
        ):
            variants.append(image.resize((new_width, new_height), Image.LANCZOS))

    grayscale = image.convert("L")
    variants.append(grayscale)

    # Mild contrast enhancement helps low-contrast scans without destroying detail.
    variants.append(ImageEnhance.Contrast(grayscale).enhance(1.8))

    # Binarization targets difficult scans; it is one candidate, not the only input.
    variants.append(grayscale.point(lambda value: 255 if value > 128 else 0))

    return variants[:MAX_IMAGE_OCR_ATTEMPTS]


def _ocr_text_score(text: str) -> tuple[int, int]:
    """Deterministic OCR-quality score; prefer meaningful alphanumeric content."""
    return (sum(character.isalnum() for character in text), len(text))


def _ocr_image_candidates(
    image, pytesseract, total_timeout_seconds: float
) -> str:
    """Run bounded OCR across preprocessing variants and return the best text.

    A single wall-clock deadline spans all attempts: the remaining time is
    computed before each variant and the loop stops once the budget is
    exhausted. Per-variant failures are caught so one bad variant does not
    discard the others; the first underlying error is re-raised only if every
    variant fails or the budget runs out without usable text.
    """
    first_error = None
    texts: list[str] = []
    deadline = time.monotonic() + total_timeout_seconds
    for candidate in _image_ocr_candidates(image):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            texts.append(_ocr_image(candidate, pytesseract, remaining))
        except UploadProcessingError as error:
            if first_error is None:
                first_error = error
            # A hard OCR timeout consumed the shared budget; stop rather than
            # starting a fresh expensive attempt.
            if error.code == "ocr_timeout":
                break
    if texts:
        return max(texts, key=_ocr_text_score)
    if first_error is not None:
        raise first_error
    raise UploadProcessingError(
        "image_ocr_failed", "Local OCR could not read this image."
    )


LANGUAGE_CODE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _resolve_ocr_languages() -> list[str]:
    """Resolve configured Tesseract language codes from the environment.

    Reads PAGEGLANCE_TESSERACT_LANGUAGES (default 'eng'). Codes may be separated
    by '+', commas, or whitespace. The configuration is validated (format plus a
    bounded number of languages); unavailable traineddata is detected at OCR time
    and reported as a controlled error rather than silently claiming support.
    """
    raw = os.getenv(
        "PAGEGLANCE_TESSERACT_LANGUAGES", DEFAULT_IMAGE_OCR_LANGUAGE
    ).strip()
    if not raw:
        return [DEFAULT_IMAGE_OCR_LANGUAGE]
    tokens = [token for token in re.split(r"[+\s,]+", raw) if token]
    if len(tokens) > MAX_IMAGE_OCR_LANGS:
        raise UploadProcessingError(
            "ocr_language_config_invalid",
            f"At most {MAX_IMAGE_OCR_LANGS} OCR languages may be configured via "
            "PAGEGLANCE_TESSERACT_LANGUAGES.",
        )
    for token in tokens:
        if not LANGUAGE_CODE_RE.match(token):
            raise UploadProcessingError(
                "ocr_language_config_invalid",
                f"Invalid OCR language code '{token}'. Use Tesseract codes such as "
                "'eng' or 'hin', e.g. 'eng+hin'.",
            )
    return tokens


def _ocr_image_layout(
    image, pytesseract, lang: str, timeout_seconds: float
) -> tuple[list[str], float, list[dict[str, object]]] | None:
    """Best-effort structured layout from Tesseract's ``image_to_data`` output.

    Returns ``(ordered paragraph pieces, mean word confidence, layout records)``
    preserving the reading order Tesseract reports (block/paragraph/line). Each
    layout record carries the paragraph's bounding box (``x``, ``y``,
    ``width``, ``height``), its mean word confidence, and ``block_index`` (the
    matching position in the returned pieces / structured blocks). Records are
    validated against the image dimensions and drop out-of-bounds or
    non-finite boxes, empty text, and paragraphs without a valid confidence.
    Returns ``None`` when structured output is unavailable, invalid, or yields
    no usable words, so the caller falls back to the plain-text paragraph split.
    No semantic labels are invented; multi-column reading order is inherited from
    Tesseract's grouping.
    """
    try:
        data = pytesseract.image_to_data(
            image,
            lang=lang,
            output_type=pytesseract.Output.DICT,
            timeout=timeout_seconds,
        )
    except Exception:
        # Structured OCR is best-effort: any failure (missing engine, timeout,
        # invalid output) falls back to the plain-text paragraph split.
        return None

    texts = data.get("text") or []
    if len(texts) == 0:
        return None

    img_width, img_height = image.size

    words: list[
        tuple[int, int, int, str, float, float, float, float, float]
    ] = []
    for index in range(len(texts)):
        word = (texts[index] or "").strip()
        if not word:
            continue
        try:
            confidence = float(data["conf"][index])
        except (KeyError, IndexError, TypeError, ValueError):
            confidence = -1.0
        try:
            block_num = int(data["block_num"][index])
            par_num = int(data["par_num"][index])
            line_num = int(data["line_num"][index])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        try:
            left = float(data["left"][index])
            top = float(data["top"][index])
            width = float(data["width"][index])
            height = float(data["height"][index])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        words.append(
            (block_num, par_num, line_num, word, confidence, left, top, width, height)
        )

    if not words:
        return None

    line_map: dict[tuple[int, int, int], list[str]] = {}
    line_order: list[tuple[int, int, int]] = []
    line_box: dict[tuple[int, int, int], list[float]] = {}
    line_conf: dict[tuple[int, int, int], list[float]] = {}
    confidences: list[float] = []
    for block_num, par_num, line_num, word, confidence, left, top, width, height in words:
        key = (block_num, par_num, line_num)
        right = left + width
        bottom = top + height
        if key not in line_map:
            line_map[key] = []
            line_order.append(key)
            line_box[key] = [left, top, right, bottom]
            line_conf[key] = []
        else:
            box = line_box[key]
            box[0] = min(box[0], left)
            box[1] = min(box[1], top)
            box[2] = max(box[2], right)
            box[3] = max(box[3], bottom)
        line_map[key].append(word)
        if confidence >= 0:
            line_conf[key].append(confidence)
            confidences.append(confidence)

    para_map: dict[tuple[int, int], list[str]] = {}
    para_order: list[tuple[int, int]] = []
    para_box: dict[tuple[int, int], list[float]] = {}
    para_conf: dict[tuple[int, int], list[float]] = {}
    for key in line_order:
        pkey = (key[0], key[1])
        if pkey not in para_map:
            para_map[pkey] = []
            para_order.append(pkey)
            para_box[pkey] = list(line_box[key])
            para_conf[pkey] = []
        else:
            box = para_box[pkey]
            line = line_box[key]
            box[0] = min(box[0], line[0])
            box[1] = min(box[1], line[1])
            box[2] = max(box[2], line[2])
            box[3] = max(box[3], line[3])
        para_map[pkey].append(" ".join(line_map[key]))
        para_conf[pkey].extend(line_conf[key])

    pieces: list[str] = []
    raw_records: list[dict[str, object] | None] = []
    for pkey in para_order:
        piece_text = normalize_text("\n".join(para_map[pkey]))
        if not piece_text:
            continue
        index = len(pieces)
        pieces.append(piece_text)
        box = para_box[pkey]
        x = box[0]
        y = box[1]
        width = box[2] - box[0]
        height = box[3] - box[1]
        record: dict[str, object] | None = None
        if (
            math.isfinite(x)
            and math.isfinite(y)
            and math.isfinite(width)
            and math.isfinite(height)
            and width >= 0
            and height >= 0
            and x >= 0
            and y >= 0
            and x + width <= img_width
            and y + height <= img_height
        ):
            paragraph_confidence = para_conf[pkey]
            if paragraph_confidence:
                record = {
                    "x": round(x, 1),
                    "y": round(y, 1),
                    "width": round(width, 1),
                    "height": round(height, 1),
                    "confidence": round(
                        sum(paragraph_confidence) / len(paragraph_confidence), 1
                    ),
                    "block_index": index,
                }
        raw_records.append(record)

    if not pieces:
        return None

    mean_confidence = (
        round(sum(confidences) / len(confidences), 1) if confidences else -1.0
    )
    records = [record for record in raw_records if record is not None]
    return pieces, mean_confidence, records


def _build_structured_blocks(
    pieces: list[str],
) -> tuple[str, tuple[BlockReference, ...]]:
    """Join ordered text pieces into a normalized flat text plus offset-accurate blocks."""
    parts: list[str] = []
    blocks: list[BlockReference] = []
    cursor = 0
    for piece in pieces:
        piece = normalize_text(piece)
        if not piece:
            continue
        if parts:
            parts.append("\n\n")
            cursor += 2
        block_start = cursor
        parts.append(piece)
        cursor += len(piece)
        blocks.append(
            BlockReference(
                block_type="paragraph",
                text=piece,
                start_char=block_start,
                end_char=cursor,
            )
        )
    combined = normalize_text("".join(parts))
    return combined, tuple(blocks)


def _extract_image(
    safe_name: str, content: bytes | BinaryIO, file_size: int
) -> ExtractedUpload:
    """Decode a raster image, run local OCR, and return normalized structured text.

    Only actual image decoding decides validity; the file extension is never
    trusted. Pixel and side-length limits are enforced before OCR so a huge or
    decompression-bomb image cannot exhaust memory.
    """
    try:
        from PIL import Image, ImageOps
    except ImportError:
        raise UploadProcessingError(
            "ocr_engine_unavailable",
            "Image OCR is unavailable because Pillow is not installed.",
        ) from None

    if isinstance(content, bytes):
        raw = content
    else:
        content.seek(0)
        raw = content.read(file_size)

    try:
        with Image.open(io.BytesIO(raw)) as image:
            # Force a real decode so extension/MIME spoofing is rejected and
            # structed/truncated files surface here as a controlled error.
            image.load()
            fmt = (image.format or "").lower()
            if fmt not in IMAGE_SUPPORTED_FORMATS:
                raise UploadValidationError(
                    "unsupported_image_format",
                    f"Image format '{fmt or 'unknown'}' is not supported. Supported formats are "
                    "PNG, JPEG, WEBP and TIFF.",
                    415,
                )
            width, height = image.size
            if width > MAX_IMAGE_SIDE_PIXELS or height > MAX_IMAGE_SIDE_PIXELS:
                raise UploadValidationError(
                    "image_dimension_limit_exceeded",
                    f"Image sides must be at most {MAX_IMAGE_SIDE_PIXELS} pixels.",
                    413,
                )
            if width * height > MAX_IMAGE_MEGAPIXELS * 1_000_000:
                raise UploadValidationError(
                    "image_pixel_limit_exceeded",
                    f"Images must be at most {MAX_IMAGE_MEGAPIXELS} megapixels.",
                    413,
                )
            # Apply EXIF orientation so OCR sees upright content where supported.
            oriented = ImageOps.exif_transpose(image)
            pytesseract = _load_pytesseract()
            languages = _resolve_ocr_languages()
            lang = "+".join(languages)
            # A single wall-clock deadline spans every OCR candidate AND the
            # structured-layout pass, so the total budget never grows.
            deadline = time.monotonic() + MAX_IMAGE_OCR_SECONDS
            remaining = deadline - time.monotonic()
            text = _ocr_image_candidates(
                oriented, pytesseract, max(remaining, 0.0)
            )

            # Best-effort structured layout within the same shared deadline; the
            # plain-text paragraph split remains the fallback when structured OCR
            # output is unavailable or unusable.
            layout_pieces: list[str] | None = None
            layout_confidence: float = -1.0
            layout_records: list[dict[str, object]] = []
            remaining = deadline - time.monotonic()
            if remaining > 1.0:
                layout = _ocr_image_layout(oriented, pytesseract, lang, remaining)
                if layout is not None:
                    layout_pieces, layout_confidence, layout_records = layout
    except (UploadValidationError, UploadProcessingError):
        raise
    except Exception:
        raise UploadValidationError(
            "invalid_image", "The image file is corrupt or could not be decoded."
        ) from None

    normalized = normalize_text(text)
    if not normalized:
        raise UploadValidationError(
            "empty_image_ocr",
            "No readable text was found in the image; OCR returned nothing.",
        )
    _check_text_size(normalized)

    # Preserve meaningful line/paragraph boundaries as ordered structured blocks.
    # When structured layout output is available and usable it drives the blocks;
    # otherwise the normalized plain text is split on paragraph breaks.
    if layout_pieces:
        pieces = list(layout_pieces)
    else:
        pieces = normalized.split("\n\n")
    combined, blocks = _build_structured_blocks(pieces)

    media_type = IMAGE_MIME_TYPES.get(fmt, "application/octet-stream")
    metadata = {
        "upload_filename": safe_name,
        "size_bytes": file_size,
        "format": fmt,
        "mime_type": media_type,
        "dimensions": [width, height],
        "extraction_status": "complete",
        "ocr_language": lang,
    }
    if layout_pieces:
        metadata["ocr_structured"] = True
        if layout_confidence >= 0:
            metadata["ocr_mean_confidence"] = layout_confidence
        if layout_records:
            bounded_records: list[dict[str, object]] = []
            total_bytes = 0
            for record in layout_records:
                if len(bounded_records) >= MAX_IMAGE_OCR_LAYOUT_RECORDS:
                    break
                encoded = json.dumps(record, separators=(",", ":"))
                if total_bytes + len(encoded) > MAX_IMAGE_OCR_LAYOUT_METADATA_BYTES:
                    break
                total_bytes += len(encoded)
                bounded_records.append(record)
            if bounded_records:
                metadata["ocr_layout"] = bounded_records
    return ExtractedUpload(
        filename=safe_name,
        source_type="image",
        media_type=media_type,
        text=combined,
        metadata=metadata,
        structured_blocks=tuple(blocks),
    )


_SRT_TIMELINE_RE = re.compile(
    r"^\s*(\d{2,3}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
    r"(\d{2,3}):(\d{2}):(\d{2})[,.](\d{3})\s*$"
)


def _parse_srt_timeline(line: str) -> tuple[int, int] | None:
    """Parse an SRT timeline line into (start_ms, end_ms), or None if invalid."""
    match = _SRT_TIMELINE_RE.match(line)
    if match is None:
        return None
    hs, ms, ss, mss, he, me, se, mse = (int(group) for group in match.groups())
    if ms > 59 or ss > 59 or mss > 999 or me > 59 or se > 59 or mse > 999:
        return None
    start = ((hs * 3600 + ms * 60 + ss) * 1000) + mss
    end = ((he * 3600 + me * 60 + se) * 1000) + mse
    if end < start:
        return None
    return start, end


def _build_subtitle_transcript(
    timed_items: list[tuple[int, int, str]],
    *,
    max_records: int,
    max_metadata_bytes: int,
    empty_error_code: str,
    empty_error_message: str,
) -> tuple[tuple[BlockReference, ...], str, list[dict[str, object]]]:
    """Build the unified timestamp-aware transcript shared by SRT and WebVTT.

    Given ordered ``(start_ms, end_ms, text)`` items, produce a ``subtitle``
    structured block for each with exact ``start_char``/``end_char`` offsets and
    a ``timestamp_sec`` start time, the normalized combined transcript text, and a
    bounded ``subtitle_stamps`` metadata list keyed by block index. Keeping this
    construction in one place guarantees both subtitle formats emit identical
    downstream retrieval structures.
    """
    blocks: list[BlockReference] = []
    parts: list[str] = []
    cursor = 0
    for start_ms, _end_ms, item_text in timed_items:
        if parts:
            parts.append("\n\n")
            cursor += 2
        block_start = cursor
        parts.append(item_text)
        cursor += len(item_text)
        blocks.append(
            BlockReference(
                block_type="subtitle",
                text=item_text,
                start_char=block_start,
                end_char=cursor,
                timestamp_sec=round(start_ms / 1000.0, 3),
            )
        )

    combined = normalize_text("".join(parts))
    if not combined:
        raise UploadValidationError(empty_error_code, empty_error_message)
    _check_text_size(combined)

    stamps: list[dict[str, object]] = []
    total_bytes = 0
    for index, (start_ms, end_ms, _text) in enumerate(timed_items):
        if len(stamps) >= max_records:
            break
        record = {"block_index": index, "start_ms": start_ms, "end_ms": end_ms}
        encoded = json.dumps(record, separators=(",", ":"))
        if total_bytes + len(encoded) > max_metadata_bytes:
            break
        total_bytes += len(encoded)
        stamps.append(record)

    return tuple(blocks), combined, stamps


def _extract_srt(
    safe_name: str, content: bytes | BinaryIO, file_size: int
) -> ExtractedUpload:
    """Parse an SRT subtitle file into normalized transcript text with timed blocks.

    Each subtitle becomes a ``subtitle`` structured block whose text is the readable
    transcript (sequence numbers and timeline clutter are dropped). The block's
    start time is carried on ``timestamp_sec`` and is preserved in bounded document
    metadata (``subtitle_stamps``) keyed to the matching block index. Structural
    problems (bad sequence number, invalid/inverted timeline) are controlled errors;
    subtitles with empty text are skipped, and a file that leaves no usable subtitle
    is rejected instead of persisting an empty document.
    """
    try:
        if isinstance(content, bytes):
            raw = content
        else:
            content.seek(0)
            raw = content.read(file_size)
    except Exception:
        raise UploadValidationError(
            "malformed_srt", "The SRT file could not be read."
        ) from None

    try:
        text = raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError:
        raise UploadValidationError(
            "invalid_text_encoding", "SRT uploads must use UTF-8 encoding."
        ) from None

    block_list = [
        block
        for block in re.split(r"\n{2,}", text.replace("\r\n", "\n"))
        if block.strip()
    ]

    subtitles: list[tuple[int, int, str]] = []
    for block in block_list:
        lines = block.strip().split("\n")
        if len(lines) < 3 or not re.fullmatch(r"\d+", lines[0].strip()):
            raise UploadValidationError(
                "malformed_srt", "The SRT file is malformed."
            )
        timeline = _parse_srt_timeline(lines[1])
        if timeline is None:
            raise UploadValidationError(
                "malformed_srt", "The SRT file contains an invalid timestamp."
            )
        start_ms, end_ms = timeline
        subtitle_text = normalize_text("\n".join(lines[2:]))
        if not subtitle_text:
            continue
        subtitles.append((start_ms, end_ms, subtitle_text))
        if len(subtitles) > MAX_SRT_SUBTITLES:
            raise UploadValidationError(
                "srt_subtitle_limit_exceeded",
                f"SRT files must have at most {MAX_SRT_SUBTITLES} subtitles.",
                413,
            )

    if not subtitles:
        raise UploadValidationError(
            "empty_srt", "No readable subtitles were found in the SRT file."
        )

    blocks, combined, stamps = _build_subtitle_transcript(
        subtitles,
        max_records=MAX_SRT_SUBTITLES,
        max_metadata_bytes=MAX_SRT_TIMESTAMP_METADATA_BYTES,
        empty_error_code="empty_srt",
        empty_error_message="No readable subtitles were found in the SRT file.",
    )

    metadata: dict[str, object] = {
        "upload_filename": safe_name,
        "size_bytes": file_size,
        "format": "srt",
        "mime_type": "application/x-subrip",
        "extraction_status": "complete",
        "subtitle_count": len(subtitles),
    }
    if stamps:
        metadata["subtitle_stamps"] = stamps

    return ExtractedUpload(
        filename=safe_name,
        source_type="text",
        media_type="application/x-subrip",
        text=combined,
        metadata=metadata,
        structured_blocks=blocks,
    )


_VTT_HEADER_RE = re.compile(r"^\ufeff?WEBVTT(?:\s.*)?$")
_VTT_TIMING_RE = re.compile(
    r"^\s*(?:(\d{2,}):)?(\d{2}):(\d{2})\.(\d{3})\s*-->\s*"
    r"(?:(\d{2,}):)?(\d{2}):(\d{2})\.(\d{3})(?:\s+.*)?\s*$"
)
_VTT_TAG_RE = re.compile(r"<[^>]*>")


def _parse_vtt_timeline(line: str) -> tuple[int, int] | None:
    """Parse a WebVTT timing line into (start_ms, end_ms), or None if invalid."""
    match = _VTT_TIMING_RE.match(line)
    if match is None:
        return None
    (h1, m1, s1, ms1, h2, m2, s2, ms2) = (g or 0 for g in match.groups())
    m1, s1, ms1, m2, s2, ms2 = (
        int(x) for x in (m1, s1, ms1, m2, s2, ms2)
    )
    h1, h2 = int(h1), int(h2)
    if (
        m1 > 59
        or s1 > 59
        or ms1 > 999
        or m2 > 59
        or s2 > 59
        or ms2 > 999
    ):
        return None
    start = ((h1 * 3600 + m1 * 60 + s1) * 1000) + ms1
    end = ((h2 * 3600 + m2 * 60 + s2) * 1000) + ms2
    if end < start:
        return None
    return start, end


def _extract_vtt(
    safe_name: str, content: bytes | BinaryIO, file_size: int
) -> ExtractedUpload:
    """Parse a WebVTT subtitle file into normalized transcript text with timed blocks.

    Supports standard WebVTT files beginning with ``WEBVTT``. Cue identifiers,
    ``HH:MM:SS.mmm`` / shorter timestamp forms, cue settings, and multiline cue
    text are parsed; NOTE, STYLE and REGION metadata blocks are excluded from the
    transcript. Presentation/timestamp tags are stripped from cue text while
    readable words are preserved. Each cue becomes a ``subtitle`` structured block
    with exact transcript offsets; the start time is carried on ``timestamp_sec``
    and preserved in bounded document metadata side to the matching block index.
    Structural problems (missing/out header, malformed cue, invalid or inverted
    timeline) are controlled errors; cues with empty text are skipped, and a file
    with no usable cue is rejected instead of persisting an empty document.
    """
    try:
        if isinstance(content, bytes):
            raw = content
        else:
            content.seek(0)
            raw = content.read(file_size)
    except Exception:
        raise UploadValidationError(
            "malformed_vtt", "The WebVTT file could not be read."
        ) from None

    try:
        text = raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError:
        raise UploadValidationError(
            "invalid_text_encoding", "WebVTT uploads must use UTF-8 encoding."
        ) from None

    blocks = [
        block
        for block in re.split(r"\n{2,}", text.replace("\r\n", "\n"))
        if block.strip()
    ]
    if not blocks or not _VTT_HEADER_RE.match(blocks[0].lstrip()):
        raise UploadValidationError(
            "invalid_vtt_header", "The file is not a valid WebVTT subtitle file."
        )

    cues: list[tuple[int, int, str]] = []
    for block in blocks[1:]:
        lines = block.strip().split("\n")
        first = lines[0].strip()
        if (
            first == "NOTE"
            or first.startswith("NOTE ")
            or first.startswith("NOTE\t")
            or first == "STYLE"
            or first.startswith("STYLE ")
            or first == "REGION"
            or first.startswith("REGION ")
        ):
            continue
        if "-->" in first:
            timeline_line = first
            text_lines = lines[1:]
        else:
            if len(lines) < 2:
                raise UploadValidationError(
                    "malformed_vtt", "The WebVTT file is malformed."
                )
            timeline_line = lines[1].strip()
            text_lines = lines[2:]
        timeline = _parse_vtt_timeline(timeline_line)
        if timeline is None:
            raise UploadValidationError(
                "malformed_vtt",
                "The WebVTT file contains an invalid cue timeline.",
            )
        start_ms, end_ms = timeline
        cue_text = normalize_text(
            "\n".join(_VTT_TAG_RE.sub("", line) for line in text_lines)
        )
        if not cue_text:
            continue
        cues.append((start_ms, end_ms, cue_text))
        if len(cues) > MAX_VTT_CUES:
            raise UploadValidationError(
                "vtt_cue_limit_exceeded",
                f"WebVTT files must have at most {MAX_VTT_CUES} cues.",
                413,
            )

    if not cues:
        raise UploadValidationError(
            "empty_vtt", "No readable cues were found in the WebVTT file."
        )

    blocks, combined, stamps = _build_subtitle_transcript(
        cues,
        max_records=MAX_VTT_CUES,
        max_metadata_bytes=MAX_VTT_TIMESTAMP_METADATA_BYTES,
        empty_error_code="empty_vtt",
        empty_error_message="No readable cues were found in the WebVTT file.",
    )

    metadata: dict[str, object] = {
        "upload_filename": safe_name,
        "size_bytes": file_size,
        "format": "vtt",
        "mime_type": "text/vtt",
        "extraction_status": "complete",
        "subtitle_count": len(cues),
    }
    if stamps:
        metadata["subtitle_stamps"] = stamps

    return ExtractedUpload(
        filename=safe_name,
        source_type="text",
        media_type="text/vtt",
        text=combined,
        metadata=metadata,
        structured_blocks=blocks,
    )


def extract_uploaded_text(
    filename: str | None,
    content: bytes | BinaryIO,
) -> ExtractedUpload:
    """Extract text incrementally, using bounded local OCR only for scanned PDF pages."""
    safe_name = _safe_filename(filename)
    file_size = _upload_size(content)
    if not file_size:
        raise UploadValidationError("empty_file", "The uploaded file is empty.")
    if file_size > get_max_upload_bytes():
        raise UploadValidationError(
            "file_too_large",
            f"Files must be {get_max_upload_mb()} MiB or smaller.",
            413,
        )

    suffix = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""
    if suffix == "txt":
        if _read_upload_range(content, 0, 5).startswith(b"%PDF-"):
            raise UploadValidationError(
                "file_type_mismatch", "The file content does not match TXT."
            )
        try:
            if isinstance(content, bytes):
                raw_text = content
            else:
                content.seek(0)
                raw_text = content.read(get_max_upload_bytes() + 1)
            text = raw_text.decode("utf-8-sig", errors="strict")
        except UnicodeDecodeError:
            raise UploadValidationError(
                "invalid_text_encoding", "TXT uploads must use UTF-8 encoding."
            ) from None
        if "\x00" in text:
            raise UploadValidationError("invalid_text_file", "The TXT file contains binary data.")
        normalized = normalize_text(text)
        if not normalized:
            raise UploadValidationError("empty_text", "No readable text was found in the file.")
        _check_text_size(normalized)
        return ExtractedUpload(
            filename=safe_name,
            source_type="text",
            media_type="text/plain",
            text=normalized,
            metadata={"upload_filename": safe_name, "size_bytes": file_size},
        )

    if suffix == "docx":
        return _extract_docx(safe_name, content, file_size)

    if suffix in {"html", "htm"}:
        return _extract_html(safe_name, content, file_size)

    if suffix in {"md", "markdown"}:
        return _extract_markdown(safe_name, content, file_size)

    if suffix == "csv":
        return _extract_csv(safe_name, content, file_size)

    if suffix == "json":
        return _extract_json(safe_name, content, file_size)

    if suffix == "xml":
        return _extract_xml(safe_name, content, file_size)

    if suffix == "srt":
        return _extract_srt(safe_name, content, file_size)

    if suffix == "vtt":
        return _extract_vtt(safe_name, content, file_size)

    if suffix in _UNSUPPORTED_MEDIA_SUFFIXES:
        raise UploadValidationError(
            "unsupported_media",
            "Raw audio/video files are not supported. Upload a transcript or "
            "caption file (.srt, .vtt or .txt) instead.",
            415,
        )

    if suffix in {"png", "jpg", "jpeg", "webp", "tiff"}:
        return _extract_image(safe_name, content, file_size)

    if suffix != "pdf":
        raise UploadValidationError(
            "unsupported_file_type",
            "Only .txt, .md, .markdown, .csv, .json, .xml, .srt, .vtt, .docx, "
            ".html, .htm, .png, .jpg, .jpeg, .webp, .tiff and text-based .pdf "
            "files are supported.",
            415,
        )
    if not _read_upload_range(content, 0, 5).startswith(b"%PDF-"):
        raise UploadValidationError("invalid_pdf", "The PDF file signature is invalid.")
    if b"%%EOF" not in _read_upload_range(content, max(file_size - 1024, 0), 1024):
        raise UploadValidationError("invalid_pdf", "The PDF file is malformed or unreadable.")

    ocr_pdf = None
    ocr_temp_path: str | None = None
    try:
        from pypdf import PdfReader

        pdf_source: BinaryIO = io.BytesIO(content) if isinstance(content, bytes) else content
        pdf_source.seek(0)
        reader = PdfReader(pdf_source, strict=True)
        if reader.is_encrypted:
            raise UploadValidationError(
                "encrypted_pdf", "Password-protected PDF files are not supported."
            )
        page_count = len(reader.pages)
        if page_count == 0:
            raise UploadValidationError("empty_pdf", "The PDF file contains no pages.")
        if page_count > MAX_PDF_PAGES:
            raise UploadValidationError(
                "pdf_page_limit_exceeded",
                f"PDF files must have at most {MAX_PDF_PAGES} pages.",
                413,
            )

        page_texts: list[tuple[int, str]] = []
        total_chars = 0
        ocr_page_count = 0
        ocr_started: float | None = None
        max_ocr_pages = get_max_ocr_pages()
        max_ocr_seconds = get_max_ocr_seconds()
        ocr_tools = None
        first_unreadable_page: int | None = None
        needs_ocr_pages: list[int] = []
        failed_ocr_pages: list[int] = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = normalize_text(page.extract_text() or "")
            if not _has_meaningful_pdf_text(page_text):
                ocr_page_count += 1
                if ocr_page_count > max_ocr_pages:
                    raise UploadValidationError(
                        "pdf_ocr_page_limit_exceeded",
                        f"PDF uploads may OCR at most {max_ocr_pages} scanned pages.",
                        413,
                    )
                if ocr_started is None:
                    ocr_started = time.monotonic()
                elapsed = time.monotonic() - ocr_started
                remaining = max_ocr_seconds - elapsed
                if remaining <= 0:
                    raise UploadValidationError(
                        "pdf_ocr_time_limit_exceeded",
                        "PDF OCR exceeded the safe processing time limit.",
                        413,
                    )
                if ocr_tools is None:
                    ocr_tools = _load_ocr_tools()
                    fitz, _pytesseract = ocr_tools
                    try:
                        ocr_temp_path = _copy_upload_to_temp(content)
                        ocr_pdf = fitz.open(filename=ocr_temp_path)
                    except Exception:
                        raise UploadProcessingError(
                            "pdf_renderer_unavailable",
                            "PyMuPDF could not open this PDF for local OCR.",
                        ) from None
                fitz, pytesseract = ocr_tools
                try:
                    ocr_page = ocr_pdf.load_page(page_number - 1)
                except Exception:
                    raise UploadProcessingError(
                        "pdf_renderer_unavailable", "PyMuPDF could not render this PDF page."
                    ) from None
                try:
                    page_text = normalize_text(
                        _ocr_pdf_page(
                            ocr_page,
                            fitz,
                            pytesseract,
                            min(MAX_OCR_PAGE_SECONDS, remaining),
                        )
                    )
                except UploadProcessingError:
                    # A per-page OCR failure must not discard successfully extracted
                    # pages: defer this page and keep going instead of aborting.
                    failed_ocr_pages.append(page_number)
                    needs_ocr_pages.append(page_number)
                    logger.warning(
                        "pdf_page_ocr_failed page_number=%d action=defer", page_number
                    )
                    page_text = ""
                if not page_text and page_number not in failed_ocr_pages:
                    # An image-only page yields no selectable text even after OCR.
                    # If other pages produced text, keep them and flag this page as
                    # needing OCR instead of discarding the whole document.
                    if first_unreadable_page is None:
                        first_unreadable_page = page_number
                    needs_ocr_pages.append(page_number)
                    logger.warning(
                        "pdf_page_needs_ocr page_number=%d action=defer", page_number
                    )
            total_chars += len(page_text) + (2 if page_texts and page_text else 0)
            if total_chars > MAX_EXTRACTED_TEXT_CHARS:
                raise UploadValidationError(
                    "extracted_text_too_large",
                    f"Extracted text must be at most {MAX_EXTRACTED_TEXT_CHARS} characters.",
                    413,
                )
            if page_text:
                page_texts.append((page_number, page_text))
            processed = page_number
            if processed % OCR_BATCH_PAGES == 0 or processed == page_count:
                logger.info(
                    "pdf_extraction_progress pages_processed=%d page_count=%d ocr_pages=%d "
                    "extracted_chars=%d",
                    processed,
                    page_count,
                    ocr_page_count,
                    total_chars,
                )
    except (UploadValidationError, UploadProcessingError):
        raise
    except ImportError:
        logger.warning(
            "document_ingestion_failed code=pdf_parser_unavailable file_bytes=%d", len(content)
        )
        raise UploadProcessingError("pdf_parser_unavailable") from None
    except Exception:
        # Parser internals can include document data; return only a stable safe error.
        raise UploadValidationError(
            "invalid_pdf", "The PDF file is malformed or unreadable."
        ) from None
    finally:
        if ocr_pdf is not None:
            ocr_pdf.close()
        if ocr_temp_path is not None:
            try:
                os.unlink(ocr_temp_path)
            except OSError:
                logger.warning("pdf_ocr_temp_cleanup_failed")

    if not page_texts:
        if failed_ocr_pages:
            logger.warning(
                "pdf_ocr_failed_all_pages pages=%s", failed_ocr_pages
            )
            raise UploadProcessingError(
                "pdf_ocr_failed", "Local OCR could not read scanned PDF pages."
            )
        if first_unreadable_page is not None:
            logger.warning(
                "pdf_page_no_readable_text page_number=%d action=require_ocr",
                first_unreadable_page,
            )
            raise UploadValidationError(
                "pdf_page_no_readable_text",
                f"No readable text could be extracted from PDF page {first_unreadable_page}; "
                "this PDF requires OCR to index.",
                422,
            )
        raise UploadValidationError(
            "empty_pdf_text",
            "No readable text was found in the PDF, even after local OCR.",
        )

    pieces: list[str] = []
    page_ranges: list[tuple[int, int, int]] = []
    cursor = 0
    for page_number, page_text in page_texts:
        if pieces:
            pieces.append("\n\n")
            cursor += 2
        start = cursor
        pieces.append(page_text)
        cursor += len(page_text)
        page_ranges.append((start, cursor, page_number))
    combined = normalize_text("".join(pieces))
    _check_text_size(combined)

    return ExtractedUpload(
        filename=safe_name,
        source_type="pdf",
        media_type="application/pdf",
        text=combined,
        metadata={
            "upload_filename": safe_name,
            "size_bytes": file_size,
            "page_count": page_count,
            "needs_ocr_pages": list(needs_ocr_pages),
            "extraction_status": "partial" if needs_ocr_pages else "complete",
        },
        page_ranges=tuple(page_ranges),
    )


def _add_page_references(extracted: ExtractedUpload, result) -> None:
    """Annotate PDF chunks with the source pages overlapping their character spans."""
    if not extracted.page_ranges:
        return
    for chunk in result.chunks:
        pages = [
            page_number
            for start, end, page_number in extracted.page_ranges
            if chunk.start_char < end and chunk.end_char > start
        ]
        if pages:
            chunk.metadata["page_numbers"] = pages
            chunk.metadata["page_start"] = pages[0]
            chunk.metadata["page_end"] = pages[-1]


def ingest_uploaded_file(
    filename: str | None,
    content: bytes | BinaryIO,
    *,
    session_factory: Callable[[], Session] | None = None,
    embedding_service: EmbeddingService | None = None,
    owner_id: str | None = None,
) -> UploadedDocument:
    """Extract, normalize, chunk, embed, and persist one upload via shared services."""
    extracted = extract_uploaded_text(filename, content)
    extracted_document = ExtractedDocument(
        title=extracted.filename,
        normalized_text=extracted.text,
        source=ContentSource(
            source_type=extracted.source_type,
            filename=extracted.filename,
            mime_type=extracted.media_type,
            metadata=dict(extracted.metadata),
        ),
        structured_blocks=list(extracted.structured_blocks),
        extraction_status=extracted.metadata.get("extraction_status", "complete"),
    )
    converted = ingest_extracted_document(
        extracted_document,
        session_factory=session_factory,
        embedding_service=embedding_service,
        max_chunks=MAX_UPLOAD_CHUNKS,
        annotate=lambda result: _add_page_references(extracted, result),
        owner_id=owner_id,
    )
    if not converted.indexed:
        code = converted.fallback_code or "document_ingestion_unavailable"
        if code == "chunk_limit_exceeded":
            raise UploadValidationError(
                "document_chunk_limit_exceeded",
                f"Documents must produce at most {MAX_UPLOAD_CHUNKS} chunks.",
                413,
            )
        if code in ("empty_page_text", "content_validation_failed"):
            raise UploadValidationError(
                "empty_text", "No readable text was found in the file."
            )
        logger.warning(
            "document_ingestion_failed code=%s file_bytes=%d", code, _upload_size(content)
        )
        raise UploadProcessingError(code) from None

    logger.info(
        "document_extracted source_type=%s extracted_chars=%d chunk_count=%d",
        extracted.source_type,
        len(extracted.text),
        converted.chunk_count,
    )
    try:
        create_session = session_factory or get_ingestion_session_factory()
        with create_session() as session:
            repository = DocumentRepository(session)
            stored = repository.get_by_id(converted.document_id, owner_id=owner_id)
        chunk_count = len(stored.chunks) if stored is not None else converted.chunk_count
        document = stored.document if stored is not None else None
    except (DatabaseConfigurationError, EmbeddingError, SQLAlchemyError, ValueError) as error:
        if isinstance(error, DatabaseConfigurationError):
            code = "database_configuration_error"
        elif isinstance(error, SQLAlchemyError):
            code = "database_unavailable"
        else:
            code = getattr(error, "code", "document_ingestion_unavailable")
        logger.warning(
            "document_ingestion_failed code=%s file_bytes=%d", code, _upload_size(content)
        )
        raise UploadProcessingError(code) from None
    except Exception:
        # Provider adapters may raise SDK-specific exceptions; keep their details private.
        logger.warning(
            "document_ingestion_failed code=document_ingestion_unavailable file_bytes=%d",
            _upload_size(content),
        )
        raise UploadProcessingError() from None

    logger.info(
        "document_indexed document_id=%s indexing_status=indexed chunk_count=%d duplicate=%s",
        document.id,
        chunk_count,
        converted.duplicate,
    )
    return UploadedDocument(
        document=document,
        chunk_count=chunk_count,
        duplicate=converted.duplicate,
        page_count=extracted.metadata.get("page_count")
        if isinstance(extracted.metadata.get("page_count"), int)
        else None,
    )
