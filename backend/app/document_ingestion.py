"""Safe TXT/PDF extraction and persistence for uploaded user documents."""

import io
import logging
import math
import os
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import BinaryIO

from dotenv import load_dotenv
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db import DatabaseConfigurationError
from app.documents import Document, DocumentSourceType
from app.embedding_pipeline import embed_and_persist
from app.embeddings import EmbeddingError, EmbeddingService
from app.ingestion import ingest_text
from app.normalization import normalize_text
from app.page_indexing import get_ingestion_embedding_service, get_ingestion_session_factory
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
DEFAULT_TESSERACT_COMMAND = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


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
    return min(_configured_ocr_limit("PAGEGLANCE_MAX_OCR_PAGES", MAX_OCR_PAGES, MAX_PDF_PAGES), MAX_PDF_PAGES)


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
        with tempfile.NamedTemporaryFile(prefix="pageglance-ocr-", suffix=".pdf", delete=False) as out:
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

    if suffix != "pdf":
        raise UploadValidationError(
            "unsupported_file_type", "Only .txt and text-based .pdf files are supported.", 415
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
                page_text = normalize_text(
                    _ocr_pdf_page(
                        ocr_page,
                        fitz,
                        pytesseract,
                        min(MAX_OCR_PAGE_SECONDS, remaining),
                    )
                )
                if not page_text:
                    logger.warning(
                        "pdf_page_no_readable_text page_number=%d action=reject_upload", page_number
                    )
                    raise UploadValidationError(
                        "pdf_page_no_readable_text",
                        f"No readable text could be extracted from PDF page {page_number}.",
                        422,
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
) -> UploadedDocument:
    """Extract, normalize, chunk, embed, and persist one upload via shared services."""
    extracted = extract_uploaded_text(filename, content)
    result = ingest_text(
        extracted.text,
        source_type=extracted.source_type,
        title=extracted.filename,
        filename=extracted.filename,
        metadata=extracted.metadata,
    )
    _add_page_references(extracted, result)
    logger.info(
        "document_extracted source_type=%s extracted_chars=%d chunk_count=%d",
        extracted.source_type,
        len(extracted.text),
        len(result.chunks),
    )
    if len(result.chunks) > MAX_UPLOAD_CHUNKS:
        raise UploadValidationError(
            "document_chunk_limit_exceeded",
            f"Documents must produce at most {MAX_UPLOAD_CHUNKS} chunks.",
            413,
        )

    try:
        create_session = session_factory or get_ingestion_session_factory()
        service = embedding_service or get_ingestion_embedding_service()
        with create_session() as session:
            repository = DocumentRepository(session)
            existing = repository.get_by_id(result.document.id)
            saved = embed_and_persist(result, repository=repository, service=service)
            duplicate = saved.id != result.document.id or (
                existing is not None
                and existing.document.content_hash == result.document.content_hash
            )
            stored = repository.get_by_id(saved.id)
            chunk_count = len(stored.chunks) if stored is not None else len(result.chunks)
        logger.info(
            "document_indexed document_id=%s indexing_status=indexed chunk_count=%d duplicate=%s",
            saved.id,
            chunk_count,
            duplicate,
        )
        return UploadedDocument(
            document=saved,
            chunk_count=chunk_count,
            duplicate=duplicate,
            page_count=extracted.metadata.get("page_count")
            if isinstance(extracted.metadata.get("page_count"), int)
            else None,
        )
    except (DatabaseConfigurationError, EmbeddingError, SQLAlchemyError, ValueError) as error:
        if isinstance(error, DatabaseConfigurationError):
            code = "database_configuration_error"
        elif isinstance(error, SQLAlchemyError):
            code = "database_unavailable"
        else:
            code = getattr(error, "code", "document_ingestion_unavailable")
        logger.warning("document_ingestion_failed code=%s file_bytes=%d", code, _upload_size(content))
        raise UploadProcessingError(code) from None
    except Exception:
        # Provider adapters may raise SDK-specific exceptions; keep their details private.
        logger.warning(
            "document_ingestion_failed code=document_ingestion_unavailable file_bytes=%d",
            _upload_size(content),
        )
        raise UploadProcessingError() from None
