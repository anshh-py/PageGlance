"""Unit coverage for bounded text/PDF document uploads."""

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import sessionmaker

from app import main
from app.document_ingestion import (
    MAX_EXTRACTED_TEXT_CHARS,
    UploadProcessingError,
    UploadValidationError,
    _add_page_references,
    extract_uploaded_text,
    ingest_uploaded_file,
)
from app.models import Base
from app.repository import DocumentRepository
from app.retrieval import build_similarity_query
from app.upload_limits import (
    MAX_MULTIPART_OVERHEAD_BYTES,
    UploadBodyLimitMiddleware,
    get_max_upload_bytes,
    get_max_upload_mb,
)


class FakeEmbeddingService:
    model_version = "test:upload:dimensions=3"
    dimensions = 3

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]


def make_text_pdf(text: str) -> bytes:
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    PdfWriter = pypdf.PdfWriter
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    stream = DecodedStreamObject()
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode())
    page[NameObject("/Contents")] = stream
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def make_pdf_pages(page_texts: list[str | None]) -> bytes:
    """Create text pages and blank scanned-page stand-ins without OCR assets."""
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = pypdf.PdfWriter()
    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        if text is None:
            continue
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        stream = DecodedStreamObject()
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode())
        page[NameObject("/Contents")] = stream
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def sqlite_factory():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_extract_txt_normalizes_utf8_and_strips_client_path() -> None:
    result = extract_uploaded_text(r"C:\fakepath\notes.txt", b"\xef\xbb\xbfHello\r\nworld")

    assert result.filename == "notes.txt"
    assert result.text == "Hello\nworld"
    assert result.source_type == "text"


def test_extract_pdf_text_records_page_reference() -> None:
    result = extract_uploaded_text("report.pdf", make_text_pdf("PageGlance PDF source text"))

    assert "PageGlance PDF source text" in result.text
    assert result.page_ranges == ((0, len(result.text), 1),)


def test_scanned_pdf_page_uses_mocked_local_ocr(monkeypatch) -> None:
    from app import document_ingestion as ingestion

    class FakePdf:
        def load_page(self, index):
            return index

        def close(self):
            pass

    class FakeFitz:
        @staticmethod
        def open(**_kwargs):
            return FakePdf()

    monkeypatch.setattr(ingestion, "_load_ocr_tools", lambda: (FakeFitz, object()))
    monkeypatch.setattr(
        ingestion, "_ocr_pdf_page", lambda page, *_args: f"Recognized page {page + 1} content"
    )
    result = extract_uploaded_text("scan.pdf", make_pdf_pages([None]))

    assert result.text == "Recognized page 1 content"
    assert result.page_ranges == ((0, len(result.text), 1),)
    assert result.metadata["page_count"] == 1


def test_scanned_pdf_over_50_pages_is_processed_incrementally_with_page_refs(
    monkeypatch, caplog
) -> None:
    from app import document_ingestion as ingestion

    caplog.set_level("INFO")
    page_count = 55
    monkeypatch.setenv("PAGEGLANCE_MAX_OCR_PAGES", str(page_count))

    class FakePdf:
        def load_page(self, index):
            return index

        def close(self):
            pass

    class FakeFitz:
        @staticmethod
        def open(**kwargs):
            assert "filename" in kwargs
            return FakePdf()

    recognized_pages = []

    def fake_ocr(page, *_args):
        recognized_pages.append(page + 1)
        return f"Scanned page {page + 1} has verifiable text."

    monkeypatch.setattr(ingestion, "_load_ocr_tools", lambda: (FakeFitz, object()))
    monkeypatch.setattr(ingestion, "_ocr_pdf_page", fake_ocr)

    result = extract_uploaded_text("large-scan.pdf", make_pdf_pages([None] * page_count))

    assert recognized_pages == list(range(1, page_count + 1))
    assert len(result.page_ranges) == page_count
    assert [page for _, _, page in result.page_ranges] == list(range(1, page_count + 1))
    assert result.metadata["page_count"] == page_count
    assert "pdf_extraction_progress pages_processed=52 page_count=55" in caplog.text
    assert "pdf_extraction_progress pages_processed=55 page_count=55" in caplog.text


def test_mixed_pdf_ocr_runs_only_for_pages_without_meaningful_text(monkeypatch) -> None:
    from app import document_ingestion as ingestion

    class FakePdf:
        def load_page(self, index):
            return index

        def close(self):
            pass

    class FakeFitz:
        @staticmethod
        def open(**_kwargs):
            return FakePdf()

    called_pages = []

    def fake_ocr(page, *_args):
        called_pages.append(page)
        return "OCR text from second page"

    monkeypatch.setattr(ingestion, "_load_ocr_tools", lambda: (FakeFitz, object()))
    monkeypatch.setattr(ingestion, "_ocr_pdf_page", fake_ocr)
    result = extract_uploaded_text("mixed.pdf", make_pdf_pages(["Short title", None]))

    assert called_pages == [1]
    assert result.page_ranges[0][2] == 1
    assert result.page_ranges[1][2] == 2
    assert "Short title" in result.text and "OCR text" in result.text


def test_pdf_ocr_failure_has_safe_processing_error(monkeypatch) -> None:
    from app import document_ingestion as ingestion

    class FakePdf:
        def load_page(self, index):
            return index

        def close(self):
            pass

    class FakeFitz:
        @staticmethod
        def open(**_kwargs):
            return FakePdf()

    monkeypatch.setattr(ingestion, "_load_ocr_tools", lambda: (FakeFitz, object()))

    def fail_ocr(*_args):
        raise UploadProcessingError("pdf_ocr_failed")

    monkeypatch.setattr(ingestion, "_ocr_pdf_page", fail_ocr)
    with pytest.raises(UploadProcessingError) as error:
        extract_uploaded_text("scan.pdf", make_pdf_pages([None]))
    assert error.value.code == "pdf_ocr_failed"
    assert "private" not in str(error.value)


def test_missing_tesseract_is_reported_as_safe_ocr_configuration_error() -> None:
    from types import SimpleNamespace

    from PIL import Image

    from app.document_ingestion import _ocr_pdf_page

    image_bytes = BytesIO()
    Image.new("RGB", (2, 2), color="white").save(image_bytes, format="PNG")

    class MissingExecutable(Exception):
        pass

    class FakePixmap:
        width = 2
        height = 2

        @staticmethod
        def tobytes(_format):
            return image_bytes.getvalue()

    class FakePage:
        rect = SimpleNamespace(width=72, height=72)

        @staticmethod
        def get_pixmap(**_kwargs):
            return FakePixmap()

    fake_provider = SimpleNamespace(
        TesseractNotFoundError=MissingExecutable,
        image_to_string=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MissingExecutable("private local path")
        ),
    )
    fake_fitz = SimpleNamespace(Matrix=lambda *_args: object())

    with pytest.raises(UploadProcessingError) as error:
        _ocr_pdf_page(FakePage(), fake_fitz, fake_provider, 1)
    assert error.value.code == "tesseract_executable_unavailable"
    assert "private" not in str(error.value)


@pytest.mark.parametrize(
    ("configured_command", "expected_command"),
    [
        (None, r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        (r"D:\OCR\tesseract.exe", r"D:\OCR\tesseract.exe"),
    ],
)
def test_ocr_tools_configure_tesseract_command(monkeypatch, configured_command, expected_command):
    import sys
    from types import SimpleNamespace

    from app import document_ingestion as ingestion

    fake_fitz = SimpleNamespace()
    fake_pytesseract = SimpleNamespace(pytesseract=SimpleNamespace(tesseract_cmd=""))
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pytesseract)
    if configured_command is None:
        monkeypatch.delenv("PAGEGLANCE_TESSERACT_CMD", raising=False)
    else:
        monkeypatch.setenv("PAGEGLANCE_TESSERACT_CMD", configured_command)

    fitz, pytesseract = ingestion._load_ocr_tools()

    assert fitz is fake_fitz
    assert pytesseract is fake_pytesseract
    assert pytesseract.pytesseract.tesseract_cmd == expected_command


def test_missing_pytesseract_has_distinct_safe_diagnostic(monkeypatch):
    import sys
    from types import SimpleNamespace

    from app import document_ingestion as ingestion

    monkeypatch.setitem(sys.modules, "fitz", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "pytesseract", None)

    with pytest.raises(UploadProcessingError) as error:
        ingestion._load_ocr_tools()

    assert error.value.code == "pytesseract_unavailable"
    assert "pytesseract" in error.value.detail


def test_pdf_with_no_readable_content_is_rejected_after_ocr(monkeypatch) -> None:
    from app import document_ingestion as ingestion

    class FakePdf:
        def load_page(self, index):
            return index

        def close(self):
            pass

    class FakeFitz:
        @staticmethod
        def open(**_kwargs):
            return FakePdf()

    monkeypatch.setattr(ingestion, "_load_ocr_tools", lambda: (FakeFitz, object()))
    monkeypatch.setattr(ingestion, "_ocr_pdf_page", lambda *_args: "  \n")
    with pytest.raises(UploadValidationError) as error:
        extract_uploaded_text("blank.pdf", make_pdf_pages([None]))
    assert error.value.code == "pdf_page_no_readable_text"
    assert "page 1" in error.value.detail


def test_ocr_page_and_chunk_limits_are_enforced(monkeypatch) -> None:
    from app import document_ingestion as ingestion

    class FakePdf:
        def load_page(self, index):
            return index

        def close(self):
            pass

    class FakeFitz:
        @staticmethod
        def open(**_kwargs):
            return FakePdf()

    monkeypatch.setattr(ingestion, "MAX_OCR_PAGES", 1)
    monkeypatch.setattr(ingestion, "_load_ocr_tools", lambda: (FakeFitz, object()))
    monkeypatch.setattr(ingestion, "_ocr_pdf_page", lambda *_args: "Readable OCR output.")
    with pytest.raises(UploadValidationError) as error:
        extract_uploaded_text("many-scans.pdf", make_pdf_pages([None, None]))
    assert error.value.code == "pdf_ocr_page_limit_exceeded"

    monkeypatch.setattr(ingestion, "MAX_UPLOAD_CHUNKS", 0)
    with pytest.raises(UploadValidationError) as chunks_error:
        ingest_uploaded_file("notes.txt", b"A valid short text document.")
    assert chunks_error.value.code == "document_chunk_limit_exceeded"


def test_mixed_pdf_page_references_reach_chunk_citation_metadata(monkeypatch) -> None:
    from app import document_ingestion as ingestion
    from app.ingestion import ingest_text

    class FakePdf:
        def load_page(self, index):
            return index

        def close(self):
            pass

    class FakeFitz:
        @staticmethod
        def open(**_kwargs):
            return FakePdf()

    monkeypatch.setattr(ingestion, "_load_ocr_tools", lambda: (FakeFitz, object()))
    monkeypatch.setattr(
        ingestion, "_ocr_pdf_page", lambda *_args: "Recognized scanned section content."
    )
    extracted = extract_uploaded_text(
        "mixed.pdf", make_pdf_pages(["Selectable opening page content.", None])
    )
    ingested = ingest_text(
        extracted.text,
        source_type="pdf",
        title=extracted.filename,
        filename=extracted.filename,
        metadata=extracted.metadata,
    )
    _add_page_references(extracted, ingested)
    assert [chunk.metadata["page_numbers"] for chunk in ingested.chunks] == [[1, 2]]


@pytest.mark.parametrize(
    ("filename", "content", "code"),
    [
        ("notes.pptx", b"text", "unsupported_file_type"),
        ("notes.txt", b"\xff", "invalid_text_encoding"),
        ("notes.txt", b"\x00binary", "invalid_text_file"),
        ("empty.txt", b"  \n", "empty_text"),
        ("report.pdf", b"%PDF-broken", "invalid_pdf"),
    ],
)
def test_rejects_invalid_uploads(filename: str, content: bytes, code: str) -> None:
    with pytest.raises(UploadValidationError) as error:
        extract_uploaded_text(filename, content)
    assert error.value.code == code


@pytest.mark.parametrize("extension", ["txt", "pdf"])
def test_upload_limit_configuration_and_file_boundaries(monkeypatch, extension) -> None:
    from app import document_ingestion as ingestion

    monkeypatch.setenv("PAGEGLANCE_MAX_UPLOAD_MB", "1")
    monkeypatch.setattr(ingestion, "MAX_EXTRACTED_TEXT_CHARS", 2 * 1024 * 1024)
    limit = get_max_upload_bytes()
    assert get_max_upload_mb() == 1

    pdf_prefix = make_text_pdf("PDF boundary test") if extension == "pdf" else b""

    def content_of(size):
        if extension == "txt":
            return b"a" * size
        before_eof, eof = pdf_prefix.rsplit(b"%%EOF", 1)
        return before_eof + b" " * (size - len(pdf_prefix)) + b"%%EOF" + eof

    below = extract_uploaded_text(f"below.{extension}", content_of(limit - 1))
    at_limit = extract_uploaded_text(f"at-limit.{extension}", content_of(limit))
    assert below.source_type == at_limit.source_type == ("pdf" if extension == "pdf" else "text")

    with pytest.raises(UploadValidationError, match="1 MiB") as oversized:
        extract_uploaded_text(f"above.{extension}", content_of(limit + 1))
    assert oversized.value.status_code == 413

    monkeypatch.setattr(ingestion, "MAX_EXTRACTED_TEXT_CHARS", MAX_EXTRACTED_TEXT_CHARS)
    with pytest.raises(UploadValidationError) as oversized_text:
        extract_uploaded_text("large.txt", b"a" * (MAX_EXTRACTED_TEXT_CHARS + 1))
    assert oversized_text.value.code == "extracted_text_too_large"


@pytest.mark.parametrize("configured", ["0", "513", "invalid"])
def test_invalid_upload_limit_uses_safe_default(monkeypatch, configured) -> None:
    monkeypatch.setenv("PAGEGLANCE_MAX_UPLOAD_MB", configured)
    assert get_max_upload_mb() == 100
    assert get_max_upload_bytes() == 100 * 1024 * 1024


def test_upload_duplicate_reuses_document_and_embeddings(caplog) -> None:
    caplog.set_level("INFO")
    factory = sqlite_factory()
    embedding = FakeEmbeddingService()
    first = ingest_uploaded_file(
        "notes.txt",
        b"A stable document body.",
        session_factory=factory,
        embedding_service=embedding,
    )
    calls_after_first = len(embedding.calls)
    repeated = ingest_uploaded_file(
        "notes.txt",
        b"A stable document body.",
        session_factory=factory,
        embedding_service=embedding,
    )

    assert repeated.document.id == first.document.id
    assert repeated.duplicate is True
    assert len(embedding.calls) == calls_after_first
    assert "document_extracted source_type=text extracted_chars=23 chunk_count=1" in caplog.text
    assert (
        f"document_indexed document_id={first.document.id} indexing_status=indexed" in caplog.text
    )
    assert "duplicate=True" in caplog.text
    assert "A stable document body." not in caplog.text


def test_uploaded_source_retrieval_query_is_document_scoped() -> None:
    statement = build_similarity_query(
        [1.0, 0.0, 0.0],
        model_version="test:upload:dimensions=3",
        dimensions=3,
        top_k=5,
        document_id="uploaded-document-id",
        source_type="pdf",
    )
    compiled = statement.compile(dialect=postgresql.dialect())
    sql = str(compiled)

    assert "document_chunks.document_id =" in sql
    assert "documents.source_type =" in sql
    assert "uploaded-document-id" in compiled.params.values()
    assert "pdf" in compiled.params.values()
    assert not any("source_url" in key for key in compiled.params)


def test_upload_route_returns_stable_metadata_and_never_calls_chat(monkeypatch) -> None:
    from datetime import datetime, timezone

    from app.document_ingestion import UploadedDocument
    from app.documents import Document

    now = datetime.now(timezone.utc)
    document = Document(
        id="doc-id",
        source_type="text",
        title="notes.txt",
        filename="notes.txt",
        content_hash="a" * 64,
        text="source",
        created_at=now,
        updated_at=now,
    )
    monkeypatch.setattr(
        main,
        "ingest_uploaded_file",
        lambda filename, content: UploadedDocument(document, 1, False, None),
    )
    response = TestClient(main.app).post(
        "/documents/upload", files={"file": ("notes.txt", b"source", "text/plain")}
    )

    assert response.status_code == 200
    assert response.json() == {
        "document_id": "doc-id",
        "filename": "notes.txt",
        "source_type": "text",
        "content_hash": "a" * 64,
        "chunk_count": 1,
        "duplicate": False,
        "page_count": None,
    }


def test_upload_route_rejects_oversized_file_without_ingesting(monkeypatch) -> None:
    monkeypatch.setenv("PAGEGLANCE_MAX_UPLOAD_MB", "1")
    monkeypatch.setattr(
        main,
        "ingest_uploaded_file",
        lambda *_args: pytest.fail("Oversized body must not be ingested"),
    )
    response = TestClient(main.app).post(
        "/documents/upload", files={"file": ("huge.txt", b"x" * (1024 * 1024 + 1))}
    )

    assert response.status_code == 413
    assert "1 MiB or smaller" in response.text


@pytest.mark.parametrize("size_delta", [-1, 0])
def test_upload_route_accepts_files_below_or_at_configured_limit(monkeypatch, size_delta) -> None:
    from datetime import datetime, timezone

    from app.document_ingestion import UploadedDocument
    from app.documents import Document

    monkeypatch.setenv("PAGEGLANCE_MAX_UPLOAD_MB", "1")
    document = Document(
        id="doc-id",
        source_type="text",
        title="boundary.txt",
        filename="boundary.txt",
        content_hash="a" * 64,
        text="x",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    monkeypatch.setattr(
        main,
        "ingest_uploaded_file",
        lambda *_args: UploadedDocument(document, 1, False, None),
    )
    content = b"x" * (1024 * 1024 + size_delta)
    response = TestClient(main.app).post(
        "/documents/upload", files={"file": ("boundary.txt", content, "text/plain")}
    )
    assert response.status_code == 200
    assert len(content) == 1024 * 1024 + size_delta


def test_upload_route_returns_safe_provider_failure(monkeypatch) -> None:
    from app.document_ingestion import UploadProcessingError

    def fail_ingestion(*_args):
        raise UploadProcessingError("private_provider_detail")

    monkeypatch.setattr(main, "ingest_uploaded_file", fail_ingestion)
    response = TestClient(main.app).post(
        "/documents/upload", files={"file": ("notes.txt", b"source", "text/plain")}
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Document ingestion is temporarily unavailable. Please try again."
    }
    assert "private_provider_detail" not in response.text


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (
            "database_configuration_error",
            "The backend database is not configured. "
            "Set DATABASE_URL in the repository root .env and restart the backend.",
        ),
        (
            "database_unavailable",
            "The backend database is unavailable. "
            "Check the local PostgreSQL service and try again.",
        ),
        (
            "embedding_configuration_error",
            "The embedding provider configuration is invalid. "
            "Check PAGEGLANCE_EMBEDDING_PROVIDER and local model settings "
            "in the repository root .env.",
        ),
        (
            "embedding_provider_error",
            "The embedding provider could not process this upload. "
            "Check its model configuration and try again.",
        ),
        (
            "embedding_model_unavailable",
            "The local embedding model is unavailable. Install backend dependencies and download "
            "or cache the configured model before retrying.",
        ),
        (
            "embedding_rate_limited",
            "The embedding provider is rate limiting uploads. Please try again shortly.",
        ),
        (
            "embedding_quota_exhausted",
            "The embedding provider quota or billing limit is exhausted. "
            "Check the provider account configuration before retrying.",
        ),
        (
            "embedding_provider_429_unknown",
            "The embedding provider rejected the upload with an unclassified rate-limit response. "
            "Check the backend logs before retrying.",
        ),
        (
            "pdf_parser_unavailable",
            "PDF uploads are unavailable because the backend PDF parser is not installed. "
            "Install backend dependencies and restart the backend.",
        ),
        (
            "pytesseract_unavailable",
            "Scanned PDF OCR requires the pytesseract Python package. Install backend dependencies "
            "and restart the backend.",
        ),
        (
            "tesseract_executable_unavailable",
            "Tesseract OCR could not be started. Set PAGEGLANCE_TESSERACT_CMD to the full path "
            "of tesseract.exe or add Tesseract to PATH, then restart the backend.",
        ),
    ],
)
def test_upload_route_returns_safe_diagnostic_for_known_failures(
    monkeypatch, caplog, code: str, expected: str
) -> None:
    from app.document_ingestion import UploadProcessingError

    def fail_ingestion(*_args):
        raise UploadProcessingError(code)

    monkeypatch.setattr(main, "ingest_uploaded_file", fail_ingestion)
    response = TestClient(main.app).post(
        "/documents/upload", files={"file": ("notes.txt", b"safe text", "text/plain")}
    )

    assert response.status_code == 503
    assert response.json() == {"detail": expected}
    assert f"document_upload_failed code={code}" in caplog.text
    assert "OPENAI_API_KEY=" not in caplog.text
    assert "postgresql" not in caplog.text


def test_missing_database_configuration_logs_safe_diagnostic(caplog) -> None:
    from app.db import DatabaseConfigurationError
    from app.document_ingestion import UploadProcessingError, ingest_uploaded_file

    def missing_database():
        raise DatabaseConfigurationError("DATABASE_URL is not configured.")

    with pytest.raises(UploadProcessingError) as failure:
        ingest_uploaded_file(
            "notes.txt",
            b"safe test text",
            session_factory=missing_database,
            embedding_service=object(),
        )

    assert failure.value.code == "database_configuration_error"
    assert str(failure.value) == "Document ingestion is temporarily unavailable. Please try again."
    assert "document_ingestion_failed code=database_configuration_error" in caplog.text
    assert "DATABASE_URL" not in caplog.text


def test_missing_pdf_parser_logs_safe_diagnostic(monkeypatch, caplog) -> None:
    import sys

    from app.document_ingestion import UploadProcessingError, extract_uploaded_text

    monkeypatch.setitem(sys.modules, "pypdf", None)
    pdf_bytes = b"%PDF-1.4\n%%EOF"
    with pytest.raises(UploadProcessingError) as failure:
        extract_uploaded_text("source.pdf", pdf_bytes)

    assert failure.value.code == "pdf_parser_unavailable"
    assert "document_ingestion_failed code=pdf_parser_unavailable file_bytes=14" in caplog.text
    assert "source.pdf" not in caplog.text


@pytest.mark.parametrize("has_content_length", [True, False])
def test_asgi_limit_rejects_oversized_body_before_multipart_parser(
    monkeypatch, has_content_length
) -> None:
    monkeypatch.setenv("PAGEGLANCE_MAX_UPLOAD_MB", "1")
    request_limit = 1024 * 1024 + MAX_MULTIPART_OVERHEAD_BYTES
    received_by_route = False
    response_messages = []

    async def route_app(_scope, receive, _send):
        nonlocal received_by_route
        received_by_route = True
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break

    chunks = [
        {
            "type": "http.request",
            "body": b"x" * (request_limit // 2 + 1),
            "more_body": True,
        },
        {
            "type": "http.request",
            "body": b"x" * (request_limit // 2 + 1),
            "more_body": False,
        },
    ]

    async def receive():
        return chunks.pop(0)

    async def send(message):
        response_messages.append(message)

    headers = []
    if has_content_length:
        headers.append(((b"content-length"), str(request_limit + 1).encode("ascii")))
    middleware = UploadBodyLimitMiddleware(route_app)
    import asyncio

    asyncio.run(
        middleware(
            {"type": "http", "method": "POST", "path": "/documents/upload", "headers": headers},
            receive,
            send,
        )
    )

    assert response_messages[0]["status"] == 413
    assert received_by_route is not has_content_length


def test_asgi_limit_accepts_declared_request_above_legacy_25_mib_cap(monkeypatch) -> None:
    import asyncio

    monkeypatch.setenv("PAGEGLANCE_MAX_UPLOAD_MB", "100")
    reached_route = False
    response_messages = []

    async def route_app(_scope, _receive, _send):
        nonlocal reached_route
        reached_route = True

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        response_messages.append(message)

    # This is the old 25 MiB file allowance + multipart overhead + one byte.
    declared_length = 25 * 1024 * 1024 + MAX_MULTIPART_OVERHEAD_BYTES + 1
    middleware = UploadBodyLimitMiddleware(route_app)
    asyncio.run(
        middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/documents/upload",
                "headers": [(b"content-length", str(declared_length).encode("ascii"))],
            },
            receive,
            send,
        )
    )

    assert reached_route
    assert response_messages == []


def test_upload_txt_metadata_preserved_through_integrated_path() -> None:
    factory = sqlite_factory()
    embedding = FakeEmbeddingService()
    uploaded = ingest_uploaded_file(
        "notes.txt",
        b"Metadata preservation text.",
        session_factory=factory,
        embedding_service=embedding,
    )

    with factory() as session:
        stored = DocumentRepository(session).get_by_id(uploaded.document.id)

    assert stored is not None
    assert stored.document.filename == "notes.txt"
    assert stored.document.metadata["upload_filename"] == "notes.txt"
    assert stored.document.metadata["size_bytes"] == len(b"Metadata preservation text.")
    assert stored.document.metadata["extraction_status"] == "complete"
    assert stored.document.metadata["mime_type"] == "text/plain"


def test_upload_pdf_page_references_reach_stored_chunks_via_integrated_path(
    monkeypatch,
) -> None:
    from app import document_ingestion as ingestion

    class FakePdf:
        def load_page(self, index):
            return index

        def close(self):
            pass

    class FakeFitz:
        @staticmethod
        def open(**_kwargs):
            return FakePdf()

    monkeypatch.setattr(ingestion, "_load_ocr_tools", lambda: (FakeFitz, object()))
    monkeypatch.setattr(
        ingestion,
        "_ocr_pdf_page",
        lambda page, *_args: f"Recognized scanned section page {page + 1} content.",
    )
    factory = sqlite_factory()
    embedding = FakeEmbeddingService()
    uploaded = ingest_uploaded_file(
        "mixed.pdf",
        make_pdf_pages(["Selectable opening page content.", None]),
        session_factory=factory,
        embedding_service=embedding,
    )

    with factory() as session:
        stored = DocumentRepository(session).get_by_id(uploaded.document.id)

    assert stored is not None
    assert uploaded.page_count == 2
    assert stored.document.metadata["page_count"] == 2
    assert stored.document.metadata["mime_type"] == "application/pdf"
    assert any("page_numbers" in chunk.metadata for chunk in stored.chunks)


def test_multipage_selectable_pdf_retains_correct_page_references() -> None:
    result = extract_uploaded_text(
        "multi.pdf",
        make_pdf_pages(["First page selectable text.", "Second page selectable text."]),
    )

    # Page boundaries are preserved with 1-based page numbers.
    assert [page for _, _, page in result.page_ranges] == [1, 2]
    # Each recorded range slices to exactly that page's text in the flattened body.
    first_start, first_end, first_page = result.page_ranges[0]
    second_start, second_end, second_page = result.page_ranges[1]
    assert (first_page, second_page) == (1, 2)
    assert result.text[first_start:first_end] == "First page selectable text."
    assert result.text[second_start:second_end] == "Second page selectable text."
    assert first_end < second_start
    assert result.text[first_end:second_start] == "\n\n"


def test_mixed_pdf_reports_image_only_pages_as_needing_ocr_and_preserves_text(
    monkeypatch,
) -> None:
    from app import document_ingestion as ingestion

    class FakePdf:
        def load_page(self, index):
            return index

        def close(self):
            pass

    class FakeFitz:
        @staticmethod
        def open(**_kwargs):
            return FakePdf()

    monkeypatch.setattr(ingestion, "_load_ocr_tools", lambda: (FakeFitz(), object()))
    # OCR yields nothing for the image-only page.
    monkeypatch.setattr(ingestion, "_ocr_pdf_page", lambda *_args: "")

    result = extract_uploaded_text(
        "mixed.pdf",
        make_pdf_pages(["Selectable page one.", None]),
    )

    # Selectable text is preserved; the image-only page is flagged, not dropped silently.
    assert "Selectable page one." in result.text
    assert result.metadata["needs_ocr_pages"] == [2]
    assert result.metadata["extraction_status"] == "partial"
    # Only the selectable page contributes text/ranges.
    assert [page for _, _, page in result.page_ranges] == [1]


def test_invalid_encrypted_and_image_only_pdfs_raise_controlled_errors(
    monkeypatch, caplog
) -> None:
    from pypdf import PdfWriter

    from app.document_ingestion import UploadValidationError

    # Encrypted PDF is rejected with a controlled code before any ingestion.
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt("hunter2")
    encrypted_bytes = BytesIO()
    writer.write(encrypted_bytes)
    with pytest.raises(UploadValidationError) as encrypted_error:
        extract_uploaded_text("locked.pdf", encrypted_bytes.getvalue())
    assert encrypted_error.value.code == "encrypted_pdf"

    # PDF with no pages at all.
    with pytest.raises(UploadValidationError) as empty_error:
        extract_uploaded_text("no-pages.pdf", make_pdf_pages([]))
    assert empty_error.value.code == "empty_pdf"

    # Fully image-only PDF (nothing extractable, even after OCR) requires OCR.
    from app import document_ingestion as ingestion

    class FakePdf:
        def load_page(self, index):
            return index

        def close(self):
            pass

    class FakeFitz:
        @staticmethod
        def open(**_kwargs):
            return FakePdf()

    monkeypatch.setattr(ingestion, "_load_ocr_tools", lambda: (FakeFitz(), object()))
    monkeypatch.setattr(ingestion, "_ocr_pdf_page", lambda *_args: " ")
    with pytest.raises(UploadValidationError) as ocr_required:
        extract_uploaded_text("scan-only.pdf", make_pdf_pages([None, None]))
    assert ocr_required.value.code == "pdf_page_no_readable_text"
    assert "requires OCR" in ocr_required.value.detail


def test_scanned_multipage_pdf_ocr_text_with_correct_page_references(
    monkeypatch,
) -> None:
    """Scanned multi-page PDF: OCR text per page with accurate 1-based refs."""
    from app import document_ingestion as ingestion

    class FakePdf:
        def load_page(self, index):
            return index

        def close(self):
            pass

    class FakeFitz:
        @staticmethod
        def open(**_kwargs):
            return FakePdf()

    monkeypatch.setattr(ingestion, "_load_ocr_tools", lambda: (FakeFitz, object()))
    monkeypatch.setattr(
        ingestion,
        "_ocr_pdf_page",
        lambda page, *_args: f"Scanned page {page + 1} of two.",
    )

    result = extract_uploaded_text("scan2.pdf", make_pdf_pages([None, None]))

    assert [page for _, _, page in result.page_ranges] == [1, 2]
    first_start, first_end, first_page = result.page_ranges[0]
    second_start, second_end, second_page = result.page_ranges[1]
    assert (first_page, second_page) == (1, 2)
    assert result.text[first_start:first_end] == "Scanned page 1 of two."
    assert result.text[second_start:second_end] == "Scanned page 2 of two."
    assert result.text[first_end:second_start] == "\n\n"


def test_mixed_pdf_combines_selectable_and_ocr_in_page_order(monkeypatch) -> None:
    """Mixed PDF: selectable and OCR text appear in correct page order."""
    from app import document_ingestion as ingestion

    class FakePdf:
        def load_page(self, index):
            return index

        def close(self):
            pass

    class FakeFitz:
        @staticmethod
        def open(**_kwargs):
            return FakePdf()

    monkeypatch.setattr(ingestion, "_load_ocr_tools", lambda: (FakeFitz, object()))
    monkeypatch.setattr(
        ingestion, "_ocr_pdf_page", lambda page, *_args: "OCR page two content."
    )

    result = extract_uploaded_text(
        "mixed.pdf", make_pdf_pages(["Selectable page one.", None])
    )

    # Selectable page one first, then OCR page two.
    assert "Selectable page one." in result.text
    assert "OCR page two content." in result.text
    assert result.text.index("Selectable page one.") < result.text.index("OCR page two content.")
    assert [page for _, _, page in result.page_ranges] == [1, 2]
    assert result.metadata["needs_ocr_pages"] == []


def test_per_page_ocr_failure_preserves_successful_pages_and_reports_partial(
    monkeypatch,
) -> None:
    """A page-level OCR failure must not discard successfully extracted pages."""
    from app import document_ingestion as ingestion

    class FakePdf:
        def load_page(self, index):
            return index

        def close(self):
            pass

    class FakeFitz:
        @staticmethod
        def open(**_kwargs):
            return FakePdf()

    monkeypatch.setattr(ingestion, "_load_ocr_tools", lambda: (FakeFitz, object()))

    def fail_ocr_on_page_two(*_args):
        raise UploadProcessingError("pdf_ocr_failed")

    monkeypatch.setattr(ingestion, "_ocr_pdf_page", fail_ocr_on_page_two)

    result = extract_uploaded_text(
        "mixed.pdf", make_pdf_pages(["Selectable page one.", None])
    )

    # Successful selectable page preserved; failed page reported, not dropped.
    assert "Selectable page one." in result.text
    assert [page for _, _, page in result.page_ranges] == [1]
    assert result.metadata["needs_ocr_pages"] == [2]
    assert result.metadata["extraction_status"] == "partial"


def test_corrupt_and_encrypted_pdfs_raise_controlled_errors() -> None:
    """Malformed/truncated and encrypted PDFs yield stable, safe errors."""
    # Truncated PDF: valid header but the %%EOF trailer is cut off.
    good = make_pdf_pages(["Readable content"])
    truncated = good[: good.rfind(b"%%EOF")]
    with pytest.raises(UploadValidationError) as corrupt:
        extract_uploaded_text("broken.pdf", truncated)
    assert corrupt.value.code == "invalid_pdf"
    # The detail is the fixed safe message; no parser internals/exceptions leak.
    assert corrupt.value.detail == "The PDF file is malformed or unreadable."

    # Encrypted PDF is rejected with its dedicated controlled code.
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt("hunter2")
    encrypted_bytes = BytesIO()
    writer.write(encrypted_bytes)
    with pytest.raises(UploadValidationError) as encrypted:
        extract_uploaded_text("locked.pdf", encrypted_bytes.getvalue())
    assert encrypted.value.code == "encrypted_pdf"


def test_oversized_and_excessive_page_pdfs_rejected_before_expensive_work(
    monkeypatch,
) -> None:
    """Size and page-count limits are enforced before any OCR work starts."""
    from app import document_ingestion as ingestion

    # Page-count limit: a 2-page PDF over a 1-page cap is rejected before OCR.
    monkeypatch.setattr(ingestion, "MAX_PDF_PAGES", 1)
    ocr_calls = []
    monkeypatch.setattr(ingestion, "_load_ocr_tools", lambda: ocr_calls.append(1))
    with pytest.raises(UploadValidationError) as page_limit:
        extract_uploaded_text("wide.pdf", make_pdf_pages(["one", "two"]))
    assert page_limit.value.code == "pdf_page_limit_exceeded"
    assert page_limit.value.status_code == 413
    assert ocr_calls == []  # rejected before any expensive OCR processing.

    # File-size limit is checked before parsing.
    monkeypatch.setattr(ingestion, "get_max_upload_bytes", lambda: 100)
    with pytest.raises(UploadValidationError) as large:
        extract_uploaded_text("big.txt", b"x" * 200)
    assert large.value.code == "file_too_large"
    assert large.value.status_code == 413


def test_ocr_failure_closes_resources_and_never_persists_empty(
    monkeypatch,
) -> None:
    """On failure the PDF document and temp copy are released; nothing persists."""
    import os
    import tempfile

    from app import document_ingestion as ingestion

    class FakePdf:
        def __init__(self) -> None:
            self.closed = False

        def load_page(self, index):
            return index

        def close(self):
            self.closed = True

    fake_pdf = FakePdf()

    class FakeFitz:
        @staticmethod
        def open(**_kwargs):
            return fake_pdf

    monkeypatch.setattr(ingestion, "_load_ocr_tools", lambda: (FakeFitz(), object()))

    def fail_ocr(*_args):
        raise UploadProcessingError("pdf_ocr_failed")

    monkeypatch.setattr(ingestion, "_ocr_pdf_page", fail_ocr)

    # Simulate a real temp copy so we can assert it is cleaned up.
    temp_file = tempfile.NamedTemporaryFile(prefix="pageglance-ocr-", suffix=".pdf", delete=False)
    temp_path = temp_file.name
    temp_file.close()
    monkeypatch.setattr(ingestion, "_copy_upload_to_temp", lambda _content: temp_path)

    with pytest.raises(UploadProcessingError) as error:
        extract_uploaded_text("scan.pdf", make_pdf_pages([None, None]))
    assert error.value.code == "pdf_ocr_failed"
    # PDF document closed and temp file removed despite the failure.
    assert fake_pdf.closed is True
    assert os.path.exists(temp_path) is False


def make_docx(paragraphs: list[dict], tables: list[list[list[str]]] | None = None) -> bytes:
    """Build a minimal OOXML DOCX archive from paragraphs and optional tables.

    Each paragraph is a dict: {"text": str, "heading": bool}.
    Tables are row-major lists of cells (cells are plain strings).
    """
    import zipfile
    from xml.etree import ElementTree as ET

    W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    ET.register_namespace("w", W)

    body = ET.Element(f"{{{W}}}body")

    def paragraph_el(text: str, heading: bool = False) -> ET.Element:
        p = ET.Element(f"{{{W}}}p")
        if heading:
            ppr = ET.SubElement(p, f"{{{W}}}pPr")
            style = ET.SubElement(ppr, f"{{{W}}}pStyle")
            style.set(f"{{{W}}}val", "Heading1")
        r = ET.SubElement(p, f"{{{W}}}r")
        t = ET.SubElement(r, f"{{{W}}}t")
        t.text = text
        return p

    for para in paragraphs:
        body.append(paragraph_el(para["text"], para.get("heading", False)))

    if tables:
        for table in tables:
            tbl = ET.SubElement(body, f"{{{W}}}tbl")
            for row in table:
                tr = ET.SubElement(tbl, f"{{{W}}}tr")
                for cell in row:
                    tc = ET.SubElement(tr, f"{{{W}}}tc")
                    p = ET.SubElement(tc, f"{{{W}}}p")
                    r = ET.SubElement(p, f"{{{W}}}r")
                    t = ET.SubElement(r, f"{{{W}}}t")
                    t.text = cell

    document = ET.Element(f"{{{W}}}document")
    document.append(body)
    xml_bytes = ET.tostring(document, encoding="utf-8", xml_declaration=True)

    out = BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", xml_bytes)
    return out.getvalue()


def test_docx_extracts_paragraphs_headings_and_tables_in_order() -> None:
    docx_bytes = make_docx(
        [
            {"text": "Project Report", "heading": True},
            {"text": "First body paragraph."},
            {"text": "Second body paragraph."},
        ],
        tables=[[["Name", "Qty"], ["Widget", "7"]]],
    )

    extracted = extract_uploaded_text("report.docx", docx_bytes)

    assert extracted.source_type == "text"
    assert (
        extracted.media_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert extracted.metadata["format"] == "docx"
    assert "Project Report" in extracted.text
    assert "First body paragraph." in extracted.text
    assert "Second body paragraph." in extracted.text
    assert "Name | Qty" in extracted.text
    assert "Widget | 7" in extracted.text
    assert extracted.text.index("Project Report") < extracted.text.index("First body paragraph.")

    kinds = [block.block_type for block in extracted.structured_blocks]
    assert kinds[0] == "heading"
    assert kinds[1] == kinds[2] == "paragraph"
    assert kinds[-1] == "table"
    for block in extracted.structured_blocks:
        assert extracted.text[block.start_char : block.end_char] == block.text


def test_docx_content_persists_through_existing_converter() -> None:
    from app.contracts import ContentSource, ExtractedDocument
    from app.converters import ingest_extracted_document
    from app.documents import DocumentSourceType

    docx_bytes = make_docx([{"text": "Intro line."}])
    extracted = extract_uploaded_text("manual.docx", docx_bytes)

    factory = sqlite_factory()
    converted = ingest_extracted_document(
        ExtractedDocument(
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
        ),
        session_factory=factory,
    )

    assert converted.indexed is True
    assert converted.document_id is not None
    assert converted.chunk_count >= 1

    # The persisted document round-trips through the shared repository.
    with factory() as s:
        stored = DocumentRepository(s).get_by_id(converted.document_id)
    assert stored is not None
    assert stored.document.source_type in DocumentSourceType.__args__
    assert stored.document.filename == "manual.docx"
    assert "Intro line." in stored.document.text


def test_empty_and_malformed_docx_return_controlled_errors() -> None:
    empty = make_docx([])
    with pytest.raises(UploadValidationError) as empty_error:
        extract_uploaded_text("empty.docx", empty)
    assert empty_error.value.code == "empty_docx"

    with pytest.raises(UploadValidationError) as bad_zip:
        extract_uploaded_text("broken.docx", b"not a zip file")
    assert bad_zip.value.code == "malformed_docx"

    import zipfile

    missing = BytesIO()
    with zipfile.ZipFile(missing, "w") as archive:
        archive.writestr("word/other.xml", "<x/>")
    with pytest.raises(UploadValidationError) as no_part:
        extract_uploaded_text("nopart.docx", missing.getvalue())
    assert no_part.value.code == "malformed_docx"


def make_html(body: str, title: str = "Sample Page") -> bytes:
    """Build a minimal HTML document as UTF-8 bytes with a title and body."""
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title></head><body>{body}</body></html>"
    ).encode()


def test_html_blocks_extract_and_boilerplate_excluded() -> None:
    html_bytes = make_html(
        "<h1>Main Heading</h1>"
        "<p>First paragraph with &amp; entity.</p>"
        "<ul><li>Item one</li><li>Item two</li></ul>"
        "<table><tr><th>Name</th><th>Qty</th></tr>"
        "<tr><td>Widget</td><td>7</td></tr></table>"
        "<pre><code>print('hi')</code></pre>"
        "<script>var x = 'should not appear';</script>"
        "<style>.hidden{color:red}</style>"
        "<nav>Home About Contact</nav>"
        "<div class='cookie-banner'>Accept cookies</div>"
    )

    extracted = extract_uploaded_text("page.html", html_bytes)

    assert extracted.source_type == "text"
    assert extracted.media_type == "text/html"
    assert extracted.metadata["format"] == "html"
    assert extracted.metadata["title"] == "Sample Page"
    assert "Main Heading" in extracted.text
    assert "First paragraph with & entity." in extracted.text
    assert "Item one" in extracted.text and "Item two" in extracted.text
    assert "Name | Qty" in extracted.text
    assert "Widget | 7" in extracted.text
    assert "print('hi')" in extracted.text

    # Boilerplate / scripts / styles / nav excluded.
    assert "should not appear" not in extracted.text
    assert ".hidden" not in extracted.text
    assert "Home About Contact" not in extracted.text
    assert "Accept cookies" not in extracted.text

    kinds = [block.block_type for block in extracted.structured_blocks]
    assert kinds[0] == "heading"
    assert "paragraph" in kinds
    assert "list_item" in kinds
    assert "table" in kinds
    assert "code" in kinds
    for block in extracted.structured_blocks:
        assert extracted.text[block.start_char : block.end_char] == block.text


def test_html_persists_through_existing_converter() -> None:
    from app.contracts import ContentSource, ExtractedDocument
    from app.converters import ingest_extracted_document
    from app.documents import DocumentSourceType

    html_bytes = make_html("<h1>Head</h1><p>Body text here.</p>", title="Rpt")
    extracted = extract_uploaded_text("report.htm", html_bytes)

    factory = sqlite_factory()
    converted = ingest_extracted_document(
        ExtractedDocument(
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
        ),
        session_factory=factory,
    )

    assert converted.indexed is True
    assert converted.document_id is not None
    assert converted.chunk_count >= 1

    with factory() as s:
        stored = DocumentRepository(s).get_by_id(converted.document_id)
    assert stored is not None
    assert stored.document.source_type in DocumentSourceType.__args__
    assert stored.document.filename == "report.htm"
    assert "Body text here." in stored.document.text


def test_empty_and_malformed_html_return_controlled_errors(monkeypatch) -> None:
    from app import document_ingestion as ingestion

    empty = make_html("", title="")
    with pytest.raises(UploadValidationError) as empty_error:
        extract_uploaded_text("empty.html", empty)
    assert empty_error.value.code == "empty_html"

    # HTMLParser is tolerant of unclosed tags, so exercise the malformed branch
    # deterministically by forcing a parser-level failure.
    class BoomParser(ingestion._DocumentHtmlParser):
        def feed(self, *_args, **_kwargs):
            raise ValueError("boom")

    monkeypatch.setattr(ingestion, "_DocumentHtmlParser", BoomParser)
    with pytest.raises(UploadValidationError) as malformed_error:
        extract_uploaded_text("broken.html", b"<html><body><p>unclosed")
    assert malformed_error.value.code == "malformed_html"


def test_markdown_blocks_extract_with_valid_offsets() -> None:
    md = """---
title: Release Notes
---
# Project Status

Intro paragraph with **bold** and `inline code` and a [link](https://example.com).

## Features
- Alpha feature
- Beta *feature*
1. First step
2. Second step

> A quoted insight.

```python
def greet(name):
    return f"hi {name}"
```

| Name | Qty |
|------|-----|
| Widget | 7 |
| Bolt | 3 |
"""

    extracted = extract_uploaded_text("notes.md", md.encode())

    assert extracted.source_type == "markdown"
    assert extracted.media_type == "text/markdown"
    assert extracted.metadata["format"] == "markdown"
    assert extracted.metadata["title"] == "Release Notes"
    assert "Project Status" in extracted.text
    assert "Intro paragraph with bold and inline code and a link." in extracted.text
    assert "Alpha feature" in extracted.text and "Beta feature" in extracted.text
    assert "First step" in extracted.text and "Second step" in extracted.text
    assert "A quoted insight." in extracted.text
    assert "def greet(name):" in extracted.text
    assert 'return f"hi {name}"' in extracted.text
    assert "Name | Qty" in extracted.text
    assert "Widget | 7" in extracted.text and "Bolt | 3" in extracted.text

    # Syntax markers are not indexed as noise.
    assert "**" not in extracted.text
    assert "###" not in extracted.text
    assert "```" not in extracted.text
    assert "---" not in extracted.text or "Release Notes" in extracted.text

    kinds = [block.block_type for block in extracted.structured_blocks]
    assert "heading" in kinds
    assert "paragraph" in kinds
    assert "list_item" in kinds
    assert "table" in kinds
    assert "code" in kinds
    for block in extracted.structured_blocks:
        assert extracted.text[block.start_char : block.end_char] == block.text


def test_markdown_content_persists_through_existing_converter() -> None:
    from app.contracts import ContentSource, ExtractedDocument
    from app.converters import ingest_extracted_document
    from app.documents import DocumentSourceType

    md = "# Head\n\nBody text here.\n\n- One\n- Two\n"
    extracted = extract_uploaded_text("doc.markdown", md.encode())

    factory = sqlite_factory()
    converted = ingest_extracted_document(
        ExtractedDocument(
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
        ),
        session_factory=factory,
    )

    assert converted.indexed is True
    assert converted.document_id is not None
    assert converted.chunk_count >= 1

    with factory() as s:
        stored = DocumentRepository(s).get_by_id(converted.document_id)
    assert stored is not None
    assert stored.document.source_type in DocumentSourceType.__args__
    assert stored.document.filename == "doc.markdown"
    assert "Body text here." in stored.document.text
    assert "One" in stored.document.text


def test_txt_compatibility_and_empty_markdown_controlled_errors() -> None:
    # TXT path unchanged and still works through the universal converter.
    txt_extracted = extract_uploaded_text("plain.txt", b"Hello world.")
    assert txt_extracted.source_type == "text"
    assert txt_extracted.media_type == "text/plain"
    assert txt_extracted.text == "Hello world."

    factory = sqlite_factory()
    from app.contracts import ContentSource, ExtractedDocument
    from app.converters import ingest_extracted_document

    converted = ingest_extracted_document(
        ExtractedDocument(
            title=txt_extracted.filename,
            normalized_text=txt_extracted.text,
            source=ContentSource(
                source_type=txt_extracted.source_type,
                filename=txt_extracted.filename,
                mime_type=txt_extracted.media_type,
                metadata=dict(txt_extracted.metadata),
            ),
        ),
        session_factory=factory,
    )
    assert converted.indexed is True
    assert converted.document_id is not None
    with factory() as s:
        stored = DocumentRepository(s).get_by_id(converted.document_id)
    assert stored is not None
    assert stored.document.filename == "plain.txt"

    # Empty/whitespace-only markdown -> controlled error, nothing persisted.
    with pytest.raises(UploadValidationError) as empty_error:
        extract_uploaded_text("empty.md", b"   \n\n")
    assert empty_error.value.code == "empty_markdown"

    # Unsupported content is still rejected.
    with pytest.raises(UploadValidationError) as bad_error:
        extract_uploaded_text("notes.pptx", b"%PDF- broken")
    assert bad_error.value.code == "unsupported_file_type"


def test_csv_quoted_fields_headers_rows_offsets_and_persistence() -> None:
    from app.contracts import ContentSource, ExtractedDocument
    from app.converters import ingest_extracted_document

    csv_bytes = (
        b"name,address,note\n"
        b"\"Doe, Jane\",\"123 Main St, Apt 4\",\"Line1\nLine2\"\n"
        b"\"O'Reilly\",\"7 Elm Ave\",\"said \"\"hi\"\"\"\n"
        b",unknown,\n"
    )

    extracted = extract_uploaded_text("people.csv", csv_bytes)

    assert extracted.source_type == "text"
    assert extracted.media_type == "text/csv"
    assert extracted.metadata["format"] == "csv"
    assert extracted.metadata["upload_filename"] == "people.csv"
    assert "name | address | note" in extracted.text
    assert "Doe, Jane | 123 Main St, Apt 4 | Line1 Line2" in extracted.text
    assert "O'Reilly" in extracted.text and "7 Elm Ave" in extracted.text
    assert " | unknown |" in extracted.text

    kinds = [block.block_type for block in extracted.structured_blocks]
    assert kinds and all(kind == "table" for kind in kinds)
    for block in extracted.structured_blocks:
        assert extracted.text[block.start_char : block.end_char] == block.text

    factory = sqlite_factory()
    converted = ingest_extracted_document(
        ExtractedDocument(
            title=extracted.filename,
            normalized_text=extracted.text,
            source=ContentSource(
                source_type=extracted.source_type,
                filename=extracted.filename,
                mime_type=extracted.media_type,
                metadata=dict(extracted.metadata),
            ),
            structured_blocks=list(extracted.structured_blocks),
        ),
        session_factory=factory,
    )
    assert converted.indexed is True
    assert converted.document_id is not None
    with factory() as s:
        stored = DocumentRepository(s).get_by_id(converted.document_id)
    assert stored is not None
    assert stored.document.filename == "people.csv"
    assert "Doe, Jane" in stored.document.text


def test_json_nested_paths_arrays_offsets_and_persistence() -> None:
    import json

    from app.contracts import ContentSource, ExtractedDocument
    from app.converters import ingest_extracted_document

    payload = {
        "customer": {"name": "Riya", "address": {"city": "Jaipur"}},
        "orders": [
            {"id": 1, "active": True},
            {"id": 2, "price": 19.99},
        ],
        "meta": {"nullable": None, "tags": ["a", "b"]},
    }
    extracted = extract_uploaded_text("data.json", json.dumps(payload).encode())

    assert extracted.source_type == "text"
    assert extracted.media_type == "application/json"
    assert extracted.metadata["format"] == "json"
    assert extracted.metadata["upload_filename"] == "data.json"
    assert "customer.name: Riya" in extracted.text
    assert "customer.address.city: Jaipur" in extracted.text
    assert "orders[0].id: 1" in extracted.text
    assert "orders[0].active: true" in extracted.text
    assert "orders[1].price: 19.99" in extracted.text
    assert "meta.nullable: null" in extracted.text
    assert "meta.tags[0]: a" in extracted.text
    assert "meta.tags[1]: b" in extracted.text

    for block in extracted.structured_blocks:
        assert extracted.text[block.start_char : block.end_char] == block.text

    factory = sqlite_factory()
    converted = ingest_extracted_document(
        ExtractedDocument(
            title=extracted.filename,
            normalized_text=extracted.text,
            source=ContentSource(
                source_type=extracted.source_type,
                filename=extracted.filename,
                mime_type=extracted.media_type,
                metadata=dict(extracted.metadata),
            ),
            structured_blocks=list(extracted.structured_blocks),
        ),
        session_factory=factory,
    )
    assert converted.indexed is True
    assert converted.document_id is not None
    with factory() as s:
        stored = DocumentRepository(s).get_by_id(converted.document_id)
    assert stored is not None
    assert stored.document.filename == "data.json"
    assert "customer.address.city: Jaipur" in stored.document.text


def test_csv_json_invalid_empty_oversized_controlled_errors() -> None:
    with pytest.raises(UploadValidationError) as empty_csv:
        extract_uploaded_text("empty.csv", b" , , \n")
    assert empty_csv.value.code == "empty_csv"

    with pytest.raises(UploadValidationError) as empty_json:
        extract_uploaded_text("empty.json", b"{}")
    assert empty_json.value.code == "empty_json"

    with pytest.raises(UploadValidationError) as bad_json:
        extract_uploaded_text("bad.json", b"{not json}")
    assert bad_json.value.code == "malformed_json"

    wide = "a" + ','.join(["x"] * 2005) + "\n"
    with pytest.raises(UploadValidationError) as wide_csv:
        extract_uploaded_text("wide.csv", wide.encode())
    assert wide_csv.value.code == "csv_column_limit_exceeded"

    deep = "0"
    for _ in range(40):
        deep = f"[{deep}]"
    with pytest.raises(UploadValidationError) as deep_json:
        extract_uploaded_text("deep.json", deep.encode())
    assert deep_json.value.code == "json_depth_limit_exceeded"


def test_xml_nested_attributes_repeated_siblings_namespaces_offsets() -> None:
    from app.contracts import ContentSource, ExtractedDocument
    from app.converters import ingest_extracted_document

    xml_bytes = (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<catalog xmlns:bk="urn:books">'
        b'  <bk:book id="123">'
        b'    <bk:title>Example</bk:title>'
        b'  </bk:book>'
        b'  <bk:book id="456">'
        b'    <bk:title>Second</bk:title>'
        b'  </bk:book>'
        b'</catalog>'
    )

    extracted = extract_uploaded_text("catalog.xml", xml_bytes)

    assert extracted.source_type == "text"
    assert extracted.media_type == "application/xml"
    assert extracted.metadata["format"] == "xml"
    assert extracted.metadata["upload_filename"] == "catalog.xml"
    # Namespaces are handled: the local element names are preserved.
    assert "catalog.book[0].title: Example" in extracted.text
    assert "catalog.book[1].title: Second" in extracted.text
    assert "catalog.book[0].@id: 123" in extracted.text
    assert "catalog.book[1].@id: 456" in extracted.text

    kinds = [block.block_type for block in extracted.structured_blocks]
    assert kinds and all(kind == "xml" for kind in kinds)
    for block in extracted.structured_blocks:
        assert extracted.text[block.start_char : block.end_char] == block.text

    factory = sqlite_factory()
    converted = ingest_extracted_document(
        ExtractedDocument(
            title=extracted.filename,
            normalized_text=extracted.text,
            source=ContentSource(
                source_type=extracted.source_type,
                filename=extracted.filename,
                mime_type=extracted.media_type,
                metadata=dict(extracted.metadata),
            ),
            structured_blocks=list(extracted.structured_blocks),
        ),
        session_factory=factory,
    )
    assert converted.indexed is True
    assert converted.document_id is not None
    with factory() as s:
        stored = DocumentRepository(s).get_by_id(converted.document_id)
    assert stored is not None
    assert stored.document.filename == "catalog.xml"
    assert "catalog.book[0].title: Example" in stored.document.text


def test_xml_mixed_text_and_tail_preserved_in_order() -> None:
    xml_bytes = b"<p>Hello <b>world</b>!</p>"
    extracted = extract_uploaded_text("mixed.xml", xml_bytes)
    assert "p: Hello" in extracted.text
    assert "p.b: world" in extracted.text
    assert "p: !" in extracted.text


def test_xml_invalid_empty_dtd_deep_controlled_errors() -> None:
    with pytest.raises(UploadValidationError) as malformed:
        extract_uploaded_text("bad.xml", b"<catalog><book></catalog>")
    assert malformed.value.code == "malformed_xml"

    with pytest.raises(UploadValidationError) as empty_xml:
        extract_uploaded_text("empty.xml", b"   ")
    assert empty_xml.value.code == "empty_xml"

    with pytest.raises(UploadValidationError) as empty_content:
        extract_uploaded_text("blank.xml", b"<root></root>")
    assert empty_content.value.code == "empty_xml"

    with pytest.raises(UploadValidationError) as dtd:
        extract_uploaded_text(
            "entity.xml",
            b'<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
            b"<foo>&xxe;</foo>",
        )
    assert dtd.value.code == "xml_dtd_blocked"


def make_image_bytes(fmt: str, size=(120, 40), color=(255, 255, 255)) -> bytes:
    """Build an in-memory image via Pillow for image-OCR tests."""
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


def test_image_valid_ocr_metadata_offsets_and_persistence(monkeypatch) -> None:
    from app import document_ingestion as ingestion

    captured: list[object] = []

    def fake_ocr(image, pytesseract, timeout):
        captured.append(image)
        return "Invoice Total\n\nAmount 42"

    monkeypatch.setattr(ingestion, "_ocr_image", fake_ocr)
    monkeypatch.setattr(ingestion, "_load_pytesseract", lambda: object())

    extracted = extract_uploaded_text("receipt.png", make_image_bytes("PNG"))
    assert extracted.source_type == "image"
    assert extracted.media_type == "image/png"
    assert extracted.metadata["format"] == "png"
    assert extracted.metadata["dimensions"] == [120, 40]
    assert extracted.metadata["extraction_status"] == "complete"
    assert "Invoice Total" in extracted.text
    assert "Amount 42" in extracted.text
    assert captured  # OCR actually received a decoded image object.

    for block in extracted.structured_blocks:
        assert extracted.text[block.start_char : block.end_char] == block.text

    factory = sqlite_factory()
    from app.contracts import ContentSource, ExtractedDocument
    from app.converters import ingest_extracted_document

    converted = ingest_extracted_document(
        ExtractedDocument(
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
        ),
        session_factory=factory,
    )
    assert converted.indexed is True
    assert converted.document_id is not None
    with factory() as s:
        stored = DocumentRepository(s).get_by_id(converted.document_id)
    assert stored is not None
    assert stored.document.source_type == "image"
    assert stored.document.filename == "receipt.png"
    assert "Invoice Total" in stored.document.text


def test_image_exif_orientation_is_applied_before_ocr(monkeypatch) -> None:
    from PIL import Image

    from app import document_ingestion as ingestion

    # Build a JPEG with EXIF orientation=6 (rotate 90 deg clockwise).
    buf = BytesIO()
    img = Image.new("RGB", (120, 40), (255, 255, 255))
    exif = img.getexif()
    exif[274] = 6  # Orientation tag.
    img.save(buf, format="JPEG", exif=exif)
    jpeg_bytes = buf.getvalue()

    seen_sizes: list[tuple[int, int]] = []

    def fake_ocr(image, pytesseract, timeout):
        seen_sizes.append(image.size)
        return "Oriented text"

    monkeypatch.setattr(ingestion, "_ocr_image", fake_ocr)
    monkeypatch.setattr(ingestion, "_load_pytesseract", lambda: object())

    extract_uploaded_text("oriented.jpg", jpeg_bytes)
    # EXIF 6 swaps width/height relative to the raw frame, so the oriented
    # fallback candidate must be 40x120.
    assert seen_sizes
    assert (40, 120) in seen_sizes


def test_image_corrupt_unsupported_oversized_and_empty_ocr_controlled_errors(
    monkeypatch,
) -> None:
    from app import document_ingestion as ingestion

    # Corrupt bytes with a plausible extension are rejected, not trusted.
    with pytest.raises(UploadValidationError) as invalid:
        extract_uploaded_text("broken.png", b"not-an-image")
    assert invalid.value.code == "invalid_image"

    # Unsupported real format (BMP) hidden behind .png is rejected.
    with pytest.raises(UploadValidationError) as unsupported:
        extract_uploaded_text("bmp.png", make_image_bytes("BMP"))
    assert unsupported.value.code == "unsupported_image_format"

    # Pixel/dimension limits are enforced before OCR.
    monkeypatch.setattr(ingestion, "MAX_IMAGE_MEGAPIXELS", 0)
    with pytest.raises(UploadValidationError) as pixels:
        extract_uploaded_text("big.png", make_image_bytes("PNG", (200, 200)))
    assert pixels.value.code == "image_pixel_limit_exceeded"

    monkeypatch.setattr(ingestion, "MAX_IMAGE_SIDE_PIXELS", 10)
    with pytest.raises(UploadValidationError) as side:
        extract_uploaded_text("wide.jpg", make_image_bytes("JPEG"))
    assert side.value.code == "image_dimension_limit_exceeded"

    # Empty OCR output -> controlled failure, never a persisted empty document.
    monkeypatch.setattr(ingestion, "MAX_IMAGE_MEGAPIXELS", 25)
    monkeypatch.setattr(ingestion, "MAX_IMAGE_SIDE_PIXELS", 16_000)
    monkeypatch.setattr(ingestion, "_load_pytesseract", lambda: object())
    monkeypatch.setattr(ingestion, "_ocr_image", lambda *_a: "")
    with pytest.raises(UploadValidationError) as empty_ocr:
        extract_uploaded_text("blank.png", make_image_bytes("PNG"))
    assert empty_ocr.value.code == "empty_image_ocr"


def test_image_preprocessing_applied_within_limits(monkeypatch) -> None:
    from app import document_ingestion as ingestion

    # A small, low-contrast image triggers upscaling + grayscale/contrast.
    seen: list[tuple[tuple[int, int], str]] = []

    def fake_ocr(image, pytesseract, timeout):
        seen.append((image.size, image.mode))
        return "Invoice"

    monkeypatch.setattr(ingestion, "_ocr_image", fake_ocr)
    monkeypatch.setattr(ingestion, "_load_pytesseract", lambda: object())

    extract_uploaded_text("small.png", make_image_bytes("PNG", (60, 30)))

    # Several bounded variants were attempted, not just one.
    assert len(seen) >= 2
    sizes = [size for size, _mode in seen]
    modes = set(mode for _size, mode in seen)
    # Original fallback is present, plus an upscaled variant.
    assert (60, 30) in sizes
    assert any(size[0] > 60 for size in sizes)
    # Grayscale variant is among the preprocessing outputs.
    assert "L" in modes
    # Every attempted variant stays inside the configured pixel/dimension budget.
    for (w, h), _mode in seen:
        assert w <= ingestion.MAX_IMAGE_SIDE_PIXELS
        assert h <= ingestion.MAX_IMAGE_SIDE_PIXELS
        assert w * h <= ingestion.MAX_IMAGE_MEGAPIXELS * 1_000_000


def test_image_ocr_candidate_selection_deterministic_with_fallback(monkeypatch) -> None:
    from app import document_ingestion as ingestion

    # Prefer the candidate with the most meaningful (alphanumeric) text.
    def fake_ocr(image, pytesseract, timeout):
        if image.size == (60, 30):
            return "Hello World"  # meaningful short text on the original
        if image.size == (120, 60):
            return "The quick brown fox jumps over the lazy dog"
        return "abc"  # weaker candidates

    monkeypatch.setattr(ingestion, "_ocr_image", fake_ocr)
    monkeypatch.setattr(ingestion, "_load_pytesseract", lambda: object())

    extracted = extract_uploaded_text("select.png", make_image_bytes("PNG", (60, 30)))
    assert "quick brown fox" in extracted.text

    # Deterministic: same input, same selection.
    extracted_again = extract_uploaded_text(
        "select.png", make_image_bytes("PNG", (60, 30))
    )
    assert extracted.text == extracted_again.text

    # Original image remains a fallback when preprocessing variants fail.
    def failing_ocr(image, pytesseract, timeout):
        if image.size == (60, 30):
            return "Fallback OK"
        raise ingestion.UploadProcessingError("image_ocr_failed")

    monkeypatch.setattr(ingestion, "_ocr_image", failing_ocr)
    fallback = extract_uploaded_text("fallback.png", make_image_bytes("PNG", (60, 30)))
    assert "Fallback OK" in fallback.text

    # The 40s OCR budget spans all attempts, not a fresh per-attempt budget.
    timeouts: list[float] = []
    clock = {"t": 0.0}
    monkeypatch.setattr(ingestion.time, "monotonic", lambda: clock["t"])

    def budget_ocr(image, pytesseract, timeout):
        clock["t"] += 25.0  # simulate each OCR call consuming 25s
        timeouts.append(timeout)
        return "ok"

    monkeypatch.setattr(ingestion, "_ocr_image", budget_ocr)
    monkeypatch.setattr(ingestion, "_load_pytesseract", lambda: object())
    extract_uploaded_text("budget.png", make_image_bytes("PNG", (60, 30)))
    # Attempt 1 gets the full 40s; attempt 2 only the 15s remainder; the loop
    # stops once the deadline is drained, so no third attempt runs.
    assert timeouts[0] == pytest.approx(40.0)
    assert timeouts[1] == pytest.approx(15.0)
    assert len(timeouts) == 2


def test_image_preprocessing_compatibility_offsets_persistence_and_empty_ocr(
    monkeypatch,
) -> None:
    from app import document_ingestion as ingestion

    def fake_ocr(image, pytesseract, timeout):
        return "Aligned total\n\nSecond line"

    monkeypatch.setattr(ingestion, "_ocr_image", fake_ocr)
    monkeypatch.setattr(ingestion, "_load_pytesseract", lambda: object())

    extracted = extract_uploaded_text("ok.png", make_image_bytes("PNG"))
    assert extracted.metadata["extraction_status"] == "complete"
    assert extracted.metadata["dimensions"] == [120, 40]
    assert extracted.metadata["format"] == "png"
    for block in extracted.structured_blocks:
        assert extracted.text[block.start_char : block.end_char] == block.text

    factory = sqlite_factory()
    from app.contracts import ContentSource, ExtractedDocument
    from app.converters import ingest_extracted_document

    converted = ingest_extracted_document(
        ExtractedDocument(
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
        ),
        session_factory=factory,
    )
    assert converted.indexed is True
    with factory() as s:
        stored = DocumentRepository(s).get_by_id(converted.document_id)
    assert stored is not None
    assert stored.document.source_type == "image"
    assert "Aligned total" in stored.document.text

    # All OCR variants empty -> controlled failure, never persisted.
    monkeypatch.setattr(ingestion, "_ocr_image", lambda *_a: "")
    with pytest.raises(UploadValidationError) as empty_ocr:
        extract_uploaded_text("blank.png", make_image_bytes("PNG"))
    assert empty_ocr.value.code == "empty_image_ocr"


def test_image_ocr_language_configuration_default_combination_and_unavailable_pack(
    monkeypatch,
) -> None:
    from app import document_ingestion as ingestion

    # The default language resolves to English.
    assert ingestion._resolve_ocr_languages() == ["eng"]

    # A configured combination resolves to two codes.
    monkeypatch.setenv("PAGEGLANCE_TESSERACT_LANGUAGES", "eng+hin")
    assert ingestion._resolve_ocr_languages() == ["eng", "hin"]

    # An invalid language configuration is rejected with a controlled error.
    monkeypatch.setenv("PAGEGLANCE_TESSERACT_LANGUAGES", "eng;;;")
    with pytest.raises(ingestion.UploadProcessingError) as invalid_config:
        ingestion._resolve_ocr_languages()
    assert invalid_config.value.code == "ocr_language_config_invalid"

    # The configured language reaches Tesseract as lang="eng+hin".
    monkeypatch.setenv("PAGEGLANCE_TESSERACT_LANGUAGES", "eng+hin")
    seen = {}

    class RecordingPytesseract:
        class TesseractError(Exception):
            pass

        class TesseractNotFoundError(Exception):
            pass

        @staticmethod
        def image_to_string(image, *, lang="", timeout=None):
            seen["lang"] = lang
            return "Readable text"

    result = ingestion._ocr_image(object(), RecordingPytesseract(), 10.0)
    assert seen["lang"] == "eng+hin"
    assert "Readable text" in result

    # An unavailable language pack surfaces a controlled, actionable error.
    class StubPytesseract:
        class TesseractError(Exception):
            pass

        class TesseractNotFoundError(Exception):
            pass

        @staticmethod
        def image_to_string(*args, **kwargs):
            raise StubPytesseract.TesseractError("Failed loading language 'hin'")

    with pytest.raises(ingestion.UploadProcessingError) as missing_pack:
        ingestion._ocr_image(object(), StubPytesseract(), 10.0)
    assert missing_pack.value.code == "ocr_language_unavailable"


def test_image_structured_ocr_layout_lines_paragraphs_confidence_and_offsets(
    monkeypatch,
) -> None:
    from PIL import Image

    from app import document_ingestion as ingestion


    class FakeDataPytesseract:
        class Output:
            DICT = "dict"

        class TesseractError(Exception):
            pass

        class TesseractNotFoundError(Exception):
            pass

        @staticmethod
        def image_to_data(image, lang, output_type, timeout):
            return {
                "level": [5] * 6,
                "page_num": [1] * 6,
                "block_num": [1, 1, 1, 1, 1, 1],
                "par_num": [1, 1, 1, 2, 2, 2],
                "line_num": [1, 1, 1, 1, 1, 1],
                "word_num": [0, 1, 2, 0, 1, 2],
                "left": [10, 20, 30, 10, 20, 30],
                "top": [0, 0, 0, 40, 40, 40],
                "width": [5, 5, 5, 5, 5, 5],
                "height": [10, 10, 10, 10, 10, 10],
                "conf": [95, 98, 90, 80, 85, 75],
                "text": ["Hello", "", "world", "Second", "", "line"],
            }

    img = Image.new("RGB", (120, 80))
    pieces, confidence, records = ingestion._ocr_image_layout(
        img, FakeDataPytesseract(), "eng", 10.0
    )
    assert pieces == ["Hello world", "Second line"]
    assert confidence == pytest.approx(85.0)
    assert records == [
        {
            "x": 10.0,
            "y": 0.0,
            "width": 25.0,
            "height": 10.0,
            "confidence": 92.5,
            "block_index": 0,
        },
        {
            "x": 10.0,
            "y": 40.0,
            "width": 25.0,
            "height": 10.0,
            "confidence": 77.5,
            "block_index": 1,
        },
    ]

    def layout_ocr(
        image, pytesseract, timeout, lang=ingestion.DEFAULT_IMAGE_OCR_LANGUAGE
    ):
        return "Hello world\n\nSecond line"

    def structured_layout(image, pytesseract, lang, timeout):
        return (
            ["Hello world", "Second line"],
            95.0,
            [
                {
                    "x": 10.0,
                    "y": 0.0,
                    "width": 25.0,
                    "height": 10.0,
                    "confidence": 95.0,
                    "block_index": 0,
                },
                {
                    "x": 10.0,
                    "y": 20.0,
                    "width": 25.0,
                    "height": 10.0,
                    "confidence": 95.0,
                    "block_index": 1,
                },
            ],
        )

    monkeypatch.setattr(ingestion, "_ocr_image", layout_ocr)
    monkeypatch.setattr(ingestion, "_ocr_image_layout", structured_layout)
    monkeypatch.setattr(ingestion, "_load_pytesseract", lambda: object())

    extracted = extract_uploaded_text("layout.png", make_image_bytes("PNG"))
    assert extracted.text == "Hello world\n\nSecond line"
    assert extracted.metadata.get("ocr_structured") is True
    assert extracted.metadata.get("ocr_mean_confidence") == 95.0
    assert [b.text for b in extracted.structured_blocks] == [
        "Hello world",
        "Second line",
    ]
    for block in extracted.structured_blocks:
        assert extracted.text[block.start_char : block.end_char] == block.text


def test_image_structured_ocr_fallback_persistence_and_deadline(
    monkeypatch,
) -> None:
    from app import document_ingestion as ingestion

    # Structured output unavailable -> plain-text fallback still extracts/persists.
    def plain_ocr(
        image, pytesseract, timeout, lang=ingestion.DEFAULT_IMAGE_OCR_LANGUAGE
    ):
        return "Para one line one\nPara one line two\n\nSecond paragraph"

    monkeypatch.setattr(ingestion, "_ocr_image", plain_ocr)
    monkeypatch.setattr(ingestion, "_ocr_image_layout", lambda *_a, **_k: None)
    monkeypatch.setattr(ingestion, "_load_pytesseract", lambda: object())

    extracted = extract_uploaded_text("fallback.png", make_image_bytes("PNG"))
    assert extracted.metadata["extraction_status"] == "complete"
    assert "ocr_structured" not in extracted.metadata
    assert [b.text for b in extracted.structured_blocks] == [
        "Para one line one\nPara one line two",
        "Second paragraph",
    ]
    for block in extracted.structured_blocks:
        assert extracted.text[block.start_char : block.end_char] == block.text

    # Persistence through the shared converter/repository.
    factory = sqlite_factory()
    from app.contracts import ContentSource, ExtractedDocument
    from app.converters import ingest_extracted_document

    converted = ingest_extracted_document(
        ExtractedDocument(
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
        ),
        session_factory=factory,
    )
    assert converted.indexed is True
    with factory() as s:
        stored = DocumentRepository(s).get_by_id(converted.document_id)
    assert stored is not None and "Second paragraph" in stored.document.text

    # A single 40s deadline spans attempts AND layout; layout gets the remainder.
    clock = {"t": 0.0}
    monkeypatch.setattr(ingestion.time, "monotonic", lambda: clock["t"])
    timeouts: list[float] = []
    layout_timeouts: list[float] = []

    def timed_ocr(
        image, pytesseract, timeout, lang=ingestion.DEFAULT_IMAGE_OCR_LANGUAGE
    ):
        clock["t"] += 2.0
        timeouts.append(timeout)
        return "ok"

    def timed_layout(image, pytesseract, lang, timeout):
        layout_timeouts.append(timeout)
        return None

    monkeypatch.setattr(ingestion, "_ocr_image", timed_ocr)
    monkeypatch.setattr(ingestion, "_ocr_image_layout", timed_layout)
    extract_uploaded_text("deadline.png", make_image_bytes("PNG"))
    assert timeouts[0] == pytest.approx(40.0)  # first candidate gets the full budget
    assert len(timeouts) == 5  # all preprocessing variants are attempted
    assert layout_timeouts[0] == pytest.approx(30.0)  # layout gets the remainder
    assert clock["t"] <= 40.0  # never exceeds the total budget


def test_image_ocr_layout_records_persist_and_correspond_to_blocks(monkeypatch) -> None:
    """M5.3: valid layout metadata is persisted and maps to the right text blocks."""
    from app import document_ingestion as ingestion
    from app.contracts import ContentSource, ExtractedDocument
    from app.converters import ingest_extracted_document

    def structured_layout(image, pytesseract, lang, timeout):
        return (
            ["Invoice Total", "Amount 42"],
            95.0,
            [
                {
                    "x": 10.0,
                    "y": 0.0,
                    "width": 60.0,
                    "height": 12.0,
                    "confidence": 96.0,
                    "block_index": 0,
                },
                {
                    "x": 10.0,
                    "y": 20.0,
                    "width": 50.0,
                    "height": 12.0,
                    "confidence": 94.0,
                    "block_index": 1,
                },
            ],
        )

    monkeypatch.setattr(
        ingestion, "_ocr_image", lambda *_a: "Invoice Total\n\nAmount 42"
    )
    monkeypatch.setattr(ingestion, "_ocr_image_layout", structured_layout)
    monkeypatch.setattr(ingestion, "_load_pytesseract", lambda: object())

    extracted = extract_uploaded_text("layout.png", make_image_bytes("PNG"))
    assert extracted.text == "Invoice Total\n\nAmount 42"
    records = extracted.metadata.get("ocr_layout")
    assert records == [
        {
            "x": 10.0,
            "y": 0.0,
            "width": 60.0,
            "height": 12.0,
            "confidence": 96.0,
            "block_index": 0,
        },
        {
            "x": 10.0,
            "y": 20.0,
            "width": 50.0,
            "height": 12.0,
            "confidence": 94.0,
            "block_index": 1,
        },
    ]
    # Each record's block_index points at the matching structured text block.
    for record in records:
        block = extracted.structured_blocks[record["block_index"]]
        assert block.block_type == "paragraph"
        assert extracted.text[block.start_char : block.end_char] == block.text

    factory = sqlite_factory()
    converted = ingest_extracted_document(
        ExtractedDocument(
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
        ),
        session_factory=factory,
    )
    with factory() as s:
        stored = DocumentRepository(s).get_by_id(converted.document_id)
    assert stored is not None
    assert stored.document.metadata["ocr_layout"] == records


def test_image_ocr_layout_records_filter_invalid_boxes_and_bound_excess(
    monkeypatch,
) -> None:
    """M5.3: invalid/excessive layout records are filtered or bounded."""
    from PIL import Image

    from app import document_ingestion as ingestion

    class FakeDataPytesseract:
        class Output:
            DICT = "dict"

        class TesseractError(Exception):
            pass

        class TesseractNotFoundError(Exception):
            pass

        @staticmethod
        def image_to_data(image, lang, output_type, timeout):
            return {
                "block_num": [1, 1, 2],
                "par_num": [1, 2, 1],
                "line_num": [1, 1, 1],
                "left": [5, 500, 5],
                "top": [5, 500, 40],
                "width": [20, 20, 20],
                "height": [10, 10, 10],
                "conf": [90, 70, 88],
                "text": ["Alpha", "Beta", ""],
            }

    img = Image.new("RGB", (200, 200))
    pieces, confidence, records = ingestion._ocr_image_layout(
        img, FakeDataPytesseract(), "eng", 10.0
    )
    assert pieces == ["Alpha", "Beta"]
    assert confidence == pytest.approx(80.0)
    # The out-of-bounds "Beta" paragraph and the empty-text entry are dropped.
    assert records == [
        {
            "x": 5.0,
            "y": 5.0,
            "width": 20.0,
            "height": 10.0,
            "confidence": 90.0,
            "block_index": 0,
        }
    ]

    # Excessive records are bounded to the configured cap inside _extract_image.
    def many_layout(image, pytesseract, lang, timeout):
        return (
            [f"line{i}" for i in range(5)],
            99.0,
            [
                {
                    "x": 0.0,
                    "y": float(i),
                    "width": 10.0,
                    "height": 5.0,
                    "confidence": 95.0,
                    "block_index": i,
                }
                for i in range(500)
            ],
        )

    monkeypatch.setattr(
        ingestion,
        "_ocr_image",
        lambda *_a: "\n\n".join(f"line{i}" for i in range(5)),
    )
    monkeypatch.setattr(ingestion, "_ocr_image_layout", many_layout)
    monkeypatch.setattr(ingestion, "_load_pytesseract", lambda: object())
    extracted = extract_uploaded_text("many.png", make_image_bytes("PNG"))
    assert (
        len(extracted.metadata["ocr_layout"]) == ingestion.MAX_IMAGE_OCR_LAYOUT_RECORDS
    )


def test_image_ocr_layout_plain_text_fallback_persists(monkeypatch) -> None:
    """M5.3: plain-text fallback and existing ingestion persistence still work."""
    from app import document_ingestion as ingestion
    from app.contracts import ContentSource, ExtractedDocument
    from app.converters import ingest_extracted_document

    monkeypatch.setattr(
        ingestion,
        "_ocr_image",
        lambda *_a: "Para one line one\nPara one line two\n\nSecond paragraph",
    )
    monkeypatch.setattr(ingestion, "_ocr_image_layout", lambda *_a, **_k: None)
    monkeypatch.setattr(ingestion, "_load_pytesseract", lambda: object())

    extracted = extract_uploaded_text("fallback.png", make_image_bytes("PNG"))
    assert "ocr_structured" not in extracted.metadata
    assert "ocr_layout" not in extracted.metadata
    assert extracted.metadata["extraction_status"] == "complete"
    assert [b.text for b in extracted.structured_blocks] == [
        "Para one line one\nPara one line two",
        "Second paragraph",
    ]
    for block in extracted.structured_blocks:
        assert extracted.text[block.start_char : block.end_char] == block.text

    factory = sqlite_factory()
    converted = ingest_extracted_document(
        ExtractedDocument(
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
        ),
        session_factory=factory,
    )
    with factory() as s:
        stored = DocumentRepository(s).get_by_id(converted.document_id)
    assert stored is not None
    assert "Para one line one\nPara one line two" in stored.document.text
    assert "ocr_layout" not in stored.document.metadata


SRT_VALID = (
    "1\n"
    "00:00:01,000 --> 00:00:03,250\n"
    "Hello world\n"
    "multiline subtitle text\n"
    "\n"
    "2\n"
    "00:00:04,500 --> 00:00:06,000\n"
    "Second subtitle line\n"
)


def test_srt_valid_multiline_timestamps_offsets_and_persistence() -> None:
    """M6.1: valid SRT yields exact offsets plus preserved, bounded timestamps."""
    from app.contracts import ContentSource, ExtractedDocument
    from app.converters import ingest_extracted_document

    extracted = extract_uploaded_text("movie.srt", SRT_VALID.encode())

    assert extracted.source_type == "text"
    assert extracted.media_type == "application/x-subrip"
    assert extracted.metadata["format"] == "srt"
    assert extracted.metadata["mime_type"] == "application/x-subrip"
    assert extracted.metadata["subtitle_count"] == 2
    # Readable transcript with no sequence-number / timeline clutter.
    assert extracted.text == "Hello world\nmultiline subtitle text\n\nSecond subtitle line"
    assert "00:00" not in extracted.text
    assert "--> " not in extracted.text

    assert [b.block_type for b in extracted.structured_blocks] == [
        "subtitle",
        "subtitle",
    ]
    for block in extracted.structured_blocks:
        assert extracted.text[block.start_char : block.end_char] == block.text
    assert extracted.structured_blocks[0].timestamp_sec == pytest.approx(1.0)
    assert extracted.structured_blocks[1].timestamp_sec == pytest.approx(4.5)
    assert extracted.metadata["subtitle_stamps"] == [
        {"block_index": 0, "start_ms": 1000, "end_ms": 3250},
        {"block_index": 1, "start_ms": 4500, "end_ms": 6000},
    ]

    factory = sqlite_factory()
    converted = ingest_extracted_document(
        ExtractedDocument(
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
        ),
        session_factory=factory,
    )
    assert converted.indexed is True
    with factory() as s:
        stored = DocumentRepository(s).get_by_id(converted.document_id)
    assert stored is not None
    assert stored.document.filename == "movie.srt"
    assert stored.document.metadata["format"] == "srt"
    assert stored.document.text == "Hello world\nmultiline subtitle text\n\nSecond subtitle line"
    # Start time propagates into chunk metadata via the structured-block references.
    assert any(
        ref.get("timestamp_sec") == pytest.approx(1.0)
        for chunk in stored.chunks
        for ref in chunk.metadata.get("structured_blocks", [])
    )


def test_srt_empty_malformed_and_invalid_controlled_errors() -> None:
    """M6.1: empty or malformed SRT is rejected and never persists empty docs."""
    with pytest.raises(UploadValidationError) as empty_error:
        extract_uploaded_text("empty.srt", b"   \n\n\n")
    assert empty_error.value.code == "empty_srt"

    # Subtitle with only sequence + timeline (no text) is malformed.
    no_text = "1\n00:00:01,000 --> 00:00:02,000\n\n"
    with pytest.raises(UploadValidationError) as no_text_error:
        extract_uploaded_text("notext.srt", no_text.encode())
    assert no_text_error.value.code == "malformed_srt"

    # Non-integer sequence number.
    bad_seq = "x\n00:00:01,000 --> 00:00:02,000\nText\n"
    with pytest.raises(UploadValidationError) as seq_error:
        extract_uploaded_text("badseq.srt", bad_seq.encode())
    assert seq_error.value.code == "malformed_srt"

    # Missing timeline line.
    bad_time = "1\nhello\nText\n"
    with pytest.raises(UploadValidationError) as time_error:
        extract_uploaded_text("badtime.srt", bad_time.encode())
    assert time_error.value.code == "malformed_srt"

    # Inverted timeline (end before start).
    inverted = "1\n00:00:05,000 --> 00:00:02,000\nText\n"
    with pytest.raises(UploadValidationError) as inv_error:
        extract_uploaded_text("inverted.srt", inverted.encode())
    assert inv_error.value.code == "malformed_srt"


def test_srt_ingestion_persists_with_format_metadata_and_deduplicates() -> None:
    """M6.1: successful SRT persists via the shared pipeline and deduplicates."""
    from app.document_ingestion import ingest_uploaded_file

    factory = sqlite_factory()
    embedding = FakeEmbeddingService()
    first = ingest_uploaded_file(
        "movie.srt",
        SRT_VALID.encode(),
        session_factory=factory,
        embedding_service=embedding,
    )
    calls_after_first = len(embedding.calls)
    repeated = ingest_uploaded_file(
        "movie.srt",
        SRT_VALID.encode(),
        session_factory=factory,
        embedding_service=embedding,
    )

    assert repeated.duplicate is True
    assert repeated.document.id == first.document.id
    assert len(embedding.calls) == calls_after_first
    assert first.document.source_type == "text"

    with factory() as s:
        stored = DocumentRepository(s).get_by_id(first.document.id)
    assert stored is not None
    assert stored.document.metadata["format"] == "srt"
    assert stored.document.metadata["mime_type"] == "application/x-subrip"
    assert stored.document.metadata["subtitle_count"] == 2
    assert stored.document.metadata["subtitle_stamps"][0] == {
        "block_index": 0,
        "start_ms": 1000,
        "end_ms": 3250,
    }
    assert "Hello world\nmultiline subtitle text" in stored.document.text


VTT_VALID = (
    "WEBVTT\n"
    "\n"
    "00:01.000 --> 00:03.500 align:start position:10%\n"
    "<i>Hello</i> world\n"
    "<c.yellow>multiline</c> cue text\n"
    "\n"
    "NOTE this note must not appear in transcript\n"
    "still note text\n"
    "\n"
    "STYLE\n"
    "::cue { color: yellow }\n"
    "\n"
    "REGION\n"
    "id:reg1 width:40%\n"
    "\n"
    "cue-id\n"
    "00:05.000 --> 00:06.000\n"
    "Second <b>cue</b> line\n"
)


def test_vtt_valid_multiline_timestamps_offsets_and_persistence() -> None:
    """M6.2: valid VTT yields exact offsets plus preserved, bounded timestamps."""
    from app.contracts import ContentSource, ExtractedDocument
    from app.converters import ingest_extracted_document

    extracted = extract_uploaded_text("movie.vtt", VTT_VALID.encode())

    assert extracted.source_type == "text"
    assert extracted.media_type == "text/vtt"
    assert extracted.metadata["format"] == "vtt"
    assert extracted.metadata["mime_type"] == "text/vtt"
    assert extracted.metadata["subtitle_count"] == 2
    # Readable transcript; no identifiers, settings, tags, or metadata blocks.
    assert extracted.text == (
        "Hello world\nmultiline cue text\n\nSecond cue line"
    )
    assert "00:01" not in extracted.text
    assert "align:start" not in extracted.text
    assert "position:10%" not in extracted.text
    assert "<i>" not in extracted.text
    assert "NOTE" not in extracted.text
    assert "STYLE" not in extracted.text
    assert "REGION" not in extracted.text

    assert [b.block_type for b in extracted.structured_blocks] == [
        "subtitle",
        "subtitle",
    ]
    for block in extracted.structured_blocks:
        assert extracted.text[block.start_char : block.end_char] == block.text
    assert extracted.structured_blocks[0].timestamp_sec == pytest.approx(1.0)
    assert extracted.structured_blocks[1].timestamp_sec == pytest.approx(5.0)
    assert extracted.metadata["subtitle_stamps"] == [
        {"block_index": 0, "start_ms": 1000, "end_ms": 3500},
        {"block_index": 1, "start_ms": 5000, "end_ms": 6000},
    ]

    factory = sqlite_factory()
    converted = ingest_extracted_document(
        ExtractedDocument(
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
        ),
        session_factory=factory,
    )
    assert converted.indexed is True
    with factory() as s:
        stored = DocumentRepository(s).get_by_id(converted.document_id)
    assert stored is not None
    assert stored.document.filename == "movie.vtt"
    assert stored.document.metadata["format"] == "vtt"
    assert stored.document.text == (
        "Hello world\nmultiline cue text\n\nSecond cue line"
    )
    # Start time propagates into chunk metadata via the structured-block references.
    assert any(
        ref.get("timestamp_sec") == pytest.approx(1.0)
        for chunk in stored.chunks
        for ref in chunk.metadata.get("structured_blocks", [])
    )


def test_vtt_blocks_excluded_and_malformed_controlled_errors() -> None:
    """M6.2: NOTE/STYLE/REGION excluded; bad header/cue/timeline rejected."""
    with pytest.raises(UploadValidationError) as header_error:
        extract_uploaded_text("bad.vtt", b"NOPE\n\n00:01.000 --> 00:02.000\nHi\n")
    assert header_error.value.code == "invalid_vtt_header"

    with pytest.raises(UploadValidationError) as empty_error:
        extract_uploaded_text("empty.vtt", b"WEBVTT\n\n\n")
    assert empty_error.value.code == "empty_vtt"

    # Cue with no timing line (a lone identifier line).
    malformed_cue = "WEBVTT\n\nplain_id\n"
    with pytest.raises(UploadValidationError) as cue_error:
        extract_uploaded_text("badcue.vtt", malformed_cue.encode())
    assert cue_error.value.code == "malformed_vtt"

    # Unparseable timeline.
    bad_time = "WEBVTT\n\nnotatime --> alsobad\nText\n"
    with pytest.raises(UploadValidationError) as time_error:
        extract_uploaded_text("badtime.vtt", bad_time.encode())
    assert time_error.value.code == "malformed_vtt"

    # Inverted timeline (end before start).
    inverted = "WEBVTT\n\n00:05.000 --> 00:02.000\nText\n"
    with pytest.raises(UploadValidationError) as inv_error:
        extract_uploaded_text("inverted.vtt", inverted.encode())
    assert inv_error.value.code == "malformed_vtt"


def test_vtt_ingestion_persists_with_format_metadata_and_deduplicates() -> None:
    """M6.2: successful VTT persists via the shared pipeline and deduplicates."""
    from app.document_ingestion import ingest_uploaded_file

    factory = sqlite_factory()
    embedding = FakeEmbeddingService()
    first = ingest_uploaded_file(
        "movie.vtt",
        VTT_VALID.encode(),
        session_factory=factory,
        embedding_service=embedding,
    )
    calls_after_first = len(embedding.calls)
    repeated = ingest_uploaded_file(
        "movie.vtt",
        VTT_VALID.encode(),
        session_factory=factory,
        embedding_service=embedding,
    )

    assert repeated.duplicate is True
    assert repeated.document.id == first.document.id
    assert len(embedding.calls) == calls_after_first
    assert first.document.source_type == "text"

    with factory() as s:
        stored = DocumentRepository(s).get_by_id(first.document.id)
    assert stored is not None
    assert stored.document.metadata["format"] == "vtt"
    assert stored.document.metadata["mime_type"] == "text/vtt"
    assert stored.document.metadata["subtitle_count"] == 2
    assert stored.document.metadata["subtitle_stamps"][0] == {
        "block_index": 0,
        "start_ms": 1000,
        "end_ms": 3500,
    }
    assert "Hello world\nmultiline cue text" in stored.document.text


# M6.3: equivalent SRT and VTT content must produce identical timestamp-aware
# transcript structures.
SRT_EQUIVALENT = (
    "1\n"
    "00:00:01,000 --> 00:00:03,250\n"
    "Hello world\n"
    "multiline subtitle text\n"
    "\n"
    "2\n"
    "00:00:04,500 --> 00:00:06,000\n"
    "Second subtitle line\n"
)
VTT_EQUIVALENT = (
    "WEBVTT\n"
    "\n"
    "00:01.000 --> 00:03.250\n"
    "Hello world\n"
    "multiline subtitle text\n"
    "\n"
    "00:04.500 --> 00:06.000\n"
    "Second subtitle line\n"
)


def _transcript_signature(extracted) -> tuple:
    """Extract the shared, timestamp-aware structure fields for comparison."""
    return (
        extracted.text,
        tuple(
            (
                block.block_type,
                block.start_char,
                block.end_char,
                block.text,
                block.timestamp_sec,
            )
            for block in extracted.structured_blocks
        ),
        tuple(
            tuple(sorted(record.items()))
            for record in extracted.metadata.get("subtitle_stamps", [])
        ),
    )


def test_srt_and_vtt_equivalent_timestamp_aware_structures() -> None:
    """M6.3: SRT and VTT yield identical transcript structures for equal content."""
    srt = extract_uploaded_text("eq.srt", SRT_EQUIVALENT.encode())
    vtt = extract_uploaded_text("eq.vtt", VTT_EQUIVALENT.encode())

    assert _transcript_signature(srt) == _transcript_signature(vtt)
    assert srt.metadata["subtitle_count"] == vtt.metadata["subtitle_count"] == 2
    assert srt.metadata["subtitle_stamps"] == vtt.metadata["subtitle_stamps"]
    assert [b.block_type for b in srt.structured_blocks] == ["subtitle", "subtitle"]


def test_subtitle_block_indexes_and_offsets_survive_filtering() -> None:
    """M6.3: block indexes/offsets stay correct after empty-cue filtering."""
    # The middle cue is empty after tag stripping and must be dropped, so the
    # surviving cues re-index as 0 and 1 and offsets are computed on the filtered
    # transcript only.
    filtered_vtt = (
        "WEBVTT\n"
        "\n"
        "00:01.000 --> 00:02.000\n"
        "First\n"
        "\n"
        "00:02.500 --> 00:03.000\n"
        "<v>\n"
        "\n"
        "00:03.500 --> 00:04.000\n"
        "Third\n"
    )
    extracted = extract_uploaded_text("filtered.vtt", filtered_vtt.encode())

    assert extracted.text == "First\n\nThird"
    assert extracted.metadata["subtitle_count"] == 2
    blocks = extracted.structured_blocks
    assert [b.block_type for b in blocks] == ["subtitle", "subtitle"]
    assert [b.start_char for b in blocks] == [0, 7]
    assert [b.end_char for b in blocks] == [5, 12]
    for block in blocks:
        assert extracted.text[block.start_char : block.end_char] == block.text
    assert extracted.metadata["subtitle_stamps"] == [
        {"block_index": 0, "start_ms": 1000, "end_ms": 2000},
        {"block_index": 1, "start_ms": 3500, "end_ms": 4000},
    ]


def test_converter_preserves_timestamps_for_both_formats() -> None:
    """M6.3: converter persists timestamp metadata for SRT and VTT alike."""
    from app.contracts import ContentSource, ExtractedDocument
    from app.converters import ingest_extracted_document

    for filename, body in (("eq.srt", SRT_EQUIVALENT), ("eq.vtt", VTT_EQUIVALENT)):
        extracted = extract_uploaded_text(filename, body.encode())
        factory = sqlite_factory()
        converted = ingest_extracted_document(
            ExtractedDocument(
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
            ),
            session_factory=factory,
        )
        with factory() as s:
            stored = DocumentRepository(s).get_by_id(converted.document_id)
        assert stored is not None
        refs = [
            ref
            for chunk in stored.chunks
            for ref in chunk.metadata.get("structured_blocks", [])
        ]
        assert any(ref.get("timestamp_sec") == pytest.approx(1.0) for ref in refs)
        assert any(ref.get("timestamp_sec") == pytest.approx(4.5) for ref in refs)
        assert stored.document.metadata["subtitle_stamps"] == [
            {"block_index": 0, "start_ms": 1000, "end_ms": 3250},
            {"block_index": 1, "start_ms": 4500, "end_ms": 6000},
        ]


def test_media_caption_filename_dispatch_and_parser_selection() -> None:
    """M6.4: multi-dot media-caption names route to the right subtitle parser."""
    # lecture.mp4.en.vtt -> last-segment suffix "vtt" -> VTT parser.
    vtt = extract_uploaded_text("lecture.mp4.en.vtt", VTT_VALID.encode())
    assert vtt.source_type == "text"
    assert vtt.media_type == "text/vtt"
    assert vtt.metadata["format"] == "vtt"
    assert vtt.metadata["mime_type"] == "text/vtt"
    assert [b.block_type for b in vtt.structured_blocks] == [
        "subtitle",
        "subtitle",
    ]
    assert vtt.metadata["subtitle_stamps"][0] == {
        "block_index": 0,
        "start_ms": 1000,
        "end_ms": 3500,
    }

    # lecture.en.srt -> last-segment suffix "srt" -> SRT parser.
    srt = extract_uploaded_text("lecture.en.srt", SRT_VALID.encode())
    assert srt.source_type == "text"
    assert srt.media_type == "application/x-subrip"
    assert srt.metadata["format"] == "srt"
    assert srt.metadata["subtitle_count"] == 2
    assert srt.structured_blocks[0].timestamp_sec == pytest.approx(1.0)

    # meeting_transcript.txt -> existing plain-text TXT path.
    txt = extract_uploaded_text("meeting_transcript.txt", b"meeting notes text")
    assert txt.source_type == "text"
    assert txt.media_type == "text/plain"
    assert "format" not in txt.metadata
    assert "subtitle_stamps" not in txt.metadata
    assert txt.text == "meeting notes text"


def test_raw_media_rejected_and_plain_text_transcript() -> None:
    """M6.4: raw binaries are rejected; .txt transcripts ingest without timestamps."""
    for ext in ("mp3", "mp4", "wav", "mkv", "mov", "webm"):
        with pytest.raises(UploadValidationError) as media_error:
            extract_uploaded_text(f"audio.{ext}", b"\xff\x00binary container bytes")
        assert media_error.value.code == "unsupported_media"
        assert "transcript" in media_error.value.detail

    # .txt transcript persists through the pipeline with no fabricated timestamps.
    from app.contracts import ContentSource, ExtractedDocument
    from app.converters import ingest_extracted_document

    extracted = extract_uploaded_text(
        "meeting_transcript.txt", b"First speaking turn.\n\nSecond speaking turn."
    )
    factory = sqlite_factory()
    converted = ingest_extracted_document(
        ExtractedDocument(
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
        ),
        session_factory=factory,
    )
    assert converted.indexed is True
    with factory() as s:
        stored = DocumentRepository(s).get_by_id(converted.document_id)
    assert stored is not None
    assert stored.document.source_type == "text"
    assert "format" not in stored.document.metadata
    assert "subtitle_stamps" not in stored.document.metadata
    # No timestamp appears anywhere in the persisted transcript or chunk metadata.
    assert "timestamp_sec" not in stored.document.metadata
    assert all(
        "timestamp_sec" not in ref
        for chunk in stored.chunks
        for ref in chunk.metadata.get("structured_blocks", [])
    )


def test_srt_vtt_existing_timestamp_aware_behavior_intact() -> None:
    """M6.4: existing SRT/VTT timestamp-aware behavior is unchanged."""
    for filename, content, fmt in (
        ("movie.srt", SRT_VALID, "srt"),
        ("movie.vtt", VTT_VALID, "vtt"),
    ):
        extracted = extract_uploaded_text(filename, content.encode())
        assert extracted.metadata["format"] == fmt
        assert extracted.structured_blocks[0].block_type == "subtitle"
        assert extracted.structured_blocks[0].timestamp_sec == pytest.approx(1.0)
        assert extracted.metadata["subtitle_stamps"][0] == {
            "block_index": 0,
            "start_ms": 1000,
            "end_ms": 3500 if fmt == "vtt" else 3250,
        }
        for block in extracted.structured_blocks:
            assert extracted.text[block.start_char : block.end_char] == block.text


