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
    MAX_UPLOAD_BYTES,
    UploadProcessingError,
    UploadValidationError,
    _add_page_references,
    extract_uploaded_text,
    ingest_uploaded_file,
)
from app.models import Base
from app.retrieval import build_similarity_query
from app.upload_limits import MAX_UPLOAD_REQUEST_BYTES, UploadBodyLimitMiddleware


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
    assert error.value.code == "ocr_engine_unavailable"
    assert "private" not in str(error.value)


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
    assert error.value.code == "empty_pdf_text"
    assert "No readable text" in error.value.detail


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
        ("notes.docx", b"text", "unsupported_file_type"),
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


def test_rejects_file_and_extracted_text_over_limits() -> None:
    with pytest.raises(UploadValidationError, match="25 MiB") as oversized:
        extract_uploaded_text("huge.txt", b"a" * (MAX_UPLOAD_BYTES + 1))
    assert oversized.value.status_code == 413

    with pytest.raises(UploadValidationError) as oversized_text:
        extract_uploaded_text("large.txt", b"a" * (MAX_EXTRACTED_TEXT_CHARS + 1))
    assert oversized_text.value.code == "extracted_text_too_large"


def test_upload_duplicate_reuses_document_and_embeddings() -> None:
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


def test_upload_route_rejects_oversized_request_before_ingestion(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "ingest_uploaded_file",
        lambda *_args: pytest.fail("Oversized body must not be ingested"),
    )
    response = TestClient(main.app).post(
        "/documents/upload", files={"file": ("huge.txt", b"x" * (MAX_UPLOAD_BYTES + 1))}
    )

    assert response.status_code == 413
    assert "25 MiB or smaller" in response.text


def test_upload_route_accepts_file_below_25_mib_with_multipart_overhead(monkeypatch) -> None:
    from datetime import datetime, timezone

    from app.document_ingestion import UploadedDocument
    from app.documents import Document

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
    content = b"x" + b" " * (MAX_UPLOAD_BYTES - 2)
    response = TestClient(main.app).post(
        "/documents/upload", files={"file": ("boundary.txt", content, "text/plain")}
    )
    assert response.status_code == 200
    assert len(content) == MAX_UPLOAD_BYTES - 1


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
            "ocr_engine_unavailable",
            "Local OCR is unavailable. Install pytesseract and Tesseract OCR, then "
            "restart the backend.",
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


@pytest.mark.parametrize("content_length", [str(MAX_UPLOAD_REQUEST_BYTES + 1), None])
def test_asgi_limit_rejects_oversized_body_before_multipart_parser(content_length) -> None:
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
            "body": b"x" * (MAX_UPLOAD_REQUEST_BYTES // 2 + 1),
            "more_body": True,
        },
        {
            "type": "http.request",
            "body": b"x" * (MAX_UPLOAD_REQUEST_BYTES // 2 + 1),
            "more_body": False,
        },
    ]

    async def receive():
        return chunks.pop(0)

    async def send(message):
        response_messages.append(message)

    headers = []
    if content_length is not None:
        headers.append((b"content-length", content_length.encode("ascii")))
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
    assert received_by_route is (content_length is None)
