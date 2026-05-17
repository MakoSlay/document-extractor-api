"""
Integration tests for main.py.

Uses FastAPI TestClient and mocks extract_document at the function level
to test the full HTTP layer: validation, error codes, caching, and the
semaphore gate — without real DeepSeek calls.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import main
from models import ExtractionResponse


# ---------------------------------------------------------------------------
# TestClient fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def client():
    """Return a TestClient bound to the FastAPI app with a clean cache."""
    main._cache.clear()
    return TestClient(main.app)


# ---------------------------------------------------------------------------
# Cache clearing — every test starts fresh
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear the in-memory cache and reset semaphore state between tests."""
    main._cache.clear()

    # Reset semaphore — release any leaked acquires from prior tests.
    # _value tracks available permits; restore to the initial count of 5.
    while main._semaphore._value < 5:
        main._semaphore._value += 1
    # _waiters is None when empty in some Python versions, a deque otherwise.
    waiters = main._semaphore._waiters
    if waiters is not None:
        waiters.clear()

    yield

    main._cache.clear()
    while main._semaphore._value < 5:
        main._semaphore._value += 1
    waiters = main._semaphore._waiters
    if waiters is not None:
        waiters.clear()


# ---------------------------------------------------------------------------
# Mock helper for extract_document
# ---------------------------------------------------------------------------
def _install_mock_extract(monkeypatch, return_value=None, side_effect=None):
    """
    Replace main.extract_document with a mock.

    If return_value is provided, the mock returns it on every call.
    If side_effect is provided, the mock raises/returns per call.
    """
    mock_fn = MagicMock()
    if side_effect is not None:
        mock_fn.side_effect = side_effect
    elif return_value is not None:
        mock_fn.return_value = return_value

    monkeypatch.setattr(main, "extract_document", mock_fn)
    return mock_fn


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# 400 — Invalid file extension
# ---------------------------------------------------------------------------
def test_invalid_extension(client):
    response = client.post(
        "/extract",
        files={"file": ("document.txt", b"some text content", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 400 — No filename
# ---------------------------------------------------------------------------
def test_no_filename(client):
    """Upload with no filename — FastAPI validates this as 422 before our handler runs."""
    response = client.post(
        "/extract",
        files={"file": ("", b"content", "application/pdf")},
    )
    # FastAPI's form validation catches the empty filename and returns 422
    # before our handler's "No filename provided" check fires.
    assert response.status_code in (400, 422)


# ---------------------------------------------------------------------------
# 400 — Magic byte / MIME mismatch
# ---------------------------------------------------------------------------
def test_mime_mismatch(client):
    """File with .pdf extension but actually a text file → extension-MIME mismatch 400."""
    response = client.post(
        "/extract",
        files={"file": ("fake.pdf", b"This is plain text not a PDF", "application/pdf")},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "does not match" in detail


# ---------------------------------------------------------------------------
# 400 — Content-Length header exceeds limit (middleware)
# ---------------------------------------------------------------------------
def test_content_length_exceeds_limit(client):
    """
    The Content-Length middleware rejects the request before the body
    is read. We send a plain POST (no file) with a huge Content-Length
    to trigger it. The actual body is small, but the header signals
    an oversized upload.
    """
    response = client.post(
        "/extract",
        content=b"x",
        headers={"Content-Length": str(11 * 1024 * 1024)},
    )
    # The middleware should reject with 400 before the endpoint handler runs.
    # httpx may override Content-Length for multipart uploads, so we use a
    # plain body here to ensure the header passes through.
    assert response.status_code == 400
    assert "10 MB" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 400 — Empty file
# ---------------------------------------------------------------------------
def test_empty_file(client):
    response = client.post(
        "/extract",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 400 — Corrupt PDF
# ---------------------------------------------------------------------------
def test_corrupt_pdf(client):
    """Garbage bytes with .pdf extension → caught by magic bytes as text/plain."""
    response = client.post(
        "/extract",
        files={"file": ("corrupt.pdf", b"this is not a pdf", "application/pdf")},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    # The magic-byte check fires before the PDF structural check, so we get
    # the MIME mismatch message rather than the "could not be read" message.
    assert "does not match" in detail.lower()


# ---------------------------------------------------------------------------
# 400 — Empty PDF (zero pages)
# ---------------------------------------------------------------------------
def test_empty_pdf(client, empty_pdf_bytes):
    response = client.post(
        "/extract",
        files={"file": ("empty.pdf", empty_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 400
    assert "no pages" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 503 — Semaphore busy (all slots occupied)
# ---------------------------------------------------------------------------
def test_semaphore_busy(client, minimal_png_bytes, monkeypatch):
    """When the semaphore is fully occupied, return 503."""

    async def fake_acquire_busy():
        raise asyncio.TimeoutError()

    monkeypatch.setattr(main._semaphore, "acquire", fake_acquire_busy)

    response = client.post(
        "/extract",
        files={"file": ("test.png", minimal_png_bytes, "image/png")},
    )
    assert response.status_code == 503
    assert "busy" in response.json()["detail"].lower()


def test_semaphore_acquire_unexpected_error(client, minimal_png_bytes, monkeypatch):
    """If semaphore acquire fails for an unexpected reason, still return 503."""

    async def fake_acquire_error():
        raise RuntimeError("Event loop is shutting down")

    monkeypatch.setattr(main._semaphore, "acquire", fake_acquire_error)

    response = client.post(
        "/extract",
        files={"file": ("test.png", minimal_png_bytes, "image/png")},
    )
    assert response.status_code == 503
    assert "busy" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 422 — Extraction validation failure
# ---------------------------------------------------------------------------
def test_extraction_value_error_422(client, minimal_pdf_bytes, monkeypatch):
    """extract_document raises ValueError → 422 Unprocessable Entity."""
    _install_mock_extract(
        monkeypatch,
        side_effect=ValueError("Failed to parse document: missing total_amount"),
    )

    response = client.post(
        "/extract",
        files={"file": ("doc.pdf", minimal_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 422
    assert "total_amount" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 500 — Unexpected extraction error
# ---------------------------------------------------------------------------
def test_extraction_unexpected_error_500(client, minimal_pdf_bytes, monkeypatch):
    """extract_document raises an unexpected Exception → 500 Internal Error."""
    _install_mock_extract(
        monkeypatch,
        side_effect=RuntimeError("DeepSeek API timed out after 30s"),
    )

    response = client.post(
        "/extract",
        files={"file": ("doc.pdf", minimal_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 500
    detail = response.json()["detail"]
    # Must not expose the internal error message
    assert "timed out" not in detail.lower()
    assert "unexpected" in detail.lower()


# ---------------------------------------------------------------------------
# 200 — Successful extraction
# ---------------------------------------------------------------------------
def test_successful_extraction(client, minimal_pdf_bytes, monkeypatch):
    """Happy path: PDF upload → mocked extraction → 200 with structured JSON."""
    result_obj = ExtractionResponse.model_validate({
        "request_id": "will-be-replaced",
        "document_type": "invoice",
        "vendor_name": "Happy Path Corp",
        "total_amount": 250.00,
    })
    mock_fn = _install_mock_extract(
        monkeypatch,
        return_value=(result_obj, 600, 300),
    )

    response = client.post(
        "/extract",
        files={"file": ("invoice.pdf", minimal_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["document_type"] == "invoice"
    assert data["vendor_name"] == "Happy Path Corp"
    assert data["total_amount"] == 250.00
    # The mocked extract_document returns the object as-is, so the request_id
    # from the mock is preserved.  In production, extract_document injects the
    # real request_id — this test verifies the HTTP layer, not the injector.
    assert data["request_id"] == "will-be-replaced"

    # Confidence: vendor_name and the default currency="USD" are populated.
    # 2 of 12 optional fields = 0.1667
    assert data["confidence_score"] == pytest.approx(0.1667, abs=0.01)

    # extract_document should have been called exactly once
    mock_fn.assert_called_once()


# ---------------------------------------------------------------------------
# Cache hit — second identical request returns cached result
# ---------------------------------------------------------------------------
def test_cache_hit(client, minimal_pdf_bytes, monkeypatch):
    """Two identical uploads → second returns cached result without calling extract."""
    result_obj = ExtractionResponse.model_validate({
        "request_id": "cached-req",
        "document_type": "receipt",
        "vendor_name": "Cache Store",
        "total_amount": 50.00,
    })
    mock_fn = _install_mock_extract(
        monkeypatch,
        return_value=(result_obj, 400, 150),
    )

    # First request — extraction happens
    resp1 = client.post(
        "/extract",
        files={"file": ("receipt.pdf", minimal_pdf_bytes, "application/pdf")},
    )
    assert resp1.status_code == 200
    assert mock_fn.call_count == 1

    # Second request with the same file — cache hit
    resp2 = client.post(
        "/extract",
        files={"file": ("receipt.pdf", minimal_pdf_bytes, "application/pdf")},
    )
    assert resp2.status_code == 200
    # extract_document should NOT have been called again
    assert mock_fn.call_count == 1

    data2 = resp2.json()
    assert data2["document_type"] == "receipt"
    assert data2["vendor_name"] == "Cache Store"


def test_cache_hit_different_request_id(client, minimal_pdf_bytes, monkeypatch):
    """Cache hit returns the cached data but with a fresh request_id."""
    result_obj = ExtractionResponse.model_validate({
        "request_id": "original-id",
        "document_type": "receipt",
        "vendor_name": "Cache Store",
        "total_amount": 50.00,
    })
    _install_mock_extract(monkeypatch, return_value=(result_obj, 400, 150))

    resp1 = client.post(
        "/extract",
        files={"file": ("r.pdf", minimal_pdf_bytes, "application/pdf")},
    )
    req_id_1 = resp1.json()["request_id"]

    resp2 = client.post(
        "/extract",
        files={"file": ("r.pdf", minimal_pdf_bytes, "application/pdf")},
    )
    req_id_2 = resp2.json()["request_id"]

    # Same content → cache hit, but each request gets a unique request_id
    assert req_id_1 != req_id_2
    assert len(req_id_2) == 36


# ---------------------------------------------------------------------------
# 200 — PNG image extraction
# ---------------------------------------------------------------------------
def test_successful_png_extraction(client, minimal_png_bytes, monkeypatch):
    """Happy path: PNG upload → mocked extraction → 200."""
    result_obj = ExtractionResponse.model_validate({
        "request_id": "will-be-replaced",
        "document_type": "receipt",
        "total_amount": 12.50,
    })
    _install_mock_extract(monkeypatch, return_value=(result_obj, 300, 100))

    response = client.post(
        "/extract",
        files={"file": ("receipt.png", minimal_png_bytes, "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["document_type"] == "receipt"


# ---------------------------------------------------------------------------
# 400 — JPEG with wrong extension
# ---------------------------------------------------------------------------
def test_jpeg_with_wrong_extension_rejected(client, minimal_jpg_bytes):
    """JPEG content uploaded as .pdf → extension-MIME mismatch 400."""
    response = client.post(
        "/extract",
        files={"file": ("document.pdf", minimal_jpg_bytes, "application/pdf")},
    )
    assert response.status_code == 400
    assert "does not match" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 200 — JPEG extraction
# ---------------------------------------------------------------------------
def test_successful_jpeg_extraction(client, minimal_jpg_bytes, monkeypatch):
    """JPEG upload works correctly."""
    result_obj = ExtractionResponse.model_validate({
        "request_id": "will-be-replaced",
        "document_type": "receipt",
        "total_amount": 5.00,
    })
    _install_mock_extract(monkeypatch, return_value=(result_obj, 300, 100))

    response = client.post(
        "/extract",
        files={"file": ("photo.jpeg", minimal_jpg_bytes, "image/jpeg")},
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# 400 — Invalid extension (no dot)
# ---------------------------------------------------------------------------
def test_no_extension(client):
    """Filename with no extension at all should be rejected."""
    response = client.post(
        "/extract",
        files={"file": ("justafile", b"content", "application/pdf")},
    )
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 503 — Semaphore released after successful request
# ---------------------------------------------------------------------------
def test_semaphore_released_after_success(client, minimal_png_bytes, monkeypatch):
    """After a successful request, the semaphore slot is released."""
    result_obj = ExtractionResponse.model_validate({
        "request_id": "x",
        "document_type": "receipt",
        "total_amount": 1.00,
    })
    _install_mock_extract(monkeypatch, return_value=(result_obj, 100, 50))

    initial_value = main._semaphore._value

    response = client.post(
        "/extract",
        files={"file": ("x.png", minimal_png_bytes, "image/png")},
    )
    assert response.status_code == 200
    # Semaphore should be back to its initial value (slot released)
    assert main._semaphore._value == initial_value


def test_semaphore_released_after_error(client, minimal_pdf_bytes, monkeypatch):
    """After a failed request, the semaphore slot is still released."""
    _install_mock_extract(
        monkeypatch,
        side_effect=ValueError("bad"),
    )

    initial_value = main._semaphore._value

    response = client.post(
        "/extract",
        files={"file": ("x.pdf", minimal_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 422
    assert main._semaphore._value == initial_value
