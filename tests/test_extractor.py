"""
Unit tests for extractor.py.

Mocks the openai.OpenAI client at the chat.completions.create level
to test the full extraction pipeline (OCR → DeepSeek → JSON parsing)
without real API calls.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

import extractor
from .helpers import (
    BOM_EXTRACTION_JSON,
    EMPTY_JSON,
    MALFORMED_JSON,
    VALID_EXTRACTION_DICT,
    VALID_EXTRACTION_JSON,
    make_mock_openai_response,
)
from models import ExtractionResponse


# ---------------------------------------------------------------------------
# Helper to install a mock DeepSeek client
# ---------------------------------------------------------------------------
def _install_mock_client(monkeypatch, mock_response):
    """Replace _get_client() so it returns a mock whose create() yields mock_response."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    def fake_get_client():
        return mock_client

    monkeypatch.setattr(extractor, "_get_client", fake_get_client)
    return mock_client


# ---------------------------------------------------------------------------
# Tests: successful extraction (PDF and image)
# ---------------------------------------------------------------------------
def test_extract_pdf_success(minimal_pdf_bytes, monkeypatch):
    """Happy path: PDF → OCR → text → valid JSON → ExtractionResponse."""
    mock_resp = make_mock_openai_response(VALID_EXTRACTION_JSON)
    _install_mock_client(monkeypatch, mock_resp)

    result, prompt_tokens, completion_tokens = extractor.extract_document(
        file_data=minimal_pdf_bytes,
        mime_type="application/pdf",
        request_id="test-pdf-001",
    )

    assert isinstance(result, ExtractionResponse)
    assert result.request_id == "test-pdf-001"
    assert result.document_type == "invoice"
    assert result.vendor_name == "Test Corp"
    assert result.total_amount == 100.0
    assert len(result.line_items) == 1
    assert result.line_items[0].description == "Widget A"
    assert prompt_tokens == 500
    assert completion_tokens == 200
    # Confidence: 9 of 12 optional fields populated = 0.75
    # (vendor_name, document_number, date, currency, subtotal, tax_amount,
    #  tax_rate, line_items, payment_terms are all non-null/non-empty)
    assert result.confidence_score == pytest.approx(0.75, abs=0.001)


def test_extract_png_success(minimal_png_bytes, monkeypatch):
    """Happy path: PNG image → OCR → text → valid JSON."""
    mock_resp = make_mock_openai_response(VALID_EXTRACTION_JSON)
    _install_mock_client(monkeypatch, mock_resp)

    result, pt, ct = extractor.extract_document(
        file_data=minimal_png_bytes,
        mime_type="image/png",
        request_id="test-png-001",
    )

    assert result.document_type == "invoice"
    assert pt == 500
    assert ct == 200


def test_extract_jpeg_success(minimal_jpg_bytes, monkeypatch):
    """Happy path: JPEG image → OCR → text → valid JSON."""
    mock_resp = make_mock_openai_response(VALID_EXTRACTION_JSON)
    _install_mock_client(monkeypatch, mock_resp)

    result, pt, ct = extractor.extract_document(
        file_data=minimal_jpg_bytes,
        mime_type="image/jpeg",
        request_id="test-jpeg-001",
    )

    assert result.document_type == "invoice"


# ---------------------------------------------------------------------------
# Tests: retry logic
# ---------------------------------------------------------------------------
def test_retry_on_malformed_json_then_succeed(minimal_pdf_bytes, monkeypatch):
    """First attempt returns bad JSON, second returns valid — should succeed."""
    bad_resp = make_mock_openai_response(MALFORMED_JSON)
    good_resp = make_mock_openai_response(VALID_EXTRACTION_JSON)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [bad_resp, good_resp]

    monkeypatch.setattr(extractor, "_get_client", lambda: mock_client)

    result, pt, ct = extractor.extract_document(
        file_data=minimal_pdf_bytes,
        mime_type="application/pdf",
        request_id="test-retry-001",
    )

    assert result.document_type == "invoice"
    assert mock_client.chat.completions.create.call_count == 2


def test_retry_exhaustion_raises(minimal_pdf_bytes, monkeypatch):
    """All 3 attempts return malformed JSON → ValueError raised."""
    bad_resp = make_mock_openai_response(MALFORMED_JSON)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [bad_resp, bad_resp, bad_resp]

    monkeypatch.setattr(extractor, "_get_client", lambda: mock_client)

    with pytest.raises(ValueError, match="Extraction failed after 3 attempts"):
        extractor.extract_document(
            file_data=minimal_pdf_bytes,
            mime_type="application/pdf",
            request_id="test-exhaust-001",
        )

    assert mock_client.chat.completions.create.call_count == 3


def test_retry_on_empty_string_response(minimal_pdf_bytes, monkeypatch):
    """Empty string response is retried, then succeeds."""
    empty_resp = make_mock_openai_response(EMPTY_JSON)
    good_resp = make_mock_openai_response(VALID_EXTRACTION_JSON)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [empty_resp, good_resp]

    monkeypatch.setattr(extractor, "_get_client", lambda: mock_client)

    result, pt, ct = extractor.extract_document(
        file_data=minimal_pdf_bytes,
        mime_type="application/pdf",
        request_id="test-empty-001",
    )

    assert result.document_type == "invoice"
    assert mock_client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# Tests: empty choices guard
# ---------------------------------------------------------------------------
def test_empty_choices_triggers_retry(minimal_pdf_bytes, monkeypatch):
    """Empty choices list is treated as retryable, not an unexpected error."""
    empty_choices_resp = make_mock_openai_response("", empty_choices=True)
    good_resp = make_mock_openai_response(VALID_EXTRACTION_JSON)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [empty_choices_resp, good_resp]

    monkeypatch.setattr(extractor, "_get_client", lambda: mock_client)

    result, pt, ct = extractor.extract_document(
        file_data=minimal_pdf_bytes,
        mime_type="application/pdf",
        request_id="test-empty-choices-001",
    )

    assert result.document_type == "invoice"
    assert mock_client.chat.completions.create.call_count == 2


def test_empty_choices_all_retries_exhausted(minimal_pdf_bytes, monkeypatch):
    """All 3 attempts return empty choices → ValueError."""
    empty_resp = make_mock_openai_response("", empty_choices=True)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [empty_resp, empty_resp, empty_resp]

    monkeypatch.setattr(extractor, "_get_client", lambda: mock_client)

    with pytest.raises(ValueError, match="Extraction failed after 3 attempts"):
        extractor.extract_document(
            file_data=minimal_pdf_bytes,
            mime_type="application/pdf",
            request_id="test-empty-choices-exhaust-001",
        )

    assert mock_client.chat.completions.create.call_count == 3


# ---------------------------------------------------------------------------
# Tests: BOM stripping
# ---------------------------------------------------------------------------
def test_bom_stripping(minimal_pdf_bytes, monkeypatch):
    """Response with UTF-8 BOM + whitespace is parsed correctly."""
    bom_resp = make_mock_openai_response(BOM_EXTRACTION_JSON)
    _install_mock_client(monkeypatch, bom_resp)

    result, pt, ct = extractor.extract_document(
        file_data=minimal_pdf_bytes,
        mime_type="application/pdf",
        request_id="test-bom-001",
    )

    assert result.document_type == "invoice"
    assert result.vendor_name == "Test Corp"


# ---------------------------------------------------------------------------
# Tests: unexpected API errors (no retry)
# ---------------------------------------------------------------------------
def test_unexpected_api_error_no_retry(minimal_pdf_bytes, monkeypatch):
    """Network/auth errors raise immediately without retry."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = ConnectionError("Network down")

    monkeypatch.setattr(extractor, "_get_client", lambda: mock_client)

    with pytest.raises(ConnectionError, match="Network down"):
        extractor.extract_document(
            file_data=minimal_pdf_bytes,
            mime_type="application/pdf",
            request_id="test-network-001",
        )

    assert mock_client.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# Tests: PDF text extraction (OCR)
# ---------------------------------------------------------------------------
def test_corrupt_pdf_raises_valueerror():
    """Bytes that are not a valid PDF → ValueError on extraction."""
    with pytest.raises(ValueError, match="Failed to open PDF"):
        extractor._extract_text_from_pdf(b"not a pdf")


def test_empty_pdf_raises_valueerror(empty_pdf_bytes):
    """PDF with zero pages → ValueError."""
    with pytest.raises(ValueError, match="PDF contains no pages"):
        extractor._extract_text_from_pdf(empty_pdf_bytes)


def test_pdf_ocr_produces_text(minimal_pdf_bytes):
    """A valid PDF with text content → OCR returns non-empty string."""
    text = extractor._extract_text_from_pdf(minimal_pdf_bytes)
    # The minimal PDF from conftest has no visible text, so OCR may return
    # empty.  We just verify no exception is raised.
    assert isinstance(text, str)


def test_image_ocr_produces_text(minimal_png_bytes):
    """A PNG image → OCR runs without error."""
    text = extractor._extract_text_from_image(minimal_png_bytes)
    assert isinstance(text, str)


# ---------------------------------------------------------------------------
# Tests: OCR on blank document
# ---------------------------------------------------------------------------
def test_blank_document_rejected(monkeypatch):
    """If OCR returns empty text, extraction raises ValueError."""
    import extractor as ex

    # Create a tiny white image that OCR will return empty for
    from PIL import Image as PILImage
    import io as _io
    from unittest.mock import MagicMock

    img = PILImage.new("RGB", (10, 10), color="white")
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    with pytest.raises(ValueError, match="OCR produced no text"):
        ex.extract_document(
            file_data=png_bytes,
            mime_type="image/png",
            request_id="test-blank-001",
        )


# ---------------------------------------------------------------------------
# Tests: compute_cost
# ---------------------------------------------------------------------------
def test_compute_cost():
    """Cost calculation matches DeepSeek pricing."""
    cost = extractor.compute_cost(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost == pytest.approx(1.37, abs=0.001)

    cost_zero = extractor.compute_cost(prompt_tokens=0, completion_tokens=0)
    assert cost_zero == 0.0


# ---------------------------------------------------------------------------
# Tests: parse_json_response
# ---------------------------------------------------------------------------
def test_parse_valid_json():
    parsed = extractor._parse_json_response(
        '{"key": "value"}', "test-parse-001"
    )
    assert parsed == {"key": "value"}


def test_parse_empty_string():
    with pytest.raises(ValueError, match="Empty response"):
        extractor._parse_json_response("", "test-parse-002")


def test_parse_bom_string():
    parsed = extractor._parse_json_response(
        '\ufeff  {"key": "value"}  ', "test-parse-003"
    )
    assert parsed == {"key": "value"}


def test_parse_malformed_json():
    with pytest.raises(ValueError, match="invalid JSON"):
        extractor._parse_json_response("{bad json}", "test-parse-004")


# ---------------------------------------------------------------------------
# Test: unsupported MIME type (defensive guard)
# ---------------------------------------------------------------------------
def test_unsupported_mime_type(minimal_png_bytes):
    """Defensive guard raises ValueError for unsupported MIME types."""
    with pytest.raises(ValueError, match="Unsupported MIME type"):
        extractor.extract_document(
            file_data=minimal_png_bytes,
            mime_type="image/gif",
            request_id="test-bad-mime",
        )
