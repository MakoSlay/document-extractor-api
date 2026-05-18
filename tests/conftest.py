"""
Shared fixtures for the document extractor test suite.

Fixtures in this file are auto-discovered by pytest.  Shared constants
and mock builders live in helpers.py so they can be imported by test
files directly (pytest does not allow conftest.py to be imported as a
regular module).
"""

import io
import json
import os

import fitz
import pytest
from PIL import Image

from models import ExtractionResponse


# ---------------------------------------------------------------------------
# Module-level environment setup — MUST run before any test imports main.py
# because main.py validates DEEPSEEK_API_KEY at import time.  These prevent
# accidental real API calls and set a short semaphore timeout for testing.
# ---------------------------------------------------------------------------
os.environ["DEEPSEEK_API_KEY"] = "sk-test-fake-key-for-testing"
os.environ["SEMAPHORE_TIMEOUT_SECONDS"] = "0.5"


# ---------------------------------------------------------------------------
# Pydantic model for assertion helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def valid_extraction_response():
    """A fully populated ExtractionResponse with a test request_id."""
    data = {
        "document_type": "invoice",
        "vendor_name": "Test Corp",
        "vendor_address": None,
        "document_number": "INV-2024-001",
        "date": "2024-01-15",
        "due_date": None,
        "currency": "USD",
        "subtotal": 90.0,
        "tax_amount": 10.0,
        "tax_rate": 11.11,
        "total_amount": 100.0,
        "line_items": [
            {
                "description": "Widget A",
                "quantity": 2.0,
                "unit_price": 45.0,
                "total": 90.0,
            }
        ],
        "payment_terms": "Net 30",
        "notes": None,
    }
    data["request_id"] = "test-request-id"
    return ExtractionResponse.model_validate(data)


# ---------------------------------------------------------------------------
# File fixtures — minimal valid PDF, PNG, JPEG, and edge-case PDFs
# ---------------------------------------------------------------------------
@pytest.fixture
def minimal_pdf_bytes():
    """A minimal valid single-page PDF with visible text for OCR."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((50, 50), "INVOICE", fontsize=20)
    page.insert_text((50, 80), "Vendor: Test Corp", fontsize=12)
    page.insert_text((50, 100), "Total: $100.00", fontsize=12)
    page.insert_text((50, 120), "Date: 2024-01-15", fontsize=12)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


@pytest.fixture
def minimal_png_bytes():
    """A small PNG image with visible text for OCR."""
    from PIL import ImageDraw, ImageFont
    img = Image.new("RGB", (400, 200), color="white")
    draw = ImageDraw.Draw(img)
    # Use default font
    draw.text((20, 20), "INVOICE", fill="black")
    draw.text((20, 50), "Vendor: Test Corp", fill="black")
    draw.text((20, 80), "Total: $100.00", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def minimal_jpg_bytes():
    """A small JPEG image with visible text for OCR."""
    from PIL import ImageDraw
    img = Image.new("RGB", (400, 200), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "INVOICE", fill="black")
    draw.text((20, 50), "Vendor: Test Corp", fill="black")
    draw.text((20, 80), "Total: $100.00", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def empty_pdf_bytes():
    """A valid PDF structure with zero pages (Count 0)."""
    # Hand-crafted minimal PDF — pymupdf won't .save() a zero-page doc.
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Count 0/Kids[]>>endobj\n"
        b"xref\n"
        b"0 3\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"trailer<</Size 3/Root 1 0 R>>\n"
        b"startxref\n"
        b"108\n"
        b"%%EOF\n"
    )


@pytest.fixture
def corrupt_pdf_bytes():
    """Bytes that are not a valid PDF at all."""
    return b"this is not a pdf file at all just garbage"
