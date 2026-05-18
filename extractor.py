"""
DeepSeek API integration for document data extraction.

Pipeline: render PDF/images to pixel data → OCR with pytesseract →
send extracted text to DeepSeek for structured JSON extraction.

DeepSeek's API is text-only, so we use OCR as the bridge from
visual documents to the LLM.
"""

import io
import json
import logging
import os
import time
from typing import Optional

import fitz  # pymupdf
import pytesseract
from openai import OpenAI
from PIL import Image

from models import ExtractionResponse

logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.0

# Token pricing for deepseek-chat (per 1M tokens, input / output).
# Source: https://api-docs.deepseek.com/quick_start/pricing
PRICE_PER_1M_PROMPT_TOKENS = 0.27  # USD (cache miss)
PRICE_PER_1M_COMPLETION_TOKENS = 1.10  # USD

EXTRACTION_SYSTEM_PROMPT = (
    "You are a document data extraction engine. Your sole purpose is to examine "
    "the provided OCR-extracted text from a document and extract all key fields "
    "into a clean, valid JSON object. You must return ONLY raw JSON — no "
    "markdown, no backticks, no code fences, no explanations before or after. "
    "Any output that is not pure JSON will be considered a failure."
)

EXTRACTION_USER_PROMPT_TEMPLATE = (
    "Extract the following fields from this OCR text of a document and return "
    "them as a single JSON object. Use null for any field you cannot find or "
    "are unsure about. Do not guess — if the value is not clearly present, "
    "set it to null.\n\n"
    "Fields to extract:\n"
    '  "document_type": "invoice | receipt | purchase_order | other"\n'
    '  "vendor_name": company or individual who issued the document\n'
    '  "vendor_address": full address of the vendor\n'
    '  "document_number": invoice number, receipt number, or PO number\n'
    '  "date": document date as YYYY-MM-DD\n'
    '  "due_date": payment due date as YYYY-MM-DD if applicable\n'
    '  "currency": currency code like USD, EUR, GBP\n'
    '  "subtotal": amount before tax\n'
    '  "tax_amount": tax amount\n'
    '  "tax_rate": tax rate as a percentage (e.g. 8.5)\n'
    '  "total_amount": final total\n'
    '  "line_items": [{{"description": "...", "quantity": 1.0, "unit_price": 50.00, "total": 50.00}}]\n'
    '  "payment_terms": e.g. Net 30, Due on Receipt\n'
    '  "notes": any additional notes or comments\n\n'
    "OCR-extracted document text:\n"
    "{ocr_text}\n\n"
    "Return ONLY the JSON object, nothing else."
)


def _get_client() -> OpenAI:
    """Create a configured OpenAI client pointed at the DeepSeek API."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def _ocr_image(pil_image: Image.Image) -> str:
    """Run pytesseract OCR on a PIL Image and return the extracted text."""
    try:
        text = pytesseract.image_to_string(pil_image)
        return text.strip()
    except Exception as exc:
        raise ValueError(f"OCR failed: {exc}") from exc


def _extract_text_from_pdf(file_data: bytes) -> str:
    """
    Render the first page of a PDF to an image at 2x scale and run OCR.

    Raises ValueError if the PDF cannot be opened, has no pages, or
    rendering fails.
    """
    try:
        doc = fitz.open(stream=file_data, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Failed to open PDF: {exc}") from exc

    if doc.page_count == 0:
        doc.close()
        raise ValueError("PDF contains no pages")

    try:
        page = doc.load_page(0)
        # Render at 2x scale for better OCR accuracy
        matrix = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=matrix)
        pil_image = Image.open(io.BytesIO(pix.tobytes("png")))
    except Exception as exc:
        raise ValueError(f"Failed to render PDF page to image: {exc}") from exc
    finally:
        doc.close()

    return _ocr_image(pil_image)


def _extract_text_from_image(file_data: bytes) -> str:
    """Run OCR directly on image bytes (PNG or JPEG)."""
    try:
        pil_image = Image.open(io.BytesIO(file_data))
    except Exception as exc:
        raise ValueError(f"Failed to open image: {exc}") from exc
    return _ocr_image(pil_image)


def _strip_bom_and_whitespace(text: str) -> str:
    """
    Remove leading BOM characters and all surrounding whitespace.

    The BOM character u'\ufeff' is not considered whitespace by Python's
    str.strip(), so we explicitly lstrip it first.
    """
    return text.strip().lstrip("\ufeff").strip()


def _parse_json_response(raw_text: str, request_id: str) -> dict:
    """
    Parse a raw LLM response string into a dict.

    Strips BOM/whitespace before attempting JSON parse.
    Raises ValueError if the text cannot be parsed as JSON.
    """
    cleaned = _strip_bom_and_whitespace(raw_text)
    if not cleaned:
        raise ValueError("Empty response from DeepSeek after cleaning")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"DeepSeek returned invalid JSON: {exc}") from exc


def compute_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Compute estimated cost in USD based on DeepSeek token pricing."""
    prompt_cost = (prompt_tokens / 1_000_000) * PRICE_PER_1M_PROMPT_TOKENS
    completion_cost = (completion_tokens / 1_000_000) * PRICE_PER_1M_COMPLETION_TOKENS
    return round(prompt_cost + completion_cost, 6)


def extract_document(
    file_data: bytes,
    mime_type: str,
    request_id: str,
) -> tuple[ExtractionResponse, int, int]:
    """
    OCR the document then send extracted text to DeepSeek for structuring.

    Args:
        file_data: Raw file bytes (PDF or image).
        mime_type: MIME type of the file (application/pdf, image/png, image/jpeg).
        request_id: Unique request identifier for logging.

    Returns:
        Tuple of (ExtractionResponse, prompt_tokens, completion_tokens).

    Raises ValueError after exhausting all retries if extraction or
    validation fails.
    """
    # Step 1: OCR the document
    try:
        if mime_type == "application/pdf":
            ocr_text = _extract_text_from_pdf(file_data)
        elif mime_type in ("image/png", "image/jpeg"):
            ocr_text = _extract_text_from_image(file_data)
        else:
            # This branch is intentionally unreachable in production because
            # main.py filters MIME types to the allowed set before calling
            # extract_document.  It exists as a defensive guard in case this
            # function is ever called from another context.
            raise ValueError(f"Unsupported MIME type for extraction: {mime_type}")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Document OCR failed: {exc}") from exc

    if not ocr_text:
        raise ValueError("OCR produced no text — document may be blank or unreadable")

    logger.info(
        "OCR complete | request_id=%s | ocr_chars=%d",
        request_id,
        len(ocr_text),
    )

    # Step 2: Build the extraction prompt with the OCR text
    prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(ocr_text=ocr_text)

    client = _get_client()
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                "DeepSeek extraction attempt %d/%d | request_id=%s",
                attempt,
                MAX_RETRIES,
                request_id,
            )

            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,  # Deterministic output for structured extraction
                max_tokens=4096,
            )

            # Guard against empty or missing choices (transient API glitch).
            if not response.choices:
                logger.warning(
                    "DeepSeek returned empty choices list on attempt %d/%d | request_id=%s",
                    attempt,
                    MAX_RETRIES,
                    request_id,
                )
                raise ValueError("DeepSeek returned empty choices list")

            raw_content = response.choices[0].message.content or ""
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0

            # Parse the JSON response
            parsed = _parse_json_response(raw_content, request_id)

            # Inject our request_id into the parsed data
            parsed["request_id"] = request_id

            # Validate against the Pydantic schema
            extraction = ExtractionResponse.model_validate(parsed)
            return extraction, prompt_tokens, completion_tokens

        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "Attempt %d/%d failed for request_id=%s: %s",
                attempt,
                MAX_RETRIES,
                request_id,
                exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
        except Exception as exc:
            # Unexpected errors (network, auth, etc.) — do not retry
            logger.error(
                "Unexpected DeepSeek API error for request_id=%s: %s",
                request_id,
                exc,
            )
            raise

    # Exhausted all retries
    raise ValueError(
        f"Extraction failed after {MAX_RETRIES} attempts: {last_error}"
    )
