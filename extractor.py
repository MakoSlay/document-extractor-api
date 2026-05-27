"""
OpenAI GPT-4o-mini integration for document data extraction.

Handles PDF page rendering, image encoding, prompt construction,
API communication, and response parsing with retry logic.

Uses gpt-4o-mini's vision capabilities to extract structured data
directly from document images — no OCR step needed.
"""

import base64
import io
import json
import logging
import os
import time
from typing import Optional

import fitz  # pymupdf
import httpx
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from PIL import Image

from models import ExtractionResponse

logger = logging.getLogger(__name__)

OPENAI_BASE_URL = "https://api.openai.com/v1"
MODEL_NAME = "gpt-4o-mini"

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.0

# Token pricing for gpt-4o-mini (per 1M tokens, input / output).
# Source: https://openai.com/api/pricing/
PRICE_PER_1M_INPUT_TOKENS = 0.15  # USD
PRICE_PER_1M_OUTPUT_TOKENS = 0.60  # USD

EXTRACTION_PROMPT = (
    "You are a document data extraction engine. Your sole purpose is to examine "
    "the provided document image and extract all key fields into a clean, valid "
    "JSON object. You must return ONLY raw JSON — no markdown, no backticks, "
    "no code fences, no explanations before or after. Any output that is not "
    "pure JSON will be considered a failure.\n\n"
    "Extract the following fields from this document image and return them as a "
    "single JSON object. Use null for any field you cannot find or are unsure about. "
    "Do not guess — if the value is not clearly present, set it to null.\n\n"
    "{\n"
    '  "document_type": "invoice | receipt | purchase_order | other",\n'
    '  "vendor_name": "Company Name",\n'
    '  "vendor_address": "123 Main St, City, Country",\n'
    '  "document_number": "INV-12345",\n'
    '  "date": "YYYY-MM-DD",\n'
    '  "due_date": "YYYY-MM-DD",\n'
    '  "currency": "USD",\n'
    '  "subtotal": 100.00,\n'
    '  "tax_amount": 8.50,\n'
    '  "tax_rate": 8.5,\n'
    '  "total_amount": 108.50,\n'
    '  "line_items": [\n'
    '    {"description": "Item name", "quantity": 1.0, "unit_price": 50.00, "total": 50.00}\n'
    '  ],\n'
    '  "payment_terms": "Net 30",\n'
    '  "notes": "Any additional notes"\n'
    "}\n\n"
    "Return ONLY the JSON object, nothing else."
)


def _get_client() -> OpenAI:
    """Create a configured OpenAI client pointed at the standard API."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    return OpenAI(
        api_key=api_key,
        base_url=OPENAI_BASE_URL,
        timeout=httpx.Timeout(60.0, connect=10.0),
    )


def _render_pdf_page_to_png(file_data: bytes) -> bytes:
    """
    Render the first page of a PDF to PNG bytes at 2x scale.

    After rendering, if either pixel dimension exceeds 4096px the image
    is scaled down proportionally via Pillow so the longest axis is at
    most 4096px. This prevents memory exhaustion from large-format PDFs.

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
        # Render at 2x scale for better vision quality
        matrix = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=matrix)
        pil_image = Image.open(io.BytesIO(pix.tobytes("png")))

        # Pixel budget check — scale down if either axis exceeds 4096px.
        # Large-format vector PDFs can produce multi-megapixel renders at 2x.
        if pil_image.width > 4096 or pil_image.height > 4096:
            scale = 4096 / max(pil_image.width, pil_image.height)
            new_size = (int(pil_image.width * scale), int(pil_image.height * scale))
            logger.warning(
                "PDF page oversized after 2x render (%dx%d), scaling to %dx%d",
                pil_image.width,
                pil_image.height,
                new_size[0],
                new_size[1],
            )
            pil_image = pil_image.resize(new_size, Image.LANCZOS)

        # Encode the (possibly scaled) image as PNG bytes
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as exc:
        raise ValueError(f"Failed to render PDF page to image: {exc}") from exc
    finally:
        doc.close()


def _encode_image_to_base64(file_data: bytes) -> str:
    """
    Open image bytes, apply the 4096px pixel budget cap if needed, then
    encode to a base64 PNG data URI string.

    Always re-encodes to PNG for consistent output regardless of input format.
    """
    pil_image = Image.open(io.BytesIO(file_data))

    # Apply the same pixel budget check that PDFs get — scale down if
    # either axis exceeds 4096px.
    if pil_image.width > 4096 or pil_image.height > 4096:
        scale = 4096 / max(pil_image.width, pil_image.height)
        new_size = (int(pil_image.width * scale), int(pil_image.height * scale))
        logger.warning(
            "Uploaded image oversized (%dx%d), scaling to %dx%d",
            pil_image.width,
            pil_image.height,
            new_size[0],
            new_size[1],
        )
        pil_image = pil_image.resize(new_size, Image.LANCZOS)

    # Always encode as PNG for the API (consistent, lossless)
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


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
        raise ValueError("Empty response from model after cleaning")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON: {exc}") from exc


def compute_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Compute estimated cost in USD based on gpt-4o-mini token pricing."""
    input_cost = (prompt_tokens / 1_000_000) * PRICE_PER_1M_INPUT_TOKENS
    output_cost = (completion_tokens / 1_000_000) * PRICE_PER_1M_OUTPUT_TOKENS
    return round(input_cost + output_cost, 6)


def extract_document(
    file_data: bytes,
    mime_type: str,
    request_id: str,
) -> tuple[ExtractionResponse, int, int]:
    """
    Send a document image to gpt-4o-mini for structured extraction.

    Args:
        file_data: Raw file bytes (PDF or image).
        mime_type: MIME type of the file (application/pdf, image/png, image/jpeg).
        request_id: Unique request identifier for logging.

    Returns:
        Tuple of (ExtractionResponse, prompt_tokens, completion_tokens).

    Raises ValueError after exhausting all retries if extraction or
    validation fails.
    """
    # Step 1: Prepare the base64 image payload
    if mime_type == "application/pdf":
        png_bytes = _render_pdf_page_to_png(file_data)
        base64_image = base64.b64encode(png_bytes).decode("utf-8")
        image_mime = "image/png"
    elif mime_type in ("image/png", "image/jpeg"):
        base64_image = _encode_image_to_base64(file_data)
        # Always output as PNG after potential resize
        image_mime = "image/png"
    else:
        # This branch is intentionally unreachable in production because
        # main.py filters MIME types to the allowed set before calling
        # extract_document.  It exists as a defensive guard in case this
        # function is ever called from another context.
        raise ValueError(f"Unsupported MIME type for extraction: {mime_type}")

    data_uri = f"data:{image_mime};base64,{base64_image}"

    client = _get_client()
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                "GPT-4o-mini extraction attempt %d/%d | request_id=%s",
                attempt,
                MAX_RETRIES,
                request_id,
            )

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_uri},
                            },
                            {
                                "type": "text",
                                "text": EXTRACTION_PROMPT,
                            },
                        ],
                    },
                ],
                temperature=0.0,  # Deterministic output for structured extraction
                max_tokens=4096,
            )

            # Guard against empty or missing choices (transient API glitch).
            if not response.choices:
                logger.warning(
                    "Model returned empty choices list on attempt %d/%d | request_id=%s",
                    attempt,
                    MAX_RETRIES,
                    request_id,
                )
                raise ValueError("Model returned empty choices list")

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
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as exc:
            last_error = exc
            logger.warning(
                "Transient OpenAI API failure on attempt %d/%d | request_id=%s: %s",
                attempt,
                MAX_RETRIES,
                request_id,
                exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
        except Exception as exc:
            # Unexpected errors (auth, schema, SDK misuse, etc.) — do not retry
            logger.error(
                "Unexpected API error for request_id=%s: %s",
                request_id,
                exc,
            )
            raise

    # Exhausted all retries
    raise ValueError(
        f"Extraction failed after {MAX_RETRIES} attempts: {last_error}"
    )
