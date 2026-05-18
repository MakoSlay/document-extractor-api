"""
FastAPI application for document extraction via GPT-4o-mini.

Provides a /extract endpoint that accepts PDF/image uploads, validates them,
caches results by content hash, and returns structured JSON via AI vision.
"""

import asyncio
import hashlib
import logging
import os
import time
import uuid
from collections import OrderedDict
from typing import Optional

import fitz  # pymupdf — used for PDF validation
import magic
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from extractor import compute_cost, extract_document
from models import ExtractionResponse

# ---------------------------------------------------------------------------
# Load environment variables from .env file (does not override existing)
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Logging configuration — consistent format, never use print()
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Startup validation — fail fast if OPENAI_API_KEY is missing
# ---------------------------------------------------------------------------
_api_key = os.environ.get("OPENAI_API_KEY", "")
if not _api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is missing or empty. "
        "Set it in your .env file or environment before starting the server."
    )

# ---------------------------------------------------------------------------
# Validation constants — must be defined before middleware that references them
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}
ALLOWED_MIME_TYPES = {"application/pdf", "image/png", "image/jpeg"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Document Extractor API",
    description="Upload PDF or image documents and extract structured data via AI.",
    version="1.0.0",
)

# CORS — allow all origins for RapidAPI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Content-Length middleware — reject oversized uploads before reading body
# ---------------------------------------------------------------------------
@app.middleware("http")
async def content_length_middleware(request: Request, call_next):
    """Check Content-Length header before the request body is consumed."""
    if request.url.path == "/extract" and request.method == "POST":
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_FILE_SIZE_BYTES:
                    logger.warning(
                        "Content-Length exceeds limit | content_length=%s",
                        content_length,
                    )
                    return JSONResponse(
                        status_code=400,
                        content={"detail": "File size exceeds the 10 MB limit"},
                    )
            except ValueError:
                pass  # Malformed header — let the body-read check handle it
    return await call_next(request)

# ---------------------------------------------------------------------------
# Concurrency control — limit to 5 simultaneous extraction requests
# ---------------------------------------------------------------------------
_semaphore = asyncio.Semaphore(5)
_semaphore_timeout = float(os.environ.get("SEMAPHORE_TIMEOUT_SECONDS", "30"))

# ---------------------------------------------------------------------------
# Response cache — keyed by MD5 hash of file content, with LRU eviction
# ---------------------------------------------------------------------------
_cache_max_size = int(os.environ.get("CACHE_MAX_SIZE", "100"))
_cache: OrderedDict[str, ExtractionResponse] = OrderedDict()
_cache_lock = asyncio.Lock()


async def _cache_get(file_hash: str) -> Optional[ExtractionResponse]:
    """Retrieve a cached extraction result by file hash. Returns None on miss."""
    async with _cache_lock:
        return _cache.get(file_hash)


async def _cache_set(file_hash: str, result: ExtractionResponse) -> None:
    """Store an extraction result in the cache, evicting oldest if at capacity."""
    async with _cache_lock:
        if file_hash in _cache:
            # Move to end (most recently used)
            _cache.move_to_end(file_hash)
            _cache[file_hash] = result
        else:
            if len(_cache) >= _cache_max_size:
                # Evict the oldest entry (first item in OrderedDict)
                evicted = _cache.popitem(last=False)
                logger.debug("Cache evicted entry with hash=%s", evicted[0])
            _cache[file_hash] = result


def _hash_content(data: bytes) -> str:
    """Return the MD5 hex digest of the given bytes."""
    return hashlib.md5(data).hexdigest()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractionResponse)
async def extract(file: UploadFile = File(...)):
    """
    Upload a PDF or image document and receive structured extraction JSON.

    The file is validated by extension, magic bytes, and size before
    being sent to the AI model for data extraction.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info("Request started | request_id=%s | filename=%s", request_id, file.filename)

    # ---- Step 1: File extension validation ----
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning(
            "Invalid extension | request_id=%s | filename=%s | ext=%s",
            request_id,
            file.filename,
            ext,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '.{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # ---- Step 2: Read file content ----
    file_data = await file.read()

    # ---- Step 3: File size validation ----
    file_size = len(file_data)
    if file_size > MAX_FILE_SIZE_BYTES:
        logger.warning(
            "File too large | request_id=%s | filename=%s | size=%d",
            request_id,
            file.filename,
            file_size,
        )
        raise HTTPException(
            status_code=400,
            detail=f"File size ({file_size / 1024 / 1024:.1f} MB) exceeds the 10 MB limit",
        )
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # ---- Step 4: Magic byte / MIME type validation ----
    try:
        detected_mime = magic.from_buffer(file_data, mime=True)
    except Exception as exc:
        logger.error("Magic byte detection failed | request_id=%s | error=%s", request_id, exc)
        raise HTTPException(status_code=400, detail="Could not determine file type from content")

    if detected_mime not in ALLOWED_MIME_TYPES:
        logger.warning(
            "MIME mismatch | request_id=%s | filename=%s | ext=%s | detected_mime=%s",
            request_id,
            file.filename,
            ext,
            detected_mime,
        )
        raise HTTPException(
            status_code=400,
            detail=(
                f"File content does not match expected type. "
                f"Extension '.{ext}' but detected content type is '{detected_mime}'. "
                f"Allowed types: {', '.join(sorted(ALLOWED_MIME_TYPES))}"
            ),
        )

    # Verify the detected MIME type is consistent with the file extension.
    # This catches extension spoofing (e.g. a PNG renamed to .pdf).
    _EXT_TO_MIME = {
        "pdf": "application/pdf",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
    }
    expected_mime = _EXT_TO_MIME.get(ext)
    if expected_mime and detected_mime != expected_mime:
        logger.warning(
            "Extension-MIME mismatch | request_id=%s | filename=%s | ext=%s | expected=%s | detected=%s",
            request_id,
            file.filename,
            ext,
            expected_mime,
            detected_mime,
        )
        raise HTTPException(
            status_code=400,
            detail=(
                f"File extension '.{ext}' does not match its content type '{detected_mime}'. "
                f"Expected '{expected_mime}' for this extension."
            ),
        )

    # ---- Step 5: PDF-specific validation (corrupt / zero pages) ----
    if detected_mime == "application/pdf":
        try:
            pdf_doc = fitz.open(stream=file_data, filetype="pdf")
            page_count = pdf_doc.page_count
            pdf_doc.close()
            if page_count == 0:
                raise HTTPException(
                    status_code=400,
                    detail="PDF could not be read or contains no pages",
                )
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning(
                "Corrupt PDF | request_id=%s | filename=%s | error=%s",
                request_id,
                file.filename,
                exc,
            )
            raise HTTPException(
                status_code=400,
                detail="PDF could not be read or contains no pages",
            )

    file_size_kb = file_size / 1024

    # ---- Step 6: Check cache ----
    file_hash = _hash_content(file_data)
    cached = await _cache_get(file_hash)
    if cached is not None:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            "Cache hit | request_id=%s | filename=%s | size_kb=%.1f | elapsed_ms=%.0f",
            request_id,
            file.filename,
            file_size_kb,
            elapsed_ms,
        )
        # Return a fresh copy with the current request_id
        result = cached.model_copy()
        result.request_id = request_id
        return JSONResponse(
            status_code=200,
            content=result.model_dump(),
        )
    logger.info("Cache miss | request_id=%s | filename=%s", request_id, file.filename)

    # ---- Step 7: Acquire concurrency semaphore ----
    acquired = False
    try:
        # Wait for a slot up to the configured timeout, then reject with 503
        try:
            acquired = await asyncio.wait_for(
                _semaphore.acquire(), timeout=_semaphore_timeout
            )
        except asyncio.TimeoutError:
            acquired = False
            logger.warning(
                "Semaphore busy (timeout=%.1fs) | request_id=%s",
                _semaphore_timeout,
                request_id,
            )
        except Exception:
            acquired = False
            logger.exception(
                "Semaphore acquire failed unexpectedly | request_id=%s",
                request_id,
            )

        if not acquired:
            raise HTTPException(
                status_code=503,
                detail="Server busy, please retry shortly",
            )

        # ---- Step 8: Call extraction ----
        try:
            extraction_result, prompt_tokens, completion_tokens = extract_document(
                file_data=file_data,
                mime_type=detected_mime,
                request_id=request_id,
            )
        except ValueError as exc:
            logger.error(
                "Extraction validation failure | request_id=%s | error=%s",
                request_id,
                exc,
            )
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            )
        except Exception:
            logger.exception(
                "Unexpected extraction error | request_id=%s",
                request_id,
            )
            raise HTTPException(
                status_code=500,
                detail="An unexpected error occurred during document extraction",
            )

        # ---- Step 9: Store in cache ----
        await _cache_set(file_hash, extraction_result)

        # ---- Step 10: Log success metrics ----
        elapsed_ms = (time.time() - start_time) * 1000
        estimated_cost = compute_cost(prompt_tokens, completion_tokens)
        logger.info(
            "Extraction complete | request_id=%s | filename=%s | size_kb=%.1f | "
            "prompt_tokens=%d | completion_tokens=%d | cost_usd=%.6f | elapsed_ms=%.0f",
            request_id,
            file.filename,
            file_size_kb,
            prompt_tokens,
            completion_tokens,
            estimated_cost,
            elapsed_ms,
        )

        return JSONResponse(
            status_code=200,
            content=extraction_result.model_dump(),
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled error | request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred",
        )
    finally:
        if acquired:
            _semaphore.release()


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=4000, reload=True)
