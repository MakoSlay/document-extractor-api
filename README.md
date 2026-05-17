# Document Extractor API

A Python FastAPI server that accepts PDF or image document uploads, sends them to the DeepSeek AI API for vision-based data extraction, and returns clean structured JSON containing all key fields (vendor, date, total, line items, tax, currency, etc.).

Designed for listing on RapidAPI.

---

## Features

- PDF and image (PNG/JPG) upload support
- Magic-byte file validation (extension spoofing is rejected)
- 10 MB file size limit
- MD5-based response cache with LRU eviction
- Concurrency limited to 5 simultaneous extraction requests
- Retry logic (up to 3 attempts) for malformed AI responses
- Dynamic confidence scoring based on field population
- Detailed structured JSON output
- CORS enabled for all origins
- Structured logging throughout — no print statements

---

## System Requirements

- Python 3.11+
- **libmagic** (system package, required by `python-magic`)

### Installing libmagic

**macOS:**
```bash
brew install libmagic
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get install libmagic1
```

**Linux (RHEL/Fedora):**
```bash
sudo dnf install file-libs
```

---

## Setup

1.  **Clone and navigate to the project:**
    ```bash
    cd document-extractor
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # macOS/Linux
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment variables:**
    ```bash
    cp .env.example .env
    ```

    Edit `.env` and fill in your DeepSeek API key:
    ```
    DEEPSEEK_API_KEY=sk-your-deepseek-api-key
    CACHE_MAX_SIZE=100
    ```

    Get your API key at [platform.deepseek.com](https://platform.deepseek.com/api_keys).

---

## Running the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or use the built-in entry point:
```bash
python main.py
```

The server will refuse to start if `DEEPSEEK_API_KEY` is missing or empty.

Verify it's running:
```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

---

## API Endpoints

### `GET /health`

Returns server status.

**Response:**
```json
{"status": "ok"}
```

---

### `POST /extract`

Upload a document for AI extraction.

**Request:** multipart/form-data with field `file`.

**curl example:**
```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@invoice.pdf"
```

**Success response (200):**
```json
{
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "document_type": "invoice",
  "vendor_name": "Acme Supplies Inc.",
  "vendor_address": "123 Industrial Blvd, Springfield, IL 62701",
  "document_number": "INV-2024-0892",
  "date": "2024-11-15",
  "due_date": "2024-12-15",
  "currency": "USD",
  "subtotal": 1250.00,
  "tax_amount": 100.00,
  "tax_rate": 8.0,
  "total_amount": 1350.00,
  "line_items": [
    {
      "description": "Widget Model A",
      "quantity": 10.0,
      "unit_price": 75.00,
      "total": 750.00
    },
    {
      "description": "Widget Model B",
      "quantity": 5.0,
      "unit_price": 100.00,
      "total": 500.00
    }
  ],
  "payment_terms": "Net 30",
  "notes": null,
  "confidence_score": 0.8333
}
```

**Error responses:**

| Status | Meaning |
|--------|---------|
| 400 | Bad input — invalid extension, MIME mismatch, file too large, empty/corrupt PDF |
| 422 | Validation failure — DeepSeek returned data that failed schema validation |
| 503 | Server busy — all 5 extraction slots are occupied, retry shortly |
| 500 | Unexpected internal error |

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | **Yes** | — | Your DeepSeek API key (starts with `sk-`) |
| `CACHE_MAX_SIZE` | No | `100` | Maximum number of cached extraction results |

---

## Cost Estimation

DeepSeek-chat pricing (as of 2024):

| Token Type | Price per 1M tokens |
|------------|---------------------|
| Input (prompt) | $0.27 |
| Output (completion) | $1.10 |

**Typical per-request costs:**

| Document Type | Input Tokens | Output Tokens | Estimated Cost |
|---------------|-------------|---------------|----------------|
| Simple receipt | ~800 | ~200 | ~$0.00044 |
| Standard invoice | ~1,200 | ~350 | ~$0.00071 |
| Complex multi-page invoice | ~1,800 | ~600 | ~$0.00115 |

Cost per request is logged alongside every extraction for monitoring.

---

## Project Structure

```
document-extractor/
├── main.py            # FastAPI app, endpoints, validation, caching
├── extractor.py       # DeepSeek API integration, image encoding, retry logic
├── models.py          # Pydantic v2 response schemas with dynamic confidence scoring
├── requirements.txt   # Pinned Python dependencies
├── .env.example       # Environment variable template
├── __init__.py        # Package marker
└── README.md          # This file
```

---

## Architecture Notes

- **Validation is defense-in-depth**: extension check → magic bytes → file size → PDF structural check. No single check is trusted in isolation.
- **Cache is content-addressable**: MD5 hash of raw file bytes. Re-uploading the identical file returns a cached result instantly.
- **Confidence score is dynamic**: computed as `populated_optional_fields / total_optional_fields` in the schema. Never hardcoded.
- **No internal details leak**: all error responses contain only actionable messages — never stack traces or internal paths.
