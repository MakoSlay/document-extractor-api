"""
Shared test constants, data, and mock builders.

Importable as a regular Python module (unlike conftest.py which pytest
treats as a configuration file and won't let you import directly).
"""

import json
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Valid extraction response JSON that DeepSeek would return
# ---------------------------------------------------------------------------
VALID_EXTRACTION_DICT = {
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

VALID_EXTRACTION_JSON = json.dumps(VALID_EXTRACTION_DICT)

# JSON with a UTF-8 BOM prefix and surrounding whitespace/newlines
BOM_EXTRACTION_JSON = "\ufeff  \n" + VALID_EXTRACTION_JSON + "\n\n"

# JSON wrapped in markdown code fences (common LLM output mistake)
MARKDOWN_WRAPPED_JSON = "```json\n" + VALID_EXTRACTION_JSON + "\n```"

# Malformed JSON for testing retry exhaustion
MALFORMED_JSON = '{"document_type": "invoice", total_amount: 100.0}'  # missing quotes

# Empty response
EMPTY_JSON = ""


# ---------------------------------------------------------------------------
# Mock OpenAI response builder
# ---------------------------------------------------------------------------
def make_mock_openai_response(
    content: str,
    prompt_tokens: int = 500,
    completion_tokens: int = 200,
    empty_choices: bool = False,
):
    """
    Build a MagicMock that mimics the openai ChatCompletion response shape.

    Args:
        content: The message content string.
        prompt_tokens: Token count for the prompt.
        completion_tokens: Token count for the completion.
        empty_choices: If True, response.choices is an empty list (transient API glitch).
    """
    mock_msg = MagicMock()
    mock_msg.content = content

    mock_choice = MagicMock()
    mock_choice.message = mock_msg

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = prompt_tokens
    mock_usage.completion_tokens = completion_tokens

    mock_response = MagicMock()
    mock_response.choices = [] if empty_choices else [mock_choice]
    mock_response.usage = mock_usage

    return mock_response
