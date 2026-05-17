"""
Pydantic v2 models defining the extraction response schema.

The confidence_score is computed dynamically as the ratio of
populated optional fields to total optional fields in the schema.
It is never hardcoded and always reflects the actual data extracted.
"""

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class LineItem(BaseModel):
    """A single line item from an invoice or receipt."""

    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total: Optional[float] = None


class ExtractionResponse(BaseModel):
    """The complete structured extraction result for a document."""

    request_id: str
    document_type: str

    vendor_name: Optional[str] = None
    vendor_address: Optional[str] = None
    document_number: Optional[str] = None
    date: Optional[str] = None
    due_date: Optional[str] = None
    currency: Optional[str] = "USD"

    subtotal: Optional[float] = None
    tax_amount: Optional[float] = None
    tax_rate: Optional[float] = None
    total_amount: float

    line_items: list[LineItem] = Field(default_factory=list)
    payment_terms: Optional[str] = None
    notes: Optional[str] = None

    confidence_score: float = Field(ge=0.0, le=1.0, default=0.0)

    @model_validator(mode="after")
    def compute_confidence(self) -> "ExtractionResponse":
        """
        Compute confidence_score as the ratio of non-empty optional fields
        to the total number of optional fields defined in this schema.

        This runs automatically after every model instantiation and
        guarantees the score is never hardcoded.
        """
        optional_fields: list[tuple[str, bool]] = [
            # (field_name, is_list_type)
            ("vendor_name", False),
            ("vendor_address", False),
            ("document_number", False),
            ("date", False),
            ("due_date", False),
            ("currency", False),
            ("subtotal", False),
            ("tax_amount", False),
            ("tax_rate", False),
            ("line_items", True),
            ("payment_terms", False),
            ("notes", False),
        ]

        populated = 0
        for field_name, is_list in optional_fields:
            value = getattr(self, field_name)
            if is_list:
                # A non-empty list counts as populated
                if value and len(value) > 0:
                    populated += 1
            else:
                if value is not None:
                    populated += 1

        self.confidence_score = round(populated / len(optional_fields), 4)
        return self
