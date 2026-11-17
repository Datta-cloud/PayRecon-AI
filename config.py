
"""Central configuration for PayRecon AI.

Everything that is a tunable setting or a shared constant lives here so the
business logic, the UI and the tests all agree on the same names.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


# --------------------------------------------------------------------------
# AI (Gemini)
# --------------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Use a model ID supported by your Google AI Studio API key.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

GEMINI_TIMEOUT_SECONDS = 30
GEMINI_MAX_RETRIES = 2

# Compatibility aliases for existing code using mixed capitalization.
Gemini_TIMEOUT_SECONDS = GEMINI_TIMEOUT_SECONDS
Gemini_MAX_RETRIES = GEMINI_MAX_RETRIES


def get_GEMINI_API_KEY() -> Optional[str]:
    """Get Gemini API key from environment or Streamlit secrets."""

    key = os.getenv("GEMINI_API_KEY", "").strip()

    if key:
        return key

    try:
        import streamlit as st

        secret = st.secrets.get("GEMINI_API_KEY")

        if secret and str(secret).strip():
            return str(secret).strip()

    except Exception:
        pass

    return None


# Compatibility aliases for different existing import spellings.
def get_Gemini_api_key() -> Optional[str]:
    return get_GEMINI_API_KEY()


def get_gemini_api_key() -> Optional[str]:
    return get_GEMINI_API_KEY()


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------

APP_NAME = "PayRecon AI"

APP_TAGLINE = (
    "Compare your payment platform records with bank settlements and find "
    "rounding differences, duplicates and refunds without an original payment."
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PLATFORM_CSV = DATA_DIR / "platform_transactions.csv"
BANK_CSV = DATA_DIR / "bank_settlements.csv"

DEFAULT_SEED = 42


# --------------------------------------------------------------------------
# Input schema
# --------------------------------------------------------------------------

SUPPORTED_CURRENCIES = ("INR", "USD", "EUR", "GBP")

TRANSACTION_TYPES = ("PAYMENT", "REFUND")

PLATFORM_STATUSES = ("SUCCESS", "FAILED")

PLATFORM_REQUIRED_COLUMNS = (
    "transaction_id",
    "customer_id",
    "transaction_date",
    "transaction_type",
    "amount",
    "currency",
    "status",
)

PLATFORM_OPTIONAL_COLUMNS = ("original_transaction_id",)

BANK_REQUIRED_COLUMNS = (
    "settlement_id",
    "transaction_id",
    "settlement_date",
    "settlement_amount",
    "currency",
    "bank_reference",
)


# --------------------------------------------------------------------------
# Match statuses
# --------------------------------------------------------------------------

MATCHED = "Matched"

ROUNDING_DIFFERENCE = "Rounding Difference"
AMOUNT_MISMATCH = "Amount Mismatch"
PENDING_SETTLEMENT = "Pending Settlement"
MISSING_FROM_BANK = "Missing from Bank"
MISSING_FROM_PLATFORM = "Missing from Platform"
STATUS_CONFLICT = "Status Conflict"
NOT_SETTLEABLE = "Not Settleable"

MATCH_STATUSES = (
    MATCHED,
    ROUNDING_DIFFERENCE,
    AMOUNT_MISMATCH,
    PENDING_SETTLEMENT,
    MISSING_FROM_BANK,
    MISSING_FROM_PLATFORM,
    STATUS_CONFLICT,
    NOT_SETTLEABLE,
)


# --------------------------------------------------------------------------
# Discrepancy categories
# --------------------------------------------------------------------------

CAT_ROUNDING = "Rounding Difference"
CAT_DUPLICATE = "Duplicate Entry"
CAT_UNMATCHED_REFUND = "Unmatched Refund"
CAT_AMOUNT_MISMATCH = "Amount Mismatch"
CAT_MISSING_BANK = "Missing from Bank"
CAT_MISSING_PLATFORM = "Missing from Platform"
CAT_STATUS_CONFLICT = "Status Conflict"
CAT_REFUND_INCONSISTENT = "Refund Inconsistency"


# --------------------------------------------------------------------------
# Refund review statuses
# --------------------------------------------------------------------------

REFUND_VALID = "Valid"
REFUND_UNMATCHED = "Unmatched Refund"
REFUND_INVALID_REF = "Invalid Reference"
REFUND_INCONSISTENT = "Inconsistent Refund"

# --------------------------------------------------------------------------
# Reconciliation settings
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ReconConfig:
    """Settings that change how reconciliation decisions are made.

    Attributes:
        settlement_min_days: Earliest normal settlement lag (calendar days).
        settlement_max_days: Latest normal settlement lag. A payment with no
            bank record yet is "Pending" while its age is <= this value.
        rounding_tolerance: Largest absolute per-transaction difference that
            is treated as a rounding difference.
        supported_currencies: Currency codes accepted by the validator.
        as_of_date: Reporting date. None means latest date in the data.
    """

    settlement_min_days: int = 1
    settlement_max_days: int = 2
    rounding_tolerance: Decimal = Decimal("0.05")
    supported_currencies: tuple = SUPPORTED_CURRENCIES
    as_of_date: Optional[date] = None

    def __post_init__(self) -> None:
        if self.settlement_min_days < 0:
            raise ValueError("settlement_min_days cannot be negative")

        if self.settlement_max_days < self.settlement_min_days:
            raise ValueError(
                "settlement_max_days must be >= settlement_min_days"
            )

        tol = Decimal(str(self.rounding_tolerance))

        if tol < 0:
            raise ValueError("rounding_tolerance cannot be negative")

        object.__setattr__(self, "rounding_tolerance", tol)
def get_groq_api_key():
    """Backward compatibility for old imports. Returns the Gemini API key."""
    return get_Gemini_api_key()