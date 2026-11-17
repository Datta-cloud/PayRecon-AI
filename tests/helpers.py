"""Small builders so each test can describe only the fields it cares about."""
from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd

from config import ReconConfig
from reconciliation.reconciliation_engine import ReconciliationResult, run_reconciliation

PLATFORM_COLS = [
    "transaction_id", "customer_id", "transaction_date", "transaction_type",
    "amount", "currency", "status", "original_transaction_id",
]
BANK_COLS = ["settlement_id", "transaction_id", "settlement_date", "settlement_amount", "currency", "bank_reference"]

_PLATFORM_DEFAULTS = dict(
    transaction_id="T1", customer_id="CUST-1", transaction_date="2026-10-10", transaction_type="PAYMENT",
    amount="100.00", currency="INR", status="SUCCESS", original_transaction_id="",
)
_BANK_DEFAULTS = dict(
    settlement_id="S1", transaction_id="T1", settlement_date="2026-10-11", settlement_amount="100.00",
    currency="INR", bank_reference="BNK1",
)


def platform_df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame([{**_PLATFORM_DEFAULTS, **r} for r in rows], columns=PLATFORM_COLS)


def bank_df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame([{**_BANK_DEFAULTS, **r} for r in rows], columns=BANK_COLS)


def empty_bank_but_valid() -> pd.DataFrame:
    """A bank file with one unrelated record (an entirely empty bank file is a fatal input error)."""
    return bank_df({"transaction_id": "UNRELATED", "settlement_id": "S-X"})


def run(platform: pd.DataFrame, bank: pd.DataFrame, **config_kwargs) -> ReconciliationResult:
    return run_reconciliation(platform, bank, ReconConfig(**config_kwargs))


def status_of(result: ReconciliationResult, txn_id: str) -> str:
    row = result.matches[result.matches["transaction_id"] == txn_id]
    assert len(row) == 1, f"expected exactly one match row for {txn_id}, got {len(row)}"
    return row.iloc[0]["match_status"]


D = Decimal
AS_OF = date(2026, 10, 31)
