"""Validate raw CSV data and convert it to clean, typed records.

Rules:
* Missing columns, or a file with no data rows, are *fatal* (nothing can be reconciled).
* A bad row (missing transaction ID, unparseable amount or date, unsupported
  currency...) is *not* silently dropped: it is excluded from matching but listed
  in ``invalid_rows`` so the user can see exactly what was left out and why.
* Amounts become ``decimal.Decimal`` so no floating point error can creep in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Optional

import pandas as pd

from config import (
    BANK_REQUIRED_COLUMNS,
    PLATFORM_OPTIONAL_COLUMNS,
    PLATFORM_REQUIRED_COLUMNS,
    PLATFORM_STATUSES,
    TRANSACTION_TYPES,
    ReconConfig,
)
from reconciliation.data_loader import normalize_columns

PLATFORM_CLEAN_COLUMNS = [
    "transaction_id", "customer_id", "transaction_date", "transaction_type", "amount",
    "currency", "status", "original_transaction_id", "signed_amount", "source_row",
]
BANK_CLEAN_COLUMNS = [
    "settlement_id", "transaction_id", "settlement_date", "settlement_amount",
    "currency", "bank_reference", "source_row",
]
INVALID_ROW_COLUMNS = ["row_number", "field", "problem", "value"]


@dataclass
class ValidationResult:
    dataset: str
    valid: pd.DataFrame
    invalid_rows: pd.DataFrame
    fatal_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_rows: int = 0

    @property
    def ok(self) -> bool:
        return not self.fatal_errors


def parse_decimal(value: object) -> Optional[Decimal]:
    """Parse '1,234.50' -> Decimal('1234.50'). Returns None for blank/invalid/NaN/inf."""
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def parse_date(value: object) -> Optional[date]:
    """Parse an ISO-style date or timestamp to a ``date``. Returns None if invalid."""
    text = str(value).strip()
    if not text:
        return None
    stamp = pd.to_datetime(text, errors="coerce")
    if pd.isna(stamp):
        return None
    return stamp.date()


def _empty_result(dataset: str, valid_cols: list[str], fatal: list[str], warnings: list[str], total: int = 0) -> ValidationResult:
    return ValidationResult(
        dataset=dataset,
        valid=pd.DataFrame(columns=valid_cols),
        invalid_rows=pd.DataFrame(columns=INVALID_ROW_COLUMNS),
        fatal_errors=fatal,
        warnings=warnings,
        total_rows=total,
    )


def _prepare(raw: object, dataset: str, required: tuple, optional: tuple, valid_cols: list[str]):
    """Shared structural checks. Returns (dataframe or None, early_result or None)."""
    if not isinstance(raw, pd.DataFrame):
        return None, _empty_result(dataset, valid_cols, [f"{dataset}: no data was provided."], [])
    df = normalize_columns(raw)
    warnings: list[str] = []
    missing = [c for c in required if c not in df.columns]
    if missing:
        found = ", ".join(df.columns) if len(df.columns) else "none"
        return None, _empty_result(
            dataset, valid_cols,
            [f"{dataset}: missing required column(s): {', '.join(missing)}. Columns found: {found}."],
            warnings, total=len(df),
        )
    if df.empty:
        return None, _empty_result(dataset, valid_cols, [f"{dataset}: the file has a header but no data rows."], warnings)
    for col in optional:
        if col not in df.columns:
            df[col] = ""
            warnings.append(f"{dataset}: optional column '{col}' not found; treated as blank.")
    return (df.reset_index(drop=True), warnings), None


def _finish(dataset: str, valid_rows: list[dict], invalid: list[dict], valid_cols: list[str], warnings: list[str], total: int) -> ValidationResult:
    fatal: list[str] = []
    if not valid_rows:
        fatal.append(f"{dataset}: none of the {total} rows are valid. See the data quality issues for details.")
    return ValidationResult(
        dataset=dataset,
        valid=pd.DataFrame(valid_rows, columns=valid_cols),
        invalid_rows=pd.DataFrame(invalid, columns=INVALID_ROW_COLUMNS),
        fatal_errors=fatal,
        warnings=warnings,
        total_rows=total,
    )


def validate_platform(raw: pd.DataFrame, config: Optional[ReconConfig] = None) -> ValidationResult:
    config = config or ReconConfig()
    dataset = "Platform transactions"
    prepared, early = _prepare(raw, dataset, PLATFORM_REQUIRED_COLUMNS, PLATFORM_OPTIONAL_COLUMNS, PLATFORM_CLEAN_COLUMNS)
    if early is not None:
        return early
    df, warnings = prepared
    valid_rows: list[dict] = []
    invalid: list[dict] = []

    for idx, row in df.iterrows():
        row_number = int(idx) + 2  # +1 for 1-based, +1 for the header line
        problems: list[tuple[str, str, str]] = []

        txn_id = str(row["transaction_id"]).strip()
        if not txn_id:
            problems.append(("transaction_id", "Missing transaction ID", ""))

        txn_date = parse_date(row["transaction_date"])
        if txn_date is None:
            problems.append(("transaction_date", "Invalid or missing date", row["transaction_date"]))

        txn_type = str(row["transaction_type"]).strip().upper()
        if txn_type not in TRANSACTION_TYPES:
            problems.append(("transaction_type", f"Must be one of {', '.join(TRANSACTION_TYPES)}", row["transaction_type"]))

        amount = parse_decimal(row["amount"])
        if amount is None:
            problems.append(("amount", "Invalid or missing amount", row["amount"]))
        elif amount <= 0:
            problems.append(("amount", "Amount must be positive (refunds are positive amounts with type REFUND)", row["amount"]))

        currency = str(row["currency"]).strip().upper()
        if currency not in config.supported_currencies:
            problems.append(("currency", f"Unsupported currency (supported: {', '.join(config.supported_currencies)})", row["currency"]))

        status = str(row["status"]).strip().upper()
        if status not in PLATFORM_STATUSES:
            problems.append(("status", f"Must be one of {', '.join(PLATFORM_STATUSES)}", row["status"]))

        if problems:
            invalid.extend({"row_number": row_number, "field": f, "problem": p, "value": v} for f, p, v in problems)
            continue

        valid_rows.append({
            "transaction_id": txn_id,
            "customer_id": str(row["customer_id"]).strip(),
            "transaction_date": txn_date,
            "transaction_type": txn_type,
            "amount": amount,
            "currency": currency,
            "status": status,
            "original_transaction_id": str(row["original_transaction_id"]).strip(),
            "signed_amount": -amount if txn_type == "REFUND" else amount,
            "source_row": row_number,
        })

    return _finish(dataset, valid_rows, invalid, PLATFORM_CLEAN_COLUMNS, warnings, len(df))


def validate_bank(raw: pd.DataFrame, config: Optional[ReconConfig] = None) -> ValidationResult:
    config = config or ReconConfig()
    dataset = "Bank settlements"
    prepared, early = _prepare(raw, dataset, BANK_REQUIRED_COLUMNS, (), BANK_CLEAN_COLUMNS)
    if early is not None:
        return early
    df, warnings = prepared
    valid_rows: list[dict] = []
    invalid: list[dict] = []

    for idx, row in df.iterrows():
        row_number = int(idx) + 2
        problems: list[tuple[str, str, str]] = []

        txn_id = str(row["transaction_id"]).strip()
        if not txn_id:
            problems.append(("transaction_id", "Missing transaction ID", ""))

        settle_date = parse_date(row["settlement_date"])
        if settle_date is None:
            problems.append(("settlement_date", "Invalid or missing date", row["settlement_date"]))

        amount = parse_decimal(row["settlement_amount"])
        if amount is None:
            problems.append(("settlement_amount", "Invalid or missing amount", row["settlement_amount"]))
        elif amount == 0:
            problems.append(("settlement_amount", "Settlement amount is zero", row["settlement_amount"]))

        currency = str(row["currency"]).strip().upper()
        if currency not in config.supported_currencies:
            problems.append(("currency", f"Unsupported currency (supported: {', '.join(config.supported_currencies)})", row["currency"]))

        if problems:
            invalid.extend({"row_number": row_number, "field": f, "problem": p, "value": v} for f, p, v in problems)
            continue

        valid_rows.append({
            "settlement_id": str(row["settlement_id"]).strip(),
            "transaction_id": txn_id,
            "settlement_date": settle_date,
            "settlement_amount": amount,
            "currency": currency,
            "bank_reference": str(row["bank_reference"]).strip(),
            "source_row": row_number,
        })

    return _finish(dataset, valid_rows, invalid, BANK_CLEAN_COLUMNS, warnings, len(df))
