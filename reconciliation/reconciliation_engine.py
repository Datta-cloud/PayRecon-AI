"""Orchestrates a full reconciliation run.

``run_reconciliation`` is the single entry point used by the UI and the tests:

    raw CSV frames -> validate -> match -> detect -> bridge + summary
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Optional

import pandas as pd

from config import (
    AMOUNT_MISMATCH,
    MATCHED,
    MISSING_FROM_BANK,
    MISSING_FROM_PLATFORM,
    NOT_SETTLEABLE,
    PENDING_SETTLEMENT,
    REFUND_INCONSISTENT,
    REFUND_INVALID_REF,
    REFUND_UNMATCHED,
    REFUND_VALID,
    ROUNDING_DIFFERENCE,
    STATUS_CONFLICT,
    ReconConfig,
)
from reconciliation.discrepancy_detector import (
    analyze_rounding,
    build_discrepancy_table,
    detect_duplicates,
    detect_unmatched_refunds,
)
from reconciliation.matcher import default_as_of_date, match_transactions, split_duplicates
from reconciliation.validator import ValidationResult, validate_bank, validate_platform

ZERO = Decimal("0")

BRIDGE_COMPONENTS = [
    "amount_differences", "currency_mismatch", "pending_settlement", "missing_from_bank",
    "missing_from_platform", "status_conflict", "platform_duplicates", "bank_duplicates",
]


class ReconciliationInputError(Exception):
    """Raised when the inputs cannot be reconciled (missing columns, empty files...)."""

    def __init__(self, errors: list[str], validations: Optional[list[ValidationResult]] = None):
        super().__init__("; ".join(errors))
        self.errors = errors
        self.validations = validations or []


@dataclass
class ReconciliationResult:
    config: ReconConfig
    as_of_date: date
    platform: pd.DataFrame
    bank: pd.DataFrame
    platform_validation: ValidationResult
    bank_validation: ValidationResult
    matches: pd.DataFrame
    duplicates: pd.DataFrame
    refunds: pd.DataFrame
    rounding_details: pd.DataFrame
    rounding_by_currency: pd.DataFrame
    rounding_by_period: pd.DataFrame
    discrepancies: pd.DataFrame
    bridge: pd.DataFrame
    daily_trend: pd.DataFrame
    summary: dict = field(default_factory=dict)

    def with_status(self, *statuses: str) -> pd.DataFrame:
        return self.matches[self.matches["match_status"].isin(statuses)].reset_index(drop=True)

    @property
    def pending(self) -> pd.DataFrame:
        return self.with_status(PENDING_SETTLEMENT)


def _dsum(values) -> Decimal:
    return sum(values, ZERO)


def build_bridge(matches: pd.DataFrame, platform: pd.DataFrame, bank: pd.DataFrame, platform_extra: pd.DataFrame, bank_extra: pd.DataFrame) -> pd.DataFrame:
    """Explain the gap between platform and bank totals, per currency.

    gap = platform total - bank total
        = amount differences + currency mismatches + pending + missing from bank
          - missing from platform - status conflicts + platform duplicates - bank duplicates
          + unexplained

    ``unexplained`` is a built-in self-check: if the engine classified every record
    correctly it is exactly zero.
    Platform totals only include SUCCESS records (failed payments move no money).
    """
    p_ok = platform[platform["status"] == "SUCCESS"]
    currencies = sorted(set(p_ok["currency"]) | set(bank["currency"]))
    rows = []
    recs = matches.to_dict("records")
    p_extra_ok = platform_extra[platform_extra["status"] == "SUCCESS"]

    for cur in currencies:
        platform_total = _dsum(p_ok.loc[p_ok["currency"] == cur, "signed_amount"])
        bank_total = _dsum(bank.loc[bank["currency"] == cur, "settlement_amount"])
        comp = {k: ZERO for k in BRIDGE_COMPONENTS}

        for r in recs:
            status = r["match_status"]
            if status in (MATCHED, ROUNDING_DIFFERENCE) or (status == AMOUNT_MISMATCH and r["platform_currency"] == r["bank_currency"]):
                if r["platform_currency"] == cur:
                    comp["amount_differences"] += r["amount_difference"]
            elif status == AMOUNT_MISMATCH:  # currency mismatch between the two sides
                if r["platform_currency"] == cur:
                    comp["currency_mismatch"] += r["platform_amount"]
                if r["bank_currency"] == cur:
                    comp["currency_mismatch"] -= r["bank_amount"]
            elif status == PENDING_SETTLEMENT and r["platform_currency"] == cur:
                comp["pending_settlement"] += r["platform_amount"]
            elif status == MISSING_FROM_BANK and r["platform_currency"] == cur:
                comp["missing_from_bank"] += r["platform_amount"]
            elif status == MISSING_FROM_PLATFORM and r["bank_currency"] == cur:
                comp["missing_from_platform"] -= r["bank_amount"]
            elif status == STATUS_CONFLICT and r["bank_currency"] == cur:
                comp["status_conflict"] -= r["bank_amount"]

        comp["platform_duplicates"] = _dsum(p_extra_ok.loc[p_extra_ok["currency"] == cur, "signed_amount"])
        comp["bank_duplicates"] = -_dsum(bank_extra.loc[bank_extra["currency"] == cur, "settlement_amount"])

        difference = platform_total - bank_total
        explained = _dsum(comp.values())
        rows.append({
            "currency": cur, "platform_total": platform_total, "bank_total": bank_total,
            "difference": difference, **comp, "explained_total": explained,
            "unexplained": difference - explained,
        })
    return pd.DataFrame(rows, columns=["currency", "platform_total", "bank_total", "difference", *BRIDGE_COMPONENTS, "explained_total", "unexplained"])


def build_daily_trend(platform: pd.DataFrame, bank: pd.DataFrame) -> pd.DataFrame:
    """Per-day, per-currency counts and net amounts (floats: for charts only)."""
    p = platform[platform["status"] == "SUCCESS"]
    rows: dict[tuple, dict] = {}
    for r in p.to_dict("records"):
        key = (r["transaction_date"], r["currency"])
        row = rows.setdefault(key, {"date": key[0], "currency": key[1], "platform_amount": 0.0, "platform_count": 0, "bank_amount": 0.0, "bank_count": 0})
        row["platform_amount"] += float(r["signed_amount"])
        row["platform_count"] += 1
    for r in bank.to_dict("records"):
        key = (r["settlement_date"], r["currency"])
        row = rows.setdefault(key, {"date": key[0], "currency": key[1], "platform_amount": 0.0, "platform_count": 0, "bank_amount": 0.0, "bank_count": 0})
        row["bank_amount"] += float(r["settlement_amount"])
        row["bank_count"] += 1
    cols = ["date", "currency", "platform_amount", "platform_count", "bank_amount", "bank_count"]
    return pd.DataFrame(sorted(rows.values(), key=lambda x: (x["currency"], x["date"])), columns=cols)


def _completion_status(discrepancies: int, pending: int) -> str:
    if discrepancies == 0 and pending == 0:
        return "Fully reconciled"
    if discrepancies == 0:
        return f"Reconciled, {pending} settlement(s) still pending"
    return f"Needs review: {discrepancies} discrepanc{'y' if discrepancies == 1 else 'ies'} found"


def run_reconciliation(platform_raw: pd.DataFrame, bank_raw: pd.DataFrame, config: Optional[ReconConfig] = None) -> ReconciliationResult:
    """Validate, match and analyse two raw (all-string) DataFrames."""
    config = config or ReconConfig()
    pv = validate_platform(platform_raw, config)
    bv = validate_bank(bank_raw, config)
    errors = pv.fatal_errors + bv.fatal_errors
    if errors:
        raise ReconciliationInputError(errors, [pv, bv])

    platform, bank = pv.valid, bv.valid
    as_of = config.as_of_date or default_as_of_date(platform, bank)

    matches = match_transactions(platform, bank, config, as_of)
    duplicates = detect_duplicates(platform, bank)
    refunds = detect_unmatched_refunds(platform)
    rounding_details, rounding_by_currency, rounding_by_period = analyze_rounding(matches, config)
    discrepancies = build_discrepancy_table(matches, duplicates, refunds)
    _, platform_extra = split_duplicates(platform)
    _, bank_extra = split_duplicates(bank)
    bridge = build_bridge(matches, platform, bank, platform_extra, bank_extra)
    daily = build_daily_trend(platform, bank)

    counts = matches["match_status"].value_counts().to_dict()
    refund_counts = refunds["refund_status"].value_counts().to_dict() if not refunds.empty else {}
    pending = int(counts.get(PENDING_SETTLEMENT, 0))
    summary = {
        "as_of_date": as_of,
        "total_platform_records": len(platform),
        "total_bank_records": len(bank),
        "platform_invalid_rows": len(pv.invalid_rows),
        "bank_invalid_rows": len(bv.invalid_rows),
        "matched": int(counts.get(MATCHED, 0)),
        "rounding_differences": int(counts.get(ROUNDING_DIFFERENCE, 0)),
        "amount_mismatches": int(counts.get(AMOUNT_MISMATCH, 0)),
        "pending_settlements": pending,
        "missing_from_bank": int(counts.get(MISSING_FROM_BANK, 0)),
        "missing_from_platform": int(counts.get(MISSING_FROM_PLATFORM, 0)),
        "status_conflicts": int(counts.get(STATUS_CONFLICT, 0)),
        "not_settleable": int(counts.get(NOT_SETTLEABLE, 0)),
        "duplicate_entries": len(duplicates),
        "platform_duplicate_ids": int((duplicates["dataset"] == "Platform").sum()) if not duplicates.empty else 0,
        "bank_duplicate_ids": int((duplicates["dataset"] == "Bank").sum()) if not duplicates.empty else 0,
        "valid_refunds": int(refund_counts.get(REFUND_VALID, 0)),
        "unmatched_refunds": int(refund_counts.get(REFUND_UNMATCHED, 0) + refund_counts.get(REFUND_INVALID_REF, 0)),
        "inconsistent_refunds": int(refund_counts.get(REFUND_INCONSISTENT, 0)),
        "late_settlements": int(matches["timing_flag"].str.startswith("Late").sum()) if len(matches) else 0,
        "total_discrepancies": len(discrepancies),
        "total_amount_difference": {r["currency"]: r["difference"] for r in bridge.to_dict("records")},
        "unexplained_difference": {r["currency"]: r["unexplained"] for r in bridge.to_dict("records")},
        "cumulative_rounding": {r["currency"]: r["net_difference"] for r in rounding_by_currency.to_dict("records")},
        "completion_status": _completion_status(len(discrepancies), pending),
    }
    return ReconciliationResult(
        config=config, as_of_date=as_of, platform=platform, bank=bank,
        platform_validation=pv, bank_validation=bv, matches=matches, duplicates=duplicates,
        refunds=refunds, rounding_details=rounding_details, rounding_by_currency=rounding_by_currency,
        rounding_by_period=rounding_by_period, discrepancies=discrepancies, bridge=bridge,
        daily_trend=daily, summary=summary,
    )
