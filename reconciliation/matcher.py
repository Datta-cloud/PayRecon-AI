"""Match platform transactions with bank settlements.

Matching rules (all deterministic, no LLM involved):

1. Records are matched on ``transaction_id``.
2. If an ID appears more than once in a dataset, the *first* occurrence is used
   for matching and the extra copies are reported separately as duplicates.
   Nothing is deleted: extra copies stay available in ``extra`` frames.
3. A platform payment/refund with no bank record is:
   * **Pending Settlement** if its age (as-of date minus transaction date) is
     within the configured settlement window, otherwise
   * **Missing from Bank**.
4. A bank record with no platform record is **Missing from Platform**.
5. For matched pairs the signed amounts are compared:
   * equal -> **Matched**
   * differ by <= tolerance -> **Rounding Difference**
   * differ by more (or currencies differ) -> **Amount Mismatch**
6. Platform records with status FAILED expect no settlement (**Not Settleable**);
   if the bank settled one anyway it is a **Status Conflict**.
"""
from __future__ import annotations

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
    ROUNDING_DIFFERENCE,
    STATUS_CONFLICT,
    ReconConfig,
)

ZERO = Decimal("0")

MATCH_COLUMNS = [
    "transaction_id", "transaction_type", "match_status",
    "platform_date", "platform_amount", "platform_currency", "platform_status",
    "bank_date", "bank_amount", "bank_currency", "settlement_id", "bank_reference",
    "amount_difference", "settlement_lag_days", "timing_flag", "age_days", "note",
]


def split_duplicates(df: pd.DataFrame, key: str = "transaction_id") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (first occurrences, extra copies). Input order is preserved; nothing is lost."""
    is_extra = df.duplicated(subset=key, keep="first")
    return df[~is_extra].reset_index(drop=True), df[is_extra].reset_index(drop=True)


def default_as_of_date(platform: pd.DataFrame, bank: pd.DataFrame) -> date:
    """Latest date seen in either dataset; used as the reporting date."""
    dates = list(platform["transaction_date"]) + list(bank["settlement_date"])
    return max(dates)


def _fmt(value: Optional[Decimal]) -> str:
    return "n/a" if value is None else format(value, "f")


def _timing_flag(lag: int, config: ReconConfig) -> str:
    if lag < 0:
        return "Settled before transaction date"
    if lag < config.settlement_min_days:
        return "Earlier than expected window"
    if lag > config.settlement_max_days:
        return f"Late settlement ({lag} days)"
    return "Within window"


def match_transactions(
    platform: pd.DataFrame,
    bank: pd.DataFrame,
    config: Optional[ReconConfig] = None,
    as_of: Optional[date] = None,
) -> pd.DataFrame:
    """Classify every transaction. Returns one row per distinct transaction_id."""
    config = config or ReconConfig()
    as_of = as_of or config.as_of_date or default_as_of_date(platform, bank)

    p_first, _ = split_duplicates(platform)
    b_first, _ = split_duplicates(bank)
    bank_by_id = {row["transaction_id"]: row for row in b_first.to_dict("records")}
    platform_ids = set(p_first["transaction_id"])

    records: list[dict] = []

    for p in p_first.to_dict("records"):
        tid = p["transaction_id"]
        base = {
            "transaction_id": tid,
            "transaction_type": p["transaction_type"],
            "platform_date": p["transaction_date"],
            "platform_amount": p["signed_amount"],
            "platform_currency": p["currency"],
            "platform_status": p["status"],
            "bank_date": None, "bank_amount": None, "bank_currency": None,
            "settlement_id": None, "bank_reference": None,
            "amount_difference": None, "settlement_lag_days": None,
            "timing_flag": "", "age_days": None, "note": "",
        }
        b = bank_by_id.get(tid)

        if b is None:
            age = (as_of - p["transaction_date"]).days
            if p["status"] != "SUCCESS":
                base.update(match_status=NOT_SETTLEABLE, note=f"Status is {p['status']} on the platform, so no settlement is expected.")
            elif age <= config.settlement_max_days:
                base.update(
                    match_status=PENDING_SETTLEMENT, age_days=age,
                    note=f"Age {age} day(s) is within the {config.settlement_max_days}-day settlement window.",
                )
            else:
                base.update(
                    match_status=MISSING_FROM_BANK, age_days=age,
                    note=f"No bank settlement after {age} days (window is {config.settlement_max_days}).",
                )
            records.append(base)
            continue

        lag = (b["settlement_date"] - p["transaction_date"]).days
        base.update(
            bank_date=b["settlement_date"], bank_amount=b["settlement_amount"], bank_currency=b["currency"],
            settlement_id=b["settlement_id"], bank_reference=b["bank_reference"],
            settlement_lag_days=lag, timing_flag=_timing_flag(lag, config),
        )

        if p["status"] != "SUCCESS":
            base.update(
                match_status=STATUS_CONFLICT,
                note=f"Bank settled this transaction but the platform status is {p['status']}.",
            )
        elif p["currency"] != b["currency"]:
            base.update(
                match_status=AMOUNT_MISMATCH,
                note=f"Currency differs: platform {p['currency']} vs bank {b['currency']}.",
            )
        else:
            diff = p["signed_amount"] - b["settlement_amount"]
            base["amount_difference"] = diff
            if diff == ZERO:
                base.update(match_status=MATCHED, note="Amounts agree exactly.")
            elif abs(diff) <= config.rounding_tolerance:
                base.update(
                    match_status=ROUNDING_DIFFERENCE,
                    note=f"Difference of {_fmt(diff)} is within the rounding tolerance of {_fmt(config.rounding_tolerance)}.",
                )
            else:
                base.update(
                    match_status=AMOUNT_MISMATCH,
                    note=f"Platform {_fmt(p['signed_amount'])} vs bank {_fmt(b['settlement_amount'])} (difference {_fmt(diff)}).",
                )
        records.append(base)

    for b in b_first.to_dict("records"):
        if b["transaction_id"] in platform_ids:
            continue
        records.append({
            "transaction_id": b["transaction_id"], "transaction_type": None,
            "match_status": MISSING_FROM_PLATFORM,
            "platform_date": None, "platform_amount": None, "platform_currency": None, "platform_status": None,
            "bank_date": b["settlement_date"], "bank_amount": b["settlement_amount"], "bank_currency": b["currency"],
            "settlement_id": b["settlement_id"], "bank_reference": b["bank_reference"],
            "amount_difference": None, "settlement_lag_days": None, "timing_flag": "", "age_days": None,
            "note": "The bank settled this transaction ID but the platform has no record of it.",
        })

    return pd.DataFrame(records, columns=MATCH_COLUMNS)
