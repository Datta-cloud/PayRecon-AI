"""Detect the three required discrepancy types and build the discrepancy report.

* Duplicate entries    -> :func:`detect_duplicates`
* Rounding differences -> :func:`analyze_rounding` (individual and cumulative)
* Unmatched refunds    -> :func:`detect_unmatched_refunds`
"""
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from typing import Optional

import pandas as pd

from config import (
    AMOUNT_MISMATCH,
    CAT_AMOUNT_MISMATCH,
    CAT_DUPLICATE,
    CAT_MISSING_BANK,
    CAT_MISSING_PLATFORM,
    CAT_REFUND_INCONSISTENT,
    CAT_ROUNDING,
    CAT_STATUS_CONFLICT,
    CAT_UNMATCHED_REFUND,
    MISSING_FROM_BANK,
    MISSING_FROM_PLATFORM,
    REFUND_INCONSISTENT,
    REFUND_INVALID_REF,
    REFUND_UNMATCHED,
    REFUND_VALID,
    ROUNDING_DIFFERENCE,
    STATUS_CONFLICT,
    ReconConfig,
)

ZERO = Decimal("0")

DUPLICATE_COLUMNS = [
    "dataset", "transaction_id", "occurrences", "extra_copies", "amounts_seen", "currency",
    "first_date", "last_date", "amount_conflict", "financial_impact", "impact_reliable",
    "source_rows", "details",
]
REFUND_COLUMNS = [
    "refund_id", "original_transaction_id", "refund_date", "refund_amount", "currency",
    "refund_status", "reason", "original_amount", "original_date", "cumulative_refunded",
]
ROUNDING_DETAIL_COLUMNS = [
    "transaction_id", "currency", "platform_date", "bank_date", "platform_amount",
    "bank_amount", "amount_difference", "abs_difference",
]
ROUNDING_SUMMARY_COLUMNS = [
    "currency", "transactions", "net_difference", "total_abs_difference",
    "largest_single_difference", "average_difference", "cumulative_exceeds_tolerance", "comment",
]
DISCREPANCY_COLUMNS = [
    "discrepancy_id", "category", "severity", "transaction_id", "dataset", "currency",
    "amount_impact", "details",
]


def _plain(value: Optional[Decimal]) -> str:
    return "n/a" if value is None else format(value, "f")


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------
def _duplicates_in(df: pd.DataFrame, dataset: str) -> list[dict]:
    is_platform = dataset == "Platform"
    amount_col = "signed_amount" if is_platform else "settlement_amount"
    date_col = "transaction_date" if is_platform else "settlement_date"
    rows: list[dict] = []

    for tid, group in df.groupby("transaction_id", sort=False):
        if len(group) < 2:
            continue
        occurrences = len(group)
        amounts = list(dict.fromkeys(group[amount_col]))
        currencies = sorted(set(group["currency"]))
        statuses = set(group["status"]) if is_platform else {"n/a"}
        consistent = len(amounts) == 1 and len(currencies) == 1 and len(statuses) == 1

        impact: Optional[Decimal] = None
        if consistent:
            counts_as_money = (not is_platform) or ("SUCCESS" in statuses)
            impact = amounts[0] * (occurrences - 1) if counts_as_money else ZERO

        if consistent:
            details = f"{occurrences} identical records; {occurrences - 1} extra cop{'y' if occurrences == 2 else 'ies'}."
        else:
            details = (f"{occurrences} records with conflicting amounts, currencies or statuses; "
                       "financial impact cannot be calculated reliably.")
        rows.append({
            "dataset": dataset,
            "transaction_id": tid,
            "occurrences": occurrences,
            "extra_copies": occurrences - 1,
            "amounts_seen": " / ".join(_plain(a) for a in amounts),
            "currency": "/".join(currencies),
            "first_date": min(group[date_col]),
            "last_date": max(group[date_col]),
            "amount_conflict": len(amounts) > 1,
            "financial_impact": impact,
            "impact_reliable": consistent,
            "source_rows": ", ".join(str(r) for r in group["source_row"]),
            "details": details,
        })
    return rows


def detect_duplicates(platform: pd.DataFrame, bank: pd.DataFrame) -> pd.DataFrame:
    """Find transaction IDs that appear more than once, separately per dataset.

    ``financial_impact`` is the value of the extra copies as recorded in that
    dataset (platform refunds are negative). It is ``None`` when the copies
    disagree on amount/currency/status, because the real impact is then unknown.
    """
    rows = _duplicates_in(platform, "Platform") + _duplicates_in(bank, "Bank")
    return pd.DataFrame(rows, columns=DUPLICATE_COLUMNS)


# ---------------------------------------------------------------------------
# Refunds
# ---------------------------------------------------------------------------
def detect_unmatched_refunds(platform: pd.DataFrame) -> pd.DataFrame:
    """Review every refund against the platform's original payments.

    Statuses:
      * Valid               - original payment found and the refund is consistent.
      * Unmatched Refund    - the referenced original transaction does not exist.
      * Invalid Reference   - blank reference, self reference, points to another
                              refund, or points to a payment that failed.
      * Inconsistent Refund - original exists but currency, date or amount do not
                              make sense (e.g. refund larger than the payment, or
                              total refunds exceeding the payment).
    """
    first = platform.drop_duplicates(subset="transaction_id", keep="first")
    payments = {r["transaction_id"]: r for r in first[first["transaction_type"] == "PAYMENT"].to_dict("records")}
    refund_ids = set(first.loc[first["transaction_type"] == "REFUND", "transaction_id"])
    refunds = first[first["transaction_type"] == "REFUND"].sort_values(["transaction_date", "source_row"], kind="stable")

    cumulative: dict[str, Decimal] = {}
    rows: list[dict] = []

    for r in refunds.to_dict("records"):
        ref = r["original_transaction_id"]
        out = {
            "refund_id": r["transaction_id"], "original_transaction_id": ref,
            "refund_date": r["transaction_date"], "refund_amount": r["amount"], "currency": r["currency"],
            "refund_status": REFUND_VALID, "reason": "Original payment found and refund is consistent.",
            "original_amount": None, "original_date": None, "cumulative_refunded": None,
        }
        original = payments.get(ref) if ref else None

        if not ref:
            out.update(refund_status=REFUND_INVALID_REF, reason="The refund has no original_transaction_id.")
        elif ref == r["transaction_id"]:
            out.update(refund_status=REFUND_INVALID_REF, reason="The refund references itself.")
        elif ref in refund_ids:
            out.update(refund_status=REFUND_INVALID_REF, reason="The refund references another refund, not a payment.")
        elif original is None:
            out.update(
                refund_status=REFUND_UNMATCHED,
                reason="Original transaction not found in the platform payment records "
                       "(it may belong to a period outside this file).",
            )
        elif original["status"] != "SUCCESS":
            out.update(
                refund_status=REFUND_INVALID_REF,
                reason=f"The original payment has status {original['status']}, so nothing can be refunded.",
                original_amount=original["amount"], original_date=original["transaction_date"],
            )
        else:
            running = cumulative.get(ref, ZERO) + r["amount"]
            cumulative[ref] = running
            out.update(original_amount=original["amount"], original_date=original["transaction_date"], cumulative_refunded=running)
            if original["currency"] != r["currency"]:
                out.update(refund_status=REFUND_INCONSISTENT,
                           reason=f"Refund currency {r['currency']} differs from original {original['currency']}.")
            elif r["transaction_date"] < original["transaction_date"]:
                out.update(refund_status=REFUND_INCONSISTENT, reason="The refund is dated before the original payment.")
            elif r["amount"] > original["amount"]:
                out.update(refund_status=REFUND_INCONSISTENT,
                           reason=f"Refund {_plain(r['amount'])} exceeds the original payment {_plain(original['amount'])}.")
            elif running > original["amount"]:
                out.update(refund_status=REFUND_INCONSISTENT,
                           reason=f"Total refunded {_plain(running)} exceeds the original payment {_plain(original['amount'])}.")
        rows.append(out)

    return pd.DataFrame(rows, columns=REFUND_COLUMNS)


# ---------------------------------------------------------------------------
# Rounding
# ---------------------------------------------------------------------------
def _week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _summarise(items: list[dict], tolerance: Decimal) -> dict:
    diffs = [i["amount_difference"] for i in items]
    net = sum(diffs, ZERO)
    total_abs = sum((abs(x) for x in diffs), ZERO)
    exceeds = abs(net) > tolerance
    if exceeds:
        comment = (f"Each difference is within {_plain(tolerance)}, but together they add up to "
                   f"{_plain(net)} (more than one tolerance).")
    else:
        comment = "Cumulative difference stays within the tolerance."
    return {
        "transactions": len(items),
        "net_difference": net,
        "total_abs_difference": total_abs,
        "largest_single_difference": max(abs(x) for x in diffs),
        "average_difference": (net / len(items)).quantize(Decimal("0.0001")),
        "cumulative_exceeds_tolerance": exceeds,
        "comment": comment,
    }


def analyze_rounding(matches: pd.DataFrame, config: Optional[ReconConfig] = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (individual differences, cumulative by currency, cumulative by settlement week).

    Individual differences are the per-transaction gaps. The cumulative tables sum
    those gaps so that a bias (e.g. the bank always truncating) becomes visible
    even though each single gap looks harmless.
    """
    config = config or ReconConfig()
    rows = matches[matches["match_status"] == ROUNDING_DIFFERENCE]

    details = pd.DataFrame(
        [{
            "transaction_id": r["transaction_id"], "currency": r["platform_currency"],
            "platform_date": r["platform_date"], "bank_date": r["bank_date"],
            "platform_amount": r["platform_amount"], "bank_amount": r["bank_amount"],
            "amount_difference": r["amount_difference"], "abs_difference": abs(r["amount_difference"]),
        } for r in rows.to_dict("records")],
        columns=ROUNDING_DETAIL_COLUMNS,
    )

    items = details.to_dict("records")
    by_currency_rows, by_period_rows = [], []
    for currency in sorted({i["currency"] for i in items}):
        subset = [i for i in items if i["currency"] == currency]
        by_currency_rows.append({"currency": currency, **_summarise(subset, config.rounding_tolerance)})
        for week in sorted({_week_start(i["bank_date"]) for i in subset}):
            week_items = [i for i in subset if _week_start(i["bank_date"]) == week]
            by_period_rows.append({"currency": currency, "settlement_week_start": week, **_summarise(week_items, config.rounding_tolerance)})

    by_currency = pd.DataFrame(by_currency_rows, columns=ROUNDING_SUMMARY_COLUMNS)
    by_period = pd.DataFrame(by_period_rows, columns=["currency", "settlement_week_start"] + ROUNDING_SUMMARY_COLUMNS[1:])
    return details, by_currency, by_period


# ---------------------------------------------------------------------------
# Unified discrepancy report
# ---------------------------------------------------------------------------
_SEVERITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}


def build_discrepancy_table(matches: pd.DataFrame, duplicates: pd.DataFrame, refunds: pd.DataFrame) -> pd.DataFrame:
    """Combine every finding into one table, sorted by severity."""
    rows: list[dict] = []

    for r in matches.to_dict("records"):
        status = r["match_status"]
        if status == ROUNDING_DIFFERENCE:
            rows.append(dict(category=CAT_ROUNDING, severity="Low", transaction_id=r["transaction_id"], dataset="Platform vs Bank",
                             currency=r["platform_currency"], amount_impact=r["amount_difference"], details=r["note"]))
        elif status == AMOUNT_MISMATCH:
            rows.append(dict(category=CAT_AMOUNT_MISMATCH, severity="High", transaction_id=r["transaction_id"], dataset="Platform vs Bank",
                             currency=r["platform_currency"], amount_impact=r["amount_difference"], details=r["note"]))
        elif status == MISSING_FROM_BANK:
            rows.append(dict(category=CAT_MISSING_BANK, severity="High", transaction_id=r["transaction_id"], dataset="Platform",
                             currency=r["platform_currency"], amount_impact=r["platform_amount"], details=r["note"]))
        elif status == MISSING_FROM_PLATFORM:
            rows.append(dict(category=CAT_MISSING_PLATFORM, severity="High", transaction_id=r["transaction_id"], dataset="Bank",
                             currency=r["bank_currency"], amount_impact=r["bank_amount"], details=r["note"]))
        elif status == STATUS_CONFLICT:
            rows.append(dict(category=CAT_STATUS_CONFLICT, severity="Medium", transaction_id=r["transaction_id"], dataset="Platform vs Bank",
                             currency=r["bank_currency"], amount_impact=r["bank_amount"], details=r["note"]))

    for d in duplicates.to_dict("records"):
        severity = "High" if (d["dataset"] == "Bank" or d["amount_conflict"]) else "Medium"
        rows.append(dict(category=CAT_DUPLICATE, severity=severity, transaction_id=d["transaction_id"], dataset=d["dataset"],
                         currency=d["currency"], amount_impact=d["financial_impact"],
                         details=f"{d['dataset']}: {d['details']}"))

    for f in refunds.to_dict("records"):
        if f["refund_status"] in (REFUND_UNMATCHED, REFUND_INVALID_REF):
            rows.append(dict(category=CAT_UNMATCHED_REFUND, severity="High", transaction_id=f["refund_id"], dataset="Platform",
                             currency=f["currency"], amount_impact=f["refund_amount"],
                             details=f"{f['refund_status']}: {f['reason']}"))
        elif f["refund_status"] == REFUND_INCONSISTENT:
            rows.append(dict(category=CAT_REFUND_INCONSISTENT, severity="Medium", transaction_id=f["refund_id"], dataset="Platform",
                             currency=f["currency"], amount_impact=f["refund_amount"], details=f["reason"]))

    table = pd.DataFrame(rows, columns=[c for c in DISCREPANCY_COLUMNS if c != "discrepancy_id"])
    if table.empty:
        return pd.DataFrame(columns=DISCREPANCY_COLUMNS)
    table["_sev"] = table["severity"].map(_SEVERITY_ORDER)
    table = table.sort_values(["_sev", "category", "transaction_id"], kind="stable").drop(columns="_sev").reset_index(drop=True)
    table.insert(0, "discrepancy_id", [f"DIS-{i:04d}" for i in range(1, len(table) + 1)])
    return table
