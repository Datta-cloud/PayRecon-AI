"""Build the downloadable reports (CSV and plain-text summary)."""
from __future__ import annotations

from config import APP_NAME
from reconciliation.reconciliation_engine import ReconciliationResult
from utils.formatting import format_amounts_by_currency


def full_report_csv(result: ReconciliationResult) -> str:
    """Every transaction with its match status, amounts, lag and note."""
    return result.matches.to_csv(index=False)


def discrepancy_csv(result: ReconciliationResult) -> str:
    """Only the findings that need attention (pending settlements are not included)."""
    return result.discrepancies.to_csv(index=False)


def summary_text(result: ReconciliationResult) -> str:
    s = result.summary
    cfg = result.config
    total_transactions = s["total_platform_records"]
    lines = [
        f"{APP_NAME} - Reconciliation summary",
        f"Reporting date: {s['as_of_date']}",
        f"Settlement window: {cfg.settlement_min_days}-{cfg.settlement_max_days} days | Rounding tolerance: {cfg.rounding_tolerance}",
        "",
        f"Total platform records:      {total_transactions}",
        f"Total bank records:          {s['total_bank_records']}",
        f"Matched transactions:        {s['matched']}",
        f"Pending settlements:         {s['pending_settlements']}",
        f"Total discrepancies:         {s['total_discrepancies']}",
        f"  Rounding differences:      {s['rounding_differences']}",
        f"  Duplicate entries:         {s['duplicate_entries']} (platform {s['platform_duplicate_ids']}, bank {s['bank_duplicate_ids']})",
        f"  Unmatched refunds:         {s['unmatched_refunds']}",
        f"  Amount mismatches:         {s['amount_mismatches']}",
        f"  Missing from bank:         {s['missing_from_bank']}",
        f"  Missing from platform:     {s['missing_from_platform']}",
        f"  Status conflicts:          {s['status_conflicts']}",
        f"  Inconsistent refunds:      {s['inconsistent_refunds']}",
        "",
        f"Platform minus bank total:   {format_amounts_by_currency(s['total_amount_difference'])}",
        f"Unexplained difference:      {format_amounts_by_currency(s['unexplained_difference'])}",
        f"Net rounding difference:     {format_amounts_by_currency(s['cumulative_rounding'])}",
        f"Rows excluded as invalid:    platform {s['platform_invalid_rows']}, bank {s['bank_invalid_rows']}",
        "",
        f"Completion status: {s['completion_status']}",
    ]
    return "\n".join(lines) + "\n"
