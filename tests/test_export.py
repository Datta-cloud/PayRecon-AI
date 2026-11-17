"""Exports: CSV row counts and summary figures must match the engine output."""
import io

import pandas as pd

from tests.helpers import bank_df, platform_df, run
from utils.export import discrepancy_csv, full_report_csv, summary_text
from utils.formatting import format_amount, format_amounts_by_currency
from decimal import Decimal


def _result():
    platform = platform_df({"transaction_id": "A", "amount": "100.00"}, {"transaction_id": "B"}, {"transaction_id": "B"})
    bank = bank_df({"transaction_id": "A", "settlement_amount": "99.99"}, {"transaction_id": "B"})
    return run(platform, bank)


def test_full_report_has_one_row_per_transaction_and_exact_amounts():
    result = _result()
    df = pd.read_csv(io.StringIO(full_report_csv(result)), dtype=str)
    assert len(df) == len(result.matches) == 2
    assert df.set_index("transaction_id").loc["A", "amount_difference"] == "0.01"
    assert df.set_index("transaction_id").loc["A", "match_status"] == "Rounding Difference"


def test_discrepancy_report_lists_exactly_the_findings():
    result = _result()
    df = pd.read_csv(io.StringIO(discrepancy_csv(result)))
    assert len(df) == len(result.discrepancies) == 2
    assert sorted(df["category"]) == ["Duplicate Entry", "Rounding Difference"]


def test_summary_contains_every_required_figure():
    text = summary_text(_result())
    assert "Total platform records:      3" in text
    assert "Matched transactions:        1" in text
    assert "Pending settlements:         0" in text
    assert "Total discrepancies:         2" in text
    assert "Rounding differences:      1" in text
    assert "Duplicate entries:         1" in text
    assert "Unmatched refunds:         0" in text
    assert "Completion status: Needs review: 2 discrepancies found" in text


def test_empty_discrepancy_report_still_has_a_header():
    result = run(platform_df({}), bank_df({}))
    assert discrepancy_csv(result).startswith("discrepancy_id,category")
    assert "Completion status: Fully reconciled" in summary_text(result)


def test_formatting_helpers():
    assert format_amount(Decimal("1234.5")) == "1,234.50"
    assert format_amount(None) == "-"
    assert format_amounts_by_currency({"USD": Decimal("-2"), "INR": Decimal("0.1")}) == "INR 0.10 | USD -2.00"
