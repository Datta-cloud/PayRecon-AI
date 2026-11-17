"""Loading and validation: bad files must produce clear errors, never silent data loss."""
from decimal import Decimal

import pandas as pd
import pytest

from config import BANK_CSV, PLATFORM_CSV, ReconConfig
from reconciliation.data_loader import DataLoadError, load_csv
from reconciliation.reconciliation_engine import ReconciliationInputError, run_reconciliation
from reconciliation.validator import parse_date, parse_decimal, validate_bank, validate_platform
from tests.helpers import bank_df, platform_df, run


def test_loader_normalises_headers_and_keeps_values_as_text():
    raw = b" Transaction ID ,Customer-ID,Amount\nT1,C1,1234.50\n"
    df = load_csv(raw)
    assert list(df.columns) == ["transaction_id", "customer_id", "amount"]
    assert df.loc[0, "amount"] == "1234.50"  # not turned into a float


def test_loader_rejects_completely_empty_file():
    with pytest.raises(DataLoadError, match="empty"):
        load_csv(b"", label="Platform CSV")


def test_loader_rejects_missing_file(tmp_path):
    with pytest.raises(DataLoadError, match="not found"):
        load_csv(tmp_path / "nope.csv")


def test_header_only_file_loads_but_validation_reports_no_rows():
    df = load_csv(b"transaction_id,customer_id,transaction_date,transaction_type,amount,currency,status\n")
    assert df.empty
    result = validate_platform(df)
    assert not result.ok
    assert "no data rows" in result.fatal_errors[0]


def test_missing_columns_are_reported_by_name():
    raw = pd.DataFrame({"transaction_id": ["T1"], "amount": ["10"]})
    result = validate_platform(raw)
    assert not result.ok
    message = result.fatal_errors[0]
    assert "missing required column" in message
    assert "customer_id" in message and "currency" in message


def test_bank_missing_columns():
    result = validate_bank(pd.DataFrame({"transaction_id": ["T1"]}))
    assert not result.ok and "settlement_amount" in result.fatal_errors[0]


def test_missing_transaction_id_row_is_excluded_but_reported():
    raw = platform_df({"transaction_id": "T1"}, {"transaction_id": ""}, {"transaction_id": "T3"})
    result = validate_platform(raw)
    assert result.ok
    assert list(result.valid["transaction_id"]) == ["T1", "T3"]
    problems = result.invalid_rows
    assert len(problems) == 1
    assert problems.iloc[0]["problem"] == "Missing transaction ID"
    assert problems.iloc[0]["row_number"] == 3  # header is row 1, so the 2nd data row is row 3


def test_bad_amount_date_and_currency_are_flagged_per_field():
    raw = platform_df(
        {"transaction_id": "A", "amount": "abc"},
        {"transaction_id": "B", "transaction_date": "not-a-date"},
        {"transaction_id": "C", "currency": "XYZ"},
        {"transaction_id": "D", "amount": "-5"},
        {"transaction_id": "E", "amount": "NaN"},
        {"transaction_id": "OK"},
    )
    result = validate_platform(raw)
    assert list(result.valid["transaction_id"]) == ["OK"]
    fields = sorted(result.invalid_rows["field"])
    assert fields == ["amount", "amount", "amount", "currency", "transaction_date"]


def test_amounts_are_exact_decimals():
    result = validate_platform(platform_df({"amount": "1,234.50"}))
    value = result.valid.loc[0, "amount"]
    assert isinstance(value, Decimal)
    assert value == Decimal("1234.50")


def test_refunds_get_negative_signed_amounts():
    result = validate_platform(platform_df({"transaction_type": "REFUND", "amount": "40.00", "original_transaction_id": "P"}))
    assert result.valid.loc[0, "signed_amount"] == Decimal("-40.00")


def test_parsers_return_none_for_garbage():
    assert parse_decimal("") is None
    assert parse_decimal("abc") is None and parse_decimal("inf") is None and parse_decimal("NaN") is None
    assert parse_date("garbage") is None and parse_date("") is None
    assert str(parse_date("2026-10-05 14:30:00")) == "2026-10-05"


def test_optional_original_transaction_id_column_can_be_absent():
    raw = platform_df({}).drop(columns=["original_transaction_id"])
    result = validate_platform(raw)
    assert result.ok and any("original_transaction_id" in w for w in result.warnings)


def test_engine_rejects_empty_datasets():
    empty_platform = platform_df().iloc[0:0]
    with pytest.raises(ReconciliationInputError) as info:
        run_reconciliation(empty_platform, bank_df({}), ReconConfig())
    assert any("no data rows" in e for e in info.value.errors)


def test_engine_rejects_invalid_columns_in_either_file():
    with pytest.raises(ReconciliationInputError) as info:
        run_reconciliation(pd.DataFrame({"foo": [1]}), pd.DataFrame({"bar": [2]}), ReconConfig())
    assert len(info.value.errors) == 2


def test_engine_rejects_file_where_every_row_is_invalid():
    with pytest.raises(ReconciliationInputError, match="none of the 1 rows are valid"):
        run_reconciliation(platform_df({"amount": "oops"}), bank_df({}), ReconConfig())


def test_invalid_rows_do_not_stop_the_rest_from_reconciling():
    result = run(platform_df({"transaction_id": "T1"}, {"transaction_id": ""}), bank_df({"transaction_id": "T1"}))
    assert result.summary["platform_invalid_rows"] == 1
    assert result.summary["matched"] == 1


def test_bundled_sample_files_load_and_reconcile():
    platform = load_csv(PLATFORM_CSV, "Platform CSV")
    bank = load_csv(BANK_CSV, "Bank CSV")
    assert 100 <= len(platform) <= 200 and 100 <= len(bank) <= 200
    result = run_reconciliation(platform, bank, ReconConfig())
    assert result.summary["platform_invalid_rows"] == 0 and result.summary["bank_invalid_rows"] == 0
