"""Refunds: valid refunds must pass, orphaned or inconsistent ones must be flagged."""
from config import (
    CAT_REFUND_INCONSISTENT,
    CAT_UNMATCHED_REFUND,
    MATCHED,
    REFUND_INCONSISTENT,
    REFUND_INVALID_REF,
    REFUND_UNMATCHED,
    REFUND_VALID,
)
from tests.helpers import D, bank_df, platform_df, run, status_of


def refund(txn_id="R1", original="P1", amount="40.00", date="2026-10-12", **extra):
    return {"transaction_id": txn_id, "transaction_type": "REFUND", "amount": amount,
            "original_transaction_id": original, "transaction_date": date, **extra}


def refund_status(result, refund_id="R1"):
    row = result.refunds[result.refunds["refund_id"] == refund_id]
    assert len(row) == 1
    return row.iloc[0]["refund_status"]


def test_refund_whose_original_does_not_exist_is_an_unmatched_refund():
    platform = platform_df(refund("R1", original="DOES-NOT-EXIST"))
    result = run(platform, bank_df({"transaction_id": "R1", "settlement_amount": "-40.00", "settlement_date": "2026-10-13"}))

    assert refund_status(result) == REFUND_UNMATCHED
    finding = result.discrepancies[result.discrepancies["category"] == CAT_UNMATCHED_REFUND]
    assert len(finding) == 1 and finding.iloc[0]["transaction_id"] == "R1"
    assert "not found" in finding.iloc[0]["details"]
    assert result.summary["unmatched_refunds"] == 1


def test_valid_refund_is_not_reported_as_unmatched():
    platform = platform_df({"transaction_id": "P1"}, refund("R1", original="P1"))
    bank = bank_df({"transaction_id": "P1"}, {"transaction_id": "R1", "settlement_amount": "-40.00", "settlement_date": "2026-10-13"})
    result = run(platform, bank)

    assert refund_status(result) == REFUND_VALID
    assert result.discrepancies.empty
    assert status_of(result, "R1") == MATCHED  # refund settled with a negative bank amount
    assert result.summary["valid_refunds"] == 1


def test_refund_listed_before_its_original_is_still_valid():
    platform = platform_df(refund("R1", original="P1"), {"transaction_id": "P1", "transaction_date": "2026-10-10"})
    result = run(platform, bank_df({"transaction_id": "P1"}))
    assert refund_status(result) == REFUND_VALID


def test_blank_reference_is_an_invalid_reference():
    result = run(platform_df(refund("R1", original="")), bank_df({"transaction_id": "R1", "settlement_amount": "-40.00"}))
    assert refund_status(result) == REFUND_INVALID_REF
    finding = result.discrepancies[result.discrepancies["transaction_id"] == "R1"]
    assert list(finding["category"]) == [CAT_UNMATCHED_REFUND]
    assert "no original_transaction_id" in finding.iloc[0]["details"]


def test_refund_of_a_refund_and_self_reference_are_invalid():
    platform = platform_df({"transaction_id": "P1"}, refund("R1", original="P1"), refund("R2", original="R1"), refund("R3", original="R3"))
    result = run(platform, bank_df({}))
    assert refund_status(result, "R1") == REFUND_VALID
    assert refund_status(result, "R2") == REFUND_INVALID_REF
    assert refund_status(result, "R3") == REFUND_INVALID_REF


def test_refund_of_a_failed_payment_is_invalid():
    platform = platform_df({"transaction_id": "P1", "status": "FAILED"}, refund("R1", original="P1"))
    result = run(platform, bank_df({}))
    assert refund_status(result) == REFUND_INVALID_REF


def test_refund_larger_than_the_payment_is_inconsistent():
    platform = platform_df({"transaction_id": "P1", "amount": "100.00"}, refund("R1", original="P1", amount="150.00"))
    result = run(platform, bank_df({}))
    assert refund_status(result) == REFUND_INCONSISTENT
    finding = result.discrepancies[result.discrepancies["category"] == CAT_REFUND_INCONSISTENT]
    assert len(finding) == 1 and "exceeds" in finding.iloc[0]["details"]


def test_partial_refunds_within_the_total_are_valid():
    platform = platform_df(
        {"transaction_id": "P1", "amount": "100.00"},
        refund("R1", original="P1", amount="40.00", date="2026-10-12"),
        refund("R2", original="P1", amount="60.00", date="2026-10-13"),
    )
    result = run(platform, bank_df({}))
    assert refund_status(result, "R1") == REFUND_VALID and refund_status(result, "R2") == REFUND_VALID
    assert result.refunds.set_index("refund_id").loc["R2", "cumulative_refunded"] == D("100.00")


def test_partial_refunds_that_add_up_to_more_than_the_payment_are_inconsistent():
    platform = platform_df(
        {"transaction_id": "P1", "amount": "100.00"},
        refund("R1", original="P1", amount="70.00", date="2026-10-12"),
        refund("R2", original="P1", amount="50.00", date="2026-10-13"),
    )
    result = run(platform, bank_df({}))
    assert refund_status(result, "R1") == REFUND_VALID
    assert refund_status(result, "R2") == REFUND_INCONSISTENT
    assert "Total refunded 120.00" in result.refunds.set_index("refund_id").loc["R2", "reason"]


def test_refund_dated_before_the_payment_or_in_another_currency_is_inconsistent():
    platform = platform_df(
        {"transaction_id": "P1", "transaction_date": "2026-10-10"},
        refund("R1", original="P1", date="2026-10-05"),
        refund("R2", original="P1", currency="USD", date="2026-10-12"),
    )
    result = run(platform, bank_df({}))
    assert refund_status(result, "R1") == REFUND_INCONSISTENT
    assert refund_status(result, "R2") == REFUND_INCONSISTENT


def test_no_refunds_gives_an_empty_review_table():
    result = run(platform_df({}), bank_df({}))
    assert result.refunds.empty and result.summary["unmatched_refunds"] == 0
