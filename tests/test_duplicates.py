"""Duplicates: detected separately per dataset, never silently removed."""
from config import CAT_DUPLICATE, MATCHED
from tests.helpers import D, bank_df, platform_df, run, status_of


def duplicate_rows(result, dataset=None):
    rows = result.duplicates
    return rows if dataset is None else rows[rows["dataset"] == dataset]


def test_same_transaction_id_twice_in_platform_is_a_duplicate_entry():
    result = run(platform_df({}, {}), bank_df({}))

    dup = duplicate_rows(result)
    assert len(dup) == 1
    row = dup.iloc[0]
    assert row["dataset"] == "Platform" and row["transaction_id"] == "T1"
    assert row["occurrences"] == 2 and row["extra_copies"] == 1
    assert row["financial_impact"] == D("100.00") and row["impact_reliable"]
    assert row["source_rows"] == "2, 3"  # CSV line numbers of both copies

    finding = result.discrepancies[result.discrepancies["category"] == CAT_DUPLICATE]
    assert len(finding) == 1 and finding.iloc[0]["severity"] == "Medium"
    assert result.summary["platform_duplicate_ids"] == 1 and result.summary["bank_duplicate_ids"] == 0


def test_same_transaction_id_twice_in_bank_is_a_duplicate_entry_with_high_severity():
    result = run(platform_df({}), bank_df({"settlement_id": "S1"}, {"settlement_id": "S2"}))

    dup = duplicate_rows(result)
    assert len(dup) == 1 and dup.iloc[0]["dataset"] == "Bank"
    assert dup.iloc[0]["financial_impact"] == D("100.00")  # the bank paid out an extra 100.00
    assert result.discrepancies.iloc[0]["severity"] == "High"
    assert result.summary["bank_duplicate_ids"] == 1


def test_duplicates_are_reported_but_not_removed_from_the_data():
    result = run(platform_df({}, {}), bank_df({}))
    assert len(result.platform) == 2  # nothing silently dropped
    assert len(result.matches) == 1  # but the transaction is matched once
    assert status_of(result, "T1") == MATCHED


def test_duplicate_in_both_datasets_gives_one_finding_per_dataset():
    result = run(platform_df({}, {}), bank_df({"settlement_id": "S1"}, {"settlement_id": "S2"}))
    assert sorted(duplicate_rows(result)["dataset"]) == ["Bank", "Platform"]
    assert result.summary["duplicate_entries"] == 2
    assert (result.discrepancies["category"] == CAT_DUPLICATE).sum() == 2
    # both sides doubled, so the totals still agree and the bridge explains the whole gap
    assert result.bridge.iloc[0]["difference"] == D("0.00")
    assert result.bridge.iloc[0]["unexplained"] == D("0.00")


def test_three_copies_count_two_extra():
    result = run(platform_df({}, {}, {}), bank_df({}))
    row = duplicate_rows(result).iloc[0]
    assert row["occurrences"] == 3 and row["extra_copies"] == 2
    assert row["financial_impact"] == D("200.00")
    assert "2 extra copies" in row["details"]


def test_duplicates_with_conflicting_amounts_have_no_reliable_impact():
    result = run(platform_df({"amount": "100.00"}, {"amount": "110.00"}), bank_df({"settlement_amount": "100.00"}))
    row = duplicate_rows(result).iloc[0]
    assert row["amount_conflict"] and not row["impact_reliable"]
    assert row["financial_impact"] is None
    assert row["amounts_seen"] == "100.00 / 110.00"
    assert result.discrepancies.iloc[0]["severity"] == "High"
    assert result.bridge.iloc[0]["unexplained"] == D("0.00")  # the totals still reconcile


def test_duplicated_refund_has_a_negative_impact():
    refund = {"transaction_id": "R1", "transaction_type": "REFUND", "amount": "40.00", "original_transaction_id": "P1"}
    result = run(platform_df({"transaction_id": "P1"}, refund, refund), bank_df({"transaction_id": "P1"}))
    row = duplicate_rows(result, "Platform").iloc[0]
    assert row["transaction_id"] == "R1"
    assert row["financial_impact"] == D("-40.00")


def test_unique_ids_produce_no_duplicates():
    result = run(platform_df({"transaction_id": "A"}, {"transaction_id": "B"}),
                 bank_df({"transaction_id": "A"}, {"transaction_id": "B"}))
    assert result.duplicates.empty
    assert result.summary["duplicate_entries"] == 0 and result.discrepancies.empty
