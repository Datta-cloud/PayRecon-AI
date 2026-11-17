"""Matching, pending settlements, the amount bridge and the generated sample data."""
from datetime import date

import pandas as pd

from config import (
    AMOUNT_MISMATCH,
    MATCHED,
    MISSING_FROM_BANK,
    MISSING_FROM_PLATFORM,
    NOT_SETTLEABLE,
    PENDING_SETTLEMENT,
    STATUS_CONFLICT,
    ReconConfig,
)
from data.generate_data import generate_datasets, write_datasets
from reconciliation.data_loader import load_csv
from reconciliation.reconciliation_engine import run_reconciliation
from tests.helpers import D, bank_df, empty_bank_but_valid, platform_df, run, status_of


def discrepancy_ids(result, txn_id):
    return result.discrepancies[result.discrepancies["transaction_id"] == txn_id]


# ---------------------------------------------------------------------------
# Basic matching
# ---------------------------------------------------------------------------
def test_same_id_and_amount_is_matched_with_no_discrepancies():
    result = run(platform_df({}), bank_df({}))
    assert status_of(result, "T1") == MATCHED
    assert result.discrepancies.empty
    assert result.summary["matched"] == 1
    assert result.summary["completion_status"] == "Fully reconciled"
    assert result.matches.iloc[0]["amount_difference"] == D("0.00")


def test_bank_amount_larger_than_tolerance_is_an_amount_mismatch():
    result = run(platform_df({"amount": "100.00"}), bank_df({"settlement_amount": "90.00"}))
    assert status_of(result, "T1") == AMOUNT_MISMATCH
    finding = result.discrepancies.iloc[0]
    assert finding["category"] == "Amount Mismatch" and finding["severity"] == "High"
    assert finding["amount_impact"] == D("10.00")


def test_currency_difference_is_an_amount_mismatch():
    result = run(platform_df({"currency": "INR"}), bank_df({"currency": "USD"}))
    assert status_of(result, "T1") == AMOUNT_MISMATCH
    assert "Currency differs" in result.matches.iloc[0]["note"]


def test_bank_record_without_platform_record_is_missing_from_platform():
    result = run(platform_df({}), bank_df({}, {"transaction_id": "GHOST", "settlement_amount": "25.00"}))
    assert status_of(result, "GHOST") == MISSING_FROM_PLATFORM
    assert discrepancy_ids(result, "GHOST").iloc[0]["category"] == "Missing from Platform"


# ---------------------------------------------------------------------------
# Settlement window
# ---------------------------------------------------------------------------
def test_unsettled_payment_inside_the_window_is_pending_not_an_error():
    for as_of in (date(2026, 10, 11), date(2026, 10, 12)):  # age 1 and age 2
        result = run(platform_df({"transaction_date": "2026-10-10"}), empty_bank_but_valid(), as_of_date=as_of)
        assert status_of(result, "T1") == PENDING_SETTLEMENT
        assert discrepancy_ids(result, "T1").empty
        assert result.pending.iloc[0]["transaction_id"] == "T1"
        assert result.summary["pending_settlements"] == 1


def test_unsettled_payment_outside_the_window_is_missing_from_bank():
    result = run(platform_df({"transaction_date": "2026-10-10"}), empty_bank_but_valid(), as_of_date=date(2026, 10, 13))  # age 3
    assert status_of(result, "T1") == MISSING_FROM_BANK
    finding = discrepancy_ids(result, "T1").iloc[0]
    assert finding["category"] == "Missing from Bank" and finding["severity"] == "High"


def test_settlement_window_is_configurable():
    platform = platform_df({"transaction_date": "2026-10-10"})
    result = run(platform, empty_bank_but_valid(), as_of_date=date(2026, 10, 13), settlement_max_days=3)
    assert status_of(result, "T1") == PENDING_SETTLEMENT


def test_default_reporting_date_is_the_latest_date_in_the_data():
    platform = platform_df({"transaction_id": "T1", "transaction_date": "2026-10-10"},
                           {"transaction_id": "T2", "transaction_date": "2026-10-20"})
    bank = bank_df({"transaction_id": "T2", "settlement_date": "2026-10-21"})
    result = run(platform, bank)
    assert result.as_of_date == date(2026, 10, 21)
    assert status_of(result, "T1") == MISSING_FROM_BANK  # 11 days old
    assert status_of(result, "T2") == MATCHED


def test_late_settlement_is_flagged_but_still_matched():
    result = run(platform_df({"transaction_date": "2026-10-01"}), bank_df({"settlement_date": "2026-10-05"}))
    row = result.matches.iloc[0]
    assert row["match_status"] == MATCHED
    assert row["settlement_lag_days"] == 4 and row["timing_flag"].startswith("Late settlement")
    assert result.summary["late_settlements"] == 1


def test_settlement_within_one_to_two_days_has_no_timing_flag_issue():
    for settle in ("2026-10-11", "2026-10-12"):
        result = run(platform_df({}), bank_df({"settlement_date": settle}))
        assert result.matches.iloc[0]["timing_flag"] == "Within window"
        assert result.summary["late_settlements"] == 0


# ---------------------------------------------------------------------------
# Failed payments
# ---------------------------------------------------------------------------
def test_failed_payment_without_settlement_is_not_settleable_and_not_an_error():
    result = run(platform_df({"status": "FAILED"}), empty_bank_but_valid(), as_of_date=date(2026, 11, 30))
    assert status_of(result, "T1") == NOT_SETTLEABLE
    assert discrepancy_ids(result, "T1").empty


def test_failed_payment_that_the_bank_settled_is_a_status_conflict():
    result = run(platform_df({"status": "FAILED"}), bank_df({}))
    assert status_of(result, "T1") == STATUS_CONFLICT
    assert discrepancy_ids(result, "T1").iloc[0]["category"] == "Status Conflict"


# ---------------------------------------------------------------------------
# The bridge: every unit of the platform-vs-bank gap must be explained
# ---------------------------------------------------------------------------
def _mixed_scenario():
    platform = platform_df(
        {"transaction_id": "T1", "amount": "100.00"},                               # matched
        {"transaction_id": "T2", "amount": "100.00"},                               # rounding (bank 99.99)
        {"transaction_id": "T3", "amount": "100.00"},                               # mismatch (bank 90.00)
        {"transaction_id": "T4", "amount": "50.00", "transaction_date": "2026-10-10"},  # pending
        {"transaction_id": "T5", "amount": "70.00", "transaction_date": "2026-10-01"},  # missing from bank
        {"transaction_id": "T6", "amount": "20.00"}, {"transaction_id": "T6", "amount": "20.00"},  # platform duplicate
        {"transaction_id": "T7", "amount": "30.00", "status": "FAILED"},            # status conflict
        {"transaction_id": "T8", "amount": "10.00"},                                # bank duplicate
    )
    bank = bank_df(
        {"transaction_id": "T1", "settlement_amount": "100.00"},
        {"transaction_id": "T2", "settlement_amount": "99.99"},
        {"transaction_id": "T3", "settlement_amount": "90.00"},
        {"transaction_id": "T6", "settlement_amount": "20.00"},
        {"transaction_id": "T7", "settlement_amount": "30.00"},
        {"transaction_id": "T8", "settlement_amount": "10.00"}, {"transaction_id": "T8", "settlement_amount": "10.00"},
        {"transaction_id": "GHOST", "settlement_amount": "25.00"},                  # missing from platform
    )
    return platform, bank


def test_mixed_scenario_counts_every_category_once():
    result = run(*_mixed_scenario(), as_of_date=date(2026, 10, 11))
    s = result.summary
    assert (s["matched"], s["rounding_differences"], s["amount_mismatches"]) == (3, 1, 1)
    assert (s["pending_settlements"], s["missing_from_bank"], s["missing_from_platform"], s["status_conflicts"]) == (1, 1, 1, 1)
    assert s["duplicate_entries"] == 2
    assert s["total_discrepancies"] == 7  # pending is NOT a discrepancy
    assert s["completion_status"] == "Needs review: 7 discrepancies found"


def test_bridge_explains_the_whole_gap_with_nothing_unexplained():
    result = run(*_mixed_scenario(), as_of_date=date(2026, 10, 11))
    row = result.bridge.iloc[0]
    assert row["platform_total"] == D("470.00") and row["bank_total"] == D("384.99")
    assert row["difference"] == D("85.01")
    assert row["amount_differences"] == D("10.01")
    assert row["pending_settlement"] == D("50.00") and row["missing_from_bank"] == D("70.00")
    assert row["missing_from_platform"] == D("-25.00") and row["status_conflict"] == D("-30.00")
    assert row["platform_duplicates"] == D("20.00") and row["bank_duplicates"] == D("-10.00")
    assert row["unexplained"] == D("0.00")
    assert result.summary["total_amount_difference"] == {"INR": D("85.01")}


def test_discrepancy_table_is_sorted_by_severity_with_stable_ids():
    result = run(*_mixed_scenario(), as_of_date=date(2026, 10, 11))
    table = result.discrepancies
    assert list(table["discrepancy_id"]) == [f"DIS-{i:04d}" for i in range(1, 8)]
    order = {"High": 0, "Medium": 1, "Low": 2}
    ranks = [order[s] for s in table["severity"]]
    assert ranks == sorted(ranks)


# ---------------------------------------------------------------------------
# Generated sample data: the engine must find exactly what was injected
# ---------------------------------------------------------------------------
def _ids(result, status):
    return set(result.with_status(status)["transaction_id"])


def _category_ids(result, category):
    table = result.discrepancies
    return set(table.loc[table["category"] == category, "transaction_id"])


def test_generator_is_reproducible_for_a_fixed_seed():
    first, second = generate_datasets(42), generate_datasets(42)
    assert first.platform.equals(second.platform) and first.bank.equals(second.bank)
    assert first.truth == second.truth
    assert not generate_datasets(7).platform.equals(first.platform)


def test_engine_finds_exactly_the_problems_injected_into_the_sample_data():
    data = generate_datasets(42)
    result = run_reconciliation(data.platform, data.bank, ReconConfig())
    truth = data.truth

    assert _ids(result, "Rounding Difference") == set(truth["rounding"])
    assert _ids(result, AMOUNT_MISMATCH) == set(truth["amount_mismatch"])
    assert _ids(result, MISSING_FROM_BANK) == set(truth["missing_from_bank"])
    assert _ids(result, MISSING_FROM_PLATFORM) == set(truth["missing_from_platform"])
    assert _ids(result, PENDING_SETTLEMENT) == set(truth["pending"])
    assert _ids(result, NOT_SETTLEABLE) == set(truth["failed"])

    dups = result.duplicates
    assert set(dups.loc[dups["dataset"] == "Platform", "transaction_id"]) == set(truth["platform_duplicates"])
    assert set(dups.loc[dups["dataset"] == "Bank", "transaction_id"]) == set(truth["bank_duplicates"])

    assert _category_ids(result, "Unmatched Refund") == set(truth["unmatched_refunds"])
    assert _category_ids(result, "Refund Inconsistency") == set(truth["inconsistent_refunds"])
    refunds = result.refunds.set_index("refund_id")
    assert all(refunds.loc[r, "refund_status"] == "Valid" for r in truth["valid_refunds"])

    assert result.summary["late_settlements"] == len(truth["late_settlement"])


def test_sample_data_bridge_has_nothing_unexplained_and_rounding_adds_up():
    data = generate_datasets(42)
    result = run_reconciliation(data.platform, data.bank, ReconConfig())
    assert (result.bridge["unexplained"] == D("0")).all()
    # 10 x 0.01 + 2 x 0.02 - 0.01 = 0.13: small gaps that only matter when summed
    inr_or_usd = result.rounding_by_currency["net_difference"].sum()
    assert inr_or_usd == D("0.13")


def test_written_csv_files_reconcile_to_the_same_result_as_in_memory_data(tmp_path):
    platform_path, bank_path = tmp_path / "p.csv", tmp_path / "b.csv"
    data = write_datasets(42, platform_path, bank_path)
    from_files = run_reconciliation(load_csv(platform_path), load_csv(bank_path), ReconConfig())
    in_memory = run_reconciliation(data.platform, data.bank, ReconConfig())
    assert from_files.summary == in_memory.summary
    assert isinstance(from_files.matches, pd.DataFrame) and len(from_files.matches) == len(in_memory.matches)
