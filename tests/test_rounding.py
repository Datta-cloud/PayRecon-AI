"""Rounding differences: individual gaps and the cumulative effect when summed."""
from config import AMOUNT_MISMATCH, MATCHED, ROUNDING_DIFFERENCE
from tests.helpers import D, bank_df, platform_df, run, status_of


def test_100_vs_99_99_is_a_rounding_difference_within_tolerance():
    result = run(platform_df({"amount": "100.00"}), bank_df({"settlement_amount": "99.99"}))
    assert status_of(result, "T1") == ROUNDING_DIFFERENCE
    row = result.rounding_details.iloc[0]
    assert row["amount_difference"] == D("0.01")
    assert row["platform_amount"] == D("100.00") and row["bank_amount"] == D("99.99")
    assert list(result.discrepancies["category"]) == ["Rounding Difference"]


def test_exactly_equal_amounts_are_matched_not_rounding():
    result = run(platform_df({"amount": "100.00"}), bank_df({"settlement_amount": "100.00"}))
    assert status_of(result, "T1") == MATCHED
    assert result.rounding_details.empty


def test_difference_equal_to_tolerance_is_rounding_but_above_it_is_a_mismatch():
    at_limit = run(platform_df({"amount": "100.00"}), bank_df({"settlement_amount": "99.95"}), rounding_tolerance=D("0.05"))
    above = run(platform_df({"amount": "100.00"}), bank_df({"settlement_amount": "99.94"}), rounding_tolerance=D("0.05"))
    assert status_of(at_limit, "T1") == ROUNDING_DIFFERENCE
    assert status_of(above, "T1") == AMOUNT_MISMATCH


def test_tolerance_is_configurable():
    platform, bank = platform_df({"amount": "100.00"}), bank_df({"settlement_amount": "99.50"})
    assert status_of(run(platform, bank), "T1") == AMOUNT_MISMATCH  # default tolerance 0.05
    assert status_of(run(platform, bank, rounding_tolerance=D("1.00")), "T1") == ROUNDING_DIFFERENCE


def test_bank_paying_more_is_also_detected_with_a_negative_difference():
    result = run(platform_df({"amount": "100.00"}), bank_df({"settlement_amount": "100.01"}))
    assert result.rounding_details.iloc[0]["amount_difference"] == D("-0.01")


def _ten_penny_gaps():
    ids = [f"T{i}" for i in range(10)]
    platform = platform_df(*[{"transaction_id": t, "amount": "50.00"} for t in ids])
    bank = bank_df(*[{"transaction_id": t, "settlement_amount": "49.99"} for t in ids])
    return platform, bank


def test_tiny_individual_gaps_become_visible_when_summed():
    result = run(*_ten_penny_gaps(), rounding_tolerance=D("0.05"))
    assert result.summary["rounding_differences"] == 10

    individual = result.rounding_details
    assert (individual["abs_difference"] == D("0.01")).all()  # each one looks harmless

    cumulative = result.rounding_by_currency.iloc[0]
    assert cumulative["transactions"] == 10
    assert cumulative["net_difference"] == D("0.10")
    assert cumulative["largest_single_difference"] == D("0.01")
    assert cumulative["cumulative_exceeds_tolerance"]
    assert result.summary["cumulative_rounding"] == {"INR": D("0.10")}


def test_sums_are_exact_decimals_not_floats():
    # 10 x 0.10 in floating point is 0.9999999999999999; Decimal must give exactly 1.00
    ids = [f"T{i}" for i in range(10)]
    platform = platform_df(*[{"transaction_id": t, "amount": "10.10"} for t in ids])
    bank = bank_df(*[{"transaction_id": t, "settlement_amount": "10.00"} for t in ids])
    result = run(platform, bank, rounding_tolerance=D("0.10"))
    assert result.rounding_by_currency.iloc[0]["net_difference"] == D("1.00")


def test_opposite_gaps_cancel_out_in_the_cumulative_view():
    platform = platform_df({"transaction_id": "A"}, {"transaction_id": "B"})
    bank = bank_df({"transaction_id": "A", "settlement_amount": "99.99"}, {"transaction_id": "B", "settlement_amount": "100.01"})
    result = run(platform, bank)
    row = result.rounding_by_currency.iloc[0]
    assert row["net_difference"] == D("0.00") and row["total_abs_difference"] == D("0.02")
    assert not row["cumulative_exceeds_tolerance"]


def test_cumulative_totals_are_kept_separate_per_currency():
    platform = platform_df({"transaction_id": "A", "currency": "INR"}, {"transaction_id": "B", "currency": "USD"})
    bank = bank_df({"transaction_id": "A", "currency": "INR", "settlement_amount": "99.99"},
                   {"transaction_id": "B", "currency": "USD", "settlement_amount": "99.98"})
    result = run(platform, bank)
    by_currency = result.rounding_by_currency.set_index("currency")
    assert by_currency.loc["INR", "net_difference"] == D("0.01")
    assert by_currency.loc["USD", "net_difference"] == D("0.02")


def test_cumulative_view_by_settlement_week():
    platform = platform_df(
        {"transaction_id": "A", "transaction_date": "2026-10-05"},   # Monday week
        {"transaction_id": "B", "transaction_date": "2026-10-06"},
        {"transaction_id": "C", "transaction_date": "2026-10-13"},   # next week
    )
    bank = bank_df(
        {"transaction_id": "A", "settlement_date": "2026-10-06", "settlement_amount": "99.99"},
        {"transaction_id": "B", "settlement_date": "2026-10-07", "settlement_amount": "99.99"},
        {"transaction_id": "C", "settlement_date": "2026-10-14", "settlement_amount": "99.99"},
    )
    result = run(platform, bank)
    weeks = result.rounding_by_period.sort_values("settlement_week_start")
    assert list(weeks["transactions"]) == [2, 1]
    assert list(weeks["net_difference"]) == [D("0.02"), D("0.01")]


def test_no_rounding_differences_gives_empty_tables():
    result = run(platform_df({}), bank_df({}))
    assert result.rounding_details.empty and result.rounding_by_currency.empty and result.rounding_by_period.empty
