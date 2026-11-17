"""Generate realistic, reproducible sample data for PayRecon AI.

How it works: build a clean "truth" of one month of payments, derive the bank
settlement file from it (1-2 day lag), then deliberately inject known problems.
Every injected problem is recorded in an answer key (``GeneratedData.truth``),
which the tests use to measure how many issues the engine finds (precision/recall).

Run directly to (re)create the CSV files:

    python data/generate_data.py            # seed 42 (default)
    python data/generate_data.py --seed 7   # a different but still reproducible dataset
"""
from __future__ import annotations

import argparse
import calendar
import random
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):  # allow `python data/generate_data.py` from the project root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BANK_CSV, DEFAULT_SEED, PLATFORM_CSV  # noqa: E402

PLATFORM_COLUMNS = [
    "transaction_id", "customer_id", "transaction_date", "transaction_type",
    "amount", "currency", "status", "original_transaction_id",
]
BANK_COLUMNS = ["settlement_id", "transaction_id", "settlement_date", "settlement_amount", "currency", "bank_reference"]
CENT = Decimal("0.01")


@dataclass
class GeneratedData:
    platform: pd.DataFrame
    bank: pd.DataFrame
    truth: dict  # injected problems: category -> list of transaction IDs


def _money(value: float) -> Decimal:
    return Decimal(str(round(value, 2))).quantize(CENT)


def generate_datasets(seed: int = DEFAULT_SEED, year: int = 2026, month: int = 10) -> GeneratedData:
    """Build the platform and bank datasets (as string DataFrames, exactly like the CSVs)."""
    rng = random.Random(seed)
    last_day = calendar.monthrange(year, month)[1]
    month_end = date(year, month, last_day)
    customers = [f"CUST-{1000 + i}" for i in range(1, 41)]

    def rand_amount(currency: str) -> Decimal:
        if currency == "INR":
            return _money(min(max(rng.lognormvariate(6.9, 0.9), 120), 60000))
        return _money(min(max(rng.lognormvariate(4.0, 0.7), 5), 1500))

    # ---- 1. clean payments -------------------------------------------------
    payments: list[dict] = []
    for _ in range(100):  # ordinary days
        payments.append({"date": date(year, month, rng.randint(1, last_day - 4))})
    for day in [last_day, last_day - 1] + [rng.randint(last_day - 3, last_day) for _ in range(12)]:  # month-end rush
        payments.append({"date": date(year, month, day)})
    payments.sort(key=lambda p: p["date"])
    for i, p in enumerate(payments):
        currency = "USD" if rng.random() < 0.10 else "INR"
        p.update(
            id=f"TXN{100001 + i}", customer=rng.choice(customers), currency=currency,
            amount=rand_amount(currency), status="SUCCESS", lag=rng.choice([1, 2]),
        )

    safe_cutoff = last_day - 7  # payments up to here always settle inside the file
    old = [p for p in payments if p["date"].day <= safe_cutoff]
    for p in rng.sample(old, 4):
        p["status"] = "FAILED"  # no settlement expected: must NOT be flagged

    eligible = [p for p in payments if p["date"].day <= safe_cutoff and p["status"] == "SUCCESS"]
    rng.shuffle(eligible)

    def take(n: int) -> list[dict]:
        if len(eligible) < n:
            raise ValueError("Not enough eligible payments to inject all scenarios")
        chosen = eligible[:n]
        del eligible[:n]
        return chosen

    rounding = take(13)
    mismatch = take(2)
    plat_dup = take(3)
    bank_dup = take(2)
    both_dup = take(1)
    missing_bank = take(3)
    late = take(2)
    refund_originals = take(12)
    inconsistent_original = take(1)[0]

    # ---- 2. platform rows --------------------------------------------------
    platform: list[dict] = []

    def platform_row(txn_id, customer, d, kind, amount, currency, status="SUCCESS", original=""):
        return {"transaction_id": txn_id, "customer_id": customer, "transaction_date": d.isoformat(),
                "transaction_type": kind, "amount": str(amount), "currency": currency,
                "status": status, "original_transaction_id": original}

    conflicting_id = plat_dup[0]["id"]
    for p in payments:
        platform.append(platform_row(p["id"], p["customer"], p["date"], "PAYMENT", p["amount"], p["currency"], p["status"]))
        if p in plat_dup or p in both_dup:
            extra_amount = p["amount"] + Decimal("10.00") if p["id"] == conflicting_id else p["amount"]
            platform.append(platform_row(p["id"], p["customer"], p["date"], "PAYMENT", extra_amount, p["currency"]))

    # ---- 3. bank rows ------------------------------------------------------
    bank: list[dict] = []

    def bank_row(txn_id, d, amount, currency):
        return {"transaction_id": txn_id, "settlement_date": d, "settlement_amount": amount, "currency": currency}

    pending_ids: list[str] = []
    rounding_deltas = [Decimal("0.01")] * 10 + [Decimal("0.02")] * 2 + [Decimal("-0.01")]  # bank usually rounds down
    rounding_by_id = {p["id"]: delta for p, delta in zip(rounding, rounding_deltas)}
    mismatch_by_id = {p["id"]: max((p["amount"] * Decimal("0.05")).quantize(CENT), Decimal("1.00")) for p in mismatch}
    missing_ids = {p["id"] for p in missing_bank}
    late_ids = {p["id"] for p in late}

    for p in payments:
        if p["status"] != "SUCCESS" or p["id"] in missing_ids:
            continue
        lag = 4 if p["id"] in late_ids else p["lag"]
        settle = p["date"] + timedelta(days=lag)
        if settle > month_end:
            pending_ids.append(p["id"])  # settles next month: pending, not an error
            continue
        amount = p["amount"] - rounding_by_id.get(p["id"], Decimal("0")) - mismatch_by_id.get(p["id"], Decimal("0"))
        bank.append(bank_row(p["id"], settle, amount, p["currency"]))

    # ---- 4. refunds --------------------------------------------------------
    refund_seq = 200001
    valid_refund_ids: list[str] = []

    def add_refund(original_id, customer, d, amount, currency):
        nonlocal refund_seq
        rid = f"RFD{refund_seq}"
        refund_seq += 1
        platform.append(platform_row(rid, customer, d, "REFUND", amount, currency, original=original_id))
        bank.append(bank_row(rid, d + timedelta(days=rng.choice([1, 2])), -amount, currency))
        return rid

    for p in refund_originals:
        share = Decimal(str(rng.choice([0.25, 0.5, 0.75, 1.0])))
        amount = (p["amount"] * share).quantize(CENT)
        valid_refund_ids.append(add_refund(p["id"], p["customer"], p["date"] + timedelta(days=rng.randint(1, 3)), amount, p["currency"]))

    too_big = (inconsistent_original["amount"] * Decimal("1.5")).quantize(CENT)
    inconsistent_id = add_refund(inconsistent_original["id"], inconsistent_original["customer"],
                                 inconsistent_original["date"] + timedelta(days=2), too_big, inconsistent_original["currency"])

    unmatched_ids = [
        add_refund("TXN099001", rng.choice(customers), date(year, month, rng.randint(5, 24)), rand_amount("INR"), "INR"),   # earlier-period original
        add_refund("TXN099002", rng.choice(customers), date(year, month, rng.randint(5, 24)), rand_amount("INR"), "INR"),   # reference never existed
        add_refund("", rng.choice(customers), date(year, month, rng.randint(5, 24)), rand_amount("INR"), "INR"),            # no reference at all
        add_refund(valid_refund_ids[0], rng.choice(customers), date(year, month, rng.randint(5, 24)), rand_amount("INR"), "INR"),  # refund of a refund
    ]

    # ---- 5. bank-only and duplicated settlements ---------------------------
    missing_platform_ids = []
    for n in (1, 2):
        d = date(year, month, rng.randint(3, 20))
        txn_id = f"TXN88000{n}"
        missing_platform_ids.append(txn_id)
        bank.append(bank_row(txn_id, d + timedelta(days=rng.choice([1, 2])), rand_amount("INR"), "INR"))

    originals = {r["transaction_id"]: r for r in list(bank)}
    for p in bank_dup + both_dup:
        first = originals[p["id"]]
        extra_day = min(first["settlement_date"] + timedelta(days=rng.choice([0, 1])), month_end)
        bank.append(bank_row(p["id"], extra_day, first["settlement_amount"], first["currency"]))

    # ---- 6. finish: sort, number, serialise -------------------------------
    platform.sort(key=lambda r: r["transaction_date"])  # stable: duplicates stay right after their original
    bank.sort(key=lambda r: r["settlement_date"])
    for i, row in enumerate(bank, start=1):
        row["settlement_id"] = f"STL{500000 + i}"
        row["bank_reference"] = f"BNK{row['settlement_date']:%Y%m%d}{rng.randint(10000, 99999)}"
        row["settlement_date"] = row["settlement_date"].isoformat()
        row["settlement_amount"] = str(row["settlement_amount"])

    truth = {
        "rounding": sorted(p["id"] for p in rounding),
        "amount_mismatch": sorted(p["id"] for p in mismatch),
        "platform_duplicates": sorted(p["id"] for p in plat_dup + both_dup),
        "bank_duplicates": sorted(p["id"] for p in bank_dup + both_dup),
        "unmatched_refunds": sorted(unmatched_ids),
        "inconsistent_refunds": [inconsistent_id],
        "valid_refunds": sorted(valid_refund_ids),
        "missing_from_bank": sorted(missing_ids),
        "missing_from_platform": sorted(missing_platform_ids),
        "pending": sorted(pending_ids),
        "failed": sorted(p["id"] for p in payments if p["status"] == "FAILED"),
        "late_settlement": sorted(late_ids),
    }
    return GeneratedData(
        platform=pd.DataFrame(platform, columns=PLATFORM_COLUMNS),
        bank=pd.DataFrame(bank, columns=BANK_COLUMNS),
        truth=truth,
    )


def write_datasets(seed: int = DEFAULT_SEED, platform_path: Path = PLATFORM_CSV, bank_path: Path = BANK_CSV) -> GeneratedData:
    """Generate and save both CSV files."""
    data = generate_datasets(seed)
    Path(platform_path).parent.mkdir(parents=True, exist_ok=True)
    data.platform.to_csv(platform_path, index=False)
    data.bank.to_csv(bank_path, index=False)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample PayRecon AI data.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed (default: %(default)s)")
    args = parser.parse_args()
    data = write_datasets(args.seed)
    print(f"Wrote {len(data.platform)} platform rows to {PLATFORM_CSV}")
    print(f"Wrote {len(data.bank)} bank rows to {BANK_CSV}")
    print("Injected problems:", {k: len(v) for k, v in data.truth.items()})


if __name__ == "__main__":
    main()
