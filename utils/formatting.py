"""Formatting helpers: Decimal-safe display and text rendering."""
from __future__ import annotations

from decimal import Decimal
from typing import Optional

import pandas as pd


def format_amount(value: Optional[Decimal], currency: str = "") -> str:
    """Decimal -> '1,234.50' (optionally prefixed by the currency). None -> '-'."""
    if value is None or (not isinstance(value, Decimal) and pd.isna(value)):
        return "-"
    text = f"{Decimal(value):,.2f}"
    return f"{currency} {text}".strip()


def format_amounts_by_currency(amounts: dict) -> str:
    """{'INR': Decimal('12.5'), 'USD': Decimal('0')} -> 'INR 12.50 | USD 0.00'."""
    if not amounts:
        return "-"
    return " | ".join(format_amount(v, c) for c, v in sorted(amounts.items()))


def decimals_to_float(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Decimal columns to floats for on-screen tables and charts only.

    Calculations always stay in Decimal; floats are used purely for display.
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            values = out[col].dropna()
            if len(values) and all(isinstance(v, Decimal) for v in values):
                out[col] = out[col].map(lambda v: None if v is None or pd.isna(v) else float(v)).astype("float64")
    return out


def to_display(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a result table for st.dataframe (floats for money, ISO strings for dates)."""
    out = decimals_to_float(df)
    for col in out.columns:
        if out[col].dtype == object:
            values = out[col].dropna()
            if len(values) and all(hasattr(v, "isoformat") for v in values):
                out[col] = out[col].map(lambda v: "" if v is None or pd.isna(v) else v.isoformat())
    return out


def titleize(column: str) -> str:
    return column.replace("_", " ").strip().capitalize()
