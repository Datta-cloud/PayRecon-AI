"""Read CSV files into raw, all-string DataFrames.

The loader deliberately does *no* interpretation of the data. Everything is read
as text so that the validator (not pandas guessing) decides what is a valid
amount or date. That keeps financial values exact.
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Union

import pandas as pd


class DataLoadError(Exception):
    """Raised when a file cannot be read as a CSV at all."""


def normalize_column_name(name: object) -> str:
    """'Transaction ID ' -> 'transaction_id'."""
    text = str(name).replace("\ufeff", "").strip().lower()
    return re.sub(r"[\s\-]+", "_", text)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_column_name(c) for c in out.columns]
    return out


def load_csv(source: Union[str, Path, bytes, io.IOBase, object], label: str = "CSV file") -> pd.DataFrame:
    """Load a CSV from a path, bytes or a file-like object (e.g. a Streamlit upload).

    Returns a DataFrame of strings with normalised column names and trimmed cells.
    A file with a header but no rows returns an empty DataFrame (the validator
    reports it). Unreadable or completely empty files raise :class:`DataLoadError`.
    """
    read_kwargs = dict(dtype=str, keep_default_na=False, skipinitialspace=True, encoding="utf-8-sig")
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise DataLoadError(f"{label}: file not found at '{path}'.")
            df = pd.read_csv(path, **read_kwargs)
        elif isinstance(source, bytes):
            df = pd.read_csv(io.BytesIO(source), **read_kwargs)
        else:
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(source, **read_kwargs)
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError(f"{label} is empty. Upload a CSV that has a header row and data.") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadError(f"{label} is not a valid CSV file: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise DataLoadError(f"{label} could not be decoded as UTF-8 text. Save it as a UTF-8 CSV.") from exc

    df = normalize_columns(df)
    if not df.empty:
        df = df.apply(lambda col: col.str.strip())
    return df
