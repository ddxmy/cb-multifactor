"""Helpers for constructing the standard factor output table."""

from __future__ import annotations

import pandas as pd

from .base import STANDARD_FACTOR_COLUMNS


def _aligned_series(
    value: pd.Series,
    signal: pd.DataFrame,
    name: str,
) -> pd.Series:
    if not isinstance(value, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if not value.index.equals(signal.index):
        raise ValueError(f"{name} index must match the signal index")
    return value


def build_factor_output(
    signal: pd.DataFrame,
    raw_factor: pd.Series,
    *,
    available_date: pd.Series | None = None,
) -> pd.DataFrame:
    """Build aligned factor output while preserving point-in-time keys."""
    if not isinstance(signal, pd.DataFrame):
        raise TypeError("signal must be a pandas DataFrame")
    missing = {"signal_date", "ts_code"}.difference(signal.columns)
    if missing:
        raise KeyError(f"signal is missing columns: {sorted(missing)}")
    raw_factor = _aligned_series(raw_factor, signal, "raw_factor")
    if not pd.api.types.is_numeric_dtype(raw_factor):
        raise TypeError("raw_factor must have a numeric dtype")

    if available_date is None:
        availability = signal["signal_date"].where(raw_factor.notna())
    else:
        availability = _aligned_series(
            available_date,
            signal,
            "available_date",
        )

    result = signal[["signal_date", "ts_code"]].copy()
    result["raw_factor"] = raw_factor
    result["available_date"] = availability
    result["signal_date"] = pd.to_datetime(
        result["signal_date"], errors="raise"
    ).dt.normalize()
    result["available_date"] = pd.to_datetime(
        result["available_date"], errors="raise"
    ).dt.normalize()
    result["ts_code"] = result["ts_code"].astype("string")
    return result.loc[:, STANDARD_FACTOR_COLUMNS].reset_index(drop=True)
