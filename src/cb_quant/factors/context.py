"""Validated point-in-time inputs supplied to factor plugins."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


_TABLE_KEYS = {
    "signal": ("signal_date", "ts_code"),
    "valuation": ("signal_date", "ts_code"),
    "cb_daily": ("trade_date", "ts_code"),
    "stock_daily": ("trade_date", "stk_code"),
    "lifecycle": ("trade_date", "ts_code"),
}
_DATE_COLUMNS = ("signal_date", "trade_date", "available_date", "effective_date")
_CODE_COLUMNS = ("ts_code", "stk_code")


def _prepare_table(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    result = frame.copy()
    if result.empty and not len(result.columns):
        return result

    keys = _TABLE_KEYS[name]
    if missing := set(keys).difference(result.columns):
        raise KeyError(f"{name} is missing key columns: {sorted(missing)}")
    for column in _DATE_COLUMNS:
        if column in result:
            result[column] = pd.to_datetime(
                result[column], errors="raise"
            ).dt.normalize()
    for column in _CODE_COLUMNS:
        if column in result:
            result[column] = result[column].astype("string")
    if result[list(keys)].isna().any().any():
        raise ValueError(f"{name} contains missing key values")
    if result.duplicated(list(keys)).any():
        raise ValueError(f"{name} contains duplicate keys")
    if name == "signal" and "available_date" in result:
        future = result["available_date"].gt(result["signal_date"])
        if future.any():
            raise ValueError("signal contains future available dates")
    return result.sort_values(list(keys)).reset_index(drop=True)


@dataclass(frozen=True)
class FactorDataBundle:
    """Point-in-time signal keys and shared histories for factor formulas."""

    signal: pd.DataFrame
    valuation: pd.DataFrame = field(default_factory=pd.DataFrame)
    cb_daily: pd.DataFrame = field(default_factory=pd.DataFrame)
    stock_daily: pd.DataFrame = field(default_factory=pd.DataFrame)
    lifecycle: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        for name in _TABLE_KEYS:
            object.__setattr__(self, name, _prepare_table(name, getattr(self, name)))


def ensure_factor_data_bundle(
    data: FactorDataBundle | pd.DataFrame,
) -> FactorDataBundle:
    """Return a validated bundle, adapting legacy signal-only DataFrames."""
    if isinstance(data, FactorDataBundle):
        return data
    if isinstance(data, pd.DataFrame):
        return FactorDataBundle(signal=data)
    raise TypeError("factor data must be a FactorDataBundle or DataFrame")
