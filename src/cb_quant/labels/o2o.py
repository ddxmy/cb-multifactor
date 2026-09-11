"""Leakage-free next-open to next-execution-open factor labels."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _normalized_index(values: Iterable) -> pd.DatetimeIndex:
    return (
        pd.DatetimeIndex(pd.to_datetime(list(values), errors="raise"))
        .normalize()
        .drop_duplicates()
        .sort_values()
    )


def build_execution_schedule(
    trading_dates: Iterable,
    signal_dates: Iterable,
) -> pd.DataFrame:
    """Map each close signal to the strict next open and next execution open."""
    calendar = _normalized_index(trading_dates)
    signals = _normalized_index(signal_dates)
    if calendar.empty:
        raise ValueError("trading calendar cannot be empty")
    if not signals.isin(calendar).all():
        raise ValueError("signal dates contain values outside the trading calendar")

    positions = calendar.searchsorted(signals.to_numpy(), side="right")
    entry_dates = pd.Series(pd.NaT, index=range(len(signals)), dtype="datetime64[ns]")
    valid = positions < len(calendar)
    entry_dates.loc[valid] = calendar.take(positions[valid]).to_numpy()
    schedule = pd.DataFrame({"signal_date": signals, "entry_date": entry_dates})
    schedule["exit_date"] = schedule["entry_date"].shift(-1)
    schedule = schedule.dropna(subset=["entry_date", "exit_date"]).reset_index(drop=True)
    schedule["label_end_date"] = schedule["exit_date"]
    return schedule


def _prepare_open_prices(open_prices: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "ts_code", "open"}
    if missing := required.difference(open_prices.columns):
        raise KeyError(f"open prices are missing columns: {sorted(missing)}")
    prices = open_prices[["trade_date", "ts_code", "open"]].copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"], errors="raise").dt.normalize()
    prices["ts_code"] = prices["ts_code"].astype("string")
    prices["open"] = pd.to_numeric(prices["open"], errors="coerce")
    if prices.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("open prices contain duplicate date and bond keys")
    return prices


def attach_o2o_returns(
    factor_frame: pd.DataFrame,
    schedule: pd.DataFrame,
    open_prices: pd.DataFrame,
    *,
    drop_unusable: bool = False,
) -> pd.DataFrame:
    """Attach locked-bond O2O returns without filling missing execution prices."""
    required_factor = {"signal_date", "ts_code", "raw_factor"}
    if missing := required_factor.difference(factor_frame.columns):
        raise KeyError(f"factor frame is missing columns: {sorted(missing)}")
    required_schedule = {"signal_date", "entry_date", "exit_date"}
    if missing := required_schedule.difference(schedule.columns):
        raise KeyError(f"execution schedule is missing columns: {sorted(missing)}")

    factors = factor_frame.copy()
    factors["signal_date"] = pd.to_datetime(
        factors["signal_date"], errors="raise"
    ).dt.normalize()
    factors["ts_code"] = factors["ts_code"].astype("string")
    if factors.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError("factor frame contains duplicate signal-date and bond keys")

    dates = schedule.copy()
    for column in ("signal_date", "entry_date", "exit_date"):
        dates[column] = pd.to_datetime(dates[column], errors="raise").dt.normalize()
    if "label_end_date" not in dates:
        dates["label_end_date"] = dates["exit_date"]
    else:
        dates["label_end_date"] = pd.to_datetime(
            dates["label_end_date"], errors="raise"
        ).dt.normalize()
    if dates.duplicated("signal_date").any():
        raise ValueError("execution schedule contains duplicate signal dates")

    prices = _prepare_open_prices(open_prices)
    result = factors.merge(dates, on="signal_date", how="inner", validate="many_to_one")
    entry = prices.rename(columns={"trade_date": "entry_date", "open": "entry_open"})
    result = result.merge(
        entry,
        on=["entry_date", "ts_code"],
        how="left",
        validate="many_to_one",
    )
    exit_prices = prices.rename(columns={"trade_date": "exit_date", "open": "exit_open"})
    result = result.merge(
        exit_prices,
        on=["exit_date", "ts_code"],
        how="left",
        validate="many_to_one",
    )
    missing_entry = result["entry_open"].isna()
    invalid_entry = ~missing_entry & (
        result["entry_open"].le(0) | ~np.isfinite(result["entry_open"])
    )
    missing_exit = result["exit_open"].isna()
    invalid_exit = ~missing_exit & (
        result["exit_open"].le(0) | ~np.isfinite(result["exit_open"])
    )
    unusable = missing_entry | invalid_entry | missing_exit | invalid_exit
    result["label_exclusion_reason"] = np.select(
        [missing_entry, invalid_entry, missing_exit, invalid_exit],
        ["missing_entry_open", "invalid_entry_open", "missing_exit_open", "invalid_exit_open"],
        default="eligible",
    )
    dropped = int(unusable.sum())
    diagnostics = result.loc[
        unusable,
        [
            "signal_date",
            "ts_code",
            "entry_date",
            "exit_date",
            "entry_open",
            "exit_open",
            "label_exclusion_reason",
        ],
    ].reset_index(drop=True)
    if dropped and not drop_unusable:
        raise ValueError(f"factor labels contain {dropped} unusable execution prices")
    if drop_unusable:
        result = result.loc[~unusable].copy()
    result["forward_return"] = result["exit_open"] / result["entry_open"] - 1.0
    result = result.sort_values(["signal_date", "ts_code"]).reset_index(drop=True)
    result.attrs["dropped_unusable_execution_rows"] = dropped
    result.attrs["unusable_execution_rows"] = diagnostics
    return result
