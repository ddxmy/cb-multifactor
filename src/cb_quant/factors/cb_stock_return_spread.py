"""Simple return spread between a convertible bond and its underlying stock."""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil

import numpy as np
import pandas as pd

from .base import FactorSpec
from .context import FactorDataBundle
from .output import build_factor_output


VALID_WINDOWS = (5, 10, 20)
MINIMUM_OBSERVATION_RATIO = 0.8

FACTOR_SPEC = FactorSpec(
    name="cb_stock_return_spread",
    family="stock_bond_linkage",
    description=(
        "Simple convertible-bond close return minus adjusted underlying-stock "
        "return over aligned market-day endpoints."
    ),
    direction=-1,
    required_fields=("stk_code",),
    required_history_fields={
        "cb_daily": ("trade_date", "ts_code", "close"),
        "stock_daily": (
            "trade_date",
            "stk_code",
            "close",
            "adj_factor",
        ),
    },
    default_parameters={"window": 20},
    neutralizers=("conversion_value", "rating", "remaining_maturity"),
)


def _validate_window(parameters: Mapping[str, object]) -> int:
    unknown = set(parameters).difference({"window"})
    if unknown:
        raise ValueError(f"unknown parameters: {sorted(unknown)}")
    window = int(parameters["window"])
    if window not in VALID_WINDOWS:
        raise ValueError("window must be 5, 10, or 20 market days")
    return window


def _return_if_complete(
    prices: pd.Series,
    market_dates: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    minimum_observations: int,
) -> float:
    window_prices = prices.reindex(
        market_dates[market_dates.get_loc(start_date) : market_dates.get_loc(end_date) + 1]
    )
    if window_prices.notna().sum() < minimum_observations:
        return np.nan
    start_price = window_prices.iloc[0]
    end_price = window_prices.iloc[-1]
    if pd.isna(start_price) or pd.isna(end_price):
        return np.nan
    if start_price <= 0.0 or end_price <= 0.0:
        return np.nan
    return float(end_price / start_price - 1.0)


def _build_return_components(
    data: FactorDataBundle,
    window: int,
) -> pd.DataFrame:
    signal = data.signal
    cb_daily = data.cb_daily.copy()
    stock_daily = data.stock_daily.copy()
    cb_daily["close"] = pd.to_numeric(cb_daily["close"], errors="coerce")
    stock_daily["close"] = pd.to_numeric(stock_daily["close"], errors="coerce")
    stock_daily["adj_factor"] = pd.to_numeric(
        stock_daily["adj_factor"], errors="coerce"
    )
    stock_daily["adjusted_close"] = (
        stock_daily["close"] * stock_daily["adj_factor"]
    )

    market_dates = pd.DatetimeIndex(
        sorted(
            set(cb_daily["trade_date"].dropna()).union(
                stock_daily["trade_date"].dropna()
            )
        )
    )
    date_positions = {date: position for position, date in enumerate(market_dates)}
    minimum_observations = ceil((window + 1) * MINIMUM_OBSERVATION_RATIO)
    cb_prices = {
        code: group.set_index("trade_date")["close"].sort_index()
        for code, group in cb_daily.groupby("ts_code", sort=False)
    }
    stock_prices = {
        code: group.set_index("trade_date")["adjusted_close"].sort_index()
        for code, group in stock_daily.groupby("stk_code", sort=False)
    }

    rows: list[dict[str, float]] = []
    for row in signal.itertuples(index=False):
        end_date = row.signal_date
        end_position = date_positions.get(end_date)
        if end_position is None or end_position < window:
            rows.append({"cb_return": np.nan, "stock_return": np.nan})
            continue
        start_date = market_dates[end_position - window]
        cb_series = cb_prices.get(row.ts_code)
        stock_series = stock_prices.get(row.stk_code)
        if cb_series is None or stock_series is None:
            rows.append({"cb_return": np.nan, "stock_return": np.nan})
            continue
        rows.append(
            {
                "cb_return": _return_if_complete(
                    cb_series,
                    market_dates,
                    start_date,
                    end_date,
                    minimum_observations,
                ),
                "stock_return": _return_if_complete(
                    stock_series,
                    market_dates,
                    start_date,
                    end_date,
                    minimum_observations,
                ),
            }
        )
    return pd.DataFrame(rows, index=signal.index)


def calculate_raw_factor(
    data: FactorDataBundle,
    parameters: Mapping[str, object],
) -> pd.Series:
    """Calculate the researcher-approved raw factor formula."""
    window = _validate_window(parameters)
    components = _build_return_components(data, window)
    return components["cb_return"] - components["stock_return"]


def calculate_factor(
    data: FactorDataBundle,
    parameters: Mapping[str, object],
) -> pd.DataFrame:
    """Apply the standard output contract to the raw spread formula."""
    raw_factor = calculate_raw_factor(data, parameters)
    return build_factor_output(data.signal, raw_factor)
