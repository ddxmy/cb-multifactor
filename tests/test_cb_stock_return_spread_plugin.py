from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cb_quant.factors import FactorDataBundle
from cb_quant.factors.cb_stock_return_spread import (
    FACTOR_SPEC,
    calculate_factor,
)


def make_bundle() -> FactorDataBundle:
    dates = pd.bdate_range("2024-01-02", periods=22)
    signal_date = dates[20]
    signal = pd.DataFrame(
        {
            "signal_date": [signal_date, signal_date],
            "ts_code": ["CB1", "CB2"],
            "stk_code": ["STK1", "STK2"],
        }
    )

    cb1 = pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": "CB1",
            "close": [*np.linspace(100.0, 110.0, 21), 150.0],
        }
    )
    cb2 = pd.DataFrame(
        {
            "trade_date": dates[5:],
            "ts_code": "CB2",
            "close": 100.0,
        }
    )
    stock1 = pd.DataFrame(
        {
            "trade_date": dates,
            "stk_code": "STK1",
            "close": [*np.linspace(50.0, 52.5, 21), 80.0],
            "adj_factor": 1.0,
        }
    )
    stock2 = pd.DataFrame(
        {
            "trade_date": dates,
            "stk_code": "STK2",
            "close": 20.0,
            "adj_factor": 1.0,
        }
    )
    return FactorDataBundle(
        signal=signal,
        cb_daily=pd.concat([cb1, cb2], ignore_index=True),
        stock_daily=pd.concat([stock1, stock2], ignore_index=True),
    )


def test_20d_spread_matches_researcher_hand_calculation() -> None:
    result = calculate_factor(make_bundle(), {"window": 20}).set_index("ts_code")

    assert result.loc["CB1", "raw_factor"] == pytest.approx(0.05)
    assert result.loc["CB1", "available_date"] == pd.Timestamp("2024-01-30")
    assert pd.isna(result.loc["CB2", "raw_factor"])
    assert pd.isna(result.loc["CB2", "available_date"])


def test_future_prices_do_not_change_signal_date_factor() -> None:
    baseline = make_bundle()
    changed = make_bundle()
    future_date = pd.Timestamp("2024-01-31")
    changed.cb_daily.loc[
        changed.cb_daily["trade_date"].eq(future_date), "close"
    ] = 1_000.0
    changed.stock_daily.loc[
        changed.stock_daily["trade_date"].eq(future_date), "close"
    ] = 1.0

    expected = calculate_factor(baseline, {"window": 20})
    actual = calculate_factor(changed, {"window": 20})

    pd.testing.assert_frame_equal(actual, expected)


def test_factor_metadata_and_window_validation_are_explicit() -> None:
    assert FACTOR_SPEC.name == "cb_stock_return_spread"
    assert FACTOR_SPEC.direction == -1
    assert FACTOR_SPEC.default_parameters == {"window": 20}

    with pytest.raises(ValueError, match="5, 10, or 20"):
        calculate_factor(make_bundle(), {"window": 7})
