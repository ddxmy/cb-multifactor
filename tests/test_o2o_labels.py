from __future__ import annotations

import pandas as pd
import pytest

from cb_quant.labels import attach_o2o_returns, build_execution_schedule


def test_schedule_maps_signal_to_strict_next_open_and_next_execution_open() -> None:
    trading_dates = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    )
    signal_dates = pd.to_datetime(["2024-01-02", "2024-01-04", "2024-01-05"])

    result = build_execution_schedule(trading_dates, signal_dates)

    assert result.to_dict("records") == [
        {
            "signal_date": pd.Timestamp("2024-01-02"),
            "entry_date": pd.Timestamp("2024-01-03"),
            "exit_date": pd.Timestamp("2024-01-05"),
            "label_end_date": pd.Timestamp("2024-01-05"),
        },
        {
            "signal_date": pd.Timestamp("2024-01-04"),
            "entry_date": pd.Timestamp("2024-01-05"),
            "exit_date": pd.Timestamp("2024-01-08"),
            "label_end_date": pd.Timestamp("2024-01-08"),
        },
    ]


def test_attach_o2o_returns_locks_bond_and_uses_entry_and_exit_opens() -> None:
    schedule = pd.DataFrame(
        {
            "signal_date": [pd.Timestamp("2024-01-02")],
            "entry_date": [pd.Timestamp("2024-01-03")],
            "exit_date": [pd.Timestamp("2024-01-05")],
            "label_end_date": [pd.Timestamp("2024-01-05")],
        }
    )
    factors = pd.DataFrame(
        {
            "signal_date": ["2024-01-02", "2024-01-02"],
            "ts_code": ["110001.SH", "110002.SH"],
            "raw_factor": [1.0, 2.0],
        }
    )
    prices = pd.DataFrame(
        {
            "trade_date": [
                "2024-01-03",
                "2024-01-05",
                "2024-01-03",
                "2024-01-05",
            ],
            "ts_code": ["110001.SH", "110001.SH", "110002.SH", "110002.SH"],
            "open": [100.0, 110.0, 120.0, 114.0],
        }
    )

    result = attach_o2o_returns(factors, schedule, prices)
    returns = result.set_index("ts_code")["forward_return"]

    assert returns["110001.SH"] == pytest.approx(0.10)
    assert returns["110002.SH"] == pytest.approx(-0.05)
    assert result["label_end_date"].eq(pd.Timestamp("2024-01-05")).all()


def test_attach_o2o_returns_can_drop_but_never_fill_missing_execution_price() -> None:
    schedule = pd.DataFrame(
        {
            "signal_date": ["2024-01-02"],
            "entry_date": ["2024-01-03"],
            "exit_date": ["2024-01-05"],
        }
    )
    factors = pd.DataFrame(
        {
            "signal_date": ["2024-01-02", "2024-01-02"],
            "ts_code": ["110001.SH", "110002.SH"],
            "raw_factor": [1.0, 2.0],
        }
    )
    prices = pd.DataFrame(
        {
            "trade_date": ["2024-01-03", "2024-01-05", "2024-01-03"],
            "ts_code": ["110001.SH", "110001.SH", "110002.SH"],
            "open": [100.0, 110.0, 120.0],
        }
    )

    with pytest.raises(ValueError, match="unusable execution prices"):
        attach_o2o_returns(factors, schedule, prices)
    result = attach_o2o_returns(factors, schedule, prices, drop_unusable=True)
    assert result["ts_code"].tolist() == ["110001.SH"]
    assert result.attrs["dropped_unusable_execution_rows"] == 1
    diagnostics = result.attrs["unusable_execution_rows"]
    assert diagnostics.loc[0, "ts_code"] == "110002.SH"
    assert diagnostics.loc[0, "label_exclusion_reason"] == "missing_exit_open"


def test_rejects_duplicate_prices_and_non_market_signal_dates() -> None:
    trading_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    with pytest.raises(ValueError, match="outside the trading calendar"):
        build_execution_schedule(trading_dates, ["2024-01-01"])

    schedule = pd.DataFrame(
        {
            "signal_date": ["2024-01-02"],
            "entry_date": ["2024-01-03"],
            "exit_date": ["2024-01-04"],
        }
    )
    factors = pd.DataFrame(
        {"signal_date": ["2024-01-02"], "ts_code": ["110001.SH"], "raw_factor": [1.0]}
    )
    duplicate_prices = pd.DataFrame(
        {
            "trade_date": ["2024-01-03", "2024-01-03", "2024-01-04"],
            "ts_code": ["110001.SH", "110001.SH", "110001.SH"],
            "open": [100.0, 101.0, 110.0],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        attach_o2o_returns(factors, schedule, duplicate_prices)
