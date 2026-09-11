from __future__ import annotations

import pandas as pd
import pytest

from cb_quant.g4_inputs import (
    build_horizon_schedules,
    build_multi_horizon_forward_return_panel,
)


def test_horizon_schedule_exits_h_market_days_after_the_entry_open() -> None:
    trading_dates = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
            "2024-01-10",
        ]
    )
    rebalance = pd.DataFrame(
        {
            "signal_date": ["2024-01-02"],
            "entry_date": ["2024-01-03"],
            "next_entry_date": ["2024-01-10"],
        }
    )

    result = build_horizon_schedules(
        rebalance,
        trading_dates=trading_dates,
        horizons=(1, 5),
        sample_partition="test_partition",
        start="2024-01-01",
        end="2024-01-10",
    ).set_index("horizon_days")

    assert result.loc[1, "entry_date"] == pd.Timestamp("2024-01-03")
    assert result.loc[1, "exit_date"] == pd.Timestamp("2024-01-04")
    assert result.loc[5, "exit_date"] == pd.Timestamp("2024-01-10")
    assert result["is_label_within_partition"].all()


def test_horizon_schedule_marks_only_the_horizon_that_crosses_the_boundary() -> None:
    trading_dates = pd.date_range("2024-01-02", periods=8, freq="B")
    rebalance = pd.DataFrame(
        {
            "signal_date": ["2024-01-02"],
            "entry_date": ["2024-01-03"],
            "next_entry_date": ["2024-01-10"],
        }
    )

    result = build_horizon_schedules(
        rebalance,
        trading_dates=trading_dates,
        horizons=(1, 5),
        sample_partition="test_partition",
        start="2024-01-01",
        end="2024-01-08",
    ).set_index("horizon_days")

    assert bool(result.loc[1, "is_label_within_partition"])
    assert not bool(result.loc[5, "is_label_within_partition"])
    assert result.loc[5, "label_exclusion_reason"] == "partition_boundary"


def test_multi_horizon_returns_lock_the_same_bond_and_preserve_horizon() -> None:
    signal_keys = pd.DataFrame(
        {"signal_date": ["2024-01-02"], "ts_code": ["110001.SH"]}
    )
    schedule = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "horizon_days": [1, 5],
            "entry_date": pd.to_datetime(["2024-01-03", "2024-01-03"]),
            "exit_date": pd.to_datetime(["2024-01-04", "2024-01-10"]),
            "label_end_date": pd.to_datetime(["2024-01-04", "2024-01-10"]),
            "sample_partition": ["test_partition", "test_partition"],
            "is_label_within_partition": [True, True],
            "label_exclusion_reason": ["eligible", "eligible"],
        }
    )
    market = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2024-01-03", "2024-01-04", "2024-01-10"]
            ),
            "ts_code": ["110001.SH"] * 3,
            "open": [100.0, 102.0, 110.0],
            "buy_allowed": [True, True, True],
            "sell_allowed": [True, True, True],
        }
    )

    result = build_multi_horizon_forward_return_panel(
        signal_keys,
        schedule,
        market,
    ).set_index("horizon_days")

    assert result.loc[1, "forward_return"] == pytest.approx(0.02)
    assert result.loc[5, "forward_return"] == pytest.approx(0.10)
    assert result.index.tolist() == [1, 5]


def test_horizons_must_be_unique_positive_integers() -> None:
    rebalance = pd.DataFrame(
        {
            "signal_date": ["2024-01-02"],
            "entry_date": ["2024-01-03"],
            "next_entry_date": ["2024-01-10"],
        }
    )
    trading_dates = pd.date_range("2024-01-02", periods=8, freq="B")

    with pytest.raises(ValueError, match="positive integers"):
        build_horizon_schedules(
            rebalance,
            trading_dates=trading_dates,
            horizons=(0, 5),
            sample_partition="test_partition",
            start="2024-01-01",
            end="2024-01-10",
        )
    with pytest.raises(ValueError, match="unique"):
        build_horizon_schedules(
            rebalance,
            trading_dates=trading_dates,
            horizons=(1, 1),
            sample_partition="test_partition",
            start="2024-01-01",
            end="2024-01-10",
        )
