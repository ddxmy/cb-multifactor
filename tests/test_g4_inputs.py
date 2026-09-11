from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cb_quant.g4_inputs import (
    build_execution_market,
    build_forward_return_panel,
    build_partition_schedule,
    enforce_partition_access,
    prepare_lifecycle_terms,
    prepare_redemption_events,
)
from cb_quant.legacy_schema import LEGACY_REDEMPTION_COLUMNS


def test_execution_market_blocks_only_one_price_limit_and_no_trade_rows() -> None:
    raw = pd.DataFrame(
        {
            "trade_date": ["2024-01-03"] * 4,
            "ts_code": ["A", "B", "C", "D"],
            "pre_close": [100.0, 100.0, 100.0, 100.0],
            "open": [120.0, 80.0, 102.0, 101.0],
            "high": [120.0, 80.0, 105.0, 101.0],
            "low": [120.0, 80.0, 99.0, 101.0],
            "close": [120.0, 80.0, 103.0, 101.0],
            "vol": [10.0, 10.0, 10.0, 0.0],
            "amount": [120.0, 80.0, 102.0, 0.0],
        }
    )

    result = build_execution_market(
        raw,
        trading_dates=pd.to_datetime(["2024-01-03"]),
        bond_codes=["A", "B", "C", "D", "E"],
    ).set_index("ts_code")

    assert bool(result.loc["A", "is_one_price_up"])
    assert not bool(result.loc["A", "buy_allowed"])
    assert bool(result.loc["A", "sell_allowed"])
    assert bool(result.loc["B", "is_one_price_down"])
    assert bool(result.loc["B", "buy_allowed"])
    assert not bool(result.loc["B", "sell_allowed"])
    assert bool(result.loc["C", "buy_allowed"])
    assert bool(result.loc["C", "sell_allowed"])
    assert not bool(result.loc["D", "buy_allowed"])
    assert not bool(result.loc["D", "sell_allowed"])
    assert not bool(result.loc["E", "has_market_observation"])
    assert not bool(result.loc["E", "buy_allowed"])


def test_partition_schedule_marks_labels_that_cross_the_boundary() -> None:
    calendar = pd.DataFrame(
        {
            "signal_date": ["2024-01-02", "2024-01-09", "2024-01-16"],
            "entry_date": ["2024-01-03", "2024-01-10", "2024-01-17"],
            "next_entry_date": ["2024-01-10", "2024-01-17", "2024-01-24"],
        }
    )

    result = build_partition_schedule(
        calendar,
        sample_partition="test_partition",
        start="2024-01-01",
        end="2024-01-12",
    )

    assert result["signal_date"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-09"),
    ]
    assert result["is_label_within_partition"].tolist() == [True, False]
    assert result.loc[0, "exit_date"] == pd.Timestamp("2024-01-10")
    assert result.loc[1, "label_exclusion_reason"] == "partition_boundary"


def test_forward_return_panel_keeps_explicit_execution_exclusions() -> None:
    signal_keys = pd.DataFrame(
        {
            "signal_date": ["2024-01-02"] * 4 + ["2024-01-09"],
            "ts_code": ["OK", "UP", "DOWN", "MISS", "BOUNDARY"],
        }
    )
    schedule = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2024-01-02", "2024-01-09"]),
            "entry_date": pd.to_datetime(["2024-01-03", "2024-01-10"]),
            "exit_date": pd.to_datetime(["2024-01-10", "2024-01-17"]),
            "label_end_date": pd.to_datetime(["2024-01-10", "2024-01-17"]),
            "is_label_within_partition": [True, False],
            "label_exclusion_reason": ["eligible", "partition_boundary"],
        }
    )
    market = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2024-01-03"] * 4 + ["2024-01-10"] * 4
            ),
            "ts_code": ["OK", "UP", "DOWN", "MISS"] * 2,
            "open": [100.0, 120.0, 100.0, np.nan, 110.0, 125.0, 90.0, 100.0],
            "buy_allowed": [True, False, True, False, True, True, True, True],
            "sell_allowed": [True, True, True, True, True, True, False, True],
        }
    )

    result = build_forward_return_panel(signal_keys, schedule, market).set_index(
        "ts_code"
    )

    assert result.loc["OK", "forward_return"] == pytest.approx(0.10)
    assert bool(result.loc["OK", "label_usable"])
    assert result.loc["UP", "label_exclusion_reason"] == "entry_not_buyable"
    assert result.loc["DOWN", "label_exclusion_reason"] == "exit_not_sellable"
    assert result.loc["MISS", "label_exclusion_reason"] == "missing_entry_open"
    assert result.loc["BOUNDARY", "label_exclusion_reason"] == "partition_boundary"
    assert result.loc[["UP", "DOWN", "MISS", "BOUNDARY"], "forward_return"].isna().all()


def test_redemption_mapping_does_not_infer_an_unobserved_last_trade_date() -> None:
    watch_status = "\u5df2\u6ee1\u8db3\u5f3a\u8d4e\u6761\u4ef6"
    announced_status = "\u516c\u544a\u5b9e\u65bd\u5f3a\u8d4e"
    columns = LEGACY_REDEMPTION_COLUMNS
    raw = pd.DataFrame(
        {
            columns["ts_code"]: ["110001.SH", "110001.SH"],
            columns["event_status"]: [watch_status, announced_status],
            columns["effective_date"]: ["2024-01-02", "2024-01-10"],
            columns["source_redeem_date"]: [None, "2024-01-26"],
            columns["call_price"]: [np.nan, 101.5],
            columns["tax_adjusted_call_price"]: [np.nan, 101.2],
            columns["payment_date"]: [None, "2024-01-29"],
            columns["call_reg_date"]: [None, "2024-01-25"],
        }
    )

    result = prepare_redemption_events(raw)

    assert result["event_status"].tolist() == [watch_status, announced_status]
    assert result["effective_date"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-10"),
    ]
    assert result["last_trade_date"].isna().all()
    assert result.loc[1, "call_reg_date"] == pd.Timestamp("2024-01-25")
    assert result.loc[1, "payment_date"] == pd.Timestamp("2024-01-29")
    assert result.loc[1, "call_price"] == pytest.approx(101.5)


def test_lifecycle_terms_do_not_infer_an_unobserved_maturity_payment_date() -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["110001.SH"],
            "list_date": [20200102],
            "delist_date": [20260102],
            "maturity_date": [20260101],
            "maturity_call_price": [108.0],
        }
    )

    result = prepare_lifecycle_terms(raw)

    assert result.loc[0, "list_date"] == pd.Timestamp("2020-01-02")
    assert result.loc[0, "maturity_date"] == pd.Timestamp("2026-01-01")
    assert pd.isna(result.loc[0, "maturity_payment_date"])


def test_holdout_materialization_requires_an_explicit_unlock() -> None:
    enforce_partition_access("replication_2018_2023", unlock_holdout=False)
    with pytest.raises(PermissionError, match="holdout"):
        enforce_partition_access("holdout_2025_onward", unlock_holdout=False)
    enforce_partition_access("holdout_2025_onward", unlock_holdout=True)
