from __future__ import annotations

import pytest
import pandas as pd

from cb_quant.factors import FactorDataBundle, build_default_factor_registry
from scripts.run_factor_batch import execute_factor_batch, validate_batch_catalog
from scripts.run_single_factor import LoadedResearchInputs, slice_research_inputs


def make_batch() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "batch_id": "test_batch",
        "sample_partition": "replication_2018_2023",
        "factors": [
            {"name": "conversion_premium", "parameters": {}},
            {"name": "cb_stock_return_spread", "parameters": {"window": 20}},
        ],
    }


def test_batch_executes_declared_factor_variants_in_order() -> None:
    calls: list[tuple[str, dict[str, object], str]] = []

    def run_one(name, parameters, sample_partition):
        calls.append((name, dict(parameters), sample_partition))
        return {"factor": name, "created": True}

    results = execute_factor_batch(make_batch(), run_one=run_one)

    assert calls == [
        ("conversion_premium", {}, "replication_2018_2023"),
        ("cb_stock_return_spread", {"window": 20}, "replication_2018_2023"),
    ]
    assert [item["factor"] for item in results] == [
        "conversion_premium",
        "cb_stock_return_spread",
    ]


def test_batch_rejects_unknown_factor_before_running_any_entry() -> None:
    batch = make_batch()
    batch["factors"][1]["name"] = "unknown_factor"

    with pytest.raises(ValueError, match="unknown_factor"):
        validate_batch_catalog(batch, build_default_factor_registry())


def test_sample_partition_is_an_executable_date_boundary() -> None:
    dates = pd.to_datetime(["2023-06-20", "2023-06-21", "2023-06-22", "2024-12-31"])
    signal = pd.DataFrame(
        {
            "signal_date": dates,
            "ts_code": "CB1",
            "conversion_premium": 0.1,
        }
    )
    market = pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": "CB1",
            "open": 100.0,
            "close": 100.0,
            "buy_allowed": True,
            "sell_allowed": True,
        }
    )
    lifecycle = pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": "CB1",
            "lifecycle_state": "active",
        }
    )
    history_dates = pd.to_datetime(["2023-06-01", *dates])
    inputs = LoadedResearchInputs(
        factor_data=FactorDataBundle(
            signal=signal,
            cb_daily=pd.DataFrame(
                {"trade_date": history_dates, "ts_code": "CB1", "close": 100.0}
            ),
            lifecycle=lifecycle,
        ),
        market=market,
        lifecycle=lifecycle,
        trading_dates=pd.DatetimeIndex(dates),
        universe_fingerprint="full-universe",
        input_fingerprints={"market": "market-hash"},
    )

    replication = slice_research_inputs(inputs, "replication_2018_2023")
    validation = slice_research_inputs(inputs, "validation_2023_2024")

    assert replication.factor_data.signal["signal_date"].max() == pd.Timestamp("2023-06-21")
    assert replication.market["trade_date"].max() == pd.Timestamp("2023-06-21")
    assert replication.factor_data.cb_daily["trade_date"].min() == pd.Timestamp("2023-06-01")
    assert validation.factor_data.signal["signal_date"].min() == pd.Timestamp("2023-06-22")
    assert validation.market["trade_date"].min() == pd.Timestamp("2023-06-22")
