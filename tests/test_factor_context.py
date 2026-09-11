from __future__ import annotations

import pandas as pd
import pytest

from cb_quant.factors import FactorDataBundle, ensure_factor_data_bundle


def test_bundle_normalizes_signal_keys_and_preserves_histories() -> None:
    signal = pd.DataFrame(
        {
            "signal_date": ["2024-01-05"],
            "ts_code": ["110001.SH"],
            "stk_code": ["600001.SH"],
        }
    )
    stock_daily = pd.DataFrame(
        {
            "trade_date": ["2024-01-04"],
            "stk_code": ["600001.SH"],
            "close": [10.0],
        }
    )

    bundle = FactorDataBundle(signal=signal, stock_daily=stock_daily)

    assert bundle.signal.loc[0, "signal_date"] == pd.Timestamp("2024-01-05")
    assert bundle.signal["ts_code"].dtype.name == "string"
    assert bundle.stock_daily.loc[0, "trade_date"] == pd.Timestamp("2024-01-04")
    assert bundle.stock_daily["stk_code"].dtype.name == "string"
    assert signal.loc[0, "signal_date"] == "2024-01-05"
    assert stock_daily.loc[0, "trade_date"] == "2024-01-04"


def test_bundle_rejects_duplicate_keys_in_each_observed_table() -> None:
    signal = pd.DataFrame(
        {
            "signal_date": ["2024-01-05"],
            "ts_code": ["110001.SH"],
        }
    )
    duplicate_cb_daily = pd.DataFrame(
        {
            "trade_date": ["2024-01-04", "2024-01-04"],
            "ts_code": ["110001.SH", "110001.SH"],
            "close": [100.0, 101.0],
        }
    )

    with pytest.raises(ValueError, match="cb_daily.*duplicate"):
        FactorDataBundle(signal=signal, cb_daily=duplicate_cb_daily)


def test_bundle_rejects_missing_required_table_keys() -> None:
    signal = pd.DataFrame(
        {
            "signal_date": ["2024-01-05"],
            "ts_code": ["110001.SH"],
        }
    )

    with pytest.raises(KeyError, match="stock_daily.*stk_code"):
        FactorDataBundle(
            signal=signal,
            stock_daily=pd.DataFrame(
                {"trade_date": ["2024-01-04"], "close": [10.0]}
            ),
        )


def test_dataframe_adapter_populates_only_the_signal_table() -> None:
    signal = pd.DataFrame(
        {
            "signal_date": ["2024-01-05"],
            "ts_code": ["110001.SH"],
        }
    )

    bundle = ensure_factor_data_bundle(signal)

    assert isinstance(bundle, FactorDataBundle)
    assert len(bundle.signal) == 1
    assert bundle.valuation.empty
    assert bundle.cb_daily.empty
    assert bundle.stock_daily.empty
    assert bundle.lifecycle.empty
    assert ensure_factor_data_bundle(bundle) is bundle
    with pytest.raises(TypeError, match="FactorDataBundle or DataFrame"):
        ensure_factor_data_bundle([signal])
