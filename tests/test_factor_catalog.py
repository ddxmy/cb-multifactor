from __future__ import annotations

import pandas as pd
import pytest

from cb_quant.factors import build_default_factor_registry


def test_default_catalog_registers_each_production_factor_explicitly() -> None:
    registry = build_default_factor_registry()

    assert registry.names() == (
        "cb_abnormal_turnover_20d",
        "cb_return_20d",
        "cb_stock_return_spread",
        "conversion_premium",
        "double_low",
        "relative_premium_to_parity_median",
        "stock_return_20d",
    )
    spread_spec = registry.get_spec("cb_stock_return_spread")
    assert spread_spec.default_parameters == {"window": 20}
    assert spread_spec.direction == -1
    assert registry.get_spec("double_low").direction == -1


def test_catalog_factor_runs_through_the_standard_contract() -> None:
    registry = build_default_factor_registry()
    context = pd.DataFrame(
        {
            "signal_date": ["2024-01-05"],
            "ts_code": ["110001.SH"],
            "conversion_premium": [0.25],
        }
    )

    result = registry.calculate("conversion_premium", context)

    assert result.loc[0, "raw_factor"] == 0.25
    assert result.loc[0, "available_date"] == pd.Timestamp("2024-01-05")


def test_registered_double_low_runs_through_the_standard_contract() -> None:
    registry = build_default_factor_registry()
    context = pd.DataFrame(
        {
            "signal_date": ["2024-01-05"],
            "ts_code": ["110001.SH"],
            "close": [108.63],
            "conversion_premium": [0.248383],
        }
    )

    result = registry.calculate("double_low", context)

    assert result.loc[0, "raw_factor"] == pytest.approx(133.4683)
    assert result.loc[0, "available_date"] == pd.Timestamp("2024-01-05")
