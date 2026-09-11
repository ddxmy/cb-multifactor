from __future__ import annotations

import pandas as pd
import pytest

from cb_quant.factors import FactorDataBundle
from cb_quant.factors.cb_abnormal_turnover_20d import (
    FACTOR_SPEC as CB_ABNORMAL_TURNOVER_SPEC,
    calculate_factor as calculate_cb_abnormal_turnover,
)
from cb_quant.factors.cb_return_20d import (
    FACTOR_SPEC as CB_RETURN_SPEC,
    calculate_factor as calculate_cb_return,
)
from cb_quant.factors.relative_premium_to_parity_median import (
    FACTOR_SPEC as RELATIVE_PREMIUM_SPEC,
    calculate_factor as calculate_relative_premium,
)
from cb_quant.factors.stock_return_20d import (
    FACTOR_SPEC as STOCK_RETURN_SPEC,
    calculate_factor as calculate_stock_return,
)


def make_bundle() -> FactorDataBundle:
    return FactorDataBundle(
        signal=pd.DataFrame(
            {
                "signal_date": ["2024-01-05", "2024-01-05"],
                "ts_code": ["110001.SH", "110002.SH"],
                "relative_premium_to_parity_median": [-0.05, float("nan")],
                "stock_return_20d": [0.12, float("nan")],
                "cb_return_20d": [0.08, float("nan")],
                "cb_abnormal_turnover_20d": [1.40, float("nan")],
            }
        )
    )


@pytest.mark.parametrize(
    ("calculator", "factor_name", "expected_value"),
    [
        (calculate_relative_premium, "relative_premium_to_parity_median", -0.05),
        (calculate_stock_return, "stock_return_20d", 0.12),
        (calculate_cb_return, "cb_return_20d", 0.08),
        (calculate_cb_abnormal_turnover, "cb_abnormal_turnover_20d", 1.40),
    ],
)
def test_plugins_expose_audited_point_in_time_fields(
    calculator,
    factor_name: str,
    expected_value: float,
) -> None:
    result = calculator(make_bundle(), {}).set_index("ts_code")

    assert result.loc["110001.SH", "raw_factor"] == pytest.approx(expected_value)
    assert result.loc["110001.SH", "available_date"] == pd.Timestamp("2024-01-05")
    assert pd.isna(result.loc["110002.SH", "raw_factor"])
    assert pd.isna(result.loc["110002.SH", "available_date"])


def test_plugin_metadata_freezes_direction_and_family_controls() -> None:
    assert RELATIVE_PREMIUM_SPEC.direction == -1
    assert RELATIVE_PREMIUM_SPEC.neutralizers == (
        "log_remaining_balance",
        "remaining_maturity_years",
        "rating",
    )
    assert STOCK_RETURN_SPEC.direction == 1
    assert STOCK_RETURN_SPEC.neutralizers == (
        "log_stock_total_market_cap",
        "ci_industry_control",
    )
    assert CB_RETURN_SPEC.direction == -1
    assert CB_ABNORMAL_TURNOVER_SPEC.direction == -1


@pytest.mark.parametrize(
    "calculator",
    [
        calculate_relative_premium,
        calculate_stock_return,
        calculate_cb_return,
        calculate_cb_abnormal_turnover,
    ],
)
def test_plugins_reject_unknown_parameters(calculator) -> None:
    with pytest.raises(ValueError, match="does not accept parameters"):
        calculator(make_bundle(), {"window": 10})
