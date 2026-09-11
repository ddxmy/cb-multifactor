from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cb_quant.preprocessing import (
    FactorPreprocessingPolicy,
    add_cb_neutralization_controls,
    attach_stock_market_cap_control,
    preprocess_factor_family,
)


def make_panel() -> pd.DataFrame:
    rows = []
    for date_index, signal_date in enumerate(["2024-01-02", "2024-01-03"]):
        for index in range(12):
            log_size = 1.0 + index / 5
            maturity = 1.0 + (index % 4)
            rating = "AA" if index < 6 else "AAA"
            rating_effect = 2.0 if rating == "AAA" else 0.0
            residual = (-1.0 if index % 2 else 1.0) * (0.1 + date_index * 0.01)
            rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": f"B{index:02d}",
                    "remain_size": np.exp(log_size),
                    "maturity_date": pd.Timestamp(signal_date) + pd.Timedelta(days=365 * maturity),
                    "rating": rating,
                    "rating_score": 17.0 if rating == "AA" else 19.0,
                    "winsorized__absolute_value": float(index),
                    "winsorized__relative_value": (
                        3.0 * log_size + 0.5 * maturity + rating_effect + residual
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_adds_log_balance_and_remaining_maturity_controls() -> None:
    result = add_cb_neutralization_controls(make_panel())

    assert result["log_remaining_balance"].notna().all()
    assert result["remaining_maturity_years"].between(0.9, 4.1).all()
    assert result["rating"].notna().all()


def test_family_policy_preserves_absolute_factor_and_neutralizes_relative_factor() -> None:
    panel = add_cb_neutralization_controls(make_panel())
    policies = (
        FactorPreprocessingPolicy(name="absolute_value"),
        FactorPreprocessingPolicy(
            name="relative_value",
            numeric_controls=("log_remaining_balance", "remaining_maturity_years"),
            categorical_controls=("rating",),
        ),
    )

    result, diagnostics = preprocess_factor_family(panel, policies)

    assert "neutralized__absolute_value" not in result
    assert result.groupby("signal_date")["standardized__absolute_value"].mean().abs().max() < 1e-12
    assert result.groupby("signal_date")["standardized__relative_value"].mean().abs().max() < 1e-10
    diagnostics = diagnostics.set_index("factor_name")
    assert diagnostics.loc["absolute_value", "neutralized"] == False  # noqa: E712
    assert diagnostics.loc["relative_value", "neutralized"] == True  # noqa: E712
    assert diagnostics.loc["relative_value", "post_mean_absolute_control_correlation"] < 1e-10
    assert diagnostics.loc["relative_value", "pre_mean_absolute_control_correlation"] > 0.2


def test_rejects_forward_labels_and_invalid_remaining_balance() -> None:
    panel = make_panel()
    with pytest.raises(ValueError, match="forward-return"):
        preprocess_factor_family(
            add_cb_neutralization_controls(panel).assign(forward_return=0.1),
            (FactorPreprocessingPolicy(name="absolute_value"),),
        )
    panel.loc[0, "remain_size"] = 0.0
    with pytest.raises(ValueError, match="remaining balance"):
        add_cb_neutralization_controls(panel)


def test_stock_market_cap_control_uses_latest_prior_observation_only() -> None:
    panel = pd.DataFrame(
        {
            "signal_date": ["2024-01-05", "2024-01-08", "2024-01-05"],
            "ts_code": ["B1", "B1", "B2"],
            "stk_code": ["600001.SH", "600001.SH", "000001.SZ"],
            "ci_industry_code": ["CI005", "CI005", "CI010"],
        }
    )
    market_cap = pd.DataFrame(
        {
            "stk_code": ["600001.SH", "000001.SZ", "600001.SH", "600001.SH"],
            "trade_date": ["2024-01-04", "2024-01-03", "2024-01-08", "2024-01-09"],
            "total_market_cap": [100.0, 80.0, 120.0, 999.0],
        }
    )

    result = attach_stock_market_cap_control(panel, market_cap)

    assert result["stock_market_cap_observation_date"].tolist() == [
        pd.Timestamp("2024-01-04"),
        pd.Timestamp("2024-01-08"),
        pd.Timestamp("2024-01-03"),
    ]
    assert result["stock_market_cap_staleness_days"].tolist() == [1, 0, 2]
    assert result["total_market_cap"].tolist() == [100.0, 120.0, 80.0]
    assert result["log_stock_total_market_cap"].tolist() == pytest.approx(
        np.log([100.0, 120.0, 80.0])
    )
    assert result["ci_industry_control"].tolist() == ["CI005", "CI005", "CI010"]


def test_stock_market_cap_control_keeps_missing_industry_explicit() -> None:
    panel = pd.DataFrame(
        {
            "signal_date": ["2024-01-05"],
            "ts_code": ["B1"],
            "stk_code": ["600001.SH"],
            "ci_industry_code": [None],
        }
    )
    market_cap = pd.DataFrame(
        {
            "stk_code": ["600001.SH"],
            "trade_date": ["2024-01-04"],
            "total_market_cap": [100.0],
        }
    )

    result = attach_stock_market_cap_control(panel, market_cap)

    assert pd.isna(result.loc[0, "ci_industry_control"])


def test_stock_market_cap_control_preserves_existing_panel_trade_date() -> None:
    panel = pd.DataFrame(
        {
            "signal_date": ["2024-01-05"],
            "trade_date": ["2024-01-05"],
            "ts_code": ["B1"],
            "stk_code": ["600001.SH"],
            "ci_industry_code": ["CI005"],
        }
    )
    market_cap = pd.DataFrame(
        {
            "stk_code": ["600001.SH"],
            "trade_date": ["2024-01-04"],
            "total_market_cap": [100.0],
        }
    )

    result = attach_stock_market_cap_control(panel, market_cap)

    assert result.loc[0, "trade_date"] == "2024-01-05"
    assert result.loc[0, "stock_market_cap_observation_date"] == pd.Timestamp(
        "2024-01-04"
    )
