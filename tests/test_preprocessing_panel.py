from __future__ import annotations

import pandas as pd
import pytest

from cb_quant.preprocessing import (
    build_mad_sensitivity,
    merge_factor_panels,
    winsorize_factor_panel,
)


def make_panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = pd.DataFrame(
        {
            "signal_date": ["2024-01-02"] * 4,
            "ts_code": ["A", "B", "C", "D"],
        }
    )
    valuation = keys.assign(
        stk_code=["SA", "SB", "SC", "SD"],
        is_eligible=True,
        rating=["AA", "AA", "AAA", "AAA"],
        remain_size=[3.0, 5.0, 8.0, 12.0],
        valuation_factor=[0.0, 1.0, 2.0, 100.0],
    )
    stock = keys.assign(stock_factor=[1.0, 2.0, 3.0, 4.0])
    bond = keys.assign(bond_factor=[4.0, 3.0, 2.0, 1.0])
    return valuation, stock, bond


def test_merge_factor_panels_preserves_context_and_exact_keys() -> None:
    valuation, stock, bond = make_panels()

    result = merge_factor_panels(
        valuation,
        stock,
        bond,
        valuation_factors=("valuation_factor",),
        stock_factors=("stock_factor",),
        bond_factors=("bond_factor",),
    )

    assert result.columns.tolist() == [
        "signal_date",
        "ts_code",
        "stk_code",
        "is_eligible",
        "rating",
        "remain_size",
        "valuation_factor",
        "stock_factor",
        "bond_factor",
    ]
    assert len(result) == 4


def test_merge_factor_panels_rejects_key_mismatch() -> None:
    valuation, stock, bond = make_panels()
    stock = stock.loc[stock["ts_code"].ne("D")]

    with pytest.raises(ValueError, match="keys do not match"):
        merge_factor_panels(
            valuation,
            stock,
            bond,
            valuation_factors=("valuation_factor",),
            stock_factors=("stock_factor",),
            bond_factors=("bond_factor",),
        )


def test_mad_sensitivity_reports_clipping_without_using_returns() -> None:
    valuation, stock, bond = make_panels()
    panel = merge_factor_panels(
        valuation,
        stock,
        bond,
        valuation_factors=("valuation_factor",),
        stock_factors=("stock_factor",),
        bond_factors=("bond_factor",),
    )

    result = build_mad_sensitivity(
        panel,
        factor_columns=("valuation_factor", "stock_factor"),
        multipliers=(1.0, 3.0),
    ).set_index(["factor_name", "mad_multiplier"])

    assert result.loc[("valuation_factor", 1.0), "clipped_count"] == 2
    assert result.loc[("valuation_factor", 3.0), "clipped_count"] == 1
    assert result.loc[("stock_factor", 3.0), "clipped_count"] == 0
    assert "forward_return" not in result.columns


def test_primary_winsorized_panel_keeps_raw_values_and_adds_audit_columns() -> None:
    valuation, stock, bond = make_panels()
    panel = merge_factor_panels(
        valuation,
        stock,
        bond,
        valuation_factors=("valuation_factor",),
        stock_factors=("stock_factor",),
        bond_factors=("bond_factor",),
    )

    result = winsorize_factor_panel(
        panel,
        factor_columns=("valuation_factor", "stock_factor"),
        n_mad=3.0,
    )

    assert result["valuation_factor"].tolist() == [0.0, 1.0, 2.0, 100.0]
    assert result.loc[3, "winsorized__valuation_factor"] < 100.0
    assert result["winsorized__stock_factor"].tolist() == [1.0, 2.0, 3.0, 4.0]
