from __future__ import annotations

import pandas as pd
import pytest

from cb_quant.factors import FactorDataBundle, build_default_factor_registry
from cb_quant.pipeline import run_single_factor_research


def make_context(signal_dates: list[str]) -> pd.DataFrame:
    rows = []
    for date_index, signal_date in enumerate(signal_dates):
        for bond_index, code in enumerate(["A", "B", "C", "D", "E"]):
            rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": code,
                    "conversion_premium": 0.10 + bond_index * 0.05,
                    "entry_allowed": True,
                    "exit_required": False,
                    "rating": "AA" if bond_index < 3 else "AAA",
                    "remaining_maturity": 2.0 + date_index,
                }
            )
    return pd.DataFrame(rows)


def make_prices(trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for date_index, trade_date in enumerate(trading_dates):
        for bond_index, code in enumerate(["A", "B", "C", "D", "E"]):
            price = 100.0 + date_index * (5 - bond_index)
            rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": code,
                    "open": price,
                    "close": price,
                    "buy_allowed": True,
                    "sell_allowed": True,
                }
            )
    return pd.DataFrame(rows)


def test_conversion_premium_runs_through_evidence_and_portfolio_layers() -> None:
    trading_dates = pd.bdate_range("2024-01-02", periods=8)
    signal_dates = ["2024-01-02", "2024-01-04", "2024-01-08"]
    context = make_context(signal_dates)
    market = make_prices(trading_dates)

    result = run_single_factor_research(
        factor_name="conversion_premium",
        factor_context=context,
        registry=build_default_factor_registry(),
        trading_dates=trading_dates,
        open_prices=market[["trade_date", "ts_code", "open"]],
        market=market,
        n_mad=3.0,
        group_count=5,
        min_ic_assets=5,
        portfolio_size=2,
        cash_reserve=0.1,
        max_weight=0.5,
        initial_cash=1_000_000.0,
        cost_rate=0.0,
        board_lot=10,
        allow_missing_lifecycle=True,
    )

    assert len(result.raw_factor) == 15
    assert result.processed_factor["processed_factor"].corr(
        result.processed_factor["raw_factor"]
    ) < 0
    assert result.factor_labels["signal_date"].nunique() == 2
    assert result.label_diagnostics.empty
    assert result.ic_series["rank_ic"].gt(0).all()
    assert set(result.quantile_returns.columns) >= {"G1", "G5", "long_short"}
    assert result.targets.groupby("execution_date")["target_weight"].sum().le(0.9).all()
    assert not result.execution.fills.empty
    assert result.execution.nav["accounting_error"].abs().max() < 1e-9


def test_pipeline_accepts_validated_factor_data_bundle() -> None:
    trading_dates = pd.bdate_range("2024-01-02", periods=8)
    signal_dates = ["2024-01-02", "2024-01-04", "2024-01-08"]
    signal = make_context(signal_dates)
    market = make_prices(trading_dates)

    result = run_single_factor_research(
        factor_name="conversion_premium",
        factor_context=FactorDataBundle(signal=signal),
        registry=build_default_factor_registry(),
        trading_dates=trading_dates,
        open_prices=market[["trade_date", "ts_code", "open"]],
        market=market,
        min_ic_assets=5,
        portfolio_size=2,
        cost_rate=0.0,
        allow_missing_lifecycle=True,
    )

    assert len(result.raw_factor) == len(signal)
    assert result.raw_factor["raw_factor"].notna().all()


def test_pipeline_rejects_unknown_factor() -> None:
    trading_dates = pd.bdate_range("2024-01-02", periods=4)
    market = make_prices(trading_dates)
    with pytest.raises(KeyError, match="unknown factor"):
        run_single_factor_research(
            factor_name="missing",
            factor_context=make_context(["2024-01-02", "2024-01-03"]),
            registry=build_default_factor_registry(),
            trading_dates=trading_dates,
            open_prices=market[["trade_date", "ts_code", "open"]],
            market=market,
            allow_missing_lifecycle=True,
        )


def test_pipeline_uses_investable_evidence_sample_and_audits_missing_labels() -> None:
    trading_dates = pd.bdate_range("2024-01-02", periods=8)
    signal_dates = ["2024-01-02", "2024-01-04", "2024-01-08"]
    context = make_context(signal_dates)
    context.loc[
        context["signal_date"].eq("2024-01-02") & context["ts_code"].eq("E"),
        "entry_allowed",
    ] = False
    market = make_prices(trading_dates)
    market = market.loc[
        ~(
            market["trade_date"].eq(pd.Timestamp("2024-01-05"))
            & market["ts_code"].eq("D")
        )
    ].copy()

    result = run_single_factor_research(
        factor_name="conversion_premium",
        factor_context=context,
        registry=build_default_factor_registry(),
        trading_dates=trading_dates,
        open_prices=market[["trade_date", "ts_code", "open"]],
        market=market,
        min_ic_assets=3,
        group_count=3,
        portfolio_size=2,
        cost_rate=0.0,
        allow_missing_lifecycle=True,
    )

    first_date_codes = result.factor_labels.loc[
        result.factor_labels["signal_date"].eq(pd.Timestamp("2024-01-02")),
        "ts_code",
    ].tolist()
    assert "E" not in first_date_codes
    assert result.label_diagnostics["ts_code"].tolist() == ["D", "D"]
    assert set(result.label_diagnostics["label_exclusion_reason"]) == {
        "missing_entry_open",
        "missing_exit_open",
    }
