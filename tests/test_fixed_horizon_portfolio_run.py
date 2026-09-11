from __future__ import annotations

import pandas as pd
import pytest

import cb_quant.portfolio as portfolio


def test_fixed_horizon_portfolio_run_executes_selected_cohort_with_real_cash() -> None:
    scores = pd.DataFrame(
        {
            "signal_date": ["2024-01-01", "2024-01-01"],
            "ts_code": ["A", "B"],
            "processed_factor": [2.0, 1.0],
            "is_eligible": [True, False],
        }
    )
    schedule = pd.DataFrame(
        {
            "signal_date": ["2024-01-01"],
            "entry_date": ["2024-01-02"],
            "exit_date": ["2024-01-04"],
        }
    )
    market = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "ts_code": ["A", "A", "A"],
            "open": [100.0, 110.0, 120.0],
            "close": [110.0, 115.0, 120.0],
            "buy_allowed": [True, True, True],
            "sell_allowed": [True, True, True],
        }
    )
    lifecycle = pd.DataFrame(
        {
            "trade_date": market["trade_date"],
            "ts_code": ["A", "A", "A"],
            "lifecycle_state": ["active", "active", "active"],
            "entry_allowed": [True, True, True],
            "exit_required": [False, False, False],
            "settlement_value": [float("nan")] * 3,
            "settlement_date": [pd.NaT] * 3,
        }
    )

    run = portfolio.run_fixed_horizon_factor_portfolio(
        scores,
        schedule,
        market,
        lifecycle,
        portfolio_size=1,
        cash_reserve=0.1,
        max_weight=1.0,
        initial_cash=1_000.0,
        cost_rate=0.0,
        board_lot=1,
    )

    assert run.plan.targets[["execution_date", "target_weight"]].to_dict("records") == [
        {"execution_date": pd.Timestamp("2024-01-02"), "target_weight": pytest.approx(0.9)},
        {"execution_date": pd.Timestamp("2024-01-04"), "target_weight": pytest.approx(0.0)},
    ]
    assert run.execution.fills[["side", "quantity", "price"]].to_dict("records") == [
        {"side": "buy", "quantity": 9, "price": 100.0},
        {"side": "sell", "quantity": 9, "price": 120.0},
    ]
    assert run.summary["final_nav"] == pytest.approx(1_180.0)
    assert run.summary["total_return"] == pytest.approx(0.18)
