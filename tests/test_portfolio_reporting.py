from __future__ import annotations

import pandas as pd
import pytest

import cb_quant.reporting as reporting
from cb_quant.execution import ExecutionResult


def test_execution_summary_includes_first_day_return_and_audited_trading_totals() -> None:
    result = ExecutionResult(
        nav=pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "cash": [10.0, 11.0],
                "holdings_value": [91.0, 88.0],
                "receivable_value": [0.0, 0.0],
                "nav": [101.0, 99.0],
                "daily_cost": [0.1, 0.2],
                "traded_notional": [90.0, 90.0],
                "blocked_buys": [1, 0],
                "blocked_sells": [0, 2],
            }
        ),
        positions=pd.DataFrame(),
        orders=pd.DataFrame(
            {"side": ["buy", "sell", "sell"], "status": ["filled", "filled", "blocked"]}
        ),
        fills=pd.DataFrame({"side": ["buy", "sell"]}),
        receivables=pd.DataFrame(),
        attribution=pd.DataFrame(),
    )

    summary = reporting.summarize_execution_result(
        result,
        initial_cash=100.0,
        periods_per_year=252,
    )

    assert summary["total_return"] == pytest.approx(-0.01)
    assert summary["annualized_return"] == pytest.approx(0.99 ** (252 / 2) - 1.0)
    assert summary["max_drawdown"] == pytest.approx(99.0 / 101.0 - 1.0)
    assert summary["total_transaction_cost"] == pytest.approx(0.3)
    assert summary["total_traded_notional"] == pytest.approx(180.0)
    assert summary["blocked_buy_orders"] == 1
    assert summary["blocked_sell_orders"] == 2
    assert summary["filled_buy_orders"] == 1
    assert summary["filled_sell_orders"] == 1
