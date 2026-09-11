"""Performance summaries for cash-constrained execution results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cb_quant.execution import ExecutionResult


def summarize_execution_result(
    result: ExecutionResult,
    *,
    initial_cash: float,
    periods_per_year: int = 252,
) -> dict[str, object]:
    """Summarize net NAV, risk, turnover, costs, and execution diagnostics."""
    if not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("initial_cash must be positive and finite")
    if not isinstance(periods_per_year, int) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer")
    if result.nav.empty:
        raise ValueError("execution NAV cannot be empty")

    nav = result.nav.sort_values("trade_date").reset_index(drop=True).copy()
    nav_values = pd.to_numeric(nav["nav"], errors="raise")
    if nav_values.le(0).any():
        raise ValueError("execution NAV must remain positive")
    daily_returns = nav_values.pct_change()
    daily_returns.iloc[0] = nav_values.iloc[0] / initial_cash - 1.0
    observations = len(daily_returns)
    total_return = float(nav_values.iloc[-1] / initial_cash - 1.0)
    annualized_return = float(
        (nav_values.iloc[-1] / initial_cash) ** (periods_per_year / observations) - 1.0
    )
    annualized_volatility = float(daily_returns.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe_ratio = (
        float(daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(periods_per_year))
        if daily_returns.std(ddof=1) > 0
        else np.nan
    )
    wealth = pd.concat(
        [pd.Series([1.0]), nav_values.div(initial_cash)], ignore_index=True
    )
    max_drawdown = float((wealth / wealth.cummax() - 1.0).min())
    calmar_ratio = (
        float(annualized_return / abs(max_drawdown)) if max_drawdown < 0 else np.nan
    )
    traded_notional = float(pd.to_numeric(nav["traded_notional"], errors="raise").sum())
    average_nav = float(nav_values.mean())
    total_turnover = traded_notional / average_nav

    fills = result.fills
    return {
        "start_date": pd.Timestamp(nav["trade_date"].iloc[0]),
        "end_date": pd.Timestamp(nav["trade_date"].iloc[-1]),
        "trading_days": observations,
        "initial_cash": float(initial_cash),
        "final_nav": float(nav_values.iloc[-1]),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "average_cash_weight": float((nav["cash"] / nav_values).mean()),
        "average_gross_exposure": float((nav["holdings_value"] / nav_values).mean()),
        "total_traded_notional": traded_notional,
        "total_turnover": total_turnover,
        "annualized_turnover": float(total_turnover * periods_per_year / observations),
        "total_transaction_cost": float(
            pd.to_numeric(nav["daily_cost"], errors="raise").sum()
        ),
        "transaction_cost_to_initial_cash": float(
            pd.to_numeric(nav["daily_cost"], errors="raise").sum() / initial_cash
        ),
        "blocked_buy_orders": int(nav["blocked_buys"].sum()),
        "blocked_sell_orders": int(nav["blocked_sells"].sum()),
        "filled_buy_orders": int((fills.get("side", pd.Series(dtype=str)) == "buy").sum()),
        "filled_sell_orders": int((fills.get("side", pd.Series(dtype=str)) == "sell").sum()),
    }
