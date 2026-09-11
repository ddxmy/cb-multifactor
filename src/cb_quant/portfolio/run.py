"""End-to-end fixed-horizon portfolio simulation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cb_quant.execution import ExecutionResult, run_execution_backtest

from .buffer import RankBufferTargetPlan, build_rank_buffer_targets
from .cohorts import CohortTargetPlan, build_fixed_horizon_cohort_targets


@dataclass(frozen=True)
class FixedHorizonPortfolioRun:
    """Portfolio plan, execution ledgers, and net performance summary."""

    plan: CohortTargetPlan
    execution: ExecutionResult
    summary: dict[str, object]


@dataclass(frozen=True)
class RankBufferPortfolioRun:
    """Rank-buffer plan, execution ledgers, and net performance summary."""

    plan: RankBufferTargetPlan
    execution: ExecutionResult
    summary: dict[str, object]


def run_fixed_horizon_factor_portfolio(
    score_panel: pd.DataFrame,
    schedule: pd.DataFrame,
    market: pd.DataFrame,
    lifecycle: pd.DataFrame,
    *,
    portfolio_size: int = 20,
    cash_reserve: float = 0.1,
    max_weight: float = 0.1,
    initial_cash: float = 1_000_000.0,
    cost_rate: float = 0.0015,
    board_lot: int = 10,
) -> FixedHorizonPortfolioRun:
    """Build fixed-horizon targets and run the production execution simulator."""
    from cb_quant.reporting.portfolio import summarize_execution_result

    plan = build_fixed_horizon_cohort_targets(
        score_panel,
        schedule,
        portfolio_size=portfolio_size,
        cash_reserve=cash_reserve,
        max_weight=max_weight,
    )
    if plan.targets.empty:
        raise ValueError("cohort plan produced no executable targets")

    start_date = pd.Timestamp(plan.targets["execution_date"].min())
    target_codes = set(plan.targets["ts_code"].astype(str))
    market_dates = pd.to_datetime(market["trade_date"], errors="raise").dt.normalize()
    market_codes = market["ts_code"].astype(str)
    market_subset = market.loc[
        market_dates.ge(start_date) & market_codes.isin(target_codes)
    ].copy()
    lifecycle_dates = pd.to_datetime(
        lifecycle["trade_date"], errors="raise"
    ).dt.normalize()
    lifecycle_codes = lifecycle["ts_code"].astype(str)
    lifecycle_subset = lifecycle.loc[
        lifecycle_dates.ge(start_date) & lifecycle_codes.isin(target_codes)
    ].copy()

    execution = run_execution_backtest(
        plan.targets,
        market_subset,
        lifecycle=lifecycle_subset,
        initial_cash=initial_cash,
        cost_rate=cost_rate,
        board_lot=board_lot,
    )
    summary = summarize_execution_result(
        execution,
        initial_cash=initial_cash,
        periods_per_year=252,
    )
    return FixedHorizonPortfolioRun(
        plan=plan,
        execution=execution,
        summary=summary,
    )


def run_rank_buffer_factor_portfolio(
    score_panel: pd.DataFrame,
    schedule: pd.DataFrame,
    market: pd.DataFrame,
    lifecycle: pd.DataFrame,
    *,
    portfolio_size: int = 20,
    exit_rank: int = 30,
    cash_reserve: float = 0.1,
    max_weight: float = 0.1,
    liquidation_date: str | pd.Timestamp | None = None,
    initial_cash: float = 1_000_000.0,
    cost_rate: float = 0.0015,
    board_lot: int = 10,
) -> RankBufferPortfolioRun:
    """Build rank-buffer targets and run the production execution simulator."""
    from cb_quant.reporting.portfolio import summarize_execution_result

    plan = build_rank_buffer_targets(
        score_panel,
        schedule,
        portfolio_size=portfolio_size,
        exit_rank=exit_rank,
        cash_reserve=cash_reserve,
        max_weight=max_weight,
        liquidation_date=liquidation_date,
    )
    if plan.targets.empty:
        raise ValueError("rank-buffer plan produced no executable targets")

    start_date = pd.Timestamp(plan.targets["execution_date"].min())
    target_codes = set(plan.targets["ts_code"].astype(str))
    market_dates = pd.to_datetime(market["trade_date"], errors="raise").dt.normalize()
    market_codes = market["ts_code"].astype(str)
    market_subset = market.loc[
        market_dates.ge(start_date) & market_codes.isin(target_codes)
    ].copy()
    lifecycle_dates = pd.to_datetime(
        lifecycle["trade_date"], errors="raise"
    ).dt.normalize()
    lifecycle_codes = lifecycle["ts_code"].astype(str)
    lifecycle_subset = lifecycle.loc[
        lifecycle_dates.ge(start_date) & lifecycle_codes.isin(target_codes)
    ].copy()

    execution = run_execution_backtest(
        plan.targets,
        market_subset,
        lifecycle=lifecycle_subset,
        initial_cash=initial_cash,
        cost_rate=cost_rate,
        board_lot=board_lot,
    )
    summary = summarize_execution_result(
        execution,
        initial_cash=initial_cash,
        periods_per_year=252,
    )
    return RankBufferPortfolioRun(
        plan=plan,
        execution=execution,
        summary=summary,
    )
