"""Shared orchestration for one convertible-bond factor research run."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from .evaluation import (
    assign_quantile_groups,
    calculate_ic_series,
    calculate_quantile_nav,
    calculate_quantile_returns,
    summarize_ic,
)
from .execution import ExecutionResult, run_execution_backtest
from .factors import FactorDataBundle, FactorRegistry, ensure_factor_data_bundle
from .labels import attach_o2o_returns, build_execution_schedule
from .portfolio import build_long_only_targets
from .preprocessing import (
    neutralize_cross_section,
    standardize_zscore,
    winsorize_mad,
)


@dataclass(frozen=True)
class SingleFactorResearchResult:
    """All evidence and portfolio artifacts from one standardized run."""

    raw_factor: pd.DataFrame
    processed_factor: pd.DataFrame
    factor_labels: pd.DataFrame
    label_diagnostics: pd.DataFrame
    ic_series: pd.DataFrame
    ic_summary: pd.DataFrame
    grouped_factor: pd.DataFrame
    quantile_returns: pd.DataFrame
    quantile_nav: pd.DataFrame
    targets: pd.DataFrame
    execution: ExecutionResult


def run_single_factor_research(
    *,
    factor_name: str,
    factor_context: FactorDataBundle | pd.DataFrame,
    registry: FactorRegistry,
    trading_dates: Iterable,
    open_prices: pd.DataFrame,
    market: pd.DataFrame,
    lifecycle: pd.DataFrame | None = None,
    factor_parameters: Mapping[str, object] | None = None,
    n_mad: float = 3.0,
    numeric_controls: Sequence[str] = (),
    categorical_controls: Sequence[str] = (),
    group_count: int = 5,
    min_ic_assets: int = 5,
    periods_per_year: int = 26,
    nw_lags: int = 5,
    portfolio_size: int = 20,
    cash_reserve: float = 0.1,
    max_weight: float = 0.1,
    initial_cash: float = 1_000_000.0,
    cost_rate: float = 0.0015,
    board_lot: int = 10,
    drop_unusable_labels: bool = True,
    allow_missing_lifecycle: bool = False,
) -> SingleFactorResearchResult:
    """Run shared preprocessing, O2O evidence, targets, and daily execution."""
    factor_data = ensure_factor_data_bundle(factor_context)
    signal = factor_data.signal
    spec = registry.get_spec(factor_name)
    raw = registry.calculate(factor_name, factor_data, factor_parameters)
    processed = winsorize_mad(raw, n_mad=n_mad)

    context_columns = ["signal_date", "ts_code"]
    optional_context = [
        column
        for column in {
            *numeric_controls,
            *categorical_controls,
            "entry_allowed",
            "exit_required",
        }
        if column in signal.columns
    ]
    context = signal[[*context_columns, *optional_context]].copy()
    context["signal_date"] = pd.to_datetime(
        context["signal_date"], errors="raise"
    ).dt.normalize()
    context["ts_code"] = context["ts_code"].astype("string")
    if context.duplicated(context_columns).any():
        raise ValueError("factor context contains duplicate signal-date and bond keys")
    processed = processed.merge(
        context,
        on=context_columns,
        how="left",
        validate="one_to_one",
    )

    preprocessing_input = "winsorized_factor"
    if numeric_controls or categorical_controls:
        processed = neutralize_cross_section(
            processed,
            value_col=preprocessing_input,
            numeric_controls=numeric_controls,
            categorical_controls=categorical_controls,
        )
        preprocessing_input = "neutralized_factor"
    processed = standardize_zscore(processed, value_col=preprocessing_input)
    processed["processed_factor"] = spec.direction * processed["standardized_factor"]
    if "entry_allowed" not in processed:
        processed["entry_allowed"] = True
    if "exit_required" not in processed:
        processed["exit_required"] = False
    processed["entry_allowed"] = processed["entry_allowed"].fillna(False).astype(bool)
    processed["exit_required"] = processed["exit_required"].fillna(False).astype(bool)

    signal_dates = sorted(processed["signal_date"].dropna().unique())
    schedule = build_execution_schedule(trading_dates, signal_dates)
    evidence_sample = processed.loc[
        processed["entry_allowed"] & ~processed["exit_required"]
    ].copy()
    labels = attach_o2o_returns(
        evidence_sample,
        schedule,
        open_prices,
        drop_unusable=drop_unusable_labels,
    )
    label_diagnostics = labels.attrs.get(
        "unusable_execution_rows", pd.DataFrame()
    ).copy()
    ic_series = calculate_ic_series(labels, min_assets=min_ic_assets)
    ic_summary = summarize_ic(
        ic_series,
        periods_per_year=periods_per_year,
        nw_lags=nw_lags,
    )
    grouped = assign_quantile_groups(labels, group_count=group_count)
    quantile_returns = calculate_quantile_returns(grouped, group_count=group_count)
    quantile_nav = calculate_quantile_nav(quantile_returns)

    target_scores = build_long_only_targets(
        processed,
        portfolio_size=portfolio_size,
        cash_reserve=cash_reserve,
        max_weight=max_weight,
    )
    target_scores = target_scores.merge(
        schedule[["signal_date", "entry_date"]],
        on="signal_date",
        how="inner",
        validate="many_to_one",
    )
    targets = target_scores.rename(columns={"entry_date": "execution_date"})
    execution = run_execution_backtest(
        targets[["execution_date", "ts_code", "target_weight"]],
        market,
        lifecycle=lifecycle,
        initial_cash=initial_cash,
        cost_rate=cost_rate,
        board_lot=board_lot,
        allow_missing_lifecycle=allow_missing_lifecycle,
    )
    return SingleFactorResearchResult(
        raw_factor=raw,
        processed_factor=processed,
        factor_labels=labels,
        label_diagnostics=label_diagnostics,
        ic_series=ic_series,
        ic_summary=ic_summary,
        grouped_factor=grouped,
        quantile_returns=quantile_returns,
        quantile_nav=quantile_nav,
        targets=targets,
        execution=execution,
    )
