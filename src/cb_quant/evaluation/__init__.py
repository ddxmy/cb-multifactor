"""Single-factor evidence and diagnostics."""

from .horizon_run import (
    MultiHorizonFactorEvaluationResult,
    run_multi_horizon_factor_evaluation,
)

from .single_factor import (
    assign_quantile_groups,
    calculate_horizon_ir_series,
    calculate_ic_series,
    calculate_multi_horizon_ic,
    calculate_quantile_nav,
    calculate_quantile_returns,
    summarize_horizon_ic,
    summarize_horizon_ir,
    summarize_ic,
)

__all__ = [
    "MultiHorizonFactorEvaluationResult",
    "assign_quantile_groups",
    "calculate_horizon_ir_series",
    "calculate_ic_series",
    "calculate_multi_horizon_ic",
    "calculate_quantile_nav",
    "calculate_quantile_returns",
    "summarize_horizon_ic",
    "summarize_horizon_ir",
    "summarize_ic",
    "run_multi_horizon_factor_evaluation",
]
