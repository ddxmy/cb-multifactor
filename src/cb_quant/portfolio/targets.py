"""Transparent long-only target construction from factor scores."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_long_only_targets(
    score_panel: pd.DataFrame,
    *,
    score_col: str = "processed_factor",
    portfolio_size: int = 20,
    cash_reserve: float = 0.1,
    max_weight: float = 1.0,
    direction: int = 1,
) -> pd.DataFrame:
    """Select eligible Top-N bonds and assign capped equal target weights."""
    if not isinstance(portfolio_size, int) or portfolio_size <= 0:
        raise ValueError("portfolio_size must be a positive integer")
    if not 0 <= cash_reserve < 1:
        raise ValueError("cash_reserve must be in [0, 1)")
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in (0, 1]")
    if direction not in {-1, 1}:
        raise ValueError("direction must be -1 or 1")
    required = {
        "signal_date",
        "ts_code",
        score_col,
        "entry_allowed",
        "exit_required",
    }
    if missing := required.difference(score_panel.columns):
        raise KeyError(f"score panel is missing columns: {sorted(missing)}")

    result = score_panel.copy()
    result["signal_date"] = pd.to_datetime(
        result["signal_date"], errors="raise"
    ).dt.normalize()
    result["ts_code"] = result["ts_code"].astype("string")
    if result.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError("score panel contains duplicate signal-date and bond keys")
    if not pd.api.types.is_numeric_dtype(result[score_col]):
        raise TypeError(f"{score_col} must have a numeric dtype")

    result["selection_score"] = direction * result[score_col]
    result["is_selected"] = False
    result["selection_rank"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["target_weight"] = 0.0
    result["implied_cash_weight"] = 1.0

    for _, group in result.groupby("signal_date", sort=True):
        candidate = (
            group["entry_allowed"].fillna(False).astype(bool)
            & ~group["exit_required"].fillna(False).astype(bool)
            & group["selection_score"].notna()
            & np.isfinite(group["selection_score"])
        )
        ranked = group.loc[candidate].sort_values(
            ["selection_score", "ts_code"], ascending=[False, True]
        )
        selected = ranked.head(portfolio_size)
        if not selected.empty:
            target = min((1.0 - cash_reserve) / len(selected), max_weight)
            result.loc[selected.index, "is_selected"] = True
            result.loc[selected.index, "selection_rank"] = range(1, len(selected) + 1)
            result.loc[selected.index, "target_weight"] = target
        total_weight = float(result.loc[group.index, "target_weight"].sum())
        result.loc[group.index, "implied_cash_weight"] = 1.0 - total_weight

    return result.sort_values(
        ["signal_date", "is_selected", "selection_rank", "ts_code"],
        ascending=[True, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
