"""Reusable multi-horizon evidence run for one registered factor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cb_quant.factors import FactorDataBundle, FactorRegistry, ensure_factor_data_bundle
from cb_quant.preprocessing import preprocess_registered_factor

from .single_factor import (
    calculate_horizon_ir_series,
    calculate_multi_horizon_ic,
    summarize_horizon_ic,
    summarize_horizon_ir,
)


@dataclass(frozen=True)
class MultiHorizonFactorEvaluationResult:
    """Auditable factor, label, IC, IR, and analytical-NAV outputs."""

    raw_factor: pd.DataFrame
    processed_factor: pd.DataFrame
    factor_labels: pd.DataFrame
    ic_series: pd.DataFrame
    ic_summary: pd.DataFrame
    ir_series: pd.DataFrame
    ir_summary: pd.DataFrame
    analytical_nav: pd.DataFrame


def _attach_labels(
    processed: pd.DataFrame,
    horizon_labels: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "signal_date",
        "ts_code",
        "horizon_days",
        "forward_return",
        "label_usable",
        "label_exclusion_reason",
    }
    if missing := required.difference(horizon_labels.columns):
        raise KeyError(f"horizon labels are missing columns: {sorted(missing)}")
    labels = horizon_labels.copy()
    labels["signal_date"] = pd.to_datetime(
        labels["signal_date"], errors="raise"
    ).dt.normalize()
    labels["ts_code"] = labels["ts_code"].astype("string")
    labels["horizon_days"] = pd.to_numeric(
        labels["horizon_days"], errors="raise"
    ).astype(int)
    if labels.duplicated(["signal_date", "ts_code", "horizon_days"]).any():
        raise ValueError("horizon labels contain duplicate keys")
    if labels["horizon_days"].le(0).any():
        raise ValueError("horizon_days must be positive")
    usable = labels["label_usable"].eq(True)
    usable_returns = pd.to_numeric(
        labels.loc[usable, "forward_return"], errors="coerce"
    )
    if usable_returns.isna().any() or not np.isfinite(usable_returns).all():
        raise ValueError("usable labels must have finite forward returns")
    labels.loc[~usable, "forward_return"] = np.nan

    result = labels.merge(
        processed,
        on=["signal_date", "ts_code"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if result["_merge"].ne("both").any():
        raise ValueError("some horizon labels do not match the factor context")
    return result.drop(columns="_merge").sort_values(
        ["horizon_days", "signal_date", "ts_code"]
    ).reset_index(drop=True)


def _build_analytical_nav(ir_series: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    return_columns = {
        "G1": "g1_nav",
        "universe_return": "universe_nav",
        "active_return": "active_nav",
        "spread_raw": "spread_nav",
    }
    for horizon, group in ir_series.groupby("horizon_days", sort=True):
        ordered = group.sort_values("signal_date").copy()
        nav = ordered[["signal_date"]].copy()
        nav["horizon_days"] = int(horizon)
        for return_column, nav_column in return_columns.items():
            nav[nav_column] = (
                1.0 + pd.to_numeric(ordered[return_column], errors="coerce")
            ).cumprod()
        frames.append(nav)
    if not frames:
        return pd.DataFrame(
            columns=["signal_date", "horizon_days", *return_columns.values()]
        )
    return pd.concat(frames, ignore_index=True).sort_values(
        ["horizon_days", "signal_date"]
    ).reset_index(drop=True)


def run_multi_horizon_factor_evaluation(
    *,
    factor_name: str,
    factor_context: FactorDataBundle | pd.DataFrame,
    horizon_labels: pd.DataFrame,
    registry: FactorRegistry,
    factor_parameters: dict[str, object] | None = None,
    n_mad: float = 3.0,
    group_count: int = 5,
    min_ic_assets: int = 5,
    periods_per_year: int = 26,
    nw_lags: int = 5,
) -> MultiHorizonFactorEvaluationResult:
    """Run return-blind preprocessing and multi-horizon factor diagnostics."""
    data = ensure_factor_data_bundle(factor_context)
    spec = registry.get_spec(factor_name)
    raw = registry.calculate(factor_name, data, factor_parameters)
    processed = preprocess_registered_factor(
        data,
        raw,
        factor_name=factor_name,
        direction=spec.direction,
        n_mad=n_mad,
    )
    labels = _attach_labels(processed, horizon_labels)
    ic_series = calculate_multi_horizon_ic(
        labels,
        min_assets=min_ic_assets,
    )
    ic_summary = summarize_horizon_ic(
        ic_series,
        periods_per_year=periods_per_year,
        nw_lags=nw_lags,
    )
    ir_series = calculate_horizon_ir_series(
        labels,
        group_count=group_count,
    )
    ir_summary = summarize_horizon_ir(
        ir_series,
        periods_per_year=periods_per_year,
    )
    analytical_nav = _build_analytical_nav(ir_series)
    return MultiHorizonFactorEvaluationResult(
        raw_factor=raw,
        processed_factor=processed,
        factor_labels=labels,
        ic_series=ic_series,
        ic_summary=ic_summary,
        ir_series=ir_series,
        ir_summary=ir_summary,
        analytical_nav=analytical_nav,
    )
