"""Merged factor panel and return-blind preprocessing diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from cb_quant.cb_trading import CB_TRADING_FACTOR_COLUMNS
from cb_quant.stock_linkage import (
    LINKAGE_FACTOR_COLUMNS,
    STOCK_FACTOR_COLUMNS,
    STOCK_LINKAGE_FACTOR_COLUMNS,
)

from .cross_section import winsorize_mad


VALUATION_FACTOR_COLUMNS = (
    "conversion_premium",
    "double_low",
    "relative_premium_to_parity_median",
    "parity_value_score",
)
ALL_FACTOR_COLUMNS = (
    *VALUATION_FACTOR_COLUMNS,
    *STOCK_LINKAGE_FACTOR_COLUMNS,
    *CB_TRADING_FACTOR_COLUMNS,
)


def _prepare_panel(
    frame: pd.DataFrame,
    *,
    name: str,
    required_columns: Sequence[str],
) -> pd.DataFrame:
    required = {"signal_date", "ts_code", *required_columns}
    if missing := required.difference(frame.columns):
        raise KeyError(f"{name} panel is missing columns: {sorted(missing)}")
    result = frame.copy()
    result["signal_date"] = pd.to_datetime(
        result["signal_date"], errors="raise"
    ).dt.normalize()
    result["ts_code"] = result["ts_code"].astype("string")
    if result[["signal_date", "ts_code"]].isna().any().any():
        raise ValueError(f"{name} panel contains missing keys")
    if result.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError(f"{name} panel contains duplicate keys")
    return result


def _key_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[["signal_date", "ts_code"]])


def merge_factor_panels(
    valuation_panel: pd.DataFrame,
    stock_linkage_panel: pd.DataFrame,
    bond_trading_panel: pd.DataFrame,
    *,
    valuation_factors: Sequence[str] = VALUATION_FACTOR_COLUMNS,
    stock_factors: Sequence[str] = STOCK_LINKAGE_FACTOR_COLUMNS,
    bond_factors: Sequence[str] = CB_TRADING_FACTOR_COLUMNS,
) -> pd.DataFrame:
    """Merge exact-key G2/G3 panels while retaining G2 point-in-time context."""
    factor_columns = tuple(valuation_factors) + tuple(stock_factors) + tuple(bond_factors)
    if len(set(factor_columns)) != len(factor_columns):
        raise ValueError("factor columns must be unique across source panels")

    valuation = _prepare_panel(
        valuation_panel,
        name="valuation",
        required_columns=valuation_factors,
    )
    stock = _prepare_panel(
        stock_linkage_panel,
        name="stock-linkage",
        required_columns=stock_factors,
    )
    bond = _prepare_panel(
        bond_trading_panel,
        name="bond-trading",
        required_columns=bond_factors,
    )
    expected_keys = _key_index(valuation)
    for name, candidate in (("stock-linkage", stock), ("bond-trading", bond)):
        if set(_key_index(candidate)) != set(expected_keys):
            raise ValueError(f"{name} panel keys do not match valuation panel keys")

    context_columns = [
        column
        for column in valuation.columns
        if column not in {"signal_date", "ts_code", *valuation_factors}
    ]
    result = valuation[
        ["signal_date", "ts_code", *context_columns, *valuation_factors]
    ].merge(
        stock[["signal_date", "ts_code", *stock_factors]],
        on=["signal_date", "ts_code"],
        how="inner",
        validate="one_to_one",
    )
    result = result.merge(
        bond[["signal_date", "ts_code", *bond_factors]],
        on=["signal_date", "ts_code"],
        how="inner",
        validate="one_to_one",
    )
    return result.sort_values(["signal_date", "ts_code"]).reset_index(drop=True)


def winsorize_factor_panel(
    panel: pd.DataFrame,
    *,
    factor_columns: Sequence[str] = ALL_FACTOR_COLUMNS,
    n_mad: float = 3.0,
) -> pd.DataFrame:
    """Add one auditable date-local MAD-winsorized column per raw factor."""
    if missing := set(factor_columns).difference(panel.columns):
        raise KeyError(f"factor panel is missing columns: {sorted(missing)}")
    result = panel.copy()
    for factor in factor_columns:
        output_column = f"winsorized__{factor}"
        transformed = winsorize_mad(
            result[["signal_date", factor]],
            value_col=factor,
            n_mad=n_mad,
            output_col=output_column,
        )
        result[output_column] = transformed[output_column]
    return result


def build_mad_sensitivity(
    panel: pd.DataFrame,
    *,
    factor_columns: Sequence[str] = ALL_FACTOR_COLUMNS,
    multipliers: Sequence[float] = (2.5, 3.0, 3.5),
) -> pd.DataFrame:
    """Summarize clipping intensity without accepting any forward-return input."""
    if not factor_columns:
        raise ValueError("factor_columns cannot be empty")
    if not multipliers:
        raise ValueError("multipliers cannot be empty")
    forbidden = {"forward_return", "label_end_date", "entry_open", "exit_open"}
    if forbidden.intersection(panel.columns):
        raise ValueError("MAD sensitivity input cannot contain forward-return labels")
    if missing := set(factor_columns).difference(panel.columns):
        raise KeyError(f"factor panel is missing columns: {sorted(missing)}")

    records: list[dict[str, object]] = []
    signal_dates = pd.to_datetime(panel["signal_date"], errors="raise").dt.normalize()
    for factor in factor_columns:
        raw = pd.to_numeric(panel[factor], errors="coerce")
        for multiplier in multipliers:
            transformed = winsorize_mad(
                panel[["signal_date", factor]],
                value_col=factor,
                n_mad=float(multiplier),
                output_col="winsorized_factor",
            )["winsorized_factor"]
            observed = raw.notna()
            changed = observed & ~np.isclose(
                raw.to_numpy(dtype=float, na_value=np.nan),
                transformed.to_numpy(dtype=float, na_value=np.nan),
                equal_nan=True,
            )
            lower = changed & transformed.gt(raw)
            upper = changed & transformed.lt(raw)
            adjustment = (transformed - raw).abs().where(observed)
            affected_by_date = pd.Series(changed, index=panel.index).groupby(
                signal_dates
            ).any()

            rank_correlations = []
            comparison = pd.DataFrame(
                {
                    "signal_date": signal_dates,
                    "raw": raw,
                    "winsorized": transformed,
                }
            )
            for _, group in comparison.groupby("signal_date", sort=False):
                valid = group[["raw", "winsorized"]].dropna()
                if valid["raw"].nunique() > 1 and valid["winsorized"].nunique() > 1:
                    rank_correlations.append(
                        valid["raw"].corr(valid["winsorized"], method="spearman")
                    )

            observations = int(observed.sum())
            clipped_count = int(changed.sum())
            records.append(
                {
                    "factor_name": factor,
                    "mad_multiplier": float(multiplier),
                    "observations": observations,
                    "clipped_count": clipped_count,
                    "clipped_rate": clipped_count / observations if observations else np.nan,
                    "lower_clipped_count": int(lower.sum()),
                    "upper_clipped_count": int(upper.sum()),
                    "affected_signal_dates": int(affected_by_date.sum()),
                    "median_absolute_adjustment": float(adjustment.dropna().median())
                    if observations
                    else np.nan,
                    "maximum_absolute_adjustment": float(adjustment.dropna().max())
                    if observations
                    else np.nan,
                    "mean_cross_sectional_rank_correlation": float(
                        pd.Series(rank_correlations, dtype=float).mean()
                    ),
                }
            )
    return pd.DataFrame(records).sort_values(
        ["factor_name", "mad_multiplier"]
    ).reset_index(drop=True)
