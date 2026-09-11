"""Shared preprocessing boundary for registered factor experiments."""

from __future__ import annotations

import pandas as pd

from cb_quant.factors import FactorDataBundle

from .cross_section import standardize_zscore, winsorize_mad


def preprocess_registered_factor(
    data: FactorDataBundle,
    raw: pd.DataFrame,
    *,
    factor_name: str,
    direction: int,
    n_mad: float,
) -> pd.DataFrame:
    """Align scores to audited G3c output or apply the return-blind fallback."""
    standardized_column = f"standardized__{factor_name}"
    winsorized_column = f"winsorized__{factor_name}"
    if standardized_column in data.signal:
        materialized_columns = [
            "signal_date",
            "ts_code",
            standardized_column,
        ]
        if winsorized_column in data.signal:
            materialized_columns.append(winsorized_column)
        materialized = data.signal[materialized_columns].copy()
        processed = raw.merge(
            materialized,
            on=["signal_date", "ts_code"],
            how="left",
            validate="one_to_one",
        )
        processed["standardized_factor"] = pd.to_numeric(
            processed.pop(standardized_column), errors="coerce"
        )
        if winsorized_column in processed:
            processed["winsorized_factor"] = pd.to_numeric(
                processed.pop(winsorized_column), errors="coerce"
            )
        else:
            processed["winsorized_factor"] = processed["raw_factor"]
    else:
        processed = winsorize_mad(raw, n_mad=n_mad)
        processed = standardize_zscore(
            processed,
            value_col="winsorized_factor",
        )
    processed["processed_factor"] = direction * processed["standardized_factor"]
    return processed
