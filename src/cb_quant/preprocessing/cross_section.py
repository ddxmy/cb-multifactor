"""Point-in-time cross-sectional factor transformations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


MAD_NORMAL_SCALE = 1.4826


def _prepare(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if "signal_date" not in frame:
        raise KeyError("frame is missing signal_date")
    if value_col not in frame:
        raise KeyError(f"frame is missing {value_col}")
    if not pd.api.types.is_numeric_dtype(frame[value_col]):
        raise TypeError(f"{value_col} must have a numeric dtype")
    result = frame.copy()
    result["signal_date"] = pd.to_datetime(
        result["signal_date"], errors="raise"
    ).dt.normalize()
    return result


def winsorize_mad(
    frame: pd.DataFrame,
    *,
    value_col: str = "raw_factor",
    n_mad: float = 3.0,
    output_col: str = "winsorized_factor",
) -> pd.DataFrame:
    """Clip each signal-date cross-section at median plus/minus scaled MAD."""
    if not np.isfinite(n_mad) or n_mad <= 0:
        raise ValueError("n_mad must be positive and finite")
    result = _prepare(frame, value_col)

    def clip_group(values: pd.Series) -> pd.Series:
        if not values.notna().any():
            return values
        median = values.median()
        mad = (values - median).abs().median()
        if pd.isna(median) or pd.isna(mad) or np.isclose(mad, 0.0):
            return values
        distance = n_mad * MAD_NORMAL_SCALE * mad
        return values.clip(median - distance, median + distance)

    result[output_col] = result.groupby("signal_date", sort=False)[value_col].transform(
        clip_group
    )
    return result


def standardize_zscore(
    frame: pd.DataFrame,
    *,
    value_col: str = "raw_factor",
    output_col: str = "standardized_factor",
) -> pd.DataFrame:
    """Compute population-standard-deviation Z-scores within each signal date."""
    result = _prepare(frame, value_col)

    def standardize(values: pd.Series) -> pd.Series:
        mean = values.mean()
        dispersion = values.std(ddof=0)
        if pd.isna(dispersion) or np.isclose(dispersion, 0.0):
            output = pd.Series(np.nan, index=values.index, dtype=float)
            output.loc[values.notna()] = 0.0
            return output
        return (values - mean) / dispersion

    result[output_col] = result.groupby("signal_date", sort=False)[value_col].transform(
        standardize
    )
    return result


def neutralize_cross_section(
    frame: pd.DataFrame,
    *,
    value_col: str,
    numeric_controls: Sequence[str] = (),
    categorical_controls: Sequence[str] = (),
    output_col: str = "neutralized_factor",
) -> pd.DataFrame:
    """Return date-local OLS residuals after controlling for specified exposures."""
    result = _prepare(frame, value_col)
    controls = tuple(numeric_controls) + tuple(categorical_controls)
    if missing := set(controls).difference(result.columns):
        raise KeyError(f"frame is missing control columns: {sorted(missing)}")
    if len(set(controls)) != len(controls):
        raise ValueError("neutralization controls cannot contain duplicates")
    result[output_col] = np.nan

    for _, group in result.groupby("signal_date", sort=False):
        design_parts = []
        if numeric_controls:
            numeric = group[list(numeric_controls)].apply(pd.to_numeric, errors="coerce")
            design_parts.append(numeric.astype(float))
        if categorical_controls:
            categorical = pd.get_dummies(
                group[list(categorical_controls)].astype("string"),
                prefix=list(categorical_controls),
                drop_first=True,
                dtype=float,
            )
            design_parts.append(categorical)
        design = (
            pd.concat(design_parts, axis=1)
            if design_parts
            else pd.DataFrame(index=group.index)
        )
        design.insert(0, "intercept", 1.0)
        complete = group[value_col].notna() & design.notna().all(axis=1)
        if categorical_controls:
            complete &= group[list(categorical_controls)].notna().all(axis=1)
        x = design.loc[complete].to_numpy(dtype=float)
        matrix_rank = int(np.linalg.matrix_rank(x)) if len(x) else 0
        if (
            int(complete.sum()) <= matrix_rank
            or matrix_rank < design.shape[1]
        ):
            continue
        y = group.loc[complete, value_col].to_numpy(dtype=float)
        coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        result.loc[group.index[complete], output_col] = y - x @ coefficients
    return result
