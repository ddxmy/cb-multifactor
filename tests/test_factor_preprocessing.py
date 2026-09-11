from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from cb_quant.preprocessing import (
    neutralize_cross_section,
    standardize_zscore,
    winsorize_mad,
)


def test_mad_winsorization_is_date_local_and_preserves_missing_values() -> None:
    frame = pd.DataFrame(
        {
            "signal_date": ["2024-01-02"] * 5 + ["2024-01-03"] * 2,
            "ts_code": [f"B{i}" for i in range(7)],
            "raw_factor": [0.0, 1.0, 2.0, 100.0, np.nan, 10.0, 10.0],
        }
    )

    result = winsorize_mad(frame, n_mad=1.0)

    first = result.loc[result["signal_date"].eq(pd.Timestamp("2024-01-02"))]
    assert first["winsorized_factor"].min() == pytest.approx(1.5 - 1.4826)
    assert first["winsorized_factor"].max() == pytest.approx(1.5 + 1.4826)
    assert pd.isna(result.loc[result["ts_code"].eq("B4"), "winsorized_factor"]).all()
    second = result.loc[result["signal_date"].eq(pd.Timestamp("2024-01-03"))]
    assert second["winsorized_factor"].tolist() == [10.0, 10.0]


def test_mad_winsorization_keeps_all_missing_cross_section_without_warning() -> None:
    frame = pd.DataFrame(
        {
            "signal_date": ["2024-01-02", "2024-01-02"],
            "raw_factor": [np.nan, np.nan],
        }
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = winsorize_mad(frame)

    assert not captured
    assert result["winsorized_factor"].isna().all()


def test_zscore_is_date_local_and_zero_dispersion_maps_to_zero() -> None:
    frame = pd.DataFrame(
        {
            "signal_date": ["2024-01-02"] * 3 + ["2024-01-03"] * 2,
            "raw_factor": [1.0, 2.0, 3.0, 5.0, 5.0],
        }
    )

    result = standardize_zscore(frame)

    first = result.loc[result["signal_date"].eq(pd.Timestamp("2024-01-02"))]
    assert first["standardized_factor"].mean() == pytest.approx(0.0)
    assert first["standardized_factor"].std(ddof=0) == pytest.approx(1.0)
    second = result.loc[result["signal_date"].eq(pd.Timestamp("2024-01-03"))]
    assert second["standardized_factor"].tolist() == [0.0, 0.0]


def test_neutralization_removes_numeric_and_categorical_exposure_by_date() -> None:
    frame = pd.DataFrame(
        {
            "signal_date": ["2024-01-02"] * 6,
            "ts_code": [f"B{i}" for i in range(6)],
            "factor": [2.0, 4.0, 6.0, 13.0, 15.0, 17.0],
            "remaining_maturity": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "rating": ["AA", "AA", "AA", "AAA", "AAA", "AAA"],
        }
    )

    result = neutralize_cross_section(
        frame,
        value_col="factor",
        numeric_controls=("remaining_maturity",),
        categorical_controls=("rating",),
    )

    assert np.abs(result["neutralized_factor"]).max() < 1e-10


def test_neutralization_leaves_incomplete_rows_missing_and_rejects_bad_parameters() -> None:
    frame = pd.DataFrame(
        {
            "signal_date": ["2024-01-02"] * 4,
            "factor": [1.0, 2.0, 3.0, 4.0],
            "size": [1.0, 2.0, np.nan, 4.0],
        }
    )

    result = neutralize_cross_section(
        frame,
        value_col="factor",
        numeric_controls=("size",),
    )
    assert pd.isna(result.loc[2, "neutralized_factor"])
    categorical = frame.assign(industry=["A", "A", None, "B"])
    categorical_result = neutralize_cross_section(
        categorical,
        value_col="factor",
        categorical_controls=("industry",),
    )
    assert pd.isna(categorical_result.loc[2, "neutralized_factor"])
    with pytest.raises(ValueError, match="positive"):
        winsorize_mad(frame.rename(columns={"factor": "raw_factor"}), n_mad=0)
    with pytest.raises(KeyError, match="rating"):
        neutralize_cross_section(
            frame,
            value_col="factor",
            categorical_controls=("rating",),
        )


def test_neutralization_requires_identified_design_and_positive_residual_degrees() -> None:
    saturated = pd.DataFrame(
        {
            "signal_date": ["2024-01-02"] * 2,
            "factor": [1.0, 2.0],
            "size": [1.0, 2.0],
        }
    )
    collinear = pd.DataFrame(
        {
            "signal_date": ["2024-01-02"] * 4,
            "factor": [1.0, 2.0, 3.0, 4.0],
            "size": [1.0, 2.0, 3.0, 4.0],
            "size_twice": [2.0, 4.0, 6.0, 8.0],
        }
    )

    saturated_result = neutralize_cross_section(
        saturated, value_col="factor", numeric_controls=("size",)
    )
    collinear_result = neutralize_cross_section(
        collinear,
        value_col="factor",
        numeric_controls=("size", "size_twice"),
    )

    assert saturated_result["neutralized_factor"].isna().all()
    assert collinear_result["neutralized_factor"].isna().all()
