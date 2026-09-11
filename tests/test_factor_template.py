from __future__ import annotations

import pandas as pd
import pytest

from cb_quant.factors import build_factor_output


def make_signal() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": ["2024-01-05", "2024-01-05"],
            "ts_code": ["110001.SH", "110002.SH"],
            "source_column": [1.0, 2.0],
        },
        index=[10, 20],
    )


def test_build_factor_output_preserves_keys_and_marks_observed_values() -> None:
    signal = make_signal()
    raw_factor = pd.Series([0.25, pd.NA], index=signal.index, dtype="Float64")

    result = build_factor_output(signal, raw_factor)

    assert result.columns.tolist() == [
        "signal_date",
        "ts_code",
        "raw_factor",
        "available_date",
    ]
    assert result["raw_factor"].tolist() == [0.25, pd.NA]
    assert result.loc[0, "available_date"] == pd.Timestamp("2024-01-05")
    assert pd.isna(result.loc[1, "available_date"])


def test_build_factor_output_rejects_misaligned_raw_factor() -> None:
    signal = make_signal()
    raw_factor = pd.Series([0.25, 0.50], index=[0, 1])

    with pytest.raises(ValueError, match="index"):
        build_factor_output(signal, raw_factor)


def test_build_factor_output_accepts_explicit_availability_series() -> None:
    signal = make_signal()
    raw_factor = pd.Series([0.25, 0.50], index=signal.index)
    available = pd.Series(
        ["2024-01-04", "2024-01-05"],
        index=signal.index,
    )

    result = build_factor_output(signal, raw_factor, available_date=available)

    assert result["available_date"].tolist() == [
        pd.Timestamp("2024-01-04"),
        pd.Timestamp("2024-01-05"),
    ]
