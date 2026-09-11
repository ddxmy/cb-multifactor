from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cb_quant.evaluation import (
    assign_quantile_groups,
    calculate_ic_series,
    calculate_quantile_nav,
    calculate_quantile_returns,
    summarize_ic,
)


def test_ic_series_matches_hand_calculated_pearson_and_rank_ic() -> None:
    panel = pd.DataFrame(
        {
            "signal_date": ["2024-01-02"] * 4 + ["2024-01-16"] * 2,
            "ts_code": ["A", "B", "C", "D", "A", "B"],
            "processed_factor": [1.0, 2.0, 3.0, 4.0, 1.0, 1.0],
            "forward_return": [0.01, 0.02, 0.03, 0.04, 0.01, -0.01],
        }
    )

    result = calculate_ic_series(panel, min_assets=3)

    assert result.loc[0, "ic"] == pytest.approx(1.0)
    assert result.loc[0, "rank_ic"] == pytest.approx(1.0)
    assert result.loc[0, "asset_count"] == 4
    assert pd.isna(result.loc[1, "ic"])


def test_ic_summary_reports_mean_hit_rate_and_annualized_icir() -> None:
    series = pd.DataFrame(
        {
            "signal_date": pd.date_range("2024-01-01", periods=4),
            "ic": [0.1, -0.1, 0.2, 0.2],
            "rank_ic": [0.2, 0.0, 0.4, 0.2],
        }
    )

    result = summarize_ic(series, periods_per_year=26, nw_lags=1).set_index("metric")

    assert result.loc["ic", "mean"] == pytest.approx(0.1)
    assert result.loc["ic", "positive_rate"] == pytest.approx(0.75)
    expected_icir = 0.1 / np.std([0.1, -0.1, 0.2, 0.2], ddof=1) * np.sqrt(26)
    assert result.loc["ic", "icir_annualized"] == pytest.approx(expected_icir)
    assert np.isfinite(result.loc["rank_ic", "nw_t_stat"])


def test_quantile_assignment_is_equal_count_descending_and_deterministic() -> None:
    panel = pd.DataFrame(
        {
            "signal_date": ["2024-01-02"] * 7,
            "ts_code": ["G", "F", "E", "D", "C", "B", "A"],
            "processed_factor": [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "forward_return": [0.0] * 7,
        }
    )

    result = assign_quantile_groups(panel, group_count=5)
    groups = result.set_index("ts_code")["group"]

    assert groups["A"] == 1
    assert groups["B"] == 1
    assert groups["C"] == 2
    assert groups["E"] == 3
    assert groups["F"] < groups["G"]
    assert len(result) == len(panel)


def test_quantile_returns_and_nav_keep_raw_and_unit_gross_spreads_separate() -> None:
    grouped = pd.DataFrame(
        {
            "signal_date": ["2024-01-02"] * 5 + ["2024-01-16"] * 5,
            "group": [1, 2, 3, 4, 5] * 2,
            "forward_return": [0.10, 0.05, 0.02, 0.00, -0.10] + [0.0] * 5,
        }
    )

    returns = calculate_quantile_returns(grouped, group_count=5)
    nav = calculate_quantile_nav(returns)

    assert returns.loc[0, "spread_raw"] == pytest.approx(0.20)
    assert returns.loc[0, "long_short"] == pytest.approx(0.10)
    assert nav.loc[0, "G1"] == pytest.approx(1.10)
    assert nav.loc[1, "G1"] == pytest.approx(1.10)
    assert nav.loc[0, "long_short"] == pytest.approx(1.10)


def test_quantile_assignment_rejects_too_small_cross_section() -> None:
    panel = pd.DataFrame(
        {
            "signal_date": ["2024-01-02"] * 4,
            "ts_code": ["A", "B", "C", "D"],
            "processed_factor": [4.0, 3.0, 2.0, 1.0],
            "forward_return": [0.04, 0.03, 0.02, 0.01],
        }
    )
    with pytest.raises(ValueError, match="fewer assets"):
        assign_quantile_groups(panel, group_count=5)
