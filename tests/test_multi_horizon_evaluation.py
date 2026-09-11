from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cb_quant.evaluation import (
    calculate_multi_horizon_ic,
    calculate_horizon_ir_series,
    summarize_horizon_ic,
    summarize_horizon_ir,
)


def test_multi_horizon_ic_keeps_each_prediction_horizon_separate() -> None:
    panel = pd.DataFrame(
        {
            "signal_date": ["2024-01-02"] * 6,
            "ts_code": ["A", "B", "C"] * 2,
            "horizon_days": [1] * 3 + [5] * 3,
            "processed_factor": [1.0, 2.0, 3.0] * 2,
            "forward_return": [0.01, 0.02, 0.03, 0.03, 0.02, 0.01],
        }
    )

    result = calculate_multi_horizon_ic(panel, min_assets=3).set_index(
        "horizon_days"
    )

    assert result.loc[1, "ic"] == pytest.approx(1.0)
    assert result.loc[1, "rank_ic"] == pytest.approx(1.0)
    assert result.loc[5, "ic"] == pytest.approx(-1.0)
    assert result.loc[5, "rank_ic"] == pytest.approx(-1.0)


def test_horizon_icir_uses_biweekly_observation_frequency() -> None:
    ic_series = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(
                ["2024-01-02", "2024-01-16", "2024-01-02", "2024-01-16"]
            ),
            "horizon_days": [1, 1, 5, 5],
            "asset_count": [20, 20, 20, 20],
            "ic": [0.1, 0.3, -0.1, 0.1],
            "rank_ic": [0.2, 0.4, 0.0, 0.2],
        }
    )

    result = summarize_horizon_ic(
        ic_series,
        periods_per_year=26,
        nw_lags=1,
    ).set_index(["horizon_days", "metric"])

    expected = 0.2 / np.std([0.1, 0.3], ddof=1) * np.sqrt(26)
    assert result.loc[(1, "ic"), "icir_annualized"] == pytest.approx(expected)
    assert result.loc[(1, "ic"), "observations"] == 2
    assert result.loc[(5, "rank_ic"), "mean"] == pytest.approx(0.1)


def _ir_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in (1, 5):
        for signal_date, scale in (("2024-01-02", 1.0), ("2024-01-16", 0.5)):
            for position in range(10):
                rows.append(
                    {
                        "signal_date": signal_date,
                        "ts_code": f"CB{position:02d}",
                        "horizon_days": horizon,
                        "processed_factor": float(10 - position),
                        "forward_return": scale * (0.10 - position * 0.01) / horizon,
                    }
                )
    return pd.DataFrame(rows)


def test_horizon_ir_uses_top_group_minus_dynamic_universe_as_primary_return() -> None:
    series = calculate_horizon_ir_series(_ir_panel(), group_count=5)
    first = series.loc[
        series["horizon_days"].eq(1)
        & series["signal_date"].eq(pd.Timestamp("2024-01-02"))
    ].iloc[0]

    assert first["G1"] == pytest.approx(0.095)
    assert first["universe_return"] == pytest.approx(0.055)
    assert first["active_return"] == pytest.approx(0.04)
    assert first["spread_raw"] == pytest.approx(0.08)

    summary = summarize_horizon_ir(series, periods_per_year=26).set_index(
        "horizon_days"
    )
    expected_active_ir = np.mean([0.04, 0.02]) / np.std(
        [0.04, 0.02], ddof=1
    ) * np.sqrt(26)
    assert summary.loc[1, "active_ir_annualized"] == pytest.approx(
        expected_active_ir
    )
    assert summary.loc[1, "observations"] == 2
