from __future__ import annotations

import pandas as pd
import pytest

from cb_quant.evaluation import run_multi_horizon_factor_evaluation
from cb_quant.factors import build_default_factor_registry
from scripts.run_g5_multi_horizon_factor import _summary_metrics


def make_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    context_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for signal_date, scale in (("2024-01-02", 1.0), ("2024-01-16", 0.5)):
        for position in range(10):
            code = f"CB{position:02d}"
            context_rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": code,
                    "close": 100.0 + position,
                    "conversion_premium": position / 100.0,
                }
            )
            for horizon in (1, 5, 10):
                label_rows.append(
                    {
                        "signal_date": signal_date,
                        "ts_code": code,
                        "horizon_days": horizon,
                        "forward_return": scale * (0.10 - position * 0.01) / horizon,
                        "label_usable": True,
                        "label_exclusion_reason": "eligible",
                    }
                )
    return pd.DataFrame(context_rows), pd.DataFrame(label_rows)


def test_multi_horizon_factor_run_preprocesses_once_and_separates_horizons() -> None:
    context, labels = make_inputs()

    result = run_multi_horizon_factor_evaluation(
        factor_name="double_low",
        factor_context=context,
        horizon_labels=labels,
        registry=build_default_factor_registry(),
        n_mad=3.0,
        group_count=5,
        min_ic_assets=5,
        periods_per_year=26,
        nw_lags=1,
    )

    processed = result.processed_factor.set_index(["signal_date", "ts_code"])
    date = pd.Timestamp("2024-01-02")
    assert processed.loc[(date, "CB00"), "processed_factor"] > 0
    assert processed.loc[(date, "CB09"), "processed_factor"] < 0
    assert len(result.factor_labels) == 60
    assert set(result.ic_summary["horizon_days"]) == {1, 5, 10}
    assert set(result.ir_summary["horizon_days"]) == {1, 5, 10}
    rank_ic = result.ic_summary.loc[
        result.ic_summary["metric"].eq("rank_ic")
    ].set_index("horizon_days")
    assert rank_ic.loc[1, "mean"] == pytest.approx(1.0)
    assert rank_ic.loc[5, "mean"] == pytest.approx(1.0)
    assert rank_ic.loc[10, "mean"] == pytest.approx(1.0)
    assert set(result.analytical_nav["horizon_days"]) == {1, 5, 10}
    summary = _summary_metrics(result).set_index("horizon_days")
    assert summary.loc[1, "g1_annualized_return"] > 0
    assert summary.loc[1, "g1_max_drawdown"] <= 0
    assert summary.loc[1, "active_max_drawdown"] <= 0


def test_multi_horizon_factor_run_preserves_unusable_label_audit_rows() -> None:
    context, labels = make_inputs()
    labels.loc[0, ["forward_return", "label_usable", "label_exclusion_reason"]] = [
        float("nan"),
        False,
        "entry_not_buyable",
    ]

    result = run_multi_horizon_factor_evaluation(
        factor_name="double_low",
        factor_context=context,
        horizon_labels=labels,
        registry=build_default_factor_registry(),
        group_count=5,
        min_ic_assets=5,
    )

    excluded = result.factor_labels.loc[
        result.factor_labels["label_exclusion_reason"].eq("entry_not_buyable")
    ]
    assert len(excluded) == 1
    assert pd.isna(excluded.iloc[0]["forward_return"])


def test_multi_horizon_factor_run_uses_materialized_g3c_standardization() -> None:
    signal_date = "2024-01-02"
    context = pd.DataFrame(
        {
            "signal_date": signal_date,
            "ts_code": [f"CB{position:02d}" for position in range(10)],
            "relative_premium_to_parity_median": range(10),
            "winsorized__relative_premium_to_parity_median": range(10),
            "standardized__relative_premium_to_parity_median": [
                1.0,
                0.8,
                0.6,
                0.4,
                0.2,
                -0.2,
                -0.4,
                -0.6,
                -0.8,
                -1.0,
            ],
        }
    )
    labels = pd.DataFrame(
        {
            "signal_date": signal_date,
            "ts_code": [f"CB{position:02d}" for position in range(10)],
            "horizon_days": 5,
            "forward_return": [
                -0.10,
                -0.08,
                -0.06,
                -0.04,
                -0.02,
                0.02,
                0.04,
                0.06,
                0.08,
                0.10,
            ],
            "label_usable": True,
            "label_exclusion_reason": "eligible",
        }
    )

    result = run_multi_horizon_factor_evaluation(
        factor_name="relative_premium_to_parity_median",
        factor_context=context,
        horizon_labels=labels,
        registry=build_default_factor_registry(),
        min_ic_assets=5,
    )

    assert result.processed_factor["processed_factor"].tolist() == pytest.approx(
        [-1.0, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    rank_ic = result.ic_summary.loc[
        result.ic_summary["metric"].eq("rank_ic"), "mean"
    ].iloc[0]
    assert rank_ic == pytest.approx(1.0)
