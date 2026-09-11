from __future__ import annotations

import pandas as pd
import pytest

from cb_quant.execution import ExecutionResult
from cb_quant.factors import FactorSpec
from cb_quant.pipeline import SingleFactorResearchResult
from cb_quant.reporting import (
    build_factor_run_index,
    build_run_identity,
    write_factor_run_record,
)


def make_spec(name: str = "test_factor") -> FactorSpec:
    return FactorSpec(
        name=name,
        family="test",
        description="Synthetic factor for reporting contract tests.",
        direction=1,
        required_fields=("source",),
        default_parameters={"window": 20},
    )


def identity_kwargs() -> dict[str, object]:
    return {
        "factor_spec": make_spec(),
        "factor_parameters": {"window": 20, "minimum": 0.8},
        "sample_partition": "replication_2018_2023",
        "universe_fingerprint": "universe-a",
        "preprocessing": {"mad": 3.0},
        "label": {"name": "next_open_to_open"},
        "rebalance": {"frequency": "biweekly"},
        "portfolio": {"size": 30},
        "execution": {"cost_bps": 15},
        "input_fingerprints": {"market": "market-a", "terms": "terms-a"},
    }


def make_result() -> SingleFactorResearchResult:
    signal_dates = pd.to_datetime(["2024-01-02", "2024-01-16"])
    raw = pd.DataFrame(
        {
            "signal_date": signal_dates.repeat(2),
            "ts_code": ["A", "B", "A", "B"],
            "raw_factor": [1.0, 2.0, 1.5, pd.NA],
            "available_date": [signal_dates[0], signal_dates[0], signal_dates[1], pd.NaT],
        }
    )
    processed = raw.assign(
        processed_factor=pd.Series([1.0, -1.0, 0.5, pd.NA], dtype="Float64")
    )
    labels = processed.iloc[:3].assign(forward_return=[0.01, -0.01, 0.02])
    ic_series = pd.DataFrame(
        {
            "signal_date": signal_dates,
            "asset_count": [2, 1],
            "ic": [0.2, pd.NA],
            "rank_ic": [0.3, pd.NA],
        }
    )
    ic_summary = pd.DataFrame(
        {
            "metric": ["ic", "rank_ic"],
            "observations": [1, 1],
            "mean": [0.2, 0.3],
            "std": [pd.NA, pd.NA],
            "positive_rate": [1.0, 1.0],
            "icir_annualized": [pd.NA, pd.NA],
            "nw_t_stat": [pd.NA, pd.NA],
            "nw_p_value": [pd.NA, pd.NA],
            "nw_lags": [0, 0],
        }
    )
    quantile_returns = pd.DataFrame(
        {
            "signal_date": signal_dates,
            "G1": [0.02, 0.01],
            "G2": [-0.01, 0.00],
            "spread_raw": [0.03, 0.01],
            "long_short": [0.015, 0.005],
        }
    )
    quantile_nav = quantile_returns.copy()
    for column in ["G1", "G2", "spread_raw", "long_short"]:
        quantile_nav[column] = (1.0 + quantile_nav[column]).cumprod()
    trade_dates = pd.bdate_range("2024-01-02", periods=4)
    nav = pd.DataFrame(
        {
            "trade_date": trade_dates,
            "cash": [100.0, 90.0, 95.0, 105.0],
            "holdings_value": [900.0, 920.0, 930.0, 940.0],
            "receivable_value": 0.0,
            "nav": [1_000.0, 1_010.0, 1_025.0, 1_045.0],
            "cash_ledger_balance": [100.0, 90.0, 95.0, 105.0],
            "daily_cost": [0.0, 1.0, 0.0, 1.0],
            "traded_notional": [900.0, 100.0, 0.0, 100.0],
            "blocked_buys": [0, 1, 0, 0],
            "blocked_sells": [0, 0, 0, 1],
            "accounting_error": 0.0,
        }
    )
    execution = ExecutionResult(
        nav=nav,
        positions=pd.DataFrame(),
        orders=pd.DataFrame(
            {
                "trade_date": trade_dates[:2],
                "ts_code": ["A", "B"],
                "side": ["buy", "sell"],
                "requested_quantity": [10, 10],
                "filled_quantity": [0, 0],
                "status": ["blocked", "blocked"],
                "reason": ["buy_not_allowed", "sell_not_allowed"],
            }
        ),
        fills=pd.DataFrame(),
        receivables=pd.DataFrame(),
        attribution=pd.DataFrame(),
    )
    return SingleFactorResearchResult(
        raw_factor=raw,
        processed_factor=processed,
        factor_labels=labels,
        label_diagnostics=pd.DataFrame(),
        ic_series=ic_series,
        ic_summary=ic_summary,
        grouped_factor=pd.DataFrame(),
        quantile_returns=quantile_returns,
        quantile_nav=quantile_nav,
        targets=pd.DataFrame(),
        execution=execution,
    )


def test_run_identity_is_deterministic_and_comparison_scope_is_separate() -> None:
    settings = identity_kwargs()
    first = build_run_identity(**settings)
    reordered = build_run_identity(
        **{
            **settings,
            "factor_parameters": {"minimum": 0.8, "window": 20},
            "input_fingerprints": {"terms": "terms-a", "market": "market-a"},
        }
    )
    assert first == reordered

    changed_window = build_run_identity(
        **{**settings, "factor_parameters": {"window": 10, "minimum": 0.8}}
    )
    changed_cost = build_run_identity(
        **{**settings, "execution": {"cost_bps": 25}}
    )
    other_factor = build_run_identity(
        **{**settings, "factor_spec": make_spec("other_factor")}
    )
    assert changed_window.run_id != first.run_id
    assert changed_cost.run_id != first.run_id
    assert changed_cost.comparison_group_id != first.comparison_group_id
    assert other_factor.run_id != first.run_id
    assert other_factor.comparison_group_id == first.comparison_group_id


def test_writer_creates_complete_immutable_evidence_package(tmp_path) -> None:
    settings = identity_kwargs()
    identity = build_run_identity(**settings)
    run_dir = write_factor_run_record(
        output_root=tmp_path,
        identity=identity,
        factor_spec=settings["factor_spec"],
        factor_parameters=settings["factor_parameters"],
        sample_partition=settings["sample_partition"],
        result=make_result(),
    )

    expected = {
        "run_manifest.json",
        "factor_summary.csv",
        "coverage_by_date.parquet",
        "raw_factor.parquet",
        "processed_factor.parquet",
        "ic_series.parquet",
        "ic_summary.csv",
        "quantile_returns.parquet",
        "quantile_nav.parquet",
        "portfolio_nav.parquet",
        "portfolio_metrics.csv",
        "annual_stability.csv",
        "execution_diagnostics.parquet",
        "ic_history.png",
        "quantile_nav.png",
        "portfolio_nav.png",
        "drawdown.png",
    }
    assert {path.name for path in run_dir.iterdir()} == expected
    manifest_before = (run_dir / "run_manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="immutable"):
        write_factor_run_record(
            output_root=tmp_path,
            identity=identity,
            factor_spec=settings["factor_spec"],
            factor_parameters=settings["factor_parameters"],
            sample_partition=settings["sample_partition"],
            result=make_result(),
        )
    assert (run_dir / "run_manifest.json").read_bytes() == manifest_before


def test_cross_factor_index_accepts_only_comparable_valid_manifests(tmp_path) -> None:
    settings = identity_kwargs()
    first_identity = build_run_identity(**settings)
    first_dir = write_factor_run_record(
        output_root=tmp_path,
        identity=first_identity,
        factor_spec=settings["factor_spec"],
        factor_parameters=settings["factor_parameters"],
        sample_partition=settings["sample_partition"],
        result=make_result(),
    )
    second_spec = make_spec("other_factor")
    second_identity = build_run_identity(**{**settings, "factor_spec": second_spec})
    write_factor_run_record(
        output_root=tmp_path,
        identity=second_identity,
        factor_spec=second_spec,
        factor_parameters=settings["factor_parameters"],
        sample_partition=settings["sample_partition"],
        result=make_result(),
    )

    index_path = tmp_path / "factor_run_index_v1.parquet"
    index = build_factor_run_index(tmp_path, output_path=index_path)

    assert len(index) == 2
    assert index["run_id"].nunique() == 2
    assert index["comparison_group_id"].nunique() == 1
    assert index_path.exists()
    assert first_dir.exists()
