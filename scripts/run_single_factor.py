"""Run one configuration-driven convertible-bond factor experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cb_quant.config import SAMPLE_PARTITION_BOUNDS, load_backtest_config
from cb_quant.factors import FactorDataBundle, build_default_factor_registry
from cb_quant.pipeline import run_single_factor_research
from cb_quant.reporting import (
    build_run_identity,
    resolve_factor_run_directory,
    write_factor_run_record,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "backtest_v1.json"
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "artifacts" / "G5" / "factor_runs"


@dataclass(frozen=True)
class LoadedResearchInputs:
    """Shared, validated inputs loaded once for one or many factor runs."""

    factor_data: FactorDataBundle
    market: pd.DataFrame
    lifecycle: pd.DataFrame
    trading_dates: pd.DatetimeIndex
    universe_fingerprint: str
    input_fingerprints: Mapping[str, str]


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported input format for {path}; use Parquet or CSV")


def _read_optional_table(path: Path | None) -> pd.DataFrame:
    return pd.DataFrame() if path is None else _read_table(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_signal_keys(signal: pd.DataFrame) -> str:
    keys = signal[["signal_date", "ts_code"]].copy()
    keys["signal_date"] = pd.to_datetime(keys["signal_date"], errors="raise").dt.normalize()
    keys["ts_code"] = keys["ts_code"].astype("string")
    keys = keys.sort_values(["signal_date", "ts_code"]).reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(keys, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def load_research_inputs(
    *,
    factor_context_path: Path,
    market_path: Path,
    lifecycle_path: Path,
    valuation_path: Path | None = None,
    cb_daily_path: Path | None = None,
    stock_daily_path: Path | None = None,
) -> LoadedResearchInputs:
    """Load shared factor and execution inputs once for a research batch."""
    signal = _read_table(factor_context_path)
    market = _read_table(market_path)
    lifecycle = _read_table(lifecycle_path)
    valuation = _read_optional_table(valuation_path)
    cb_daily = _read_optional_table(cb_daily_path)
    stock_daily = _read_optional_table(stock_daily_path)
    if not stock_daily.empty and "stk_code" not in stock_daily and "ts_code" in stock_daily:
        stock_daily = stock_daily.rename(columns={"ts_code": "stk_code"})
    factor_data = FactorDataBundle(
        signal=signal,
        valuation=valuation,
        cb_daily=cb_daily,
        stock_daily=stock_daily,
        lifecycle=lifecycle,
    )
    paths = {
        "factor_context": factor_context_path,
        "market": market_path,
        "lifecycle": lifecycle_path,
        "valuation": valuation_path,
        "cb_daily": cb_daily_path,
        "stock_daily": stock_daily_path,
    }
    fingerprints = {
        name: _sha256_file(path)
        for name, path in paths.items()
        if path is not None
    }
    trading_dates = pd.DatetimeIndex(
        pd.to_datetime(market["trade_date"], errors="raise").unique()
    ).sort_values()
    return LoadedResearchInputs(
        factor_data=factor_data,
        market=market,
        lifecycle=lifecycle,
        trading_dates=trading_dates,
        universe_fingerprint=_fingerprint_signal_keys(signal),
        input_fingerprints=fingerprints,
    )


def slice_research_inputs(
    inputs: LoadedResearchInputs,
    sample_partition: str,
) -> LoadedResearchInputs:
    """Enforce one chronological sample boundary while preserving lookback history."""
    try:
        start_text, end_text = SAMPLE_PARTITION_BOUNDS[sample_partition]
    except KeyError as error:
        raise ValueError(f"unknown sample partition: {sample_partition}") from error
    start = pd.Timestamp(start_text).normalize()
    market_dates = pd.to_datetime(inputs.market["trade_date"], errors="raise").dt.normalize()
    end = (
        pd.Timestamp(end_text).normalize()
        if end_text is not None
        else market_dates.max()
    )

    signal = inputs.factor_data.signal
    signal_dates = pd.to_datetime(signal["signal_date"], errors="raise").dt.normalize()
    signal = signal.loc[signal_dates.between(start, end)].copy()
    if signal.empty:
        raise ValueError(f"sample partition {sample_partition} contains no signal rows")

    valuation = inputs.factor_data.valuation
    if not valuation.empty:
        valuation_dates = pd.to_datetime(
            valuation["signal_date"], errors="raise"
        ).dt.normalize()
        valuation = valuation.loc[valuation_dates.between(start, end)].copy()

    cb_daily = inputs.factor_data.cb_daily
    if not cb_daily.empty:
        cb_dates = pd.to_datetime(cb_daily["trade_date"], errors="raise").dt.normalize()
        cb_daily = cb_daily.loc[cb_dates.le(end)].copy()
    stock_daily = inputs.factor_data.stock_daily
    if not stock_daily.empty:
        stock_dates = pd.to_datetime(
            stock_daily["trade_date"], errors="raise"
        ).dt.normalize()
        stock_daily = stock_daily.loc[stock_dates.le(end)].copy()

    market = inputs.market.loc[market_dates.between(start, end)].copy()
    lifecycle_dates = pd.to_datetime(
        inputs.lifecycle["trade_date"], errors="raise"
    ).dt.normalize()
    lifecycle = inputs.lifecycle.loc[lifecycle_dates.between(start, end)].copy()
    factor_data = FactorDataBundle(
        signal=signal,
        valuation=valuation,
        cb_daily=cb_daily,
        stock_daily=stock_daily,
        lifecycle=lifecycle,
    )
    trading_dates = pd.DatetimeIndex(
        pd.to_datetime(market["trade_date"], errors="raise").unique()
    ).sort_values()
    return LoadedResearchInputs(
        factor_data=factor_data,
        market=market,
        lifecycle=lifecycle,
        trading_dates=trading_dates,
        universe_fingerprint=_fingerprint_signal_keys(signal),
        input_fingerprints=inputs.input_fingerprints,
    )


def run_factor_experiment(
    *,
    factor_name: str,
    factor_parameters: Mapping[str, object],
    sample_partition: str,
    inputs: LoadedResearchInputs,
    backtest_config: Mapping[str, object],
    result_root: Path,
) -> dict[str, object]:
    """Run one factor and write or reuse its immutable evidence package."""
    inputs = slice_research_inputs(inputs, sample_partition)
    registry = build_default_factor_registry()
    spec = registry.get_spec(factor_name)
    parameters = dict(spec.default_parameters)
    parameters.update(dict(factor_parameters))
    preprocessing = backtest_config["preprocessing"]
    evaluation = backtest_config["evaluation"]
    portfolio = backtest_config["portfolio"]
    execution = backtest_config["execution"]
    timing = backtest_config["timing"]
    identity = build_run_identity(
        factor_spec=spec,
        factor_parameters=parameters,
        sample_partition=sample_partition,
        universe_fingerprint=inputs.universe_fingerprint,
        preprocessing=preprocessing,
        label={
            "primary_label": timing["primary_label"],
            "signal_time": timing["signal_time"],
            "execution_time": timing["execution_time"],
            "evaluation": evaluation,
        },
        rebalance={"signal_dates_fingerprint": inputs.universe_fingerprint},
        portfolio=portfolio,
        execution=execution,
        input_fingerprints=inputs.input_fingerprints,
    )
    run_dir = resolve_factor_run_directory(
        result_root,
        spec,
        parameters,
        sample_partition,
        identity.run_id,
    )
    if run_dir.exists():
        return {
            "factor": factor_name,
            "parameters": parameters,
            "run_id": identity.run_id,
            "run_dir": str(run_dir),
            "created": False,
        }
    result = run_single_factor_research(
        factor_name=factor_name,
        factor_context=inputs.factor_data,
        registry=registry,
        trading_dates=inputs.trading_dates,
        open_prices=inputs.market[["trade_date", "ts_code", "open"]],
        market=inputs.market,
        lifecycle=inputs.lifecycle,
        factor_parameters=parameters,
        n_mad=preprocessing["mad_multiplier"],
        numeric_controls=preprocessing["numeric_controls"],
        categorical_controls=preprocessing["categorical_controls"],
        group_count=evaluation["quantile_groups"],
        min_ic_assets=evaluation["minimum_ic_assets"],
        periods_per_year=evaluation["periods_per_year"],
        nw_lags=evaluation["newey_west_lags"],
        portfolio_size=portfolio["portfolio_size"],
        cash_reserve=portfolio["cash_reserve"],
        max_weight=portfolio["maximum_single_name_weight"],
        initial_cash=portfolio["initial_cash_yuan"],
        cost_rate=execution["one_way_cost_bps"] / 10_000.0,
        board_lot=execution["board_lot"],
        allow_missing_lifecycle=False,
    )
    write_factor_run_record(
        output_root=result_root,
        identity=identity,
        factor_spec=spec,
        factor_parameters=parameters,
        sample_partition=sample_partition,
        result=result,
    )
    return {
        "factor": factor_name,
        "parameters": parameters,
        "run_id": identity.run_id,
        "run_dir": str(run_dir),
        "created": True,
    }


def _write_result_tables(result, output_dir: Path, *, overwrite: bool) -> list[Path]:
    tables = {
        "raw_factor": result.raw_factor,
        "processed_factor": result.processed_factor,
        "factor_labels": result.factor_labels,
        "label_diagnostics": result.label_diagnostics,
        "ic_series": result.ic_series,
        "ic_summary": result.ic_summary,
        "grouped_factor": result.grouped_factor,
        "quantile_returns": result.quantile_returns,
        "quantile_nav": result.quantile_nav,
        "targets": result.targets,
        **{
            f"execution_{name}": table
            for name, table in asdict(result.execution).items()
        },
    }
    paths = [output_dir / f"{name}.parquet" for name in tables]
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing outputs: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, table in zip(paths, tables.values(), strict=True):
        table.to_parquet(path, index=False)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor-context", type=Path, required=True)
    parser.add_argument("--market", type=Path, required=True)
    parser.add_argument("--lifecycle", type=Path, required=True)
    parser.add_argument("--valuation", type=Path)
    parser.add_argument("--cb-daily", type=Path)
    parser.add_argument("--stock-daily", type=Path)
    parser.add_argument("--factor", default="conversion_premium")
    parser.add_argument("--factor-parameters", default="{}")
    parser.add_argument(
        "--sample-partition",
        required=True,
        choices=(
            "replication_2018_2023",
            "validation_2023_2024",
            "holdout_2025_onward",
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    args = parser.parse_args()

    config = load_backtest_config(args.config)
    parameters = json.loads(args.factor_parameters)
    if not isinstance(parameters, dict):
        raise ValueError("--factor-parameters must decode to a JSON object")
    inputs = load_research_inputs(
        factor_context_path=args.factor_context,
        market_path=args.market,
        lifecycle_path=args.lifecycle,
        valuation_path=args.valuation,
        cb_daily_path=args.cb_daily,
        stock_daily_path=args.stock_daily,
    )
    record = run_factor_experiment(
        factor_name=args.factor,
        factor_parameters=parameters,
        sample_partition=args.sample_partition,
        inputs=inputs,
        backtest_config=config,
        result_root=args.result_root,
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
