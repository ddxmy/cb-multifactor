"""Run an explicitly configured batch of comparable factor experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from pathlib import Path

from cb_quant.config import load_backtest_config, load_factor_batch_config
from cb_quant.factors import FactorRegistry, build_default_factor_registry
from cb_quant.reporting import build_factor_run_index
from scripts.run_single_factor import (
    DEFAULT_CONFIG,
    DEFAULT_RESULT_ROOT,
    load_research_inputs,
    run_factor_experiment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_CONFIG = PROJECT_ROOT / "config" / "factor_batch_v1.json"


def validate_batch_catalog(
    batch_config: Mapping[str, object],
    registry: FactorRegistry,
) -> None:
    """Reject unknown factors before any experiment in the batch starts."""
    unknown = [
        item["name"]
        for item in batch_config["factors"]
        if item["name"] not in registry.names()
    ]
    if unknown:
        raise ValueError(f"batch contains unknown factors: {sorted(set(unknown))}")


def execute_factor_batch(
    batch_config: Mapping[str, object],
    *,
    run_one: Callable[[str, Mapping[str, object], str], Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """Execute declared factor variants in stable configuration order."""
    sample_partition = str(batch_config["sample_partition"])
    return [
        run_one(item["name"], item["parameters"], sample_partition)
        for item in batch_config["factors"]
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor-context", type=Path, required=True)
    parser.add_argument("--market", type=Path, required=True)
    parser.add_argument("--lifecycle", type=Path, required=True)
    parser.add_argument("--valuation", type=Path)
    parser.add_argument("--cb-daily", type=Path)
    parser.add_argument("--stock-daily", type=Path)
    parser.add_argument("--batch-config", type=Path, default=DEFAULT_BATCH_CONFIG)
    parser.add_argument("--backtest-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    args = parser.parse_args()

    batch_config = load_factor_batch_config(args.batch_config)
    backtest_config = load_backtest_config(args.backtest_config)
    registry = build_default_factor_registry()
    validate_batch_catalog(batch_config, registry)
    inputs = load_research_inputs(
        factor_context_path=args.factor_context,
        market_path=args.market,
        lifecycle_path=args.lifecycle,
        valuation_path=args.valuation,
        cb_daily_path=args.cb_daily,
        stock_daily_path=args.stock_daily,
    )

    def run_one(name, parameters, sample_partition):
        return run_factor_experiment(
            factor_name=name,
            factor_parameters=parameters,
            sample_partition=sample_partition,
            inputs=inputs,
            backtest_config=backtest_config,
            result_root=args.result_root,
        )

    runs = execute_factor_batch(batch_config, run_one=run_one)
    index_path = args.result_root.parent / "factor_run_index_v1.parquet"
    index = build_factor_run_index(args.result_root, output_path=index_path)
    output = {
        "batch_id": batch_config["batch_id"],
        "sample_partition": batch_config["sample_partition"],
        "runs": runs,
        "factor_run_index": str(index_path),
        "indexed_run_count": len(index),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
