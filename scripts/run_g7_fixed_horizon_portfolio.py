"""Run the G7 cash-constrained fixed-horizon factor portfolio."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from cb_quant.config import load_backtest_config
from cb_quant.factors import build_default_factor_registry, ensure_factor_data_bundle
from cb_quant.portfolio import run_fixed_horizon_factor_portfolio
from cb_quant.preprocessing import preprocess_registered_factor


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", default="double_low")
    parser.add_argument(
        "--sample-partition",
        choices=("replication_2018_2023",),
        default="replication_2018_2023",
    )
    parser.add_argument("--holding-days", type=int, default=5)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "backtest_v1.json",
    )
    parser.add_argument(
        "--g4-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "G4",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "G7" / "fixed_horizon_portfolio",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _run_id(payload: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def _materialize_scores(
    context: pd.DataFrame,
    *,
    factor_name: str,
    mad_multiplier: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    registry = build_default_factor_registry()
    spec = registry.get_spec(factor_name)
    data = ensure_factor_data_bundle(context)
    raw = registry.calculate(factor_name, data)
    processed = preprocess_registered_factor(
        data,
        raw,
        factor_name=factor_name,
        direction=spec.direction,
        n_mad=mad_multiplier,
    )
    eligibility = context[["signal_date", "ts_code", "is_eligible"]].copy()
    scores = processed.merge(
        eligibility,
        on=["signal_date", "ts_code"],
        how="inner",
        validate="one_to_one",
    )
    metadata = {
        "name": spec.name,
        "version": spec.version,
        "direction": spec.direction,
        "raw_formula_field": factor_name,
        "ranking_field": "processed_factor",
        "ranking_order": "descending_after_direction_alignment",
        "preprocessing_source": (
            "materialized_g3c"
            if f"standardized__{factor_name}" in context
            else "return_blind_mad_zscore_fallback"
        ),
    }
    return scores, metadata


def _write_charts(nav: pd.DataFrame, output_dir: Path, factor_name: str) -> None:
    import matplotlib.pyplot as plt

    ordered = nav.sort_values("trade_date").copy()
    ordered["net_nav"] = ordered["nav"] / ordered["nav"].iloc[0]
    wealth = pd.concat(
        [pd.Series([1.0]), ordered["net_nav"]], ignore_index=True
    )
    ordered["drawdown"] = (wealth / wealth.cummax() - 1.0).iloc[1:].to_numpy()
    ordered["cash_weight"] = ordered["cash"] / ordered["nav"]
    ordered["gross_exposure"] = ordered["holdings_value"] / ordered["nav"]

    display_name = factor_name.replace("_", " ").title()
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(ordered["trade_date"], ordered["net_nav"], linewidth=1.3)
    axis.set_title(f"{display_name}: Net Executable Portfolio NAV")
    axis.set_xlabel("Trade Date")
    axis.set_ylabel("NAV (First Execution Day = 1)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "net_nav.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 4.5))
    axis.fill_between(
        ordered["trade_date"], ordered["drawdown"], 0.0, alpha=0.65
    )
    axis.set_title(f"{display_name}: Drawdown")
    axis.set_xlabel("Trade Date")
    axis.set_ylabel("Drawdown")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "drawdown.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 4.5))
    axis.plot(ordered["trade_date"], ordered["gross_exposure"], label="Gross exposure")
    axis.plot(ordered["trade_date"], ordered["cash_weight"], label="Cash weight")
    axis.set_title(f"{display_name}: Exposure and Cash")
    axis.set_xlabel("Trade Date")
    axis.set_ylabel("Portfolio Weight")
    axis.legend(loc="upper right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "exposure_and_cash.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.holding_days != 5:
        raise ValueError("this frozen G7 run currently supports holding_days=5 only")

    config = load_backtest_config(args.config)
    partition_root = args.g4_root / args.sample_partition
    paths = {
        "factor_context": partition_root / "factor_context_v1.parquet",
        "horizon_schedule": partition_root / "execution_horizon_schedule_v1.parquet",
        "execution_market": partition_root / "execution_market_v1.parquet",
        "lifecycle_panel": partition_root / "lifecycle_panel_v1.parquet",
    }
    if missing := [path for path in paths.values() if not path.exists()]:
        raise FileNotFoundError("missing G4 inputs: " + ", ".join(map(str, missing)))

    context = pd.read_parquet(paths["factor_context"])
    schedule_all = pd.read_parquet(paths["horizon_schedule"])
    market = pd.read_parquet(paths["execution_market"])
    lifecycle = pd.read_parquet(paths["lifecycle_panel"])
    schedule = schedule_all.loc[
        schedule_all["horizon_days"].eq(args.holding_days)
        & schedule_all["is_label_within_partition"].fillna(False)
    ].copy()
    scores, factor_metadata = _materialize_scores(
        context,
        factor_name=args.factor,
        mad_multiplier=float(config["preprocessing"]["mad_multiplier"]),
    )

    portfolio_config = config["portfolio"]
    execution_config = config["execution"]
    payload = {
        "schema_version": "1.0",
        "stage": "G7",
        "factor": factor_metadata,
        "sample_partition": args.sample_partition,
        "holding_days_market_open_to_open": args.holding_days,
        "preprocessing": config["preprocessing"],
        "portfolio": portfolio_config,
        "execution": execution_config,
        "overlap_policy": (
            "exit_before_entry; active cohorts reserve gross target capacity; "
            "new cohorts receive only remaining capacity"
        ),
        "inputs": {name: _sha256(path) for name, path in paths.items()},
        "implementation": {
            "runner": _sha256(Path(__file__)),
            "cohort_targets": _sha256(
                PROJECT_ROOT / "src" / "cb_quant" / "portfolio" / "cohorts.py"
            ),
            "execution_engine": _sha256(
                PROJECT_ROOT / "src" / "cb_quant" / "execution" / "engine.py"
            ),
        },
    }
    run_id = _run_id(payload)
    run_dir = args.output_root / args.factor / args.sample_partition / run_id
    if run_dir.exists():
        print((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        print(pd.read_csv(run_dir / "summary_metrics.csv").to_string(index=False))
        return 0

    run = run_fixed_horizon_factor_portfolio(
        scores,
        schedule,
        market,
        lifecycle,
        portfolio_size=int(portfolio_config["portfolio_size"]),
        cash_reserve=float(portfolio_config["cash_reserve"]),
        max_weight=float(portfolio_config["maximum_single_name_weight"]),
        initial_cash=float(portfolio_config["initial_cash_yuan"]),
        cost_rate=float(execution_config["one_way_cost_bps"]) / 10_000.0,
        board_lot=int(execution_config["board_lot"]),
    )
    cohort_counts = run.plan.cohort_audit["status"].value_counts().to_dict()
    summary = {
        **run.summary,
        "factor": args.factor,
        "holding_days": args.holding_days,
        "signal_cohorts": int(len(run.plan.cohort_audit)),
        "allocated_cohorts": int(cohort_counts.get("allocated", 0)),
        "partially_allocated_cohorts": int(cohort_counts.get("partially_allocated", 0)),
        "skipped_no_capacity_cohorts": int(cohort_counts.get("skipped_no_capacity", 0)),
        "skipped_no_candidate_cohorts": int(cohort_counts.get("skipped_no_candidates", 0)),
        "selected_bond_signal_rows": int(run.plan.selections["is_selected"].sum()),
    }
    manifest = {
        **payload,
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "pass",
        "result_scope": "G7 cash-constrained executable portfolio",
    }

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".g7_portfolio_", dir=run_dir.parent) as temp:
        destination = Path(temp)
        run.plan.selections.to_parquet(destination / "selections.parquet", index=False)
        run.plan.cohort_audit.to_csv(destination / "cohort_audit.csv", index=False)
        run.plan.targets.to_parquet(destination / "targets.parquet", index=False)
        run.execution.nav.to_parquet(destination / "nav.parquet", index=False)
        run.execution.positions.to_parquet(destination / "positions.parquet", index=False)
        run.execution.orders.to_parquet(destination / "orders.parquet", index=False)
        run.execution.fills.to_parquet(destination / "fills.parquet", index=False)
        run.execution.receivables.to_parquet(destination / "receivables.parquet", index=False)
        run.execution.attribution.to_parquet(destination / "attribution.parquet", index=False)
        pd.DataFrame([summary]).to_csv(destination / "summary_metrics.csv", index=False)
        (destination / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        _write_charts(run.execution.nav, destination, args.factor)
        os.replace(destination, run_dir)

    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    print(pd.DataFrame([summary]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
