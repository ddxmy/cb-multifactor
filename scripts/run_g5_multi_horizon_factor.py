"""Run immutable G5 multi-horizon evidence for one registered factor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cb_quant.config import load_backtest_config
from cb_quant.evaluation import run_multi_horizon_factor_evaluation
from cb_quant.factors import build_default_factor_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", default="double_low")
    parser.add_argument(
        "--sample-partition",
        choices=("replication_2018_2023",),
        default="replication_2018_2023",
    )
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
        default=PROJECT_ROOT / "artifacts" / "G5" / "multi_horizon_factor_runs",
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


def _verify_materialized_factor(
    context: pd.DataFrame,
    factor_name: str,
    raw_factor: pd.DataFrame,
    processed_factor: pd.DataFrame,
    direction: int,
) -> dict[str, int]:
    keys = ["signal_date", "ts_code"]
    checks: dict[str, int] = {}
    if factor_name in context:
        raw_expected = context[keys + [factor_name]].merge(
            raw_factor[keys + ["raw_factor"]],
            on=keys,
            how="outer",
            validate="one_to_one",
        )
        valid = raw_expected[[factor_name, "raw_factor"]].notna().all(axis=1)
        checks["raw_factor_mismatches"] = int(
            (~np.isclose(
                raw_expected.loc[valid, factor_name],
                raw_expected.loc[valid, "raw_factor"],
                rtol=1e-10,
                atol=1e-10,
            )).sum()
        )
    standardized_column = f"standardized__{factor_name}"
    if standardized_column in context:
        expected = context[keys + [standardized_column]].merge(
            processed_factor[keys + ["processed_factor"]],
            on=keys,
            how="outer",
            validate="one_to_one",
        )
        expected["expected_processed"] = direction * expected[standardized_column]
        valid = expected[["expected_processed", "processed_factor"]].notna().all(axis=1)
        checks["processed_factor_mismatches"] = int(
            (~np.isclose(
                expected.loc[valid, "expected_processed"],
                expected.loc[valid, "processed_factor"],
                rtol=1e-10,
                atol=1e-10,
            )).sum()
        )
    if any(checks.values()):
        raise ValueError(f"factor values do not match materialized G3c evidence: {checks}")
    return checks


def _summary_metrics(result) -> pd.DataFrame:
    ic = result.ic_summary.pivot(
        index="horizon_days",
        columns="metric",
        values=["mean", "positive_rate", "icir_annualized", "nw_t_stat", "nw_p_value"],
    )
    ic.columns = [f"{metric}_{stat}" for stat, metric in ic.columns]
    summary = result.ir_summary.set_index("horizon_days").join(ic)
    labels = result.factor_labels.groupby("horizon_days").agg(
        label_rows=("ts_code", "size"),
        usable_label_rows=("label_usable", "sum"),
    )
    labels["label_exclusion_rate"] = 1.0 - (
        labels["usable_label_rows"] / labels["label_rows"]
    )
    summary = summary.join(labels)
    final_nav = (
        result.analytical_nav.sort_values("signal_date")
        .groupby("horizon_days")
        .tail(1)
        .set_index("horizon_days")
        [["g1_nav", "universe_nav", "active_nav", "spread_nav"]]
    )
    nav_metrics: list[dict[str, float | int]] = []
    for horizon, group in result.analytical_nav.groupby("horizon_days", sort=True):
        ordered = group.sort_values("signal_date")
        observations = len(ordered)
        record: dict[str, float | int] = {"horizon_days": int(horizon)}
        for prefix, column in (
            ("g1", "g1_nav"),
            ("universe", "universe_nav"),
            ("active", "active_nav"),
            ("spread", "spread_nav"),
        ):
            nav = pd.to_numeric(ordered[column], errors="coerce").dropna()
            final_value = float(nav.iloc[-1])
            record[f"{prefix}_annualized_return"] = (
                final_value ** (26.0 / observations) - 1.0
            )
            running_peak = nav.cummax().clip(lower=1.0)
            record[f"{prefix}_max_drawdown"] = float(
                (nav / running_peak - 1.0).min()
            )
        nav_metrics.append(record)
    nav_metric_frame = pd.DataFrame(nav_metrics).set_index("horizon_days")
    return summary.join(final_nav).join(nav_metric_frame).reset_index()


def _write_charts(result, output_dir: Path, factor_name: str) -> None:
    import matplotlib.pyplot as plt

    horizons = sorted(result.ic_series["horizon_days"].unique())
    figure, axes = plt.subplots(len(horizons), 1, figsize=(11, 8), sharex=True)
    for axis, horizon in zip(np.atleast_1d(axes), horizons, strict=True):
        group = result.ic_series.loc[result.ic_series["horizon_days"].eq(horizon)]
        axis.plot(group["signal_date"], group["ic"], label="IC", linewidth=1.0)
        axis.plot(group["signal_date"], group["rank_ic"], label="Rank IC", linewidth=1.0)
        axis.axhline(0.0, color="black", linewidth=0.6)
        axis.set_title(f"{horizon}-Market-Day Horizon")
        axis.legend(loc="upper right")
    display_name = factor_name.replace("_", " ").title()
    figure.suptitle(f"{display_name} Multi-Horizon IC History")
    figure.tight_layout()
    figure.savefig(output_dir / "ic_rank_ic_history.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(len(horizons), 1, figsize=(11, 8), sharex=True)
    for axis, horizon in zip(np.atleast_1d(axes), horizons, strict=True):
        group = result.analytical_nav.loc[
            result.analytical_nav["horizon_days"].eq(horizon)
        ]
        axis.plot(group["signal_date"], group["g1_nav"], label="G1")
        axis.plot(group["signal_date"], group["universe_nav"], label="Universe")
        axis.plot(group["signal_date"], group["active_nav"], label="G1 Active")
        axis.plot(group["signal_date"], group["spread_nav"], label="G1-G5")
        axis.set_title(f"{horizon}-Market-Day Horizon")
        axis.legend(loc="upper left", ncol=4)
    figure.suptitle(
        f"{display_name} Analytical NAV (No Costs or Position Constraints)"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "analytical_nav.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    config = load_backtest_config(args.config)
    evaluation = config["evaluation"]
    partition_root = args.g4_root / args.sample_partition
    factor_context_path = partition_root / "factor_context_v1.parquet"
    labels_path = partition_root / "o2o_forward_returns_horizons_v1.parquet"
    missing = [path for path in (factor_context_path, labels_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing G4 inputs: " + ", ".join(str(path) for path in missing))

    context = pd.read_parquet(factor_context_path)
    labels = pd.read_parquet(labels_path)
    expected_horizons = evaluation["forward_return_horizons_market_days"]
    if sorted(labels["horizon_days"].unique().tolist()) != expected_horizons:
        raise ValueError("G4 label horizons do not match the frozen backtest configuration")

    registry = build_default_factor_registry()
    spec = registry.get_spec(args.factor)
    payload = {
        "schema_version": "1.0",
        "factor": {"name": spec.name, "version": spec.version, "direction": spec.direction},
        "sample_partition": args.sample_partition,
        "preprocessing": config["preprocessing"],
        "evaluation": evaluation,
        "inputs": {
            "factor_context": _sha256(factor_context_path),
            "horizon_labels": _sha256(labels_path),
            "backtest_config": _sha256(args.config),
        },
        "implementation": {
            "runner": _sha256(Path(__file__)),
            "evaluation": _sha256(
                PROJECT_ROOT / "src" / "cb_quant" / "evaluation" / "horizon_run.py"
            ),
        },
    }
    run_id = _run_id(payload)
    run_dir = args.output_root / args.factor / args.sample_partition / run_id
    if run_dir.exists():
        print((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        print(pd.read_csv(run_dir / "summary_metrics.csv").to_string(index=False))
        return 0

    result = run_multi_horizon_factor_evaluation(
        factor_name=args.factor,
        factor_context=context,
        horizon_labels=labels,
        registry=registry,
        n_mad=float(config["preprocessing"]["mad_multiplier"]),
        group_count=int(evaluation["quantile_groups"]),
        min_ic_assets=int(evaluation["minimum_ic_assets"]),
        periods_per_year=int(evaluation["periods_per_year"]),
        nw_lags=int(evaluation["newey_west_lags"]),
    )
    factor_checks = _verify_materialized_factor(
        context,
        args.factor,
        result.raw_factor,
        result.processed_factor,
        spec.direction,
    )
    summary = _summary_metrics(result)
    exclusions = (
        result.factor_labels.groupby(["horizon_days", "label_exclusion_reason"])
        .size()
        .rename("rows")
        .reset_index()
    )
    manifest = {
        **payload,
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "pass",
        "factor_value_checks": factor_checks,
        "result_scope": "G5 analytical factor evidence; not a G7 implementable portfolio",
    }

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".g5_horizon_", dir=run_dir.parent) as temp:
        destination = Path(temp)
        result.raw_factor.to_parquet(destination / "raw_factor.parquet", index=False)
        result.processed_factor.to_parquet(destination / "processed_factor.parquet", index=False)
        result.factor_labels.to_parquet(destination / "factor_labels.parquet", index=False)
        result.ic_series.to_parquet(destination / "ic_series.parquet", index=False)
        result.ic_summary.to_csv(destination / "ic_summary.csv", index=False)
        result.ir_series.to_parquet(destination / "ir_series.parquet", index=False)
        result.ir_summary.to_csv(destination / "ir_summary.csv", index=False)
        result.analytical_nav.to_parquet(destination / "analytical_nav.parquet", index=False)
        summary.to_csv(destination / "summary_metrics.csv", index=False)
        exclusions.to_csv(destination / "label_exclusions.csv", index=False)
        (destination / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_charts(result, destination, args.factor)
        os.replace(destination, run_dir)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
