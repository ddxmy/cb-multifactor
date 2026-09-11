"""Materialize leakage-free 1/5/10-market-day O2O labels for G4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cb_quant.config import (
    SAMPLE_PARTITION_BOUNDS,
    SAMPLE_PARTITIONS,
    load_backtest_config,
)
from cb_quant.data_catalog import DataCatalog
from cb_quant.g4_inputs import (
    build_horizon_schedules,
    build_multi_horizon_forward_return_panel,
    enforce_partition_access,
)
from cb_quant.universe import load_open_trading_dates


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-partition",
        choices=sorted(SAMPLE_PARTITIONS),
        default="replication_2018_2023",
    )
    parser.add_argument(
        "--backtest-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "backtest_v1.json",
    )
    parser.add_argument(
        "--g1-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "G1",
    )
    parser.add_argument(
        "--g4-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "G4",
    )
    parser.add_argument("--unlock-holdout", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _calendar_fingerprint(trading_dates: pd.DatetimeIndex) -> str:
    payload = "\n".join(trading_dates.strftime("%Y-%m-%d")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _assert_outputs_absent(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing G4 horizon artifacts: "
            + ", ".join(str(path) for path in existing)
        )


def main() -> int:
    args = parse_args()
    enforce_partition_access(
        args.sample_partition,
        unlock_holdout=args.unlock_holdout,
    )
    start_text, end_text = SAMPLE_PARTITION_BOUNDS[args.sample_partition]
    start = pd.Timestamp(start_text).normalize()
    end = pd.Timestamp(end_text).normalize() if end_text is not None else None
    if end is None:
        raise ValueError("an open-ended partition requires an explicit data-end policy")

    config = load_backtest_config(args.backtest_config)
    horizons = config["evaluation"]["forward_return_horizons_market_days"]
    partition_root = args.g4_root / args.sample_partition
    factor_context_path = partition_root / "factor_context_v1.parquet"
    execution_market_path = partition_root / "execution_market_v1.parquet"
    rebalance_path = args.g1_root / "rebalance_calendar_v1.parquet"
    required_inputs = [factor_context_path, execution_market_path, rebalance_path]
    missing = [path for path in required_inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing prerequisite artifacts: " + ", ".join(str(path) for path in missing)
        )

    catalog = DataCatalog.from_environment()
    calendar_path = (
        catalog.tushare_raw_root / "trade_cal_sse_open_20180101_20260723.csv"
    )
    trading_dates = load_open_trading_dates(
        calendar_path,
        historical_calendar_path=catalog.ci_l1_daily_path,
    )
    rebalance_calendar = pd.read_parquet(rebalance_path)
    factor_context = pd.read_parquet(
        factor_context_path,
        columns=["signal_date", "ts_code"],
    )
    execution_market = pd.read_parquet(execution_market_path)

    schedule = build_horizon_schedules(
        rebalance_calendar,
        trading_dates=trading_dates,
        horizons=horizons,
        sample_partition=args.sample_partition,
        start=start,
        end=end,
    )
    labels = build_multi_horizon_forward_return_panel(
        factor_context,
        schedule,
        execution_market,
    )

    expected_horizons = set(horizons)
    if set(schedule["horizon_days"].unique()) != expected_horizons:
        raise ValueError("schedule horizons do not match the frozen configuration")
    if set(labels["horizon_days"].unique()) != expected_horizons:
        raise ValueError("label horizons do not match the frozen configuration")
    if len(labels) != len(factor_context) * len(horizons):
        raise ValueError("multi-horizon label rows do not match signal rows times horizons")
    if labels.duplicated(["signal_date", "ts_code", "horizon_days"]).any():
        raise ValueError("multi-horizon labels contain duplicate keys")
    usable = labels.loc[labels["label_usable"]]
    if not np.isfinite(usable["forward_return"]).all():
        raise ValueError("usable multi-horizon returns contain non-finite values")
    if usable["label_end_date"].gt(end).any():
        raise ValueError("usable multi-horizon labels cross the sample boundary")

    outputs = {
        "schedule": partition_root / "execution_horizon_schedule_v1.parquet",
        "labels": partition_root / "o2o_forward_returns_horizons_v1.parquet",
        "audit": partition_root / "g4_horizon_label_audit_v1.json",
    }
    _assert_outputs_absent(list(outputs.values()))
    partition_root.mkdir(parents=True, exist_ok=True)
    schedule.to_parquet(outputs["schedule"], index=False)
    labels.to_parquet(outputs["labels"], index=False)

    horizon_summary: dict[str, dict[str, object]] = {}
    for horizon, group in labels.groupby("horizon_days", sort=True):
        horizon_summary[str(int(horizon))] = {
            "label_rows": int(len(group)),
            "usable_label_rows": int(group["label_usable"].sum()),
            "exclusion_reason_counts": {
                str(reason): int(count)
                for reason, count in group["label_exclusion_reason"]
                .value_counts()
                .items()
            },
        }
    audit = {
        "status": "pass",
        "sample_partition": args.sample_partition,
        "sample_start": str(start.date()),
        "sample_end": str(end.date()),
        "holdout_unlocked": bool(args.unlock_holdout),
        "horizons_market_days": horizons,
        "signal_rows": int(len(factor_context)),
        "signal_dates": int(factor_context["signal_date"].nunique()),
        "schedule_rows": int(len(schedule)),
        "label_rows": int(len(labels)),
        "usable_label_rows": int(labels["label_usable"].sum()),
        "horizon_summary": horizon_summary,
        "source_fingerprints": {
            "backtest_config": _sha256(args.backtest_config),
            "rebalance_calendar": _sha256(rebalance_path),
            "factor_context": _sha256(factor_context_path),
            "execution_market": _sha256(execution_market_path),
            "normalized_trading_calendar": _calendar_fingerprint(trading_dates),
        },
        "artifact_paths": {name: str(path) for name, path in outputs.items()},
    }
    outputs["audit"].write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
