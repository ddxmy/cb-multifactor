"""Materialize G4 execution inputs and leakage-free O2O labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from cb_quant.config import SAMPLE_PARTITION_BOUNDS, SAMPLE_PARTITIONS
from cb_quant.data_catalog import DataCatalog
from cb_quant.g4_inputs import (
    build_execution_market,
    build_forward_return_panel,
    build_partition_schedule,
    enforce_partition_access,
    prepare_lifecycle_terms,
    prepare_redemption_events,
)
from cb_quant.lifecycle import build_lifecycle_panel
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
        "--research-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "research_v1.json",
    )
    parser.add_argument(
        "--g1-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "G1",
    )
    parser.add_argument(
        "--factor-context-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "G3C" / "preprocessed_factor_panel_v1",
    )
    parser.add_argument(
        "--output-root",
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


def _load_factor_context(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    paths = [
        root / f"year={year}" / "part-0000.parquet"
        for year in range(start.year, end.year + 1)
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing factor-context partitions: " + ", ".join(str(path) for path in missing)
        )
    result = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    result["signal_date"] = pd.to_datetime(result["signal_date"], errors="raise").dt.normalize()
    result = result.loc[result["signal_date"].between(start, end)].copy()
    if "is_eligible" in result:
        result = result.loc[result["is_eligible"].fillna(False)].copy()
    if result.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError("factor context contains duplicate signal-date and bond keys")
    return result.sort_values(["signal_date", "ts_code"]).reset_index(drop=True)


def _load_terms(path: Path) -> pd.DataFrame:
    with duckdb.connect(str(path), read_only=True) as connection:
        raw = connection.execute(
            """
            SELECT
                ts_code,
                list_date,
                delist_date,
                maturity_date,
                maturity_call_price
            FROM default_table
            WHERE cb_type = 'CB'
              AND ts_code IS NOT NULL
              AND list_date IS NOT NULL
              AND maturity_date IS NOT NULL
            """
        ).df()
    return prepare_lifecycle_terms(raw)


def _load_redemption_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"redemption database does not exist: {path}")
    with duckdb.connect(str(path), read_only=True) as connection:
        raw = connection.execute("SELECT * FROM default_table").df()
    return prepare_redemption_events(raw)


def _assert_outputs_absent(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing G4 artifacts: "
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
    research_config = json.loads(args.research_config.read_text(encoding="utf-8"))
    seasoning_days = int(
        research_config["universe"]["minimum_listing_age_market_days"]
    )

    catalog = DataCatalog.from_environment()
    raw_daily = pd.read_csv(catalog.cb_daily_path, low_memory=False)
    raw_daily["trade_date"] = pd.to_datetime(
        raw_daily["trade_date"].astype("string"), format="%Y%m%d", errors="coerce"
    ).dt.normalize()
    observed_end = raw_daily["trade_date"].max()
    end = pd.Timestamp(end_text).normalize() if end_text is not None else observed_end
    if pd.isna(observed_end) or end > observed_end:
        raise ValueError("sample end exceeds the available CB daily market history")

    calendar_path = catalog.tushare_raw_root / "trade_cal_sse_open_20180101_20260723.csv"
    trading_dates = load_open_trading_dates(
        calendar_path,
        historical_calendar_path=catalog.ci_l1_daily_path,
    )
    trading_dates = trading_dates[trading_dates.to_series().between(start, end).to_numpy()]
    if trading_dates.empty:
        raise ValueError("sample partition contains no trading dates")

    rebalance_path = args.g1_root / "rebalance_calendar_v1.parquet"
    rebalance_calendar = pd.read_parquet(rebalance_path)
    schedule = build_partition_schedule(
        rebalance_calendar,
        sample_partition=args.sample_partition,
        start=start,
        end=end,
    )
    factor_context = _load_factor_context(args.factor_context_root, start, end)

    terms = _load_terms(catalog.cb_terms_path)
    overlaps_sample = terms["list_date"].le(end) & (
        terms["delist_date"].isna() | terms["delist_date"].ge(start)
    )
    terms = terms.loc[overlaps_sample].copy()
    missing_terms = set(factor_context["ts_code"].astype(str)).difference(
        set(terms["ts_code"].astype(str))
    )
    if missing_terms:
        raise ValueError(
            f"{len(missing_terms)} factor-context bonds are missing lifecycle terms"
        )

    raw_daily = raw_daily.loc[
        raw_daily["trade_date"].between(start, end)
        & raw_daily["ts_code"].astype(str).isin(terms["ts_code"].astype(str))
    ].copy()
    market = build_execution_market(
        raw_daily,
        trading_dates=trading_dates,
        bond_codes=terms["ts_code"],
    )
    redemption_events = _load_redemption_events(catalog.redemption_path)
    redemption_events = redemption_events.loc[
        redemption_events["effective_date"].le(end)
        & redemption_events["ts_code"].isin(terms["ts_code"])
    ].copy()
    lifecycle = build_lifecycle_panel(
        terms,
        trading_dates,
        redemption_events,
        tradability=market[["trade_date", "ts_code", "is_tradable"]],
        seasoning_days=seasoning_days,
    )
    forward_returns = build_forward_return_panel(
        factor_context[["signal_date", "ts_code"]],
        schedule,
        market,
    )

    if len(market) != len(lifecycle) or not market[
        ["trade_date", "ts_code"]
    ].equals(lifecycle[["trade_date", "ts_code"]]):
        raise ValueError("execution market and lifecycle keys are not identical")
    if len(forward_returns) != len(factor_context):
        raise ValueError("forward-return rows do not match factor-context rows")
    if forward_returns.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError("forward-return panel contains duplicate keys")
    usable = forward_returns.loc[forward_returns["label_usable"]]
    if not np.isfinite(usable["forward_return"]).all():
        raise ValueError("usable forward returns contain non-finite values")
    if usable["label_end_date"].gt(end).any():
        raise ValueError("usable labels cross the sample boundary")

    partition_root = args.output_root / args.sample_partition
    outputs = {
        "factor_context": partition_root / "factor_context_v1.parquet",
        "execution_schedule": partition_root / "execution_schedule_v1.parquet",
        "execution_market": partition_root / "execution_market_v1.parquet",
        "lifecycle": partition_root / "lifecycle_panel_v1.parquet",
        "forward_returns": partition_root / "o2o_forward_returns_v1.parquet",
        "audit": partition_root / "g4_audit_v1.json",
    }
    _assert_outputs_absent(list(outputs.values()))
    partition_root.mkdir(parents=True, exist_ok=True)
    factor_context.to_parquet(outputs["factor_context"], index=False)
    schedule.to_parquet(outputs["execution_schedule"], index=False)
    market.to_parquet(outputs["execution_market"], index=False)
    lifecycle.to_parquet(outputs["lifecycle"], index=False)
    forward_returns.to_parquet(outputs["forward_returns"], index=False)

    reason_counts = {
        str(reason): int(count)
        for reason, count in forward_returns["label_exclusion_reason"].value_counts().items()
    }
    audit = {
        "status": "pass",
        "sample_partition": args.sample_partition,
        "sample_start": str(start.date()),
        "sample_end": str(end.date()),
        "holdout_unlocked": bool(args.unlock_holdout),
        "signal_rows": int(len(factor_context)),
        "signal_dates": int(factor_context["signal_date"].nunique()),
        "bond_count": int(terms["ts_code"].nunique()),
        "seasoning_market_days": seasoning_days,
        "market_rows": int(len(market)),
        "lifecycle_rows": int(len(lifecycle)),
        "label_rows": int(len(forward_returns)),
        "usable_label_rows": int(forward_returns["label_usable"].sum()),
        "label_exclusion_reason_counts": reason_counts,
        "one_price_up_rows": int(market["is_one_price_up"].sum()),
        "one_price_down_rows": int(market["is_one_price_down"].sum()),
        "missing_market_grid_rows": int((~market["has_market_observation"]).sum()),
        "redemption_event_rows": int(len(redemption_events)),
        "lifecycle_state_counts": {
            str(state): int(count)
            for state, count in lifecycle["lifecycle_state"].value_counts().items()
        },
        "source_fingerprints": {
            "research_config": _sha256(args.research_config),
            "rebalance_calendar": _sha256(rebalance_path),
            "cb_daily": _sha256(catalog.cb_daily_path),
            "cb_terms": _sha256(catalog.cb_terms_path),
            "redemption_events": _sha256(catalog.redemption_path),
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
