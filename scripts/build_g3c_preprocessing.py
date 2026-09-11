"""Build the return-blind G3c-A merged and MAD-winsorized factor panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from cb_quant import DataCatalog, load_a_share_market_cap
from cb_quant.preprocessing import (
    ALL_FACTOR_COLUMNS,
    STOCK_FACTOR_COLUMNS,
    VALUATION_FACTOR_COLUMNS,
    FactorPreprocessingPolicy,
    add_cb_neutralization_controls,
    attach_stock_market_cap_control,
    build_mad_sensitivity,
    merge_factor_panels,
    preprocess_factor_family,
    winsorize_factor_panel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "research_v1.json",
    )
    parser.add_argument(
        "--g2-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "G2",
    )
    parser.add_argument(
        "--g3-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "G3",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "G3C",
    )
    parser.add_argument("--start", help="Optional bounded-sample start signal date")
    parser.add_argument("--end", help="Optional bounded-sample end signal date")
    return parser.parse_args()


def _partition_files(root: Path, dataset_name: str) -> list[Path]:
    files = sorted((root / dataset_name).glob("year=*/part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No partitions found under {root / dataset_name}")
    return files


def load_partitioned_panel(
    root: Path,
    dataset_name: str,
    *,
    start: str | None = None,
    end: str | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    files = _partition_files(root, dataset_name)
    panel = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    panel["signal_date"] = pd.to_datetime(panel["signal_date"], errors="raise").dt.normalize()
    if start is not None:
        panel = panel.loc[panel["signal_date"].ge(pd.Timestamp(start).normalize())]
    if end is not None:
        panel = panel.loc[panel["signal_date"].le(pd.Timestamp(end).normalize())]
    if panel.empty:
        raise ValueError(f"Selected {dataset_name} panel is empty")
    return panel.reset_index(drop=True), files


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_manifest(files: list[Path]) -> dict[str, object]:
    records = [
        {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]
    combined = hashlib.sha256(
        json.dumps(records, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {"partitions": records, "combined_sha256": combined}


def _frame_manifest(frame: pd.DataFrame) -> dict[str, object]:
    normalized = frame.sort_values(["stk_code", "trade_date"]).reset_index(drop=True)
    row_hashes = pd.util.hash_pandas_object(
        normalized,
        index=False,
        categorize=True,
    )
    return {
        "row_count": int(len(normalized)),
        "stock_count": int(normalized["stk_code"].nunique()),
        "sample_start": str(normalized["trade_date"].min().date()),
        "sample_end": str(normalized["trade_date"].max().date()),
        "content_sha256": hashlib.sha256(row_hashes.to_numpy().tobytes()).hexdigest(),
    }


def _diagnostic_records(diagnostics: pd.DataFrame) -> dict[str, object]:
    return {
        row.factor_name: {
            "neutralized": bool(row.neutralized),
            "numeric_controls": row.numeric_controls,
            "categorical_controls": row.categorical_controls,
            "processed_coverage": float(row.processed_coverage),
            "pre_mean_absolute_control_correlation": (
                float(row.pre_mean_absolute_control_correlation)
                if pd.notna(row.pre_mean_absolute_control_correlation)
                else None
            ),
            "post_mean_absolute_control_correlation": (
                float(row.post_mean_absolute_control_correlation)
                if pd.notna(row.post_mean_absolute_control_correlation)
                else None
            ),
            "winsorized_to_processed_rank_correlation": float(
                row.winsorized_to_processed_rank_correlation
            ),
        }
        for row in diagnostics.itertuples(index=False)
    }


def build_audit(
    panel: pd.DataFrame,
    sensitivity: pd.DataFrame,
    valuation_diagnostics: pd.DataFrame,
    stock_diagnostics: pd.DataFrame,
    *,
    config_path: Path,
    primary_multiplier: float,
    source_manifests: dict[str, object],
) -> dict[str, object]:
    main = sensitivity.loc[sensitivity["mad_multiplier"].eq(primary_multiplier)]
    clipping_order = sensitivity.pivot(
        index="factor_name", columns="mad_multiplier", values="clipped_count"
    )
    monotone = clipping_order.apply(
        lambda row: row.sort_index().diff().dropna().le(0).all(), axis=1
    )
    winsorized_columns = [f"winsorized__{factor}" for factor in ALL_FACTOR_COLUMNS]
    standardized_valuation_columns = [
        f"standardized__{factor}" for factor in VALUATION_FACTOR_COLUMNS
    ]
    standardized_stock_columns = [
        f"standardized__{factor}" for factor in STOCK_FACTOR_COLUMNS
    ]
    missing_output_columns = sorted(
        set(
            [
                *winsorized_columns,
                *standardized_valuation_columns,
                *standardized_stock_columns,
            ]
        ).difference(panel.columns)
    )
    stock_staleness = panel["stock_market_cap_staleness_days"].dropna()
    audit = {
        "schema_version": "1.0",
        "stage": "G3c",
        "completed_substages": ["G3c-A", "G3c-B-valuation", "G3c-B-stock"],
        "status": "pass",
        "config_sha256": _sha256_file(config_path),
        "source_manifests": source_manifests,
        "sample_start": str(panel["signal_date"].min().date()),
        "sample_end": str(panel["signal_date"].max().date()),
        "row_count": int(len(panel)),
        "signal_date_count": int(panel["signal_date"].nunique()),
        "bond_count": int(panel["ts_code"].nunique()),
        "factor_count": len(ALL_FACTOR_COLUMNS),
        "primary_mad_multiplier": float(primary_multiplier),
        "sensitivity_multipliers": sorted(
            sensitivity["mad_multiplier"].unique().tolist()
        ),
        "duplicate_key_count": int(
            panel.duplicated(["signal_date", "ts_code"]).sum()
        ),
        "missing_output_columns": missing_output_columns,
        "forward_label_column_count": int(
            len(
                {"forward_return", "label_end_date", "entry_open", "exit_open"}.intersection(
                    panel.columns
                )
            )
        ),
        "coverage_by_factor": {
            factor: float(panel[factor].notna().mean()) for factor in ALL_FACTOR_COLUMNS
        },
        "primary_clipped_rate_by_factor": {
            row.factor_name: float(row.clipped_rate)
            for row in main.itertuples(index=False)
        },
        "primary_rank_correlation_by_factor": {
            row.factor_name: float(row.mean_cross_sectional_rank_correlation)
            for row in main.itertuples(index=False)
        },
        "completed_g3cb_factors": [
            *VALUATION_FACTOR_COLUMNS,
            *STOCK_FACTOR_COLUMNS,
        ],
        "valuation_preprocessing": _diagnostic_records(valuation_diagnostics),
        "stock_preprocessing": _diagnostic_records(stock_diagnostics),
        "stock_control_coverage": {
            "market_cap": float(panel["total_market_cap"].notna().mean()),
            "citic_level_1_industry": float(
                panel["ci_industry_control"].notna().mean()
            ),
            "joint": float(
                panel[["total_market_cap", "ci_industry_control"]]
                .notna()
                .all(axis=1)
                .mean()
            ),
            "market_cap_staleness_max_days": (
                int(stock_staleness.max()) if not stock_staleness.empty else None
            ),
        },
        "sensitivity_clipping_is_monotone_by_factor": {
            str(factor): bool(value) for factor, value in monotone.items()
        },
    }
    hard_failures = (
        audit["duplicate_key_count"]
        + len(missing_output_columns)
        + audit["forward_label_column_count"]
        + sum(not value for value in monotone)
    )
    if hard_failures:
        audit["status"] = "fail"
    return audit


def write_artifacts(
    panel: pd.DataFrame,
    sensitivity: pd.DataFrame,
    valuation_diagnostics: pd.DataFrame,
    stock_diagnostics: pd.DataFrame,
    audit: dict[str, object],
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for year, frame in panel.groupby(panel["signal_date"].dt.year, sort=True):
        year_root = output_root / "preprocessed_factor_panel_v1" / f"year={year}"
        year_root.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(year_root / "part-0000.parquet", index=False)
    sensitivity.to_parquet(output_root / "mad_sensitivity_v1.parquet", index=False)
    valuation_diagnostics.to_parquet(
        output_root / "valuation_preprocessing_diagnostics_v1.parquet",
        index=False,
    )
    stock_diagnostics.to_parquet(
        output_root / "stock_preprocessing_diagnostics_v1.parquet",
        index=False,
    )
    (output_root / "preprocessing_audit_v1.json").write_text(
        json.dumps(audit, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    winsorization = config["preprocessing"]["winsorization"]
    neutralization = config["preprocessing"]["neutralization"]
    factor_policies = neutralization["factor_policies"]
    primary_multiplier = float(winsorization["primary_multiplier"])
    multipliers = tuple(float(value) for value in winsorization["sensitivity_multipliers"])

    valuation, valuation_files = load_partitioned_panel(
        args.g2_root,
        "valuation_panel_v1",
        start=args.start,
        end=args.end,
    )
    stock, stock_files = load_partitioned_panel(
        args.g3_root,
        "stock_linkage_factor_panel_v1",
        start=args.start,
        end=args.end,
    )
    bond, bond_files = load_partitioned_panel(
        args.g3_root,
        "cb_trading_factor_panel_v1",
        start=args.start,
        end=args.end,
    )
    merged = merge_factor_panels(valuation, stock, bond)
    sensitivity = build_mad_sensitivity(
        merged,
        factor_columns=ALL_FACTOR_COLUMNS,
        multipliers=multipliers,
    )
    panel = winsorize_factor_panel(
        merged,
        factor_columns=ALL_FACTOR_COLUMNS,
        n_mad=primary_multiplier,
    )
    panel = add_cb_neutralization_controls(panel)
    market_cap_control = neutralization["stock_market_cap_control"]
    market_cap_start = (
        panel["signal_date"].min()
        - pd.Timedelta(days=market_cap_control["lookback_buffer_calendar_days"])
    ).strftime("%Y%m%d")
    market_cap_end = panel["signal_date"].max().strftime("%Y%m%d")
    market_cap_history = load_a_share_market_cap(
        DataCatalog.from_environment().a_share_daily_path,
        panel["stk_code"].dropna().unique(),
        start_date=market_cap_start,
        end_date=market_cap_end,
    )
    panel = attach_stock_market_cap_control(panel, market_cap_history)
    valuation_policies = tuple(
        FactorPreprocessingPolicy(
            name=factor,
            numeric_controls=tuple(factor_policies[factor]["numeric_controls"]),
            categorical_controls=tuple(
                factor_policies[factor]["categorical_controls"]
            ),
        )
        for factor in VALUATION_FACTOR_COLUMNS
    )
    panel, valuation_diagnostics = preprocess_factor_family(
        panel,
        valuation_policies,
    )
    stock_policies = tuple(
        FactorPreprocessingPolicy(
            name=factor,
            numeric_controls=tuple(factor_policies[factor]["numeric_controls"]),
            categorical_controls=tuple(
                factor_policies[factor]["categorical_controls"]
            ),
        )
        for factor in STOCK_FACTOR_COLUMNS
    )
    panel, stock_diagnostics = preprocess_factor_family(panel, stock_policies)
    audit = build_audit(
        panel,
        sensitivity,
        valuation_diagnostics,
        stock_diagnostics,
        config_path=args.config,
        primary_multiplier=primary_multiplier,
        source_manifests={
            "valuation": _source_manifest(valuation_files),
            "stock_linkage": _source_manifest(stock_files),
            "bond_trading": _source_manifest(bond_files),
            "stock_market_cap": _frame_manifest(market_cap_history),
        },
    )
    write_artifacts(
        panel,
        sensitivity,
        valuation_diagnostics,
        stock_diagnostics,
        audit,
        args.output_root,
    )
    print(json.dumps(audit, indent=2, default=str))
    if audit["status"] != "pass":
        raise SystemExit("G3c preprocessing audit failed")


if __name__ == "__main__":
    main()
