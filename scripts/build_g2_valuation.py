"""Build and audit the frozen G2 valuation factor panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from cb_quant import (
    DataCatalog,
    audit_valuation_panel,
    build_full_valuation_panel,
    build_valuation_anomaly_review,
    load_a_share_daily,
    summarize_parity_bucket_coverage,
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
        "--g1-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "G1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "G2",
    )
    parser.add_argument("--start", help="Optional small-sample start signal date")
    parser.add_argument("--end", help="Optional small-sample end signal date")
    parser.add_argument("--minimum-group-size", type=int, default=5)
    return parser.parse_args()


def load_g1_universe_panel(
    g1_root: Path,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    files = sorted((g1_root / "investable_universe_panel_v1").glob("year=*/part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No G1 universe partitions found under {g1_root}")
    panel = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    panel["signal_date"] = pd.to_datetime(panel["signal_date"]).dt.normalize()
    if start is not None:
        panel = panel.loc[panel["signal_date"].ge(pd.Timestamp(start).normalize())]
    if end is not None:
        panel = panel.loc[panel["signal_date"].le(pd.Timestamp(end).normalize())]
    if panel.empty:
        raise ValueError("Selected G1 universe panel is empty")
    return panel.reset_index(drop=True)


def write_valuation_artifacts(
    valuation_panel: pd.DataFrame,
    parity_coverage: pd.DataFrame,
    audit: dict[str, object],
    anomaly_review: pd.DataFrame,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for year, frame in valuation_panel.groupby(
        pd.to_datetime(valuation_panel["signal_date"]).dt.year,
        sort=True,
    ):
        year_root = output_root / "valuation_panel_v1" / f"year={year}"
        year_root.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(year_root / "part-0000.parquet", index=False)
    parity_coverage.to_parquet(
        output_root / "parity_bucket_coverage_v1.parquet",
        index=False,
    )
    anomaly_review.to_parquet(
        output_root / "valuation_anomaly_review_v1.parquet",
        index=False,
    )
    (output_root / "valuation_factor_audit_v1.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def enrich_valuation_audit(
    audit: dict[str, object],
    valuation_panel: pd.DataFrame,
    *,
    config_path: Path,
    research_id: str,
    minimum_group_size: int,
) -> dict[str, object]:
    enriched = dict(audit)
    factors = [
        "conversion_premium",
        "double_low",
        "parity_value_score",
    ]
    correlations = valuation_panel[factors].corr(method="spearman")
    enriched.update(
        {
            "research_id": research_id,
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "minimum_parity_group_size": int(minimum_group_size),
            "sample_start": str(valuation_panel["signal_date"].min().date()),
            "sample_end": str(valuation_panel["signal_date"].max().date()),
            "missing_stock_close_count": int(valuation_panel["stock_close"].isna().sum()),
            "missing_convert_price_count": int(
                valuation_panel["convert_price"].isna().sum()
            ),
            "valuation_coverage_by_year": {
                str(year): float(frame["is_valuation_available"].mean())
                for year, frame in valuation_panel.groupby(
                    valuation_panel["signal_date"].dt.year, sort=True
                )
            },
            "relative_value_coverage_by_year": {
                str(year): float(frame["parity_value_score"].notna().mean())
                for year, frame in valuation_panel.groupby(
                    valuation_panel["signal_date"].dt.year, sort=True
                )
            },
            "factor_spearman_correlation": {
                row: {
                    column: float(correlations.loc[row, column])
                    for column in correlations.columns
                }
                for row in correlations.index
            },
            "conversion_premium_quantiles": {
                str(quantile): float(value)
                for quantile, value in valuation_panel["conversion_premium"]
                .dropna()
                .quantile([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
                .items()
            },
        }
    )
    return enriched


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    catalog = DataCatalog.from_environment()
    catalog.validate_required_sources(2018, 2026)

    universe_panel = load_g1_universe_panel(
        args.g1_root,
        start=args.start,
        end=args.end,
    )
    investable = universe_panel.loc[universe_panel["is_eligible"]]
    stock_daily = load_a_share_daily(
        catalog.a_share_daily_path,
        investable["stk_code"].dropna().unique(),
        start_date=universe_panel["signal_date"].min().strftime("%Y%m%d"),
        end_date=universe_panel["signal_date"].max().strftime("%Y%m%d"),
    )
    valuation_panel = build_full_valuation_panel(
        universe_panel,
        stock_daily,
        minimum_group_size=args.minimum_group_size,
    )
    parity_coverage = summarize_parity_bucket_coverage(valuation_panel)
    anomaly_review = build_valuation_anomaly_review(valuation_panel)
    audit = audit_valuation_panel(
        valuation_panel,
        universe_panel,
        minimum_group_size=args.minimum_group_size,
    )
    audit = enrich_valuation_audit(
        audit,
        valuation_panel,
        config_path=args.config,
        research_id=config["research_id"],
        minimum_group_size=args.minimum_group_size,
    )
    write_valuation_artifacts(
        valuation_panel,
        parity_coverage,
        audit,
        anomaly_review,
        args.output_root,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, default=str))
    if audit["status"] != "pass":
        raise SystemExit("G2 valuation audit failed")


if __name__ == "__main__":
    main()
