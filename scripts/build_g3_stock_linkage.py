"""Build and audit G3 underlying-stock and stock-bond linkage factors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from cb_quant import (
    DataCatalog,
    STOCK_LINKAGE_FACTOR_COLUMNS,
    audit_stock_linkage_factor_panel,
    build_stock_linkage_factor_panel,
    load_a_share_daily,
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
        default=PROJECT_ROOT / "artifacts" / "G3",
    )
    parser.add_argument("--start", help="Optional small-sample start signal date")
    parser.add_argument("--end", help="Optional small-sample end signal date")
    parser.add_argument("--minimum-observation-ratio", type=float, default=0.8)
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


def load_cb_daily_history(
    path: Path,
    bond_codes: set[str],
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    daily = pd.read_csv(
        path,
        usecols=["ts_code", "trade_date", "close", "pct_chg"],
        dtype={"ts_code": str, "trade_date": str},
    )
    daily["ts_code"] = daily["ts_code"].str.lstrip("\ufeff")
    daily["trade_date"] = pd.to_datetime(
        daily["trade_date"], format="%Y%m%d", errors="coerce"
    )
    return daily.loc[
        daily["ts_code"].isin(bond_codes)
        & daily["trade_date"].between(start_date, end_date)
    ].reset_index(drop=True)


def build_anomaly_review(
    factor_panel: pd.DataFrame,
    *,
    tail_count: int = 10,
) -> pd.DataFrame:
    if tail_count <= 0:
        raise ValueError("tail_count must be positive")
    reviews = []
    for factor in STOCK_LINKAGE_FACTOR_COLUMNS:
        valid = factor_panel.loc[factor_panel[factor].notna()]
        if valid.empty:
            continue
        low = valid.nsmallest(tail_count, factor)[
            ["signal_date", "ts_code", "stk_code", factor]
        ].copy()
        low["review_factor"] = factor
        low["review_tail"] = "lowest"
        low = low.rename(columns={factor: "factor_value"})
        high = valid.nlargest(tail_count, factor)[
            ["signal_date", "ts_code", "stk_code", factor]
        ].copy()
        high["review_factor"] = factor
        high["review_tail"] = "highest"
        high = high.rename(columns={factor: "factor_value"})
        reviews.extend([low, high])
    if not reviews:
        return pd.DataFrame(
            columns=[
                "review_factor",
                "review_tail",
                "signal_date",
                "ts_code",
                "stk_code",
                "factor_value",
            ]
        )
    return pd.concat(reviews, ignore_index=True)[
        [
            "review_factor",
            "review_tail",
            "signal_date",
            "ts_code",
            "stk_code",
            "factor_value",
        ]
    ].sort_values(["review_factor", "review_tail", "factor_value"])


def enrich_audit(
    audit: dict[str, object],
    factor_panel: pd.DataFrame,
    *,
    config_path: Path,
    research_id: str,
    minimum_observation_ratio: float,
) -> dict[str, object]:
    correlations = factor_panel[STOCK_LINKAGE_FACTOR_COLUMNS].corr(method="spearman")
    enriched = dict(audit)
    enriched.update(
        {
            "research_id": research_id,
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "sample_start": str(factor_panel["signal_date"].min().date()),
            "sample_end": str(factor_panel["signal_date"].max().date()),
            "minimum_observation_ratio": float(minimum_observation_ratio),
            "factor_definitions": {
                "stock_returns": "adjusted close t / adjusted close t-N - 1",
                "stock_volatility_20d": "20-market-day standard deviation of daily pct_chg, annualized by sqrt(252)",
                "stock_rsi_20d": "20-market-day positive adjusted price change / absolute adjusted price change * 100",
                "stock_price_to_high_20d": "adjusted close / 20-market-day adjusted close maximum",
                "stock_percent_b_20d": "(adjusted close - 20-day lower Bollinger band) / band width; bands use mean plus/minus 2 population standard deviations",
                "stock_amihud_20d": "mean(abs(daily return) / amount_yuan) * 1e8",
                "stock_mfi_20d": "20-market-day positive adjusted typical-price money flow / total directional money flow * 100",
                "cb_stock_return_spread": "raw CB close return - adjusted stock close return",
                "cb_stock_correlation": "rolling paired-observation Pearson correlation of CB and stock daily returns",
                "cb_stock_beta": "rolling paired-observation covariance(CB, stock) / variance(stock)",
            },
            "factor_spearman_correlation": {
                row: {
                    column: float(correlations.loc[row, column])
                    if pd.notna(correlations.loc[row, column])
                    else None
                    for column in correlations.columns
                }
                for row in correlations.index
            },
            "factor_quantiles": {
                factor: {
                    str(quantile): float(value)
                    for quantile, value in factor_panel[factor]
                    .dropna()
                    .quantile([0.01, 0.05, 0.5, 0.95, 0.99])
                    .items()
                }
                for factor in STOCK_LINKAGE_FACTOR_COLUMNS
            },
        }
    )
    return enriched


def write_stock_linkage_artifacts(
    factor_panel: pd.DataFrame,
    anomaly_review: pd.DataFrame,
    audit: dict[str, object],
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for year, frame in factor_panel.groupby(
        pd.to_datetime(factor_panel["signal_date"]).dt.year,
        sort=True,
    ):
        year_root = output_root / "stock_linkage_factor_panel_v1" / f"year={year}"
        year_root.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(year_root / "part-0000.parquet", index=False)
    anomaly_review.to_parquet(
        output_root / "stock_linkage_anomaly_review_v1.parquet",
        index=False,
    )
    (output_root / "stock_linkage_factor_audit_v1.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


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
    first_signal = investable["signal_date"].min()
    last_signal = investable["signal_date"].max()
    history_start = first_signal - pd.Timedelta(days=90)
    stock_daily = load_a_share_daily(
        catalog.a_share_daily_path,
        investable["stk_code"].dropna().unique(),
        start_date=history_start.strftime("%Y%m%d"),
        end_date=last_signal.strftime("%Y%m%d"),
    )
    cb_daily = load_cb_daily_history(
        catalog.cb_daily_path,
        set(investable["ts_code"].unique()),
        start_date=history_start,
        end_date=last_signal,
    )
    factor_panel = build_stock_linkage_factor_panel(
        universe_panel,
        stock_daily,
        cb_daily,
        minimum_observation_ratio=args.minimum_observation_ratio,
    )
    audit = audit_stock_linkage_factor_panel(
        factor_panel,
        universe_panel,
        stock_daily,
        cb_daily,
        minimum_observation_ratio=args.minimum_observation_ratio,
    )
    audit = enrich_audit(
        audit,
        factor_panel,
        config_path=args.config,
        research_id=config["research_id"],
        minimum_observation_ratio=args.minimum_observation_ratio,
    )
    anomaly_review = build_anomaly_review(factor_panel)
    write_stock_linkage_artifacts(
        factor_panel,
        anomaly_review,
        audit,
        args.output_root,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, default=str))
    if audit["status"] != "pass":
        raise SystemExit("G3 stock-linkage factor audit failed")


if __name__ == "__main__":
    main()
