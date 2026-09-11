"""Build and audit G3 convertible-bond daily trading factors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from cb_quant import (
    CB_TRADING_FACTOR_COLUMNS,
    DataCatalog,
    audit_cb_trading_factor_panel,
    build_cb_trading_factor_panel,
    load_cb_reference_universe,
    load_open_trading_dates,
    prepare_share_events,
)
from cb_quant.cb_trading import (
    TUSHARE_CB_DAILY_AMOUNT_CNY_PER_REPORTED_UNIT,
    TUSHARE_CB_DAILY_BONDS_PER_HAND,
    TUSHARE_CB_DAILY_VWAP_CNY_PER_HAND,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VWAP_RANGE_TOLERANCE = 0.02


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


def load_cb_daily(path: Path) -> pd.DataFrame:
    columns = [
        "ts_code",
        "trade_date",
        "pre_close",
        "open",
        "high",
        "low",
        "close",
        "pct_chg",
        "vol",
        "amount",
    ]
    return pd.read_csv(
        path,
        usecols=columns,
        dtype={"ts_code": str, "trade_date": str},
        low_memory=False,
    )


def build_full_history_vwap_unit_audit(cb_daily: pd.DataFrame) -> dict[str, object]:
    daily = cb_daily.copy()
    for column in ("high", "low", "vol", "amount"):
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    valid = (
        daily["high"].gt(0.0)
        & daily["low"].gt(0.0)
        & daily["vol"].gt(0.0)
        & daily["amount"].gt(0.0)
    )
    review = daily.loc[valid, ["ts_code", "trade_date", "high", "low", "vol", "amount"]].copy()
    review["vwap"] = (
        review["amount"] * TUSHARE_CB_DAILY_VWAP_CNY_PER_HAND / review["vol"]
    )
    review["inside_range"] = (
        review["vwap"].ge(review["low"] - VWAP_RANGE_TOLERANCE)
        & review["vwap"].le(review["high"] + VWAP_RANGE_TOLERANCE)
    )
    review["exchange"] = review["ts_code"].astype(str).str[-2:]
    review["year"] = review["trade_date"].astype(str).str[:4]
    return {
        "formula": "amount_ten_thousand_cny * 10000 / (volume_hands * 10)",
        "vwap_formula": "amount_ten_thousand_cny * 10000 / (volume_hands * 10)",
        "actual_amount_cny_formula": "amount_ten_thousand_cny * 10000",
        "amount_cny_per_reported_unit": TUSHARE_CB_DAILY_AMOUNT_CNY_PER_REPORTED_UNIT,
        "bonds_per_hand": TUSHARE_CB_DAILY_BONDS_PER_HAND,
        "valid_row_count": int(len(review)),
        "inside_daily_range_count": int(review["inside_range"].sum()),
        "inside_daily_range_rate": float(review["inside_range"].mean()),
        "by_exchange": {
            str(exchange): {
                "count": int(len(frame)),
                "inside_daily_range_rate": float(frame["inside_range"].mean()),
            }
            for exchange, frame in review.groupby("exchange", sort=True)
        },
        "by_year": {
            str(year): {
                "count": int(len(frame)),
                "inside_daily_range_rate": float(frame["inside_range"].mean()),
            }
            for year, frame in review.groupby("year", sort=True)
        },
    }


def build_anomaly_review(
    factor_panel: pd.DataFrame,
    *,
    tail_count: int = 10,
) -> pd.DataFrame:
    if tail_count <= 0:
        raise ValueError("tail_count must be positive")
    reviews = []
    source_columns = [
        "signal_date",
        "ts_code",
        "cb_close",
        "cb_vol",
        "cb_amount",
        "remain_size",
    ]
    for factor in CB_TRADING_FACTOR_COLUMNS:
        valid = factor_panel.loc[factor_panel[factor].notna()]
        if valid.empty:
            continue
        for tail, frame in (
            ("lowest", valid.nsmallest(tail_count, factor)),
            ("highest", valid.nlargest(tail_count, factor)),
        ):
            review = frame[[*source_columns, factor]].copy()
            review["review_factor"] = factor
            review["review_tail"] = tail
            review = review.rename(columns={factor: "factor_value"})
            reviews.append(review)
    if not reviews:
        return pd.DataFrame(
            columns=[
                "review_factor",
                "review_tail",
                *source_columns,
                "factor_value",
            ]
        )
    return pd.concat(reviews, ignore_index=True)[
        [
            "review_factor",
            "review_tail",
            *source_columns,
            "factor_value",
        ]
    ].sort_values(["review_factor", "review_tail", "factor_value"])


def enrich_audit(
    audit: dict[str, object],
    factor_panel: pd.DataFrame,
    cb_daily: pd.DataFrame,
    *,
    config_path: Path,
    research_id: str,
    minimum_observation_ratio: float,
) -> dict[str, object]:
    correlations = factor_panel[CB_TRADING_FACTOR_COLUMNS].corr(method="spearman")
    enriched = dict(audit)
    enriched.update(
        {
            "research_id": research_id,
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "sample_start": str(factor_panel["signal_date"].min().date()),
            "sample_end": str(factor_panel["signal_date"].max().date()),
            "minimum_observation_ratio": float(minimum_observation_ratio),
            "factor_definitions": {
                "cb_return": "raw close t / raw close t-N - 1",
                "cb_short_long_momentum_5_20d": "5-day return minus 20-day return",
                "cb_daily_turnover": "observed source volume * 1000 / point-in-time remaining principal; observed vol=0 and amount=0 maps to zero, while an absent source row remains missing",
                "cb_abnormal_turnover": "current turnover / prior-window mean turnover using at least 80% genuinely observed rows; absent source rows remain missing and do not enter as zero",
                "cb_log_amount_zscore_20d": "current log actual amount_cny relative to prior 20-market-day mean and sample standard deviation",
                "cb_amihud_20d": "mean(abs(daily return) / actual_amount_cny) * 1e8",
                "cb_vwap": "amount_ten_thousand_cny * 10000 / (volume_hands * 10)",
                "cb_close_to_vwap": "close / VWAP - 1",
                "cb_close_to_vwap_mean_5d": "five-market-day mean close-to-VWAP deviation",
            },
            "missing_data_policy": {
                "observed_zero_trade": "A source row with vol=0 and amount=0 has zero daily turnover; VWAP, log amount, Amihud, and close-to-VWAP factors remain missing on that date.",
                "unknown_market_gap": "A calendar bond-date without a source market row remains missing for turnover, returns, amount, VWAP, and dependent factors.",
                "rolling_turnover": "Unknown market gaps are excluded from the rolling mean and observation count; observed zero-turnover rows are valid observations.",
            },
            "factor_quantiles": {
                factor: {
                    str(quantile): float(value)
                    for quantile, value in factor_panel[factor]
                    .dropna()
                    .quantile([0.01, 0.05, 0.5, 0.95, 0.99])
                    .items()
                }
                for factor in CB_TRADING_FACTOR_COLUMNS
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
            "full_history_vwap_unit_audit": build_full_history_vwap_unit_audit(
                cb_daily
            ),
        }
    )
    return enriched


def write_cb_trading_artifacts(
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
        year_root = output_root / "cb_trading_factor_panel_v1" / f"year={year}"
        year_root.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(year_root / "part-0000.parquet", index=False)
    anomaly_review.to_parquet(
        output_root / "cb_trading_anomaly_review_v1.parquet",
        index=False,
    )
    (output_root / "cb_trading_factor_audit_v1.json").write_text(
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
    cb_daily = load_cb_daily(catalog.cb_daily_path)
    reference = load_cb_reference_universe(catalog.cb_terms_path, cb_daily["ts_code"])
    trading_dates = load_open_trading_dates(
        catalog.tushare_raw_root / "trade_cal_sse_open_20180101_20260723.csv",
        historical_calendar_path=catalog.ci_l1_daily_path,
    )
    share_events = prepare_share_events(
        pd.read_parquet(
            catalog.tushare_history_root / "cb_share_history.parquet"
        ),
        trading_dates,
    )
    factor_panel = build_cb_trading_factor_panel(
        universe_panel,
        cb_daily,
        reference,
        share_events,
        trading_dates,
        minimum_observation_ratio=args.minimum_observation_ratio,
    )
    audit = audit_cb_trading_factor_panel(
        factor_panel,
        universe_panel,
        cb_daily,
        reference,
        share_events,
        trading_dates,
        minimum_observation_ratio=args.minimum_observation_ratio,
    )
    audit = enrich_audit(
        audit,
        factor_panel,
        cb_daily,
        config_path=args.config,
        research_id=config["research_id"],
        minimum_observation_ratio=args.minimum_observation_ratio,
    )
    anomaly_review = build_anomaly_review(factor_panel)
    write_cb_trading_artifacts(
        factor_panel,
        anomaly_review,
        audit,
        args.output_root,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, default=str))
    if audit["status"] != "pass":
        raise SystemExit("G3 convertible-bond trading-factor audit failed")


if __name__ == "__main__":
    main()
