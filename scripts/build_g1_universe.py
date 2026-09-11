"""Build and audit the frozen G1 biweekly investable-universe panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from cb_quant import (
    DataCatalog,
    UniverseRules,
    audit_event_boundaries,
    audit_universe_panel,
    build_biweekly_universe_panel,
    build_rebalance_calendar,
    build_universe_transitions,
    load_cb_reference_universe,
    load_open_trading_dates,
    prepare_rating_events,
    prepare_share_events,
    summarize_universe_by_date,
)
from cb_quant.event_state import RATING_SCORES


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "research_v1.json",
    )
    parser.add_argument("--start", help="Optional small-sample start date")
    parser.add_argument("--end", help="Optional small-sample end date")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "G1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    catalog = DataCatalog.from_environment()
    catalog.validate_required_sources(2018, 2026)

    sample = config["data"]["formal_sample"]
    start = args.start or sample["start"]
    end = args.end or sample["end"]
    universe = config["universe"]
    sampling = config["sampling"]
    rules = UniverseRules(
        minimum_age_days=int(universe["minimum_listing_age_market_days"]),
        turnover_window=int(universe["turnover_window_market_days"]),
        maximum_turnover=float(universe["maximum_cumulative_turnover"]),
        minimum_remain_size=float(universe["minimum_remaining_balance_yuan"]),
        minimum_rating_score=RATING_SCORES[universe["minimum_rating"]],
    )

    calendar_path = (
        catalog.tushare_raw_root / "trade_cal_sse_open_20180101_20260723.csv"
    )
    trading_dates = load_open_trading_dates(
        calendar_path, historical_calendar_path=catalog.ci_l1_daily_path
    )
    calendar = build_rebalance_calendar(
        trading_dates,
        anchor_date=sampling["anchor_signal_date"],
        sample_start=start,
        sample_end=end,
    )

    cb_daily = pd.read_csv(
        catalog.cb_daily_path,
        usecols=["ts_code", "trade_date", "close", "vol", "amount"],
        low_memory=False,
    )
    reference = load_cb_reference_universe(catalog.cb_terms_path, cb_daily["ts_code"])
    rating_events = prepare_rating_events(
        pd.read_parquet(
            catalog.tushare_history_root / "cb_rating_history.parquet"
        ),
        trading_dates,
    )
    share_events = prepare_share_events(
        pd.read_parquet(catalog.tushare_history_root / "cb_share_history.parquet"),
        trading_dates,
    )
    panel, funnel = build_biweekly_universe_panel(
        calendar["signal_date"],
        cb_daily,
        reference,
        trading_dates,
        rating_events,
        share_events,
        rules=rules,
        industry_database_path=catalog.ci_l1_daily_path,
    )
    counts = summarize_universe_by_date(panel)
    transitions = build_universe_transitions(panel)
    event_boundary_checks = audit_event_boundaries(
        rating_events,
        share_events,
        trading_dates,
        sample_start=start,
        sample_end=end,
    )
    audit = audit_universe_panel(panel, calendar)
    audit.update(
        {
            "research_id": config["research_id"],
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "sample_start": str(pd.Timestamp(start).date()),
            "sample_end": str(pd.Timestamp(end).date()),
            "calendar_rule": sampling["planned_signal_rule"],
            "universe_rules": rules.__dict__,
            "rating_event_count": int(len(rating_events)),
            "share_event_count": int(len(share_events)),
            "transition_count": int(len(transitions)),
            "exit_reason_counts": {
                str(key): int(value)
                for key, value in transitions.loc[
                    transitions["transition"].eq("exited"), "transition_reason"
                ]
                .value_counts()
                .items()
            },
            "event_boundary_check_count": int(len(event_boundary_checks)),
            "event_boundary_check_failures": int(
                (~event_boundary_checks["passed"]).sum()
            )
            if len(event_boundary_checks)
            else 0,
            "market_observation_ratio_20d": {
                key: float(value)
                for key, value in panel.loc[panel["is_eligible"]]
                ["market_observation_ratio_20d"]
                .describe()[["min", "25%", "50%", "75%", "max"]]
                .items()
            },
        }
    )
    if audit["event_boundary_check_failures"]:
        audit["status"] = "fail"

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    calendar.to_parquet(output_root / "rebalance_calendar_v1.parquet", index=False)
    counts.to_parquet(
        output_root / "universe_counts_by_signal_date_v1.parquet", index=False
    )
    funnel.to_parquet(output_root / "universe_funnel_by_signal_date_v1.parquet", index=False)
    event_boundary_checks.to_parquet(
        output_root / "event_boundary_checks_v1.parquet", index=False
    )
    transitions.to_parquet(
        output_root / "universe_transitions_v1.parquet", index=False
    )
    for year, frame in panel.groupby(panel["signal_date"].dt.year, sort=True):
        year_root = output_root / "investable_universe_panel_v1" / f"year={year}"
        year_root.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(year_root / "part-0000.parquet", index=False)
    (output_root / "universe_audit_v1.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, default=str))
    if audit["status"] != "pass":
        raise SystemExit("G1 universe audit failed")


if __name__ == "__main__":
    main()
