"""Point-in-time daily investable-universe construction for convertible bonds."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .event_state import attach_asof_event_state
from .legacy_schema import (
    LEGACY_G1_FUNNEL_COUNT_COLUMN,
    LEGACY_G1_FUNNEL_STAGE_COLUMN,
    LEGACY_G1_FUNNEL_STAGES,
)


def load_open_trading_dates(
    calendar_path: Path,
    historical_calendar_path: Path | None = None,
) -> pd.DatetimeIndex:
    """Load open dates, optionally extending history from a daily DuckDB source."""
    calendar = pd.read_csv(calendar_path, low_memory=False)
    dates = pd.to_datetime(calendar.loc[calendar["is_open"].eq(1), "cal_date"].astype(str))
    all_dates = pd.DatetimeIndex(dates.dropna().unique()).normalize()
    if historical_calendar_path is not None:
        if not historical_calendar_path.exists():
            raise FileNotFoundError(
                f"Historical calendar database does not exist: {historical_calendar_path}"
            )
        with duckdb.connect(str(historical_calendar_path), read_only=True) as connection:
            history = connection.execute(
                "SELECT DISTINCT trade_date::DATE AS trade_date FROM default_table ORDER BY trade_date"
            ).df()
        historical_dates = pd.DatetimeIndex(
            pd.to_datetime(history["trade_date"], errors="coerce").dropna().unique()
        ).normalize()
        all_dates = all_dates.union(historical_dates)
    return all_dates.sort_values()


def load_cb_reference_universe(cb_terms_path: Path, daily_codes: pd.Series) -> pd.DataFrame:
    """Return ordinary CBs that actually occur in the research-period daily panel."""
    observed_codes = set(daily_codes.astype(str).str.lstrip("\ufeff"))
    with duckdb.connect(str(cb_terms_path), read_only=True) as connection:
        terms = connection.execute(
            """
            SELECT
                ts_code,
                bond_short_name,
                stk_code,
                list_date,
                delist_date,
                maturity_date,
                issue_size,
                first_conv_price
            FROM default_table
            WHERE cb_type = 'CB'
              AND ts_code IS NOT NULL
              AND list_date IS NOT NULL
              AND issue_size IS NOT NULL
            """
        ).df()
    terms["ts_code"] = terms["ts_code"].astype(str)
    terms = terms.loc[terms["ts_code"].isin(observed_codes)].copy()
    for column in ("list_date", "delist_date", "maturity_date"):
        terms[column] = pd.to_datetime(
            terms[column].astype("Int64").astype("string"),
            format="%Y%m%d",
            errors="coerce",
        )
    terms["issue_size"] = pd.to_numeric(terms["issue_size"], errors="coerce")
    terms["first_conv_price"] = pd.to_numeric(terms["first_conv_price"], errors="coerce")
    return terms.dropna(subset=["list_date", "issue_size"]).drop_duplicates("ts_code").sort_values("ts_code")


def build_daily_universe_snapshot(
    as_of_date: str | pd.Timestamp,
    cb_daily: pd.DataFrame,
    reference_universe: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    rating_events: pd.DataFrame,
    share_events: pd.DataFrame,
    *,
    minimum_age_days: int = 11,
    turnover_window: int = 20,
    maximum_turnover: float = 1.0,
    minimum_remain_size: float = 200_000_000.0,
    minimum_rating_score: int = 14,
) -> pd.DataFrame:
    """Build the full daily candidate snapshot using information available at ``t`` close.

    The return includes every ordinary CB in the reference universe and boolean
    filter columns. It deliberately keeps failed candidates for a transparent
    sample funnel instead of returning only the final eligible set.
    """
    if turnover_window <= 0:
        raise ValueError("turnover_window must be positive")
    signal_date = pd.Timestamp(as_of_date).normalize()
    signal_index = trading_dates.searchsorted(signal_date)
    if signal_index == len(trading_dates) or trading_dates[signal_index] != signal_date:
        raise ValueError(f"as_of_date must be an open trading day: {signal_date.date()}")

    window_start = max(0, signal_index - turnover_window + 1)
    window_dates = trading_dates[window_start : signal_index + 1]
    reference = reference_universe.copy()
    reference["list_date"] = pd.to_datetime(reference["list_date"]).dt.normalize()
    reference["list_trade_date"] = _next_or_same_trade_date(reference["list_date"], trading_dates)

    window = pd.MultiIndex.from_product(
        [reference["ts_code"], window_dates], names=["ts_code", "trade_date"]
    ).to_frame(index=False)
    window = window.merge(reference, on="ts_code", how="left", validate="m:1")
    window["is_listed"] = window["trade_date"] >= window["list_trade_date"]

    market = cb_daily[["ts_code", "trade_date", "close", "vol", "amount"]].copy()
    market["ts_code"] = market["ts_code"].astype(str).str.lstrip("\ufeff")
    market["trade_date"] = pd.to_datetime(market["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    market = market.loc[market["trade_date"].isin(window_dates)]
    window = window.merge(market, on=["ts_code", "trade_date"], how="left", validate="1:1")
    window["has_market_observation"] = window["close"].notna()
    window[["vol", "amount"]] = window[["vol", "amount"]].fillna(0.0)

    share_state_columns = [
        column
        for column in ("remain_size", "convert_price", "publish_date", "end_date")
        if column in share_events
    ]
    window = attach_asof_event_state(
        window,
        share_events,
        share_state_columns,
        effective_date_output="share_effective_date",
    )
    window["remain_size"] = pd.to_numeric(window["remain_size"], errors="coerce")
    window["convert_price"] = pd.to_numeric(window["convert_price"], errors="coerce")
    window["remain_size"] = window["remain_size"].fillna(window["issue_size"])
    window["conversion_price_source"] = np.where(
        window["convert_price"].notna(),
        "disclosed_event",
        np.where(window["first_conv_price"].notna(), "initial_terms", "missing"),
    )
    window["convert_price"] = window["convert_price"].fillna(window["first_conv_price"])
    valid_turnover = window["is_listed"] & window["vol"].gt(0) & window["remain_size"].gt(0)
    window["daily_turnover"] = np.where(
        valid_turnover,
        window["vol"] * 1000.0 / window["remain_size"],
        0.0,
    )
    window = window.sort_values(["ts_code", "trade_date"])
    window["turnover_20d"] = window.groupby("ts_code")["daily_turnover"].transform(
        lambda series: series.rolling(turnover_window, min_periods=1).sum()
    )
    window["market_observations_20d"] = window.groupby("ts_code")[
        "has_market_observation"
    ].transform(lambda series: series.rolling(turnover_window, min_periods=1).sum())
    window["market_observation_ratio_20d"] = (
        window["market_observations_20d"] / turnover_window
    )
    snapshot = window.loc[window["trade_date"].eq(signal_date)].copy()
    rating_state_columns = [
        column
        for column in (
            "rating",
            "rating_score",
            "ann_date",
            "rating_com_name",
            "rating_type",
            "rating_outlook",
        )
        if column in rating_events
    ]
    snapshot = attach_asof_event_state(
        snapshot,
        rating_events,
        rating_state_columns,
        effective_date_output="rating_effective_date",
    )

    list_indices = trading_dates.searchsorted(snapshot["list_trade_date"].to_numpy())
    snapshot["listing_age_days"] = signal_index - list_indices + 1
    snapshot["listing_age_is_lower_bound"] = snapshot["list_date"].lt(trading_dates.min())
    snapshot.loc[snapshot["listing_age_is_lower_bound"], "listing_age_days"] = minimum_age_days
    snapshot["is_age_eligible"] = snapshot["listing_age_days"].ge(minimum_age_days)
    snapshot["is_turnover_eligible"] = snapshot["turnover_20d"].le(maximum_turnover)
    snapshot["is_size_eligible"] = snapshot["remain_size"].ge(minimum_remain_size)
    snapshot["is_rating_eligible"] = snapshot["rating_score"].ge(minimum_rating_score)
    snapshot["is_signal_tradable"] = (
        snapshot["close"].gt(0) & snapshot["vol"].gt(0) & snapshot["amount"].gt(0)
    )
    filter_columns = [
        "is_age_eligible",
        "is_turnover_eligible",
        "is_size_eligible",
        "is_rating_eligible",
        "is_signal_tradable",
    ]
    snapshot["is_eligible"] = snapshot[filter_columns].all(axis=1)
    snapshot["primary_exclusion_reason"] = _primary_exclusion_reason(snapshot)
    return snapshot.sort_values("ts_code", ignore_index=True)


def universe_funnel(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Summarize sequential sample-pool filters for a single signal date."""
    filters = [
        (LEGACY_G1_FUNNEL_STAGES[0], pd.Series(True, index=snapshot.index)),
        (LEGACY_G1_FUNNEL_STAGES[1], snapshot["is_age_eligible"]),
        (LEGACY_G1_FUNNEL_STAGES[2], snapshot["is_turnover_eligible"]),
        (LEGACY_G1_FUNNEL_STAGES[3], snapshot["is_size_eligible"]),
        (LEGACY_G1_FUNNEL_STAGES[4], snapshot["is_rating_eligible"]),
        (LEGACY_G1_FUNNEL_STAGES[5], snapshot["is_signal_tradable"]),
    ]
    running = pd.Series(True, index=snapshot.index)
    rows = []
    for name, passed in filters:
        running &= passed.fillna(False)
        rows.append(
            {
                LEGACY_G1_FUNNEL_STAGE_COLUMN: name,
                LEGACY_G1_FUNNEL_COUNT_COLUMN: int(running.sum()),
            }
        )
    return pd.DataFrame(rows)


def _next_or_same_trade_date(dates: pd.Series, trading_dates: pd.DatetimeIndex) -> pd.Series:
    positions = trading_dates.searchsorted(dates.to_numpy(), side="left")
    result = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    valid = (positions >= 0) & (positions < len(trading_dates)) & dates.notna().to_numpy()
    result.loc[valid] = trading_dates.take(positions[valid]).to_numpy()
    return result


def _primary_exclusion_reason(snapshot: pd.DataFrame) -> pd.Series:
    """Return the first failed sequential universe rule for each candidate."""
    conditions = [
        ~snapshot["is_listed"].fillna(False),
        ~snapshot["is_age_eligible"].fillna(False),
        ~snapshot["is_turnover_eligible"].fillna(False),
        ~snapshot["is_size_eligible"].fillna(False),
        snapshot["rating_score"].isna(),
        snapshot["rating_score"].notna() & ~snapshot["is_rating_eligible"].fillna(False),
        ~snapshot["is_signal_tradable"].fillna(False),
    ]
    labels = [
        "not_yet_listed",
        "listing_age_below_minimum",
        "turnover_above_maximum",
        "remaining_balance_below_minimum",
        "rating_missing",
        "rating_below_minimum",
        "no_signal_day_trade",
    ]
    return pd.Series(
        np.select(conditions, labels, default="eligible"),
        index=snapshot.index,
        dtype="string",
    )
