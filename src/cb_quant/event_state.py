"""Convert disclosed convertible-bond events into point-in-time state series."""

from __future__ import annotations

import numpy as np
import pandas as pd


RATING_SCORES = {
    "AAA": 19, "AA+": 18, "AA": 17, "AA-": 16,
    "A+": 15, "A": 14, "A-": 13, "BBB+": 12,
    "BBB": 11, "BBB-": 10, "BB+": 9, "BB": 8,
    "BB-": 7, "B+": 6, "B": 5, "B-": 4,
    "CCC": 3, "CC": 2, "C": 1,
}


def next_trade_dates(event_dates: pd.Series, trading_dates: pd.Series) -> pd.Series:
    """Map a disclosure date to the strictly next available market trading day."""
    calendar = pd.DatetimeIndex(pd.to_datetime(trading_dates).dropna().unique()).sort_values()
    dates = pd.to_datetime(event_dates, errors="coerce").dt.normalize()
    positions = calendar.searchsorted(dates.to_numpy(), side="right")
    result = pd.Series(pd.NaT, index=event_dates.index, dtype="datetime64[ns]")
    valid = (positions >= 0) & (positions < len(calendar)) & dates.notna().to_numpy()
    result.loc[valid] = calendar.take(positions[valid]).to_numpy()
    return result


def prepare_rating_events(rating_history: pd.DataFrame, trading_dates: pd.Series) -> pd.DataFrame:
    """Prepare lowest effective long-term rating per bond and effective date."""
    events = rating_history.copy()
    for column in ("rating_com_name", "rating_type", "rating_outlook"):
        if column not in events:
            events[column] = pd.NA
    events["rating"] = events["rating"].astype("string").str.strip().str.upper()
    events["rating_score"] = events["rating"].map(RATING_SCORES)
    events["ann_date"] = pd.to_datetime(events["ann_date"], errors="coerce")
    events["effective_date"] = next_trade_dates(events["ann_date"], trading_dates)
    events = events.dropna(subset=["ts_code", "effective_date", "rating_score"])
    events = events.sort_values(["ts_code", "effective_date", "rating_score", "ann_date"])
    events = events.drop_duplicates(["ts_code", "effective_date"], keep="first")
    return events[
        [
            "ts_code",
            "effective_date",
            "ann_date",
            "rating",
            "rating_score",
            "rating_com_name",
            "rating_type",
            "rating_outlook",
        ]
    ].reset_index(drop=True)


def prepare_share_events(share_history: pd.DataFrame, trading_dates: pd.Series) -> pd.DataFrame:
    """Prepare remaining-size and conversion-price states from disclosed reports."""
    events = share_history.copy()
    events["publish_date"] = pd.to_datetime(events["publish_date"], errors="coerce")
    events["end_date"] = pd.to_datetime(events["end_date"], errors="coerce")
    events["effective_date"] = next_trade_dates(events["publish_date"], trading_dates)
    events = events.dropna(subset=["ts_code", "effective_date", "remain_size", "convert_price"])
    events = events.sort_values(["ts_code", "effective_date", "end_date", "publish_date"])
    events = events.drop_duplicates(["ts_code", "effective_date"], keep="last")
    return events[
        ["ts_code", "effective_date", "publish_date", "end_date", "remain_size", "convert_price"]
    ].reset_index(drop=True)


def attach_asof_event_state(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    state_columns: list[str],
    *,
    date_column: str = "trade_date",
    code_column: str = "ts_code",
    effective_date_output: str | None = None,
) -> pd.DataFrame:
    """Attach latest event state available on each panel date without look-ahead."""
    required_panel = {date_column, code_column}
    required_events = {"effective_date", code_column, *state_columns}
    if missing := required_panel.difference(panel.columns):
        raise KeyError(f"Panel is missing columns: {sorted(missing)}")
    if missing := required_events.difference(events.columns):
        raise KeyError(f"Events are missing columns: {sorted(missing)}")

    left = panel.copy()
    left[date_column] = pd.to_datetime(left[date_column])
    right = events[[code_column, "effective_date", *state_columns]].copy()
    right["effective_date"] = pd.to_datetime(right["effective_date"])
    left = left.sort_values([date_column, code_column])
    right = right.sort_values(["effective_date", code_column])
    attached = pd.merge_asof(
        left,
        right,
        left_on=date_column,
        right_on="effective_date",
        by=code_column,
        direction="backward",
    )
    if effective_date_output is not None:
        attached = attached.rename(columns={"effective_date": effective_date_output})
    else:
        attached = attached.drop(columns="effective_date")
    return attached.sort_index()
