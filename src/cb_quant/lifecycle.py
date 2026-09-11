"""Point-in-time lifecycle states for convertible bonds."""

from __future__ import annotations

from enum import Enum
from typing import Iterable

import numpy as np
import pandas as pd


class LifecycleState(str, Enum):
    PRE_LISTED = "pre_listed"
    SEASONING = "seasoning"
    ACTIVE = "active"
    REDEMPTION_WATCH = "redemption_watch"
    REDEMPTION_ANNOUNCED = "redemption_announced"
    LAST_TRADING_WINDOW = "last_trading_window"
    SUSPENDED_OR_UNQUOTED = "suspended_or_unquoted"
    SETTLEMENT_RECEIVABLE = "settlement_receivable"
    SETTLEMENT_UNRESOLVED = "settlement_unresolved"
    SETTLED = "settled"
    DELISTED_UNRESOLVED = "delisted_unresolved"


_WATCH_STATUSES = {
    "redemption_watch",
    "triggered",
    "\u5df2\u6ee1\u8db3\u5f3a\u8d4e\u6761\u4ef6",
    "\u516c\u544a\u63d0\u793a\u5f3a\u8d4e",
}
_ANNOUNCED_STATUSES = {
    "implementation_announced",
    "announced",
    "\u516c\u544a\u5b9e\u65bd\u5f3a\u8d4e",
    "\u516c\u544a\u5f3a\u8d4e",
    "\u516c\u544a\u5230\u671f\u8d4e\u56de",
}
_WAIVED_STATUSES = {
    "not_called",
    "waived",
    "\u516c\u544a\u4e0d\u5f3a\u8d4e",
}
_DATE_COLUMNS = (
    "list_date",
    "delist_date",
    "maturity_date",
    "maturity_payment_date",
)
_EVENT_DATE_COLUMNS = (
    "effective_date",
    "last_trade_date",
    "call_reg_date",
    "payment_date",
)


def _normalize_dates(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.replace(r"\.0$", "", regex=True)
    compact = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    generic = pd.to_datetime(values, errors="coerce")
    return compact.fillna(generic).dt.normalize()


def _prepare_terms(terms: pd.DataFrame) -> pd.DataFrame:
    required = {"ts_code", "list_date", "maturity_date"}
    if missing := required.difference(terms.columns):
        raise KeyError(f"terms are missing columns: {sorted(missing)}")
    result = terms.copy()
    result["ts_code"] = result["ts_code"].astype("string")
    if result["ts_code"].isna().any():
        raise ValueError("terms contain missing ts_code")
    if result.duplicated("ts_code").any():
        raise ValueError("terms contain duplicate ts_code")
    for column in _DATE_COLUMNS:
        if column not in result:
            result[column] = pd.NaT
        result[column] = _normalize_dates(result[column])
    if "maturity_call_price" not in result:
        result["maturity_call_price"] = np.nan
    result["maturity_call_price"] = pd.to_numeric(
        result["maturity_call_price"], errors="coerce"
    )
    if result["list_date"].isna().any():
        raise ValueError("list_date cannot be missing")
    return result


def _prepare_events(events: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "ts_code",
        "effective_date",
        "event_status",
        "last_trade_date",
        "call_reg_date",
        "payment_date",
        "call_price",
    ]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)
    required = {"ts_code", "effective_date", "event_status"}
    if missing := required.difference(events.columns):
        raise KeyError(f"redemption events are missing columns: {sorted(missing)}")
    result = events.copy()
    for column in columns:
        if column not in result:
            result[column] = np.nan if column == "call_price" else pd.NaT
    result["ts_code"] = result["ts_code"].astype("string")
    result["event_status"] = result["event_status"].astype("string").str.strip()
    for column in _EVENT_DATE_COLUMNS:
        result[column] = _normalize_dates(result[column])
    result["call_price"] = pd.to_numeric(result["call_price"], errors="coerce")
    result = result.dropna(subset=["ts_code", "effective_date", "event_status"])
    if result.duplicated(["ts_code", "effective_date"]).any():
        raise ValueError("redemption events contain duplicate effective dates")

    waived = result["event_status"].isin(_WAIVED_STATUSES)
    result.loc[waived, ["last_trade_date", "call_reg_date", "payment_date"]] = pd.NaT
    result.loc[waived, "call_price"] = np.nan

    for row in result.itertuples(index=False):
        ordered = [
            row.effective_date,
            row.last_trade_date,
            row.call_reg_date,
            row.payment_date,
        ]
        known = [date for date in ordered if pd.notna(date)]
        if known != sorted(known):
            raise ValueError("redemption event timeline is inconsistent")
    return result[columns].sort_values(["ts_code", "effective_date"]).reset_index(drop=True)


def _attach_latest_events(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    result["_panel_order"] = np.arange(len(result))
    event_groups = {
        str(code): group.drop(columns="ts_code")
        .rename(columns={"effective_date": "event_effective_date"})
        .sort_values("event_effective_date")
        for code, group in events.groupby("ts_code", sort=False)
    }
    parts: list[pd.DataFrame] = []
    for code, bond_panel in result.groupby("ts_code", sort=False):
        bond_events = event_groups.get(str(code))
        if bond_events is None:
            empty = bond_panel.copy()
            empty["event_status"] = pd.Series(pd.NA, index=empty.index, dtype="string")
            for column in (
                "event_effective_date",
                "last_trade_date",
                "call_reg_date",
                "payment_date",
            ):
                empty[column] = pd.NaT
            empty["call_price"] = np.nan
            parts.append(empty)
            continue
        parts.append(
            pd.merge_asof(
                bond_panel.sort_values("trade_date"),
                bond_events,
                left_on="trade_date",
                right_on="event_effective_date",
                direction="backward",
                allow_exact_matches=True,
            )
        )
    attached = pd.concat(parts, ignore_index=True)
    attached["event_status"] = attached["event_status"].astype("string")
    return (
        attached.sort_values("_panel_order")
        .drop(columns="_panel_order")
        .reset_index(drop=True)
    )


def build_lifecycle_panel(
    terms: pd.DataFrame,
    trading_dates: Iterable,
    redemption_events: pd.DataFrame | None = None,
    *,
    tradability: pd.DataFrame | None = None,
    seasoning_days: int = 11,
) -> pd.DataFrame:
    """Build one lifecycle and entry/exit state per bond and market date."""
    if not isinstance(seasoning_days, int) or seasoning_days < 1:
        raise ValueError("seasoning_days must be a positive integer")
    reference = _prepare_terms(terms)
    events = _prepare_events(redemption_events)
    calendar = pd.DatetimeIndex(pd.to_datetime(list(trading_dates))).normalize()
    calendar = calendar.dropna().drop_duplicates().sort_values()
    if calendar.empty:
        raise ValueError("trading_dates cannot be empty")

    panel = pd.MultiIndex.from_product(
        [reference["ts_code"], calendar], names=["ts_code", "trade_date"]
    ).to_frame(index=False)
    panel = panel.merge(reference, on="ts_code", how="left", validate="many_to_one")
    list_positions = calendar.searchsorted(panel["list_date"].to_numpy(), side="left")
    date_positions = calendar.searchsorted(panel["trade_date"].to_numpy(), side="left")
    panel["listing_age_market_days"] = date_positions - list_positions + 1
    panel.loc[panel["trade_date"].lt(panel["list_date"]), "listing_age_market_days"] = 0
    listed_before_calendar = panel["list_date"].lt(calendar.min())
    panel.loc[listed_before_calendar, "listing_age_market_days"] = seasoning_days

    panel["lifecycle_state"] = LifecycleState.ACTIVE.value
    panel.loc[
        panel["trade_date"].lt(panel["list_date"]), "lifecycle_state"
    ] = LifecycleState.PRE_LISTED.value
    seasoning = panel["listing_age_market_days"].between(1, seasoning_days - 1)
    panel.loc[seasoning, "lifecycle_state"] = LifecycleState.SEASONING.value

    panel = _attach_latest_events(panel, events)
    watch = panel["event_status"].isin(_WATCH_STATUSES)
    announced = panel["event_status"].isin(_ANNOUNCED_STATUSES)
    panel.loc[watch, "lifecycle_state"] = LifecycleState.REDEMPTION_WATCH.value
    panel.loc[announced, "lifecycle_state"] = LifecycleState.REDEMPTION_ANNOUNCED.value

    event_last = announced & panel["last_trade_date"].notna()
    panel.loc[
        event_last & panel["trade_date"].eq(panel["last_trade_date"]),
        "lifecycle_state",
    ] = LifecycleState.LAST_TRADING_WINDOW.value
    after_event_last = event_last & panel["trade_date"].gt(panel["last_trade_date"])
    panel.loc[after_event_last, "lifecycle_state"] = LifecycleState.SUSPENDED_OR_UNQUOTED.value
    valid_event_settlement = (
        announced
        & panel["call_price"].notna()
        & panel["payment_date"].notna()
    )
    reached_event_registration = (
        announced
        & panel["call_reg_date"].notna()
        & panel["trade_date"].ge(panel["call_reg_date"])
    )
    event_receivable = reached_event_registration & valid_event_settlement
    panel.loc[event_receivable, "lifecycle_state"] = LifecycleState.SETTLEMENT_RECEIVABLE.value
    unresolved_event = reached_event_registration & ~valid_event_settlement
    panel.loc[unresolved_event, "lifecycle_state"] = LifecycleState.SETTLEMENT_UNRESOLVED.value
    event_settled = valid_event_settlement & panel["trade_date"].ge(
        panel["payment_date"]
    )
    panel.loc[event_settled, "lifecycle_state"] = LifecycleState.SETTLED.value

    no_announced_event = ~announced
    after_delist = (
        no_announced_event
        & panel["delist_date"].notna()
        & panel["trade_date"].gt(panel["delist_date"])
    )
    panel.loc[after_delist, "lifecycle_state"] = LifecycleState.DELISTED_UNRESOLVED.value

    reached_maturity = (
        no_announced_event
        & panel["maturity_date"].notna()
        & panel["trade_date"].ge(panel["maturity_date"])
    )
    valid_maturity_settlement = (
        panel["maturity_call_price"].notna()
        & panel["maturity_payment_date"].notna()
    )
    maturity_receivable = reached_maturity & valid_maturity_settlement
    panel.loc[maturity_receivable, "lifecycle_state"] = LifecycleState.SETTLEMENT_RECEIVABLE.value
    panel.loc[
        reached_maturity & ~valid_maturity_settlement, "lifecycle_state"
    ] = LifecycleState.SETTLEMENT_UNRESOLVED.value
    maturity_settled = (
        maturity_receivable
        & panel["maturity_payment_date"].notna()
        & panel["trade_date"].ge(panel["maturity_payment_date"])
    )
    panel.loc[maturity_settled, "lifecycle_state"] = LifecycleState.SETTLED.value

    panel["is_tradable"] = True
    if tradability is not None:
        required = {"trade_date", "ts_code", "is_tradable"}
        if missing := required.difference(tradability.columns):
            raise KeyError(f"tradability is missing columns: {sorted(missing)}")
        market = tradability[list(required)].copy()
        market["trade_date"] = _normalize_dates(market["trade_date"])
        market["ts_code"] = market["ts_code"].astype("string")
        if market.duplicated(["trade_date", "ts_code"]).any():
            raise ValueError("tradability contains duplicate date and bond keys")
        panel = panel.drop(columns="is_tradable").merge(
            market,
            on=["trade_date", "ts_code"],
            how="left",
            validate="one_to_one",
        )
        panel["is_tradable"] = panel["is_tradable"].fillna(False).astype(bool)

    normal_market_states = panel["lifecycle_state"].isin(
        [LifecycleState.ACTIVE.value, LifecycleState.REDEMPTION_WATCH.value]
    )
    panel.loc[
        normal_market_states & ~panel["is_tradable"], "lifecycle_state"
    ] = LifecycleState.SUSPENDED_OR_UNQUOTED.value
    panel["entry_allowed"] = panel["lifecycle_state"].isin(
        [LifecycleState.ACTIVE.value, LifecycleState.REDEMPTION_WATCH.value]
    ) & panel["is_tradable"]
    panel["exit_required"] = panel["lifecycle_state"].isin(
        [LifecycleState.REDEMPTION_ANNOUNCED.value, LifecycleState.LAST_TRADING_WINDOW.value]
    )
    panel["settlement_value"] = np.where(
        event_receivable | event_settled,
        panel["call_price"],
        np.where(maturity_receivable | maturity_settled, panel["maturity_call_price"], np.nan),
    )
    panel["settlement_date"] = np.where(
        event_receivable | event_settled,
        panel["payment_date"],
        np.where(maturity_receivable | maturity_settled, panel["maturity_payment_date"], pd.NaT),
    )
    panel["settlement_date"] = pd.to_datetime(panel["settlement_date"])
    panel["lifecycle_state"] = panel["lifecycle_state"].astype("string")
    return panel.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
