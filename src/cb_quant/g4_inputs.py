"""Materialize execution-aligned inputs for the G4 research gate."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .legacy_schema import LEGACY_REDEMPTION_COLUMNS


def _normalize_dates(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.replace(r"\.0$", "", regex=True)
    compact = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    generic = pd.to_datetime(values, errors="coerce")
    return compact.fillna(generic).dt.normalize()


def _valid_price(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.gt(0) & np.isfinite(numeric)


def build_execution_market(
    cb_daily: pd.DataFrame,
    *,
    trading_dates: Iterable,
    bond_codes: Iterable[str],
) -> pd.DataFrame:
    """Create a complete daily market grid with conservative open-fill flags.

    A one-price session above the previous close blocks buys, while a one-price
    session below the previous close blocks sells. Full-session turnover is not
    used as a capacity threshold; positive volume and amount only establish that
    the source records an actual trade.
    """
    required = {
        "trade_date",
        "ts_code",
        "pre_close",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
    }
    if missing := required.difference(cb_daily.columns):
        raise KeyError(f"CB daily market is missing columns: {sorted(missing)}")

    observed = cb_daily[list(required)].copy()
    observed["trade_date"] = _normalize_dates(observed["trade_date"])
    observed["ts_code"] = observed["ts_code"].astype("string").str.lstrip("\ufeff")
    numeric_columns = [
        "pre_close",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
    ]
    for column in numeric_columns:
        observed[column] = pd.to_numeric(observed[column], errors="coerce")
    observed = observed.dropna(subset=["trade_date", "ts_code"])
    if observed.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("CB daily market contains duplicate date and bond keys")
    observed["has_market_observation"] = True

    calendar = pd.DatetimeIndex(pd.to_datetime(list(trading_dates), errors="raise"))
    calendar = calendar.normalize().drop_duplicates().sort_values()
    codes = pd.Index(pd.Series(list(bond_codes), dtype="string").dropna().unique())
    codes = codes.astype("string").sort_values()
    if calendar.empty:
        raise ValueError("trading_dates cannot be empty")
    if codes.empty:
        raise ValueError("bond_codes cannot be empty")

    grid = pd.MultiIndex.from_product(
        [calendar, codes], names=["trade_date", "ts_code"]
    ).to_frame(index=False)
    result = grid.merge(
        observed,
        on=["trade_date", "ts_code"],
        how="left",
        validate="one_to_one",
    )
    result["has_market_observation"] = result["has_market_observation"].eq(True)
    result["has_positive_trade"] = result["vol"].gt(0) & result["amount"].gt(0)
    result["has_valid_open"] = _valid_price(result["open"])
    result["has_valid_close"] = _valid_price(result["close"])

    flat_session = (
        result["has_valid_open"]
        & _valid_price(result["high"])
        & _valid_price(result["low"])
        & np.isclose(result["open"], result["high"], rtol=0.0, atol=1e-8)
        & np.isclose(result["open"], result["low"], rtol=0.0, atol=1e-8)
    )
    valid_previous_close = _valid_price(result["pre_close"])
    result["is_one_price_up"] = (
        flat_session
        & valid_previous_close
        & result["open"].gt(result["pre_close"])
    )
    result["is_one_price_down"] = (
        flat_session
        & valid_previous_close
        & result["open"].lt(result["pre_close"])
    )
    open_execution_observed = result["has_valid_open"] & result["has_positive_trade"]
    result["buy_allowed"] = open_execution_observed & ~result["is_one_price_up"]
    result["sell_allowed"] = open_execution_observed & ~result["is_one_price_down"]
    result["is_tradable"] = open_execution_observed
    result["amount_yuan"] = result["amount"] * 1000.0
    return result.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def build_partition_schedule(
    rebalance_calendar: pd.DataFrame,
    *,
    sample_partition: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None,
) -> pd.DataFrame:
    """Slice the frozen calendar and flag labels crossing the sample boundary."""
    required = {"signal_date", "entry_date", "next_entry_date"}
    if missing := required.difference(rebalance_calendar.columns):
        raise KeyError(f"rebalance calendar is missing columns: {sorted(missing)}")
    schedule = rebalance_calendar.copy()
    for column in required:
        schedule[column] = _normalize_dates(schedule[column])
    if schedule.duplicated("signal_date").any():
        raise ValueError("rebalance calendar contains duplicate signal dates")

    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize() if end is not None else None
    signal_in_partition = schedule["signal_date"].ge(start_date)
    if end_date is not None:
        signal_in_partition &= schedule["signal_date"].le(end_date)
    schedule = schedule.loc[signal_in_partition].copy()
    schedule["exit_date"] = schedule["next_entry_date"]
    schedule["label_end_date"] = schedule["exit_date"]
    within = (
        schedule["entry_date"].notna()
        & schedule["exit_date"].notna()
        & schedule["entry_date"].gt(schedule["signal_date"])
    )
    if end_date is not None:
        within &= schedule["exit_date"].le(end_date)
    schedule["sample_partition"] = sample_partition
    schedule["is_label_within_partition"] = within
    schedule["label_exclusion_reason"] = np.where(
        within, "eligible", "partition_boundary"
    )
    columns = [
        "signal_date",
        "entry_date",
        "exit_date",
        "label_end_date",
        "sample_partition",
        "is_label_within_partition",
        "label_exclusion_reason",
    ]
    return schedule[columns].sort_values("signal_date").reset_index(drop=True)


def build_horizon_schedules(
    rebalance_calendar: pd.DataFrame,
    *,
    trading_dates: Iterable,
    horizons: Iterable[int],
    sample_partition: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None,
) -> pd.DataFrame:
    """Build next-open to h-market-day-later-open schedules."""
    required = {"signal_date", "entry_date"}
    if missing := required.difference(rebalance_calendar.columns):
        raise KeyError(f"rebalance calendar is missing columns: {sorted(missing)}")
    horizon_values = tuple(horizons)
    if not horizon_values or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in horizon_values
    ):
        raise ValueError("horizons must be positive integers")
    if len(set(horizon_values)) != len(horizon_values):
        raise ValueError("horizons must be unique")

    calendar = pd.DatetimeIndex(pd.to_datetime(list(trading_dates), errors="raise"))
    calendar = calendar.normalize().drop_duplicates().sort_values()
    if calendar.empty:
        raise ValueError("trading_dates cannot be empty")

    base = rebalance_calendar.copy()
    for column in required:
        base[column] = _normalize_dates(base[column])
    if base.duplicated("signal_date").any():
        raise ValueError("rebalance calendar contains duplicate signal dates")
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize() if end is not None else None
    in_partition = base["signal_date"].ge(start_date)
    if end_date is not None:
        in_partition &= base["signal_date"].le(end_date)
    base = base.loc[in_partition, ["signal_date", "entry_date"]].copy()

    entry_positions = calendar.get_indexer(base["entry_date"])
    invalid_entries = base["entry_date"].notna().to_numpy() & (entry_positions < 0)
    if invalid_entries.any():
        raise ValueError("entry dates contain values outside the trading calendar")

    frames: list[pd.DataFrame] = []
    for horizon in horizon_values:
        schedule = base.copy()
        schedule["horizon_days"] = horizon
        exit_positions = entry_positions + horizon
        valid_exit = (entry_positions >= 0) & (exit_positions < len(calendar))
        schedule["exit_date"] = pd.NaT
        schedule.loc[valid_exit, "exit_date"] = calendar.take(
            exit_positions[valid_exit]
        ).to_numpy()
        schedule["label_end_date"] = schedule["exit_date"]
        within = (
            schedule["entry_date"].notna()
            & schedule["exit_date"].notna()
            & schedule["entry_date"].gt(schedule["signal_date"])
        )
        if end_date is not None:
            within &= schedule["exit_date"].le(end_date)
        schedule["sample_partition"] = sample_partition
        schedule["is_label_within_partition"] = within
        schedule["label_exclusion_reason"] = np.where(
            within, "eligible", "partition_boundary"
        )
        frames.append(schedule)

    columns = [
        "signal_date",
        "horizon_days",
        "entry_date",
        "exit_date",
        "label_end_date",
        "sample_partition",
        "is_label_within_partition",
        "label_exclusion_reason",
    ]
    return (
        pd.concat(frames, ignore_index=True)[columns]
        .sort_values(["signal_date", "horizon_days"])
        .reset_index(drop=True)
    )


def enforce_partition_access(
    sample_partition: str,
    *,
    unlock_holdout: bool,
) -> None:
    """Prevent accidental materialization of the sealed final holdout."""
    if sample_partition == "holdout_2025_onward" and not unlock_holdout:
        raise PermissionError(
            "holdout materialization is locked; pass an explicit unlock only at the final gate"
        )


def build_forward_return_panel(
    signal_keys: pd.DataFrame,
    schedule: pd.DataFrame,
    execution_market: pd.DataFrame,
) -> pd.DataFrame:
    """Attach achievable O2O labels while retaining every explicit exclusion."""
    required_keys = {"signal_date", "ts_code"}
    if missing := required_keys.difference(signal_keys.columns):
        raise KeyError(f"signal keys are missing columns: {sorted(missing)}")
    required_schedule = {
        "signal_date",
        "entry_date",
        "exit_date",
        "label_end_date",
        "is_label_within_partition",
        "label_exclusion_reason",
    }
    if missing := required_schedule.difference(schedule.columns):
        raise KeyError(f"schedule is missing columns: {sorted(missing)}")
    required_market = {"trade_date", "ts_code", "open", "buy_allowed", "sell_allowed"}
    if missing := required_market.difference(execution_market.columns):
        raise KeyError(f"execution market is missing columns: {sorted(missing)}")

    keys = signal_keys[["signal_date", "ts_code"]].copy()
    keys["signal_date"] = _normalize_dates(keys["signal_date"])
    keys["ts_code"] = keys["ts_code"].astype("string")
    if keys.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError("signal keys contain duplicates")

    dates = schedule.copy()
    for column in ("signal_date", "entry_date", "exit_date", "label_end_date"):
        dates[column] = _normalize_dates(dates[column])
    if dates.duplicated("signal_date").any():
        raise ValueError("schedule contains duplicate signal dates")

    market = execution_market[list(required_market)].copy()
    market["trade_date"] = _normalize_dates(market["trade_date"])
    market["ts_code"] = market["ts_code"].astype("string")
    market["open"] = pd.to_numeric(market["open"], errors="coerce")
    if market.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("execution market contains duplicate date and bond keys")

    result = keys.merge(dates, on="signal_date", how="left", validate="many_to_one")
    entry = market.rename(
        columns={
            "trade_date": "entry_date",
            "open": "entry_open",
            "buy_allowed": "entry_buy_allowed",
            "sell_allowed": "entry_sell_allowed",
        }
    )
    result = result.merge(
        entry[["entry_date", "ts_code", "entry_open", "entry_buy_allowed"]],
        on=["entry_date", "ts_code"],
        how="left",
        validate="many_to_one",
    )
    exit_market = market.rename(
        columns={
            "trade_date": "exit_date",
            "open": "exit_open",
            "buy_allowed": "exit_buy_allowed",
            "sell_allowed": "exit_sell_allowed",
        }
    )
    result = result.merge(
        exit_market[["exit_date", "ts_code", "exit_open", "exit_sell_allowed"]],
        on=["exit_date", "ts_code"],
        how="left",
        validate="many_to_one",
    )

    boundary = ~result["is_label_within_partition"].fillna(False)
    missing_entry = ~boundary & result["entry_open"].isna()
    invalid_entry = ~boundary & ~missing_entry & ~_valid_price(result["entry_open"])
    blocked_entry = (
        ~boundary
        & ~missing_entry
        & ~invalid_entry
        & ~result["entry_buy_allowed"].eq(True)
    )
    missing_exit = (
        ~boundary
        & ~missing_entry
        & ~invalid_entry
        & ~blocked_entry
        & result["exit_open"].isna()
    )
    invalid_exit = (
        ~boundary
        & ~missing_entry
        & ~invalid_entry
        & ~blocked_entry
        & ~missing_exit
        & ~_valid_price(result["exit_open"])
    )
    blocked_exit = (
        ~boundary
        & ~missing_entry
        & ~invalid_entry
        & ~blocked_entry
        & ~missing_exit
        & ~invalid_exit
        & ~result["exit_sell_allowed"].eq(True)
    )
    result["label_exclusion_reason"] = np.select(
        [
            boundary,
            missing_entry,
            invalid_entry,
            blocked_entry,
            missing_exit,
            invalid_exit,
            blocked_exit,
        ],
        [
            "partition_boundary",
            "missing_entry_open",
            "invalid_entry_open",
            "entry_not_buyable",
            "missing_exit_open",
            "invalid_exit_open",
            "exit_not_sellable",
        ],
        default="eligible",
    )
    result["label_usable"] = result["label_exclusion_reason"].eq("eligible")
    result["forward_return"] = np.where(
        result["label_usable"],
        result["exit_open"] / result["entry_open"] - 1.0,
        np.nan,
    )
    return result.sort_values(["signal_date", "ts_code"]).reset_index(drop=True)


def build_multi_horizon_forward_return_panel(
    signal_keys: pd.DataFrame,
    horizon_schedules: pd.DataFrame,
    execution_market: pd.DataFrame,
) -> pd.DataFrame:
    """Build a long-form O2O label panel for multiple prediction horizons."""
    if "horizon_days" not in horizon_schedules:
        raise KeyError("horizon schedules are missing horizon_days")
    schedules = horizon_schedules.copy()
    schedules["horizon_days"] = pd.to_numeric(
        schedules["horizon_days"], errors="raise"
    ).astype(int)
    if schedules.duplicated(["signal_date", "horizon_days"]).any():
        raise ValueError("horizon schedules contain duplicate signal-date horizons")

    frames: list[pd.DataFrame] = []
    for horizon, schedule in schedules.groupby("horizon_days", sort=True):
        labels = build_forward_return_panel(
            signal_keys,
            schedule.drop(columns="horizon_days"),
            execution_market,
        )
        labels["horizon_days"] = int(horizon)
        frames.append(labels)
    if not frames:
        raise ValueError("horizon schedules cannot be empty")
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["signal_date", "horizon_days", "ts_code"])
        .reset_index(drop=True)
    )


def prepare_redemption_events(raw_events: pd.DataFrame) -> pd.DataFrame:
    """Map the licensed redemption table to the lifecycle event contract."""
    source_columns = {source: alias for alias, source in LEGACY_REDEMPTION_COLUMNS.items()}
    required = {
        LEGACY_REDEMPTION_COLUMNS["ts_code"],
        LEGACY_REDEMPTION_COLUMNS["event_status"],
        LEGACY_REDEMPTION_COLUMNS["effective_date"],
    }
    if missing := required.difference(raw_events.columns):
        raise KeyError(f"redemption data are missing columns: {sorted(missing)}")
    result = raw_events.copy()
    for source in source_columns:
        if source not in result:
            result[source] = np.nan
    result = result[list(source_columns)].rename(columns=source_columns)
    result["ts_code"] = result["ts_code"].astype("string")
    result["event_status"] = result["event_status"].astype("string").str.strip()
    for column in (
        "effective_date",
        "source_redeem_date",
        "payment_date",
        "call_reg_date",
    ):
        result[column] = _normalize_dates(result[column])
    result["call_price"] = pd.to_numeric(result["call_price"], errors="coerce")
    result["tax_adjusted_call_price"] = pd.to_numeric(
        result["tax_adjusted_call_price"], errors="coerce"
    )
    result["call_price"] = result["call_price"].fillna(
        result["tax_adjusted_call_price"]
    )
    result["last_trade_date"] = pd.NaT
    result = result.dropna(subset=["ts_code", "effective_date", "event_status"])
    if result.duplicated(["ts_code", "effective_date"]).any():
        raise ValueError("redemption data contain duplicate effective dates")
    columns = [
        "ts_code",
        "effective_date",
        "event_status",
        "last_trade_date",
        "call_reg_date",
        "payment_date",
        "call_price",
        "source_redeem_date",
        "tax_adjusted_call_price",
    ]
    return result[columns].sort_values(["ts_code", "effective_date"]).reset_index(drop=True)


def prepare_lifecycle_terms(raw_terms: pd.DataFrame) -> pd.DataFrame:
    """Normalize bond terms without inventing an unavailable payment date."""
    required = {
        "ts_code",
        "list_date",
        "delist_date",
        "maturity_date",
        "maturity_call_price",
    }
    if missing := required.difference(raw_terms.columns):
        raise KeyError(f"bond terms are missing columns: {sorted(missing)}")
    result = raw_terms[list(required)].copy()
    result["ts_code"] = result["ts_code"].astype("string").str.lstrip("\ufeff")
    for column in ("list_date", "delist_date", "maturity_date"):
        result[column] = _normalize_dates(result[column])
    result["maturity_call_price"] = pd.to_numeric(
        result["maturity_call_price"], errors="coerce"
    )
    result["maturity_payment_date"] = pd.NaT
    result = result.dropna(subset=["ts_code", "list_date", "maturity_date"])
    if result.duplicated("ts_code").any():
        raise ValueError("bond terms contain duplicate ts_code")
    return result.sort_values("ts_code").reset_index(drop=True)
