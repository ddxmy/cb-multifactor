"""Point-in-time daily trading factors for convertible bonds."""

from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd

from .event_state import attach_asof_event_state


RETURN_WINDOWS = (5, 10, 20)
TURNOVER_WINDOWS = (20, 60)
RISK_WINDOW = 20
VWAP_MEAN_WINDOW = 5
MINIMUM_OBSERVATION_RATIO = 0.8
AMIHUD_SCALE = 1e8
VWAP_RANGE_TOLERANCE = 0.02
TUSHARE_CB_DAILY_AMOUNT_CNY_PER_REPORTED_UNIT = 10_000.0
TUSHARE_CB_DAILY_BONDS_PER_HAND = 10.0
TUSHARE_CB_DAILY_VWAP_CNY_PER_HAND = (
    TUSHARE_CB_DAILY_AMOUNT_CNY_PER_REPORTED_UNIT
    / TUSHARE_CB_DAILY_BONDS_PER_HAND
)

CB_TRADING_FACTOR_COLUMNS = [
    "cb_return_5d",
    "cb_return_10d",
    "cb_return_20d",
    "cb_short_long_momentum_5_20d",
    "cb_daily_turnover",
    "cb_abnormal_turnover_20d",
    "cb_abnormal_turnover_60d",
    "cb_log_amount_zscore_20d",
    "cb_amihud_20d",
    "cb_close_to_vwap",
    "cb_close_to_vwap_mean_5d",
]


def _normalize_dates(values: pd.Series) -> pd.Series:
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        return pd.to_datetime(values).dt.tz_localize(None).dt.normalize()
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values).dt.normalize()
    text = values.astype(str).str.replace(r"\.0$", "", regex=True)
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text.loc[missing], errors="coerce")
    return parsed.dt.normalize()


def _prepare_cb_daily(cb_daily: pd.DataFrame) -> pd.DataFrame:
    required = {
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
    }
    if missing := required.difference(cb_daily.columns):
        raise KeyError(f"Convertible-bond daily data is missing columns: {sorted(missing)}")

    daily = cb_daily[list(required)].copy()
    daily["_has_market_observation"] = True
    daily["ts_code"] = daily["ts_code"].astype(str).str.lstrip("\ufeff")
    daily["trade_date"] = _normalize_dates(daily["trade_date"])
    for column in required.difference({"ts_code", "trade_date"}):
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    if daily.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("Convertible-bond daily data has duplicate bond/date rows")
    return daily.dropna(subset=["trade_date"]).sort_values(
        ["ts_code", "trade_date"]
    )


def _prepare_reference(reference_universe: pd.DataFrame) -> pd.DataFrame:
    required = {"ts_code", "list_date", "issue_size"}
    if missing := required.difference(reference_universe.columns):
        raise KeyError(f"Reference universe is missing columns: {sorted(missing)}")
    reference = reference_universe[list(required)].copy()
    reference["ts_code"] = reference["ts_code"].astype(str).str.lstrip("\ufeff")
    reference["list_date"] = _normalize_dates(reference["list_date"])
    reference["issue_size"] = pd.to_numeric(reference["issue_size"], errors="coerce")
    if reference.duplicated("ts_code").any():
        raise ValueError("Reference universe has duplicate bond codes")
    return reference


def _prepare_share_events(share_events: pd.DataFrame) -> pd.DataFrame:
    required = {"ts_code", "effective_date", "remain_size"}
    if missing := required.difference(share_events.columns):
        raise KeyError(f"Share events are missing columns: {sorted(missing)}")
    events = share_events[list(required)].copy()
    events["ts_code"] = events["ts_code"].astype(str).str.lstrip("\ufeff")
    events["effective_date"] = _normalize_dates(events["effective_date"])
    events["remain_size"] = pd.to_numeric(events["remain_size"], errors="coerce")
    events = events.dropna(subset=["effective_date", "remain_size"])
    return events.sort_values(["effective_date", "ts_code"]).drop_duplicates(
        ["ts_code", "effective_date"], keep="last"
    )


def _rolling_group_stat(
    values: pd.Series,
    *,
    window: int,
    minimum_observations: int,
    statistic: str,
) -> pd.Series:
    rolling = values.groupby(level="ts_code", sort=False).rolling(
        window,
        min_periods=minimum_observations,
    )
    if statistic == "mean":
        result = rolling.mean()
    elif statistic == "std":
        result = rolling.std(ddof=1)
    elif statistic == "count":
        result = rolling.count()
    else:
        raise ValueError(f"Unsupported rolling statistic: {statistic}")
    return result.droplevel(0)


def _build_daily_factor_history(
    cb_daily: pd.DataFrame,
    reference_universe: pd.DataFrame,
    share_events: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    bond_codes: list[str],
    *,
    last_signal_date: pd.Timestamp,
    return_windows: tuple[int, ...],
    turnover_windows: tuple[int, ...],
    risk_window: int,
    vwap_mean_window: int,
    minimum_observation_ratio: float,
) -> pd.DataFrame:
    daily = cb_daily.loc[
        cb_daily["ts_code"].isin(bond_codes)
        & cb_daily["trade_date"].le(last_signal_date)
    ].copy()
    if daily.empty:
        raise ValueError("Convertible-bond daily history is empty for the selected universe")
    first_history_date = daily["trade_date"].min()
    calendar = trading_dates[
        trading_dates.to_series().between(first_history_date, last_signal_date).to_numpy()
    ]
    index = pd.MultiIndex.from_product(
        [bond_codes, calendar], names=["ts_code", "trade_date"]
    )
    history = daily.set_index(["ts_code", "trade_date"]).reindex(index).reset_index()
    history = history.merge(
        reference_universe,
        on="ts_code",
        how="left",
        validate="m:1",
    )
    history = attach_asof_event_state(
        history,
        share_events,
        ["remain_size"],
        effective_date_output="share_effective_date",
    )
    history["remain_size"] = pd.to_numeric(history["remain_size"], errors="coerce")
    history["remain_size"] = history["remain_size"].fillna(history["issue_size"])
    history["is_listed"] = history["trade_date"].ge(history["list_date"])
    history["has_market_observation"] = history.pop(
        "_has_market_observation"
    ).notna()
    vwap_notional_cny_per_hand = (
        history["amount"] * TUSHARE_CB_DAILY_VWAP_CNY_PER_HAND
    )
    vwap_candidate = (vwap_notional_cny_per_hand / history["vol"]).where(
        history["vol"].gt(0.0) & vwap_notional_cny_per_hand.gt(0.0)
    )
    history["vwap_unit_valid"] = (
        vwap_candidate.notna()
        & history["low"].gt(0.0)
        & history["high"].gt(0.0)
        & vwap_candidate.ge(history["low"] - VWAP_RANGE_TOLERANCE)
        & vwap_candidate.le(history["high"] + VWAP_RANGE_TOLERANCE)
    )
    observed_zero_trade = (
        history["has_market_observation"]
        & history["vol"].eq(0.0)
        & history["amount"].eq(0.0)
    )
    inconsistent_trade = (
        history["has_market_observation"]
        & ~history["vwap_unit_valid"]
        & ~observed_zero_trade
    )
    listed_volume = history["vol"].where(
        history["is_listed"] & history["has_market_observation"]
    )
    listed_volume = listed_volume.mask(inconsistent_trade)
    history["daily_turnover"] = (
        listed_volume * 1_000.0 / history["remain_size"]
    ).where(history["remain_size"].gt(0.0))
    history["daily_return"] = (history["pct_chg"] / 100.0).where(
        history["has_market_observation"]
    )
    history["vwap"] = vwap_candidate.where(history["vwap_unit_valid"])
    history["close_to_vwap"] = (
        history["close"] / history["vwap"] - 1.0
    ).where(history["vwap"].gt(0.0))

    history = history.sort_values(["ts_code", "trade_date"]).set_index(
        ["ts_code", "trade_date"]
    )
    actual_amount_cny = (
        history["amount"] * TUSHARE_CB_DAILY_AMOUNT_CNY_PER_REPORTED_UNIT
    ).where(
        history["vwap_unit_valid"]
    )
    grouped = history.groupby(level="ts_code", sort=False)
    for window in return_windows:
        previous_close = grouped["close"].shift(window)
        observations = (
            grouped["close"]
            .rolling(window + 1, min_periods=1)
            .count()
            .droplevel(0)
        )
        required = ceil((window + 1) * minimum_observation_ratio)
        history[f"cb_return_{window}d"] = (
            history["close"] / previous_close - 1.0
        ).where(observations.ge(required))
    history["cb_short_long_momentum_5_20d"] = (
        history["cb_return_5d"] - history["cb_return_20d"]
    )

    prior_turnover = grouped["daily_turnover"].shift(1)
    for window in turnover_windows:
        required = ceil(window * minimum_observation_ratio)
        prior_mean = _rolling_group_stat(
            prior_turnover,
            window=window,
            minimum_observations=required,
            statistic="mean",
        )
        history[f"cb_abnormal_turnover_{window}d"] = (
            history["daily_turnover"] / prior_mean
        ).where(prior_mean.gt(0.0))

    valid_log_amount = np.log(actual_amount_cny.where(actual_amount_cny.gt(0.0)))
    prior_log_amount = valid_log_amount.groupby(level="ts_code", sort=False).shift(1)
    minimum_risk_observations = ceil(risk_window * minimum_observation_ratio)
    prior_amount_mean = _rolling_group_stat(
        prior_log_amount,
        window=risk_window,
        minimum_observations=minimum_risk_observations,
        statistic="mean",
    )
    prior_amount_std = _rolling_group_stat(
        prior_log_amount,
        window=risk_window,
        minimum_observations=minimum_risk_observations,
        statistic="std",
    )
    history["cb_log_amount_zscore_20d"] = (
        (valid_log_amount - prior_amount_mean) / prior_amount_std
    ).where(prior_amount_std.gt(0.0))

    daily_amihud = (history["daily_return"].abs() / actual_amount_cny).where(
        actual_amount_cny.gt(0.0)
    )
    history["cb_amihud_20d"] = (
        _rolling_group_stat(
            daily_amihud,
            window=risk_window,
            minimum_observations=minimum_risk_observations,
            statistic="mean",
        )
        * AMIHUD_SCALE
    ).where(actual_amount_cny.notna())
    minimum_vwap_observations = ceil(
        vwap_mean_window * minimum_observation_ratio
    )
    history["cb_close_to_vwap_mean_5d"] = (
        _rolling_group_stat(
            history["close_to_vwap"],
            window=vwap_mean_window,
            minimum_observations=minimum_vwap_observations,
            statistic="mean",
        ).where(history["close_to_vwap"].notna())
    )
    history["cb_daily_turnover"] = history["daily_turnover"]
    history["cb_close_to_vwap"] = history["close_to_vwap"]
    history["cb_vwap"] = history["vwap"]
    history["cb_vwap_unit_valid"] = history["vwap_unit_valid"].astype(bool)
    return history.reset_index()


def build_cb_trading_factor_panel(
    universe_panel: pd.DataFrame,
    cb_daily: pd.DataFrame,
    reference_universe: pd.DataFrame,
    share_events: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    *,
    return_windows: tuple[int, ...] = RETURN_WINDOWS,
    turnover_windows: tuple[int, ...] = TURNOVER_WINDOWS,
    risk_window: int = RISK_WINDOW,
    vwap_mean_window: int = VWAP_MEAN_WINDOW,
    minimum_observation_ratio: float = MINIMUM_OBSERVATION_RATIO,
) -> pd.DataFrame:
    """Build daily CB trading factors for investable signal-date observations."""
    required_universe = {"signal_date", "ts_code", "is_eligible"}
    if missing := required_universe.difference(universe_panel.columns):
        raise KeyError(f"Universe panel is missing columns: {sorted(missing)}")
    if not return_windows or any(window <= 0 for window in return_windows):
        raise ValueError("return_windows must contain positive integers")
    if not turnover_windows or any(window <= 0 for window in turnover_windows):
        raise ValueError("turnover_windows must contain positive integers")
    if risk_window <= 1 or vwap_mean_window <= 0:
        raise ValueError("Rolling windows must be positive and risk_window must exceed one")
    if not 0.0 < minimum_observation_ratio <= 1.0:
        raise ValueError("minimum_observation_ratio must be in (0, 1]")

    investable = universe_panel.loc[
        universe_panel["is_eligible"], ["signal_date", "ts_code"]
    ].copy()
    investable["signal_date"] = _normalize_dates(investable["signal_date"])
    investable["ts_code"] = investable["ts_code"].astype(str).str.lstrip("\ufeff")
    if investable.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError("Universe panel has duplicate signal-date/bond rows")
    if investable.empty:
        return pd.DataFrame(
            columns=["signal_date", "ts_code", *CB_TRADING_FACTOR_COLUMNS]
        )

    daily = _prepare_cb_daily(cb_daily)
    reference = _prepare_reference(reference_universe)
    events = _prepare_share_events(share_events)
    calendar = pd.DatetimeIndex(pd.to_datetime(trading_dates).dropna().unique()).sort_values()
    bond_codes = sorted(investable["ts_code"].unique())
    history = _build_daily_factor_history(
        daily,
        reference.loc[reference["ts_code"].isin(bond_codes)],
        events.loc[events["ts_code"].isin(bond_codes)],
        calendar,
        bond_codes,
        last_signal_date=investable["signal_date"].max(),
        return_windows=return_windows,
        turnover_windows=turnover_windows,
        risk_window=risk_window,
        vwap_mean_window=vwap_mean_window,
        minimum_observation_ratio=minimum_observation_ratio,
    )
    signal_history = history.rename(columns={"trade_date": "signal_date"})
    source_columns = [
        "remain_size",
        "close",
        "high",
        "low",
        "vol",
        "amount",
        "cb_vwap",
        "cb_vwap_unit_valid",
    ]
    panel = investable.merge(
        signal_history[
            ["signal_date", "ts_code", *source_columns, *CB_TRADING_FACTOR_COLUMNS]
        ],
        on=["signal_date", "ts_code"],
        how="left",
        validate="1:1",
    ).rename(
        columns={
            "close": "cb_close",
            "high": "cb_high",
            "low": "cb_low",
            "vol": "cb_vol",
            "amount": "cb_amount",
        }
    )
    output_columns = [
        "signal_date",
        "ts_code",
        "remain_size",
        "cb_close",
        "cb_high",
        "cb_low",
        "cb_vol",
        "cb_amount",
        "cb_vwap",
        "cb_vwap_unit_valid",
        *CB_TRADING_FACTOR_COLUMNS,
    ]
    return panel[output_columns].sort_values(
        ["signal_date", "ts_code"]
    ).reset_index(drop=True)


def audit_cb_trading_factor_panel(
    factor_panel: pd.DataFrame,
    universe_panel: pd.DataFrame,
    cb_daily: pd.DataFrame,
    reference_universe: pd.DataFrame,
    share_events: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    *,
    minimum_vwap_range_rate: float = 0.999,
    return_windows: tuple[int, ...] = RETURN_WINDOWS,
    turnover_windows: tuple[int, ...] = TURNOVER_WINDOWS,
    risk_window: int = RISK_WINDOW,
    vwap_mean_window: int = VWAP_MEAN_WINDOW,
    minimum_observation_ratio: float = MINIMUM_OBSERVATION_RATIO,
) -> dict[str, object]:
    """Audit keys, formulas, factor coverage, and the inferred VWAP unit rule."""
    required = {
        "signal_date",
        "ts_code",
        "cb_close",
        "cb_high",
        "cb_low",
        "cb_vol",
        "cb_amount",
        "cb_vwap",
        "cb_vwap_unit_valid",
        *CB_TRADING_FACTOR_COLUMNS,
    }
    if missing := required.difference(factor_panel.columns):
        raise KeyError(f"Factor panel is missing columns: {sorted(missing)}")

    actual = factor_panel.copy()
    actual["signal_date"] = _normalize_dates(actual["signal_date"])
    expected = build_cb_trading_factor_panel(
        universe_panel,
        cb_daily,
        reference_universe,
        share_events,
        trading_dates,
        return_windows=return_windows,
        turnover_windows=turnover_windows,
        risk_window=risk_window,
        vwap_mean_window=vwap_mean_window,
        minimum_observation_ratio=minimum_observation_ratio,
    )
    expected_keys = set(map(tuple, expected[["signal_date", "ts_code"]].to_numpy()))
    actual_keys = set(map(tuple, actual[["signal_date", "ts_code"]].to_numpy()))
    duplicate_keys = int(actual.duplicated(["signal_date", "ts_code"]).sum())
    comparison = actual.merge(
        expected,
        on=["signal_date", "ts_code"],
        how="inner",
        suffixes=("_actual", "_expected"),
    )
    formula_columns = ["cb_vwap", *CB_TRADING_FACTOR_COLUMNS]
    mismatch_count = 0
    for column in formula_columns:
        actual_values = comparison[f"{column}_actual"]
        expected_values = comparison[f"{column}_expected"]
        mismatch_count += int(actual_values.isna().ne(expected_values.isna()).sum())
        comparable = actual_values.notna() & expected_values.notna()
        mismatch_count += int(
            (~np.isclose(
                actual_values.loc[comparable],
                expected_values.loc[comparable],
                rtol=1e-10,
                atol=1e-10,
            )).sum()
        )

    mismatch_count += int(
        comparison["cb_vwap_unit_valid_actual"]
        .ne(comparison["cb_vwap_unit_valid_expected"])
        .sum()
    )

    valid_vwap = (
        actual["cb_vol"].gt(0.0)
        & actual["cb_amount"].gt(0.0)
        & actual["cb_low"].gt(0.0)
        & actual["cb_high"].gt(0.0)
    )
    raw_vwap_candidate = (
        actual["cb_amount"] * TUSHARE_CB_DAILY_VWAP_CNY_PER_HAND / actual["cb_vol"]
    )
    inside_range = (
        raw_vwap_candidate.ge(actual["cb_low"] - VWAP_RANGE_TOLERANCE)
        & raw_vwap_candidate.le(actual["cb_high"] + VWAP_RANGE_TOLERANCE)
    )
    vwap_rate = float(inside_range.loc[valid_vwap].mean()) if valid_vwap.any() else 0.0
    audit = {
        "row_count": int(len(actual)),
        "signal_date_count": int(actual["signal_date"].nunique()),
        "duplicate_signal_bond_keys": duplicate_keys,
        "missing_expected_keys": int(len(expected_keys - actual_keys)),
        "extra_unexpected_keys": int(len(actual_keys - expected_keys)),
        "formula_mismatch_count": int(mismatch_count),
        "vwap_valid_count": int(valid_vwap.sum()),
        "vwap_inside_daily_range_count": int((valid_vwap & inside_range).sum()),
        "vwap_excluded_count": int((valid_vwap & ~inside_range).sum()),
        "vwap_inside_daily_range_rate": vwap_rate,
        "vwap_range_check_passed": bool(vwap_rate >= minimum_vwap_range_rate),
        "factor_coverage": {
            column: float(actual[column].notna().mean()) if len(actual) else 0.0
            for column in CB_TRADING_FACTOR_COLUMNS
        },
    }
    audit["status"] = "pass" if (
        audit["duplicate_signal_bond_keys"] == 0
        and audit["missing_expected_keys"] == 0
        and audit["extra_unexpected_keys"] == 0
        and audit["formula_mismatch_count"] == 0
        and audit["vwap_range_check_passed"]
    ) else "fail"
    return audit
