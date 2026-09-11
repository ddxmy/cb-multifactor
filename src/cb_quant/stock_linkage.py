"""Point-in-time underlying-stock and stock-bond linkage factors."""

from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd


RETURN_WINDOWS = (5, 10, 20)
RISK_WINDOW = 20
LINKAGE_WINDOWS = (20, 60)
MINIMUM_OBSERVATION_RATIO = 0.8
AMIHUD_SCALE = 1e8
BOLLINGER_STANDARD_DEVIATIONS = 2.0

STOCK_FACTOR_COLUMNS = [
    "stock_return_5d",
    "stock_return_10d",
    "stock_return_20d",
    "stock_volatility_20d",
    "stock_rsi_20d",
    "stock_price_to_high_20d",
    "stock_percent_b_20d",
    "stock_amihud_20d",
    "stock_mfi_20d",
]
LINKAGE_FACTOR_COLUMNS = [
    "cb_stock_return_spread_5d",
    "cb_stock_return_spread_10d",
    "cb_stock_return_spread_20d",
    "cb_stock_correlation_20d",
    "cb_stock_correlation_60d",
    "cb_stock_beta_20d",
    "cb_stock_beta_60d",
]
STOCK_LINKAGE_FACTOR_COLUMNS = [*STOCK_FACTOR_COLUMNS, *LINKAGE_FACTOR_COLUMNS]


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


def _prepare_stock_daily(stock_daily: pd.DataFrame) -> pd.DataFrame:
    required = {
        "ts_code",
        "trade_date",
        "close",
        "high",
        "low",
        "pre_close",
        "pct_chg",
        "vol",
        "amount",
        "adj_factor",
    }
    if missing := required.difference(stock_daily.columns):
        raise KeyError(f"Stock daily data is missing columns: {sorted(missing)}")

    stock = stock_daily[list(required)].copy()
    stock["ts_code"] = stock["ts_code"].astype(str).str.lstrip("\ufeff")
    stock["trade_date"] = _normalize_dates(stock["trade_date"])
    for column in required.difference({"ts_code", "trade_date"}):
        stock[column] = pd.to_numeric(stock[column], errors="coerce")
    if stock.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("Stock daily data has duplicate stock/date rows")
    return stock.dropna(subset=["trade_date"])


def _prepare_cb_daily(cb_daily: pd.DataFrame) -> pd.DataFrame:
    required = {"ts_code", "trade_date", "close"}
    if missing := required.difference(cb_daily.columns):
        raise KeyError(f"Convertible-bond daily data is missing columns: {sorted(missing)}")

    columns = ["ts_code", "trade_date", "close"]
    if "pct_chg" in cb_daily.columns:
        columns.append("pct_chg")
    bond = cb_daily[columns].copy()
    bond["ts_code"] = bond["ts_code"].astype(str).str.lstrip("\ufeff")
    bond["trade_date"] = _normalize_dates(bond["trade_date"])
    bond["close"] = pd.to_numeric(bond["close"], errors="coerce")
    if "pct_chg" in bond:
        bond["pct_chg"] = pd.to_numeric(bond["pct_chg"], errors="coerce")
    if bond.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("Convertible-bond daily data has duplicate bond/date rows")
    bond = bond.dropna(subset=["trade_date"]).sort_values(
        ["ts_code", "trade_date"]
    )
    observed_return = bond.groupby("ts_code", sort=False)["close"].pct_change(
        fill_method=None
    )
    if "pct_chg" in bond:
        bond["daily_return"] = (bond["pct_chg"] / 100.0).combine_first(
            observed_return
        )
    else:
        bond["daily_return"] = observed_return
    return bond


def _complete_history_grid(
    frame: pd.DataFrame,
    *,
    codes: list[str],
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [codes, trading_dates], names=["ts_code", "trade_date"]
    )
    return frame.set_index(["ts_code", "trade_date"]).reindex(index).sort_index()


def _build_stock_history_factors(
    stock_daily: pd.DataFrame,
    *,
    stock_codes: list[str],
    trading_dates: pd.DatetimeIndex,
    return_windows: tuple[int, ...],
    risk_window: int,
    minimum_observation_ratio: float,
) -> pd.DataFrame:
    stock = stock_daily.loc[stock_daily["ts_code"].isin(stock_codes)].copy()
    stock = _complete_history_grid(
        stock,
        codes=stock_codes,
        trading_dates=trading_dates,
    )
    stock["adjusted_close"] = stock["close"] * stock["adj_factor"]
    stock["daily_return"] = stock["pct_chg"] / 100.0
    stock["adjusted_change"] = (
        (stock["close"] - stock["pre_close"]) * stock["adj_factor"]
    )
    stock["adjusted_typical_price"] = (
        (stock["high"] + stock["low"] + stock["close"])
        / 3.0
        * stock["adj_factor"]
    )
    grouped = stock.groupby(level="ts_code", sort=False)

    for window in return_windows:
        previous = grouped["adjusted_close"].shift(window)
        observations = grouped["adjusted_close"].rolling(
            window + 1,
            min_periods=1,
        ).count().droplevel(0)
        required = ceil((window + 1) * minimum_observation_ratio)
        stock[f"stock_return_{window}d"] = (
            stock["adjusted_close"] / previous - 1.0
        ).where(observations.ge(required))

    minimum_risk_observations = ceil(risk_window * minimum_observation_ratio)
    stock["stock_volatility_20d"] = (
        grouped["daily_return"]
        .rolling(risk_window, min_periods=minimum_risk_observations)
        .std(ddof=1)
        .droplevel(0)
        * np.sqrt(252.0)
    )

    positive_change = stock["adjusted_change"].clip(lower=0.0)
    absolute_change = stock["adjusted_change"].abs()
    positive_sum = (
        positive_change.groupby(level="ts_code", sort=False)
        .rolling(risk_window, min_periods=minimum_risk_observations)
        .sum()
        .droplevel(0)
    )
    absolute_sum = (
        absolute_change.groupby(level="ts_code", sort=False)
        .rolling(risk_window, min_periods=minimum_risk_observations)
        .sum()
        .droplevel(0)
    )
    stock["stock_rsi_20d"] = positive_sum / absolute_sum * 100.0
    flat_window = absolute_sum.eq(0.0) & absolute_sum.notna()
    stock.loc[flat_window, "stock_rsi_20d"] = 50.0

    rolling_high = (
        grouped["adjusted_close"]
        .rolling(risk_window, min_periods=minimum_risk_observations)
        .max()
        .droplevel(0)
    )
    stock["stock_price_to_high_20d"] = stock["adjusted_close"] / rolling_high

    rolling_middle = (
        grouped["adjusted_close"]
        .rolling(risk_window, min_periods=minimum_risk_observations)
        .mean()
        .droplevel(0)
    )
    rolling_standard_deviation = (
        grouped["adjusted_close"]
        .rolling(risk_window, min_periods=minimum_risk_observations)
        .std(ddof=0)
        .droplevel(0)
    )
    lower_band = (
        rolling_middle
        - BOLLINGER_STANDARD_DEVIATIONS * rolling_standard_deviation
    )
    band_width = (
        2.0 * BOLLINGER_STANDARD_DEVIATIONS * rolling_standard_deviation
    )
    stock["stock_percent_b_20d"] = (
        (stock["adjusted_close"] - lower_band) / band_width
    ).where(band_width.gt(0.0))

    amount_yuan = stock["amount"] * 1_000.0
    daily_amihud = (stock["daily_return"].abs() / amount_yuan).where(
        amount_yuan.gt(0.0)
    )
    stock["stock_amihud_20d"] = (
        daily_amihud.groupby(level="ts_code", sort=False)
        .rolling(risk_window, min_periods=minimum_risk_observations)
        .mean()
        .droplevel(0)
        * AMIHUD_SCALE
    )

    typical_change = grouped["adjusted_typical_price"].diff()
    raw_money_flow = (
        stock["adjusted_typical_price"] * stock["vol"]
    ).where(stock["vol"].gt(0.0))
    valid_money_flow = raw_money_flow.notna() & typical_change.notna()
    positive_money_flow = raw_money_flow.where(
        typical_change.gt(0.0), 0.0
    ).where(valid_money_flow)
    negative_money_flow = raw_money_flow.where(
        typical_change.lt(0.0), 0.0
    ).where(valid_money_flow)
    positive_sum = (
        positive_money_flow.groupby(level="ts_code", sort=False)
        .rolling(risk_window, min_periods=minimum_risk_observations)
        .sum()
        .droplevel(0)
    )
    negative_sum = (
        negative_money_flow.groupby(level="ts_code", sort=False)
        .rolling(risk_window, min_periods=minimum_risk_observations)
        .sum()
        .droplevel(0)
    )
    total_money_flow = positive_sum + negative_sum
    stock["stock_mfi_20d"] = positive_sum / total_money_flow * 100.0
    no_directional_flow = total_money_flow.eq(0.0) & total_money_flow.notna()
    stock.loc[no_directional_flow, "stock_mfi_20d"] = 50.0
    return stock.reset_index()


def _build_cb_return_factors(
    cb_daily: pd.DataFrame,
    *,
    bond_codes: list[str],
    trading_dates: pd.DatetimeIndex,
    return_windows: tuple[int, ...],
    minimum_observation_ratio: float,
) -> pd.DataFrame:
    bond = cb_daily.loc[cb_daily["ts_code"].isin(bond_codes)].copy()
    bond = _complete_history_grid(
        bond,
        codes=bond_codes,
        trading_dates=trading_dates,
    )
    grouped = bond.groupby(level="ts_code", sort=False)
    for window in return_windows:
        previous = grouped["close"].shift(window)
        observations = grouped["close"].rolling(
            window + 1,
            min_periods=1,
        ).count().droplevel(0)
        required = ceil((window + 1) * minimum_observation_ratio)
        bond[f"cb_return_{window}d"] = (
            bond["close"] / previous - 1.0
        ).where(observations.ge(required))
    return bond.reset_index()


def _rolling_group_sum(
    values: pd.Series,
    groups: pd.Series,
    *,
    window: int,
) -> pd.Series:
    return (
        values.groupby(groups, sort=False)
        .rolling(window, min_periods=1)
        .sum()
        .droplevel(0)
    )


def _build_stock_bond_linkage_state(
    stock_history: pd.DataFrame,
    cb_history: pd.DataFrame,
    bond_stock_map: pd.DataFrame,
    *,
    linkage_windows: tuple[int, ...],
    minimum_observation_ratio: float,
) -> pd.DataFrame:
    stock_returns = stock_history[["ts_code", "trade_date", "daily_return"]].rename(
        columns={"ts_code": "stk_code", "daily_return": "stock_daily_return"}
    )
    cb_returns = cb_history[["ts_code", "trade_date", "daily_return"]].rename(
        columns={"daily_return": "cb_daily_return"}
    )
    pairs = cb_returns.merge(
        bond_stock_map,
        on="ts_code",
        how="left",
        validate="m:1",
    ).merge(
        stock_returns,
        on=["stk_code", "trade_date"],
        how="left",
        validate="m:1",
    ).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    valid_pair = pairs["stock_daily_return"].notna() & pairs["cb_daily_return"].notna()
    stock_return = pairs["stock_daily_return"].where(valid_pair)
    cb_return = pairs["cb_daily_return"].where(valid_pair)
    groups = pairs["ts_code"]
    for window in linkage_windows:
        count = _rolling_group_sum(
            valid_pair.astype(float), groups, window=window
        )
        stock_sum = _rolling_group_sum(stock_return, groups, window=window)
        cb_sum = _rolling_group_sum(cb_return, groups, window=window)
        stock_square_sum = _rolling_group_sum(
            stock_return.pow(2), groups, window=window
        )
        cb_square_sum = _rolling_group_sum(
            cb_return.pow(2), groups, window=window
        )
        cross_sum = _rolling_group_sum(
            stock_return * cb_return, groups, window=window
        )
        covariance_numerator = cross_sum - stock_sum * cb_sum / count
        stock_variance_numerator = (
            stock_square_sum - stock_sum.pow(2) / count
        ).clip(lower=0.0)
        cb_variance_numerator = (
            cb_square_sum - cb_sum.pow(2) / count
        ).clip(lower=0.0)
        required = ceil(window * minimum_observation_ratio)
        sufficient = count.ge(required)
        valid_variance = stock_variance_numerator.gt(0.0)
        pairs[f"cb_stock_beta_{window}d"] = (
            covariance_numerator / stock_variance_numerator
        ).where(sufficient & valid_variance)
        pairs[f"cb_stock_correlation_{window}d"] = (
            covariance_numerator
            / np.sqrt(stock_variance_numerator * cb_variance_numerator)
        ).where(
            sufficient
            & valid_variance
            & cb_variance_numerator.gt(0.0)
        ).clip(-1.0, 1.0)
    columns = ["trade_date", "ts_code"]
    for window in linkage_windows:
        columns.extend(
            [
                f"cb_stock_correlation_{window}d",
                f"cb_stock_beta_{window}d",
            ]
        )
    return pairs[columns]


def build_stock_linkage_factor_panel(
    universe_panel: pd.DataFrame,
    stock_daily: pd.DataFrame,
    cb_daily: pd.DataFrame,
    *,
    return_windows: tuple[int, ...] = RETURN_WINDOWS,
    risk_window: int = RISK_WINDOW,
    linkage_windows: tuple[int, ...] = LINKAGE_WINDOWS,
    minimum_observation_ratio: float = MINIMUM_OBSERVATION_RATIO,
) -> pd.DataFrame:
    """Build confirmed stock and stock-bond factors on investable signal dates.

    Return factors use adjusted underlying-stock closes and raw convertible-bond
    closes. Rolling statistics require the configured share of valid market days;
    suspended days remain missing and are never forward-filled.
    """
    required_universe = {"signal_date", "ts_code", "stk_code", "is_eligible"}
    if missing := required_universe.difference(universe_panel.columns):
        raise KeyError(f"Universe panel is missing columns: {sorted(missing)}")
    if not return_windows or any(window <= 0 for window in return_windows):
        raise ValueError("return_windows must contain positive integers")
    if risk_window <= 1:
        raise ValueError("risk_window must be greater than one")
    if not linkage_windows or any(window <= 1 for window in linkage_windows):
        raise ValueError("linkage_windows must contain integers greater than one")
    if not 0.0 < minimum_observation_ratio <= 1.0:
        raise ValueError("minimum_observation_ratio must be in (0, 1]")

    investable = universe_panel.loc[
        universe_panel["is_eligible"],
        ["signal_date", "ts_code", "stk_code"],
    ].copy()
    investable["signal_date"] = _normalize_dates(investable["signal_date"])
    investable["ts_code"] = investable["ts_code"].astype(str).str.lstrip("\ufeff")
    investable["stk_code"] = investable["stk_code"].astype(str).str.lstrip("\ufeff")
    if investable.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError("Universe panel has duplicate signal-date/bond rows")
    if investable.empty:
        return pd.DataFrame(
            columns=["signal_date", "ts_code", "stk_code", *STOCK_LINKAGE_FACTOR_COLUMNS]
        )

    stock = _prepare_stock_daily(stock_daily)
    bond = _prepare_cb_daily(cb_daily)
    last_signal_date = investable["signal_date"].max()
    stock = stock.loc[stock["trade_date"].le(last_signal_date)]
    bond = bond.loc[bond["trade_date"].le(last_signal_date)]
    trading_dates = pd.DatetimeIndex(
        sorted(set(stock["trade_date"]).union(bond["trade_date"]))
    )
    if trading_dates.empty:
        raise ValueError("Stock and convertible-bond histories contain no valid dates")

    stock_history = _build_stock_history_factors(
        stock,
        stock_codes=sorted(investable["stk_code"].dropna().unique()),
        trading_dates=trading_dates,
        return_windows=return_windows,
        risk_window=risk_window,
        minimum_observation_ratio=minimum_observation_ratio,
    )
    stock_factor_columns = [
        f"stock_return_{window}d" for window in return_windows
    ] + [
        "stock_volatility_20d",
        "stock_rsi_20d",
        "stock_price_to_high_20d",
        "stock_percent_b_20d",
        "stock_amihud_20d",
        "stock_mfi_20d",
    ]
    stock_factors = stock_history.rename(
        columns={"ts_code": "stk_code", "trade_date": "signal_date"}
    )[["signal_date", "stk_code", *stock_factor_columns]]

    cb_history = _build_cb_return_factors(
        bond,
        bond_codes=sorted(investable["ts_code"].unique()),
        trading_dates=trading_dates,
        return_windows=return_windows,
        minimum_observation_ratio=minimum_observation_ratio,
    )
    bond_stock_map = investable[["ts_code", "stk_code"]].drop_duplicates()
    if bond_stock_map.duplicated("ts_code").any():
        raise ValueError("A convertible bond maps to multiple underlying stocks")
    linkage_state = _build_stock_bond_linkage_state(
        stock_history,
        cb_history,
        bond_stock_map,
        linkage_windows=linkage_windows,
        minimum_observation_ratio=minimum_observation_ratio,
    ).rename(columns={"trade_date": "signal_date"})
    cb_return_columns = [f"cb_return_{window}d" for window in return_windows]
    cb_factors = cb_history.rename(columns={"trade_date": "signal_date"})[
        ["signal_date", "ts_code", *cb_return_columns]
    ]

    panel = investable.merge(
        stock_factors,
        on=["signal_date", "stk_code"],
        how="left",
        validate="m:1",
    ).merge(
        cb_factors,
        on=["signal_date", "ts_code"],
        how="left",
        validate="1:1",
    ).merge(
        linkage_state,
        on=["signal_date", "ts_code"],
        how="left",
        validate="1:1",
    )
    for window in return_windows:
        panel[f"cb_stock_return_spread_{window}d"] = (
            panel[f"cb_return_{window}d"] - panel[f"stock_return_{window}d"]
        )
    panel = panel.drop(columns=cb_return_columns)
    expected_columns = [
        "signal_date",
        "ts_code",
        "stk_code",
        *[f"stock_return_{window}d" for window in return_windows],
        "stock_volatility_20d",
        "stock_rsi_20d",
        "stock_price_to_high_20d",
        "stock_percent_b_20d",
        "stock_amihud_20d",
        "stock_mfi_20d",
        *[f"cb_stock_return_spread_{window}d" for window in return_windows],
        *[
            column
            for window in linkage_windows
            for column in (
                f"cb_stock_correlation_{window}d",
                f"cb_stock_beta_{window}d",
            )
        ],
    ]
    return panel[expected_columns].sort_values(
        ["signal_date", "ts_code"]
    ).reset_index(drop=True)


def audit_stock_linkage_factor_panel(
    factor_panel: pd.DataFrame,
    universe_panel: pd.DataFrame,
    stock_daily: pd.DataFrame,
    cb_daily: pd.DataFrame,
    *,
    return_windows: tuple[int, ...] = RETURN_WINDOWS,
    risk_window: int = RISK_WINDOW,
    linkage_windows: tuple[int, ...] = LINKAGE_WINDOWS,
    minimum_observation_ratio: float = MINIMUM_OBSERVATION_RATIO,
) -> dict[str, object]:
    """Recompute the panel and audit keys, formulas and factor coverage."""
    required = {"signal_date", "ts_code", "stk_code", *STOCK_LINKAGE_FACTOR_COLUMNS}
    if missing := required.difference(factor_panel.columns):
        raise KeyError(f"Factor panel is missing columns: {sorted(missing)}")

    actual = factor_panel.copy()
    actual["signal_date"] = _normalize_dates(actual["signal_date"])
    duplicate_keys = int(actual.duplicated(["signal_date", "ts_code"]).sum())
    expected = build_stock_linkage_factor_panel(
        universe_panel,
        stock_daily,
        cb_daily,
        return_windows=return_windows,
        risk_window=risk_window,
        linkage_windows=linkage_windows,
        minimum_observation_ratio=minimum_observation_ratio,
    )
    expected_keys = set(map(tuple, expected[["signal_date", "ts_code"]].to_numpy()))
    actual_keys = set(map(tuple, actual[["signal_date", "ts_code"]].to_numpy()))
    comparison = actual.merge(
        expected,
        on=["signal_date", "ts_code", "stk_code"],
        how="inner",
        suffixes=("_actual", "_expected"),
    )
    mismatch_count = 0
    factor_columns = list(STOCK_LINKAGE_FACTOR_COLUMNS)
    for column in factor_columns:
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

    audit = {
        "row_count": int(len(actual)),
        "signal_date_count": int(actual["signal_date"].nunique()),
        "duplicate_signal_bond_keys": duplicate_keys,
        "missing_expected_keys": int(len(expected_keys - actual_keys)),
        "extra_unexpected_keys": int(len(actual_keys - expected_keys)),
        "formula_mismatch_count": int(mismatch_count),
        "factor_coverage": {
            column: float(actual[column].notna().mean()) if len(actual) else 0.0
            for column in factor_columns
        },
    }
    audit["status"] = "pass" if all(
        audit[key] == 0
        for key in (
            "duplicate_signal_bond_keys",
            "missing_expected_keys",
            "extra_unexpected_keys",
            "formula_mismatch_count",
        )
    ) else "fail"
    return audit
