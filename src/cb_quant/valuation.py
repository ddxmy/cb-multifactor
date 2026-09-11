"""Daily point-in-time convertible-bond valuation factor construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


PARITY_BINS = [0.0, 70.0, 90.0, 100.0, 110.0, 130.0, float("inf")]
PARITY_LABELS = ["<70", "70-90", "90-100", "100-110", "110-130", ">=130"]


def build_conversion_valuation(
    universe_snapshot: pd.DataFrame,
    stock_daily: pd.DataFrame,
    *,
    snapshot_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Attach conversion value, conversion premium and double-low to a snapshot.

    ``convert_price`` must already be a point-in-time state derived from disclosed
    conversion-price events. Missing conversion prices are intentionally kept as
    missing rather than filled from a current static terms field.
    """
    required_snapshot = {"ts_code", "stk_code", "close", "convert_price"}
    required_stock = {"ts_code", "trade_date", "close"}
    if missing := required_snapshot.difference(universe_snapshot.columns):
        raise KeyError(f"Universe snapshot is missing columns: {sorted(missing)}")
    if missing := required_stock.difference(stock_daily.columns):
        raise KeyError(f"Stock daily data is missing columns: {sorted(missing)}")

    date = pd.Timestamp(snapshot_date).normalize()
    stock = stock_daily[["ts_code", "trade_date", "close"]].copy()
    stock["ts_code"] = stock["ts_code"].astype(str).str.lstrip("\ufeff")
    stock["trade_date"] = pd.to_datetime(stock["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    stock = stock.loc[stock["trade_date"].eq(date)].rename(
        columns={"ts_code": "stk_code", "close": "stock_close"}
    )
    if stock.duplicated("stk_code").any():
        raise ValueError(f"Stock daily data has duplicate stock/date rows on {date.date()}")

    valuation = universe_snapshot.merge(stock[["stk_code", "stock_close"]], on="stk_code", how="left", validate="m:1")
    valid = (
        valuation["close"].gt(0)
        & valuation["stock_close"].gt(0)
        & valuation["convert_price"].gt(0)
    )
    valuation["conversion_value"] = pd.NA
    valuation.loc[valid, "conversion_value"] = (
        valuation.loc[valid, "stock_close"] / valuation.loc[valid, "convert_price"] * 100.0
    )
    valuation["conversion_value"] = pd.to_numeric(valuation["conversion_value"], errors="coerce")
    valuation["conversion_premium"] = valuation["close"] / valuation["conversion_value"] - 1.0
    valuation["double_low"] = valuation["close"] + valuation["conversion_premium"] * 100.0
    valuation["is_valuation_available"] = valuation["conversion_premium"].notna()
    return valuation


def build_parity_relative_premium(
    investable_valuation: pd.DataFrame,
    *,
    minimum_group_size: int = 5,
) -> pd.DataFrame:
    """Measure a bond's premium against the same-parity investable cross-section.

    The input must already be limited to the investable universe on one signal day.
    A negative relative premium means the bond is cheaper than its same-parity
    peers; ``parity_value_score`` reverses the sign so larger is cheaper.
    """
    if minimum_group_size <= 0:
        raise ValueError("minimum_group_size must be positive")
    required = {"conversion_value", "conversion_premium"}
    if missing := required.difference(investable_valuation.columns):
        raise KeyError(f"Investable valuation is missing columns: {sorted(missing)}")

    factor = investable_valuation.copy()
    valid = factor["conversion_value"].gt(0) & factor["conversion_premium"].notna()
    factor["parity_bucket"] = pd.cut(
        factor["conversion_value"],
        bins=PARITY_BINS,
        labels=PARITY_LABELS,
        right=False,
        include_lowest=True,
    )
    grouped = factor.loc[valid].groupby("parity_bucket", observed=True)["conversion_premium"]
    factor["parity_group_count"] = grouped.transform("size")
    factor["parity_group_median_premium"] = grouped.transform("median")
    sufficient = factor["parity_group_count"].ge(minimum_group_size)
    factor["relative_premium_to_parity_median"] = (
        factor["conversion_premium"] - factor["parity_group_median_premium"]
    ).where(sufficient)
    factor["parity_value_score"] = -factor["relative_premium_to_parity_median"]
    return factor


def build_full_valuation_panel(
    universe_panel: pd.DataFrame,
    stock_daily: pd.DataFrame,
    *,
    minimum_group_size: int = 5,
) -> pd.DataFrame:
    """Build raw valuation factors for every investable signal-date observation."""
    if minimum_group_size <= 0:
        raise ValueError("minimum_group_size must be positive")
    required_universe = {
        "signal_date",
        "ts_code",
        "stk_code",
        "close",
        "convert_price",
        "is_eligible",
    }
    required_stock = {"ts_code", "trade_date", "close"}
    if missing := required_universe.difference(universe_panel.columns):
        raise KeyError(f"Universe panel is missing columns: {sorted(missing)}")
    if missing := required_stock.difference(stock_daily.columns):
        raise KeyError(f"Stock daily data is missing columns: {sorted(missing)}")

    panel = universe_panel.loc[universe_panel["is_eligible"]].copy()
    panel["signal_date"] = pd.to_datetime(panel["signal_date"]).dt.normalize()
    if panel.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError("Universe panel has duplicate signal-date/bond rows")

    stock = stock_daily[["ts_code", "trade_date", "close"]].copy()
    stock["ts_code"] = stock["ts_code"].astype(str).str.lstrip("\ufeff")
    stock["trade_date"] = pd.to_datetime(
        stock["trade_date"].astype(str), format="%Y%m%d", errors="coerce"
    ).dt.normalize()
    stock["close"] = pd.to_numeric(stock["close"], errors="coerce")
    if stock.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("Stock daily data has duplicate stock/date rows")
    stock = stock.rename(
        columns={
            "ts_code": "stk_code",
            "trade_date": "stock_trade_date",
            "close": "stock_close",
        }
    )

    valuation = panel.merge(
        stock,
        left_on=["stk_code", "signal_date"],
        right_on=["stk_code", "stock_trade_date"],
        how="left",
        validate="m:1",
    )
    valid = (
        pd.to_numeric(valuation["close"], errors="coerce").gt(0)
        & valuation["stock_close"].gt(0)
        & pd.to_numeric(valuation["convert_price"], errors="coerce").gt(0)
    )
    valuation["conversion_value"] = np.where(
        valid,
        valuation["stock_close"] / valuation["convert_price"] * 100.0,
        np.nan,
    )
    valuation["conversion_premium"] = (
        valuation["close"] / valuation["conversion_value"] - 1.0
    )
    valuation["double_low"] = (
        valuation["close"] + valuation["conversion_premium"] * 100.0
    )
    valuation["is_valuation_available"] = valuation["conversion_premium"].notna()
    valuation["parity_bucket"] = pd.cut(
        valuation["conversion_value"],
        bins=PARITY_BINS,
        labels=PARITY_LABELS,
        right=False,
        include_lowest=True,
    )
    grouped = valuation.loc[valid].groupby(
        ["signal_date", "parity_bucket"], observed=True
    )["conversion_premium"]
    valuation["parity_group_count"] = grouped.transform("size")
    valuation["parity_group_median_premium"] = grouped.transform("median")
    sufficient = valuation["parity_group_count"].ge(minimum_group_size)
    valuation["relative_premium_to_parity_median"] = (
        valuation["conversion_premium"]
        - valuation["parity_group_median_premium"]
    ).where(sufficient)
    valuation["parity_value_score"] = (
        -valuation["relative_premium_to_parity_median"]
    )
    return valuation.sort_values(["signal_date", "ts_code"]).reset_index(drop=True)


def summarize_parity_bucket_coverage(
    valuation_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize absolute and relative valuation coverage by date and parity."""
    required = {
        "signal_date",
        "ts_code",
        "parity_bucket",
        "is_valuation_available",
        "parity_value_score",
        "conversion_value",
        "conversion_premium",
    }
    if missing := required.difference(valuation_panel.columns):
        raise KeyError(f"Valuation panel is missing columns: {sorted(missing)}")
    summary = (
        valuation_panel.dropna(subset=["parity_bucket"])
        .groupby(["signal_date", "parity_bucket"], observed=True)
        .agg(
            bond_count=("ts_code", "size"),
            valuation_count=("is_valuation_available", "sum"),
            relative_value_count=("parity_value_score", "count"),
            median_conversion_value=("conversion_value", "median"),
            median_conversion_premium=("conversion_premium", "median"),
        )
        .reset_index()
    )
    summary["valuation_coverage"] = (
        summary["valuation_count"] / summary["bond_count"]
    )
    summary["relative_value_coverage"] = (
        summary["relative_value_count"] / summary["bond_count"]
    )
    return summary.sort_values(["signal_date", "parity_bucket"]).reset_index(drop=True)


def build_valuation_anomaly_review(
    valuation_panel: pd.DataFrame,
    *,
    tail_count: int = 20,
) -> pd.DataFrame:
    """Return traceable low/high premium observations for manual source review."""
    if tail_count <= 0:
        raise ValueError("tail_count must be positive")
    required = {
        "signal_date",
        "ts_code",
        "stk_code",
        "close",
        "stock_close",
        "convert_price",
        "conversion_value",
        "conversion_premium",
    }
    if missing := required.difference(valuation_panel.columns):
        raise KeyError(f"Valuation panel is missing columns: {sorted(missing)}")

    valid = valuation_panel.loc[
        valuation_panel["conversion_premium"].notna()
    ].copy()
    lowest = valid.nsmallest(tail_count, "conversion_premium").copy()
    lowest["review_reason"] = "lowest"
    lowest_keys = pd.MultiIndex.from_frame(lowest[["signal_date", "ts_code"]])
    remaining = valid.loc[
        ~pd.MultiIndex.from_frame(valid[["signal_date", "ts_code"]]).isin(lowest_keys)
    ]
    highest = remaining.nlargest(tail_count, "conversion_premium").copy()
    highest["review_reason"] = "highest"
    review = pd.concat([lowest, highest], ignore_index=True)

    review_columns = [
        "review_reason",
        "signal_date",
        "ts_code",
        "bond_short_name",
        "stk_code",
        "close",
        "stock_close",
        "convert_price",
        "conversion_value",
        "conversion_premium",
        "double_low",
        "parity_bucket",
        "parity_group_count",
        "conversion_price_source",
        "share_effective_date",
    ]
    for column in review_columns:
        if column not in review:
            review[column] = pd.NA
    return review[review_columns].sort_values(
        ["review_reason", "conversion_premium", "signal_date", "ts_code"]
    ).reset_index(drop=True)


def _formula_mismatch_count(
    actual: pd.Series,
    expected: pd.Series,
    valid: pd.Series,
) -> int:
    missing_mismatches = valid & actual.isna().ne(expected.isna())
    comparable = valid & actual.notna() & expected.notna()
    value_mismatches = int(
        (
            ~np.isclose(
                actual.loc[comparable].astype(float),
                expected.loc[comparable].astype(float),
                rtol=1e-10,
                atol=1e-10,
            )
        ).sum()
    )
    return int(missing_mismatches.sum()) + value_mismatches


def audit_valuation_panel(
    valuation_panel: pd.DataFrame,
    universe_panel: pd.DataFrame,
    *,
    minimum_group_size: int = 5,
) -> dict[str, object]:
    """Audit keys, formulas and same-parity construction without future returns."""
    required = {
        "signal_date",
        "ts_code",
        "close",
        "stock_close",
        "convert_price",
        "conversion_value",
        "conversion_premium",
        "double_low",
        "parity_bucket",
        "parity_group_count",
        "parity_group_median_premium",
        "relative_premium_to_parity_median",
        "parity_value_score",
        "is_valuation_available",
    }
    if missing := required.difference(valuation_panel.columns):
        raise KeyError(f"Valuation panel is missing columns: {sorted(missing)}")
    if minimum_group_size <= 0:
        raise ValueError("minimum_group_size must be positive")

    panel = valuation_panel.copy()
    panel["signal_date"] = pd.to_datetime(panel["signal_date"]).dt.normalize()
    eligible = universe_panel.loc[
        universe_panel["is_eligible"], ["signal_date", "ts_code"]
    ].copy()
    eligible["signal_date"] = pd.to_datetime(eligible["signal_date"]).dt.normalize()
    expected_keys = set(map(tuple, eligible.drop_duplicates().to_numpy()))
    actual_keys = set(
        map(tuple, panel[["signal_date", "ts_code"]].drop_duplicates().to_numpy())
    )

    valid = (
        pd.to_numeric(panel["close"], errors="coerce").gt(0)
        & pd.to_numeric(panel["stock_close"], errors="coerce").gt(0)
        & pd.to_numeric(panel["convert_price"], errors="coerce").gt(0)
    )
    expected_conversion_value = (
        panel["stock_close"] / panel["convert_price"] * 100.0
    )
    expected_premium = panel["close"] / panel["conversion_value"] - 1.0
    expected_double_low = panel["close"] + panel["conversion_premium"] * 100.0
    conversion_mismatches = _formula_mismatch_count(
        panel["conversion_value"], expected_conversion_value, valid
    )
    premium_mismatches = _formula_mismatch_count(
        panel["conversion_premium"], expected_premium, valid
    )
    double_low_mismatches = _formula_mismatch_count(
        panel["double_low"], expected_double_low, valid
    )
    availability_mismatches = int(
        (
            panel["is_valuation_available"].astype(bool)
            != panel["conversion_premium"].notna()
        ).sum()
    )
    valid_group = (
        panel["conversion_value"].gt(0)
        & panel["conversion_premium"].notna()
        & panel["parity_bucket"].notna()
    )
    expected_group_count = pd.Series(np.nan, index=panel.index)
    expected_group_median = pd.Series(np.nan, index=panel.index)
    grouped = panel.loc[valid_group].groupby(
        ["signal_date", "parity_bucket"], observed=True
    )["conversion_premium"]
    expected_group_count.loc[valid_group] = grouped.transform("size")
    expected_group_median.loc[valid_group] = grouped.transform("median")
    group_count_mismatches = _formula_mismatch_count(
        panel["parity_group_count"],
        expected_group_count,
        valid_group,
    )
    group_median_mismatches = _formula_mismatch_count(
        panel["parity_group_median_premium"],
        expected_group_median,
        valid_group,
    )
    expected_relative_premium = (
        panel["conversion_premium"] - expected_group_median
    ).where(expected_group_count.ge(minimum_group_size))
    relative_formula_mismatches = _formula_mismatch_count(
        panel["relative_premium_to_parity_median"],
        expected_relative_premium,
        valid_group,
    )
    insufficient_group_scores = int(
        (
            expected_group_count.lt(minimum_group_size)
            & panel["parity_value_score"].notna()
        ).sum()
    )
    score_sign_mismatches = _formula_mismatch_count(
        panel["parity_value_score"],
        -panel["relative_premium_to_parity_median"],
        panel["relative_premium_to_parity_median"].notna(),
    )
    duplicate_keys = int(panel.duplicated(["signal_date", "ts_code"]).sum())

    audit = {
        "row_count": int(len(panel)),
        "expected_eligible_count": int(len(expected_keys)),
        "signal_date_count": int(panel["signal_date"].nunique()),
        "duplicate_signal_bond_keys": duplicate_keys,
        "missing_eligible_keys": int(len(expected_keys - actual_keys)),
        "extra_ineligible_keys": int(len(actual_keys - expected_keys)),
        "valuation_available_count": int(panel["is_valuation_available"].sum()),
        "valuation_coverage": float(panel["is_valuation_available"].mean())
        if len(panel)
        else 0.0,
        "relative_value_available_count": int(panel["parity_value_score"].notna().sum()),
        "relative_value_coverage": float(panel["parity_value_score"].notna().mean())
        if len(panel)
        else 0.0,
        "conversion_value_formula_mismatches": conversion_mismatches,
        "conversion_premium_formula_mismatches": premium_mismatches,
        "double_low_formula_mismatches": double_low_mismatches,
        "valuation_availability_mismatches": availability_mismatches,
        "parity_group_count_mismatches": group_count_mismatches,
        "parity_group_median_mismatches": group_median_mismatches,
        "relative_premium_formula_mismatches": relative_formula_mismatches,
        "insufficient_group_scores": insufficient_group_scores,
        "parity_value_score_sign_mismatches": score_sign_mismatches,
    }
    hard_failures = (
        "duplicate_signal_bond_keys",
        "missing_eligible_keys",
        "extra_ineligible_keys",
        "conversion_value_formula_mismatches",
        "conversion_premium_formula_mismatches",
        "double_low_formula_mismatches",
        "valuation_availability_mismatches",
        "parity_group_count_mismatches",
        "parity_group_median_mismatches",
        "relative_premium_formula_mismatches",
        "insufficient_group_scores",
        "parity_value_score_sign_mismatches",
    )
    audit["status"] = (
        "pass" if all(audit[key] == 0 for key in hard_failures) else "fail"
    )
    return audit
