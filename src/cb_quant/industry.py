"""Point-in-time industry membership helpers for convertible-bond underlyings."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd


IndustryLevel = Literal["l1", "l2"]
_VALID_LEVELS = {"l1", "l2"}


def load_ci_industry_membership(
    database_path: Path,
    as_of_date: date | str,
    level: IndustryLevel = "l1",
) -> pd.DataFrame:
    """Return the CITIC constituent map that was observable on ``as_of_date``.

    The two local CI DuckDB files store a daily snapshot in ``con_codes``. This
    function intentionally queries one exact date: it neither backfills from a
    later snapshot nor forward-fills a missing industry assignment.
    """
    if level not in _VALID_LEVELS:
        raise ValueError("level must be either 'l1' or 'l2'")
    if not database_path.exists():
        raise FileNotFoundError(f"CI industry database does not exist: {database_path}")

    name_column = f"{level}_name"
    query = f"""
        SELECT
            code AS stk_code,
            ts_code AS ci_industry_code,
            {name_column} AS ci_industry_name,
            trade_date::DATE AS membership_date,
            '{level}' AS ci_industry_level
        FROM default_table,
             UNNEST(con_codes) AS constituent(code)
        WHERE trade_date::DATE = ?
          AND code IS NOT NULL
    """
    with duckdb.connect(str(database_path), read_only=True) as connection:
        membership = connection.execute(query, [as_of_date]).df()

    membership["membership_date"] = pd.to_datetime(membership["membership_date"]).dt.date
    membership = _resolve_ambiguous_membership(
        membership, key_columns=["stk_code"]
    )
    return membership.sort_values("stk_code", ignore_index=True)


def load_ci_industry_membership_panel(
    database_path: Path,
    signal_dates: pd.Series | pd.DatetimeIndex,
    level: IndustryLevel = "l1",
) -> pd.DataFrame:
    """Load exact-date CITIC membership for several signal dates in one query."""
    if level not in _VALID_LEVELS:
        raise ValueError("level must be either 'l1' or 'l2'")
    if not database_path.exists():
        raise FileNotFoundError(f"CI industry database does not exist: {database_path}")

    requested = pd.DataFrame(
        {
            "signal_date": pd.DatetimeIndex(
                pd.to_datetime(signal_dates, errors="coerce")
            ).normalize().unique()
        }
    ).dropna()
    if requested.empty:
        return pd.DataFrame(
            columns=[
                "signal_date",
                "stk_code",
                "ci_industry_code",
                "ci_industry_name",
                "membership_date",
                "ci_industry_level",
            ]
        )

    name_column = f"{level}_name"
    query = f"""
        SELECT
            requested.signal_date,
            code AS stk_code,
            source.ts_code AS ci_industry_code,
            source.{name_column} AS ci_industry_name,
            source.trade_date::DATE AS membership_date,
            '{level}' AS ci_industry_level
        FROM default_table AS source
        INNER JOIN requested
            ON source.trade_date::DATE = requested.signal_date::DATE,
        UNNEST(source.con_codes) AS constituent(code)
        WHERE code IS NOT NULL
    """
    with duckdb.connect(str(database_path), read_only=True) as connection:
        connection.register("requested", requested)
        membership = connection.execute(query).df()

    membership["signal_date"] = pd.to_datetime(membership["signal_date"]).dt.normalize()
    membership["membership_date"] = pd.to_datetime(membership["membership_date"]).dt.normalize()
    key_columns = ["signal_date", "stk_code"]
    membership = _resolve_ambiguous_membership(
        membership, key_columns=key_columns
    )
    return membership.sort_values(key_columns, ignore_index=True)


def _resolve_ambiguous_membership(
    membership: pd.DataFrame,
    *,
    key_columns: list[str],
) -> pd.DataFrame:
    """Keep one row per stock/date and mark overlapping classifications missing."""
    if membership.empty:
        membership["is_industry_ambiguous"] = pd.Series(dtype=bool)
        membership["industry_candidate_names"] = pd.Series(dtype="string")
        return membership

    candidate_names = (
        membership.groupby(key_columns, dropna=False)["ci_industry_name"]
        .agg(lambda values: "|".join(sorted(set(map(str, values.dropna())))))
        .rename("industry_candidate_names")
        .reset_index()
    )
    candidate_counts = (
        membership.groupby(key_columns, dropna=False)
        .size()
        .rename("industry_candidate_count")
        .reset_index()
    )
    resolved = membership.drop_duplicates(key_columns, keep="first").merge(
        candidate_names, on=key_columns, how="left", validate="1:1"
    ).merge(candidate_counts, on=key_columns, how="left", validate="1:1")
    resolved["is_industry_ambiguous"] = resolved["industry_candidate_count"].gt(1)
    resolved.loc[
        resolved["is_industry_ambiguous"],
        ["ci_industry_code", "ci_industry_name"],
    ] = pd.NA
    return resolved


def attach_ci_industry(
    frame: pd.DataFrame,
    database_path: Path,
    as_of_date: date | str,
    level: IndustryLevel = "l1",
    stock_code_column: str = "stk_code",
) -> pd.DataFrame:
    """Attach the same-day CITIC industry to a frame containing stock codes."""
    if stock_code_column not in frame.columns:
        raise KeyError(f"Missing stock code column: {stock_code_column}")

    membership = load_ci_industry_membership(database_path, as_of_date, level)
    return frame.merge(
        membership,
        how="left",
        left_on=stock_code_column,
        right_on="stk_code",
        validate="m:1",
        suffixes=("", "_industry"),
    )
