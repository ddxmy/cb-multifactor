"""Read-only adapter for the local A-share daily OHLCV DuckDB dataset."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable

import duckdb
import pandas as pd

from .legacy_schema import LEGACY_A_SHARE_COLUMNS, quote_identifier


STOCK_DAILY_COLUMNS = [
    "ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
    "pct_chg", "vol", "amount", "turnover_rate", "adj_factor",
]


def load_a_share_daily(
    database_path: Path,
    stock_codes: Iterable[str],
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load local A-share OHLCV for requested codes and an inclusive date range.

    Raw prices match the Tushare close convention and are suitable for conversion
    value. ``adj_factor`` is supplied separately for continuous-price factors.
    """
    codes = sorted(set(map(str, stock_codes)))
    if not database_path.exists():
        raise FileNotFoundError(f"A-share daily database does not exist: {database_path}")
    if not codes:
        return pd.DataFrame(columns=STOCK_DAILY_COLUMNS)

    select_columns = ",\n            ".join(
        f"d.{quote_identifier(LEGACY_A_SHARE_COLUMNS[alias])} AS {quote_identifier(alias)}"
        for alias in STOCK_DAILY_COLUMNS
    )
    code_column = quote_identifier(LEGACY_A_SHARE_COLUMNS["ts_code"])
    date_column = quote_identifier(LEGACY_A_SHARE_COLUMNS["trade_date"])
    query = f"""
        SELECT
            {select_columns}
        FROM daily_adj AS d
        INNER JOIN UNNEST(?) AS requested(ts_code)
            ON d.{code_column} = requested.ts_code
        WHERE d.{date_column} BETWEEN ? AND ?
        ORDER BY d.{code_column}, d.{date_column}
    """
    with duckdb.connect(str(database_path), read_only=True) as connection:
        return connection.execute(query, [codes, start_date, end_date]).df()


def load_a_share_market_cap(
    database_path: Path,
    stock_codes: Iterable[str],
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load point-in-time total market capitalization for requested stocks."""
    codes = sorted(set(map(str, stock_codes)))
    columns = ["stk_code", "trade_date", "total_market_cap"]
    if not database_path.exists():
        raise FileNotFoundError(f"A-share daily database does not exist: {database_path}")
    if not codes:
        return pd.DataFrame(columns=columns)

    code_column = quote_identifier(LEGACY_A_SHARE_COLUMNS["ts_code"])
    date_column = quote_identifier(LEGACY_A_SHARE_COLUMNS["trade_date"])
    market_cap_column = quote_identifier(
        LEGACY_A_SHARE_COLUMNS["total_market_cap"]
    )
    query = f"""
        SELECT
            d.{code_column} AS stk_code,
            d.{date_column} AS trade_date,
            d.{market_cap_column} AS total_market_cap
        FROM daily_adj AS d
        INNER JOIN UNNEST(?) AS requested(stk_code)
            ON d.{code_column} = requested.stk_code
        WHERE d.{date_column} BETWEEN ? AND ?
        ORDER BY d.{code_column}, d.{date_column}
    """
    with duckdb.connect(str(database_path), read_only=True) as connection:
        result = connection.execute(query, [codes, start_date, end_date]).df()
    result["trade_date"] = pd.to_datetime(
        result["trade_date"].astype(str), format="%Y%m%d", errors="raise"
    ).dt.normalize()
    result["total_market_cap"] = pd.to_numeric(
        result["total_market_cap"], errors="coerce"
    )
    return result[columns]
