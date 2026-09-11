"""Compatibility constants for locally licensed legacy database schemas.

Public research code uses English aliases. Source-system identifiers that cannot
be renamed are isolated here and represented with escaped Unicode code points.
"""

from __future__ import annotations


LEGACY_DATABASE_DIRECTORY = "\u6570\u636e\u5e93"
LEGACY_CB_INTRADAY_DIRECTORY = (
    "\u53ef\u8f6c\u503a\u6570\u636e\u5e93",
    "\u53ef\u8f6c\u503a\u5206\u65f6\u65e5\u7ebf",
)
LEGACY_CB_MARKET_DIRECTORY = (
    "\u53ef\u8f6c\u503a\u6570\u636e\u5e93",
    "\u53ef\u8f6c\u503a\u884c\u60c5",
)
LEGACY_A_SHARE_DAILY_DIRECTORY = "A\u80a1\u65e5\u7ebf\u6570\u636e\u5e93"
LEGACY_REDEMPTION_FILENAME = "\u5f3a\u8d4e.duckdb"
LEGACY_2025_DAILY_FILENAME = "\u65e5k_7z.duckdb"

LEGACY_REDEMPTION_COLUMNS = {
    "ts_code": "\u8f6c\u503a\u4ee3\u7801",
    "event_status": "\u662f\u5426\u8d4e\u56de",
    "effective_date": "\u516c\u544a\u65e5\u671f",
    "source_redeem_date": "\u8d4e\u56de\u65e5\u671f",
    "call_price": "\u8d4e\u56de\u4ef7\u683c",
    "tax_adjusted_call_price": "\u8d4e\u56de\u4ef7\u683c\u542b\u7a0e",
    "payment_date": "\u652f\u4ed8\u65e5\u671f",
    "call_reg_date": "\u8d4e\u56de\u767b\u8bb0\u65e5",
}

LEGACY_A_SHARE_COLUMNS = {
    "ts_code": "\u4ee3\u7801",
    "trade_date": "\u65e5\u671f",
    "open": "\u5f00\u76d8",
    "high": "\u6700\u9ad8",
    "low": "\u6700\u4f4e",
    "close": "\u6536\u76d8",
    "pre_close": "\u6628\u6536",
    "pct_chg": "\u6da8\u8dcc\u5e45",
    "vol": "\u6210\u4ea4\u91cf",
    "amount": "\u6210\u4ea4\u989d",
    "turnover_rate": "\u6362\u624b\u7387",
    "total_market_cap": "\u603b\u5e02\u503c",
    "adj_factor": "\u590d\u6743\u56e0\u5b50",
}

LEGACY_G1_FUNNEL_STAGE_COLUMN = "\u7b5b\u9009\u5c42"
LEGACY_G1_FUNNEL_COUNT_COLUMN = "\u5269\u4f59\u503a\u5238\u6570"
LEGACY_G1_FUNNEL_STAGES = (
    "\u666e\u901aCB\u4e14\u7814\u7a76\u671f\u5185\u6709\u65e5\u7ebf",
    "\u4e0a\u5e02\u7b2c11\u4e2a\u5e02\u573a\u4ea4\u6613\u65e5\u540e",
    "20\u65e5\u7d2f\u8ba1\u6362\u624b\u7387\u4e0d\u9ad8\u4e8e100%",
    "\u5269\u4f59\u89c4\u6a21\u4e0d\u4f4e\u4e8e2\u4ebf\u5143",
    "\u8bc4\u7ea7\u4e0d\u4f4e\u4e8eA",
    "\u4fe1\u53f7\u65e5\u771f\u5b9e\u6210\u4ea4",
)

CJK_GLYPH_PROBE = "\u4e2d\u6587\u53ef\u8f6c\u503a\u6ea2\u4ef7\u7387"


def quote_identifier(identifier: str) -> str:
    """Return a safely quoted SQL identifier for a trusted schema constant."""
    return '"' + identifier.replace('"', '""') + '"'
