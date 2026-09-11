import tempfile
import unittest
from pathlib import Path

import duckdb

from cb_quant.legacy_schema import (
    LEGACY_A_SHARE_COLUMNS,
    LEGACY_A_SHARE_DAILY_DIRECTORY,
    LEGACY_CB_INTRADAY_DIRECTORY,
    quote_identifier,
)
from cb_quant.stock_data import load_a_share_daily, load_a_share_market_cap


class LegacySchemaAdapterTests(unittest.TestCase):
    def test_legacy_identifiers_are_exposed_through_one_adapter(self):
        self.assertEqual(
            LEGACY_CB_INTRADAY_DIRECTORY,
            ("\u53ef\u8f6c\u503a\u6570\u636e\u5e93", "\u53ef\u8f6c\u503a\u5206\u65f6\u65e5\u7ebf"),
        )
        self.assertEqual(LEGACY_A_SHARE_DAILY_DIRECTORY, "A\u80a1\u65e5\u7ebf\u6570\u636e\u5e93")
        self.assertEqual(LEGACY_A_SHARE_COLUMNS["ts_code"], "\u4ee3\u7801")
        self.assertEqual(LEGACY_A_SHARE_COLUMNS["adj_factor"], "\u590d\u6743\u56e0\u5b50")
        self.assertEqual(LEGACY_A_SHARE_COLUMNS["total_market_cap"], "\u603b\u5e02\u503c")

    def test_stock_loader_uses_legacy_schema_and_returns_english_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "a_share.duckdb"
            definitions = ", ".join(
                f"{quote_identifier(source)} {data_type}"
                for source, data_type in (
                    (LEGACY_A_SHARE_COLUMNS["ts_code"], "VARCHAR"),
                    (LEGACY_A_SHARE_COLUMNS["trade_date"], "VARCHAR"),
                    *(
                        (LEGACY_A_SHARE_COLUMNS[column], "DOUBLE")
                        for column in (
                            "open", "high", "low", "close", "pre_close", "pct_chg",
                            "vol", "amount", "turnover_rate", "adj_factor",
                            "total_market_cap",
                        )
                    ),
                )
            )
            with duckdb.connect(str(database_path)) as connection:
                connection.execute(f"CREATE TABLE daily_adj ({definitions})")
                connection.execute(
                    "INSERT INTO daily_adj VALUES "
                    "('600001.SH', '20200103', 10, 11, 9, 10.5, 10, 5, 100, 1000, 1, 2, 50000)"
                )

            loaded = load_a_share_daily(
                database_path,
                ["600001.SH"],
                start_date="20200101",
                end_date="20200103",
            )

            self.assertEqual(
                loaded.columns.tolist(),
                [
                    "ts_code", "trade_date", "open", "high", "low", "close",
                    "pre_close", "pct_chg", "vol", "amount", "turnover_rate", "adj_factor",
                ],
            )
            self.assertEqual(loaded.loc[0, "high"], 11.0)
            self.assertEqual(loaded.loc[0, "adj_factor"], 2.0)

            market_cap = load_a_share_market_cap(
                database_path,
                ["600001.SH"],
                start_date="20200101",
                end_date="20200103",
            )
            self.assertEqual(
                market_cap.columns.tolist(),
                ["stk_code", "trade_date", "total_market_cap"],
            )
            self.assertEqual(market_cap.loc[0, "total_market_cap"], 50000.0)


if __name__ == "__main__":
    unittest.main()
