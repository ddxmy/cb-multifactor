import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from cb_quant.stock_linkage import (
    LINKAGE_FACTOR_COLUMNS,
    STOCK_FACTOR_COLUMNS,
    STOCK_LINKAGE_FACTOR_COLUMNS,
    audit_stock_linkage_factor_panel,
    build_stock_linkage_factor_panel,
)


class StockLinkageFactorPanelTests(unittest.TestCase):
    def test_factor_families_form_an_exact_partition(self):
        self.assertEqual(len(STOCK_FACTOR_COLUMNS), 9)
        self.assertEqual(len(LINKAGE_FACTOR_COLUMNS), 7)
        self.assertFalse(set(STOCK_FACTOR_COLUMNS) & set(LINKAGE_FACTOR_COLUMNS))
        self.assertEqual(
            [*STOCK_FACTOR_COLUMNS, *LINKAGE_FACTOR_COLUMNS],
            STOCK_LINKAGE_FACTOR_COLUMNS,
        )

    def make_inputs(self):
        dates = pd.bdate_range("2020-01-02", periods=62)
        stock_close = np.arange(100.0, 162.0)
        bond_close = np.arange(100.0, 131.0, 0.5)
        signal_date = dates[60]

        universe = pd.DataFrame(
            {
                "signal_date": [signal_date, signal_date],
                "ts_code": ["CB1", "CB2"],
                "stk_code": ["STK1", "STK2"],
                "is_eligible": [True, False],
            }
        )
        stock = pd.DataFrame(
            {
                "ts_code": "STK1",
                "trade_date": dates.strftime("%Y%m%d"),
                "high": stock_close + 1.0,
                "low": stock_close - 1.0,
                "close": stock_close,
                "pre_close": np.r_[np.nan, stock_close[:-1]],
                "pct_chg": pd.Series(stock_close).pct_change().to_numpy() * 100.0,
                "vol": 10_000.0,
                "amount": 1_000.0,
                "adj_factor": 1.0,
            }
        )
        bond = pd.DataFrame(
            {
                "ts_code": "CB1",
                "trade_date": dates.strftime("%Y%m%d"),
                "close": bond_close,
            }
        )
        return universe, stock, bond, signal_date

    def test_builds_confirmed_stock_and_stock_bond_factors(self):
        universe, stock, bond, signal_date = self.make_inputs()

        panel = build_stock_linkage_factor_panel(universe, stock, bond)

        self.assertEqual(panel["ts_code"].tolist(), ["CB1"])
        row = panel.iloc[0]
        stock_close = stock["close"]
        bond_close = bond["close"]
        stock_daily_return = stock_close.pct_change()
        self.assertAlmostEqual(
            row["stock_return_5d"], stock_close.iloc[60] / stock_close.iloc[55] - 1.0
        )
        self.assertAlmostEqual(
            row["stock_return_10d"], stock_close.iloc[60] / stock_close.iloc[50] - 1.0
        )
        self.assertAlmostEqual(
            row["stock_return_20d"], stock_close.iloc[60] / stock_close.iloc[40] - 1.0
        )
        self.assertAlmostEqual(
            row["stock_volatility_20d"],
            stock_daily_return.iloc[41:61].std(ddof=1) * np.sqrt(252.0),
        )
        self.assertAlmostEqual(row["stock_rsi_20d"], 100.0)
        self.assertAlmostEqual(row["stock_price_to_high_20d"], 1.0)
        expected_amihud = (
            (stock_daily_return.iloc[41:61].abs() / (1_000.0 * 1_000.0)).mean()
            * 1e8
        )
        self.assertAlmostEqual(row["stock_amihud_20d"], expected_amihud)
        rolling_close = stock_close.iloc[41:61]
        middle = rolling_close.mean()
        standard_deviation = rolling_close.std(ddof=0)
        expected_percent_b = (
            stock_close.iloc[60] - (middle - 2.0 * standard_deviation)
        ) / (4.0 * standard_deviation)
        self.assertAlmostEqual(row["stock_percent_b_20d"], expected_percent_b)
        self.assertAlmostEqual(row["stock_mfi_20d"], 100.0)
        for window in (5, 10, 20):
            expected = (
                bond_close.iloc[60] / bond_close.iloc[60 - window] - 1.0
                - (stock_close.iloc[60] / stock_close.iloc[60 - window] - 1.0)
            )
            self.assertAlmostEqual(row[f"cb_stock_return_spread_{window}d"], expected)
        bond_daily_return = bond_close.pct_change()
        paired = pd.DataFrame(
            {
                "stock": stock_daily_return.iloc[41:61],
                "bond": bond_daily_return.iloc[41:61],
            }
        )
        self.assertAlmostEqual(
            row["cb_stock_correlation_20d"], paired["bond"].corr(paired["stock"])
        )
        self.assertAlmostEqual(
            row["cb_stock_beta_20d"],
            paired["bond"].cov(paired["stock"]) / paired["stock"].var(ddof=1),
        )
        paired_60d = pd.DataFrame(
            {
                "stock": stock_daily_return.iloc[1:61],
                "bond": bond_daily_return.iloc[1:61],
            }
        )
        self.assertAlmostEqual(
            row["cb_stock_correlation_60d"],
            paired_60d["bond"].corr(paired_60d["stock"]),
        )
        self.assertAlmostEqual(
            row["cb_stock_beta_60d"],
            paired_60d["bond"].cov(paired_60d["stock"])
            / paired_60d["stock"].var(ddof=1),
        )
        self.assertEqual(row["signal_date"], signal_date)

    def test_uses_adjusted_stock_prices_for_continuous_return_factors(self):
        universe, stock, bond, _ = self.make_inputs()
        stock.loc[10:, "adj_factor"] = 2.0

        panel = build_stock_linkage_factor_panel(universe, stock, bond)

        expected = (
            stock.loc[60, "close"] * stock.loc[60, "adj_factor"]
            / (stock.loc[55, "close"] * stock.loc[55, "adj_factor"])
            - 1.0
        )
        self.assertAlmostEqual(panel.iloc[0]["stock_return_5d"], expected)

    def test_does_not_use_observations_after_the_signal_date(self):
        universe, stock, bond, signal_date = self.make_inputs()
        baseline = build_stock_linkage_factor_panel(universe, stock, bond)
        future_stock = stock.copy()
        future_stock.loc[61, "close"] = 10_000.0
        future_bond = bond.copy()
        future_bond.loc[61, "close"] = 10_000.0

        with_future = build_stock_linkage_factor_panel(
            universe,
            future_stock,
            future_bond,
        )

        factor_columns = [
            column
            for column in baseline.columns
            if column.startswith("stock_") or column.startswith("cb_stock_")
        ]
        pd.testing.assert_series_equal(
            baseline.loc[0, factor_columns],
            with_future.loc[0, factor_columns],
        )

    def test_mfi_is_neutral_when_typical_price_never_changes(self):
        universe, stock, bond, _ = self.make_inputs()
        stock[["high", "close", "pre_close"]] = 100.0
        stock["low"] = 100.0
        stock["pct_chg"] = 0.0

        panel = build_stock_linkage_factor_panel(universe, stock, bond)

        self.assertAlmostEqual(panel.iloc[0]["stock_mfi_20d"], 50.0)
        self.assertTrue(pd.isna(panel.iloc[0]["stock_percent_b_20d"]))

    def test_near_zero_linkage_variance_does_not_emit_runtime_warning(self):
        universe, stock, bond, _ = self.make_inputs()
        stock["pct_chg"] = 1.0423606739764034e-10
        bond["pct_chg"] = np.linspace(0.01, 0.02, len(bond))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_stock_linkage_factor_panel(universe, stock, bond)

        runtime_warnings = [
            warning
            for warning in caught
            if issubclass(warning.category, RuntimeWarning)
        ]
        self.assertEqual(runtime_warnings, [])

    def test_audit_detects_formula_tampering(self):
        universe, stock, bond, _ = self.make_inputs()
        panel = build_stock_linkage_factor_panel(universe, stock, bond)

        clean = audit_stock_linkage_factor_panel(panel, universe, stock, bond)
        self.assertEqual(clean["status"], "pass")
        self.assertEqual(clean["formula_mismatch_count"], 0)

        broken = panel.copy()
        broken.loc[0, "stock_return_5d"] = 999.0
        audit = audit_stock_linkage_factor_panel(broken, universe, stock, bond)
        self.assertEqual(audit["status"], "fail")
        self.assertEqual(audit["formula_mismatch_count"], 1)

    def test_rejects_duplicate_daily_keys(self):
        universe, stock, bond, _ = self.make_inputs()
        duplicate_stock = pd.concat([stock, stock.iloc[[0]]], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "duplicate stock/date"):
            build_stock_linkage_factor_panel(universe, duplicate_stock, bond)

    def test_writes_partitioned_panel_review_and_audit(self):
        from scripts.build_g3_stock_linkage import write_stock_linkage_artifacts

        universe, stock, bond, _ = self.make_inputs()
        panel = build_stock_linkage_factor_panel(universe, stock, bond)
        audit = audit_stock_linkage_factor_panel(panel, universe, stock, bond)
        review = panel.copy()

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "G3"
            write_stock_linkage_artifacts(panel, review, audit, output_root)

            self.assertTrue(
                (
                    output_root
                    / "stock_linkage_factor_panel_v1"
                    / "year=2020"
                    / "part-0000.parquet"
                ).is_file()
            )
            self.assertTrue(
                (output_root / "stock_linkage_anomaly_review_v1.parquet").is_file()
            )
            audit_payload = json.loads(
                (output_root / "stock_linkage_factor_audit_v1.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(audit_payload["status"], "pass")


if __name__ == "__main__":
    unittest.main()
