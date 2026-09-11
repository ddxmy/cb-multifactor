import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cb_quant.valuation import (
    audit_valuation_panel,
    build_full_valuation_panel,
    build_valuation_anomaly_review,
    summarize_parity_bucket_coverage,
)


class FullValuationPanelTests(unittest.TestCase):
    def make_universe_panel(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "signal_date": pd.to_datetime(
                    [
                        "2020-01-02",
                        "2020-01-02",
                        "2020-01-02",
                        "2020-01-03",
                        "2020-01-03",
                    ]
                ),
                "trade_date": pd.to_datetime(
                    [
                        "2020-01-02",
                        "2020-01-02",
                        "2020-01-02",
                        "2020-01-03",
                        "2020-01-03",
                    ]
                ),
                "ts_code": ["CB1", "CB2", "CB3", "CB1", "CB4"],
                "stk_code": ["STK1", "STK2", "STK3", "STK1", "STK4"],
                "close": [110.0, 120.0, 500.0, 126.0, 100.0],
                "convert_price": [100.0, 100.0, 100.0, 100.0, 100.0],
                "is_eligible": [True, True, False, True, True],
            }
        )

    def make_stock_daily(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ts_code": ["STK1", "STK2", "STK3", "STK1"],
                "trade_date": ["20200102", "20200102", "20200102", "20200103"],
                "close": [100.0, 100.0, 100.0, 120.0],
            }
        )

    def test_builds_each_signal_date_from_investable_rows_and_same_day_stock_close(self):
        panel = build_full_valuation_panel(
            self.make_universe_panel(),
            self.make_stock_daily(),
            minimum_group_size=2,
        )

        self.assertEqual(panel["ts_code"].tolist(), ["CB1", "CB2", "CB1", "CB4"])
        first_date = panel.loc[panel["signal_date"].eq(pd.Timestamp("2020-01-02"))]
        self.assertEqual(first_date["stock_close"].tolist(), [100.0, 100.0])
        self.assertAlmostEqual(first_date.iloc[0]["conversion_value"], 100.0)
        self.assertAlmostEqual(first_date.iloc[0]["conversion_premium"], 0.10)
        self.assertAlmostEqual(
            first_date.iloc[0]["relative_premium_to_parity_median"], -0.05
        )
        self.assertAlmostEqual(first_date.iloc[0]["parity_value_score"], 0.05)

        second_date = panel.loc[panel["signal_date"].eq(pd.Timestamp("2020-01-03"))]
        self.assertAlmostEqual(second_date.iloc[0]["stock_close"], 120.0)
        self.assertAlmostEqual(second_date.iloc[0]["conversion_value"], 120.0)
        self.assertFalse(second_date.iloc[1]["is_valuation_available"])
        self.assertTrue(second_date["parity_value_score"].isna().all())

    def test_rejects_duplicate_stock_date_rows(self):
        stock_daily = pd.concat(
            [self.make_stock_daily(), self.make_stock_daily().iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaisesRegex(ValueError, "duplicate stock/date"):
            build_full_valuation_panel(
                self.make_universe_panel(),
                stock_daily,
                minimum_group_size=2,
            )

    def test_summarizes_parity_coverage_by_signal_date_and_bucket(self):
        panel = build_full_valuation_panel(
            self.make_universe_panel(),
            self.make_stock_daily(),
            minimum_group_size=2,
        )

        coverage = summarize_parity_bucket_coverage(panel)

        row = coverage.loc[
            coverage["signal_date"].eq(pd.Timestamp("2020-01-02"))
            & coverage["parity_bucket"].astype(str).eq("100-110")
        ].iloc[0]
        self.assertEqual(row["bond_count"], 2)
        self.assertEqual(row["relative_value_count"], 2)
        self.assertAlmostEqual(row["valuation_coverage"], 1.0)

    def test_audit_detects_key_and_formula_mismatches(self):
        universe = self.make_universe_panel()
        panel = build_full_valuation_panel(
            universe,
            self.make_stock_daily(),
            minimum_group_size=2,
        )
        clean = audit_valuation_panel(panel, universe, minimum_group_size=2)
        self.assertEqual(clean["status"], "pass")
        self.assertEqual(clean["duplicate_signal_bond_keys"], 0)
        self.assertEqual(clean["missing_eligible_keys"], 0)
        self.assertEqual(clean["conversion_value_formula_mismatches"], 0)

        broken = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
        broken.loc[0, "conversion_value"] = 999.0
        audit = audit_valuation_panel(broken, universe, minimum_group_size=2)

        self.assertEqual(audit["status"], "fail")
        self.assertEqual(audit["duplicate_signal_bond_keys"], 1)
        self.assertEqual(audit["conversion_value_formula_mismatches"], 1)

    def test_audit_treats_missing_formula_output_as_a_mismatch(self):
        universe = self.make_universe_panel()
        panel = build_full_valuation_panel(
            universe,
            self.make_stock_daily(),
            minimum_group_size=2,
        )
        panel.loc[0, "conversion_value"] = pd.NA

        audit = audit_valuation_panel(panel, universe, minimum_group_size=2)

        self.assertEqual(audit["status"], "fail")
        self.assertEqual(audit["conversion_value_formula_mismatches"], 1)

    def test_audit_recomputes_same_parity_group_statistics(self):
        universe = self.make_universe_panel()
        panel = build_full_valuation_panel(
            universe,
            self.make_stock_daily(),
            minimum_group_size=2,
        )
        same_group = panel["signal_date"].eq(pd.Timestamp("2020-01-02"))
        panel.loc[same_group, "parity_group_median_premium"] += 0.10
        panel.loc[same_group, "relative_premium_to_parity_median"] = (
            panel.loc[same_group, "conversion_premium"]
            - panel.loc[same_group, "parity_group_median_premium"]
        )
        panel.loc[same_group, "parity_value_score"] = -panel.loc[
            same_group, "relative_premium_to_parity_median"
        ]

        audit = audit_valuation_panel(panel, universe, minimum_group_size=2)

        self.assertEqual(audit["status"], "fail")
        self.assertEqual(audit["parity_group_median_mismatches"], 2)

    def test_writes_year_partitions_coverage_and_audit(self):
        from scripts.build_g2_valuation import write_valuation_artifacts

        universe = self.make_universe_panel()
        panel = build_full_valuation_panel(
            universe,
            self.make_stock_daily(),
            minimum_group_size=2,
        )
        coverage = summarize_parity_bucket_coverage(panel)
        audit = audit_valuation_panel(panel, universe, minimum_group_size=2)
        anomalies = build_valuation_anomaly_review(panel, tail_count=1)

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "G2"
            write_valuation_artifacts(
                panel,
                coverage,
                audit,
                anomalies,
                output_root,
            )

            self.assertTrue(
                (
                    output_root
                    / "valuation_panel_v1"
                    / "year=2020"
                    / "part-0000.parquet"
                ).is_file()
            )
            self.assertTrue(
                (output_root / "parity_bucket_coverage_v1.parquet").is_file()
            )
            self.assertTrue(
                (output_root / "valuation_anomaly_review_v1.parquet").is_file()
            )
            audit_payload = json.loads(
                (output_root / "valuation_factor_audit_v1.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(audit_payload["status"], "pass")
            self.assertEqual(audit_payload["row_count"], len(panel))

    def test_builds_deduplicated_extreme_premium_review(self):
        panel = build_full_valuation_panel(
            self.make_universe_panel(),
            self.make_stock_daily(),
            minimum_group_size=2,
        )

        review = build_valuation_anomaly_review(panel, tail_count=2)

        self.assertFalse(review.duplicated(["signal_date", "ts_code"]).any())
        self.assertEqual(set(review["review_reason"]), {"lowest", "highest"})
        self.assertIn("conversion_premium", review.columns)
        self.assertIn("conversion_price_source", review.columns)


if __name__ == "__main__":
    unittest.main()
