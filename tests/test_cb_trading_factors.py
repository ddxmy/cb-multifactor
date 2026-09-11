import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cb_quant.cb_trading import (
    audit_cb_trading_factor_panel,
    build_cb_trading_factor_panel,
)


class ConvertibleBondTradingFactorTests(unittest.TestCase):
    def make_inputs(self):
        dates = pd.bdate_range("2020-01-02", periods=65)
        close = pd.Series(100.0 + np.arange(len(dates)) * 0.25)
        volume = pd.Series(1_000.0 + np.arange(len(dates)) * 10.0)
        vwap = close - 0.20
        amount = vwap * volume / 1_000.0
        signal_date = dates[60]

        universe = pd.DataFrame(
            {
                "signal_date": [signal_date, signal_date],
                "ts_code": ["CB1", "CB2"],
                "is_eligible": [True, False],
            }
        )
        daily = pd.DataFrame(
            {
                "ts_code": "CB1",
                "trade_date": dates.strftime("%Y%m%d"),
                "pre_close": np.r_[np.nan, close.iloc[:-1]],
                "open": close - 0.10,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "pct_chg": close.pct_change().to_numpy() * 100.0,
                "vol": volume,
                "amount": amount,
            }
        )
        reference = pd.DataFrame(
            {
                "ts_code": ["CB1", "CB2"],
                "list_date": [dates[0], dates[0]],
                "issue_size": [100_000_000.0, 100_000_000.0],
            }
        )
        share_events = pd.DataFrame(
            {
                "ts_code": ["CB1"],
                "effective_date": [dates[50]],
                "remain_size": [80_000_000.0],
            }
        )
        return universe, daily, reference, share_events, dates, signal_date

    def test_builds_confirmed_cb_daily_trading_factors(self):
        universe, daily, reference, share_events, dates, signal_date = self.make_inputs()

        panel = build_cb_trading_factor_panel(
            universe,
            daily,
            reference,
            share_events,
            dates,
        )

        self.assertEqual(panel["ts_code"].tolist(), ["CB1"])
        row = panel.iloc[0]
        close = daily["close"]
        for window in (5, 10, 20):
            expected = close.iloc[60] / close.iloc[60 - window] - 1.0
            self.assertAlmostEqual(row[f"cb_return_{window}d"], expected)
        self.assertAlmostEqual(
            row["cb_short_long_momentum_5_20d"],
            row["cb_return_5d"] - row["cb_return_20d"],
        )

        volume = daily["vol"]
        remaining_size = pd.Series(100_000_000.0, index=daily.index)
        remaining_size.loc[50:] = 80_000_000.0
        turnover = volume * 1_000.0 / remaining_size
        self.assertAlmostEqual(row["cb_daily_turnover"], turnover.iloc[60])
        self.assertAlmostEqual(
            row["cb_abnormal_turnover_20d"],
            turnover.iloc[60] / turnover.iloc[40:60].mean(),
        )
        self.assertAlmostEqual(
            row["cb_abnormal_turnover_60d"],
            turnover.iloc[60] / turnover.iloc[0:60].mean(),
        )

        log_amount = np.log(daily["amount"] * 10_000.0)
        expected_amount_zscore = (
            log_amount.iloc[60] - log_amount.iloc[40:60].mean()
        ) / log_amount.iloc[40:60].std(ddof=1)
        self.assertAlmostEqual(row["cb_log_amount_zscore_20d"], expected_amount_zscore)

        daily_return = daily["pct_chg"] / 100.0
        expected_amihud = (
            (daily_return.iloc[41:61].abs() / (daily["amount"].iloc[41:61] * 10_000.0)).mean()
            * 1e8
        )
        self.assertAlmostEqual(row["cb_amihud_20d"], expected_amihud)

        vwap = daily["amount"] * 1_000.0 / daily["vol"]
        close_to_vwap = daily["close"] / vwap - 1.0
        self.assertAlmostEqual(row["cb_vwap"], vwap.iloc[60])
        self.assertAlmostEqual(row["cb_close_to_vwap"], close_to_vwap.iloc[60])
        self.assertAlmostEqual(
            row["cb_close_to_vwap_mean_5d"], close_to_vwap.iloc[56:61].mean()
        )
        self.assertEqual(row["signal_date"], signal_date)
        self.assertAlmostEqual(row["remain_size"], 80_000_000.0)

    def test_separates_vwap_hand_scale_from_actual_amount_notional_scale(self):
        universe, daily, reference, share_events, dates, _ = self.make_inputs()

        panel = build_cb_trading_factor_panel(
            universe, daily, reference, share_events, dates
        )

        row = panel.iloc[0]
        expected_vwap = daily.loc[60, "amount"] * 1_000.0 / daily.loc[60, "vol"]
        daily_return = daily["pct_chg"] / 100.0
        expected_amihud = (
            (
                daily_return.iloc[41:61].abs()
                / (daily["amount"].iloc[41:61] * 10_000.0)
            ).mean()
            * 1e8
        )
        hand_scaled_amihud = (
            (
                daily_return.iloc[41:61].abs()
                / (daily["amount"].iloc[41:61] * 1_000.0)
            ).mean()
            * 1e8
        )

        self.assertAlmostEqual(row["cb_vwap"], expected_vwap)
        self.assertAlmostEqual(row["cb_amihud_20d"], expected_amihud)
        self.assertNotAlmostEqual(row["cb_amihud_20d"], hand_scaled_amihud)

    def test_missing_market_row_stays_missing_and_is_excluded_from_turnover_mean(self):
        universe, daily, reference, share_events, dates, _ = self.make_inputs()
        missing_index = 45
        universe = pd.concat(
            [
                universe,
                pd.DataFrame(
                    {
                        "signal_date": [dates[missing_index]],
                        "ts_code": ["CB1"],
                        "is_eligible": [True],
                    }
                ),
            ],
            ignore_index=True,
        )
        daily_with_gap = daily.drop(index=missing_index).reset_index(drop=True)

        panel = build_cb_trading_factor_panel(
            universe,
            daily_with_gap,
            reference,
            share_events,
            dates,
        )

        gap_row = panel.loc[panel["signal_date"].eq(dates[missing_index])].iloc[0]
        for column in (
            "cb_close",
            "cb_vol",
            "cb_amount",
            "cb_vwap",
            "cb_daily_turnover",
            "cb_return_5d",
            "cb_return_10d",
            "cb_return_20d",
            "cb_abnormal_turnover_20d",
            "cb_log_amount_zscore_20d",
            "cb_amihud_20d",
            "cb_close_to_vwap",
            "cb_close_to_vwap_mean_5d",
        ):
            with self.subTest(column=column):
                self.assertTrue(pd.isna(gap_row[column]))
        self.assertFalse(gap_row["cb_vwap_unit_valid"])

        later_row = panel.loc[panel["signal_date"].eq(dates[60])].iloc[0]
        remaining_size = pd.Series(100_000_000.0, index=daily.index)
        remaining_size.loc[50:] = 80_000_000.0
        turnover = daily["vol"] * 1_000.0 / remaining_size
        observed_prior = turnover.iloc[40:60].drop(index=missing_index)
        expected = turnover.iloc[60] / observed_prior.mean()
        gap_as_zero = turnover.iloc[60] / (observed_prior.sum() / 20.0)
        self.assertAlmostEqual(later_row["cb_abnormal_turnover_20d"], expected)
        self.assertNotAlmostEqual(later_row["cb_abnormal_turnover_20d"], gap_as_zero)

    def test_observed_zero_volume_row_has_zero_turnover_but_no_amount_features(self):
        universe, daily, reference, share_events, dates, _ = self.make_inputs()
        daily.loc[60, "vol"] = 0.0
        daily.loc[60, "amount"] = 0.0

        panel = build_cb_trading_factor_panel(
            universe,
            daily,
            reference,
            share_events,
            dates,
        )

        row = panel.iloc[0]
        self.assertEqual(row["cb_daily_turnover"], 0.0)
        self.assertEqual(row["cb_abnormal_turnover_20d"], 0.0)
        self.assertEqual(row["cb_amount"], 0.0)
        self.assertFalse(row["cb_vwap_unit_valid"])
        self.assertTrue(pd.notna(row["cb_return_5d"]))
        for column in (
            "cb_vwap",
            "cb_close_to_vwap",
            "cb_close_to_vwap_mean_5d",
            "cb_log_amount_zscore_20d",
            "cb_amihud_20d",
        ):
            with self.subTest(column=column):
                self.assertTrue(pd.isna(row[column]))

    def test_abnormal_turnover_20d_requires_16_valid_prior_observations(self):
        universe, daily, reference, share_events, dates, _ = self.make_inputs()
        zero_index = 40
        daily.loc[zero_index, ["vol", "amount"]] = 0.0

        daily_with_16 = daily.drop(index=[41, 42, 43, 44]).reset_index(drop=True)
        panel_with_16 = build_cb_trading_factor_panel(
            universe,
            daily_with_16,
            reference,
            share_events,
            dates,
        )

        remaining_size = pd.Series(100_000_000.0, index=daily.index)
        remaining_size.loc[50:] = 80_000_000.0
        turnover = daily["vol"] * 1_000.0 / remaining_size
        valid_prior_indices = [zero_index, *range(45, 60)]
        prior_turnover = turnover.loc[valid_prior_indices]
        self.assertEqual(len(prior_turnover), 16)
        self.assertEqual(prior_turnover.loc[zero_index], 0.0)
        expected = turnover.loc[60] / prior_turnover.mean()
        without_zero = turnover.loc[60] / prior_turnover.loc[lambda values: values.gt(0)].mean()
        observed = panel_with_16.iloc[0]["cb_abnormal_turnover_20d"]
        self.assertAlmostEqual(observed, expected)
        self.assertNotAlmostEqual(observed, without_zero)

        daily_with_15 = daily.drop(index=[41, 42, 43, 44, 45]).reset_index(drop=True)
        panel_with_15 = build_cb_trading_factor_panel(
            universe,
            daily_with_15,
            reference,
            share_events,
            dates,
        )
        self.assertTrue(pd.isna(panel_with_15.iloc[0]["cb_abnormal_turnover_20d"]))

    def test_does_not_use_market_or_event_data_after_signal_date(self):
        universe, daily, reference, share_events, dates, _ = self.make_inputs()
        baseline = build_cb_trading_factor_panel(
            universe, daily, reference, share_events, dates
        )
        changed_daily = daily.copy()
        changed_daily.loc[61:, ["high", "close", "vol", "amount"]] = 99_999.0
        future_event = pd.DataFrame(
            {
                "ts_code": ["CB1"],
                "effective_date": [dates[61]],
                "remain_size": [1_000_000.0],
            }
        )

        changed = build_cb_trading_factor_panel(
            universe,
            changed_daily,
            reference,
            pd.concat([share_events, future_event], ignore_index=True),
            dates,
        )

        factor_columns = [
            column
            for column in baseline.columns
            if column.startswith("cb_")
        ]
        pd.testing.assert_series_equal(
            baseline.loc[0, factor_columns],
            changed.loc[0, factor_columns],
        )

    def test_audit_detects_formula_tampering_and_validates_vwap_units(self):
        universe, daily, reference, share_events, dates, _ = self.make_inputs()
        panel = build_cb_trading_factor_panel(
            universe, daily, reference, share_events, dates
        )

        clean = audit_cb_trading_factor_panel(
            panel, universe, daily, reference, share_events, dates
        )
        self.assertEqual(clean["status"], "pass")
        self.assertEqual(clean["formula_mismatch_count"], 0)
        self.assertAlmostEqual(clean["vwap_inside_daily_range_rate"], 1.0)

        broken = panel.copy()
        broken.loc[0, "cb_return_5d"] = 999.0
        audit = audit_cb_trading_factor_panel(
            broken, universe, daily, reference, share_events, dates
        )
        self.assertEqual(audit["status"], "fail")
        self.assertEqual(audit["formula_mismatch_count"], 1)

    def test_rejects_duplicate_daily_keys(self):
        universe, daily, reference, share_events, dates, _ = self.make_inputs()
        duplicate = pd.concat([daily, daily.iloc[[0]]], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "duplicate bond/date"):
            build_cb_trading_factor_panel(
                universe, duplicate, reference, share_events, dates
            )

    def test_excludes_internally_inconsistent_amount_volume_observation(self):
        universe, daily, reference, share_events, dates, _ = self.make_inputs()
        daily.loc[60, "amount"] = daily.loc[60, "amount"] * 0.50

        panel = build_cb_trading_factor_panel(
            universe, daily, reference, share_events, dates
        )

        row = panel.iloc[0]
        self.assertFalse(row["cb_vwap_unit_valid"])
        self.assertTrue(pd.isna(row["cb_vwap"]))
        self.assertTrue(pd.isna(row["cb_close_to_vwap"]))
        self.assertTrue(pd.isna(row["cb_daily_turnover"]))
        self.assertTrue(pd.isna(row["cb_abnormal_turnover_20d"]))
        self.assertTrue(pd.isna(row["cb_log_amount_zscore_20d"]))

    def test_writes_partitioned_panel_review_and_audit(self):
        from scripts.build_g3_cb_trading import write_cb_trading_artifacts

        universe, daily, reference, share_events, dates, _ = self.make_inputs()
        panel = build_cb_trading_factor_panel(
            universe, daily, reference, share_events, dates
        )
        audit = audit_cb_trading_factor_panel(
            panel, universe, daily, reference, share_events, dates
        )
        review = panel.copy()

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "G3"
            write_cb_trading_artifacts(panel, review, audit, output_root)

            self.assertTrue(
                (
                    output_root
                    / "cb_trading_factor_panel_v1"
                    / "year=2020"
                    / "part-0000.parquet"
                ).is_file()
            )
            self.assertTrue(
                (output_root / "cb_trading_anomaly_review_v1.parquet").is_file()
            )
            payload = json.loads(
                (output_root / "cb_trading_factor_audit_v1.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["status"], "pass")


if __name__ == "__main__":
    unittest.main()
