import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from cb_quant.data_catalog import DataCatalog
from cb_quant.event_state import attach_asof_event_state, prepare_rating_events, prepare_share_events
from cb_quant.g1_universe import (
    audit_universe_panel,
    build_rebalance_calendar,
    build_universe_transitions,
)
from cb_quant.industry import attach_ci_industry, load_ci_industry_membership
from cb_quant.legacy_schema import (
    CJK_GLYPH_PROBE,
    LEGACY_A_SHARE_COLUMNS,
    LEGACY_A_SHARE_DAILY_DIRECTORY,
    LEGACY_CB_INTRADAY_DIRECTORY,
    LEGACY_DATABASE_DIRECTORY,
    LEGACY_G1_FUNNEL_COUNT_COLUMN,
    quote_identifier,
)
from cb_quant.tushare_history import _read_completed_windows, load_cb_codes, load_cb_codes_from_terms, month_windows
from cb_quant.universe import build_daily_universe_snapshot, universe_funnel
from cb_quant.valuation import build_conversion_valuation, build_parity_relative_premium
from cb_quant.stock_data import load_a_share_daily
from cb_quant.plotting import configure_matplotlib


class DataCatalogTests(unittest.TestCase):
    def test_frozen_research_config_passes_stage_validator(self):
        from scripts.validate_research_config import validate_config

        config_path = Path(__file__).resolve().parents[1] / "config" / "research_v1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(validate_config(config), [])

    def test_environment_catalog_requires_explicit_database_root(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(EnvironmentError, "CB_DATABASE_ROOT"):
                DataCatalog.from_environment()

    def test_stage_scripts_resolve_repository_root(self):
        from scripts import (
            build_g1_universe,
            build_g2_valuation,
            build_g3_cb_trading,
            build_g3_stock_linkage,
            validate_research_config,
        )

        project_root = Path(__file__).resolve().parents[1]
        for script in (
            validate_research_config,
            build_g1_universe,
            build_g2_valuation,
            build_g3_stock_linkage,
            build_g3_cb_trading,
        ):
            with self.subTest(script=script.__name__):
                self.assertEqual(script.PROJECT_ROOT, project_root)

    def make_catalog(self, root: Path) -> DataCatalog:
        return DataCatalog(
            research_root=root / "research",
            legacy_project_root=root / "legacy",
            intraday_root=root / "intraday",
            database_root=root / "database",
        )

    def test_intraday_path_before_2025(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = self.make_catalog(Path(directory))
            expected = catalog.intraday_root / "2024" / "2024_1m_7z.duckdb"
            self.assertEqual(catalog.intraday_db_path(2024, "1m"), expected)

    def test_environment_catalog_uses_current_database_tree_and_latest_a_share_daily(self):
        with tempfile.TemporaryDirectory() as directory:
            database_root = Path(directory) / LEGACY_DATABASE_DIRECTORY
            intraday_root = database_root.joinpath(*LEGACY_CB_INTRADAY_DIRECTORY)
            a_share_root = database_root / LEGACY_A_SHARE_DAILY_DIRECTORY
            intraday_root.mkdir(parents=True)
            a_share_root.mkdir(parents=True)
            older = a_share_root / "daily_adj_19901219_20260721.duckdb"
            latest = a_share_root / "daily_adj_19901219_20260722.duckdb"
            older.touch()
            latest.touch()

            with patch.dict(
                os.environ,
                {"CB_DATABASE_ROOT": str(database_root)},
                clear=True,
            ):
                catalog = DataCatalog.from_environment()

            self.assertEqual(catalog.intraday_root, intraday_root.resolve())
            self.assertEqual(catalog.a_share_daily_path, latest.resolve())

    def test_plotting_configuration_renders_cjk_without_missing_glyphs(self):
        font_path = Path.home() / "Library/Fonts/Kaiti.ttc"
        if not font_path.exists():
            self.skipTest("Kaiti font is not installed on this machine")

        font_config = configure_matplotlib(chinese_font_path=font_path)
        figure, axis = plt.subplots()
        axis.set_title(f"{CJK_GLYPH_PROBE} Return")
        axis.set_xlabel(CJK_GLYPH_PROBE)
        axis.plot([0, 1], [-1, 1])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            figure.canvas.draw()
        plt.close(figure)

        glyph_warnings = [item for item in caught if "Glyph" in str(item.message)]
        self.assertEqual(glyph_warnings, [])
        self.assertEqual(font_config.chinese_family, "Kaiti SC")

    def test_intraday_path_handles_2025_and_2026_names(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = self.make_catalog(Path(directory))
            self.assertEqual(
                catalog.intraday_db_path(2025, "15m"),
                catalog.intraday_root / "2025" / "15min_7z.duckdb",
            )
            self.assertEqual(
                catalog.intraday_db_path(2026, "day"),
                catalog.intraday_root / "2026" / "2026_day_zip.duckdb",
            )

    def test_invalid_frequency_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = self.make_catalog(Path(directory))
            with self.assertRaises(ValueError):
                catalog.intraday_db_path(2024, "tick")

    def test_manifest_marks_missing_required_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = self.make_catalog(Path(directory))
            manifest = catalog.source_manifest(2018, 2018)
            required = [item for item in manifest if item["required"]]
            self.assertTrue(required)
            self.assertTrue(all(not item["exists"] for item in required))
            with self.assertRaises(FileNotFoundError):
                catalog.validate_required_sources(2018, 2018)

    def test_ci_paths_use_database_root(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = self.make_catalog(Path(directory))
            self.assertEqual(
                catalog.ci_l1_daily_path,
                catalog.local_database_root / "ci_l1_daily.duckdb",
            )
            self.assertEqual(
                catalog.ci_l2_daily_path,
                catalog.local_database_root / "ci_l2_daily.duckdb",
            )

    def test_ci_membership_uses_exact_history_date(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "ci_l1_daily.duckdb"
            with duckdb.connect(str(database_path)) as connection:
                connection.execute(
                    """
                    CREATE TABLE default_table (
                        trade_date DATE,
                        ts_code VARCHAR,
                        l1_name VARCHAR,
                        con_codes VARCHAR[]
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO default_table VALUES
                    (DATE '2020-01-02', 'CI005001.CI', 'Legacy Industry', ['000001.SZ']),
                    (DATE '2020-01-03', 'CI005002.CI', 'Current Industry', ['000001.SZ', '000002.SZ'])
                    """
                )

            membership = load_ci_industry_membership(database_path, "2020-01-02")
            self.assertEqual(membership["ci_industry_name"].tolist(), ["Legacy Industry"])
            self.assertEqual(membership["membership_date"].iloc[0].isoformat(), "2020-01-02")

            stocks = pd.DataFrame({"stk_code": ["000001.SZ", "999999.SZ"]})
            attached = attach_ci_industry(stocks, database_path, "2020-01-03")
            self.assertEqual(attached.loc[0, "ci_industry_name"], "Current Industry")
            self.assertTrue(pd.isna(attached.loc[1, "ci_industry_name"]))

    def test_ci_membership_rejects_invalid_level(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "missing.duckdb"
            with self.assertRaises(ValueError):
                load_ci_industry_membership(database_path, "2020-01-02", level="l3")  # type: ignore[arg-type]

    def test_ci_membership_marks_overlapping_industries_as_ambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "ci_l1.duckdb"
            with duckdb.connect(str(database_path)) as connection:
                connection.execute(
                    """
                    CREATE TABLE default_table (
                        trade_date DATE,
                        ts_code VARCHAR,
                        l1_name VARCHAR,
                        con_codes VARCHAR[]
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO default_table VALUES
                    (DATE '2020-01-02', 'CI001.CI', 'Industry One', ['000001.SZ']),
                    (DATE '2020-01-02', 'CI002.CI', 'Industry Two', ['000001.SZ'])
                    """
                )
            membership = load_ci_industry_membership(
                database_path, "2020-01-02"
            )
            self.assertEqual(len(membership), 1)
            self.assertTrue(membership.loc[0, "is_industry_ambiguous"])
            self.assertTrue(pd.isna(membership.loc[0, "ci_industry_name"]))
            self.assertEqual(
                membership.loc[0, "industry_candidate_names"], "Industry One|Industry Two"
            )

    def test_month_windows_cover_range_without_overlap(self):
        self.assertEqual(
            month_windows("20200130", "20200302"),
            [("20200130", "20200131"), ("20200201", "20200229"), ("20200301", "20200302")],
        )

    def test_load_cb_codes_excludes_exchangeable_bonds(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "cb_basic.csv"
            pd.DataFrame(
                {
                    "ts_code": ["110001.SH", "132001.SZ", "110001.SH"],
                    "cb_type": ["CB", "EB", "CB"],
                }
            ).to_csv(source, index=False)
            self.assertEqual(load_cb_codes(source), ["110001.SH"])

    def test_load_cb_codes_from_terms_excludes_exchangeable_bonds(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "cb_list.duckdb"
            with duckdb.connect(str(source)) as connection:
                connection.execute("CREATE TABLE default_table (ts_code VARCHAR, cb_type VARCHAR)")
                connection.execute(
                    "INSERT INTO default_table VALUES ('110002.SH', 'CB'), ('132001.SZ', 'EB'), ('110001.SH', 'CB')"
                )
            self.assertEqual(load_cb_codes_from_terms(source), ["110001.SH", "110002.SH"])

    def test_completed_share_windows_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "cb_share_completed_windows.csv"
            pd.DataFrame(
                {"start_date": ["20200101"], "end_date": ["20200131"]}
            ).to_csv(source, index=False)
            self.assertEqual(_read_completed_windows(source), {("20200101", "20200131")})

    def test_disclosed_events_become_visible_on_next_trade_day(self):
        calendar = pd.Series(pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]))
        ratings = pd.DataFrame({
            "ts_code": ["110001.SH", "110001.SH"],
            "ann_date": ["20200102", "20200102"],
            "rating": ["AA", "A"],
        })
        shares = pd.DataFrame({
            "ts_code": ["110001.SH", "110001.SH"],
            "publish_date": ["20200102", "20200102"],
            "end_date": ["20191231", "20200101"],
            "remain_size": [100_000_000, 90_000_000],
            "convert_price": [10.0, 9.0],
        })
        rating_events = prepare_rating_events(ratings, calendar)
        share_events = prepare_share_events(shares, calendar)
        self.assertEqual(rating_events.loc[0, "effective_date"], pd.Timestamp("2020-01-03"))
        self.assertEqual(rating_events.loc[0, "rating"], "A")
        self.assertEqual(share_events.loc[0, "remain_size"], 90_000_000)

        panel = pd.DataFrame({"ts_code": ["110001.SH", "110001.SH"], "trade_date": ["2020-01-02", "2020-01-03"]})
        attached = attach_asof_event_state(panel, rating_events, ["rating", "rating_score"])
        self.assertTrue(pd.isna(attached.iloc[0]["rating"]))
        self.assertEqual(attached.iloc[1]["rating"], "A")

        attached_with_date = attach_asof_event_state(
            panel,
            rating_events,
            ["rating", "rating_score", "ann_date"],
            effective_date_output="rating_effective_date",
        )
        self.assertTrue(pd.isna(attached_with_date.iloc[0]["rating_effective_date"]))
        self.assertEqual(
            attached_with_date.iloc[1]["rating_effective_date"],
            pd.Timestamp("2020-01-03"),
        )

    def test_daily_universe_snapshot_applies_point_in_time_filters(self):
        dates = pd.DatetimeIndex(pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]))
        reference = pd.DataFrame(
            {
                "ts_code": ["110001.SH", "110002.SH"],
                "stk_code": ["600001.SH", "600002.SH"],
                "list_date": ["2020-01-02", "2020-01-02"],
                "issue_size": [300_000_000.0, 300_000_000.0],
                "first_conv_price": [10.0, 10.0],
            }
        )
        daily = pd.DataFrame(
            {
                "ts_code": ["110001.SH", "110001.SH", "110002.SH", "110002.SH"],
                "trade_date": [20200102, 20200103, 20200102, 20200103],
                "close": [100.0, 101.0, 100.0, 100.0],
                "vol": [10_000.0, 10_000.0, 10_000.0, 10_000.0],
                "amount": [1_000.0, 1_000.0, 1_000.0, 1_000.0],
            }
        )
        ratings = pd.DataFrame(
            {
                "ts_code": ["110001.SH"],
                "effective_date": ["2020-01-03"],
                "rating": ["A"],
                "rating_score": [14],
            }
        )
        shares = pd.DataFrame(
            {
                "ts_code": ["110001.SH"],
                "effective_date": ["2020-01-03"],
                "remain_size": [250_000_000.0],
                "convert_price": [10.0],
            }
        )
        snapshot = build_daily_universe_snapshot(
            "2020-01-03",
            daily,
            reference,
            dates,
            ratings,
            shares,
            minimum_age_days=2,
            turnover_window=2,
        )
        eligible = snapshot.set_index("ts_code")
        self.assertTrue(eligible.loc["110001.SH", "is_eligible"])
        self.assertFalse(eligible.loc["110002.SH", "is_rating_eligible"])
        self.assertFalse(eligible.loc["110002.SH", "is_eligible"])
        self.assertAlmostEqual(eligible.loc["110001.SH", "turnover_20d"], 0.0733333333)
        self.assertEqual(eligible.loc["110002.SH", "conversion_price_source"], "initial_terms")
        self.assertEqual(
            eligible.loc["110002.SH", "primary_exclusion_reason"], "rating_missing"
        )
        self.assertEqual(
            universe_funnel(snapshot).iloc[-1][LEGACY_G1_FUNNEL_COUNT_COLUMN], 1
        )

    def test_old_listing_is_not_treated_as_new_when_calendar_history_is_truncated(self):
        dates = pd.DatetimeIndex(pd.to_datetime(["2020-01-02", "2020-01-03"]))
        reference = pd.DataFrame(
            {
                "ts_code": ["110001.SH"],
                "stk_code": ["600001.SH"],
                "list_date": ["2019-01-02"],
                "issue_size": [300_000_000.0],
                "first_conv_price": [10.0],
            }
        )
        daily = pd.DataFrame(
            {
                "ts_code": ["110001.SH"],
                "trade_date": [20200102],
                "close": [100.0],
                "vol": [1_000.0],
                "amount": [1_000.0],
            }
        )
        ratings = pd.DataFrame(
            {
                "ts_code": ["110001.SH"],
                "effective_date": ["2020-01-02"],
                "rating": ["AA"],
                "rating_score": [17],
            }
        )
        shares = pd.DataFrame(
            columns=["ts_code", "effective_date", "remain_size", "convert_price"]
        )
        snapshot = build_daily_universe_snapshot(
            "2020-01-02", daily, reference, dates, ratings, shares
        )
        self.assertTrue(snapshot.loc[0, "listing_age_is_lower_bound"])
        self.assertTrue(snapshot.loc[0, "is_age_eligible"])

    def test_rebalance_calendar_preserves_fixed_anchors_across_holidays(self):
        dates = pd.DatetimeIndex(
            pd.to_datetime(
                ["2020-01-10", "2020-01-13", "2020-01-23", "2020-02-03", "2020-02-04"]
            )
        )
        calendar = build_rebalance_calendar(
            dates,
            anchor_date="2020-01-10",
            sample_start="2020-01-10",
            sample_end="2020-02-04",
        )
        self.assertEqual(
            calendar["anchor_date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2020-01-10", "2020-01-24"],
        )
        self.assertEqual(
            calendar["signal_date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2020-01-10", "2020-02-03"],
        )
        self.assertEqual(calendar["holiday_shift_days"].tolist(), [0, 10])

    def test_universe_audit_detects_future_event_and_formula_mismatch(self):
        panel = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(["2020-01-03"]),
                "ts_code": ["110001.SH"],
                "is_age_eligible": [True],
                "is_turnover_eligible": [True],
                "is_size_eligible": [True],
                "is_rating_eligible": [True],
                "is_signal_tradable": [True],
                "is_eligible": [False],
                "share_effective_date": pd.to_datetime(["2020-01-06"]),
                "rating_effective_date": pd.to_datetime(["2020-01-03"]),
                "membership_date": pd.to_datetime(["2020-01-03"]),
                "primary_exclusion_reason": ["eligible"],
            }
        )
        calendar = pd.DataFrame({"signal_date": pd.to_datetime(["2020-01-03"])})
        audit = audit_universe_panel(panel, calendar, random_date_count=1)
        self.assertEqual(audit["status"], "fail")
        self.assertEqual(audit["eligibility_formula_mismatches"], 1)
        self.assertEqual(audit["event_timing_violations"]["share_effective_date"], 1)

    def test_universe_transitions_preserve_entry_and_exit_reasons(self):
        panel = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(
                    ["2020-01-02", "2020-01-16", "2020-01-30"]
                ),
                "ts_code": ["110001.SH"] * 3,
                "is_eligible": [False, True, False],
                "primary_exclusion_reason": [
                    "listing_age_below_minimum",
                    "eligible",
                    "turnover_above_maximum",
                ],
            }
        )
        transitions = build_universe_transitions(panel)
        self.assertEqual(transitions["transition"].tolist(), ["entered", "exited"])
        self.assertEqual(
            transitions["transition_reason"].tolist(),
            ["listing_age_below_minimum", "turnover_above_maximum"],
        )

    def test_conversion_valuation_uses_same_day_stock_close_and_event_price(self):
        snapshot = pd.DataFrame(
            {
                "ts_code": ["110001.SH", "110002.SH"],
                "stk_code": ["600001.SH", "600002.SH"],
                "close": [120.0, 100.0],
                "convert_price": [10.0, None],
            }
        )
        stocks = pd.DataFrame(
            {
                "ts_code": ["600001.SH", "600002.SH"],
                "trade_date": [20200103, 20200103],
                "close": [12.0, 8.0],
            }
        )
        valuation = build_conversion_valuation(snapshot, stocks, snapshot_date="2020-01-03").set_index("ts_code")
        self.assertAlmostEqual(valuation.loc["110001.SH", "conversion_value"], 120.0)
        self.assertAlmostEqual(valuation.loc["110001.SH", "conversion_premium"], 0.0)
        self.assertAlmostEqual(valuation.loc["110001.SH", "double_low"], 120.0)
        self.assertFalse(valuation.loc["110002.SH", "is_valuation_available"])

    def test_local_a_share_adapter_maps_ohlcv_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "a_share.duckdb"
            with duckdb.connect(str(database_path)) as connection:
                definitions = ", ".join(
                    f"{quote_identifier(LEGACY_A_SHARE_COLUMNS[column])} {data_type}"
                    for column, data_type in (
                        ("ts_code", "VARCHAR"), ("trade_date", "VARCHAR"),
                        ("open", "DOUBLE"), ("high", "DOUBLE"), ("low", "DOUBLE"),
                        ("close", "DOUBLE"), ("pre_close", "DOUBLE"),
                        ("pct_chg", "DOUBLE"), ("vol", "DOUBLE"), ("amount", "DOUBLE"),
                        ("turnover_rate", "DOUBLE"), ("adj_factor", "DOUBLE"),
                    )
                )
                connection.execute(f"CREATE TABLE daily_adj ({definitions})")
                connection.execute("INSERT INTO daily_adj VALUES ('600001.SH', '20200103', 10, 11, 9, 10.5, 10, 5, 100, 1000, 1, 2)")
            loaded = load_a_share_daily(database_path, ['600001.SH'], start_date='20200101', end_date='20200103')
            self.assertEqual(loaded.columns.tolist(), [
                'ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'pct_chg', 'vol', 'amount', 'turnover_rate', 'adj_factor'
            ])
            self.assertEqual(loaded.loc[0, 'high'], 11.0)
            self.assertEqual(loaded.loc[0, 'adj_factor'], 2.0)

    def test_relative_premium_compares_only_same_parity_group(self):
        valuation = pd.DataFrame(
            {
                'conversion_value': [80.0, 80.0, 80.0, 115.0, 115.0],
                'conversion_premium': [0.20, 0.30, 0.40, 0.10, 0.20],
            }
        )
        factor = build_parity_relative_premium(valuation, minimum_group_size=2)
        self.assertEqual(factor.loc[0, 'parity_bucket'], '70-90')
        self.assertAlmostEqual(factor.loc[0, 'parity_group_median_premium'], 0.30)
        self.assertAlmostEqual(factor.loc[0, 'relative_premium_to_parity_median'], -0.10)
        self.assertAlmostEqual(factor.loc[0, 'parity_value_score'], 0.10)
        self.assertAlmostEqual(factor.loc[3, 'relative_premium_to_parity_median'], -0.05)


if __name__ == "__main__":
    unittest.main()
