import json
import re
import subprocess
import tomllib
import unittest
from pathlib import Path


EXPECTED_NOTEBOOKS = {
    "00_data_sources_and_quality.ipynb",
    "01_point_in_time_universe.ipynb",
    "02_valuation_factors.ipynb",
    "03_equity_and_linkage_factors.ipynb",
    "04_cb_trading_factors.ipynb",
    "05_single_factor_backtest_framework.ipynb",
    "06_factor_preprocessing.ipynb",
}
G0_G2_NOTEBOOKS = (
    "00_data_sources_and_quality.ipynb",
    "01_point_in_time_universe.ipynb",
    "02_valuation_factors.ipynb",
)
G3_NOTEBOOKS = (
    "03_equity_and_linkage_factors.ipynb",
    "04_cb_trading_factors.ipynb",
)
NOTEBOOK_SECTION_ORDER = (
    "## TL;DR",
    "## Context & Methods",
    "## Data Quality Gate",
    "## Results",
    "## Takeaways",
)
MAX_NOTEBOOK_OUTPUT_BYTES = 300_000
LOCAL_ONLY_ROOTS = {"artifacts", "outputs", ".local_workspace", ".local_archive"}
TRACKED_LOCAL_ONLY_PREFIXES = (
    "artifacts/",
    "outputs/",
    ".local_archive/",
    ".local_workspace/",
    "data/interim/",
    "data/processed/",
)
PUBLIC_RECURSIVE_ROOTS = (
    ".github",
    "config",
    "docs",
    "notebooks",
    "scripts",
    "src",
    "tests",
)
PUBLIC_DATA_FILES = ("data/README.md",)
METHODOLOGY_DOCUMENTS = {
    "research_decisions.md",
    "factor_research_roadmap.md",
    "reference_report_factor_map.md",
    "implementation_status.md",
    "backtest_framework.md",
    "adding_a_factor.md",
    "empirical_results.md",
}
ROOT_METHODOLOGY_DOCUMENTS = {
    "DECISIONS.md",
    "FACTOR_RESEARCH_ROADMAP.md",
    "REPORT_FACTOR_MAP.md",
    "IMPLEMENTATION_CHECKLIST.md",
}
README_REPRODUCTION_COMMANDS = (
    "PYTHONPATH=src python scripts/validate_research_config.py --check-only",
    "PYTHONPATH=src python scripts/build_g1_universe.py",
    "PYTHONPATH=src python scripts/build_g2_valuation.py",
    "PYTHONPATH=src python scripts/build_g3_stock_linkage.py",
    "PYTHONPATH=src python scripts/build_g3_cb_trading.py",
    "PYTHONPATH=src python scripts/build_g3c_preprocessing.py",
    "PYTHONPATH=src python scripts/run_single_factor.py --help",
)
README_SETUP_COMMANDS = (
    "conda create -n cb-multifactor python=3.11 -y",
    "conda activate cb-multifactor",
    'python -m pip install -e ".[dev]"',
)
README_SHOWCASE_ASSETS = {
    "G0 data-quality coverage": "docs/assets/g0_data_quality_coverage.png",
    "G1 point-in-time universe diagnostics": (
        "docs/assets/g1_point_in_time_universe_diagnostics.png"
    ),
    "G2 valuation diagnostics": "docs/assets/g2_valuation_factor_diagnostics.png",
    "G3a equity and stock-bond linkage diagnostics": (
        "docs/assets/g3a_equity_linkage_diagnostics.png"
    ),
    "G3b convertible-bond trading diagnostics": (
        "docs/assets/g3b_cb_trading_diagnostics.png"
    ),
    "G5 replication-sample Rank IC": "docs/assets/g5_rank_ic_comparison.png",
    "G7 executable five-day portfolio comparison": (
        "docs/assets/g7_portfolio_comparison.png"
    ),
}
README_NOTEBOOK_LINKS = {
    notebook_name: f"notebooks/{notebook_name}"
    for notebook_name in EXPECTED_NOTEBOOKS
}
README_METHODOLOGY_LINKS = {
    "Research decisions": "docs/methodology/research_decisions.md",
    "Factor research roadmap": "docs/methodology/factor_research_roadmap.md",
    "Reference-report factor map": "docs/methodology/reference_report_factor_map.md",
    "Implementation stage status": "docs/methodology/implementation_status.md",
    "Backtest framework": "docs/methodology/backtest_framework.md",
    "Adding a factor": "docs/methodology/adding_a_factor.md",
    "Replication-sample empirical results": (
        "docs/methodology/empirical_results.md"
    ),
}
EXPECTED_STAGE_STATUS_ROWS = (
    "| G0 | Complete |",
    "| G1 | Complete |",
    "| G2 | Complete |",
    "| G3a | Complete |",
    "| G3b | Complete |",
    "| G3c | In Progress |",
    "| G4 | Replication Complete |",
    "| G5 | Representative Evidence |",
    "| G6 | Pending |",
    "| G7 | Representative Evidence |",
    "| G8 | Release Candidate |",
)
DATA_README_REQUIREMENTS = (
    "Convertible-bond daily market data",
    "Point-in-time terms and event data",
    "A-share daily data",
    "CITIC industry data",
    "CB_MULTIFACTOR_ROOT",
    "CB_LEGACY_PROJECT_ROOT",
    "CB_DATABASE_ROOT",
    "CB_A_SHARE_DAILY_PATH",
    "TUSHARE_TOKEN",
    "non-redistribution",
    "synthetic fixtures",
)
CREDENTIAL_ASSIGNMENT_PATTERN = (
    r"(?i:(?:[a-z0-9_-]*(?:token|secret|api[_-]?key|access[_-]?key)[a-z0-9_-]*)"
    r"\s*[:=]\s*['\"]?[0-9a-f]{40,})"
)
FORBIDDEN_PUBLIC_PATTERNS = (
    r"[\u4e00-\u9fff]",
    re.escape("/" + "Users" + "/mingyuxu/"),
    CREDENTIAL_ASSIGNMENT_PATTERN,
)
EXPECTED_STAGE_PROTOCOL = [
    "define_economic_rationale_and_formula",
    "freeze_parameters_and_information_timing",
    "validate_bounded_sample_formulas_and_reasonableness",
    "materialize_full_sample_artifacts",
    "run_independent_audit",
    "publish_research_notebook",
    "approve_stage_gate_before_advancing",
]
EXPECTED_BEHAVIOR_CONFIG = {
    "identity": {
        "schema_version": "1.0",
        "research_id": "cb_daily_multifactor_v1",
        "status": "frozen_core_v1",
    },
    "formal_sample": {"start": "2018-01-01", "end": "2026-07-07"},
    "research_partitions": {
        "report_replication_end": "2023-06-21",
        "extension_validation_start": "2023-06-22",
        "extension_validation_end": "2024-12-31",
        "final_holdout_start": "2025-01-01",
    },
    "universe": {
        "security_type": "ordinary_convertible_bond",
        "minimum_listing_age_market_days": 11,
        "turnover_window_market_days": 20,
        "maximum_cumulative_turnover": 1.0,
        "minimum_remaining_balance_yuan": 200000000,
        "minimum_rating": "A",
        "signal_day_real_trade_required": True,
        "current_static_event_backfill_forbidden": True,
    },
    "sampling": {
        "frequency": "biweekly",
        "anchor_signal_date": "2018-01-10",
        "planned_signal_rule": "every_14_calendar_days",
        "holiday_rule": "move_to_next_market_trading_day_without_shifting_future_anchors",
        "signal_time": "signal_day_close",
        "entry_time": "next_market_trading_day_open",
        "exit_time": "next_rebalance_execution_open",
    },
    "returns": {
        "primary_label": "next_execution_open_to_open_return",
        "benchmark": "same_investable_universe_equal_weight",
        "market_reference": "000832.CSI",
        "missing_execution_price_policy": "mark_order_failed_without_forward_fill",
        "label_end_date_purge_required": True,
    },
    "factor_evaluation": {
        "primary_metric": "cross_sectional_rank_ic",
        "required_outputs": [
            "rank_ic",
            "icir",
            "positive_ic_rate",
            "quantile_monotonicity",
            "holding_period_decay",
            "annual_stability",
            "market_state_stability",
        ],
        "test_period_parameter_selection_forbidden": True,
    },
    "preprocessing": {
        "winsorization": {
            "method": "cross_sectional_scaled_median_absolute_deviation",
            "scope": "independent_within_signal_date",
            "normal_consistency_scale": 1.4826,
            "primary_multiplier": 3.0,
            "sensitivity_multipliers": [2.5, 3.0, 3.5],
            "missing_value_policy": "preserve",
            "observation_policy": "clip_without_deleting_rows",
            "selection_uses_forward_returns": False,
        },
        "neutralization": {
            "status": "valuation_and_stock_complete_other_families_pending",
            "candidate_controls": [
                "log_remaining_balance",
                "log_stock_total_market_cap",
                "rating",
                "remaining_maturity",
                "citic_level_1_industry",
            ],
            "stock_market_cap_control": {
                "source": "point_in_time_a_share_daily_total_market_cap",
                "match_rule": "latest_observation_on_or_before_signal_date",
                "lookback_buffer_calendar_days": 365,
                "future_fill_forbidden": True,
            },
            "factor_policies": {
                "conversion_premium": {
                    "numeric_controls": [],
                    "categorical_controls": [],
                    "rationale": "Preserve the full parity-dependent convertible-bond valuation structure.",
                },
                "double_low": {
                    "numeric_controls": [],
                    "categorical_controls": [],
                    "rationale": "Preserve the intended joint price and conversion-premium signal.",
                },
                "relative_premium_to_parity_median": {
                    "numeric_controls": [
                        "log_remaining_balance",
                        "remaining_maturity_years",
                    ],
                    "categorical_controls": ["rating"],
                    "rationale": "Parity is controlled by construction; remove residual balance, credit, and maturity exposures.",
                },
                "parity_value_score": {
                    "numeric_controls": [
                        "log_remaining_balance",
                        "remaining_maturity_years",
                    ],
                    "categorical_controls": ["rating"],
                    "rationale": "Process the direction-reversed audit counterpart consistently; retain only one version during redundancy selection.",
                },
                "stock_return_5d": {
                    "numeric_controls": ["log_stock_total_market_cap"],
                    "categorical_controls": ["ci_industry_control"],
                    "rationale": "Remove broad industry and underlying-equity size exposures from the short-horizon stock return signal.",
                },
                "stock_return_10d": {
                    "numeric_controls": ["log_stock_total_market_cap"],
                    "categorical_controls": ["ci_industry_control"],
                    "rationale": "Remove broad industry and underlying-equity size exposures from the medium-horizon stock return signal.",
                },
                "stock_return_20d": {
                    "numeric_controls": ["log_stock_total_market_cap"],
                    "categorical_controls": ["ci_industry_control"],
                    "rationale": "Remove broad industry and underlying-equity size exposures from the monthly stock return signal.",
                },
                "stock_volatility_20d": {
                    "numeric_controls": ["log_stock_total_market_cap"],
                    "categorical_controls": ["ci_industry_control"],
                    "rationale": "Measure stock volatility beyond systematic industry and company-size effects.",
                },
                "stock_rsi_20d": {
                    "numeric_controls": ["log_stock_total_market_cap"],
                    "categorical_controls": ["ci_industry_control"],
                    "rationale": "Measure stock price strength beyond systematic industry and company-size effects.",
                },
                "stock_price_to_high_20d": {
                    "numeric_controls": ["log_stock_total_market_cap"],
                    "categorical_controls": ["ci_industry_control"],
                    "rationale": "Measure proximity to the recent high beyond systematic industry and company-size effects.",
                },
                "stock_percent_b_20d": {
                    "numeric_controls": ["log_stock_total_market_cap"],
                    "categorical_controls": ["ci_industry_control"],
                    "rationale": "Measure Bollinger-band position beyond systematic industry and company-size effects.",
                },
                "stock_amihud_20d": {
                    "numeric_controls": ["log_stock_total_market_cap"],
                    "categorical_controls": ["ci_industry_control"],
                    "rationale": "Separate stock illiquidity from its mechanical industry and company-size exposures.",
                },
                "stock_mfi_20d": {
                    "numeric_controls": ["log_stock_total_market_cap"],
                    "categorical_controls": ["ci_industry_control"],
                    "rationale": "Measure volume-weighted stock buying pressure beyond systematic industry and company-size effects.",
                },
            },
        },
        "standardization": {
            "method": "cross_sectional_population_zscore",
            "status": "valuation_and_stock_complete_other_families_pending",
        },
    },
    "portfolio": {
        "execution": "next_market_trading_day_open",
        "one_way_cost_bps": 15,
        "sell_before_buy": True,
        "failed_sell_policy": "continue_holding_and_retry",
        "failed_buy_policy": "retain_cash",
        "execution_day_full_day_liquidity_lookahead_forbidden": True,
    },
    "factor_scope_enums": {
        "report_groups": [
            "convertible_bond_valuation",
            "underlying_stock_price_volume",
            "stock_bond_linkage",
            "convertible_bond_daily_price_volume",
        ],
        "implemented_foundations": [
            "conversion_value",
            "conversion_premium",
            "double_low",
            "same_parity_relative_premium",
        ],
        "daily_primary_research": True,
        "intraday_factors_in_primary_research": False,
    },
}


class RepositoryReleaseTests(unittest.TestCase):
    project_root = Path(__file__).resolve().parents[1]

    def public_files(self):
        root_files = [path for path in self.project_root.iterdir() if path.is_file()]
        nested_files = []
        for root_name in PUBLIC_RECURSIVE_ROOTS:
            root = self.project_root / root_name
            if root.exists():
                nested_files.extend(path for path in root.rglob("*") if path.is_file())
        nested_files.extend(
            path for relative_path in PUBLIC_DATA_FILES
            if (path := self.project_root / relative_path).is_file()
        )
        return sorted(
            root_files + nested_files,
            key=lambda path: str(path.relative_to(self.project_root)),
        )

    def test_public_notebook_set_matches_release_sequence(self):
        notebooks_dir = self.project_root / "notebooks"
        actual_notebooks = {path.name for path in notebooks_dir.glob("*.ipynb")}
        self.assertEqual(actual_notebooks, EXPECTED_NOTEBOOKS)

    def test_g0_g2_notebooks_are_executed_reader_facing_reports(self):
        for notebook_name in G0_G2_NOTEBOOKS:
            with self.subTest(notebook=notebook_name):
                notebook_path = self.project_root / "notebooks" / notebook_name
                self.assertTrue(notebook_path.is_file(), f"Missing {notebook_name}")
                notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
                cells = notebook.get("cells", [])
                markdown = "\n".join(
                    "".join(cell.get("source", []))
                    for cell in cells
                    if cell.get("cell_type") == "markdown"
                )
                section_positions = [markdown.find(section) for section in NOTEBOOK_SECTION_ORDER]
                self.assertTrue(
                    all(position >= 0 for position in section_positions),
                    f"{notebook_name} is missing a required section",
                )
                self.assertEqual(
                    section_positions,
                    sorted(section_positions),
                    f"{notebook_name} sections are out of order",
                )

                code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
                self.assertTrue(code_cells, f"{notebook_name} has no executable analysis")
                self.assertTrue(
                    all(cell.get("execution_count") is not None for cell in code_cells),
                    f"{notebook_name} contains unexecuted code cells",
                )
                error_outputs = [
                    output
                    for cell in code_cells
                    for output in cell.get("outputs", [])
                    if output.get("output_type") == "error"
                ]
                self.assertEqual(error_outputs, [])

                public_text = "\n".join(
                    "".join(cell.get("source", []))
                    for cell in cells
                ) + "\n" + "\n".join(
                    json.dumps(output, ensure_ascii=False)
                    for cell in code_cells
                    for output in cell.get("outputs", [])
                    if output.get("output_type") != "display_data"
                    or "image/png" not in output.get("data", {})
                )
                self.assertIsNone(re.search(r"[\u4e00-\u9fff]", public_text))
                self.assertNotIn("/Users/", public_text)
                self.assertNotIn("config_path", public_text)
                output_size = sum(
                    len(json.dumps(cell.get("outputs", []), ensure_ascii=False).encode("utf-8"))
                    for cell in code_cells
                )
                self.assertLessEqual(output_size, MAX_NOTEBOOK_OUTPUT_BYTES)

    def test_g3_notebooks_are_executed_reader_facing_reports(self):
        required_terms = {
            "03_equity_and_linkage_factors.ipynb": (
                "Convertible Bond (CB)",
                "Relative Strength Index (RSI)",
                "Money Flow Index (MFI)",
                "Percent B",
                "Pearson correlation",
                "rolling beta",
            ),
            "04_cb_trading_factors.ipynb": (
                "Volume (`vol`) is reported in hands of ten bonds",
                "Amount (`amount`) is reported in ten-thousand CNY",
                "amount * 1,000 / vol",
                "amount * 10,000",
                "Volume-Weighted Average Price (VWAP)",
                "Amihud illiquidity",
            ),
        }
        forbidden_claims = (
            "forward_return",
            "future_return",
            "predictive performance",
            "predicts returns",
            "outperforms",
        )

        for notebook_name in G3_NOTEBOOKS:
            with self.subTest(notebook=notebook_name):
                notebook_path = self.project_root / "notebooks" / notebook_name
                self.assertTrue(notebook_path.is_file(), f"Missing {notebook_name}")
                notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
                cells = notebook.get("cells", [])
                markdown = "\n".join(
                    "".join(cell.get("source", []))
                    for cell in cells
                    if cell.get("cell_type") == "markdown"
                )
                section_positions = [markdown.find(section) for section in NOTEBOOK_SECTION_ORDER]
                self.assertTrue(
                    all(position >= 0 for position in section_positions),
                    f"{notebook_name} is missing a required section",
                )
                self.assertEqual(
                    section_positions,
                    sorted(section_positions),
                    f"{notebook_name} sections are out of order",
                )

                code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
                self.assertTrue(code_cells, f"{notebook_name} has no executable analysis")
                self.assertTrue(
                    all(cell.get("execution_count") is not None for cell in code_cells),
                    f"{notebook_name} contains unexecuted code cells",
                )
                error_outputs = [
                    output
                    for cell in code_cells
                    for output in cell.get("outputs", [])
                    if output.get("output_type") == "error"
                ]
                self.assertEqual(error_outputs, [])

                public_text = "\n".join(
                    "".join(cell.get("source", []))
                    for cell in cells
                ) + "\n" + "\n".join(
                    json.dumps(output, ensure_ascii=False)
                    for cell in code_cells
                    for output in cell.get("outputs", [])
                    if output.get("output_type") != "display_data"
                    or "image/png" not in output.get("data", {})
                )
                self.assertIsNone(re.search(r"[\u4e00-\u9fff]", public_text))
                self.assertNotIn("/Users/", public_text)
                for term in required_terms[notebook_name]:
                    self.assertIn(term, public_text)
                for claim in forbidden_claims:
                    self.assertNotIn(claim, public_text.casefold())

                output_size = sum(
                    len(json.dumps(cell.get("outputs", []), ensure_ascii=False).encode("utf-8"))
                    for cell in code_cells
                )
                self.assertLessEqual(output_size, MAX_NOTEBOOK_OUTPUT_BYTES)

    def test_notebook_lineage_gate_is_derived_from_stored_audit_hashes(self):
        try:
            from cb_quant.notebook_quality import build_lineage_gate
        except ModuleNotFoundError as error:
            self.fail(f"Missing notebook lineage helper: {error}")

        differing = build_lineage_gate(
            {
                "G0": {"config_sha256": "current"},
                "G1": {"config_sha256": "legacy"},
                "G2": {"config_sha256": "current"},
            }
        )
        matching = build_lineage_gate(
            {
                "G0": {"config_sha256": "unified"},
                "G1": {"config_sha256": "unified"},
                "G2": {"config_sha256": "unified"},
            }
        )

        self.assertEqual(differing["status"], "pending")
        self.assertIn("administratively reconciled metadata difference", differing["detail"])
        self.assertIn("unified final build pending", differing["detail"])
        self.assertEqual(matching["status"], "pass")
        self.assertIn("match", matching["detail"].casefold())
        self.assertNotIn("current", str(differing))
        self.assertNotIn("legacy", str(differing))

        for notebook_name in EXPECTED_NOTEBOOKS:
            notebook = json.loads(
                (self.project_root / "notebooks" / notebook_name).read_text(encoding="utf-8")
            )
            source = "\n".join(
                "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
            )
            with self.subTest(notebook=notebook_name):
                self.assertIn("build_lineage_gate", source)
                self.assertNotIn("cell-for-cell identical", source)
                self.assertNotIn("investable keys verified", source.casefold())

    def test_g1_notebook_rejects_stale_pending_lineage_prose_after_unified_build(self):
        audit_paths = (
            self.project_root / "artifacts" / "G0" / "research_config_validation_v1.json",
            self.project_root / "artifacts" / "G1" / "universe_audit_v1.json",
            self.project_root / "artifacts" / "G2" / "valuation_factor_audit_v1.json",
            self.project_root / "artifacts" / "G3" / "stock_linkage_factor_audit_v1.json",
            self.project_root / "artifacts" / "G3" / "cb_trading_factor_audit_v1.json",
        )
        if not all(path.is_file() for path in audit_paths):
            self.skipTest(
                "licensed-data audit artifacts are intentionally excluded from the release"
            )
        audits = [json.loads(path.read_text(encoding="utf-8")) for path in audit_paths]
        self.assertEqual(len({audit["config_sha256"] for audit in audits}), 1)

        notebook = json.loads(
            (self.project_root / "notebooks" / "01_point_in_time_universe.ipynb").read_text(
                encoding="utf-8"
            )
        )
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )
        outputs = "\n".join(
            json.dumps(output, ensure_ascii=False)
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
            for output in cell.get("outputs", [])
        )
        for stale_phrase in (
            "fingerprints differ",
            "administratively reconciled metadata difference",
            "unified final build pending",
        ):
            with self.subTest(phrase=stale_phrase):
                self.assertNotIn(stale_phrase, source.casefold())
        self.assertIn("Stored stage configuration fingerprints match.", outputs)

    def test_g1_notebook_headline_counts_are_derived_from_audit(self):
        try:
            from cb_quant.notebook_quality import build_g1_headline
        except ModuleNotFoundError as error:
            self.fail(f"Missing G1 headline helper: {error}")

        headline = build_g1_headline(
            {
                "signal_date_count": 7,
                "row_count": 80,
                "eligible_row_count": 9,
                "eligible_bond_count": 10,
            }
        )
        self.assertEqual(headline["Observed value"].tolist(), [7, 80, 9, 10])

        notebook = json.loads(
            (self.project_root / "notebooks" / "01_point_in_time_universe.ipynb").read_text(
                encoding="utf-8"
            )
        )
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )
        self.assertIn("build_g1_headline(g1_audit)", source)

    def test_notebook_00_rating_coverage_is_derived_from_g1_counts(self):
        notebook = json.loads(
            (self.project_root / "notebooks" / "00_data_sources_and_quality.ipynb").read_text(
                encoding="utf-8"
            )
        )
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )
        self.assertIn(
            'g1_counts["eligible_rating_coverage"].mean()',
            source,
        )
        self.assertNotIn('["G1 rating coverage", 1.0]', source)

    def test_g1_funnel_translation_uses_explicit_upstream_stage_names(self):
        try:
            from cb_quant.notebook_quality import (
                G1_FUNNEL_STAGE_LABELS,
                translate_g1_funnel_stages,
            )
        except ModuleNotFoundError as error:
            self.fail(f"Missing G1 funnel translation helper: {error}")

        expected_upstream_names = {
            "\u666e\u901aCB\u4e14\u7814\u7a76\u671f\u5185\u6709\u65e5\u7ebf",
            "\u4e0a\u5e02\u7b2c11\u4e2a\u5e02\u573a\u4ea4\u6613\u65e5\u540e",
            "20\u65e5\u7d2f\u8ba1\u6362\u624b\u7387\u4e0d\u9ad8\u4e8e100%",
            "\u5269\u4f59\u89c4\u6a21\u4e0d\u4f4e\u4e8e2\u4ebf\u5143",
            "\u8bc4\u7ea7\u4e0d\u4f4e\u4e8eA",
            "\u4fe1\u53f7\u65e5\u771f\u5b9e\u6210\u4ea4",
        }
        self.assertEqual(set(G1_FUNNEL_STAGE_LABELS), expected_upstream_names)
        translated = translate_g1_funnel_stages(list(reversed(sorted(expected_upstream_names))))
        self.assertEqual(len(translated), 6)
        self.assertTrue(all(label in G1_FUNNEL_STAGE_LABELS.values() for label in translated))
        with self.assertRaisesRegex(ValueError, "Unrecognized upstream G1 funnel stage"):
            translate_g1_funnel_stages(["unexpected-stage"])

        notebook = json.loads(
            (self.project_root / "notebooks" / "01_point_in_time_universe.ipynb").read_text(
                encoding="utf-8"
            )
        )
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )
        self.assertIn("translate_g1_funnel_stages", source)

    def test_local_only_roots_are_ignored(self):
        for root_name in LOCAL_ONLY_ROOTS:
            with self.subTest(root=root_name):
                result = subprocess.run(
                    [
                        "git",
                        "check-ignore",
                        "--no-index",
                        "--quiet",
                        f"{root_name}/release-audit-placeholder",
                    ],
                    cwd=self.project_root,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, f"{root_name}/ must be ignored")

    def test_tracked_local_only_payloads_are_absent_from_the_worktree(self):
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=self.project_root,
            check=True,
            capture_output=True,
        )
        tracked_paths = result.stdout.decode("utf-8").split("\0")
        offenders = [
            relative_path
            for relative_path in tracked_paths
            if relative_path
            and relative_path.startswith(TRACKED_LOCAL_ONLY_PREFIXES)
            and (self.project_root / relative_path).is_file()
        ]
        self.assertEqual(offenders, [])

    def test_ignored_interim_data_is_not_scanned(self):
        ignored_file = self.project_root / "data/interim/private_local_input.csv"
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", str(ignored_file)],
            cwd=self.project_root,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("data", PUBLIC_RECURSIVE_ROOTS)
        self.assertNotIn(ignored_file, self.public_files())

    def test_public_filenames_are_release_safe(self):
        pattern = re.compile("|".join(FORBIDDEN_PUBLIC_PATTERNS))
        offenders = [
            str(path.relative_to(self.project_root))
            for path in self.public_files()
            if pattern.search(str(path.relative_to(self.project_root)))
        ]
        self.assertEqual(offenders, [])

    def test_public_source_is_release_safe(self):
        pattern = re.compile("|".join(FORBIDDEN_PUBLIC_PATTERNS))
        offenders = []
        for path in self.public_files():
            if path.suffix not in {
                ".cff",
                ".csv",
                ".example",
                ".ipynb",
                ".json",
                ".md",
                ".py",
                ".toml",
                ".txt",
                ".yaml",
                ".yml",
            }:
                continue
            source = path.read_text(encoding="utf-8")
            if pattern.search(source):
                offenders.append(str(path.relative_to(self.project_root)))
        self.assertEqual(offenders, [])

    def test_generic_credential_scanner_catches_synthetic_assignment_and_scans_itself(self):
        pattern = re.compile(CREDENTIAL_ASSIGNMENT_PATTERN)
        synthetic_assignment = "research_api_token = " + ("ab" * 24)
        self.assertIsNotNone(pattern.search(synthetic_assignment))

        test_file = Path(__file__).resolve()
        scanned_files = {path.resolve() for path in self.public_files()}
        self.assertIn(test_file, scanned_files)
        self.assertIsNone(pattern.search(test_file.read_text(encoding="utf-8")))

    def test_research_config_preserves_frozen_behavior_parameters(self):
        config = json.loads(
            (self.project_root / "config" / "research_v1.json").read_text(encoding="utf-8")
        )
        actual = {
            "identity": {
                key: config[key] for key in ("schema_version", "research_id", "status")
            },
            "formal_sample": config["data"]["formal_sample"],
            "research_partitions": config["data"]["research_partitions"],
            "universe": {
                key: value
                for key, value in config["universe"].items()
                if key != "mother_universe"
            },
            "sampling": config["sampling"],
            "returns": config["returns"],
            "factor_evaluation": config["factor_evaluation"],
            "preprocessing": config["preprocessing"],
            "portfolio": config["portfolio"],
            "factor_scope_enums": {
                key: config["factor_scope"][key]
                for key in (
                    "report_groups",
                    "implemented_foundations",
                    "daily_primary_research",
                    "intraday_factors_in_primary_research",
                )
            },
        }
        self.assertEqual(actual, EXPECTED_BEHAVIOR_CONFIG)

    def test_research_config_is_professional_english_release_specification(self):
        config = json.loads(
            (self.project_root / "config" / "research_v1.json").read_text(encoding="utf-8")
        )
        public_text = json.dumps(config, ensure_ascii=False)
        self.assertIsNone(re.search(r"[\u4e00-\u9fff]", public_text))
        self.assertEqual(config["stage_protocol"], EXPECTED_STAGE_PROTOCOL)

        normalized = public_text.casefold()
        for forbidden_term in (
            "teach",
            "tutorial",
            "lesson",
            "reader_facing",
            "walkthrough",
            "resume",
            "interview",
        ):
            with self.subTest(term=forbidden_term):
                self.assertNotIn(forbidden_term, normalized)

        source_roles = json.dumps(config["data"]["source_roles"]).casefold()
        for private_locator in (".duckdb", ".sqlite", ".db", "/users/", "\\users\\"):
            with self.subTest(locator=private_locator):
                self.assertNotIn(private_locator, source_roles)

    def test_research_config_stage_state_matches_completed_pipeline(self):
        config = json.loads(
            (self.project_root / "config" / "research_v1.json").read_text(encoding="utf-8")
        )
        actual = {stage["id"]: stage["status"] for stage in config["stages"]}
        expected = {
            **{f"G{stage}": "complete" for stage in range(4)},
            "G4": "complete",
            "G5": "in_progress",
            "G6": "pending",
            "G7": "in_progress",
            "G8": "pending",
        }
        self.assertEqual(actual, expected)

    def test_methodology_documents_are_consolidated(self):
        methodology_dir = self.project_root / "docs" / "methodology"
        self.assertEqual(
            {path.name for path in methodology_dir.glob("*.md")},
            METHODOLOGY_DOCUMENTS,
        )
        root_copies = [
            filename
            for filename in ROOT_METHODOLOGY_DOCUMENTS
            if (self.project_root / filename).exists()
        ]
        self.assertEqual(root_copies, [])

    def test_public_docs_only_contain_methodology_or_assets(self):
        public_docs = {
            path.name
            for path in (self.project_root / "docs").iterdir()
            if path.is_dir()
        }
        self.assertTrue(public_docs.issubset({"assets", "methodology", "results"}))
        self.assertIn("methodology", public_docs)

    def test_public_markdown_local_links_resolve(self):
        unresolved = []
        link_pattern = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")
        for path in self.public_files():
            if path.suffix != ".md":
                continue
            for target in link_pattern.findall(path.read_text(encoding="utf-8")):
                target = target.strip().split(maxsplit=1)[0]
                if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                    continue
                local_target = target.split("#", maxsplit=1)[0]
                if not (path.parent / local_target).exists():
                    unresolved.append(
                        f"{path.relative_to(self.project_root)} -> {target}"
                    )
        self.assertEqual(unresolved, [])

    def test_showcase_readme_uses_script_reproduction_commands(self):
        readme = (self.project_root / "README.md").read_text(encoding="utf-8")
        for command in README_REPRODUCTION_COMMANDS:
            with self.subTest(command=command):
                self.assertIn(command, readme)

    def test_showcase_readme_provides_ordered_python_311_installation_path(self):
        readme = (self.project_root / "README.md").read_text(encoding="utf-8")
        self.assertIn("Python >=3.11", readme)
        positions = []
        for command in README_SETUP_COMMANDS:
            with self.subTest(command=command):
                self.assertIn(command, readme)
                positions.append(readme.index(command))
        self.assertEqual(positions, sorted(positions))

        activation_position = readme.index("conda activate cb-multifactor")
        python_command_positions = [
            match.start()
            for match in re.finditer(r"(?m)^(?:PYTHONPATH=src )?python\b", readme)
        ]
        self.assertTrue(python_command_positions)
        self.assertGreater(min(python_command_positions), activation_position)

    def test_showcase_readme_links_only_approved_stage_and_result_assets(self):
        readme = (self.project_root / "README.md").read_text(encoding="utf-8")
        image_links = re.findall(r"!\[([^]]+)\]\(([^)]+\.png)\)", readme)
        self.assertLessEqual(len(image_links), 7)
        self.assertEqual(dict(image_links), README_SHOWCASE_ASSETS)
        for caption, target in image_links:
            with self.subTest(asset=target):
                self.assertRegex(caption, r"^G(?:0|1|2|3a|3b|5|7)\b")
                self.assertRegex(Path(target).name, r"^[a-z0-9_]+\.png$")
                self.assertTrue((self.project_root / target).is_file())

    def test_showcase_readme_links_public_notebooks_and_methodology(self):
        readme = (self.project_root / "README.md").read_text(encoding="utf-8")
        markdown_links = dict(re.findall(r"(?<!!)\[([^]]+)\]\(([^)]+)\)", readme))
        for label, target in {
            **README_NOTEBOOK_LINKS,
            **README_METHODOLOGY_LINKS,
        }.items():
            with self.subTest(target=target):
                self.assertEqual(markdown_links.get(label), target)
                self.assertTrue((self.project_root / target).is_file())

    def test_implementation_status_matches_current_stage_gate(self):
        status = (
            self.project_root / "docs" / "methodology" / "implementation_status.md"
        ).read_text(encoding="utf-8")
        for expected_row in EXPECTED_STAGE_STATUS_ROWS:
            with self.subTest(row=expected_row):
                self.assertIn(expected_row, status)
        for document in (
            (self.project_root / "README.md").read_text(encoding="utf-8"),
            status,
        ):
            normalized = " ".join(document.casefold().split())
            self.assertIn("factor predictive power", normalized)
            self.assertIn("strategy performance", normalized)
            self.assertIn("have not yet passed", normalized)

    def test_data_readme_documents_public_data_contract(self):
        data_readme = (self.project_root / "data" / "README.md").read_text(encoding="utf-8")
        for requirement in DATA_README_REQUIREMENTS:
            with self.subTest(requirement=requirement):
                self.assertIn(requirement.casefold(), data_readme.casefold())
        self.assertNotIn("/Users/", data_readme)

    def test_public_release_governance_files_are_present(self):
        required_files = (
            "LICENSE",
            "SECURITY.md",
            "CONTRIBUTING.md",
            "CITATION.cff",
            ".github/workflows/ci.yml",
            ".github/dependabot.yml",
        )
        for relative_path in required_files:
            with self.subTest(path=relative_path):
                self.assertTrue((self.project_root / relative_path).is_file())

    def test_license_and_citation_identify_the_release(self):
        license_text = (self.project_root / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("MIT License", license_text)
        self.assertIn("Copyright (c) 2026 Mingyu Xu", license_text)

        citation = (self.project_root / "CITATION.cff").read_text(encoding="utf-8")
        for term in (
            "cff-version: 1.2.0",
            'title: "Convertible Bond Multi-Factor Research"',
            'version: "0.1.0"',
            "family-names: Xu",
            "given-names: Mingyu",
            "https://github.com/ddxmy/cb-multifactor",
        ):
            with self.subTest(term=term):
                self.assertIn(term, citation)

    def test_pyproject_contains_public_package_metadata(self):
        with (self.project_root / "pyproject.toml").open("rb") as file:
            project = tomllib.load(file)["project"]
        self.assertEqual(project["readme"], "README.md")
        self.assertEqual(project["license"], "MIT")
        self.assertEqual(project["license-files"], ["LICENSE"])
        self.assertEqual(project["authors"], [{"name": "Mingyu Xu"}])
        self.assertEqual(
            project["urls"]["Repository"],
            "https://github.com/ddxmy/cb-multifactor",
        )
        self.assertIn("pandas>=2.1,<3", project["dependencies"])
        self.assertIn("setuptools>=77", project["optional-dependencies"]["dev"])
        self.assertIn("Development Status :: 3 - Alpha", project["classifiers"])
        self.assertIn("Programming Language :: Python :: 3.11", project["classifiers"])
        self.assertNotIn("License :: OSI Approved :: MIT License", project["classifiers"])

    def test_ci_workflow_runs_release_gates(self):
        workflow = (self.project_root / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        for term in (
            "pull_request:",
            "push:",
            "contents: read",
            'python-version: ["3.11", "3.12", "3.13"]',
            'python -m pip install -e ".[dev]"',
            "python -m pytest",
            "python -m compileall -q src scripts",
            "python scripts/validate_research_config.py --check-only",
            "python -m pip check",
            "python -m build",
            "persist-credentials: false",
        ):
            with self.subTest(term=term):
                self.assertIn(term, workflow)
        action_refs = re.findall(r"uses:\s+[^@\s]+@([^\s#]+)", workflow)
        self.assertTrue(action_refs)
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in action_refs))

    def test_dependabot_covers_python_and_github_actions(self):
        dependabot = (self.project_root / ".github" / "dependabot.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn('package-ecosystem: "pip"', dependabot)
        self.assertIn('package-ecosystem: "github-actions"', dependabot)
        self.assertIn('dependency-name: "pandas"', dependabot)
        self.assertIn('update-types: ["version-update:semver-major"]', dependabot)
        self.assertIn('versions: [">=3"]', dependabot)

    def test_readme_displays_ci_and_license_badges(self):
        readme = (self.project_root / "README.md").read_text(encoding="utf-8")
        self.assertIn("actions/workflows/ci.yml/badge.svg", readme)
        self.assertIn("License-MIT", readme)
        self.assertIn("G3c preprocessing review in progress", readme)
        self.assertNotIn("G0-G4 replication pipeline complete", readme)

    def test_security_policy_has_a_safe_reporting_fallback(self):
        policy = (self.project_root / "SECURITY.md").read_text(encoding="utf-8")
        self.assertIn("If the private reporting button is unavailable", policy)
        self.assertIn("do not disclose vulnerability details", policy)
