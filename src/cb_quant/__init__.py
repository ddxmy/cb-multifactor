"""Reusable components for the convertible bond multi-factor project."""

from .data_catalog import DataCatalog
from .event_state import attach_asof_event_state, prepare_rating_events, prepare_share_events
from .industry import attach_ci_industry, load_ci_industry_membership
from .g1_universe import (
    UniverseRules,
    audit_event_boundaries,
    audit_universe_panel,
    build_biweekly_universe_panel,
    build_rebalance_calendar,
    build_universe_transitions,
    summarize_universe_by_date,
)
from .industry import load_ci_industry_membership_panel
from .tushare_history import (
    create_tushare_client,
    download_cb_rating_history,
    download_cb_share_history,
    load_cb_codes_from_terms,
)
from .universe import build_daily_universe_snapshot, load_cb_reference_universe, load_open_trading_dates, universe_funnel
from .valuation import (
    audit_valuation_panel,
    build_conversion_valuation,
    build_full_valuation_panel,
    build_parity_relative_premium,
    build_valuation_anomaly_review,
    summarize_parity_bucket_coverage,
)
from .stock_data import load_a_share_daily, load_a_share_market_cap
from .stock_linkage import (
    LINKAGE_FACTOR_COLUMNS,
    STOCK_FACTOR_COLUMNS,
    STOCK_LINKAGE_FACTOR_COLUMNS,
    audit_stock_linkage_factor_panel,
    build_stock_linkage_factor_panel,
)
from .cb_trading import (
    CB_TRADING_FACTOR_COLUMNS,
    audit_cb_trading_factor_panel,
    build_cb_trading_factor_panel,
)
from .plotting import PlotFontConfig, configure_matplotlib
from .pipeline import SingleFactorResearchResult, run_single_factor_research

__all__ = [
    "DataCatalog",
    "attach_ci_industry",
    "UniverseRules",
    "audit_event_boundaries",
    "audit_universe_panel",
    "build_biweekly_universe_panel",
    "build_rebalance_calendar",
    "build_universe_transitions",
    "attach_asof_event_state",
    "create_tushare_client",
    "download_cb_rating_history",
    "download_cb_share_history",
    "load_ci_industry_membership",
    "load_ci_industry_membership_panel",
    "load_cb_codes_from_terms",
    "prepare_rating_events",
    "prepare_share_events",
    "summarize_universe_by_date",
    "build_daily_universe_snapshot",
    "load_cb_reference_universe",
    "load_open_trading_dates",
    "universe_funnel",
    "build_conversion_valuation",
    "build_full_valuation_panel",
    "build_parity_relative_premium",
    "build_valuation_anomaly_review",
    "summarize_parity_bucket_coverage",
    "audit_valuation_panel",
    "load_a_share_daily",
    "load_a_share_market_cap",
    "LINKAGE_FACTOR_COLUMNS",
    "STOCK_FACTOR_COLUMNS",
    "STOCK_LINKAGE_FACTOR_COLUMNS",
    "audit_stock_linkage_factor_panel",
    "build_stock_linkage_factor_panel",
    "CB_TRADING_FACTOR_COLUMNS",
    "audit_cb_trading_factor_panel",
    "build_cb_trading_factor_panel",
    "PlotFontConfig",
    "configure_matplotlib",
    "SingleFactorResearchResult",
    "run_single_factor_research",
]
