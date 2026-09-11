"""Explicit catalog of production factor plugins."""

from __future__ import annotations

from .cb_abnormal_turnover_20d import (
    FACTOR_SPEC as CB_ABNORMAL_TURNOVER_20D_SPEC,
    calculate_factor as calculate_cb_abnormal_turnover_20d,
)
from .cb_return_20d import (
    FACTOR_SPEC as CB_RETURN_20D_SPEC,
    calculate_factor as calculate_cb_return_20d,
)
from .cb_stock_return_spread import (
    FACTOR_SPEC as CB_STOCK_RETURN_SPREAD_SPEC,
    calculate_factor as calculate_cb_stock_return_spread,
)
from .double_low import (
    FACTOR_SPEC as DOUBLE_LOW_SPEC,
    calculate_factor as calculate_double_low,
)
from .relative_premium_to_parity_median import (
    FACTOR_SPEC as RELATIVE_PREMIUM_TO_PARITY_MEDIAN_SPEC,
    calculate_factor as calculate_relative_premium_to_parity_median,
)
from .registry import FactorRegistry
from .stock_return_20d import (
    FACTOR_SPEC as STOCK_RETURN_20D_SPEC,
    calculate_factor as calculate_stock_return_20d,
)
from .valuation import CONVERSION_PREMIUM_SPEC, calculate_conversion_premium


def build_default_factor_registry() -> FactorRegistry:
    """Return a fresh registry containing approved production factors."""
    registry = FactorRegistry()
    registry.register(
        CB_ABNORMAL_TURNOVER_20D_SPEC,
        calculate_cb_abnormal_turnover_20d,
    )
    registry.register(CB_RETURN_20D_SPEC, calculate_cb_return_20d)
    registry.register(
        CB_STOCK_RETURN_SPREAD_SPEC,
        calculate_cb_stock_return_spread,
    )
    registry.register(CONVERSION_PREMIUM_SPEC, calculate_conversion_premium)
    registry.register(DOUBLE_LOW_SPEC, calculate_double_low)
    registry.register(
        RELATIVE_PREMIUM_TO_PARITY_MEDIAN_SPEC,
        calculate_relative_premium_to_parity_median,
    )
    registry.register(STOCK_RETURN_20D_SPEC, calculate_stock_return_20d)
    return registry
