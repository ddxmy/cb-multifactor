"""Underlying-stock 20-market-day return factor."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .base import FactorSpec
from .context import FactorDataBundle
from .output import build_factor_output


FACTOR_SPEC = FactorSpec(
    name="stock_return_20d",
    family="underlying_equity",
    description=(
        "Twenty-market-day simple return of the underlying stock using "
        "point-in-time adjusted closes."
    ),
    direction=1,
    required_fields=("stock_return_20d",),
    default_parameters={},
    neutralizers=("log_stock_total_market_cap", "ci_industry_control"),
)


def calculate_factor(
    data: FactorDataBundle,
    parameters: Mapping[str, object],
) -> pd.DataFrame:
    """Expose the audited point-in-time stock-return field."""
    if parameters:
        unknown = sorted(parameters)
        raise ValueError(f"stock_return_20d does not accept parameters: {unknown}")
    raw_factor = pd.to_numeric(data.signal["stock_return_20d"], errors="coerce")
    return build_factor_output(data.signal, raw_factor)
