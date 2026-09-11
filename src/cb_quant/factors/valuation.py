"""Valuation-factor plugins backed by the existing point-in-time panel."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .base import FactorSpec
from .context import FactorDataBundle


CONVERSION_PREMIUM_SPEC = FactorSpec(
    name="conversion_premium",
    family="valuation",
    description="Bond price premium relative to point-in-time conversion value.",
    direction=-1,
    required_fields=("conversion_premium",),
    default_parameters={},
    neutralizers=("conversion_value", "rating", "remaining_maturity", "remaining_balance"),
)


def calculate_conversion_premium(
    data: FactorDataBundle,
    parameters: Mapping[str, object],
) -> pd.DataFrame:
    """Expose the validated G2 conversion premium through the factor contract."""
    if parameters:
        unknown = sorted(parameters)
        raise ValueError(f"conversion_premium does not accept parameters: {unknown}")
    result = data.signal[
        ["signal_date", "ts_code", "conversion_premium"]
    ].rename(
        columns={"conversion_premium": "raw_factor"}
    )
    result["available_date"] = result["signal_date"]
    return result
