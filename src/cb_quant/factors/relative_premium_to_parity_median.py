"""Same-parity relative conversion-premium factor."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .base import FactorSpec
from .context import FactorDataBundle
from .output import build_factor_output


FACTOR_SPEC = FactorSpec(
    name="relative_premium_to_parity_median",
    family="valuation",
    description=(
        "Conversion premium minus the signal-date median premium of bonds in "
        "the same conversion-value bucket."
    ),
    direction=-1,
    required_fields=("relative_premium_to_parity_median",),
    default_parameters={},
    neutralizers=("log_remaining_balance", "remaining_maturity_years", "rating"),
)


def calculate_factor(
    data: FactorDataBundle,
    parameters: Mapping[str, object],
) -> pd.DataFrame:
    """Expose the audited point-in-time relative-premium field."""
    if parameters:
        unknown = sorted(parameters)
        raise ValueError(
            "relative_premium_to_parity_median does not accept parameters: "
            f"{unknown}"
        )
    raw_factor = pd.to_numeric(
        data.signal["relative_premium_to_parity_median"], errors="coerce"
    )
    return build_factor_output(data.signal, raw_factor)
