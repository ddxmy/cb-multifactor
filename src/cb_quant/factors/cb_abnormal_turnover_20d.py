"""Convertible-bond 20-market-day abnormal-turnover factor."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .base import FactorSpec
from .context import FactorDataBundle
from .output import build_factor_output


FACTOR_SPEC = FactorSpec(
    name="cb_abnormal_turnover_20d",
    family="convertible_bond_trading",
    description=(
        "Current convertible-bond turnover divided by its prior 20-market-day "
        "mean, oriented as a crowding-reversal signal."
    ),
    direction=-1,
    required_fields=("cb_abnormal_turnover_20d",),
    default_parameters={},
    neutralizers=(),
)


def calculate_factor(
    data: FactorDataBundle,
    parameters: Mapping[str, object],
) -> pd.DataFrame:
    """Expose the audited point-in-time abnormal-turnover field."""
    if parameters:
        unknown = sorted(parameters)
        raise ValueError(
            f"cb_abnormal_turnover_20d does not accept parameters: {unknown}"
        )
    raw_factor = pd.to_numeric(
        data.signal["cb_abnormal_turnover_20d"], errors="coerce"
    )
    return build_factor_output(data.signal, raw_factor)
