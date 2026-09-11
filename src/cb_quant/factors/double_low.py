"""Double-low valuation factor for convertible bonds."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .base import FactorSpec
from .context import FactorDataBundle
from .output import build_factor_output


FACTOR_SPEC = FactorSpec(
    name="double_low",
    family="valuation",
    description=(
        "Convertible-bond price combined with its point-in-time conversion "
        "premium percentage."
    ),
    direction=-1,
    required_fields=("close", "conversion_premium"),
    required_history_fields={},
    default_parameters={},
    neutralizers=(),
)


def calculate_raw_factor(
    data: FactorDataBundle,
    parameters: Mapping[str, object],
) -> pd.Series:
    """Calculate the researcher-owned raw double-low formula."""
    if parameters:
        unknown = sorted(parameters)
        raise ValueError(f"double_low does not accept parameters: {unknown}")
    signal = data.signal

    raw_factor = signal['close'] + 100 * signal['conversion_premium']
    return raw_factor


def calculate_factor(
    data: FactorDataBundle,
    parameters: Mapping[str, object],
) -> pd.DataFrame:
    """Wrap the raw formula in the framework's standard output contract."""
    raw_factor = calculate_raw_factor(data, parameters)
    return build_factor_output(data.signal, raw_factor)
