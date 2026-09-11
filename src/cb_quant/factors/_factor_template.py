"""Copy this module when implementing a reviewed production factor."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .base import FactorSpec
from .context import FactorDataBundle
from .output import build_factor_output


FACTOR_SPEC = FactorSpec(
    name="factor_template",
    family="replace_with_family",
    description="Replace with the factor's economic interpretation.",
    direction=1,
    required_fields=("source_column",),
    required_history_fields={},
    default_parameters={"window": 20},
    neutralizers=(),
)


def calculate_raw_factor(
    data: FactorDataBundle,
    parameters: Mapping[str, object],
) -> pd.Series:
    """Implement only the reviewed point-in-time formula in this function."""
    raise NotImplementedError("replace with the factor formula")


def calculate_factor(
    data: FactorDataBundle,
    parameters: Mapping[str, object],
) -> pd.DataFrame:
    """Wrap the raw formula in the framework's standard output contract."""
    raw_factor = calculate_raw_factor(data, parameters)
    return build_factor_output(data.signal, raw_factor)
