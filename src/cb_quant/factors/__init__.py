"""Reusable convertible-bond factor plugin interfaces."""

from .base import FactorSpec, STANDARD_FACTOR_COLUMNS, validate_factor_frame
from .catalog import build_default_factor_registry
from .context import FactorDataBundle, ensure_factor_data_bundle
from .output import build_factor_output
from .registry import FactorCalculator, FactorRegistry

__all__ = [
    "FactorCalculator",
    "FactorDataBundle",
    "FactorRegistry",
    "FactorSpec",
    "STANDARD_FACTOR_COLUMNS",
    "build_factor_output",
    "build_default_factor_registry",
    "ensure_factor_data_bundle",
    "validate_factor_frame",
]
