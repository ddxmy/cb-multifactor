"""Standard contracts for convertible-bond factor plugins."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd


_FACTOR_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
STANDARD_FACTOR_COLUMNS = ("signal_date", "ts_code", "raw_factor", "available_date")
FACTOR_HISTORY_TABLES = frozenset(
    {"valuation", "cb_daily", "stock_daily", "lifecycle"}
)


@dataclass(frozen=True)
class FactorSpec:
    """Immutable metadata describing a point-in-time factor formula."""

    name: str
    family: str
    description: str
    direction: int
    required_fields: tuple[str, ...]
    required_history_fields: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    default_parameters: Mapping[str, object] = field(default_factory=dict)
    neutralizers: tuple[str, ...] = ()
    version: str = "1.0"

    def __post_init__(self) -> None:
        if not _FACTOR_NAME.fullmatch(self.name):
            raise ValueError("factor name must be lower snake_case")
        if not self.family.strip():
            raise ValueError("factor family cannot be empty")
        if not self.description.strip():
            raise ValueError("factor description cannot be empty")
        if self.direction not in {-1, 1}:
            raise ValueError("factor direction must be -1 or 1")
        if len(set(self.required_fields)) != len(self.required_fields):
            raise ValueError("required_fields cannot contain duplicates")
        history_fields: dict[str, tuple[str, ...]] = {}
        for table, columns in self.required_history_fields.items():
            if table not in FACTOR_HISTORY_TABLES:
                raise ValueError(f"unknown factor history table: {table}")
            normalized = tuple(columns)
            if not normalized:
                raise ValueError(f"history table {table} must declare fields")
            if len(set(normalized)) != len(normalized):
                raise ValueError(
                    f"history table {table} fields cannot contain duplicates"
                )
            history_fields[table] = normalized
        object.__setattr__(
            self,
            "required_history_fields",
            MappingProxyType(history_fields),
        )
        object.__setattr__(
            self,
            "default_parameters",
            MappingProxyType(dict(self.default_parameters)),
        )


def validate_factor_frame(frame: pd.DataFrame, spec: FactorSpec) -> pd.DataFrame:
    """Validate and normalize a factor plugin's standard output."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("factor output must be a pandas DataFrame")
    missing = set(STANDARD_FACTOR_COLUMNS).difference(frame.columns)
    if missing:
        raise KeyError(f"factor output is missing columns: {sorted(missing)}")

    result = frame.copy()
    result["signal_date"] = pd.to_datetime(result["signal_date"], errors="raise").dt.normalize()
    result["ts_code"] = result["ts_code"].astype("string")
    if result[["signal_date", "ts_code"]].isna().any().any():
        raise ValueError("signal_date and ts_code cannot be missing")
    if result.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError("factor output contains duplicate signal-date and bond keys")
    if not pd.api.types.is_numeric_dtype(result["raw_factor"]):
        raise TypeError("raw_factor must have a numeric dtype")
    finite = result["raw_factor"].dropna().to_numpy(dtype=float)
    if not np.isfinite(finite).all():
        raise ValueError("raw_factor values must be finite or missing")

    result["available_date"] = pd.to_datetime(
        result["available_date"], errors="raise"
    ).dt.normalize()
    factor_observed = result["raw_factor"].notna()
    if result.loc[factor_observed, "available_date"].isna().any():
        raise ValueError("available_date cannot be missing for an observed factor")
    future = result["available_date"].notna() & result["available_date"].gt(
        result["signal_date"]
    )
    if future.any():
        raise ValueError("available_date cannot be later than signal_date")

    return result.sort_values(["signal_date", "ts_code"]).reset_index(drop=True)
