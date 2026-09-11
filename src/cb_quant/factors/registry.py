"""Registry and execution boundary for factor plugins."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pandas as pd

from .base import FactorSpec, validate_factor_frame
from .context import FactorDataBundle, ensure_factor_data_bundle


FactorCalculator = Callable[
    [FactorDataBundle, Mapping[str, object]],
    pd.DataFrame,
]


class FactorRegistry:
    """Register named factor formulas behind one validated interface."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[FactorSpec, FactorCalculator]] = {}

    def register(self, spec: FactorSpec, calculator: FactorCalculator) -> None:
        if spec.name in self._entries:
            raise ValueError(f"factor {spec.name!r} is already registered")
        if not callable(calculator):
            raise TypeError("factor calculator must be callable")
        self._entries[spec.name] = (spec, calculator)

    def get_spec(self, name: str) -> FactorSpec:
        try:
            return self._entries[name][0]
        except KeyError as error:
            raise KeyError(f"unknown factor: {name}") from error

    def calculate(
        self,
        name: str,
        context: FactorDataBundle | pd.DataFrame,
        parameters: Mapping[str, object] | None = None,
    ) -> pd.DataFrame:
        try:
            spec, calculator = self._entries[name]
        except KeyError as error:
            raise KeyError(f"unknown factor: {name}") from error
        data = ensure_factor_data_bundle(context)
        missing = set(spec.required_fields).difference(data.signal.columns)
        if missing:
            raise KeyError(f"factor context is missing required fields: {sorted(missing)}")
        for table, required in spec.required_history_fields.items():
            frame = getattr(data, table)
            if missing := set(required).difference(frame.columns):
                raise KeyError(
                    f"factor history table {table} is missing fields: {sorted(missing)}"
                )
        settings = dict(spec.default_parameters)
        settings.update(dict(parameters or {}))
        result = validate_factor_frame(calculator(data, settings), spec)
        context_keys = data.signal[["signal_date", "ts_code"]].copy()
        context_keys["signal_date"] = pd.to_datetime(
            context_keys["signal_date"], errors="raise"
        ).dt.normalize()
        context_keys["ts_code"] = context_keys["ts_code"].astype("string")
        allowed = pd.MultiIndex.from_frame(context_keys)
        output = pd.MultiIndex.from_frame(result[["signal_date", "ts_code"]])
        if not output.isin(allowed).all():
            raise ValueError("factor output contains keys outside the supplied context")
        return result

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))
