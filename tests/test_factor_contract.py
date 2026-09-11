from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cb_quant.factors import (
    FactorDataBundle,
    FactorRegistry,
    FactorSpec,
    validate_factor_frame,
)


def make_spec() -> FactorSpec:
    return FactorSpec(
        name="test_value",
        family="valuation",
        description="Synthetic valuation factor used by contract tests.",
        direction=-1,
        required_fields=("cb_close", "conversion_value"),
        default_parameters={"scale": 1.0},
        neutralizers=("rating",),
    )


def test_validate_factor_frame_normalizes_and_sorts_standard_contract() -> None:
    frame = pd.DataFrame(
        {
            "signal_date": ["2024-01-05", "2024-01-04"],
            "ts_code": ["110002.SH", "110001.SH"],
            "raw_factor": [0.2, np.nan],
            "available_date": ["2024-01-05", "2024-01-03"],
        }
    )

    result = validate_factor_frame(frame, make_spec())

    assert result["signal_date"].tolist() == [
        pd.Timestamp("2024-01-04"),
        pd.Timestamp("2024-01-05"),
    ]
    assert pd.api.types.is_datetime64_ns_dtype(result["available_date"])
    assert result["ts_code"].dtype.name == "string"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(columns="raw_factor"), "missing columns"),
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "duplicate",
        ),
        (lambda frame: frame.assign(raw_factor="not-numeric"), "numeric"),
        (lambda frame: frame.assign(raw_factor=np.inf), "finite"),
        (
            lambda frame: frame.assign(available_date="2024-01-06"),
            "later than signal_date",
        ),
        (lambda frame: frame.assign(available_date=pd.NaT), "cannot be missing"),
    ],
)
def test_validate_factor_frame_rejects_invalid_contract(mutation, message) -> None:
    valid = pd.DataFrame(
        {
            "signal_date": ["2024-01-05"],
            "ts_code": ["110001.SH"],
            "raw_factor": [0.1],
            "available_date": ["2024-01-05"],
        }
    )

    with pytest.raises((KeyError, TypeError, ValueError), match=message):
        validate_factor_frame(mutation(valid), make_spec())


def test_registry_merges_parameters_and_validates_required_context() -> None:
    registry = FactorRegistry()
    captured: dict[str, object] = {}

    def calculator(data: FactorDataBundle, parameters) -> pd.DataFrame:
        context = data.signal
        captured.update(parameters)
        return context.assign(
            raw_factor=(
                context["cb_close"] / context["conversion_value"] - 1.0
            ),
            available_date=context["signal_date"],
        ).assign(
            raw_factor=lambda frame: frame["raw_factor"]
            * float(parameters["scale"])
        )[["signal_date", "ts_code", "raw_factor", "available_date"]]

    registry.register(make_spec(), calculator)
    context = pd.DataFrame(
        {
            "signal_date": ["2024-01-05"],
            "ts_code": ["110001.SH"],
            "cb_close": [120.0],
            "conversion_value": [100.0],
        }
    )

    result = registry.calculate("test_value", context, {"scale": 2.0})

    assert captured == {"scale": 2.0}
    assert result.loc[0, "raw_factor"] == pytest.approx(0.4)
    with pytest.raises(KeyError, match="conversion_value"):
        registry.calculate("test_value", context.drop(columns="conversion_value"))


def test_registry_rejects_duplicate_names_and_unknown_factor() -> None:
    registry = FactorRegistry()
    calculator = lambda context, parameters: context
    registry.register(make_spec(), calculator)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(make_spec(), calculator)
    with pytest.raises(KeyError, match="unknown factor"):
        registry.calculate("missing", pd.DataFrame())


def test_registry_requires_availability_audit_and_context_key_subset() -> None:
    context = pd.DataFrame(
        {
            "signal_date": ["2024-01-05"],
            "ts_code": ["110001.SH"],
            "cb_close": [120.0],
            "conversion_value": [100.0],
        }
    )
    registry = FactorRegistry()

    def missing_availability(data, parameters):
        context = data.signal
        return context.assign(raw_factor=1.0)[
            ["signal_date", "ts_code", "raw_factor"]
        ]

    registry.register(make_spec(), missing_availability)
    with pytest.raises(KeyError, match="available_date"):
        registry.calculate("test_value", context)

    registry = FactorRegistry()

    def invented_key(context, parameters):
        return pd.DataFrame(
            {
                "signal_date": ["2024-01-05"],
                "ts_code": ["999999.SH"],
                "raw_factor": [1.0],
                "available_date": ["2024-01-05"],
            }
        )

    registry.register(make_spec(), invented_key)
    with pytest.raises(ValueError, match="outside the supplied context"):
        registry.calculate("test_value", context)


def test_registry_validates_declared_history_table_fields() -> None:
    spec = FactorSpec(
        name="stock_history_value",
        family="underlying_equity",
        description="Synthetic factor that requires underlying-equity history.",
        direction=1,
        required_fields=("stk_code",),
        required_history_fields={
            "stock_daily": ("trade_date", "stk_code", "close", "adj_factor"),
        },
    )
    registry = FactorRegistry()
    captured: dict[str, object] = {}

    def calculator(data: FactorDataBundle, parameters) -> pd.DataFrame:
        captured["data"] = data
        return data.signal.assign(
            raw_factor=1.0,
            available_date=data.signal["signal_date"],
        )[["signal_date", "ts_code", "raw_factor", "available_date"]]

    registry.register(spec, calculator)
    signal = pd.DataFrame(
        {
            "signal_date": ["2024-01-05"],
            "ts_code": ["110001.SH"],
            "stk_code": ["600001.SH"],
        }
    )
    incomplete = FactorDataBundle(
        signal=signal,
        stock_daily=pd.DataFrame(
            {
                "trade_date": ["2024-01-04"],
                "stk_code": ["600001.SH"],
                "adj_factor": [1.0],
            }
        ),
    )

    with pytest.raises(KeyError, match="stock_daily.*close"):
        registry.calculate("stock_history_value", incomplete)

    complete = FactorDataBundle(
        signal=signal,
        stock_daily=incomplete.stock_daily.assign(close=10.0),
    )
    result = registry.calculate("stock_history_value", complete)

    assert captured["data"] is complete
    assert result.loc[0, "raw_factor"] == 1.0


def test_factor_spec_rejects_unknown_history_table() -> None:
    with pytest.raises(ValueError, match="history table"):
        FactorSpec(
            name="bad_history",
            family="test",
            description="Invalid history-table declaration.",
            direction=1,
            required_fields=(),
            required_history_fields={"database": ("close",)},
        )
