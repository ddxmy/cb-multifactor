"""Validation for the reusable single-factor backtest configuration."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping


REQUIRED_SECTIONS = {
    "schema_version",
    "configuration_id",
    "status",
    "timing",
    "preprocessing",
    "evaluation",
    "portfolio",
    "execution",
    "lifecycle",
    "parameter_status",
}
FACTOR_BATCH_REQUIRED_FIELDS = {
    "schema_version",
    "batch_id",
    "sample_partition",
    "factors",
}
SAMPLE_PARTITION_BOUNDS = {
    "replication_2018_2023": ("2018-01-01", "2023-06-21"),
    "validation_2023_2024": ("2023-06-22", "2024-12-31"),
    "holdout_2025_onward": ("2025-01-01", None),
}
SAMPLE_PARTITIONS = set(SAMPLE_PARTITION_BOUNDS)
FACTOR_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def validate_backtest_config(config: Mapping[str, Any]) -> list[str]:
    """Return all structural and behavioral configuration errors."""
    errors: list[str] = []
    missing = REQUIRED_SECTIONS.difference(config)
    if missing:
        return [f"missing top-level fields: {sorted(missing)}"]

    timing = config["timing"]
    expected_timing = {
        "signal_time": "signal_day_close",
        "execution_time": "next_market_trading_day_open",
        "primary_label": "next_execution_open_to_open",
        "daily_mark": "market_close",
    }
    for key, expected in expected_timing.items():
        if timing.get(key) != expected:
            errors.append(f"timing.{key} must remain {expected!r}")

    preprocessing = config["preprocessing"]
    if not _positive_number(preprocessing.get("mad_multiplier")):
        errors.append("preprocessing.mad_multiplier must be positive")

    evaluation = config["evaluation"]
    for key in ("quantile_groups", "minimum_ic_assets", "periods_per_year"):
        if not _positive_integer(evaluation.get(key)):
            errors.append(f"evaluation.{key} must be a positive integer")
    if not _non_negative_integer(evaluation.get("newey_west_lags")):
        errors.append("evaluation.newey_west_lags must be a non-negative integer")
    horizons = evaluation.get("forward_return_horizons_market_days")
    valid_horizons = (
        isinstance(horizons, list)
        and bool(horizons)
        and all(_positive_integer(value) for value in horizons)
        and len(set(horizons)) == len(horizons)
    )
    if not valid_horizons or horizons != [1, 5, 10]:
        errors.append(
            "evaluation.forward_return_horizons_market_days must remain [1, 5, 10]"
        )

    portfolio = config["portfolio"]
    portfolio_size = portfolio.get("portfolio_size")
    cash_reserve = portfolio.get("cash_reserve")
    maximum_weight = portfolio.get("maximum_single_name_weight")
    if not _positive_integer(portfolio_size):
        errors.append("portfolio.portfolio_size must be a positive integer")
    if not _unit_interval(cash_reserve, include_one=False):
        errors.append("portfolio.cash_reserve must be in [0, 1)")
    if not _unit_interval(maximum_weight, include_zero=False):
        errors.append("portfolio.maximum_single_name_weight must be in (0, 1]")
    if not _positive_number(portfolio.get("initial_cash_yuan")):
        errors.append("portfolio.initial_cash_yuan must be positive")
    if (
        _positive_integer(portfolio_size)
        and _unit_interval(cash_reserve, include_one=False)
        and _unit_interval(maximum_weight, include_zero=False)
        and portfolio_size * maximum_weight < 1.0 - cash_reserve - 1e-12
    ):
        errors.append(
            "portfolio constraints cannot deploy the requested non-cash allocation"
        )

    execution = config["execution"]
    if not _non_negative_number(execution.get("one_way_cost_bps")):
        errors.append("execution.one_way_cost_bps must be non-negative")
    if not _positive_integer(execution.get("board_lot")):
        errors.append("execution.board_lot must be a positive integer")
    if execution.get("sell_before_buy") is not True:
        errors.append("execution.sell_before_buy must remain true")
    if execution.get("missing_open_policy") != "no_fill_without_forward_fill":
        errors.append("execution.missing_open_policy cannot assume a fill")
    if (
        execution.get("input_coverage_policy")
        != "require_complete_tradability_and_lifecycle"
    ):
        errors.append(
            "execution.input_coverage_policy must require complete tradability and lifecycle"
        )
    expected_execution_policies = {
        "failed_sell_policy": "continue_holding_and_retry",
        "failed_buy_policy": "retain_cash_without_automatic_retry",
    }
    for key, expected in expected_execution_policies.items():
        if execution.get(key) != expected:
            errors.append(f"execution.{key} must remain {expected!r}")

    lifecycle = config["lifecycle"]
    expected_lifecycle_policies = {
        "announced_redemption_blocks_entry": True,
        "validated_terminal_terms_create_receivable": True,
        "unresolved_terminal_settlement_policy": "raise_and_audit",
    }
    for key, expected in expected_lifecycle_policies.items():
        if lifecycle.get(key) != expected:
            errors.append(f"lifecycle.{key} must remain {expected!r}")
    return errors


def load_backtest_config(path: str | Path) -> dict[str, Any]:
    """Load JSON configuration and fail with a consolidated error message."""
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    errors = validate_backtest_config(config)
    if errors:
        raise ValueError("invalid backtest configuration: " + "; ".join(errors))
    return config


def validate_factor_batch_config(config: Mapping[str, Any]) -> list[str]:
    """Return structural errors for a batch of comparable factor runs."""
    errors: list[str] = []
    missing = FACTOR_BATCH_REQUIRED_FIELDS.difference(config)
    if missing:
        return [f"missing top-level fields: {sorted(missing)}"]
    if config.get("schema_version") != "1.0":
        errors.append("schema_version must be '1.0'")
    batch_id = config.get("batch_id")
    if not isinstance(batch_id, str) or not FACTOR_NAME_PATTERN.fullmatch(batch_id):
        errors.append("batch_id must be lower snake_case")
    if config.get("sample_partition") not in SAMPLE_PARTITIONS:
        errors.append(
            "sample_partition must be replication_2018_2023, "
            "validation_2023_2024, or holdout_2025_onward"
        )

    factors = config.get("factors")
    if not isinstance(factors, list) or not factors:
        errors.append("factors must be a non-empty list")
        return errors
    signatures: set[str] = set()
    for index, item in enumerate(factors):
        prefix = f"factors[{index}]"
        if not isinstance(item, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        if set(item) != {"name", "parameters"}:
            errors.append(f"{prefix} must contain only name and parameters")
        name = item.get("name")
        if not isinstance(name, str) or not FACTOR_NAME_PATTERN.fullmatch(name):
            errors.append(f"{prefix}.name must be lower snake_case")
        parameters = item.get("parameters")
        if not isinstance(parameters, Mapping):
            errors.append(f"{prefix}.parameters must be an object")
            continue
        signature = json.dumps(
            {"name": name, "parameters": dict(parameters)},
            sort_keys=True,
            separators=(",", ":"),
        )
        if signature in signatures:
            errors.append(f"{prefix} duplicates an earlier factor and parameter set")
        signatures.add(signature)
    return errors


def load_factor_batch_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a factor-batch JSON configuration."""
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    errors = validate_factor_batch_config(config)
    if errors:
        raise ValueError("invalid factor batch configuration: " + "; ".join(errors))
    return config


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _non_negative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_number(value: object) -> bool:
    return _non_negative_number(value) and float(value) > 0


def _non_negative_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value) >= 0
    )


def _unit_interval(
    value: object,
    *,
    include_zero: bool = True,
    include_one: bool = True,
) -> bool:
    if not _non_negative_number(value):
        return False
    numeric = float(value)
    lower = numeric >= 0 if include_zero else numeric > 0
    upper = numeric <= 1 if include_one else numeric < 1
    return lower and upper
