from __future__ import annotations

import json
from pathlib import Path

import pytest

from cb_quant.config import (
    load_backtest_config,
    load_factor_batch_config,
    validate_backtest_config,
    validate_factor_batch_config,
)
from scripts.run_single_factor import _sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_repository_backtest_config_is_valid_and_keeps_frozen_timing() -> None:
    config = load_backtest_config(PROJECT_ROOT / "config" / "backtest_v1.json")

    assert config["timing"]["signal_time"] == "signal_day_close"
    assert config["timing"]["execution_time"] == "next_market_trading_day_open"
    assert config["timing"]["primary_label"] == "next_execution_open_to_open"
    assert config["evaluation"]["forward_return_horizons_market_days"] == [1, 5, 10]
    assert config["execution"]["board_lot"] == 10


def test_rejects_infeasible_portfolio_and_timing_changes() -> None:
    config = json.loads(
        (PROJECT_ROOT / "config" / "backtest_v1.json").read_text(encoding="utf-8")
    )
    config["timing"]["execution_time"] = "signal_day_close"
    config["portfolio"]["portfolio_size"] = 2
    config["portfolio"]["maximum_single_name_weight"] = 0.1

    errors = validate_backtest_config(config)

    assert any("execution_time" in error for error in errors)
    assert any("cannot deploy" in error for error in errors)


def test_loader_raises_with_all_validation_errors(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text('{"schema_version": "1.0"}', encoding="utf-8")

    with pytest.raises(ValueError, match="missing top-level fields"):
        load_backtest_config(path)


@pytest.mark.parametrize(
    "horizons", [[], [0, 5, 10], [1, 5, 5], [1, 5], "1,5,10"]
)
def test_rejects_invalid_forward_return_horizons(horizons: object) -> None:
    config = json.loads(
        (PROJECT_ROOT / "config" / "backtest_v1.json").read_text(encoding="utf-8")
    )
    config["evaluation"]["forward_return_horizons_market_days"] = horizons

    errors = validate_backtest_config(config)

    assert any("forward_return_horizons_market_days" in error for error in errors)


def test_rejects_changes_to_frozen_execution_and_lifecycle_policies() -> None:
    config = json.loads(
        (PROJECT_ROOT / "config" / "backtest_v1.json").read_text(encoding="utf-8")
    )
    config["execution"]["failed_sell_policy"] = "drop_order"
    config["execution"]["failed_buy_policy"] = "retry_next_day"
    config["lifecycle"]["announced_redemption_blocks_entry"] = False
    config["lifecycle"]["validated_terminal_terms_create_receivable"] = False
    config["lifecycle"]["unresolved_terminal_settlement_policy"] = "carry_last_mark"

    errors = validate_backtest_config(config)

    assert len(errors) == 5
    assert any("failed_sell_policy" in error for error in errors)
    assert any("failed_buy_policy" in error for error in errors)
    assert sum("lifecycle." in error for error in errors) == 3


def test_input_fingerprint_changes_when_file_content_changes(tmp_path: Path) -> None:
    path = tmp_path / "factor_context.csv"
    path.write_text("value\n1\n", encoding="utf-8")
    first = _sha256_file(path)
    path.write_text("value\n2\n", encoding="utf-8")

    assert _sha256_file(path) != first


def test_repository_factor_batch_config_is_valid() -> None:
    config = load_factor_batch_config(
        PROJECT_ROOT / "config" / "factor_batch_v1.json"
    )

    assert config["sample_partition"] == "replication_2018_2023"
    assert config["factors"] == [
        {"name": "conversion_premium", "parameters": {}},
        {"name": "double_low", "parameters": {}},
        {"name": "relative_premium_to_parity_median", "parameters": {}},
        {"name": "stock_return_20d", "parameters": {}},
        {"name": "cb_stock_return_spread", "parameters": {"window": 20}},
        {"name": "cb_return_20d", "parameters": {}},
        {"name": "cb_abnormal_turnover_20d", "parameters": {}},
    ]


def test_factor_batch_config_rejects_duplicate_factors_and_invalid_parameters() -> None:
    config = {
        "schema_version": "1.0",
        "batch_id": "test_batch",
        "sample_partition": "replication_2018_2023",
        "factors": [
            {"name": "conversion_premium", "parameters": {}},
            {"name": "conversion_premium", "parameters": {}},
            {"name": "cb_stock_return_spread", "parameters": []},
        ],
    }

    errors = validate_factor_batch_config(config)

    assert any("duplicate" in error for error in errors)
    assert any("parameters" in error for error in errors)
