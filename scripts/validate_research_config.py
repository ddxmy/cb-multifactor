"""Validate the frozen core research definition and emit a G0 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "research_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "G0" / "research_config_validation_v1.json"
EXPECTED_STAGE_IDS = [f"G{number}" for number in range(9)]
ALLOWED_STAGE_STATUS = {"complete", "in_progress", "pending", "blocked"}


def validate_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_top_level = {
        "schema_version",
        "research_id",
        "status",
        "title",
        "research_questions",
        "data",
        "universe",
        "sampling",
        "returns",
        "factor_scope",
        "factor_evaluation",
        "preprocessing",
        "portfolio",
        "stage_protocol",
        "stages",
        "pending_decisions",
    }
    missing = required_top_level.difference(config)
    if missing:
        errors.append(f"missing top-level fields: {sorted(missing)}")
        return errors

    sample = config["data"]["formal_sample"]
    partitions = config["data"]["research_partitions"]
    sample_start = _parse_date(sample["start"], "formal_sample.start", errors)
    sample_end = _parse_date(sample["end"], "formal_sample.end", errors)
    replication_end = _parse_date(
        partitions["report_replication_end"], "report_replication_end", errors
    )
    validation_start = _parse_date(
        partitions["extension_validation_start"], "extension_validation_start", errors
    )
    validation_end = _parse_date(
        partitions["extension_validation_end"], "extension_validation_end", errors
    )
    holdout_start = _parse_date(
        partitions["final_holdout_start"], "final_holdout_start", errors
    )
    if all(
        value is not None
        for value in [
            sample_start,
            sample_end,
            replication_end,
            validation_start,
            validation_end,
            holdout_start,
        ]
    ):
        if not (
            sample_start
            <= replication_end
            < validation_start
            <= validation_end
            < holdout_start
            <= sample_end
        ):
            errors.append("research partitions are not strictly chronological")

    universe = config["universe"]
    if universe["minimum_listing_age_market_days"] < 1:
        errors.append("minimum listing age must be positive")
    if universe["turnover_window_market_days"] < 1:
        errors.append("turnover window must be positive")
    if universe["minimum_remaining_balance_yuan"] <= 0:
        errors.append("minimum remaining balance must be positive")

    sampling = config["sampling"]
    if sampling["signal_time"] != "signal_day_close":
        errors.append("v1 signal_time must remain signal_day_close")
    if sampling["entry_time"] != "next_market_trading_day_open":
        errors.append("v1 entry_time must remain next_market_trading_day_open")

    stage_ids = [stage.get("id") for stage in config["stages"]]
    if stage_ids != EXPECTED_STAGE_IDS:
        errors.append(f"stage ids must be exactly {EXPECTED_STAGE_IDS}")
    invalid_status = [
        stage for stage in config["stages"] if stage.get("status") not in ALLOWED_STAGE_STATUS
    ]
    if invalid_status:
        errors.append(f"invalid stage status: {invalid_status}")

    if config["portfolio"]["one_way_cost_bps"] < 0:
        errors.append("one-way transaction cost cannot be negative")
    winsorization = config["preprocessing"]["winsorization"]
    if winsorization["primary_multiplier"] != 3.0:
        errors.append("v1 primary MAD multiplier must remain 3.0")
    if winsorization["sensitivity_multipliers"] != [2.5, 3.0, 3.5]:
        errors.append("v1 MAD sensitivity multipliers must remain [2.5, 3.0, 3.5]")
    if winsorization["selection_uses_forward_returns"]:
        errors.append("MAD selection cannot use forward returns")
    factor_policies = config["preprocessing"]["neutralization"]["factor_policies"]
    required_factor_policies = {
        "conversion_premium",
        "double_low",
        "relative_premium_to_parity_median",
        "parity_value_score",
        "stock_return_5d",
        "stock_return_10d",
        "stock_return_20d",
        "stock_volatility_20d",
        "stock_rsi_20d",
        "stock_price_to_high_20d",
        "stock_percent_b_20d",
        "stock_amihud_20d",
        "stock_mfi_20d",
    }
    if missing_policies := required_factor_policies.difference(factor_policies):
        errors.append(
            f"v1 factor preprocessing policies are incomplete: {sorted(missing_policies)}"
        )
    if config["factor_scope"]["intraday_factors_in_primary_research"]:
        errors.append("v1 primary research is frozen as daily-frequency only")
    if not config["pending_decisions"]:
        errors.append("pending decisions must remain explicit until resolved")

    if _contains_secret_value(config):
        errors.append("a secret-like key contains a non-placeholder value")
    return errors


def _parse_date(value: str, field: str, errors: list[str]):
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        errors.append(f"{field} is not a valid YYYY-MM-DD date: {value!r}")
        return None


def _contains_secret_value(value: Any) -> bool:
    placeholders = {"", "none", "null", "your_token", "set_in_environment"}
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = str(key).lower()
            if normalized_key in {"token", "tushare_token", "api_key", "secret"}:
                if str(child).strip().lower() not in placeholders:
                    return True
            if _contains_secret_value(child):
                return True
    elif isinstance(value, list):
        return any(_contains_secret_value(child) for child in value)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    raw = args.config.read_bytes()
    config = json.loads(raw)
    errors = validate_config(config)
    audit = {
        "schema_version": "1.0",
        "research_id": config.get("research_id"),
        "config_path": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "stage_status": {
            stage["id"]: stage["status"] for stage in config.get("stages", [])
        },
        "pending_decision_count": len(config.get("pending_decisions", [])),
    }
    if not args.check_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
