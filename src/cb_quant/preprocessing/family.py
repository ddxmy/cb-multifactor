"""Factor-family preprocessing policies and exposure diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cross_section import neutralize_cross_section, standardize_zscore


FORWARD_LABEL_COLUMNS = {"forward_return", "label_end_date", "entry_open", "exit_open"}
DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class FactorPreprocessingPolicy:
    """Explicit controls applied to one already-winsorized factor."""

    name: str
    numeric_controls: tuple[str, ...] = ()
    categorical_controls: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("factor policy name cannot be empty")
        controls = self.numeric_controls + self.categorical_controls
        if len(set(controls)) != len(controls):
            raise ValueError("factor policy controls cannot contain duplicates")

    @property
    def is_neutralized(self) -> bool:
        return bool(self.numeric_controls or self.categorical_controls)


def add_cb_neutralization_controls(panel: pd.DataFrame) -> pd.DataFrame:
    """Add point-in-time balance and maturity controls to a merged CB panel."""
    required = {"signal_date", "remain_size", "maturity_date", "rating"}
    if missing := required.difference(panel.columns):
        raise KeyError(f"factor panel is missing control sources: {sorted(missing)}")
    result = panel.copy()
    result["signal_date"] = pd.to_datetime(
        result["signal_date"], errors="raise"
    ).dt.normalize()
    result["maturity_date"] = pd.to_datetime(
        result["maturity_date"], errors="coerce"
    ).dt.normalize()
    balance = pd.to_numeric(result["remain_size"], errors="coerce")
    if balance.notna().any() and balance.dropna().le(0).any():
        raise ValueError("remaining balance must be positive when observed")
    result["log_remaining_balance"] = np.log(balance.where(balance.gt(0)))
    result["remaining_maturity_years"] = (
        result["maturity_date"] - result["signal_date"]
    ).dt.days / DAYS_PER_YEAR
    if result["remaining_maturity_years"].dropna().lt(0).any():
        raise ValueError("remaining maturity cannot be negative")
    result["rating"] = result["rating"].astype("string")
    return result


def attach_stock_market_cap_control(
    panel: pd.DataFrame,
    market_cap_history: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the latest non-future stock market cap and industry control."""
    required_panel = {"signal_date", "ts_code", "stk_code", "ci_industry_code"}
    if missing := required_panel.difference(panel.columns):
        raise KeyError(f"factor panel is missing stock control keys: {sorted(missing)}")
    required_history = {"stk_code", "trade_date", "total_market_cap"}
    if missing := required_history.difference(market_cap_history.columns):
        raise KeyError(f"market-cap history is missing columns: {sorted(missing)}")

    result = panel.copy()
    result["signal_date"] = pd.to_datetime(
        result["signal_date"], errors="raise"
    ).dt.normalize()
    result["stk_code"] = result["stk_code"].astype("string")
    result["_original_order"] = np.arange(len(result))
    history = market_cap_history[
        ["stk_code", "trade_date", "total_market_cap"]
    ].copy()
    history["stk_code"] = history["stk_code"].astype("string")
    history["trade_date"] = pd.to_datetime(
        history["trade_date"], errors="raise"
    ).dt.normalize()
    history["total_market_cap"] = pd.to_numeric(
        history["total_market_cap"], errors="coerce"
    )
    if history.duplicated(["stk_code", "trade_date"]).any():
        raise ValueError("market-cap history contains duplicate stock-date keys")
    if history["total_market_cap"].dropna().le(0).any():
        raise ValueError("stock total market cap must be positive when observed")

    history = history.rename(
        columns={"trade_date": "stock_market_cap_observation_date"}
    )

    left = result.sort_values(["signal_date", "stk_code"])
    right = history.sort_values(
        ["stock_market_cap_observation_date", "stk_code"]
    )
    attached = pd.merge_asof(
        left,
        right,
        left_on="signal_date",
        right_on="stock_market_cap_observation_date",
        by="stk_code",
        direction="backward",
        allow_exact_matches=True,
    )
    future = attached["stock_market_cap_observation_date"].gt(
        attached["signal_date"]
    )
    if future.any():
        raise ValueError("stock market-cap control contains future observations")
    attached["stock_market_cap_staleness_days"] = (
        attached["signal_date"] - attached["stock_market_cap_observation_date"]
    ).dt.days.astype("Int64")
    attached["log_stock_total_market_cap"] = np.log(
        attached["total_market_cap"].where(attached["total_market_cap"].gt(0))
    )
    attached["ci_industry_control"] = attached["ci_industry_code"].astype(
        "string"
    )
    return attached.sort_values("_original_order").drop(
        columns="_original_order"
    ).reset_index(drop=True)


def _control_design(
    group: pd.DataFrame,
    *,
    numeric_controls: Sequence[str],
    categorical_controls: Sequence[str],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    if numeric_controls:
        parts.append(
            group[list(numeric_controls)].apply(pd.to_numeric, errors="coerce")
        )
    if categorical_controls:
        parts.append(
            pd.get_dummies(
                group[list(categorical_controls)].astype("string"),
                prefix=list(categorical_controls),
                drop_first=True,
                dtype=float,
            )
        )
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=group.index)


def _mean_absolute_control_correlation(
    frame: pd.DataFrame,
    *,
    value_col: str,
    numeric_controls: Sequence[str],
    categorical_controls: Sequence[str],
) -> float:
    correlations: list[float] = []
    for _, group in frame.groupby("signal_date", sort=False):
        design = _control_design(
            group,
            numeric_controls=numeric_controls,
            categorical_controls=categorical_controls,
        )
        for control in design.columns:
            valid = pd.DataFrame(
                {
                    "value": pd.to_numeric(group[value_col], errors="coerce"),
                    "control": pd.to_numeric(design[control], errors="coerce"),
                }
            ).dropna()
            if valid["value"].nunique() > 1 and valid["control"].nunique() > 1:
                correlations.append(abs(valid["value"].corr(valid["control"])))
    return float(pd.Series(correlations, dtype=float).mean())


def _mean_cross_sectional_rank_correlation(
    frame: pd.DataFrame,
    left: str,
    right: str,
) -> float:
    correlations = []
    for _, group in frame.groupby("signal_date", sort=False):
        valid = group[[left, right]].dropna()
        if valid[left].nunique() > 1 and valid[right].nunique() > 1:
            correlations.append(valid[left].corr(valid[right], method="spearman"))
    return float(pd.Series(correlations, dtype=float).mean())


def preprocess_factor_family(
    panel: pd.DataFrame,
    policies: Sequence[FactorPreprocessingPolicy],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply explicit neutralization policies and date-local z-scoring."""
    if FORWARD_LABEL_COLUMNS.intersection(panel.columns):
        raise ValueError("factor preprocessing cannot contain forward-return labels")
    if not policies:
        raise ValueError("factor preprocessing policies cannot be empty")
    names = [policy.name for policy in policies]
    if len(set(names)) != len(names):
        raise ValueError("factor preprocessing policies contain duplicate factors")

    result = panel.copy()
    diagnostics: list[dict[str, object]] = []
    for policy in policies:
        source = f"winsorized__{policy.name}"
        if source not in result:
            raise KeyError(f"factor panel is missing {source}")
        controls = policy.numeric_controls + policy.categorical_controls
        if missing := set(controls).difference(result.columns):
            raise KeyError(f"factor panel is missing controls: {sorted(missing)}")

        pre_exposure = (
            _mean_absolute_control_correlation(
                result,
                value_col=source,
                numeric_controls=policy.numeric_controls,
                categorical_controls=policy.categorical_controls,
            )
            if policy.is_neutralized
            else np.nan
        )
        standardized_input = source
        neutralized_column = f"neutralized__{policy.name}"
        if policy.is_neutralized:
            result = neutralize_cross_section(
                result,
                value_col=source,
                numeric_controls=policy.numeric_controls,
                categorical_controls=policy.categorical_controls,
                output_col=neutralized_column,
            )
            standardized_input = neutralized_column
        standardized_column = f"standardized__{policy.name}"
        result = standardize_zscore(
            result,
            value_col=standardized_input,
            output_col=standardized_column,
        )
        post_exposure = (
            _mean_absolute_control_correlation(
                result,
                value_col=neutralized_column,
                numeric_controls=policy.numeric_controls,
                categorical_controls=policy.categorical_controls,
            )
            if policy.is_neutralized
            else np.nan
        )
        diagnostics.append(
            {
                "factor_name": policy.name,
                "neutralized": policy.is_neutralized,
                "numeric_controls": ",".join(policy.numeric_controls),
                "categorical_controls": ",".join(policy.categorical_controls),
                "observations": int(result[source].notna().sum()),
                "processed_observations": int(result[standardized_column].notna().sum()),
                "processed_coverage": float(result[standardized_column].notna().mean()),
                "pre_mean_absolute_control_correlation": pre_exposure,
                "post_mean_absolute_control_correlation": post_exposure,
                "winsorized_to_processed_rank_correlation": (
                    _mean_cross_sectional_rank_correlation(
                        result, source, standardized_column
                    )
                ),
            }
        )
    return result, pd.DataFrame(diagnostics)
