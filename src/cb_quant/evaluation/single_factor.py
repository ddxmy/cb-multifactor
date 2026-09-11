"""Cross-sectional evidence for one convertible-bond factor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _prepare_panel(
    panel: pd.DataFrame,
    factor_col: str,
    return_col: str,
) -> pd.DataFrame:
    required = {"signal_date", "ts_code", factor_col, return_col}
    if missing := required.difference(panel.columns):
        raise KeyError(f"factor-label panel is missing columns: {sorted(missing)}")
    result = panel.copy()
    result["signal_date"] = pd.to_datetime(
        result["signal_date"], errors="raise"
    ).dt.normalize()
    result["ts_code"] = result["ts_code"].astype("string")
    if result.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError("factor-label panel contains duplicate keys")
    return result


def calculate_ic_series(
    panel: pd.DataFrame,
    *,
    factor_col: str = "processed_factor",
    return_col: str = "forward_return",
    min_assets: int = 5,
) -> pd.DataFrame:
    """Calculate Pearson IC and Spearman Rank IC for each signal date."""
    if min_assets < 2:
        raise ValueError("min_assets must be at least two")
    data = _prepare_panel(panel, factor_col, return_col)
    records: list[dict[str, object]] = []
    for signal_date, group in data.groupby("signal_date", sort=True):
        valid = group[[factor_col, return_col]].dropna()
        enough = (
            len(valid) >= min_assets
            and valid[factor_col].nunique() > 1
            and valid[return_col].nunique() > 1
        )
        records.append(
            {
                "signal_date": signal_date,
                "asset_count": int(len(valid)),
                "ic": valid[factor_col].corr(valid[return_col]) if enough else np.nan,
                "rank_ic": (
                    valid[factor_col].corr(valid[return_col], method="spearman")
                    if enough
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(records)


def summarize_ic(
    ic_series: pd.DataFrame,
    *,
    periods_per_year: int = 26,
    nw_lags: int = 5,
) -> pd.DataFrame:
    """Summarize IC distributions with heteroskedasticity/autocorrelation robust inference."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    if nw_lags < 0:
        raise ValueError("nw_lags cannot be negative")
    required = {"ic", "rank_ic"}
    if missing := required.difference(ic_series.columns):
        raise KeyError(f"IC series is missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for metric in ("ic", "rank_ic"):
        values = pd.to_numeric(ic_series[metric], errors="coerce").dropna()
        mean = values.mean()
        standard_deviation = values.std(ddof=1)
        effective_lags = min(nw_lags, max(len(values) - 1, 0))
        annualized_icir = (
            mean / standard_deviation * np.sqrt(periods_per_year)
            if len(values) >= 2 and standard_deviation > 0
            else np.nan
        )
        if len(values) >= 2 and standard_deviation > 0:
            model = sm.OLS(values.to_numpy(dtype=float), np.ones((len(values), 1))).fit(
                cov_type="HAC", cov_kwds={"maxlags": effective_lags}
            )
            nw_t_stat = float(model.tvalues[0])
            nw_p_value = float(model.pvalues[0])
        else:
            nw_t_stat = np.nan
            nw_p_value = np.nan
        rows.append(
            {
                "metric": metric,
                "observations": int(len(values)),
                "mean": mean,
                "std": standard_deviation,
                "positive_rate": (values > 0).mean() if len(values) else np.nan,
                "icir_annualized": annualized_icir,
                "nw_t_stat": nw_t_stat,
                "nw_p_value": nw_p_value,
                "nw_lags": effective_lags,
            }
        )
    return pd.DataFrame(rows)


def calculate_multi_horizon_ic(
    panel: pd.DataFrame,
    *,
    horizon_col: str = "horizon_days",
    factor_col: str = "processed_factor",
    return_col: str = "forward_return",
    min_assets: int = 5,
) -> pd.DataFrame:
    """Calculate one IC and RankIC time series per prediction horizon."""
    if horizon_col not in panel:
        raise KeyError(f"factor-label panel is missing {horizon_col}")
    data = panel.copy()
    data["signal_date"] = pd.to_datetime(
        data["signal_date"], errors="raise"
    ).dt.normalize()
    data["ts_code"] = data["ts_code"].astype("string")
    data[horizon_col] = pd.to_numeric(data[horizon_col], errors="raise").astype(int)
    if data.duplicated(["signal_date", "ts_code", horizon_col]).any():
        raise ValueError("multi-horizon panel contains duplicate keys")
    frames: list[pd.DataFrame] = []
    for horizon, group in data.groupby(horizon_col, sort=True):
        series = calculate_ic_series(
            group.drop(columns=horizon_col),
            factor_col=factor_col,
            return_col=return_col,
            min_assets=min_assets,
        )
        series[horizon_col] = int(horizon)
        frames.append(series)
    if not frames:
        return pd.DataFrame(
            columns=["signal_date", "asset_count", "ic", "rank_ic", horizon_col]
        )
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values([horizon_col, "signal_date"])
        .reset_index(drop=True)
    )


def summarize_horizon_ic(
    ic_series: pd.DataFrame,
    *,
    horizon_col: str = "horizon_days",
    periods_per_year: int = 26,
    nw_lags: int = 5,
) -> pd.DataFrame:
    """Summarize IC and RankIC separately for every prediction horizon."""
    if horizon_col not in ic_series:
        raise KeyError(f"IC series is missing {horizon_col}")
    frames: list[pd.DataFrame] = []
    for horizon, group in ic_series.groupby(horizon_col, sort=True):
        summary = summarize_ic(
            group,
            periods_per_year=periods_per_year,
            nw_lags=nw_lags,
        )
        summary[horizon_col] = int(horizon)
        frames.append(summary)
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values([horizon_col, "metric"])
        .reset_index(drop=True)
    )


def assign_quantile_groups(
    panel: pd.DataFrame,
    *,
    factor_col: str = "processed_factor",
    return_col: str = "forward_return",
    group_count: int = 5,
) -> pd.DataFrame:
    """Assign deterministic equal-count groups with the highest score in group one."""
    if group_count < 2:
        raise ValueError("group_count must be at least two")
    data = _prepare_panel(panel, factor_col, return_col)
    valid = data.loc[data[[factor_col, return_col]].notna().all(axis=1)].copy()
    counts = valid.groupby("signal_date")["ts_code"].transform("count")
    if counts.lt(group_count).any():
        raise ValueError("some signal dates have fewer assets than quantile groups")
    valid = valid.sort_values(
        ["signal_date", factor_col, "ts_code"],
        ascending=[True, False, True],
    )
    position = valid.groupby("signal_date").cumcount()
    counts = valid.groupby("signal_date")["ts_code"].transform("count")
    valid["group"] = (position * group_count // counts + 1).astype(int)
    return valid.reset_index(drop=True)


def calculate_quantile_returns(
    grouped_panel: pd.DataFrame,
    *,
    return_col: str = "forward_return",
    group_count: int = 5,
) -> pd.DataFrame:
    """Calculate equal-weight quantile returns and two explicit spread conventions."""
    required = {"signal_date", "group", return_col}
    if missing := required.difference(grouped_panel.columns):
        raise KeyError(f"grouped panel is missing columns: {sorted(missing)}")
    data = grouped_panel.copy()
    data["signal_date"] = pd.to_datetime(data["signal_date"], errors="raise").dt.normalize()
    returns = data.groupby(["signal_date", "group"])[return_col].mean().unstack("group")
    returns = returns.reindex(columns=range(1, group_count + 1))
    returns.columns = [f"G{group}" for group in range(1, group_count + 1)]
    returns = returns.reset_index()
    returns["spread_raw"] = returns["G1"] - returns[f"G{group_count}"]
    returns["long_short"] = 0.5 * returns["spread_raw"]
    return returns.sort_values("signal_date").reset_index(drop=True)


def calculate_quantile_nav(quantile_returns: pd.DataFrame) -> pd.DataFrame:
    """Compound each quantile and spread return series independently."""
    if "signal_date" not in quantile_returns:
        raise KeyError("quantile returns are missing signal_date")
    ordered = quantile_returns.copy()
    ordered["signal_date"] = pd.to_datetime(
        ordered["signal_date"], errors="raise"
    ).dt.normalize()
    ordered = ordered.sort_values("signal_date").reset_index(drop=True)
    return_columns = [column for column in ordered.columns if column != "signal_date"]
    nav = ordered[["signal_date"]].copy()
    nav[return_columns] = (1.0 + ordered[return_columns]).cumprod()
    return nav


def calculate_horizon_ir_series(
    panel: pd.DataFrame,
    *,
    horizon_col: str = "horizon_days",
    factor_col: str = "processed_factor",
    return_col: str = "forward_return",
    group_count: int = 5,
) -> pd.DataFrame:
    """Build top-group active and G1-minus-bottom-group return series."""
    if horizon_col not in panel:
        raise KeyError(f"factor-label panel is missing {horizon_col}")
    data = panel.copy()
    data["signal_date"] = pd.to_datetime(
        data["signal_date"], errors="raise"
    ).dt.normalize()
    data["ts_code"] = data["ts_code"].astype("string")
    data[horizon_col] = pd.to_numeric(data[horizon_col], errors="raise").astype(int)
    if data.duplicated(["signal_date", "ts_code", horizon_col]).any():
        raise ValueError("multi-horizon panel contains duplicate keys")

    frames: list[pd.DataFrame] = []
    for horizon, group in data.groupby(horizon_col, sort=True):
        horizon_panel = group.drop(columns=horizon_col)
        grouped = assign_quantile_groups(
            horizon_panel,
            factor_col=factor_col,
            return_col=return_col,
            group_count=group_count,
        )
        returns = calculate_quantile_returns(
            grouped,
            return_col=return_col,
            group_count=group_count,
        )
        valid = horizon_panel.loc[
            horizon_panel[[factor_col, return_col]].notna().all(axis=1)
        ]
        universe = (
            valid.groupby("signal_date", as_index=False)[return_col]
            .mean()
            .rename(columns={return_col: "universe_return"})
        )
        returns = returns.merge(
            universe,
            on="signal_date",
            how="left",
            validate="one_to_one",
        )
        returns["active_return"] = returns["G1"] - returns["universe_return"]
        returns[horizon_col] = int(horizon)
        frames.append(returns)
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values([horizon_col, "signal_date"])
        .reset_index(drop=True)
    )


def _annualized_information_ratio(
    values: pd.Series,
    periods_per_year: int,
) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    standard_deviation = numeric.std(ddof=1)
    if len(numeric) < 2 or not np.isfinite(standard_deviation) or standard_deviation <= 0:
        return np.nan
    return float(numeric.mean() / standard_deviation * np.sqrt(periods_per_year))


def summarize_horizon_ir(
    ir_series: pd.DataFrame,
    *,
    horizon_col: str = "horizon_days",
    periods_per_year: int = 26,
) -> pd.DataFrame:
    """Summarize active and analytical spread IR by prediction horizon."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    required = {horizon_col, "active_return", "spread_raw"}
    if missing := required.difference(ir_series.columns):
        raise KeyError(f"IR series is missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for horizon, group in ir_series.groupby(horizon_col, sort=True):
        active = pd.to_numeric(group["active_return"], errors="coerce").dropna()
        spread = pd.to_numeric(group["spread_raw"], errors="coerce").dropna()
        rows.append(
            {
                horizon_col: int(horizon),
                "observations": int(len(active)),
                "active_return_mean": active.mean(),
                "active_return_std": active.std(ddof=1),
                "active_positive_rate": (active > 0).mean() if len(active) else np.nan,
                "active_ir_annualized": _annualized_information_ratio(
                    active, periods_per_year
                ),
                "spread_return_mean": spread.mean(),
                "spread_return_std": spread.std(ddof=1),
                "spread_positive_rate": (spread > 0).mean() if len(spread) else np.nan,
                "spread_ir_annualized": _annualized_information_ratio(
                    spread, periods_per_year
                ),
            }
        )
    return pd.DataFrame(rows)
