"""Immutable evidence packages for comparable single-factor research runs."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..factors import FactorSpec
from ..pipeline import SingleFactorResearchResult


@dataclass(frozen=True)
class FactorRunIdentity:
    """Deterministic identifiers for one run and its comparison cohort."""

    run_id: str
    comparison_group_id: str
    run_payload: Mapping[str, object]
    comparison_payload: Mapping[str, object]


def _normalize_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_json(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_json(payload: Mapping[str, object]) -> str:
    normalized = _normalize_json(payload)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def build_run_identity(
    *,
    factor_spec: FactorSpec,
    factor_parameters: Mapping[str, object],
    sample_partition: str,
    universe_fingerprint: str,
    preprocessing: Mapping[str, object],
    label: Mapping[str, object],
    rebalance: Mapping[str, object],
    portfolio: Mapping[str, object],
    execution: Mapping[str, object],
    input_fingerprints: Mapping[str, str],
) -> FactorRunIdentity:
    """Build deterministic run and comparison identities from frozen settings."""
    comparison_payload = {
        "sample_partition": sample_partition,
        "universe_fingerprint": universe_fingerprint,
        "preprocessing": dict(preprocessing),
        "label": dict(label),
        "rebalance": dict(rebalance),
        "portfolio": dict(portfolio),
        "execution": dict(execution),
        "input_fingerprints": dict(input_fingerprints),
    }
    run_payload = {
        **comparison_payload,
        "factor": {
            "name": factor_spec.name,
            "version": factor_spec.version,
            "parameters": dict(factor_parameters),
        },
    }
    normalized_run = _normalize_json(run_payload)
    normalized_comparison = _normalize_json(comparison_payload)
    return FactorRunIdentity(
        run_id=_digest(normalized_run),
        comparison_group_id=_digest(normalized_comparison),
        run_payload=normalized_run,
        comparison_payload=normalized_comparison,
    )


def _parameter_signature(parameters: Mapping[str, object]) -> str:
    if not parameters:
        return "default"
    parts = []
    for key, value in sorted(parameters.items()):
        text = re.sub(r"[^A-Za-z0-9_.=-]+", "-", f"{key}={value}")
        parts.append(text.strip("-"))
    return "__".join(parts)


def resolve_factor_run_directory(
    output_root: Path,
    factor_spec: FactorSpec,
    factor_parameters: Mapping[str, object],
    sample_partition: str,
    run_id: str,
) -> Path:
    """Return the canonical directory for one immutable factor run."""
    return (
        Path(output_root)
        / factor_spec.name
        / _parameter_signature(factor_parameters)
        / sample_partition
        / run_id
    )


def _coverage_by_date(result: SingleFactorResearchResult) -> pd.DataFrame:
    raw = result.raw_factor.copy()
    processed = result.processed_factor.copy()
    labels = result.factor_labels.copy()
    raw_counts = raw.groupby("signal_date").agg(
        eligible_count=("ts_code", "size"),
        raw_observed_count=("raw_factor", "count"),
    )
    processed_counts = processed.groupby("signal_date").agg(
        processed_count=("processed_factor", "count")
    )
    label_counts = labels.groupby("signal_date").agg(
        labeled_count=("forward_return", "count")
    )
    return (
        raw_counts.join(processed_counts, how="left")
        .join(label_counts, how="left")
        .fillna(0)
        .reset_index()
    )


def _portfolio_metrics(result: SingleFactorResearchResult) -> dict[str, object]:
    nav = result.execution.nav.copy()
    if nav.empty:
        return {
            "portfolio_total_return": np.nan,
            "portfolio_annualized_return": np.nan,
            "portfolio_annualized_volatility": np.nan,
            "portfolio_sharpe": np.nan,
            "portfolio_max_drawdown": np.nan,
            "portfolio_calmar": np.nan,
            "annualized_turnover": np.nan,
            "annualized_cost": np.nan,
            "average_cash_ratio": np.nan,
            "blocked_buy_count": 0,
            "blocked_sell_count": 0,
        }
    nav = nav.sort_values("trade_date").reset_index(drop=True)
    nav_values = pd.to_numeric(nav["nav"], errors="coerce")
    daily_returns = nav_values.pct_change().dropna()
    periods = max(len(daily_returns), 1)
    total_return = nav_values.iloc[-1] / nav_values.iloc[0] - 1.0
    annualized_return = (1.0 + total_return) ** (252.0 / periods) - 1.0
    annualized_volatility = daily_returns.std(ddof=1) * np.sqrt(252.0)
    sharpe = (
        daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(252.0)
        if len(daily_returns) >= 2 and daily_returns.std(ddof=1) > 0
        else np.nan
    )
    drawdown = nav_values / nav_values.cummax() - 1.0
    maximum_drawdown = float(drawdown.min())
    calmar = (
        annualized_return / abs(maximum_drawdown)
        if maximum_drawdown < 0
        else np.nan
    )
    average_nav = nav_values.mean()
    annualization = 252.0 / max(len(nav), 1)
    return {
        "portfolio_total_return": float(total_return),
        "portfolio_annualized_return": float(annualized_return),
        "portfolio_annualized_volatility": float(annualized_volatility),
        "portfolio_sharpe": float(sharpe),
        "portfolio_max_drawdown": maximum_drawdown,
        "portfolio_calmar": float(calmar),
        "annualized_turnover": float(
            nav["traded_notional"].sum() / average_nav * annualization
        ),
        "annualized_cost": float(nav["daily_cost"].sum() / average_nav * annualization),
        "average_cash_ratio": float((nav["cash"] / nav_values).mean()),
        "blocked_buy_count": int(nav["blocked_buys"].sum()),
        "blocked_sell_count": int(nav["blocked_sells"].sum()),
    }


def _annual_stability(result: SingleFactorResearchResult) -> pd.DataFrame:
    ic = result.ic_series.copy()
    nav = result.execution.nav.copy()
    records: dict[int, dict[str, object]] = {}
    if not ic.empty:
        ic["year"] = pd.to_datetime(ic["signal_date"]).dt.year
        for year, group in ic.groupby("year"):
            records.setdefault(int(year), {})["mean_rank_ic"] = group["rank_ic"].mean()
            records[int(year)]["rank_ic_positive_rate"] = group["rank_ic"].gt(0).mean()
    if not nav.empty:
        nav = nav.sort_values("trade_date").copy()
        nav["year"] = pd.to_datetime(nav["trade_date"]).dt.year
        nav["daily_return"] = nav["nav"].pct_change()
        for year, group in nav.groupby("year"):
            records.setdefault(int(year), {})["portfolio_return"] = (
                1.0 + group["daily_return"].dropna()
            ).prod() - 1.0
    return pd.DataFrame(
        [{"year": year, **values} for year, values in sorted(records.items())]
    )


def _rank_ic_metrics(result: SingleFactorResearchResult) -> dict[str, object]:
    summary = result.ic_summary.set_index("metric") if not result.ic_summary.empty else None
    if summary is None or "rank_ic" not in summary.index:
        return {
            "mean_rank_ic": np.nan,
            "rank_icir_annualized": np.nan,
            "rank_ic_positive_rate": np.nan,
            "rank_ic_nw_t_stat": np.nan,
        }
    row = summary.loc["rank_ic"]
    return {
        "mean_rank_ic": row.get("mean", np.nan),
        "rank_icir_annualized": row.get("icir_annualized", np.nan),
        "rank_ic_positive_rate": row.get("positive_rate", np.nan),
        "rank_ic_nw_t_stat": row.get("nw_t_stat", np.nan),
    }


def _factor_summary(
    identity: FactorRunIdentity,
    factor_spec: FactorSpec,
    factor_parameters: Mapping[str, object],
    sample_partition: str,
    result: SingleFactorResearchResult,
) -> pd.DataFrame:
    raw = result.raw_factor
    processed = result.processed_factor
    labels = result.factor_labels
    denominator = len(labels) + len(result.label_diagnostics)
    annual = _annual_stability(result)
    summary = {
        "run_id": identity.run_id,
        "comparison_group_id": identity.comparison_group_id,
        "factor_name": factor_spec.name,
        "factor_family": factor_spec.family,
        "factor_version": factor_spec.version,
        "factor_parameters": _canonical_json(dict(factor_parameters)),
        "sample_partition": sample_partition,
        "observation_count": len(raw),
        "signal_date_count": raw["signal_date"].nunique(),
        "unique_bond_count": raw["ts_code"].nunique(),
        "raw_coverage": raw["raw_factor"].notna().mean(),
        "processed_coverage": processed["processed_factor"].notna().mean(),
        "label_exclusion_rate": (
            len(result.label_diagnostics) / denominator if denominator else np.nan
        ),
        **_rank_ic_metrics(result),
        "long_short_total_return": (
            result.quantile_nav["long_short"].iloc[-1] - 1.0
            if not result.quantile_nav.empty and "long_short" in result.quantile_nav
            else np.nan
        ),
        **_portfolio_metrics(result),
        "positive_year_rank_ic_share": (
            annual["mean_rank_ic"].gt(0).mean()
            if "mean_rank_ic" in annual and annual["mean_rank_ic"].notna().any()
            else np.nan
        ),
        "worst_year_rank_ic": (
            annual["mean_rank_ic"].min() if "mean_rank_ic" in annual else np.nan
        ),
        "worst_year_portfolio_return": (
            annual["portfolio_return"].min()
            if "portfolio_return" in annual
            else np.nan
        ),
    }
    return pd.DataFrame([summary])


def _save_line_chart(
    frame: pd.DataFrame,
    date_column: str,
    value_columns: list[str],
    path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 4.8))
    dates = pd.to_datetime(frame[date_column])
    for column in value_columns:
        if column in frame:
            axis.plot(dates, pd.to_numeric(frame[column], errors="coerce"), label=column)
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _write_charts(result: SingleFactorResearchResult, output_dir: Path) -> None:
    _save_line_chart(
        result.ic_series,
        "signal_date",
        ["ic", "rank_ic"],
        output_dir / "ic_history.png",
        "IC History",
    )
    quantile_columns = [
        column for column in result.quantile_nav.columns if column != "signal_date"
    ]
    _save_line_chart(
        result.quantile_nav,
        "signal_date",
        quantile_columns,
        output_dir / "quantile_nav.png",
        "Quantile NAV",
    )
    nav = result.execution.nav.copy()
    nav["normalized_nav"] = nav["nav"] / nav["nav"].iloc[0]
    _save_line_chart(
        nav,
        "trade_date",
        ["normalized_nav"],
        output_dir / "portfolio_nav.png",
        "Portfolio NAV",
    )
    nav["drawdown"] = nav["nav"] / nav["nav"].cummax() - 1.0
    _save_line_chart(
        nav,
        "trade_date",
        ["drawdown"],
        output_dir / "drawdown.png",
        "Portfolio Drawdown",
    )


def write_factor_run_record(
    *,
    output_root: Path,
    identity: FactorRunIdentity,
    factor_spec: FactorSpec,
    factor_parameters: Mapping[str, object],
    sample_partition: str,
    result: SingleFactorResearchResult,
) -> Path:
    """Write one complete evidence package without overwriting prior evidence."""
    run_dir = resolve_factor_run_directory(
        output_root,
        factor_spec,
        factor_parameters,
        sample_partition,
        identity.run_id,
    )
    if run_dir.exists():
        raise FileExistsError(f"factor run is immutable and already exists: {run_dir}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".factor_run_", dir=run_dir.parent) as temp:
        destination = Path(temp)
        coverage = _coverage_by_date(result)
        portfolio_metrics = pd.DataFrame([_portfolio_metrics(result)])
        annual_stability = _annual_stability(result)
        factor_summary = _factor_summary(
            identity,
            factor_spec,
            factor_parameters,
            sample_partition,
            result,
        )
        manifest = {
            "schema_version": "1.0",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "run_id": identity.run_id,
            "comparison_group_id": identity.comparison_group_id,
            "factor": {
                "name": factor_spec.name,
                "family": factor_spec.family,
                "version": factor_spec.version,
                "direction": factor_spec.direction,
                "parameters": dict(factor_parameters),
            },
            "sample_partition": sample_partition,
            "identity_payload": identity.run_payload,
        }
        (destination / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        factor_summary.to_csv(destination / "factor_summary.csv", index=False)
        coverage.to_parquet(destination / "coverage_by_date.parquet", index=False)
        result.raw_factor.to_parquet(destination / "raw_factor.parquet", index=False)
        result.processed_factor.to_parquet(
            destination / "processed_factor.parquet", index=False
        )
        result.ic_series.to_parquet(destination / "ic_series.parquet", index=False)
        result.ic_summary.to_csv(destination / "ic_summary.csv", index=False)
        result.quantile_returns.to_parquet(
            destination / "quantile_returns.parquet", index=False
        )
        result.quantile_nav.to_parquet(
            destination / "quantile_nav.parquet", index=False
        )
        result.execution.nav.to_parquet(
            destination / "portfolio_nav.parquet", index=False
        )
        portfolio_metrics.to_csv(destination / "portfolio_metrics.csv", index=False)
        annual_stability.to_csv(destination / "annual_stability.csv", index=False)
        result.execution.orders.to_parquet(
            destination / "execution_diagnostics.parquet", index=False
        )
        _write_charts(result, destination)
        destination.rename(run_dir)
    return run_dir


def build_factor_run_index(
    factor_runs_root: Path,
    *,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Build a validated cross-factor index from immutable run manifests."""
    rows: list[pd.DataFrame] = []
    for manifest_path in sorted(Path(factor_runs_root).rglob("run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary_path = manifest_path.with_name("factor_summary.csv")
        if not summary_path.exists():
            raise FileNotFoundError(f"run summary is missing: {summary_path}")
        summary = pd.read_csv(summary_path, dtype={"run_id": str})
        if len(summary) != 1:
            raise ValueError(f"run summary must contain exactly one row: {summary_path}")
        if str(summary.loc[0, "run_id"]) != str(manifest.get("run_id")):
            raise ValueError(f"manifest and summary run_id disagree: {manifest_path}")
        rows.append(summary)
    index = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not index.empty and index["run_id"].duplicated().any():
        raise ValueError("factor run index contains duplicate run_id values")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        index.to_parquet(output_path, index=False)
    return index
