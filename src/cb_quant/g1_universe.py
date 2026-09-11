"""G1 production helpers for the biweekly point-in-time investable universe."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .event_state import attach_asof_event_state
from .industry import load_ci_industry_membership_panel
from .universe import build_daily_universe_snapshot, universe_funnel


@dataclass(frozen=True)
class UniverseRules:
    """Frozen G1 sample-pool parameters."""

    minimum_age_days: int = 11
    turnover_window: int = 20
    maximum_turnover: float = 1.0
    minimum_remain_size: float = 200_000_000.0
    minimum_rating_score: int = 14


def build_rebalance_calendar(
    trading_dates: pd.DatetimeIndex,
    *,
    anchor_date: str | pd.Timestamp,
    sample_start: str | pd.Timestamp,
    sample_end: str | pd.Timestamp,
    interval_calendar_days: int = 14,
) -> pd.DataFrame:
    """Map fixed calendar anchors to the next open market day.

    Holidays move one observation to the next trading day without changing any
    later anchors. This matches the frozen biweekly sampling convention.
    """
    if interval_calendar_days <= 0:
        raise ValueError("interval_calendar_days must be positive")
    start = pd.Timestamp(sample_start).normalize()
    end = pd.Timestamp(sample_end).normalize()
    anchor = pd.Timestamp(anchor_date).normalize()
    if end < start:
        raise ValueError("sample_end must not be earlier than sample_start")
    if anchor > end:
        raise ValueError("anchor_date must not be later than sample_end")

    anchors = pd.date_range(anchor, end, freq=f"{interval_calendar_days}D")
    anchors = anchors[anchors >= start]
    positions = trading_dates.searchsorted(anchors.to_numpy(), side="left")
    valid = positions < len(trading_dates)
    calendar = pd.DataFrame(
        {
            "anchor_date": anchors[valid],
            "signal_date": trading_dates.take(positions[valid]),
        }
    )
    calendar = calendar.loc[calendar["signal_date"].le(end)].copy()
    calendar["holiday_shift_days"] = (
        calendar["signal_date"] - calendar["anchor_date"]
    ).dt.days
    calendar["entry_date"] = _next_trading_date(
        calendar["signal_date"], trading_dates
    )
    calendar["next_signal_date"] = calendar["signal_date"].shift(-1)
    calendar["next_entry_date"] = calendar["entry_date"].shift(-1)
    if calendar["signal_date"].duplicated().any():
        raise ValueError("Calendar rule generated duplicate signal dates")
    return calendar.reset_index(drop=True)


def build_biweekly_universe_panel(
    signal_dates: pd.Series,
    cb_daily: pd.DataFrame,
    reference_universe: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    rating_events: pd.DataFrame,
    share_events: pd.DataFrame,
    *,
    rules: UniverseRules,
    industry_database_path=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build all candidate snapshots and sequential funnel counts."""
    snapshots = []
    funnels = []
    normalized_signal_dates = pd.DatetimeIndex(
        pd.to_datetime(signal_dates, errors="raise")
    ).normalize()

    for signal_date in normalized_signal_dates:
        snapshot = build_daily_universe_snapshot(
            signal_date,
            cb_daily,
            reference_universe,
            trading_dates,
            rating_events,
            share_events,
            minimum_age_days=rules.minimum_age_days,
            turnover_window=rules.turnover_window,
            maximum_turnover=rules.maximum_turnover,
            minimum_remain_size=rules.minimum_remain_size,
            minimum_rating_score=rules.minimum_rating_score,
        )
        snapshot["signal_date"] = signal_date
        snapshot["turnover_history_start"] = trading_dates[
            max(0, trading_dates.searchsorted(signal_date) - rules.turnover_window + 1)
        ]
        snapshots.append(snapshot)

        funnel = universe_funnel(snapshot)
        funnel["signal_date"] = signal_date
        funnels.append(funnel)

    panel = pd.concat(snapshots, ignore_index=True)
    funnel_panel = pd.concat(funnels, ignore_index=True)
    if industry_database_path is not None:
        membership = load_ci_industry_membership_panel(
            industry_database_path, normalized_signal_dates, level="l1"
        )
        panel = panel.merge(
            membership,
            how="left",
            on=["signal_date", "stk_code"],
            validate="m:1",
        )
    panel["is_industry_available"] = panel.get(
        "ci_industry_name", pd.Series(pd.NA, index=panel.index)
    ).notna()
    return panel.sort_values(["signal_date", "ts_code"], ignore_index=True), funnel_panel


def summarize_universe_by_date(panel: pd.DataFrame) -> pd.DataFrame:
    """Produce one audit row per signal date."""
    required = {
        "signal_date",
        "ts_code",
        "is_eligible",
        "rating_score",
        "is_industry_available",
        "conversion_price_source",
    }
    if missing := required.difference(panel.columns):
        raise KeyError(f"Universe panel is missing columns: {sorted(missing)}")

    rows = []
    for signal_date, frame in panel.groupby("signal_date", sort=True):
        eligible = frame.loc[frame["is_eligible"]]
        rows.append(
            {
                "signal_date": signal_date,
                "mother_universe_count": int(len(frame)),
                "eligible_count": int(len(eligible)),
                "eligible_share": float(frame["is_eligible"].mean()),
                "eligible_rating_coverage": float(eligible["rating_score"].notna().mean())
                if len(eligible)
                else np.nan,
                "eligible_industry_coverage": float(
                    eligible["is_industry_available"].mean()
                )
                if len(eligible)
                else np.nan,
                "eligible_industry_count": int(
                    eligible["ci_industry_name"].nunique(dropna=True)
                )
                if "ci_industry_name" in eligible
                else 0,
                "eligible_industry_ambiguous_count": int(
                    eligible["is_industry_ambiguous"]
                    .astype("boolean")
                    .fillna(False)
                    .sum()
                )
                if "is_industry_ambiguous" in eligible
                else 0,
                "eligible_disclosed_share_event_share": float(
                    eligible["conversion_price_source"].eq("disclosed_event").mean()
                )
                if len(eligible)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_universe_transitions(panel: pd.DataFrame) -> pd.DataFrame:
    """Explain every entry to and exit from the eligible sample."""
    ordered = panel.sort_values(["ts_code", "signal_date"]).copy()
    ordered["previous_signal_date"] = ordered.groupby("ts_code")["signal_date"].shift()
    ordered["previous_is_eligible"] = ordered.groupby("ts_code")["is_eligible"].shift()
    ordered["previous_exclusion_reason"] = ordered.groupby("ts_code")[
        "primary_exclusion_reason"
    ].shift()
    has_previous = ordered["previous_signal_date"].notna()
    previous_eligible = ordered["previous_is_eligible"].astype("boolean").fillna(False)
    entered = (
        has_previous
        & ~previous_eligible
        & ordered["is_eligible"]
    )
    exited = (
        has_previous
        & previous_eligible
        & ~ordered["is_eligible"]
    )
    transitions = ordered.loc[entered | exited].copy()
    transitions["transition"] = np.where(
        transitions["is_eligible"], "entered", "exited"
    )
    transitions["transition_reason"] = np.where(
        transitions["is_eligible"],
        transitions["previous_exclusion_reason"],
        transitions["primary_exclusion_reason"],
    )
    columns = [
        "signal_date",
        "previous_signal_date",
        "ts_code",
        "bond_short_name",
        "stk_code",
        "transition",
        "transition_reason",
        "primary_exclusion_reason",
        "previous_exclusion_reason",
        "turnover_20d",
        "remain_size",
        "rating",
        "vol",
        "amount",
    ]
    return transitions[[column for column in columns if column in transitions]].sort_values(
        ["signal_date", "transition", "ts_code"], ignore_index=True
    )


def audit_universe_panel(
    panel: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    random_seed: int = 20260728,
    random_date_count: int = 12,
) -> dict:
    """Independently verify panel keys, timing and the frozen eligibility formula."""
    filter_columns = [
        "is_age_eligible",
        "is_turnover_eligible",
        "is_size_eligible",
        "is_rating_eligible",
        "is_signal_tradable",
    ]
    recomputed = panel[filter_columns].fillna(False).all(axis=1)
    formula_mismatch = int(recomputed.ne(panel["is_eligible"]).sum())
    duplicate_keys = int(panel.duplicated(["signal_date", "ts_code"]).sum())
    expected_dates = set(pd.to_datetime(calendar["signal_date"]))
    actual_dates = set(pd.to_datetime(panel["signal_date"]))

    timing_violations = {}
    for event_column in ("share_effective_date", "rating_effective_date"):
        if event_column in panel:
            event_dates = pd.to_datetime(panel[event_column], errors="coerce")
            timing_violations[event_column] = int(
                event_dates.gt(pd.to_datetime(panel["signal_date"])).sum()
            )

    eligible = panel.loc[panel["is_eligible"]]
    rng = np.random.default_rng(random_seed)
    unique_dates = np.array(sorted(actual_dates), dtype="datetime64[ns]")
    sample_size = min(random_date_count, len(unique_dates))
    sampled_dates = pd.to_datetime(
        rng.choice(unique_dates, size=sample_size, replace=False)
    ).sort_values()
    random_checks = []
    for signal_date in sampled_dates:
        frame = panel.loc[panel["signal_date"].eq(signal_date)]
        random_checks.append(
            {
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "rows": int(len(frame)),
                "eligible": int(frame["is_eligible"].sum()),
                "formula_mismatches": int(
                    frame["is_eligible"].ne(
                        frame[filter_columns].fillna(False).all(axis=1)
                    ).sum()
                ),
            }
        )

    industry_date_mismatches = 0
    if "membership_date" in panel:
        membership_dates = pd.to_datetime(panel["membership_date"], errors="coerce")
        industry_date_mismatches = int(
            (
                membership_dates.notna()
                & membership_dates.ne(pd.to_datetime(panel["signal_date"]))
            ).sum()
        )

    status = "pass"
    hard_failures = (
        duplicate_keys
        + formula_mismatch
        + sum(timing_violations.values())
        + industry_date_mismatches
        + len(expected_dates.symmetric_difference(actual_dates))
    )
    if hard_failures:
        status = "fail"

    return {
        "status": status,
        "row_count": int(len(panel)),
        "signal_date_count": int(panel["signal_date"].nunique()),
        "bond_count": int(panel["ts_code"].nunique()),
        "eligible_row_count": int(panel["is_eligible"].sum()),
        "eligible_bond_count": int(eligible["ts_code"].nunique()),
        "duplicate_signal_bond_keys": duplicate_keys,
        "eligibility_formula_mismatches": formula_mismatch,
        "missing_signal_dates": sorted(
            date.strftime("%Y-%m-%d") for date in expected_dates - actual_dates
        ),
        "unexpected_signal_dates": sorted(
            date.strftime("%Y-%m-%d") for date in actual_dates - expected_dates
        ),
        "event_timing_violations": timing_violations,
        "industry_membership_date_mismatches": industry_date_mismatches,
        "industry_ambiguous_rows": int(
            panel.get(
                "is_industry_ambiguous", pd.Series(False, index=panel.index)
            )
            .astype("boolean")
            .fillna(False)
            .sum()
        ),
        "eligible_count_summary": {
            key: float(value)
            for key, value in panel.groupby("signal_date")["is_eligible"]
            .sum()
            .describe()[["min", "25%", "50%", "75%", "max"]]
            .items()
        },
        "primary_exclusion_reason_counts": {
            str(key): int(value)
            for key, value in panel["primary_exclusion_reason"]
            .value_counts(dropna=False)
            .items()
        },
        "random_date_checks": random_checks,
    }


def audit_event_boundaries(
    rating_events: pd.DataFrame,
    share_events: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    *,
    sample_start: str | pd.Timestamp,
    sample_end: str | pd.Timestamp,
    samples_per_event_type: int = 12,
    random_seed: int = 20260728,
) -> pd.DataFrame:
    """Check that sampled disclosed states appear on, but not before, effective dates."""
    rng = np.random.default_rng(random_seed)
    checks = []
    specifications = [
        ("rating", rating_events, "rating", "ann_date"),
        ("share", share_events, "remain_size", "publish_date"),
    ]
    start = pd.Timestamp(sample_start).normalize()
    end = pd.Timestamp(sample_end).normalize()

    for event_type, events, value_column, disclosure_column in specifications:
        candidates = events.loc[
            events["effective_date"].between(start, end)
            & events[value_column].notna()
        ].copy()
        if candidates.empty:
            continue
        sample_size = min(samples_per_event_type, len(candidates))
        sampled_indices = rng.choice(candidates.index.to_numpy(), sample_size, replace=False)
        for event_index in sampled_indices:
            event = candidates.loc[event_index]
            effective_date = pd.Timestamp(event["effective_date"]).normalize()
            position = trading_dates.searchsorted(effective_date)
            if position == 0 or position >= len(trading_dates):
                continue
            previous_date = trading_dates[position - 1]
            probe = pd.DataFrame(
                {
                    "ts_code": [event["ts_code"], event["ts_code"]],
                    "trade_date": [previous_date, effective_date],
                }
            )
            attached = attach_asof_event_state(
                probe,
                events,
                [value_column],
                effective_date_output="state_effective_date",
            ).sort_values("trade_date")
            previous_state_date = attached.iloc[0]["state_effective_date"]
            effective_state_date = attached.iloc[1]["state_effective_date"]
            before_pass = pd.isna(previous_state_date) or pd.Timestamp(
                previous_state_date
            ) < effective_date
            effective_pass = (
                pd.notna(effective_state_date)
                and pd.Timestamp(effective_state_date) == effective_date
            )
            checks.append(
                {
                    "event_type": event_type,
                    "ts_code": event["ts_code"],
                    "disclosure_date": event.get(disclosure_column),
                    "effective_date": effective_date,
                    "previous_trade_date": previous_date,
                    "previous_state_effective_date": previous_state_date,
                    "effective_state_effective_date": effective_state_date,
                    "not_visible_early": bool(before_pass),
                    "visible_on_effective_date": bool(effective_pass),
                    "passed": bool(before_pass and effective_pass),
                }
            )
    return pd.DataFrame(checks)


def _next_trading_date(
    dates: pd.Series, trading_dates: pd.DatetimeIndex
) -> pd.Series:
    normalized = pd.to_datetime(dates).dt.normalize()
    positions = trading_dates.searchsorted(normalized.to_numpy(), side="right")
    result = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    valid = positions < len(trading_dates)
    result.loc[valid] = trading_dates.take(positions[valid]).to_numpy()
    return result
