"""Fixed-horizon cohort targets for implementable factor portfolios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CohortTargetPlan:
    """Auditable selections, cohort allocations, and full target snapshots."""

    selections: pd.DataFrame
    cohort_audit: pd.DataFrame
    targets: pd.DataFrame


def _normalize_dates(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_datetime(result[column], errors="raise").dt.normalize()
    return result


def build_fixed_horizon_cohort_targets(
    score_panel: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    score_col: str = "processed_factor",
    eligibility_col: str = "is_eligible",
    portfolio_size: int = 20,
    cash_reserve: float = 0.1,
    max_weight: float = 0.1,
) -> CohortTargetPlan:
    """Select Top-N bonds and allocate non-overlapping fixed-horizon cohorts.

    The score is assumed to be direction-aligned, so larger values rank first.
    Cohorts reserve target-weight capacity until their scheduled exit. Exits are
    processed before entries on the same date, matching sell-before-buy execution.
    """
    if not isinstance(portfolio_size, int) or portfolio_size <= 0:
        raise ValueError("portfolio_size must be a positive integer")
    if not 0 <= cash_reserve < 1:
        raise ValueError("cash_reserve must be in [0, 1)")
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in (0, 1]")

    score_required = {"signal_date", "ts_code", score_col, eligibility_col}
    if missing := score_required.difference(score_panel.columns):
        raise KeyError(f"score panel is missing columns: {sorted(missing)}")
    schedule_required = {"signal_date", "entry_date", "exit_date"}
    if missing := schedule_required.difference(schedule.columns):
        raise KeyError(f"schedule is missing columns: {sorted(missing)}")

    scores = _normalize_dates(score_panel, ("signal_date",))
    dates = _normalize_dates(schedule[list(schedule_required)], tuple(schedule_required))
    scores["ts_code"] = scores["ts_code"].astype("string")
    scores[score_col] = pd.to_numeric(scores[score_col], errors="coerce")
    scores[eligibility_col] = scores[eligibility_col].fillna(False).astype(bool)
    if scores.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError("score panel contains duplicate signal-date and bond keys")
    if dates.duplicated("signal_date").any():
        raise ValueError("schedule contains duplicate signal dates")
    if dates["entry_date"].ge(dates["exit_date"]).any():
        raise ValueError("each cohort exit must occur after its entry")

    selections = scores.merge(dates, on="signal_date", how="inner", validate="many_to_one")
    selections["is_selected"] = False
    selections["selection_rank"] = pd.Series(pd.NA, index=selections.index, dtype="Int64")
    selections["desired_weight"] = 0.0
    selections["allocated_weight"] = 0.0

    cohort_specs: list[dict[str, object]] = []
    investable_weight = 1.0 - cash_reserve
    for signal_date, group in selections.groupby("signal_date", sort=True):
        candidate = (
            group[eligibility_col]
            & group[score_col].notna()
            & np.isfinite(group[score_col])
        )
        selected = group.loc[candidate].sort_values(
            [score_col, "ts_code"], ascending=[False, True]
        ).head(portfolio_size)
        per_name_weight = (
            min(investable_weight / len(selected), max_weight)
            if len(selected)
            else 0.0
        )
        if len(selected):
            selections.loc[selected.index, "is_selected"] = True
            selections.loc[selected.index, "selection_rank"] = range(1, len(selected) + 1)
            selections.loc[selected.index, "desired_weight"] = per_name_weight
        schedule_row = group.iloc[0]
        cohort_specs.append(
            {
                "signal_date": signal_date,
                "entry_date": schedule_row["entry_date"],
                "exit_date": schedule_row["exit_date"],
                "selected_indices": selected.index.tolist(),
                "desired_weight": per_name_weight * len(selected),
            }
        )

    active: dict[pd.Timestamp, pd.DataFrame] = {}
    audit_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    event_dates = sorted(
        {spec["entry_date"] for spec in cohort_specs}
        | {spec["exit_date"] for spec in cohort_specs}
    )
    specs_by_entry = {
        date: sorted(
            (spec for spec in cohort_specs if spec["entry_date"] == date),
            key=lambda item: item["signal_date"],
        )
        for date in event_dates
    }

    for event_date in event_dates:
        changed = False
        exited_codes: set[str] = set()
        for cohort_id in list(active):
            spec = next(item for item in cohort_specs if item["signal_date"] == cohort_id)
            if spec["exit_date"] == event_date:
                exited_codes.update(active[cohort_id]["ts_code"].astype(str))
                active.pop(cohort_id)
                changed = True

        for spec in specs_by_entry[event_date]:
            indices = spec["selected_indices"]
            desired_weight = float(spec["desired_weight"])
            active_weight = sum(float(frame["allocated_weight"].sum()) for frame in active.values())
            capacity = max(investable_weight - active_weight, 0.0)
            allocated_weight = min(desired_weight, capacity)
            if not indices:
                status = "skipped_no_candidates"
            elif allocated_weight <= 1e-12:
                status = "skipped_no_capacity"
            elif allocated_weight + 1e-12 < desired_weight:
                status = "partially_allocated"
            else:
                status = "allocated"

            if allocated_weight > 1e-12:
                scale = allocated_weight / desired_weight
                selections.loc[indices, "allocated_weight"] = (
                    selections.loc[indices, "desired_weight"] * scale
                )
                active_frame = selections.loc[
                    indices, ["ts_code", "allocated_weight"]
                ].copy()
                active[spec["signal_date"]] = active_frame
                changed = True
            audit_rows.append(
                {
                    "signal_date": spec["signal_date"],
                    "entry_date": spec["entry_date"],
                    "exit_date": spec["exit_date"],
                    "selected_count": len(indices),
                    "desired_weight": desired_weight,
                    "capacity_before_entry": capacity,
                    "allocated_weight": allocated_weight,
                    "status": status,
                }
            )

        if not changed:
            continue
        if active:
            snapshot = (
                pd.concat(active.values(), ignore_index=True)
                .groupby("ts_code", as_index=False)["allocated_weight"]
                .sum()
            )
            target_rows.extend(
                {
                    "execution_date": event_date,
                    "ts_code": row.ts_code,
                    "target_weight": float(row.allocated_weight),
                }
                for row in snapshot.itertuples(index=False)
            )
        else:
            target_rows.extend(
                {
                    "execution_date": event_date,
                    "ts_code": code,
                    "target_weight": 0.0,
                }
                for code in sorted(exited_codes)
            )

    selection_columns = [
        *scores.columns,
        "entry_date",
        "exit_date",
        "is_selected",
        "selection_rank",
        "desired_weight",
        "allocated_weight",
    ]
    return CohortTargetPlan(
        selections=selections[selection_columns]
        .sort_values(
            ["signal_date", "is_selected", "selection_rank", "ts_code"],
            ascending=[True, False, True, True],
            na_position="last",
        )
        .reset_index(drop=True),
        cohort_audit=pd.DataFrame(audit_rows).sort_values("signal_date").reset_index(drop=True),
        targets=pd.DataFrame(
            target_rows,
            columns=["execution_date", "ts_code", "target_weight"],
        )
        .sort_values(["execution_date", "ts_code"])
        .reset_index(drop=True),
    )
