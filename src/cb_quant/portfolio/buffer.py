"""Rank-buffer target construction for continuous long-only portfolios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RankBufferTargetPlan:
    """Auditable selections, rebalance changes, and execution targets."""

    selections: pd.DataFrame
    rebalance_audit: pd.DataFrame
    targets: pd.DataFrame


def build_rank_buffer_targets(
    score_panel: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    score_col: str = "processed_factor",
    eligibility_col: str = "is_eligible",
    portfolio_size: int = 20,
    exit_rank: int = 30,
    cash_reserve: float = 0.1,
    max_weight: float = 0.1,
    liquidation_date: str | pd.Timestamp | None = None,
) -> RankBufferTargetPlan:
    """Enter from Top-N and retain existing names until they rank below K."""
    if not isinstance(portfolio_size, int) or portfolio_size <= 0:
        raise ValueError("portfolio_size must be a positive integer")
    if not isinstance(exit_rank, int) or exit_rank < portfolio_size:
        raise ValueError("exit_rank must be an integer at least as large as portfolio_size")
    if not 0 <= cash_reserve < 1:
        raise ValueError("cash_reserve must be in [0, 1)")
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in (0, 1]")

    score_required = {"signal_date", "ts_code", score_col, eligibility_col}
    if missing := score_required.difference(score_panel.columns):
        raise KeyError(f"score panel is missing columns: {sorted(missing)}")
    schedule_required = {"signal_date", "entry_date"}
    if missing := schedule_required.difference(schedule.columns):
        raise KeyError(f"schedule is missing columns: {sorted(missing)}")

    scores = score_panel.copy()
    scores["signal_date"] = pd.to_datetime(
        scores["signal_date"], errors="raise"
    ).dt.normalize()
    scores["ts_code"] = scores["ts_code"].astype("string")
    scores[score_col] = pd.to_numeric(scores[score_col], errors="coerce")
    scores[eligibility_col] = scores[eligibility_col].fillna(False).astype(bool)
    if scores.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError("score panel contains duplicate signal-date and bond keys")

    dates = schedule[["signal_date", "entry_date"]].copy()
    for column in ("signal_date", "entry_date"):
        dates[column] = pd.to_datetime(dates[column], errors="raise").dt.normalize()
    if dates.duplicated("signal_date").any():
        raise ValueError("schedule contains duplicate signal dates")
    if dates["entry_date"].le(dates["signal_date"]).any():
        raise ValueError("entry dates must occur after signal dates")

    selections = scores.merge(dates, on="signal_date", how="inner", validate="many_to_one")
    selections["factor_rank"] = pd.Series(pd.NA, index=selections.index, dtype="Int64")
    selections["is_selected"] = False
    selections["portfolio_slot"] = pd.Series(pd.NA, index=selections.index, dtype="Int64")
    selections["selection_reason"] = "not_selected"
    selections["target_weight"] = 0.0

    current_slots: dict[int, str] = {}
    target_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    investable_weight = 1.0 - cash_reserve

    for signal_date, group in selections.groupby("signal_date", sort=True):
        candidate = (
            group[eligibility_col]
            & group[score_col].notna()
            & np.isfinite(group[score_col])
        )
        ranked = group.loc[candidate].sort_values(
            [score_col, "ts_code"], ascending=[False, True]
        )
        selections.loc[ranked.index, "factor_rank"] = range(1, len(ranked) + 1)
        rank_by_code = pd.Series(
            range(1, len(ranked) + 1), index=ranked["ts_code"].astype(str)
        )

        previous_codes = set(current_slots.values())
        retained_slots = {
            slot: code
            for slot, code in current_slots.items()
            if code in rank_by_code.index and int(rank_by_code.loc[code]) <= exit_rank
        }
        available = [
            str(code)
            for code in ranked["ts_code"]
            if str(code) not in retained_slots.values()
        ]
        next_slots = retained_slots.copy()
        vacant_slots = [slot for slot in range(1, portfolio_size + 1) if slot not in next_slots]
        for slot, code in zip(vacant_slots, available, strict=False):
            next_slots[slot] = code

        target_per_name = (
            min(investable_weight / len(next_slots), max_weight)
            if next_slots
            else 0.0
        )
        code_to_index = {
            str(code): index for code, index in zip(group["ts_code"], group.index, strict=True)
        }
        retained_codes = set(retained_slots.values())
        for slot, code in next_slots.items():
            index = code_to_index[code]
            selections.loc[index, "is_selected"] = True
            selections.loc[index, "portfolio_slot"] = slot
            selections.loc[index, "selection_reason"] = (
                "retained_within_buffer"
                if code in retained_codes
                else "entered_to_fill_portfolio"
            )
            selections.loc[index, "target_weight"] = target_per_name

        execution_date = group["entry_date"].iloc[0]
        target_rows.extend(
            {
                "execution_date": execution_date,
                "ts_code": code,
                "target_weight": target_per_name,
            }
            for _, code in sorted(next_slots.items())
        )
        selected_codes = set(next_slots.values())
        audit_rows.append(
            {
                "signal_date": signal_date,
                "execution_date": execution_date,
                "eligible_count": len(ranked),
                "selected_count": len(next_slots),
                "retained_count": len(previous_codes & selected_codes),
                "entered_count": len(selected_codes - previous_codes),
                "exited_count": len(previous_codes - selected_codes),
                "exit_rank": exit_rank,
                "target_gross_weight": target_per_name * len(next_slots),
            }
        )
        current_slots = next_slots

    if liquidation_date is not None and current_slots:
        terminal_date = pd.Timestamp(liquidation_date).normalize()
        last_execution = pd.Timestamp(selections["entry_date"].max()).normalize()
        if terminal_date <= last_execution:
            raise ValueError("liquidation_date must be after the last entry date")
        target_rows.extend(
            {
                "execution_date": terminal_date,
                "ts_code": code,
                "target_weight": 0.0,
            }
            for _, code in sorted(current_slots.items())
        )

    return RankBufferTargetPlan(
        selections=selections.sort_values(
            ["signal_date", "is_selected", "portfolio_slot", "factor_rank", "ts_code"],
            ascending=[True, False, True, True, True],
            na_position="last",
        ).reset_index(drop=True),
        rebalance_audit=pd.DataFrame(audit_rows).sort_values("signal_date").reset_index(drop=True),
        targets=pd.DataFrame(
            target_rows,
            columns=["execution_date", "ts_code", "target_weight"],
        )
        .sort_values(["execution_date", "ts_code"])
        .reset_index(drop=True),
    )
