from __future__ import annotations

import pandas as pd
import pytest

import cb_quant.portfolio as portfolio


def test_fixed_horizon_cohort_target_builder_is_public() -> None:
    assert callable(getattr(portfolio, "build_fixed_horizon_cohort_targets", None))


def test_selects_only_pit_eligible_bonds_and_emits_entry_and_exit_targets() -> None:
    scores = pd.DataFrame(
        {
            "signal_date": ["2024-01-02"] * 4,
            "ts_code": ["A", "B", "C", "D"],
            "processed_factor": [4.0, 3.0, 2.0, 1.0],
            "is_eligible": [True, False, True, True],
        }
    )
    schedule = pd.DataFrame(
        {
            "signal_date": ["2024-01-02"],
            "entry_date": ["2024-01-03"],
            "exit_date": ["2024-01-10"],
        }
    )

    plan = portfolio.build_fixed_horizon_cohort_targets(
        scores,
        schedule,
        portfolio_size=2,
        cash_reserve=0.1,
        max_weight=0.6,
    )

    selected = plan.selections.loc[plan.selections["is_selected"]]
    assert selected["ts_code"].tolist() == ["A", "C"]
    assert selected["selection_rank"].tolist() == [1, 2]
    entry = plan.targets.loc[
        plan.targets["execution_date"].eq(pd.Timestamp("2024-01-03"))
    ].set_index("ts_code")
    assert entry["target_weight"].to_dict() == pytest.approx({"A": 0.45, "C": 0.45})
    exit_targets = plan.targets.loc[
        plan.targets["execution_date"].eq(pd.Timestamp("2024-01-10"))
    ]
    assert set(exit_targets["ts_code"]) == {"A", "C"}
    assert exit_targets["target_weight"].eq(0.0).all()


def test_full_active_cohort_skips_overlapping_entry_without_retargeting_holdings() -> None:
    scores = pd.DataFrame(
        {
            "signal_date": ["2024-01-02", "2024-01-05"],
            "ts_code": ["A", "B"],
            "processed_factor": [1.0, 1.0],
            "is_eligible": [True, True],
        }
    )
    schedule = pd.DataFrame(
        {
            "signal_date": ["2024-01-02", "2024-01-05"],
            "entry_date": ["2024-01-03", "2024-01-08"],
            "exit_date": ["2024-01-10", "2024-01-15"],
        }
    )

    plan = portfolio.build_fixed_horizon_cohort_targets(
        scores,
        schedule,
        portfolio_size=1,
        cash_reserve=0.1,
        max_weight=1.0,
    )

    audit = plan.cohort_audit.set_index("signal_date")
    assert audit.loc[pd.Timestamp("2024-01-02"), "status"] == "allocated"
    assert audit.loc[pd.Timestamp("2024-01-05"), "status"] == "skipped_no_capacity"
    assert audit.loc[pd.Timestamp("2024-01-05"), "allocated_weight"] == pytest.approx(0.0)
    assert pd.Timestamp("2024-01-08") not in set(plan.targets["execution_date"])
    assert set(plan.targets["execution_date"]) == {
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-10"),
    }


def test_exit_before_entry_allows_same_day_cohort_replacement() -> None:
    scores = pd.DataFrame(
        {
            "signal_date": ["2024-01-02", "2024-01-09"],
            "ts_code": ["A", "B"],
            "processed_factor": [1.0, 1.0],
            "is_eligible": [True, True],
        }
    )
    schedule = pd.DataFrame(
        {
            "signal_date": ["2024-01-02", "2024-01-09"],
            "entry_date": ["2024-01-03", "2024-01-10"],
            "exit_date": ["2024-01-10", "2024-01-17"],
        }
    )

    plan = portfolio.build_fixed_horizon_cohort_targets(
        scores,
        schedule,
        portfolio_size=1,
        cash_reserve=0.1,
        max_weight=1.0,
    )

    replacement = plan.targets.loc[
        plan.targets["execution_date"].eq(pd.Timestamp("2024-01-10"))
    ]
    assert replacement[["ts_code", "target_weight"]].to_dict("records") == [
        {"ts_code": "B", "target_weight": pytest.approx(0.9)}
    ]
    assert plan.cohort_audit["status"].tolist() == ["allocated", "allocated"]
