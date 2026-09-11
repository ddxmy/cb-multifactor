from __future__ import annotations

import pandas as pd
import pytest

import cb_quant.portfolio as portfolio


def make_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": ["2024-01-02"] * 3 + ["2024-01-09"] * 3,
            "ts_code": ["A", "B", "C", "A", "B", "C"],
            "processed_factor": [3.0, 2.0, 1.0, 3.0, 1.0, 2.0],
            "is_eligible": [True] * 6,
        }
    )


def make_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": ["2024-01-02", "2024-01-09"],
            "entry_date": ["2024-01-03", "2024-01-10"],
        }
    )


def test_exit_buffer_retains_old_name_and_blocks_higher_ranked_new_name() -> None:
    plan = portfolio.build_rank_buffer_targets(
        make_scores(),
        make_schedule(),
        portfolio_size=2,
        exit_rank=3,
        cash_reserve=0.1,
        max_weight=1.0,
        liquidation_date="2024-01-17",
    )

    second = plan.selections.loc[
        plan.selections["signal_date"].eq(pd.Timestamp("2024-01-09"))
        & plan.selections["is_selected"]
    ]
    assert second["ts_code"].tolist() == ["A", "B"]
    assert second.set_index("ts_code")["selection_reason"].to_dict() == {
        "A": "retained_within_buffer",
        "B": "retained_within_buffer",
    }
    assert plan.rebalance_audit.iloc[1]["retained_count"] == 2
    assert plan.rebalance_audit.iloc[1]["entered_count"] == 0
    assert plan.rebalance_audit.iloc[1]["exited_count"] == 0


def test_name_outside_exit_buffer_is_replaced_by_best_available_candidate() -> None:
    plan = portfolio.build_rank_buffer_targets(
        make_scores(),
        make_schedule(),
        portfolio_size=2,
        exit_rank=2,
        cash_reserve=0.1,
        max_weight=1.0,
        liquidation_date="2024-01-17",
    )

    second = plan.selections.loc[
        plan.selections["signal_date"].eq(pd.Timestamp("2024-01-09"))
        & plan.selections["is_selected"]
    ].set_index("ts_code")
    assert set(second.index) == {"A", "C"}
    assert second.loc["A", "selection_reason"] == "retained_within_buffer"
    assert second.loc["C", "selection_reason"] == "entered_to_fill_portfolio"
    assert plan.rebalance_audit.iloc[1]["retained_count"] == 1
    assert plan.rebalance_audit.iloc[1]["entered_count"] == 1
    assert plan.rebalance_audit.iloc[1]["exited_count"] == 1


def test_targets_are_equal_weighted_and_terminal_target_liquidates_last_portfolio() -> None:
    plan = portfolio.build_rank_buffer_targets(
        make_scores(),
        make_schedule(),
        portfolio_size=2,
        exit_rank=3,
        cash_reserve=0.1,
        max_weight=1.0,
        liquidation_date="2024-01-17",
    )

    entry_targets = plan.targets.loc[
        plan.targets["execution_date"].eq(pd.Timestamp("2024-01-03"))
    ].set_index("ts_code")
    assert entry_targets["target_weight"].to_dict() == pytest.approx(
        {"A": 0.45, "B": 0.45}
    )
    terminal = plan.targets.loc[
        plan.targets["execution_date"].eq(pd.Timestamp("2024-01-17"))
    ]
    assert set(terminal["ts_code"]) == {"A", "B"}
    assert terminal["target_weight"].eq(0.0).all()
