from __future__ import annotations

import pandas as pd
import pytest

from cb_quant.portfolio import build_long_only_targets


def make_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": ["2024-01-02"] * 5,
            "ts_code": ["E", "D", "C", "B", "A"],
            "processed_factor": [1.0, 2.0, 3.0, 4.0, 5.0],
            "entry_allowed": [True, False, True, True, True],
            "exit_required": [False, False, False, False, False],
        }
    )


def test_selects_highest_eligible_scores_and_assigns_equal_weights() -> None:
    result = build_long_only_targets(
        make_scores(), portfolio_size=3, cash_reserve=0.1, max_weight=0.5
    )
    selected = result.loc[result["is_selected"]].set_index("ts_code")

    assert selected.index.tolist() == ["A", "B", "C"]
    assert selected["target_weight"].tolist() == pytest.approx([0.3, 0.3, 0.3])
    assert result["target_weight"].sum() == pytest.approx(0.9)


def test_direction_can_reverse_ranking_and_ties_use_bond_code() -> None:
    scores = make_scores()
    scores.loc[scores["ts_code"].isin(["A", "C"]), "processed_factor"] = 3.0

    result = build_long_only_targets(
        scores,
        portfolio_size=2,
        cash_reserve=0.0,
        max_weight=1.0,
        direction=-1,
    )

    assert result.loc[result["is_selected"], "ts_code"].tolist() == ["E", "A"]


def test_entry_and_mandatory_exit_flags_remove_candidates() -> None:
    scores = make_scores()
    scores.loc[scores["ts_code"].eq("A"), "exit_required"] = True

    result = build_long_only_targets(
        scores, portfolio_size=3, cash_reserve=0.1, max_weight=0.5
    )

    assert not bool(result.set_index("ts_code").loc["A", "is_selected"])
    assert not bool(result.set_index("ts_code").loc["D", "is_selected"])
    assert result.loc[result["is_selected"], "ts_code"].tolist() == ["B", "C", "E"]


def test_weight_cap_leaves_unallocatable_capital_in_cash() -> None:
    result = build_long_only_targets(
        make_scores(), portfolio_size=2, cash_reserve=0.1, max_weight=0.2
    )

    assert result["target_weight"].sum() == pytest.approx(0.4)
    assert result["implied_cash_weight"].drop_duplicates().tolist() == pytest.approx([0.6])


def test_empty_eligible_universe_returns_zero_targets() -> None:
    scores = make_scores().assign(entry_allowed=False)

    result = build_long_only_targets(scores, portfolio_size=3)

    assert not result["is_selected"].any()
    assert result["target_weight"].eq(0.0).all()
    assert result["implied_cash_weight"].eq(1.0).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"portfolio_size": 0},
        {"cash_reserve": -0.1},
        {"cash_reserve": 1.0},
        {"max_weight": 0.0},
        {"direction": 0},
    ],
)
def test_rejects_invalid_portfolio_parameters(kwargs) -> None:
    with pytest.raises(ValueError):
        build_long_only_targets(make_scores(), **kwargs)
