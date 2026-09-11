from __future__ import annotations

import pandas as pd
import pytest

from cb_quant.execution import run_execution_backtest


def market_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["trade_date", "ts_code", "open", "close", "buy_allowed", "sell_allowed"],
    )


def permissive_backtest(*args, **kwargs):
    """Use only in compact tests that intentionally omit full lifecycle coverage."""
    return run_execution_backtest(*args, allow_missing_lifecycle=True, **kwargs)


def test_sells_before_buys_and_reuses_actual_proceeds() -> None:
    targets = pd.DataFrame(
        {
            "execution_date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "ts_code": ["A", "B", "A", "B"],
            "target_weight": [0.5, 0.0, 0.0, 1.0],
        }
    )
    market = market_frame(
        [
            ("2024-01-02", "A", 10.0, 10.0, True, True),
            ("2024-01-02", "B", 10.0, 10.0, True, True),
            ("2024-01-03", "A", 10.0, 10.0, True, True),
            ("2024-01-03", "B", 10.0, 10.0, True, True),
        ]
    )

    result = permissive_backtest(
        targets, market, initial_cash=1_000.0, cost_rate=0.0, board_lot=10
    )
    day_two_fills = result.fills.loc[result.fills["trade_date"].eq(pd.Timestamp("2024-01-03"))]

    assert day_two_fills["side"].tolist() == ["sell", "buy"]
    assert day_two_fills["ts_code"].tolist() == ["A", "B"]
    assert day_two_fills.loc[day_two_fills["ts_code"].eq("B"), "quantity"].item() == 100
    assert result.nav.iloc[-1]["cash"] == pytest.approx(0.0)


def test_board_lot_rounding_and_cost_apply_only_to_filled_notional() -> None:
    targets = pd.DataFrame(
        {
            "execution_date": ["2024-01-02"],
            "ts_code": ["A"],
            "target_weight": [0.9],
        }
    )
    market = market_frame(
        [("2024-01-02", "A", 11.0, 11.0, True, True)]
    )

    result = permissive_backtest(
        targets, market, initial_cash=1_000.0, cost_rate=0.001, board_lot=10
    )

    fill = result.fills.iloc[0]
    assert fill["quantity"] == 80
    assert fill["gross_notional"] == pytest.approx(880.0)
    assert fill["cost"] == pytest.approx(0.88)
    assert result.nav.iloc[0]["cash"] == pytest.approx(119.12)
    assert result.nav.iloc[0]["nav"] == pytest.approx(999.12)


def test_blocked_sell_carries_position_and_limits_new_buy_to_remaining_cash() -> None:
    targets = pd.DataFrame(
        {
            "execution_date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "ts_code": ["A", "B", "A", "B"],
            "target_weight": [0.8, 0.0, 0.0, 1.0],
        }
    )
    market = market_frame(
        [
            ("2024-01-02", "A", 10.0, 10.0, True, True),
            ("2024-01-02", "B", 10.0, 10.0, True, True),
            ("2024-01-03", "A", 10.0, 10.0, True, False),
            ("2024-01-03", "B", 10.0, 10.0, True, True),
            ("2024-01-04", "A", 10.0, 10.0, True, True),
            ("2024-01-04", "B", 10.0, 10.0, True, True),
        ]
    )

    result = permissive_backtest(
        targets, market, initial_cash=1_000.0, cost_rate=0.0, board_lot=10
    )
    day_two = result.nav.set_index("trade_date").loc[pd.Timestamp("2024-01-03")]
    day_two_positions = result.positions.loc[
        result.positions["trade_date"].eq(pd.Timestamp("2024-01-03"))
    ].set_index("ts_code")

    assert day_two["blocked_sells"] == 1
    assert day_two_positions.loc["A", "quantity"] == 80
    assert day_two_positions.loc["B", "quantity"] == 20
    day_three_positions = result.positions.loc[
        result.positions["trade_date"].eq(pd.Timestamp("2024-01-04"))
    ]
    assert "A" not in day_three_positions["ts_code"].tolist()


def test_blocked_or_missing_open_buy_leaves_cash_and_is_not_retried() -> None:
    targets = pd.DataFrame(
        {
            "execution_date": ["2024-01-02", "2024-01-02"],
            "ts_code": ["A", "B"],
            "target_weight": [0.5, 0.5],
        }
    )
    market = market_frame(
        [
            ("2024-01-02", "A", 10.0, 10.0, False, True),
            ("2024-01-02", "B", None, 10.0, True, True),
            ("2024-01-03", "A", 10.0, 10.0, True, True),
            ("2024-01-03", "B", 10.0, 10.0, True, True),
        ]
    )

    result = permissive_backtest(
        targets, market, initial_cash=1_000.0, cost_rate=0.0, board_lot=10
    )

    assert result.fills.empty
    assert result.nav["cash"].tolist() == pytest.approx([1_000.0, 1_000.0])
    assert result.nav.iloc[0]["blocked_buys"] == 2


def test_redemption_registration_becomes_receivable_then_cash() -> None:
    targets = pd.DataFrame(
        {
            "execution_date": ["2024-01-02"],
            "ts_code": ["A"],
            "target_weight": [1.0],
        }
    )
    market = market_frame(
        [
            ("2024-01-02", "A", 10.0, 10.0, True, True),
            ("2024-01-03", "A", 10.0, 10.0, True, False),
            ("2024-01-04", "A", None, None, False, False),
            ("2024-01-05", "A", None, None, False, False),
            ("2024-01-08", "A", None, None, False, False),
        ]
    )
    lifecycle = pd.DataFrame(
        {
            "trade_date": ["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"],
            "ts_code": ["A", "A", "A", "A"],
            "lifecycle_state": [
                "last_trading_window",
                "settlement_receivable",
                "settled",
                "settled",
            ],
            "exit_required": [True, False, False, False],
            "settlement_value": [None, 9.0, 9.0, 9.0],
            "settlement_date": [None, "2024-01-05", "2024-01-05", "2024-01-05"],
        }
    )

    result = permissive_backtest(
        targets,
        market,
        lifecycle=lifecycle,
        initial_cash=1_000.0,
        cost_rate=0.0,
        board_lot=10,
    )
    nav = result.nav.set_index("trade_date")

    assert nav.loc["2024-01-03", "blocked_sells"] == 1
    assert nav.loc["2024-01-04", "holdings_value"] == pytest.approx(0.0)
    assert nav.loc["2024-01-04", "receivable_value"] == pytest.approx(900.0)
    assert nav.loc["2024-01-05", "cash"] == pytest.approx(0.0)
    assert nav.loc["2024-01-05", "receivable_value"] == pytest.approx(900.0)
    assert nav.loc["2024-01-08", "cash"] == pytest.approx(900.0)
    assert nav.loc["2024-01-08", "receivable_value"] == pytest.approx(0.0)
    attribution = result.attribution.set_index("trade_date")
    assert attribution.loc["2024-01-04", "settlement_pnl"] == pytest.approx(-100.0)
    assert attribution["attribution_residual"].abs().max() < 1e-9


def test_accounting_identity_holds_each_day() -> None:
    targets = pd.DataFrame(
        {
            "execution_date": ["2024-01-02", "2024-01-03"],
            "ts_code": ["A", "A"],
            "target_weight": [0.9, 0.5],
        }
    )
    market = market_frame(
        [
            ("2024-01-02", "A", 10.0, 11.0, True, True),
            ("2024-01-03", "A", 12.0, 13.0, True, True),
        ]
    )

    result = permissive_backtest(targets, market, initial_cash=1_000.0)

    identity = (
        result.nav["cash"]
        + result.nav["holdings_value"]
        + result.nav["receivable_value"]
    )
    assert result.nav["nav"].tolist() == pytest.approx(identity.tolist())
    assert result.nav["cash_ledger_balance"].tolist() == pytest.approx(
        result.nav["cash"].tolist()
    )
    assert result.nav["accounting_error"].abs().max() < 1e-9


def test_daily_pnl_attribution_reconciles_entry_continuation_and_exit() -> None:
    targets = pd.DataFrame(
        {
            "execution_date": ["2024-01-02", "2024-01-03"],
            "ts_code": ["A", "A"],
            "target_weight": [1.0, 0.5],
        }
    )
    market = market_frame(
        [
            ("2024-01-02", "A", 10.0, 11.0, True, True),
            ("2024-01-03", "A", 12.0, 13.0, True, True),
        ]
    )

    result = permissive_backtest(
        targets, market, initial_cash=1_000.0, cost_rate=0.0, board_lot=10
    )
    attribution = result.attribution.set_index("trade_date")

    assert attribution.loc["2024-01-02", "entry_pnl"] == pytest.approx(100.0)
    assert attribution.loc["2024-01-03", "continuation_pnl"] == pytest.approx(100.0)
    assert attribution.loc["2024-01-03", "exit_pnl"] == pytest.approx(50.0)
    assert attribution["attribution_residual"].abs().max() < 1e-9


def test_mandatory_exit_cannot_be_rebought_from_a_stale_positive_target() -> None:
    targets = pd.DataFrame(
        {
            "execution_date": ["2024-01-02", "2024-01-03"],
            "ts_code": ["A", "A"],
            "target_weight": [1.0, 1.0],
        }
    )
    market = market_frame(
        [
            ("2024-01-02", "A", 10.0, 10.0, True, True),
            ("2024-01-03", "A", 10.0, 10.0, True, True),
        ]
    )
    lifecycle = pd.DataFrame(
        {
            "trade_date": ["2024-01-03"],
            "ts_code": ["A"],
            "lifecycle_state": ["redemption_announced"],
            "entry_allowed": [False],
            "exit_required": [True],
        }
    )

    result = permissive_backtest(
        targets, market, lifecycle=lifecycle, initial_cash=1_000.0, cost_rate=0.0
    )

    day_two = result.fills.loc[result.fills["trade_date"].eq(pd.Timestamp("2024-01-03"))]
    assert day_two["side"].tolist() == ["sell"]
    assert result.positions.loc[
        result.positions["trade_date"].eq(pd.Timestamp("2024-01-03"))
    ].empty


def test_positive_target_with_missing_open_preserves_existing_position() -> None:
    targets = pd.DataFrame(
        {
            "execution_date": ["2024-01-02", "2024-01-03"],
            "ts_code": ["A", "A"],
            "target_weight": [1.0, 1.0],
        }
    )
    market = market_frame(
        [
            ("2024-01-02", "A", 10.0, 10.0, True, True),
            ("2024-01-03", "A", None, 10.0, True, True),
            ("2024-01-04", "A", 10.0, 10.0, True, True),
        ]
    )

    result = permissive_backtest(
        targets, market, initial_cash=1_000.0, cost_rate=0.0, board_lot=10
    )
    quantities = result.positions.groupby("trade_date")["quantity"].sum()

    assert quantities.tolist() == [100, 100, 100]
    assert not result.fills.loc[
        result.fills["trade_date"].gt(pd.Timestamp("2024-01-02"))
    ].shape[0]


def test_unresolved_terminal_settlement_raises_instead_of_using_stale_mark() -> None:
    targets = pd.DataFrame(
        {
            "execution_date": ["2024-01-02"],
            "ts_code": ["A"],
            "target_weight": [1.0],
        }
    )
    market = market_frame(
        [
            ("2024-01-02", "A", 10.0, 10.0, True, True),
            ("2024-01-03", "A", None, None, False, False),
        ]
    )
    lifecycle = pd.DataFrame(
        {
            "trade_date": ["2024-01-03"],
            "ts_code": ["A"],
            "lifecycle_state": ["settlement_unresolved"],
            "entry_allowed": [False],
            "exit_required": [False],
        }
    )

    with pytest.raises(ValueError, match="unresolved terminal settlement"):
        permissive_backtest(targets, market, lifecycle=lifecycle, initial_cash=1_000.0)


def test_production_mode_requires_explicit_tradability_and_complete_lifecycle() -> None:
    targets = pd.DataFrame(
        {
            "execution_date": ["2024-01-02"],
            "ts_code": ["A"],
            "target_weight": [1.0],
        }
    )
    incomplete_market = pd.DataFrame(
        {
            "trade_date": ["2024-01-02"],
            "ts_code": ["A"],
            "open": [10.0],
            "close": [10.0],
        }
    )
    with pytest.raises(KeyError, match="buy_allowed"):
        run_execution_backtest(targets, incomplete_market, initial_cash=1_000.0)

    market = market_frame(
        [("2024-01-02", "A", 10.0, 10.0, True, True)]
    )
    with pytest.raises(ValueError, match="lifecycle coverage"):
        run_execution_backtest(targets, market, initial_cash=1_000.0)


def test_production_mode_rejects_held_bond_missing_from_daily_lifecycle() -> None:
    targets = pd.DataFrame(
        {
            "execution_date": ["2024-01-02"],
            "ts_code": ["A"],
            "target_weight": [1.0],
        }
    )
    market = market_frame(
        [
            ("2024-01-02", "A", 10.0, 10.0, True, True),
            ("2024-01-03", "B", 10.0, 10.0, True, True),
        ]
    )
    lifecycle = pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-03"],
            "ts_code": ["A", "B"],
            "lifecycle_state": ["active", "active"],
            "entry_allowed": [True, True],
            "exit_required": [False, False],
        }
    )

    with pytest.raises(ValueError, match="held bond A"):
        run_execution_backtest(
            targets,
            market,
            lifecycle=lifecycle,
            initial_cash=1_000.0,
            cost_rate=0.0,
        )


def test_payment_date_receivable_is_not_spendable_until_next_market_open() -> None:
    targets = pd.DataFrame(
        {
            "execution_date": ["2024-01-02", "2024-01-05", "2024-01-08"],
            "ts_code": ["A", "B", "B"],
            "target_weight": [1.0, 1.0, 1.0],
        }
    )
    market = market_frame(
        [
            ("2024-01-02", "A", 10.0, 10.0, True, True),
            ("2024-01-02", "B", 10.0, 10.0, True, True),
            ("2024-01-05", "A", None, None, False, False),
            ("2024-01-05", "B", 10.0, 10.0, True, True),
            ("2024-01-08", "A", None, None, False, False),
            ("2024-01-08", "B", 10.0, 10.0, True, True),
        ]
    )
    lifecycle = pd.DataFrame(
        {
            "trade_date": ["2024-01-05", "2024-01-08"],
            "ts_code": ["A", "A"],
            "lifecycle_state": ["settlement_receivable", "settled"],
            "entry_allowed": [False, False],
            "exit_required": [False, False],
            "settlement_value": [10.0, 10.0],
            "settlement_date": ["2024-01-05", "2024-01-05"],
        }
    )

    result = permissive_backtest(
        targets,
        market,
        lifecycle=lifecycle,
        initial_cash=1_000.0,
        cost_rate=0.0,
        board_lot=10,
    )
    buys = result.fills.loc[result.fills["side"].eq("buy")]

    assert buys.loc[buys["trade_date"].eq(pd.Timestamp("2024-01-05")), "ts_code"].empty
    assert buys.loc[buys["trade_date"].eq(pd.Timestamp("2024-01-08")), "ts_code"].tolist() == ["B"]


def test_new_positive_target_cancels_stale_failed_ordinary_sell() -> None:
    targets = pd.DataFrame(
        {
            "execution_date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "ts_code": ["A", "A", "A"],
            "target_weight": [1.0, 0.0, 1.0],
        }
    )
    market = market_frame(
        [
            ("2024-01-02", "A", 10.0, 10.0, True, True),
            ("2024-01-03", "A", 10.0, 10.0, True, False),
            ("2024-01-04", "A", 10.0, 10.0, True, True),
        ]
    )

    result = permissive_backtest(
        targets, market, initial_cash=1_000.0, cost_rate=0.0, board_lot=10
    )
    final_day_fills = result.fills.loc[
        result.fills["trade_date"].eq(pd.Timestamp("2024-01-04"))
    ]

    assert final_day_fills.empty
    assert result.positions.iloc[-1]["quantity"] == 100
