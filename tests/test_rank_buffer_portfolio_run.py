from __future__ import annotations

import pandas as pd

import cb_quant.portfolio as portfolio


def test_rank_buffer_run_keeps_retained_names_without_round_trip() -> None:
    scores = pd.DataFrame(
        {
            "signal_date": ["2024-01-01"] * 3 + ["2024-01-03"] * 3,
            "ts_code": ["A", "B", "C", "A", "B", "C"],
            "processed_factor": [3.0, 2.0, 1.0, 3.0, 1.0, 2.0],
            "is_eligible": [True] * 6,
        }
    )
    schedule = pd.DataFrame(
        {
            "signal_date": ["2024-01-01", "2024-01-03"],
            "entry_date": ["2024-01-02", "2024-01-04"],
        }
    )
    dates = pd.date_range("2024-01-02", "2024-01-08", freq="B")
    market = pd.DataFrame(
        [(date, code, 100.0, 100.0, True, True) for date in dates for code in ["A", "B"]],
        columns=["trade_date", "ts_code", "open", "close", "buy_allowed", "sell_allowed"],
    )
    lifecycle = market[["trade_date", "ts_code"]].assign(
        lifecycle_state="active",
        entry_allowed=True,
        exit_required=False,
        settlement_value=float("nan"),
        settlement_date=pd.NaT,
    )

    run = portfolio.run_rank_buffer_factor_portfolio(
        scores,
        schedule,
        market,
        lifecycle,
        portfolio_size=2,
        exit_rank=3,
        cash_reserve=0.1,
        max_weight=1.0,
        liquidation_date="2024-01-08",
        initial_cash=1_000.0,
        cost_rate=0.0,
        board_lot=1,
    )

    assert run.execution.fills[["trade_date", "ts_code", "side"]].to_dict("records") == [
        {"trade_date": pd.Timestamp("2024-01-02"), "ts_code": "A", "side": "buy"},
        {"trade_date": pd.Timestamp("2024-01-02"), "ts_code": "B", "side": "buy"},
        {"trade_date": pd.Timestamp("2024-01-08"), "ts_code": "A", "side": "sell"},
        {"trade_date": pd.Timestamp("2024-01-08"), "ts_code": "B", "side": "sell"},
    ]
    assert run.summary["final_nav"] == 1_000.0
