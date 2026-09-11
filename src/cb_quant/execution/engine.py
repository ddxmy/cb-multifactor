"""Cash-constrained daily execution for long-only convertible-bond portfolios."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .models import ExecutionResult


ORDER_COLUMNS = [
    "trade_date",
    "ts_code",
    "side",
    "requested_quantity",
    "filled_quantity",
    "status",
    "reason",
]
FILL_COLUMNS = [
    "trade_date",
    "ts_code",
    "side",
    "quantity",
    "price",
    "gross_notional",
    "cost",
    "net_cash_flow",
]
POSITION_COLUMNS = [
    "trade_date",
    "ts_code",
    "quantity",
    "mark_price",
    "market_value",
]
RECEIVABLE_COLUMNS = [
    "trade_date",
    "ts_code",
    "quantity",
    "settlement_price",
    "settlement_date",
    "cash_available_date",
    "receivable_value",
    "status",
]
ATTRIBUTION_COLUMNS = [
    "trade_date",
    "continuation_pnl",
    "entry_pnl",
    "exit_pnl",
    "settlement_pnl",
    "transaction_cost",
    "attributed_pnl",
    "nav_change",
    "attribution_residual",
]


def _normalize_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="raise").dt.normalize()


def _prepare_targets(targets: pd.DataFrame) -> pd.DataFrame:
    required = {"execution_date", "ts_code", "target_weight"}
    if missing := required.difference(targets.columns):
        raise KeyError(f"targets are missing columns: {sorted(missing)}")
    result = targets[list(required)].copy()
    result["execution_date"] = _normalize_date(result["execution_date"])
    result["ts_code"] = result["ts_code"].astype("string")
    result["target_weight"] = pd.to_numeric(result["target_weight"], errors="coerce")
    if result.duplicated(["execution_date", "ts_code"]).any():
        raise ValueError("targets contain duplicate execution-date and bond keys")
    if result["target_weight"].isna().any() or result["target_weight"].lt(0).any():
        raise ValueError("target weights must be non-negative finite values")
    totals = result.groupby("execution_date")["target_weight"].sum()
    if totals.gt(1.0 + 1e-12).any():
        raise ValueError("target weights cannot exceed one on an execution date")
    return result.sort_values(["execution_date", "ts_code"]).reset_index(drop=True)


def _prepare_market(market: pd.DataFrame) -> pd.DataFrame:
    required = {
        "trade_date",
        "ts_code",
        "open",
        "close",
        "buy_allowed",
        "sell_allowed",
    }
    if missing := required.difference(market.columns):
        raise KeyError(f"market data are missing columns: {sorted(missing)}")
    result = market.copy()
    result["trade_date"] = _normalize_date(result["trade_date"])
    result["ts_code"] = result["ts_code"].astype("string")
    for column in ("open", "close"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in ("buy_allowed", "sell_allowed"):
        result[column] = result[column].fillna(False).astype(bool)
    if result.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("market data contain duplicate date and bond keys")
    return result.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def _prepare_lifecycle(lifecycle: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "trade_date",
        "ts_code",
        "lifecycle_state",
        "entry_allowed",
        "exit_required",
        "settlement_value",
        "settlement_date",
    ]
    if lifecycle is None or lifecycle.empty:
        return pd.DataFrame(columns=columns)
    required = {"trade_date", "ts_code", "lifecycle_state"}
    if missing := required.difference(lifecycle.columns):
        raise KeyError(f"lifecycle data are missing columns: {sorted(missing)}")
    result = lifecycle.copy()
    for column in columns:
        if column not in result:
            if column == "exit_required":
                result[column] = False
            elif column == "entry_allowed":
                result[column] = result["lifecycle_state"].isin(
                    ["active", "redemption_watch"]
                )
            elif column == "settlement_value":
                result[column] = np.nan
            else:
                result[column] = pd.NaT
    result["trade_date"] = _normalize_date(result["trade_date"])
    result["ts_code"] = result["ts_code"].astype("string")
    result["lifecycle_state"] = result["lifecycle_state"].astype("string")
    result["exit_required"] = result["exit_required"].fillna(False).astype(bool)
    result["entry_allowed"] = result["entry_allowed"].fillna(False).astype(bool)
    result["settlement_value"] = pd.to_numeric(
        result["settlement_value"], errors="coerce"
    )
    result["settlement_date"] = pd.to_datetime(
        result["settlement_date"], errors="coerce"
    ).dt.normalize()
    if result.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("lifecycle data contain duplicate date and bond keys")
    return result[columns].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def _valid_price(value: object) -> bool:
    return pd.notna(value) and np.isfinite(float(value)) and float(value) > 0


def _round_lot(quantity: float, board_lot: int) -> int:
    return max(int(np.floor(quantity / board_lot + 1e-12)) * board_lot, 0)


def _market_lookup(frame: pd.DataFrame) -> Mapping[tuple[pd.Timestamp, str], dict]:
    return {
        (row.trade_date, str(row.ts_code)): {
            "open": row.open,
            "close": row.close,
            "buy_allowed": bool(row.buy_allowed),
            "sell_allowed": bool(row.sell_allowed),
        }
        for row in frame.itertuples(index=False)
    }


def _lifecycle_lookup(frame: pd.DataFrame) -> Mapping[tuple[pd.Timestamp, str], dict]:
    return {
        (row.trade_date, str(row.ts_code)): {
            "lifecycle_state": row.lifecycle_state,
            "entry_allowed": bool(row.entry_allowed),
            "exit_required": bool(row.exit_required),
            "settlement_value": row.settlement_value,
            "settlement_date": row.settlement_date,
        }
        for row in frame.itertuples(index=False)
    }


def run_execution_backtest(
    targets: pd.DataFrame,
    market: pd.DataFrame,
    *,
    lifecycle: pd.DataFrame | None = None,
    initial_cash: float = 1_000_000.0,
    cost_rate: float = 0.0015,
    board_lot: int = 10,
    allow_missing_lifecycle: bool = False,
) -> ExecutionResult:
    """Simulate next-open orders and daily close marking with explicit cash."""
    if not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("initial_cash must be positive and finite")
    if not np.isfinite(cost_rate) or cost_rate < 0:
        raise ValueError("cost_rate must be non-negative and finite")
    if not isinstance(board_lot, int) or board_lot <= 0:
        raise ValueError("board_lot must be a positive integer")
    if not isinstance(allow_missing_lifecycle, bool):
        raise TypeError("allow_missing_lifecycle must be boolean")

    target_data = _prepare_targets(targets)
    market_data = _prepare_market(market)
    lifecycle_data = _prepare_lifecycle(lifecycle)
    dates = pd.DatetimeIndex(market_data["trade_date"].unique()).sort_values()
    if dates.empty:
        raise ValueError("market data cannot be empty")
    if not target_data["execution_date"].isin(dates).all():
        raise ValueError("some target execution dates are outside the market calendar")
    if not allow_missing_lifecycle:
        market_keys = set(
            zip(
                market_data["trade_date"],
                market_data["ts_code"].astype(str),
                strict=True,
            )
        )
        lifecycle_keys = set(
            zip(
                lifecycle_data["trade_date"],
                lifecycle_data["ts_code"].astype(str),
                strict=True,
            )
        )
        missing_lifecycle = market_keys.difference(lifecycle_keys)
        if missing_lifecycle:
            raise ValueError(
                "lifecycle coverage is incomplete for "
                f"{len(missing_lifecycle)} market rows"
            )

    market_by_key = _market_lookup(market_data)
    lifecycle_by_key = _lifecycle_lookup(lifecycle_data)
    target_by_date = {
        date: group.set_index("ts_code")["target_weight"].to_dict()
        for date, group in target_data.groupby("execution_date", sort=True)
    }
    all_codes = sorted(
        set(market_data["ts_code"].astype(str))
        | set(target_data["ts_code"].astype(str))
        | set(lifecycle_data["ts_code"].astype(str))
    )

    cash = float(initial_cash)
    cumulative_trade_cash_flow = 0.0
    cumulative_settlement_cash_flow = 0.0
    previous_nav = float(initial_cash)
    positions: dict[str, int] = {}
    pending_sell_target: dict[str, int] = {}
    mandatory_sell_codes: set[str] = set()
    receivables: list[dict[str, object]] = []
    last_mark: dict[str, float] = {}
    nav_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []
    order_rows: list[dict[str, object]] = []
    fill_rows: list[dict[str, object]] = []
    receivable_rows: list[dict[str, object]] = []
    attribution_rows: list[dict[str, object]] = []

    for trade_date in dates:
        previous_positions = positions.copy()
        previous_marks = last_mark.copy()
        day_cost = 0.0
        day_turnover = 0.0
        settlement_pnl = 0.0
        settled_codes: set[str] = set()
        day_buy_fills: dict[str, tuple[int, float]] = {}
        day_sell_fills: dict[str, tuple[int, float]] = {}
        blocked_buys = 0
        blocked_sells = 0

        if not allow_missing_lifecycle:
            missing_held_states = [
                code
                for code in positions
                if (trade_date, code) not in lifecycle_by_key
            ]
            if missing_held_states:
                raise ValueError(
                    f"held bond {missing_held_states[0]} is missing lifecycle coverage "
                    f"on {trade_date.date()}"
                )

        for receivable in receivables:
            if (
                receivable["status"] == "outstanding"
                and pd.notna(receivable["cash_available_date"])
                and trade_date >= receivable["cash_available_date"]
            ):
                paid_value = float(receivable["receivable_value"])
                cash += paid_value
                cumulative_settlement_cash_flow += paid_value
                receivable["status"] = "paid"
                receivable["paid_date"] = trade_date

        for code in list(positions):
            state = lifecycle_by_key.get((trade_date, code), {})
            lifecycle_state = state.get("lifecycle_state")
            if lifecycle_state == "settlement_unresolved":
                raise ValueError(
                    f"unresolved terminal settlement for held bond {code} "
                    f"on {trade_date.date()}"
                )
            if lifecycle_state not in {"settlement_receivable", "settled"}:
                continue
            settlement_price = state.get("settlement_value")
            settlement_date = state.get("settlement_date")
            if not _valid_price(settlement_price) or pd.isna(settlement_date):
                continue
            quantity = positions.pop(code)
            pending_sell_target.pop(code, None)
            mandatory_sell_codes.discard(code)
            value = quantity * float(settlement_price)
            if code not in previous_marks:
                raise ValueError(
                    f"cannot attribute terminal settlement for {code} on "
                    f"{trade_date.date()} without a prior mark"
                )
            settlement_pnl += quantity * (
                float(settlement_price) - previous_marks[code]
            )
            settled_codes.add(code)
            record = {
                "ts_code": code,
                "quantity": quantity,
                "settlement_price": float(settlement_price),
                "settlement_date": pd.Timestamp(settlement_date),
                "cash_available_date": (
                    dates[next_position]
                    if (
                        next_position := dates.searchsorted(
                            pd.Timestamp(settlement_date), side="right"
                        )
                    )
                    < len(dates)
                    else pd.NaT
                ),
                "receivable_value": value,
                "status": "outstanding",
            }
            receivables.append(record)

        for code, quantity in positions.items():
            state = lifecycle_by_key.get((trade_date, code), {})
            if state.get("exit_required", False):
                pending_sell_target[code] = 0
                mandatory_sell_codes.add(code)

        scheduled_targets = target_by_date.get(trade_date)
        desired_quantities: dict[str, int] = {}
        unpriced_buy_targets: list[str] = []
        if scheduled_targets is not None:
            open_marks: dict[str, float] = {}
            for code in all_codes:
                market_row = market_by_key.get((trade_date, code), {})
                open_price = market_row.get("open")
                if _valid_price(open_price):
                    open_marks[code] = float(open_price)
                elif code in last_mark:
                    open_marks[code] = last_mark[code]
            pretrade_holdings = sum(
                quantity * open_marks.get(code, last_mark.get(code, 0.0))
                for code, quantity in positions.items()
            )
            outstanding_receivables = sum(
                float(item["receivable_value"])
                for item in receivables
                if item["status"] == "outstanding"
            )
            pretrade_nav = cash + pretrade_holdings + outstanding_receivables
            for code in all_codes:
                weight = float(scheduled_targets.get(code, 0.0))
                open_price = market_by_key.get((trade_date, code), {}).get("open")
                lifecycle_state = lifecycle_by_key.get((trade_date, code))
                entry_allowed = (
                    lifecycle_state is None
                    or bool(lifecycle_state.get("entry_allowed", False))
                )
                if weight > 0 and entry_allowed and _valid_price(open_price):
                    desired_quantities[code] = _round_lot(
                        pretrade_nav * weight / float(open_price), board_lot
                    )
                elif weight > 0 and entry_allowed:
                    unpriced_buy_targets.append(code)
                elif weight > 0:
                    desired_quantities[code] = 0
                elif weight <= 0:
                    desired_quantities[code] = 0
            for code, current_quantity in positions.items():
                target_weight = float(scheduled_targets.get(code, 0.0))
                if target_weight > 0 and code not in desired_quantities:
                    if code not in mandatory_sell_codes:
                        pending_sell_target.pop(code, None)
                    continue
                desired = desired_quantities.get(code, 0)
                if current_quantity > desired:
                    pending_sell_target[code] = desired
                elif code not in mandatory_sell_codes:
                    pending_sell_target.pop(code, None)

        sell_codes = sorted(pending_sell_target)
        for code in sell_codes:
            current_quantity = positions.get(code, 0)
            desired_quantity = pending_sell_target[code]
            quantity = max(current_quantity - desired_quantity, 0)
            if quantity == 0:
                pending_sell_target.pop(code, None)
                continue
            market_row = market_by_key.get((trade_date, code), {})
            open_price = market_row.get("open")
            can_fill = bool(market_row.get("sell_allowed", False)) and _valid_price(open_price)
            if not can_fill:
                blocked_sells += 1
                order_rows.append(
                    {
                        "trade_date": trade_date,
                        "ts_code": code,
                        "side": "sell",
                        "requested_quantity": quantity,
                        "filled_quantity": 0,
                        "status": "blocked",
                        "reason": "sell_not_tradable_or_missing_open",
                    }
                )
                continue
            price = float(open_price)
            gross_notional = quantity * price
            cost = gross_notional * cost_rate
            cash += gross_notional - cost
            cumulative_trade_cash_flow += gross_notional - cost
            remaining = current_quantity - quantity
            if remaining > 0:
                positions[code] = remaining
            else:
                positions.pop(code, None)
            pending_sell_target.pop(code, None)
            mandatory_sell_codes.discard(code)
            day_cost += cost
            day_turnover += gross_notional
            order_rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": code,
                    "side": "sell",
                    "requested_quantity": quantity,
                    "filled_quantity": quantity,
                    "status": "filled",
                    "reason": "target_reduction_or_mandatory_exit",
                }
            )
            fill_rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": code,
                    "side": "sell",
                    "quantity": quantity,
                    "price": price,
                    "gross_notional": gross_notional,
                    "cost": cost,
                    "net_cash_flow": gross_notional - cost,
                }
            )
            day_sell_fills[code] = (quantity, price)

        if scheduled_targets is not None:
            for code in sorted(unpriced_buy_targets):
                blocked_buys += 1
                order_rows.append(
                    {
                        "trade_date": trade_date,
                        "ts_code": code,
                        "side": "buy",
                        "requested_quantity": 0,
                        "filled_quantity": 0,
                        "status": "blocked",
                        "reason": "missing_valid_open_for_target_sizing",
                    }
                )
            buy_candidates = sorted(
                (
                    code
                    for code, desired in desired_quantities.items()
                    if desired > positions.get(code, 0)
                ),
                key=lambda code: (-float(scheduled_targets.get(code, 0.0)), code),
            )
            for code in buy_candidates:
                desired = desired_quantities[code]
                requested = desired - positions.get(code, 0)
                lifecycle_state = lifecycle_by_key.get((trade_date, code))
                if lifecycle_state is not None and not lifecycle_state.get(
                    "entry_allowed", False
                ):
                    blocked_buys += 1
                    order_rows.append(
                        {
                            "trade_date": trade_date,
                            "ts_code": code,
                            "side": "buy",
                            "requested_quantity": requested,
                            "filled_quantity": 0,
                            "status": "blocked",
                            "reason": "lifecycle_entry_blocked",
                        }
                    )
                    continue
                market_row = market_by_key.get((trade_date, code), {})
                open_price = market_row.get("open")
                can_fill = bool(market_row.get("buy_allowed", False)) and _valid_price(open_price)
                if not can_fill:
                    blocked_buys += 1
                    order_rows.append(
                        {
                            "trade_date": trade_date,
                            "ts_code": code,
                            "side": "buy",
                            "requested_quantity": requested,
                            "filled_quantity": 0,
                            "status": "blocked",
                            "reason": "buy_not_tradable_or_missing_open",
                        }
                    )
                    continue
                price = float(open_price)
                affordable = _round_lot(cash / (price * (1.0 + cost_rate)), board_lot)
                quantity = min(requested, affordable)
                if quantity <= 0:
                    order_rows.append(
                        {
                            "trade_date": trade_date,
                            "ts_code": code,
                            "side": "buy",
                            "requested_quantity": requested,
                            "filled_quantity": 0,
                            "status": "unfilled",
                            "reason": "insufficient_cash_after_rounding",
                        }
                    )
                    continue
                gross_notional = quantity * price
                cost = gross_notional * cost_rate
                cash -= gross_notional + cost
                cumulative_trade_cash_flow -= gross_notional + cost
                positions[code] = positions.get(code, 0) + quantity
                day_cost += cost
                day_turnover += gross_notional
                status = "filled" if quantity == requested else "partially_filled"
                order_rows.append(
                    {
                        "trade_date": trade_date,
                        "ts_code": code,
                        "side": "buy",
                        "requested_quantity": requested,
                        "filled_quantity": quantity,
                        "status": status,
                        "reason": "target_increase",
                    }
                )
                fill_rows.append(
                    {
                        "trade_date": trade_date,
                        "ts_code": code,
                        "side": "buy",
                        "quantity": quantity,
                        "price": price,
                        "gross_notional": gross_notional,
                        "cost": cost,
                        "net_cash_flow": -(gross_notional + cost),
                    }
                )
                day_buy_fills[code] = (quantity, price)

        holdings_value = 0.0
        for code, quantity in sorted(positions.items()):
            market_row = market_by_key.get((trade_date, code), {})
            close_price = market_row.get("close")
            open_price = market_row.get("open")
            if _valid_price(close_price):
                mark_price = float(close_price)
            elif code in last_mark:
                mark_price = last_mark[code]
            elif _valid_price(open_price):
                mark_price = float(open_price)
            else:
                raise ValueError(f"cannot mark held bond {code} on {trade_date.date()}")
            last_mark[code] = mark_price
            market_value = quantity * mark_price
            holdings_value += market_value
            position_rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": code,
                    "quantity": quantity,
                    "mark_price": mark_price,
                    "market_value": market_value,
                }
            )

        outstanding_value = sum(
            float(item["receivable_value"])
            for item in receivables
            if item["status"] == "outstanding"
        )
        nav = cash + holdings_value + outstanding_value
        cash_ledger_balance = (
            initial_cash
            + cumulative_trade_cash_flow
            + cumulative_settlement_cash_flow
        )
        accounting_error = cash - cash_ledger_balance

        continuation_pnl = 0.0
        entry_pnl = 0.0
        exit_pnl = 0.0
        attribution_codes = set(previous_positions) | set(positions)
        for code in attribution_codes.difference(settled_codes):
            previous_quantity = previous_positions.get(code, 0)
            current_quantity = positions.get(code, 0)
            continuing_quantity = min(previous_quantity, current_quantity)
            if continuing_quantity:
                continuation_pnl += continuing_quantity * (
                    last_mark[code] - previous_marks[code]
                )
            if current_quantity > previous_quantity:
                quantity, price = day_buy_fills[code]
                if quantity != current_quantity - previous_quantity:
                    raise ValueError("buy fill does not reconcile with position change")
                entry_pnl += quantity * (last_mark[code] - price)
            if previous_quantity > current_quantity:
                quantity, price = day_sell_fills[code]
                if quantity != previous_quantity - current_quantity:
                    raise ValueError("sell fill does not reconcile with position change")
                exit_pnl += quantity * (price - previous_marks[code])

        nav_change = nav - previous_nav
        attributed_pnl = (
            continuation_pnl
            + entry_pnl
            + exit_pnl
            + settlement_pnl
            - day_cost
        )
        attribution_residual = nav_change - attributed_pnl
        attribution_rows.append(
            {
                "trade_date": trade_date,
                "continuation_pnl": continuation_pnl,
                "entry_pnl": entry_pnl,
                "exit_pnl": exit_pnl,
                "settlement_pnl": settlement_pnl,
                "transaction_cost": -day_cost,
                "attributed_pnl": attributed_pnl,
                "nav_change": nav_change,
                "attribution_residual": attribution_residual,
            }
        )
        previous_nav = nav
        nav_rows.append(
            {
                "trade_date": trade_date,
                "cash": cash,
                "holdings_value": holdings_value,
                "receivable_value": outstanding_value,
                "nav": nav,
                "cash_ledger_balance": cash_ledger_balance,
                "daily_cost": day_cost,
                "traded_notional": day_turnover,
                "blocked_buys": blocked_buys,
                "blocked_sells": blocked_sells,
                "accounting_error": accounting_error,
            }
        )
        for item in receivables:
            receivable_rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": item["ts_code"],
                    "quantity": item["quantity"],
                    "settlement_price": item["settlement_price"],
                    "settlement_date": item["settlement_date"],
                    "cash_available_date": item["cash_available_date"],
                    "receivable_value": item["receivable_value"],
                    "status": item["status"],
                }
            )

    return ExecutionResult(
        nav=pd.DataFrame(nav_rows),
        positions=pd.DataFrame(position_rows, columns=POSITION_COLUMNS),
        orders=pd.DataFrame(order_rows, columns=ORDER_COLUMNS),
        fills=pd.DataFrame(fill_rows, columns=FILL_COLUMNS),
        receivables=pd.DataFrame(receivable_rows, columns=RECEIVABLE_COLUMNS),
        attribution=pd.DataFrame(attribution_rows, columns=ATTRIBUTION_COLUMNS),
    )
