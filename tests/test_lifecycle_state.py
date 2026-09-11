from __future__ import annotations

import pandas as pd
import pytest

from cb_quant.lifecycle import LifecycleState, build_lifecycle_panel


def state_by_date(panel: pd.DataFrame, code: str = "110001.SH") -> pd.DataFrame:
    return panel.loc[panel["ts_code"].eq(code)].set_index("trade_date")


def test_builds_pre_listed_seasoning_and_active_states_on_trading_calendar() -> None:
    dates = pd.bdate_range("2024-01-02", periods=6)
    terms = pd.DataFrame(
        {
            "ts_code": ["110001.SH"],
            "list_date": ["2024-01-03"],
            "delist_date": [pd.NaT],
            "maturity_date": ["2030-01-03"],
        }
    )

    result = state_by_date(
        build_lifecycle_panel(terms, dates, seasoning_days=3)
    )

    assert result.loc["2024-01-02", "lifecycle_state"] == LifecycleState.PRE_LISTED
    assert result.loc["2024-01-03", "lifecycle_state"] == LifecycleState.SEASONING
    assert result.loc["2024-01-04", "lifecycle_state"] == LifecycleState.SEASONING
    assert result.loc["2024-01-05", "lifecycle_state"] == LifecycleState.ACTIVE
    assert bool(result.loc["2024-01-05", "entry_allowed"])


def test_forced_redemption_moves_from_watch_to_exit_and_cash_settlement() -> None:
    dates = pd.bdate_range("2024-01-02", "2024-01-12")
    terms = pd.DataFrame(
        {
            "ts_code": ["110001.SH"],
            "list_date": ["2020-01-02"],
            "delist_date": [pd.NaT],
            "maturity_date": ["2030-01-02"],
        }
    )
    events = pd.DataFrame(
        {
            "ts_code": ["110001.SH", "110001.SH"],
            "effective_date": ["2024-01-03", "2024-01-05"],
            "event_status": ["redemption_watch", "implementation_announced"],
            "last_trade_date": [pd.NaT, "2024-01-08"],
            "call_reg_date": [pd.NaT, "2024-01-09"],
            "payment_date": [pd.NaT, "2024-01-11"],
            "call_price": [pd.NA, 100.8],
        }
    )

    result = state_by_date(build_lifecycle_panel(terms, dates, events))

    assert result.loc["2024-01-03", "lifecycle_state"] == LifecycleState.REDEMPTION_WATCH
    assert bool(result.loc["2024-01-03", "entry_allowed"])
    assert result.loc["2024-01-05", "lifecycle_state"] == LifecycleState.REDEMPTION_ANNOUNCED
    assert bool(result.loc["2024-01-05", "exit_required"])
    assert not bool(result.loc["2024-01-05", "entry_allowed"])
    assert result.loc["2024-01-08", "lifecycle_state"] == LifecycleState.LAST_TRADING_WINDOW
    assert result.loc["2024-01-09", "lifecycle_state"] == LifecycleState.SETTLEMENT_RECEIVABLE
    assert result.loc["2024-01-11", "lifecycle_state"] == LifecycleState.SETTLED
    assert result.loc["2024-01-09", "settlement_value"] == pytest.approx(100.8)
    assert result.loc["2024-01-11", "settlement_date"] == pd.Timestamp("2024-01-11")


def test_maturity_creates_receivable_and_settlement_when_terms_are_known() -> None:
    dates = pd.bdate_range("2024-01-02", "2024-01-10")
    terms = pd.DataFrame(
        {
            "ts_code": ["110001.SH"],
            "list_date": ["2020-01-02"],
            "delist_date": ["2024-01-05"],
            "maturity_date": ["2024-01-08"],
            "maturity_payment_date": ["2024-01-10"],
            "maturity_call_price": [108.0],
        }
    )

    result = state_by_date(build_lifecycle_panel(terms, dates))

    # A static delist date has no point-in-time availability field, so it must not
    # create a look-ahead exit signal before the known maturity event takes effect.
    assert result.loc["2024-01-05", "lifecycle_state"] == LifecycleState.ACTIVE
    assert result.loc["2024-01-08", "lifecycle_state"] == LifecycleState.SETTLEMENT_RECEIVABLE
    assert result.loc["2024-01-10", "lifecycle_state"] == LifecycleState.SETTLED
    assert result.loc["2024-01-08", "settlement_value"] == pytest.approx(108.0)


def test_unvalidated_delisting_is_never_silently_settled() -> None:
    dates = pd.bdate_range("2024-01-02", "2024-01-08")
    terms = pd.DataFrame(
        {
            "ts_code": ["110001.SH"],
            "list_date": ["2020-01-02"],
            "delist_date": ["2024-01-05"],
            "maturity_date": ["2030-01-02"],
        }
    )

    result = state_by_date(build_lifecycle_panel(terms, dates))

    assert result.loc["2024-01-05", "lifecycle_state"] == LifecycleState.ACTIVE
    assert result.loc["2024-01-08", "lifecycle_state"] == LifecycleState.DELISTED_UNRESOLVED
    assert pd.isna(result.loc["2024-01-08", "settlement_value"])


def test_active_unquoted_bond_is_not_entry_eligible() -> None:
    dates = pd.bdate_range("2024-01-02", periods=2)
    terms = pd.DataFrame(
        {
            "ts_code": ["110001.SH"],
            "list_date": ["2020-01-02"],
            "delist_date": [pd.NaT],
            "maturity_date": ["2030-01-02"],
        }
    )
    tradability = pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": ["110001.SH", "110001.SH"],
            "is_tradable": [True, False],
        }
    )

    result = state_by_date(build_lifecycle_panel(terms, dates, tradability=tradability))

    assert result.iloc[1]["lifecycle_state"] == LifecycleState.SUSPENDED_OR_UNQUOTED
    assert not bool(result.iloc[1]["entry_allowed"])


def test_rejects_duplicate_terms_and_inconsistent_event_timeline() -> None:
    dates = pd.bdate_range("2024-01-02", periods=5)
    terms = pd.DataFrame(
        {
            "ts_code": ["110001.SH"],
            "list_date": ["2020-01-02"],
            "maturity_date": ["2030-01-02"],
        }
    )
    events = pd.DataFrame(
        {
            "ts_code": ["110001.SH"],
            "effective_date": ["2024-01-03"],
            "event_status": ["implementation_announced"],
            "last_trade_date": ["2024-01-02"],
            "call_reg_date": ["2024-01-05"],
            "payment_date": ["2024-01-08"],
            "call_price": [100.5],
        }
    )

    with pytest.raises(ValueError, match="event timeline"):
        build_lifecycle_panel(terms, dates, events)
    with pytest.raises(ValueError, match="duplicate"):
        build_lifecycle_panel(pd.concat([terms, terms]), dates)


def test_incomplete_redemption_terms_are_explicitly_unresolved() -> None:
    dates = pd.bdate_range("2024-01-02", "2024-01-08")
    terms = pd.DataFrame(
        {
            "ts_code": ["110001.SH"],
            "list_date": ["2020-01-02"],
            "maturity_date": ["2030-01-02"],
        }
    )
    events = pd.DataFrame(
        {
            "ts_code": ["110001.SH"],
            "effective_date": ["2024-01-03"],
            "event_status": ["implementation_announced"],
            "last_trade_date": ["2024-01-04"],
            "call_reg_date": ["2024-01-05"],
            "payment_date": [pd.NaT],
            "call_price": [pd.NA],
        }
    )

    result = state_by_date(build_lifecycle_panel(terms, dates, events))

    assert result.loc["2024-01-05", "lifecycle_state"] == LifecycleState.SETTLEMENT_UNRESOLVED
    assert not bool(result.loc["2024-01-05", "entry_allowed"])
