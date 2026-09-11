"""Result containers for the daily execution simulator."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ExecutionResult:
    """Auditable tables emitted by one portfolio simulation."""

    nav: pd.DataFrame
    positions: pd.DataFrame
    orders: pd.DataFrame
    fills: pd.DataFrame
    receivables: pd.DataFrame
    attribution: pd.DataFrame
