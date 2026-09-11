# Adding a Factor

## Scope Boundary

A new factor changes the factor layer only. Label construction, preprocessing, evaluation,
portfolio construction, execution, lifecycle handling, cash accounting, and reporting remain
shared framework responsibilities.

Each production factor requires:

1. one factor module under `src/cb_quant/factors/`; and
2. one explicit registration line in `src/cb_quant/factors/catalog.py`.

Automatic module discovery is intentionally avoided. Explicit registration keeps experimental
formulas out of production runs until they are reviewed.

## Factor Module Template

```python
"""Short description of the factor and its data source."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .base import FactorSpec
from .context import FactorDataBundle
from .output import build_factor_output


FACTOR_SPEC = FactorSpec(
    name="factor_name",
    family="factor_family",
    description="Economic meaning of the factor.",
    direction=1,
    required_fields=("source_column",),
    required_history_fields={
        "stock_daily": ("trade_date", "stk_code", "close", "adj_factor"),
    },
    default_parameters={"window": 20},
    neutralizers=("remaining_balance", "rating"),
)


def calculate_raw_factor(
    data: FactorDataBundle,
    parameters: Mapping[str, object],
) -> pd.Series:
    """Calculate the reviewed point-in-time formula on signal rows."""
    window = int(parameters["window"])
    if window <= 0:
        raise ValueError("window must be positive")
    return data.signal["source_column"]


def calculate_factor(
    data: FactorDataBundle,
    parameters: Mapping[str, object],
) -> pd.DataFrame:
    """Apply the shared output contract to the raw formula."""
    raw_factor = calculate_raw_factor(data, parameters)
    return build_factor_output(data.signal, raw_factor)
```

The corresponding catalog change is one line:

```python
registry.register(FACTOR_SPEC, calculate_factor)
```

The module import is also added at the top of `catalog.py` so the two registered objects are
available. No pipeline or backtest-engine file should change.

## Required Metadata

- `name`: stable lower-snake-case identifier used in configurations and outputs.
- `family`: economic grouping such as valuation, equity, linkage, liquidity, or technical.
- `description`: concise economic interpretation rather than only a mathematical formula.
- `direction`: `1` when larger raw values are preferred and `-1` when smaller values are preferred.
- `required_fields`: context columns that must exist before the formula can run.
- `required_history_fields`: fields required from validated valuation, convertible-bond daily,
  underlying-equity daily, or lifecycle histories.
- `default_parameters`: visible formula parameters, including windows and thresholds.
- `neutralizers`: candidate controls to discuss at the preprocessing stage; listing them does not
  silently apply neutralization.

## Standard Output Contract

Every calculator returns exactly one row per supplied signal-date and bond key, with at least:

```text
signal_date | ts_code | raw_factor | available_date
```

`available_date` is the date on which the input information became usable. It must be non-missing
for every observed factor and cannot be later than `signal_date`. This field is the main defense
against introducing financial statements, rating changes, conversion terms, or event information
before their historical disclosure time.

Optional formula components may be retained as audit columns. The registry rejects duplicate keys,
non-finite factor values, missing source fields, future availability dates, and output securities
that were not present in the supplied point-in-time context.

`FactorDataBundle` separates signal-date rows from reusable histories. Formula code reads only the
declared tables, while the registry checks their required fields before calculation. Copy
`src/cb_quant/factors/_factor_template.py` for a new factor, implement `calculate_raw_factor`, and
leave the shared wrapper unchanged unless the factor has a genuinely different availability date.

## Review Checklist

- State the economic rationale before testing returns.
- Define the formula, units, direction, and missing-value behavior.
- Confirm every input is available by the signal-date close.
- Test the formula on a small hand-calculated fixture.
- Register the factor explicitly in the production catalog.
- Run the shared factor-contract and full test suites.
- Freeze parameters before evaluating the holdout period.
