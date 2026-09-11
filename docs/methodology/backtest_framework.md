# Convertible-Bond Factor Backtest Framework

## Purpose

The framework separates factor evidence from implementable portfolio performance. A factor can
show cross-sectional predictive content without implying that the same return can be captured
after cash, trading, lifecycle, and capacity constraints. Both layers therefore consume the same
point-in-time factor panel but publish separate outputs.

## Information and Execution Timing

The signal is calculated from information available by the close of signal date `t`. Portfolio
orders are first attempted at the next market-trading-day open. The primary factor label locks the
same bond from that entry open to the following rebalance execution open. Missing execution prices
are never forward-filled into assumed trades.

This convention is written compactly as `C -> O2O`:

- `C`: factor calculation at the signal-date close;
- first `O`: execution at the strict next trading-day open; and
- second `O`: exit at the next rebalance execution open.

The portfolio is marked at each daily close so drawdown, cash utilization, and blocked positions
remain observable between rebalances.

### Multi-Horizon Factor Labels

The frozen signal calendar remains biweekly. For factor-evidence diagnostics only, each signal is
also paired with 1-, 5-, and 10-market-day forward returns. All three labels enter at the strict
next market open and lock the same bond until the open exactly `h` market days later:

`forward_return_h = exit_open_h / entry_open - 1`

These are prediction horizons, not three alternative rebalance schedules. Labels that cross the
sample boundary or fail the explicit entry/exit execution checks remain in the audit panel with a
reason code but are excluded from factor statistics.

## Factor Evidence Layer

Each factor plugin returns `signal_date`, `ts_code`, `raw_factor`, and `available_date`. The registry
checks unique keys, finite numeric values, required source fields, point-in-time availability, and
that the calculator cannot create securities outside its input context.

Preprocessing is applied independently within each signal-date cross-section:

1. median-absolute-deviation winsorization;
2. optional regression neutralization; and
3. cross-sectional z-score standardization.

The evidence sample is limited to bonds eligible for entry and not subject to a mandatory exit on
the signal date. Evaluation outputs include Pearson IC, Rank IC, Newey-West-adjusted IC summary,
equal-count quantile returns, quantile NAV, and an analytical long-short spread. The spread is a
research diagnostic rather than an assumption that ordinary convertible bonds can be shorted.

Pearson Information Coefficient (IC) measures linear cross-sectional association, while Spearman
Rank IC measures monotonic ordering. Both are calculated independently for the 1-, 5-, and
10-market-day labels. Because observations follow the biweekly signal calendar, annualized ICIR
uses `sqrt(26)` rather than a daily-frequency scaling. Newey-West inference is reported to reduce
the impact of serial correlation and heteroskedasticity in the signal-date series.

The primary Information Ratio (IR) uses the equal-weight top factor quintile's active return over
the contemporaneous eligible-universe equal-weight return. The raw `G1 - G5` spread IR is retained
as an auxiliary ranking diagnostic; it is not treated as an executable short portfolio.

## Long-Only Portfolio Layer

Processed factor scores are converted into deterministic long-only target weights. The portfolio
supports a cash reserve, a maximum single-name weight, a fixed number of selected bonds, and
board-lot rounding. Targets and actual holdings remain separate throughout the simulation.

At each execution open, mandatory and target-reduction sells are processed before buys. A failed
sell remains an actual position and its expected proceeds are unavailable. A failed buy leaves cash
idle and is not retried automatically. Costs are charged only on filled notional.

Production runs require explicit `buy_allowed` and `sell_allowed` flags plus complete daily
lifecycle coverage. Missing coverage fails the run instead of defaulting to a tradable state. A
permissive lifecycle mode exists only for compact synthetic tests and must be requested explicitly.

## Convertible-Bond Lifecycle Treatment

Lifecycle states distinguish pre-listing, seasoning, active trading, redemption watch, announced
redemption, the final trading window, suspended or unquoted status, settlement receivables, and
settled or unresolved terminal states.

Static delisting dates without an information-availability timestamp do not create historical exit
signals. Point-in-time forced-redemption announcements block new entries and require an exit. If a
market sale fails but validated registration, payment-date, and call-price terms are available, the
holding becomes a receivable and later cash. Missing terminal terms raise an auditable error instead
of silently carrying a stale mark.

When only a payment date is known, settlement proceeds remain a receivable through that date and
become deployable at the following market open. This avoids assuming an unsupported pre-open cash
arrival time.

## Accounting and Return Attribution

Cash is reconciled independently from filled-trade cash flows and settlement receipts. Daily NAV is
also decomposed into:

- continuation P&L from the quantity held across the close;
- entry P&L from the execution price to the current close;
- exit P&L from the previous close to the execution price;
- forced-redemption or maturity settlement P&L; and
- transaction costs.

The attribution residual must reconcile to the daily NAV change. This mirrors the explanatory role
of continuation, entry, and exit attribution in a commodity cross-sectional framework while
replacing futures roll logic with convertible-bond lifecycle transitions.

## Configuration Status

`config/backtest_v1.json` records both frozen behavior and parameters awaiting a research-stage
decision. Signal timing, next-open execution, O2O labels, board-lot handling, sell-before-buy, and
failed-order policies are frozen. The 1-, 5-, and 10-market-day prediction horizons are also
frozen. Winsorization strength, neutralizers, quantile count, portfolio size, cash reserve, and
weight cap remain explicit defaults until sensitivity tests pass the appropriate stage gate.

## Reproducible Outputs

One factor run emits raw and processed factors, execution-aligned labels, excluded-label reasons,
IC evidence, quantile results, target weights, orders, fills, positions, receivables, NAV, P&L
attribution, and a run audit. Source datasets remain read-only, and the command-line runner refuses
to overwrite an existing output unless explicitly instructed.
