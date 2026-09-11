# Research Decision Log

This document records choices that can affect the research conclusions. A parameter marked
**Pending** must not be presented as a frozen production rule.

## Decision Register

| ID | Topic | Frozen decision | Status |
|---|---|---|---|
| D000 | Research framework | Orient Securities daily-factor framework as the replication baseline; extensions labeled separately | Confirmed |
| D001 | Time partition | Replication through 2023-06-21, validation from 2023-06-22 to 2024-12-31, final holdout from 2025-01-01 | Confirmed |
| D002 | Signal schedule | Fixed 14-calendar-day anchors from 2018-01-10; holidays move to the next market-trading day without shifting future anchors | Confirmed |
| D003 | Signal timestamp | Signal-date close; execution no earlier than the next market-trading day | Confirmed |
| D004 | Execution price | Next-market-trading-day open; failed sells remain holdings and failed buys remain cash | Confirmed |
| D005 | Transaction cost | One-way 15 basis points on actual traded notional | Confirmed |
| D006 | Primary benchmark | Equal-weight contemporaneous investable universe; CSI Convertible Bond Index as market reference | Confirmed |
| D007 | Universe | Listing age, turnover, remaining balance, rating, and signal-date-trading filters | Confirmed |
| D008 | Industry mapping | Point-in-time CITIC level-1 membership; no future or static backfill | Confirmed |
| D009 | Outlier treatment | Median-absolute-deviation method and sensitivity range | Pending |
| D010 | Neutralization | Factor-specific industry, balance, rating, maturity, and parity controls | Pending |
| D011 | Portfolio size | Report TOP30 and type-balanced baselines followed by capacity sensitivity | Pending |
| D012 | Capacity | Historical average daily value and order-participation constraint | Pending |
| D013 | Cash flows and terminal events | Price return plus validated redemption or maturity settlement in phase one; complete total return in phase two | Partially confirmed |
| D014 | Same-parity relative premium | Fixed parity buckets and signal-date within-bucket median premium | Confirmed |
| D015 | Equity and linkage windows | Equity windows of 5/10/20 days; linkage correlation and beta windows of 20/60 days; 80% minimum valid observations | Confirmed as G3 baseline |
| D016 | Convertible-bond trading factors | Momentum, turnover, activity shock, Amihud, and close-to-VWAP definitions with an 80% minimum-valid rule | Confirmed as G3 baseline |
| D017 | CB-stock return spread | Simple CB close return minus adjusted-stock return over aligned 5/10/20 market-day windows | Confirmed as production baseline |
| D018 | Factor experiment records | Explicit batch configuration, deterministic run identities, immutable evidence packages, and like-for-like comparison groups | Confirmed |

## D000: Research Framework

The primary report provides a coherent structure spanning convertible-bond valuation,
underlying-equity information, stock-bond linkage, bond trading activity, factor combination,
and portfolio construction. Report-linked factors and project extensions are tagged separately.
An extension is retained only when it has an interpretable mechanism and incremental
out-of-sample evidence.

## D001: Chronological Partitions

- **Replication:** 2018-01-01 to 2023-06-21.
- **Extension validation:** 2023-06-22 to 2024-12-31.
- **Final holdout:** 2025-01-01 onward.
- **Current data boundary:** daily data through 2026-07-07.

The validation period may be used for a limited comparison of extensions and parameter grids.
The final holdout period is excluded from factor direction, feature selection, model selection,
and hyperparameter tuning.

## D002: Signal Schedule

The report specifies biweekly rebalancing but does not fully disclose the weekday anchor or
holiday convention. The production schedule starts on Wednesday, 10 January 2018, and advances
in fixed 14-calendar-day increments. If an anchor is not a market-trading day, the signal moves to
the next market-trading day while subsequent planned anchors remain unchanged.

This rule reproduces the reported sample endpoints but is documented as an implementation
inference rather than an explicit report statement. Robustness tests will include ten-market-day,
weekly, and month-end schedules.

## D003: Information Timestamp

The universe, factors, preprocessing, rankings, and target portfolio are formed after the signal-
date close using only information available by that time. Daily rolling windows include the
signal date. Events are usable only after their defined effective timestamp.

The earliest execution is the next market-trading day. This separation prevents the use of a
complete signal-day bar while simultaneously assuming an execution at that same close.

## D004: Execution and Rebalancing

- Positions are rebalanced at the next market-trading-day open.
- Securities retained in the target portfolio remain held and do not generate artificial round
  trips.
- Required sells are processed before new buys.
- A failed sell remains in the portfolio and is retried; proceeds are unavailable until sale.
- A failed buy leaves the corresponding capital in cash.
- A signal-date suspension prevents a security from entering the new candidate set.
- Execution-day full-session amount is not used to decide whether the opening trade was feasible.

Alternative next-open, opening-volume-weighted-average-price, and next-close executions will be
reported as sensitivity tests rather than mixed with the primary result.

## D005: Transaction Costs

The replication baseline applies a one-way cost of 15 basis points to actual traded notional:

```text
buy cash outflow = fill notional * (1 + 0.0015)
sell cash inflow = fill notional * (1 - 0.0015)
```

Retained positions and failed orders incur no transaction cost. The baseline interprets 15 basis
points as a combined commission and execution-friction assumption and does not add a second fixed
slippage term. Sensitivity tests will use 0, 5, 10, 15, 25, and 50 basis points.

## D006: Benchmark

The primary benchmark is an equal-weight portfolio of the same contemporaneous investable
universe, using the same signal dates, execution timing, transaction costs, and failed-order rules.
This isolates security selection from universe filtering and broad market appreciation.

The CSI Convertible Bond Price Index (`000832.CSI`) is a market-state reference rather than a
fully comparable portfolio. A total-return index may be added after coupon and terminal cash-flow
coverage is complete.

## D007: Investable-Universe Rules

### D007-1: Listing Age

Listing day is market-trading day one. A bond becomes eligible on market-trading day 11. Market
calendar age continues through individual-security suspensions. Sensitivity analysis will compare
5-, 10-, and 20-day exclusions.

### D007-2: Twenty-Day Cumulative Turnover

Daily turnover is:

```text
daily turnover = traded volume in lots * 1,000 / point-in-time remaining principal in CNY
```

The 20-market-day cumulative turnover includes the signal date and must not exceed 100%.
Suspended and zero-trade dates contribute zero turnover. Historical remaining balance becomes
effective on the first market-trading day after disclosure. Sensitivity thresholds are 50%, 100%,
150%, and 200%.

### D007-3: Remaining Balance

Point-in-time remaining principal must be at least CNY 200 million. Initial issue size is used
until the first disclosed remaining-balance update becomes effective. Current balance is never
backfilled into history. Sensitivity thresholds are CNY 100, 200, 300, and 500 million.

### D007-4: Rating

The target rule is issuer rating of A or above. The current implementation uses point-in-time
convertible-bond ratings as a labeled proxy because a complete historical issuer-rating series is
not yet available. A, A+, AA-, AA, AA+, and AAA are retained; A- and below are excluded. Rating
changes become effective on the first market-trading day after disclosure, and the lowest valid
rating is used when multiple ratings coexist.

## D008: Historical Industry Membership

The primary industry system is CITIC level 1. Signal-date membership is mapped from the exact
daily constituent snapshot. Future, current-static, forward, and backward membership fills are
forbidden. Ambiguous multiple-industry matches remain missing and retain an audit flag.

Industry-median imputation will be introduced only for factor families where the economic meaning
and leakage implications are explicitly justified. Prices, returns, execution prices, and event
states are never industry-median-imputed.

## D013: Cash Flows and Terminal Events

The phase-one backtest is defined as market-price return plus validated terminal cash settlement,
not complete total return. An implemented forced-redemption or maturity-redemption event may
create a receivable on the record date and available cash on the payment date. Non-executable
terminal exits without a validated event remain frozen at the last available mark and are reported
as unresolved exits.

Coupon, put, default, and exceptional-delisting cash flows require a separate event ledger before
the project can claim complete total-return performance.

## D014: Same-Parity Relative Premium

Signal-date investable bonds are assigned to conversion-value buckets:

```text
<70 | 70-90 | 90-100 | 100-110 | 110-130 | >=130
```

Relative premium equals the bond's conversion premium minus the signal-date median premium of its
bucket. A bucket must contain at least five bonds. The value score reverses the sign so that a
higher score represents a cheaper bond relative to same-parity peers.

This factor is a project extension, not an exact report definition. Across 44,728 investable
observations, absolute valuation coverage is 100% and relative-value coverage is 99.12%. Its
Spearman correlations with conversion premium and double-low are -0.312 and -0.397,
respectively, indicating material but non-redundant exposure.

## D015: Underlying-Equity and Stock-Bond Linkage Factors

- Equity returns use adjusted closes over 5, 10, and 20 market days.
- Twenty-day rolling statistics require at least 16 valid observations.
- Sixty-day linkage statistics require at least 48 paired stock-bond observations.
- Suspensions remain missing and are not converted to zero returns.
- Percent B uses a 20-day mean and two population standard deviations.
- Money Flow Index uses adjusted typical price and volume; an all-flat valid window is assigned 50.
- Correlation is the paired-observation Pearson return correlation.
- Beta is paired covariance of bond and stock returns divided by stock-return variance.

The G3a panel contains 16 factors across 44,728 observations. Coverage ranges from 95.65% to
99.92%, with zero key or formula mismatches. Correlation and beta are treated as state variables
until standalone and interaction effects are evaluated in G5.

## D016: Convertible-Bond Daily Trading Factors

- Raw-close returns use 5-, 10-, and 20-market-day windows; short-long momentum is the 5-day
  return minus the 20-day return.
- Daily turnover equals volume in lots times 1,000 divided by point-in-time remaining principal,
  but only when a source market row exists. An observed row with both `vol = 0` and `amount = 0`
  contributes zero turnover; a calendar bond-date with no source row remains missing.
- Abnormal turnover equals current turnover divided by the prior 20- or 60-day mean; the current
  day is excluded from the historical baseline. The 80% valid-observation rule counts genuinely
  observed rows, including observed zero-turnover rows, and excludes unknown market-data gaps.
- Tushare `amount` is reported in ten-thousand CNY and `vol` in hands of 10 bonds. Actual traded
  notional equals `amount * 10,000` CNY.
- The amount-shock factor is the current log actual amount standardized against the prior 20-day
  mean and sample standard deviation; its standardized values are invariant to this constant
  rescaling.
- Bond Amihud illiquidity is the 20-day mean of absolute daily return divided by actual amount in
  CNY, scaled by 100 million.
- Volume-weighted average price equals `amount * 10,000 / (volume * 10)`, equivalently
  `amount * 1,000 / volume`. Close-to-VWAP is retained at the current-day and 5-day-mean horizons.
- Every rolling window requires at least 80% valid observations.

A positive amount-volume pair is accepted only when its inferred VWAP lies within the reported
daily low-high range, with a CNY 0.02 tolerance. An invalid pair is excluded from turnover,
amount-shock, Amihud, and VWAP-dependent calculations for that date, while valid close-based
returns are preserved. The raw exception remains visible in the audit rather than being silently
removed.

An observed zero-volume row cannot produce VWAP. When both reported volume and amount are zero,
turnover is retained as an observed zero, while VWAP, log-amount shock, Amihud, and current or
rolling close-to-VWAP factors remain missing on that date. When the source market row itself is
absent, price, return, amount, turnover, VWAP, and all dependent factors remain missing. No
dedicated suspension status is inferred from an absent row.

The G3b panel contains 11 factors across 44,728 observations. Coverage ranges from 95.48% to
100.00%, with zero key or formula mismatches. The full-history VWAP unit audit passes on 99.995%
of 729,970 observations; one signal-date amount-volume pair is excluded by the quality gate.

## D017: Convertible-Bond versus Underlying-Stock Return Spread

The production baseline uses simple returns:

```text
spread(N) = (CB close[t] / CB close[t-N] - 1)
          - (adjusted stock close[t] / adjusted stock close[t-N] - 1)
```

The convertible bond uses raw close, while the underlying equity uses close multiplied by its
adjustment factor. Both legs share the same market-day endpoints, with the signal date included.
The 5-, 10-, and 20-day windows require at least 80% valid observations including valid start and
end prices. Insufficient history remains missing and is never filled. Information after the
signal-date close is excluded.

The registered direction is `-1`: a larger positive spread represents stronger convertible-bond
performance relative to its underlying equity and tests a relative-overreaction reversal
hypothesis. Delta- or beta-adjusted spreads are separate extensions and must be evaluated against
this simple baseline rather than replacing it retrospectively.

## D018: Factor Experiment Records

Production experiments are declared in an explicit factor-batch configuration. The batch file
owns factor names, factor-specific parameters, and the chronological sample partition. Shared
preprocessing, label, evaluation, portfolio, and execution assumptions remain in the backtest
configuration and cannot vary silently by factor.

Each run receives a deterministic identity derived from its factor version and parameters,
executable sample boundary, universe keys, shared research settings, and source-data
fingerprints. Existing run directories are immutable. Cross-factor comparisons require the same
comparison-group identity, which excludes the factor name and parameters but retains every shared
research and data assumption. Replication, validation, and final-holdout labels therefore refer
to enforced date ranges rather than descriptive folder names.

## Pending Decisions

The following parameters must be frozen before their respective production stages:

- median-absolute-deviation threshold and sensitivity range;
- factor-specific neutralization variables and estimation method;
- TOP-N portfolio size and bond-type balancing rule;
- capacity limit, historical average daily value window, and order participation rate;
- coupon and non-standard terminal-event treatment; and
- the exact boundary between linear composite models and nonlinear extensions.
