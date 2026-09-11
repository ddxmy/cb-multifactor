# Research Stage Status

The project advances through explicit research gates. A stage is complete only after its formulas,
information timing, keys, coverage, and materialized outputs pass automated and independent audit.
Later-stage returns cannot be used to revise earlier-stage definitions.

## Current Status

| Stage | Status | Scope |
|---|---|---|
| G0 | Complete | Research specification and engineering baseline |
| G1 | Complete | Point-in-time data and investable universe |
| G2 | Complete | Valuation coordinate system |
| G3a | Complete | Underlying-equity and stock-bond linkage factors |
| G3b | Complete | Convertible-bond daily trading factors |
| G3c | In Progress | Return-blind MAD gate complete; neutralization review pending |
| G4 | Replication Complete | Replication labels and execution inputs audited |
| G5 | Representative Evidence | Six pre-declared factors evaluated on 1/5/10-day horizons |
| G6 | Pending | Factor combination and model comparison |
| G7 | Representative Evidence | Three screened factors tested in the executable five-day portfolio |
| G8 | Release Candidate | Public replication checkpoint prepared; validation and holdout remain locked |

Factor predictive power and strategy performance have not yet passed the research gates. No
completed-stage diagnostic should be interpreted as evidence of future return or implementable
portfolio performance.

## Completed Gates

### G0: Research Specification

- Frozen sample period, biweekly signal schedule, execution convention, transaction-cost
  assumption, chronological partitions, and final holdout boundary.
- Machine-readable configuration validation and explicit tracking of unresolved decisions.

Primary records: `config/research_v1.json` and
`artifacts/G0/research_config_validation_v1.json`.

### G1: Point-in-Time Universe

- Historical conversion-price, balance, and rating events become effective on the first market
  trading day after disclosure.
- Full panel audit: 44,728 investable rows across 222 signal dates and 846 distinct bonds.
- Duplicate keys, eligibility formula mismatches, event-boundary failures, and industry-date
  mismatches are all zero.

Primary records: `artifacts/G1/universe_audit_v1.json` and
`notebooks/01_point_in_time_universe.ipynb`.

### G2: Valuation Factors

- Conversion value, conversion premium, double-low, and same-parity relative value use only the
  contemporaneous point-in-time cross-section.
- Absolute-valuation coverage is 100.00%; same-parity relative-value coverage is 99.12%.
- Eligible-key and formula mismatches are zero.

Primary records: `artifacts/G2/valuation_factor_audit_v1.json` and
`notebooks/02_valuation_factors.ipynb`.

### G3a: Equity and Stock-Bond Linkage

- Sixteen daily factors cover equity return, volatility, technical state, liquidity, return
  spread, rolling correlation, and rolling beta.
- Factor coverage ranges from 95.65% to 99.92%; eligible-key and formula mismatches are zero.
- Rolling statistics require at least 80% valid paired observations and use signal-date or earlier
  information only.

Primary records: `artifacts/G3/stock_linkage_factor_audit_v1.json` and
`notebooks/03_equity_and_linkage_factors.ipynb`.

### G3b: Convertible-Bond Trading Factors

- Eleven daily factors cover return, momentum, turnover, amount surprise, Amihud illiquidity, and
  close-to-Volume-Weighted Average Price deviation.
- Factor coverage ranges from 95.48% to 100.00% across the same 44,728 rows, 222 signal dates, and
  846 bonds.
- One signal-date VWAP observation is excluded because its amount-volume pair is inconsistent with
  the daily price range. Formula mismatches, missing expected keys, extra keys, and duplicate keys
  are zero.

Primary records: `artifacts/G3/cb_trading_factor_audit_v1.json` and
`notebooks/04_cb_trading_factors.ipynb`.

## Current Gate: G3c

G3c freezes factor preprocessing before any return label is introduced. G3c-A is complete:

- G2/G3 panels reconcile to 44,728 rows, 222 signal dates, 846 bonds, and 31 factors;
- date-local scaled-MAD winsorization uses 3.0 as the primary multiplier;
- 2.5/3.0/3.5 sensitivity is evaluated without any forward-return columns;
- raw values remain available beside auditable `winsorized__*` columns; and
- source partitions and configuration are content-fingerprinted.

G3c-B remains in progress. Its acceptance scope is:

- valuation-family preprocessing is complete: conversion premium and double-low preserve their
  original economic structure, while same-parity relative valuation is neutralized against log
  remaining balance, rating, and remaining maturity before standardization;
- same-parity control exposure falls from approximately 0.122 to numerical zero while retaining
  approximately 0.912 cross-sectional rank correlation with the winsorized input;
- underlying-equity preprocessing is complete: all nine factors are neutralized against
  point-in-time log stock total market capitalization and CITIC Level-1 industry;
- stock market capitalization covers 99.97% of rows and joint size/industry coverage is 99.30%;
- equity-factor control exposure falls to numerical zero; processed rank correlation is
  approximately 0.76-0.81 for eight factors and 0.377 for stock Amihud, identifying material
  size and industry content in the raw illiquidity signal;
- stock-bond linkage and convertible-bond trading families remain pending; and

- cross-sectional standardization after the neutralization decision;
- factor-family-specific neutralization controls;
- an auditable factor catalog covering formula, direction, window, input, and timestamp; and
- a merged valuation, equity, linkage, and bond-trading panel.

The factor catalog must match the materialized columns exactly. Missing-value treatment,
outlier handling, and neutralization may use only the current or earlier cross-section.

## G4 Replication Checkpoint

The replication partition now has a materialized close-signal to next-open execution schedule,
complete daily market and lifecycle grids, and bond-locked O2O labels. Of 23,877 signal rows,
23,171 have usable execution-aligned labels. Another 647 rows cross the 21 June 2023 partition
boundary and are explicitly excluded; the remaining exclusions are missing or non-positive source
opens. Market and lifecycle keys reconcile one-to-one across 1,019,904 rows.

The same replication inputs now also support frozen 1-, 5-, and 10-market-day prediction labels
without changing the biweekly signal calendar. The long-form panel contains 71,631 rows from
23,877 signal-bond observations. Usable labels total 23,537, 23,518, and 23,172 for the respective
horizons; all boundary, missing-open, and invalid-open observations retain explicit exclusion
reasons. The corresponding evidence is stored in
`artifacts/G4/replication_2018_2023/g4_horizon_label_audit_v1.json`.

This checkpoint does not evaluate factor performance. The validation partition remains unopened
for G5 model selection, and final-holdout materialization requires an explicit release flag.

## Representative G5 and G7 Evidence

Six factors have been evaluated on the replication partition with factor directions fixed before
labels were joined. Double-low, same-parity relative premium, and underlying-stock 20-day momentum
pass the medium-horizon Rank-IC screen. The abnormal-turnover reversal hypothesis is rejected in
its pre-fixed direction rather than being sign-flipped after observation.

The three screened factors were then tested in the five-market-day executable portfolio. Only
double-low remains positive after constraints and 15-basis-point one-way costs, with a 1.84%
annualized return, 0.27 Sharpe ratio, and -14.00% maximum drawdown. Full definitions and results are
reported in `docs/methodology/empirical_results.md`.

## Remaining Evaluation Gates

- **G3c:** freeze neutralization policies for stock-bond linkage and bond-trading families.
- **G5:** expand beyond the representative factor set and add market-state stability diagnostics.
- **G6:** compare transparent linear composites before considering regularized alternatives.
- **G7:** test turnover controls and capacity assumptions on the selected composite.
- **G8:** open validation only under a separately approved protocol; keep the final holdout locked.

Only results explicitly labeled as replication evidence may be presented before those remaining
gates are completed.

## Validated Framework Infrastructure

The reusable G4-G7 infrastructure is implemented and unit-tested before the empirical stage gates:

- immutable factor metadata, point-in-time factor validation, and a plugin registry;
- explicit one-module factor plugins, validated shared history bundles, and a production catalog;
- configuration-driven factor batches with shared input loading and pre-run catalog validation;
- immutable factor-run evidence packages and a comparison-group-constrained cross-factor index;
- close-signal to next-open execution schedules and locked-bond O2O labels;
- date-local preprocessing, multi-horizon IC/Rank IC/ICIR analysis, primary top-quintile active
  IR, auxiliary G1-minus-G5 IR, and quantile diagnostics;
- long-only target construction with cash reserve and weight constraints;
- sell-before-buy execution, board-lot rounding, failed-order state, costs, and daily marking;
- forced-redemption and maturity receivables with explicit unresolved-state failures; and
- independent cash-ledger reconciliation plus continuation, entry, exit, settlement, and cost P&L
  attribution.

This checkpoint advances representative G5 and G7 evidence only. Completing the remaining gates
requires frozen preprocessing for every factor family, broader sensitivity analysis, composite-
model evaluation, validation data, and an independent leakage and execution audit.
