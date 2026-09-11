# Replication-Sample Factor Evidence

## Scope and Interpretation

This checkpoint evaluates six representative daily factors from 11 January 2018 through
21 June 2023. Factor definitions and directions were fixed before forward returns were joined.
Signals are sampled every 14 calendar days, executed at the next market open, and evaluated
against 1-, 5-, and 10-market-day open-to-open returns.

G5 results are analytical diagnostics without transaction costs, cash constraints, or overlapping
position accounting. G7 results are separate executable simulations with CNY 1 million initial
capital, a 10% cash reserve, 10-bond board lots, a 10% single-name cap, sell-before-buy execution,
failed-order handling, and 15 basis points of one-way cost.

## G5: Cross-Sectional Ranking

![Replication-sample Rank IC](../assets/g5_rank_ic_comparison.png)

| Factor | Pre-fixed orientation | 5-day Rank IC | 5-day Rank ICIR | 10-day Rank IC | 10-day Rank ICIR | Evidence |
|---|---|---:|---:|---:|---:|---|
| Conversion premium | Lower is better | 2.55% | 0.60 | 0.25% | 0.06 | Weak and horizon-unstable |
| Double-low | Lower is better | 5.89% | 1.76 | 4.02% | 1.14 | Pass |
| Same-parity relative premium | Lower is better | 4.62% | 2.43 | 3.51% | 1.66 | Pass |
| Underlying-stock 20-day return | Higher is better | 3.35% | 1.72 | 3.96% | 2.10 | Pass |
| CB 20-day return | Lower is better | 0.92% | 0.24 | 1.23% | 0.33 | Weak |
| Abnormal turnover | Lower is better | -4.17% | -1.86 | -4.16% | -2.09 | Rejected in the pre-fixed direction |

The checkpoint reporting screen requires a mean Rank IC above 2% and a Newey-West p-value below
5% at either the 5- or 10-day horizon. This screen summarizes the replication evidence and is not
presented as a preregistered selection rule. Double-low, same-parity relative premium, and underlying-
stock momentum pass. Conversion premium and bond-return reversal lack stable ranking evidence.
Abnormal turnover is statistically strong in the opposite direction to the crowding-reversal
hypothesis; its sign is not flipped after observing the result.

The complete machine-readable summary is available in
[`g5_factor_summary.csv`](../results/g5_factor_summary.csv).

## G7: Five-Day Executable Portfolios

![Executable five-day portfolio comparison](../assets/g7_portfolio_comparison.png)

| Factor | Annualized return | Sharpe ratio | Maximum drawdown | Average cash | Annualized turnover |
|---|---:|---:|---:|---:|---:|
| Double-low | 1.84% | 0.27 | -14.00% | 52.62% | 46.46x |
| Same-parity relative premium | -1.58% | -0.21 | -19.08% | 52.79% | 46.91x |
| Underlying-stock momentum | -0.52% | -0.02 | -15.90% | 54.75% | 44.89x |

Only double-low remains positive after executable constraints and costs, and its risk-adjusted
return is modest. The simulation accumulates transaction costs equal to 37.37% of initial capital
for double-low over the full sample. This is not a one-year fee rate; it is the sum of all charged
costs relative to initial capital. The large gap between G5 Rank IC and G7 performance shows that
cross-sectional predictability alone is insufficient when a short holding period creates high
turnover and overlapping cohorts leave roughly half of capital in cash.

The complete machine-readable summary is available in
[`g7_portfolio_summary.csv`](../results/g7_portfolio_summary.csv).

## Research Decision

Double-low is retained as the primary transparent baseline. Same-parity relative valuation and
underlying-stock momentum remain candidate components for a later low-turnover composite, but the
current standalone five-day implementations are not promoted as tradable strategies. Weak and
negative findings remain in the record to prevent retrospective factor selection.

Validation data from 22 June 2023 through 31 December 2024 and the final holdout beginning on
1 January 2025 remain unopened in this checkpoint.
