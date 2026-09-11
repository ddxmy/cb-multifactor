# Factor Research Roadmap

## 1. Scope

This project uses the daily-frequency portion of the Orient Securities convertible-bond
multi-factor framework as a replication baseline. Extensions are evaluated only when they have:

1. a clear economic interpretation;
2. a point-in-time implementation using the available local data;
3. standalone cross-sectional evidence;
4. incremental out-of-sample information beyond the corresponding baseline; and
5. positive contribution after turnover and transaction costs.

The primary research is daily-frequency. Intraday data are reserved for a separate extension and
are not mixed into the main model.

## 2. Data Availability and Boundaries

| Dataset | Coverage and fields | Supported research | Known boundary |
|---|---|---|---|
| Convertible-bond daily market data | 2018-01-02 to 2026-07-07; open, high, low, close, volume, amount | Returns, liquidity, turnover, and execution prices | Tushare `amount` is in ten-thousand CNY and `vol` is in hands of 10 bonds; actual notional is `amount * 10,000` CNY |
| A-share daily DuckDB | Open, high, low, close, volume, amount, turnover, and adjustment factor | Momentum, volatility, RSI, Percent B, price-to-high, Amihud, MFI, and stock-bond linkage | Raw prices are used for conversion value; adjusted prices are used for continuous-return factors |
| CITIC industry snapshots | Daily level-1 and level-2 constituents from 2010 | Historical industry mapping, industry adjustment, and neutralization | Missing constituent snapshots are not forward- or backward-filled |
| Historical bond ratings | Disclosure date, rating date, and long-term rating | Point-in-time universe filters and rating-change extensions | The current field is a bond-rating proxy, not a complete issuer-rating history |
| Conversion and balance events | Disclosure date, period end, remaining balance, and conversion price | Historical parity, premium, turnover denominator, and capacity | Events become usable on the next market-trading day after disclosure |
| Redemption events | Announcement, record, payment date, and redemption price | Terminal states and redemption-risk extensions | Coupon, put, default, and exceptional delisting cash flows remain phase-two work |
| Convertible-bond intraday data | Multiple bar frequencies from 2018 | Independent intraday extension | Excluded from the primary daily model |

The first production model excludes implied volatility, risk-free and credit curves, complete
coupon total return, and clause-level call-probability forecasts. These components require
additional point-in-time inputs and will not be approximated with current static data.

## 3. Universe and Return Label

- **Mother universe:** ordinary convertible bonds in the local terms database intersected with
  observed daily market codes; 940 bonds before signal-date filtering.
- **Signal timestamp:** after the signal-date close, using information available by that time.
- **Primary label:** next execution open to the following execution open.
- **Investability filters:** listing age of at least 11 market-trading days, 20-day cumulative
  turnover no greater than 100%, remaining balance of at least CNY 200 million, rating proxy of A
  or above, and a valid signal-date market observation.

## 4. Factor Taxonomy

### 4.1 Convertible-Bond Valuation

The valuation coordinate system begins with:

```text
conversion value   = stock close / effective conversion price * 100
conversion premium = CB close / conversion value - 1
```

| Factor | Baseline | Extension | Economic rationale |
|---|---|---|---|
| Conversion premium | Absolute signal-date premium | Relative premium within a parity bucket | Controls the structural relationship between parity and premium |
| Double-low | Price plus 100 times conversion premium | Parity-, maturity-, and credit-adjusted residual | Separates valuation from systematic bond-state exposure |
| Historical valuation position | Rolling premium z-score or percentile | Joint time-series and cross-sectional relative value | Distinguishes cheap versus peers from cheap versus own history |
| Bond-floor premium | Bond price relative to estimated straight-bond value | Rating- and maturity-matched term-structure estimate | Measures the price paid above credit-adjusted bond value |
| Implied volatility | Option-implied underlying volatility | Implied-minus-realized volatility and historical z-score | Deferred until point-in-time curves and cash flows are available |

### 4.2 Underlying-Equity Information

| Factor | Production definition | Economic rationale |
|---|---|---|
| Equity momentum | 5-, 10-, and 20-day adjusted-close return | Measures recent information and trend in the underlying stock |
| Realized volatility | 20-day standard deviation of daily return, annualized | Captures underlying risk and embedded-option relevance |
| Relative Strength Index | Positive adjusted price changes divided by total absolute changes | Measures directional price strength |
| Price-to-high | Adjusted close divided by the 20-day adjusted-close maximum | Measures proximity to a recent high |
| Percent B | Position within a 20-day Bollinger band using two population standard deviations | Measures price position after scaling by recent volatility |
| Amihud illiquidity | Mean absolute daily return divided by trading amount in CNY | Estimates price impact per unit of traded capital |
| Money Flow Index | Positive typical-price money flow divided by total directional money flow | Tests whether price direction is confirmed by volume |

Relative Strength Index (RSI), Money Flow Index (MFI), and Percent B are defined in full at first
use in public documentation; code columns retain concise, stable names.

### 4.3 Stock-Bond Linkage

| Factor | Production definition | Research role |
|---|---|---|
| Return spread | Convertible-bond return minus adjusted stock return over 5, 10, and 20 days | Identifies delayed or excessive bond-price response |
| Return correlation | 20- and 60-day paired-observation Pearson correlation | Describes the stability of stock-to-bond transmission |
| Equity beta | Rolling covariance of bond and stock returns divided by stock-return variance | Measures bond-price sensitivity to the underlying stock |
| Residual bond return | Bond return after removing estimated stock exposure | Planned extension to isolate bond-specific repricing |
| Premium change | Change in conversion premium over the linkage window | Planned decomposition of return spread into parity and valuation components |

Correlation and beta are treated primarily as state variables and interaction terms. They are not
assumed to be standalone alpha signals.

### 4.4 Convertible-Bond Daily Trading Activity

| Factor | Production definition | Economic rationale |
|---|---|---|
| Bond momentum and reversal | 5-, 10-, and 20-day raw-close returns; 5-day minus 20-day return | Tests continuation versus sentiment reversal |
| Turnover | Volume times 1,000 divided by point-in-time remaining principal | Scales trading activity by the outstanding bond balance |
| Abnormal turnover | Current turnover divided by the prior 20- or 60-day mean | Measures unusual trading intensity and crowding |
| Abnormal trading amount | Current log actual amount in CNY standardized against the prior 20-day mean and sample standard deviation | Detects abrupt capital participation |
| Bond Amihud illiquidity | 20-day mean absolute bond return divided by actual amount in CNY, scaled by 100 million | Estimates bond-market price impact |
| Close-to-VWAP deviation | Close divided by volume-weighted average price minus one; current and 5-day mean | Measures late-session pricing pressure |

Tushare `amount` is in ten-thousand CNY and `vol` is in hands of 10 bonds. Actual traded notional
is therefore `amount * 10,000` CNY, while VWAP per bond is `amount * 10,000 / (vol * 10)`,
equivalently `amount * 1,000 / vol`. Amihud and log-amount calculations use actual traded
notional; the latter's standardized values are invariant to the constant rescaling. A positive
amount-volume pair is accepted only when the inferred VWAP lies within the daily low-high range,
with a CNY 0.02 tolerance. A source row with both volume and amount equal to zero contributes zero
turnover but leaves VWAP, log amount, Amihud, and close-to-VWAP factors missing on that date. A
calendar bond-date without a source market row remains missing for all market fields and factors;
it is not treated as a suspension or a zero-turnover observation. Rolling abnormal turnover counts
observed zero-turnover rows but excludes unknown gaps under the 80% valid-observation rule. Across
729,970 full-history observations, 99.995%
pass this unit check. The G3b panel contains 11 factors across 44,728 investable observations;
coverage ranges from 95.48% to 100.00%, with zero key or formula mismatches.

## 5. Market-State Variables

Signal-date market states will be built from contemporaneously observable cross-sections:

- median conversion premium;
- median short-maturity conversion premium;
- aggregate trading amount and turnover;
- share of high-price convertible bonds;
- underlying-equity and industry volatility; and
- distribution of parity and stock-bond beta.

Market-state thresholds will be estimated on the replication or validation period only. The final
holdout period will not be used to define regimes.

## 6. Validation Sequence

1. Materialize point-in-time raw factors and retain field-level availability timestamps.
2. Audit keys, formulas, missingness, extreme values, and cross-factor correlation.
3. Apply pre-specified winsorization, standardization, and neutralization.
4. Evaluate Rank Information Coefficient, Information Coefficient Information Ratio, hit rate,
   quantile monotonicity, holding-period decay, and annual stability.
5. Compare each extension against its report-replication baseline on identical samples.
6. Build equal-weight and rolling information-coefficient composites.
7. Evaluate Ridge and Elastic Net models under expanding-window estimation.
8. Test nonlinear models only for incremental out-of-sample value.
9. Run portfolio simulation with next-open execution, one-way 15-basis-point cost, failed-order
   handling, turnover, capacity, and terminal-event states.

## 7. Acceptance Standard

A factor is eligible for the production model only if:

- its formula and information timing pass independent recomputation;
- it has sufficient and stable cross-sectional coverage;
- its direction is determined without using the final holdout period;
- it provides interpretable incremental information after controlling for correlated factors;
- its performance is not concentrated in a small number of dates or securities; and
- the effect remains economically relevant after realistic turnover and cost assumptions.
