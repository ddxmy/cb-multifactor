# Reference Report Factor Map

## 1. Reference and Classification

The primary reference is Orient Securities, *An Initial Exploration of a Convertible-Bond
Multi-Factor Model*, Macro Fixed-Income Quantitative Research Series No. 10, published on
1 July 2023. A public report index is available
[here](https://www.fxbaogao.com/detail/3773206).

This document separates three implementation classes:

- **Report baseline:** a factor family or portfolio method explicitly used in the report.
- **Daily replication:** a point-in-time implementation supported by the current local data.
- **Project extension:** a modified or additional construction motivated by the same economic
  mechanism but not presented as an exact report result.

Where the report does not disclose a complete mathematical specification, this project records
the chosen implementation as a variant rather than claiming exact replication.

## 2. Report Architecture

```text
Convertible-bond valuation
          +
Underlying-equity price and volume / stock-bond linkage
          +
Convertible-bond price and volume
          |
          v
Single-factor evaluation -> within-group composite -> cross-group composite
          |
          v
TOP-N and bond-type-balanced portfolios
```

The report is a cross-sectional security-selection study. It does not reduce the problem to a
directional forecast for either the equity market or the bond market.

## 3. Factor Mapping

### 3.1 Convertible-Bond Valuation

| Report factor family | Economic interpretation | Project status | Project implementation or extension |
|---|---|---|---|
| Conversion premium | Price paid above immediate conversion value | Implemented | Absolute premium plus same-parity relative premium |
| Double-low | Joint preference for low price and low premium | Implemented | Report baseline; parity-adjusted residual remains planned |
| Six-month premium z-score | Current premium relative to the bond's own history | Planned | 120-market-day time-series standardization with minimum-history control |
| Implied volatility | Volatility embedded in the convertible-bond option value | Deferred | Requires point-in-time rates, dividends, cash flows, and validated terms |
| Implied-minus-realized volatility | Relative option valuation | Deferred | Implemented only after the implied-volatility engine is validated |
| Bond-floor premium | Price relative to straight-bond value | Planned | Rating- and maturity-matched historical discount curve |

Implemented valuation identities:

```text
conversion value   = stock close / effective conversion price * 100
conversion premium = convertible-bond close / conversion value - 1
double-low          = convertible-bond close + 100 * conversion premium
```

The same-parity relative-value factor is a project extension. It subtracts the signal-date median
premium of bonds in the same parity bucket and reverses the sign so that a larger score represents
a cheaper bond relative to comparable parity.

### 3.2 Underlying-Equity Price and Volume

| Report factor family | Report role | Project status | Production treatment |
|---|---|---|---|
| Equity return and historical z-score | Trend and delayed information transmission | Partially implemented | 5-, 10-, and 20-day adjusted-close returns; historical z-score planned |
| Percent B | Position within a rolling Bollinger band | Implemented variant | 20-day window and two population standard deviations |
| Relative Strength Index | Directional price strength | Implemented variant | 20-day adjusted-price-change ratio |
| Price-to-high | Current price relative to recent high | Implemented variant | 20-day adjusted-close high |
| Amihud illiquidity | Price impact per unit of trading amount | Implemented variant | 20-day mean absolute return divided by amount in CNY |
| Money Flow Index | Volume-confirmed price direction | Implemented variant | 20-day adjusted typical-price money flow |

The report uses longer windows for several equity indicators. Current 20-day definitions are
explicit variants and will be compared with report-aligned windows before model selection.

### 3.3 Stock-Bond Linkage

| Report factor family | Economic interpretation | Project status | Production treatment |
|---|---|---|---|
| Bond-minus-stock return spread | Delayed or excessive bond response to stock information | Implemented | 5-, 10-, and 20-day spread |
| Rolling stock-bond correlation | Strength and stability of equity transmission | Implemented | 20- and 60-day paired-observation correlation |
| Stock-bond beta | Magnitude of bond sensitivity to stock returns | Extension implemented | 20- and 60-day covariance divided by stock variance |
| Residual bond return | Bond-specific return after equity exposure | Planned extension | Rolling-beta residual and parity-conditioned analysis |
| Conversion-premium change | Valuation component of the stock-bond return gap | Planned extension | Return-spread decomposition |

Correlation and beta use only dates with valid returns for both securities. Suspended observations
are missing, not zero.

### 3.4 Convertible-Bond Daily Trading Factors

| Report factor family | Economic interpretation | Project status | Production implementation |
|---|---|---|---|
| Bond momentum and reversal | Continuation or correction of bond-specific price pressure | Implemented variant | 5-, 10-, and 20-day raw-close returns plus 5-minus-20-day momentum |
| Bond turnover | Activity, crowding, and speculative demand | Implemented variant | Current turnover plus current-to-prior-mean ratios over 20 and 60 days |
| Bond liquidity / Amihud | Bond-market price impact | Implemented variant | 20-day mean absolute return divided by amount in CNY |
| Trading-amount shock | Abrupt participation by market capital | Project extension implemented | Current log amount standardized against the prior 20-day history |
| Close-to-VWAP deviation | Late-session pricing pressure | Project extension implemented | Close divided by volume-weighted average price minus one, current and 5-day mean |

All rolling definitions require at least 80% valid observations. Tushare `amount` is reported in
ten-thousand CNY and `vol` in hands of 10 bonds: actual notional is `amount * 10,000` CNY, while
VWAP per bond is `amount * 1,000 / vol`. Amihud and log-amount calculations use actual notional.
Amount and volume are converted to a valid volume-weighted average price before dependent factors
are calculated; internally inconsistent pairs are excluded rather than silently propagated.
An observed source row with `vol = 0` and `amount = 0` contributes zero turnover but has no valid
VWAP or amount-dependent factor on that date. A calendar bond-date with no source row remains
missing and is excluded from rolling turnover means and valid-observation counts rather than being
classified as a suspension or zero trading.

### 3.5 Intraday Factors

The report includes intraday price-volume factors based on five-minute and minute-level data,
including intraday Relative Strength Index, moderate-return averages, intraday return variance,
price-volume correlation volatility, volume-change skewness, and opening-volume share.

The primary project is deliberately restricted to daily-frequency factors. Intraday data are
retained for a separate extension and will not be combined with the daily results without an
independent specification and validation process.

## 4. Portfolio and Evaluation Mapping

| Research component | Report framework | Project implementation |
|---|---|---|
| Universe | Excludes new, high-turnover, small-balance, and low-rated bonds | Point-in-time daily implementation with a complete exclusion funnel |
| Single-factor evaluation | Information coefficient and grouped returns | Rank Information Coefficient, Information Coefficient Information Ratio, hit rate, monotonicity, decay, and stability |
| Preprocessing | Outlier treatment and cross-sectional standardization | Explicit median-absolute-deviation winsorization, standardization, and neutralization parameters |
| Composite construction | Within-group and cross-group equal weighting; orthogonalized variant | Equal-weight replication plus rolling information-coefficient and regularized-linear alternatives |
| Portfolio construction | TOP-N and bond-type-balanced portfolios | Next-open execution with costs, failed orders, cash, turnover, and capacity |

The report selected a broad set of in-sample factors and evaluated equal-weight and orthogonalized
composites. This project retains those approaches as baselines but requires chronological
validation and a final holdout period before reporting incremental performance.

## 5. Replication Standard

Every report-linked factor must record:

- the evidence that the factor family appears in the reference report;
- the exact local formula, window, required fields, and information timestamp;
- whether the implementation is an exact baseline, a report variant, or an extension;
- differences caused by unavailable point-in-time inputs; and
- standalone and incremental out-of-sample evidence on identical samples.

Negative or non-incremental findings remain part of the final research record.
