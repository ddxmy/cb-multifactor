# Data Contract

This repository publishes no raw market data, commercial databases, credentials, or derived
research panels. The pipeline resolves approved local sources at runtime and materializes only
ignored local artifacts.

## Convertible-Bond Daily Market Data

Each row must be unique on `ts_code` and `trade_date`. Required fields and units are:

| Field | Unit | Use |
|---|---|---|
| `ts_code` | source security identifier | Convertible-bond key |
| `trade_date` | market date | Point-in-time observation date |
| `pre_close`, `open`, `high`, `low`, `close` | CNY per bond face value of CNY 100 | Returns, execution, and valuation |
| `pct_chg` | percent, not decimal | Daily return when supplied |
| `vol` | lots of 10 bonds | Turnover and VWAP denominator |
| `amount` | ten-thousand CNY | Raw Tushare traded amount |

Actual traded notional is `amount * 10,000` CNY. `vol` is reported in hands of 10 bonds, so
`amount * 10,000 / (vol * 10)`, equivalently `amount * 1,000 / vol`, is the inferred CNY VWAP per
bond. Amihud and every other calculation documented as an amount in CNY use actual traded notional,
not the VWAP per-hand scale. The log-amount z-score also uses actual traded notional; its
standardized values are invariant to this constant rescaling. Amount-volume pairs are eligible for
dependent factors only when the inferred VWAP lies within the reported daily high-low range,
allowing a CNY 0.02 tolerance.

## Point-in-Time Terms and Event Data

The convertible-bond reference universe requires `ts_code`, `list_date`, and `issue_size`
(CNY 100 million). Daily universe construction uses point-in-time terms rather than current
static fields.

Rating history requires `ts_code`, `ann_date`, and `rating`; ratings are normalized to the
long-term rating scale. Remaining-principal and conversion-price history requires `ts_code`,
`publish_date`, `end_date`, `remain_size` (CNY 100 million), and `convert_price` (CNY per share).
Ratings, remaining-principal changes, and conversion-price events first become usable on the
strictly next market trading day after disclosure. No later event or current static value may be
backfilled into an earlier snapshot.

## A-Share and Industry Inputs

### A-share Daily Data

Each row must be unique on `ts_code` and `trade_date`. Required fields are `close`, `high`,
`low`, `pre_close`, `pct_chg` (percent), `vol`, `amount` (thousand CNY), and `adj_factor`.
Unadjusted stock close is used for conversion value; adjusted prices are used for continuous
stock-return factors. Missing observations represent unavailable trading data and are not
converted to zero returns.

### CITIC Industry Data

CITIC industry data must provide daily constituent snapshots with a security identifier,
snapshot date, industry code, and industry name for level 1 (and level 2 when available).
Industry membership is joined as of the observation date. Missing snapshots are neither
forward-filled nor backfilled.

## Runtime Configuration

| Environment variable | Purpose |
|---|---|
| `CB_MULTIFACTOR_ROOT` | Optional repository-root override |
| `CB_LEGACY_PROJECT_ROOT` | Parent location of approved historical raw and processed inputs |
| `CB_DATABASE_ROOT` | Root of approved local DuckDB sources |
| `CB_INTRADAY_ROOT` | Optional intraday-data root override; intraday data are outside the primary daily model |
| `CB_A_SHARE_DAILY_PATH` | Optional A-share daily DuckDB override |
| `TUSHARE_TOKEN` | Tushare credential used only for authorized downloads |

Do not commit credentials or environment-specific paths. `TUSHARE_TOKEN` must be supplied only
through the environment.

## Licensing, Redistribution, and Tests

Tushare, exchange, CITIC, and other third-party datasets remain subject to their respective
licenses and access agreements. This project provides no license to copy, publish, or redistribute
those inputs, derived raw extracts, or commercial database contents; non-redistribution is a
release requirement.

Public tests use synthetic fixtures only. Tests must not require proprietary data, local database
files, credentials, or machine-specific paths.
