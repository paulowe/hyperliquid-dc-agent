/* =============================================================================
Microstructure Extract (Event-Time, Deduped) — Price, Volume, CVD
Source: {project}.{dataset_id}.{table_id}
Dialect: BigQuery Standard SQL

PURPOSE
- Event-time ordered trade stream with price, volumes, and CVD for modeling.

CHOICES
- Event time for ordering; LOAD_TIME only for partition pruning & tiebreaks.
- Dedup by (MARKET, INSTRUMENT, trade_id), keeping earliest LOAD_TIME.

OUTPUT (per trade)
- MARKET, INSTRUMENT, trade_id, trade_ts, trade_time_toronto
- price, vol_base, vol_quote, cvd_base, cvd_quote, log_return, inter_trade_duration_seconds
============================================================================= */

WITH t AS (
  SELECT
    MARKET,
    INSTRUMENT,
    CAST(LAST_PROCESSED_TRADE_CCSEQ AS STRING) AS trade_id,  -- venue trade sequence/id
    TIMESTAMP_MICROS(
      CAST(LAST_PROCESSED_TRADE_TS AS INT64) * 1000000
      + CAST(DIV(LAST_PROCESSED_TRADE_TS_NS, 1000) AS INT64)
    ) AS trade_ts,                                            -- event timestamp (µs)
    LAST_PROCESSED_TRADE_PRICE          AS price,             -- executed trade price
    LAST_PROCESSED_TRADE_QUANTITY       AS vol_base,          -- base units (e.g., BTC)
    LAST_PROCESSED_TRADE_QUOTE_QUANTITY AS vol_quote,         -- quote units (≈ vol_base * price)
    LAST_PROCESSED_TRADE_SIDE           AS side,              -- 'BUY' / 'SELL'
    LOAD_TIME
  FROM `{{ project }}.{{ dataset_id }}.{{ table_id }}`
  WHERE INSTRUMENT = '{{ instrument }}'
    AND LOAD_TIME BETWEEN '{{start_time}}' AND '{{end_time}}'          -- partition prune
    -- AND MOD(ABS(FARM_FINGERPRINT(CAST(LAST_PROCESSED_TRADE_CCSEQ AS STRING))), 100) < 10
),

base AS (
  SELECT *
  FROM t
  WHERE trade_ts BETWEEN '{{start_time}}' AND '{{end_time}}'           -- event-time window
  QUALIFY ROW_NUMBER() OVER (                                -- dedup snapshots
    PARTITION BY MARKET, INSTRUMENT, trade_id
    ORDER BY trade_ts, LOAD_TIME
  ) = 1
),

w AS (
  SELECT
    b.*,
    CASE WHEN UPPER(side) = 'BUY'  THEN  1
         WHEN UPPER(side) = 'SELL' THEN -1
         ELSE 0 END AS sgn,
    LAG(price) OVER (
      PARTITION BY MARKET, INSTRUMENT
      ORDER BY trade_ts, trade_id, LOAD_TIME
    ) AS prev_price,
    LAG(trade_ts) OVER (
      PARTITION BY MARKET, INSTRUMENT
      ORDER BY trade_ts, trade_id, LOAD_TIME
    ) AS prev_ts
  FROM base b
)

SELECT
  MARKET,
  INSTRUMENT,
  trade_id,
  trade_ts,
  DATETIME(trade_ts, "America/Toronto") AS load_time_toronto,
  price AS PRICE,
  vol_base,
  vol_quote,

  -- CVD per venue (base & quote units)
  SUM(sgn * vol_base)  OVER (
    PARTITION BY MARKET, INSTRUMENT
    ORDER BY trade_ts, trade_id, LOAD_TIME
  ) AS cvd_base,
  SUM(sgn * vol_quote) OVER (
    PARTITION BY MARKET, INSTRUMENT
    ORDER BY trade_ts, trade_id, LOAD_TIME
  ) AS cvd_quote,

  -- Microstructure extras
  SAFE.LN(price / prev_price) AS log_return,
  TIMESTAMP_DIFF(trade_ts, prev_ts, MICROSECOND) / 1e6 AS inter_trade_duration_seconds

FROM w
ORDER BY trade_ts, trade_id
