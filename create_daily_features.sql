-- 1. Drop the old view
DROP MATERIALIZED VIEW IF EXISTS stock_features;

-- 2. Recreate with proper T-1 alignment
CREATE MATERIALIZED VIEW stock_features AS
WITH diffs AS (
  SELECT
    symbol      AS stock,
    date,
    close,
    high,
    low,
    volume,
    close - lag(close) OVER (PARTITION BY symbol ORDER BY date) AS diff,
    GREATEST(
      high - low,
      abs(high - lag(close) OVER (PARTITION BY symbol ORDER BY date)),
      abs(low  - lag(close) OVER (PARTITION BY symbol ORDER BY date))
    ) AS true_range
  FROM stock_price_history
  WHERE close IS NOT NULL
),
windows AS (
  SELECT
    stock,
    date,
    avg(close)         OVER w20    AS sma_short,
    avg(close)         OVER w50    AS sma_long,
    avg(close)         OVER w20    AS mean_20,
    stddev_samp(close) OVER w20    AS sigma_20,
    sum(close * volume) OVER w20   AS vwap_num,
    avg(volume)        OVER w20    AS avg_vol20,
    stddev_samp(close) OVER w10    AS volatility_10,
    volume,
    avg(GREATEST(diff, 0))   OVER rsi14  AS avg_gain,
    avg(GREATEST(-diff, 0))  OVER rsi14  AS avg_loss,
    avg(true_range)         OVER atr14   AS atr_14
  FROM diffs
  WINDOW
    w20   AS (PARTITION BY stock ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
    w50   AS (PARTITION BY stock ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW),
    w10   AS (PARTITION BY stock ORDER BY date ROWS BETWEEN 9  PRECEDING AND CURRENT ROW),
    rsi14 AS (PARTITION BY stock ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW),
    atr14 AS (PARTITION BY stock ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW)
),
final AS (
  SELECT
    stock,
    date,
    sma_short,
    sma_long,
    (sma_short - sma_long) AS macd,
    ((sma_short - sma_long)
      - avg(sma_short - sma_long) 
        OVER (PARTITION BY stock ORDER BY date ROWS BETWEEN 8 PRECEDING AND CURRENT ROW)
    ) AS macd_histogram,
    (100 - 100 / (1 + avg_gain / NULLIF(avg_loss, 0)))::numeric(5,2) AS rsi_thresh,
    atr_14,
    (mean_20 + 2 * sigma_20) - (mean_20 - 2 * sigma_20) AS bb_width,
    vwap_num / NULLIF(avg_vol20, 0) AS vwap,
    ((vwap_num / NULLIF(avg_vol20, 0)) - mean_20) / NULLIF(mean_20, 0) AS vwap_dev,
    volume / NULLIF(avg_vol20, 0) AS volume_spike,
    volatility_10,
    (mean_20 + 2 * sigma_20) - (mean_20 - 2 * sigma_20) AS price_compression,
    dense_rank() OVER (ORDER BY stock) AS stock_encoded
  FROM windows
)
SELECT *
FROM final
WHERE date < (
  SELECT MAX(date)
  FROM stock_price_history AS s
  WHERE s.symbol = final.stock
)
AND sma_short IS DISTINCT FROM sma_long;

-- 3. Refresh view
REFRESH MATERIALIZED VIEW stock_features;
