DROP MATERIALIZED VIEW IF EXISTS stock_features_60m;

CREATE MATERIALIZED VIEW stock_features_60m AS
WITH base AS (
    SELECT
        symbol AS stock,
        date,
        close,
        high,
        low,
        volume
    FROM stock_price_history
    WHERE interval = '60minute'
),
price_diff AS (
    SELECT *,
        LAG(close) OVER (PARTITION BY stock ORDER BY date) AS prev_close,
        close - LAG(close) OVER (PARTITION BY stock ORDER BY date) AS price_change
    FROM base
),
gain_loss AS (
    SELECT *,
        GREATEST(price_change, 0) AS gain,
        GREATEST(-price_change, 0) AS loss
    FROM price_diff
),
avg_gain_loss AS (
    SELECT *,
        AVG(gain) OVER (PARTITION BY stock ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS avg_gain,
        AVG(loss) OVER (PARTITION BY stock ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS avg_loss
    FROM gain_loss
),
smas AS (
    SELECT *,
        AVG(close) OVER (PARTITION BY stock ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS sma_short_val,
        AVG(close) OVER (PARTITION BY stock ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS sma_long_val
    FROM avg_gain_loss
),
macd_base AS (
    SELECT *,
        close - sma_short_val AS macd,
        LAG(close - sma_short_val) OVER (PARTITION BY stock ORDER BY date) AS macd_lag
    FROM smas
),
atr_input AS (
    SELECT *,
        GREATEST(high - low, ABS(close - prev_close)) AS tr
    FROM macd_base
),
atr_final AS (
    SELECT *,
        AVG(tr) OVER (PARTITION BY stock ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS atr_14
    FROM atr_input
),
vwap_calc AS (
    SELECT *,
        SUM(close * volume) OVER (PARTITION BY stock ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) /
        NULLIF(SUM(volume) OVER (PARTITION BY stock ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW), 0) AS vwap
    FROM atr_final
),
final AS (
    SELECT
        stock,
        date,
        ROUND(sma_short_val::numeric, 2) AS sma_short,
        ROUND(sma_long_val::numeric, 2) AS sma_long,
        ROUND(macd::numeric, 2) AS macd,
        ROUND((macd - macd_lag)::numeric, 2) AS macd_histogram,
        ROUND((100 - (100 / (1 + avg_gain / NULLIF(avg_loss, 0))))::numeric, 2) AS rsi_thresh,
        ROUND(atr_14::numeric, 2) AS atr_14,
        ROUND((MAX(close) OVER (PARTITION BY stock ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) -
              MIN(close) OVER (PARTITION BY stock ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW))::numeric, 2) AS bb_width,
        ROUND(vwap::numeric, 2) AS vwap,
        ROUND((close - AVG(close) OVER (PARTITION BY stock ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW))::numeric, 2) AS vwap_dev,
        CASE WHEN volume > 1.5 * AVG(volume) OVER (PARTITION BY stock ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) THEN 1 ELSE 0 END AS volume_spike,
        ROUND(STDDEV(close) OVER (PARTITION BY stock ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)::numeric, 2) AS volatility_10,
        CASE WHEN MAX(close) OVER (PARTITION BY stock ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) -
                   MIN(close) OVER (PARTITION BY stock ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)
              < 0.02 * AVG(close) OVER (PARTITION BY stock ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)
             THEN 1 ELSE 0 END AS price_compression,
        DENSE_RANK() OVER (ORDER BY stock) AS stock_encoded
    FROM vwap_calc
)
SELECT * FROM final WHERE sma_long IS NOT NULL;
