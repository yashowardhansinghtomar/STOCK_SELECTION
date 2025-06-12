# bootstrap_filter_training_data.py

import pandas as pd
from datetime import datetime, timedelta
from core.data_provider.data_provider import fetch_stock_data, save_data
from core.feature_engineering.feature_enricher_multi import enrich_multi_interval_features
from core.backtest_bt import run_backtest_config as run_backtest
from db.postgres_manager import run_query, get_all_symbols
from core.logger.logger import logger
from core.config.strategy_config import StrategyConfig, ExitRule
import pandas_market_calendars as mcal
from utils.time_utils import make_naive  # ✅ ensure run_date is naive
import traceback

LABEL_THRESHOLDS = {
    "sharpe": 1.0,
    "avg_trade_return": 0.015,  # 1.5%
}

START_DATE = datetime.today() - timedelta(days=60)
END_DATE   = datetime.today() - timedelta(days=1)


def get_strategy_config():
    return {
        "sma_short": 10,
        "sma_long": 30,
        "rsi_thresh": 55,
    }


def label_from_backtest(rec):
    if pd.notna(rec["sharpe"]) and rec["sharpe"] >= LABEL_THRESHOLDS["sharpe"]:
        return 1
    if pd.notna(rec["avg_trade_return"]) and rec["avg_trade_return"] >= LABEL_THRESHOLDS["avg_trade_return"]:
        return 1
    return 0


def bootstrap():
    symbols = get_all_symbols()
    all_recs = []

    # ✅ Restrict to actual NSE trading days (tz-naive)
    nse = mcal.get_calendar("NSE")
    valid_days = nse.valid_days(start_date=START_DATE, end_date=END_DATE)

    for symbol in symbols:
        for sim_date in valid_days:
            try:
                # ✅ Ensure sim_date is completely offset-naive
                sim_date = make_naive(pd.to_datetime(sim_date))
                date_str = sim_date.strftime("%Y-%m-%d")

                # fetch and enrich features
                enriched = enrich_multi_interval_features(symbol, sim_date, intervals=["day"])
                if enriched.empty:
                    logger.warning(f"⚠️ No features for {symbol} on {date_str}, skipping.")
                    continue

                # build strategy config
                cfg = StrategyConfig(
                    sma_short = get_strategy_config()["sma_short"],
                    sma_long  = get_strategy_config()["sma_long"],
                    rsi_entry = get_strategy_config()["rsi_thresh"],
                    exit_rule = ExitRule(
                        kind             = "fixed_pct",
                        stop_loss        = 0.03,
                        take_profit      = 0.06,
                        max_holding_days = 10
                    )
                )

                # run backtest with naive datetime
                result = run_backtest(stock=symbol, cfg=cfg, run_date=sim_date)

                rec = {
                    "stock":            symbol,
                    "date":             sim_date,
                    "sma_short":        cfg.sma_short,
                    "sma_long":         cfg.sma_long,
                    "rsi_thresh":       cfg.rsi_entry,
                    "total_return":     result.get("total_return", 0.0),
                    "avg_trade_return": result.get("avg_trade_return", 0.0),
                    "sharpe":           result.get("sharpe", 0.0),
                    "trade_count":      result.get("trade_count", 0),
                    "interval":         "day",
                    "source":           "bootstrap",
                    "trade_triggered":  0,
                    "confidence":       0.0,
                    "label":            0,
                    "created_at":       datetime.utcnow(),
                }

                rec["label"] = label_from_backtest(rec)
                all_recs.append(rec)

                logger.info(f"✅ {symbol} @ {date_str} | Label = {rec['label']}")

            except Exception as e:
                logger.warning(f"❌ Backtest failed for {symbol} @ {date_str}: {e}")
                traceback.print_exc()  # ✅ This will print the full stack trace

    # compile and persist
    df = pd.DataFrame(all_recs)
    if df.empty:
        logger.warning("🚫 No bootstrap records generated.")
        return

    # refresh created_at, dedupe, and upsert
    df["created_at"] = pd.Timestamp.utcnow()
    df.drop_duplicates(subset=["stock", "date"], inplace=True)

    min_date = df["date"].min().date()
    max_date = df["date"].max().date()

    run_query(
        "DELETE FROM recommendations WHERE date BETWEEN %s AND %s",
        (min_date, max_date)
    )

    save_data(df, "recommendations", if_exists="append")
    logger.success(f"Bootstrapped {len(df)} labeled recommendations.")


if __name__ == "__main__":
    bootstrap()
