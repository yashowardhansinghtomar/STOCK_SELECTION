# bootstrap_filter_training_data.py

import pandas as pd
from datetime import datetime, timedelta
from core.data_provider.data_provider import get_all_symbols, fetch_stock_data, save_data
from core.feature_engineering.feature_enricher_multi import enrich_multi_interval_features
from core.backtest_bt import run_backtest
from db.postgres_manager import run_query
from core.logger.logger import logger

LABEL_THRESHOLDS = {
    "sharpe": 1.0,
    "avg_trade_return": 0.015,  # 1.5%
}

START_DATE = datetime.today() - timedelta(days=60)
END_DATE = datetime.today() - timedelta(days=1)

# Placeholder for backtest config
def get_strategy_config():
    return {
        "sma_short": 10,
        "sma_long": 30,
        "rsi_thresh": 40,
    }

def label_from_backtest(row):
    if pd.notna(row["sharpe"]) and row["sharpe"] >= LABEL_THRESHOLDS["sharpe"]:
        return 1
    if pd.notna(row["avg_trade_return"]) and row["avg_trade_return"] >= LABEL_THRESHOLDS["avg_trade_return"]:
        return 1
    return 0

def bootstrap():
    symbols = get_all_symbols()
    all_recs = []
    for symbol in symbols:
        for sim_date in pd.date_range(START_DATE, END_DATE):
            date_str = sim_date.strftime("%Y-%m-%d")
            try:
                enriched = enrich_multi_interval_features(symbol, sim_date, intervals=["day"])
                if enriched.empty:
                    continue
                strategy_config = get_strategy_config()
                result = run_backtest(
                    stock=symbol,
                    run_date=sim_date,
                    sma_short=strategy_config["sma_short"],
                    sma_long=strategy_config["sma_long"],
                    rsi_thresh=strategy_config["rsi_thresh"]
                )
                rec = {
                    "stock": symbol,
                    "date": sim_date,
                    "sma_short": strategy_config["sma_short"],
                    "sma_long": strategy_config["sma_long"],
                    "rsi_thresh": strategy_config["rsi_thresh"],
                    "total_return": result.get("total_return", 0.0),
                    "avg_trade_return": result.get("avg_trade_return", 0.0),
                    "sharpe": result.get("sharpe", 0.0),
                    "trade_count": result.get("trade_count", 0),
                    "interval": "day",
                    "source": "bootstrap",
                    "trade_triggered": 0,
                    "confidence": 0.0,
                }
                rec["label"] = label_from_backtest(rec)
                all_recs.append(rec)
                logger.info(f"✅ {symbol} @ {date_str} | Label = {rec['label']}")
            except Exception as e:
                logger.warning(f"❌ Backtest failed for {symbol} @ {date_str}: {e}")
    
    df = pd.DataFrame(all_recs)
    if not df.empty:
        save_data(df, "recommendations", if_exists="append")
        logger.success(f"Bootstrapped {len(df)} labeled recommendations.")
    else:
        logger.warning("No recommendations generated during bootstrap.")

if __name__ == "__main__":
    bootstrap()
