import pandas as pd
from datetime import datetime, timedelta
from prefect import flow, task, get_run_logger

from core.logger.logger import logger
from core.time_context.time_context import get_simulation_date
from core.data_provider.data_provider import fetch_stock_data, save_data
from core.feature_engineering.feature_enricher_multi import enrich_multi_interval_features
from core.notifications.redis_notifier import push_feature_ready
from db.postgres_manager import get_all_symbols

LOOKBACK_MINUTES = 30
INTERVAL = "minute"

@task
def backfill_for_symbol(symbol: str, lookback_minutes: int = 30):
    log = get_run_logger()
    end_time = pd.to_datetime(get_simulation_date())
    start_time = end_time - timedelta(minutes=lookback_minutes)

    log.info(f"Backfilling {INTERVAL} features for {symbol} from {start_time} to {end_time}...")

    try:
        df = fetch_stock_data(symbol, start=start_time, end=end_time, interval=INTERVAL)
        if df is None or df.empty:
            log.log_warning(f"No price data available for {symbol} during backfill window.")
            return

        df["date"] = pd.to_datetime(df["date"])
        for dt in df["date"].unique():
            enriched = enrich_multi_interval_features(symbol, sim_date=dt, intervals=[INTERVAL])
            if not enriched.empty:
                save_data(enriched, table="stock_features_1m", if_exists="append")
                push_feature_ready(symbol, queue="feature_ready_1m")
                log.info(f"✅ Enriched + queued {symbol} @ {dt}")
            else:
                log.log_warning(f"⚠️ No enriched output for {symbol} @ {dt}")

    except Exception as e:
        log.error(f"❌ Backfill failed for {symbol}: {e}")

@flow(name="backfill-1m-feature-flow")
def backfill_1m_feature_flow():
    symbols = get_all_symbols()
    for symbol in symbols:
        backfill_for_symbol(symbol, lookback_minutes=LOOKBACK_MINUTES)

if __name__ == "__main__":
    backfill_1m_feature_flow()
