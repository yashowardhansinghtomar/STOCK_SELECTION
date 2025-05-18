import pandas as pd
from core.logger import logger
from core.time_context import get_simulation_date
from core.data_provider import load_data, save_data
from core.config import settings

def update_training_data():
    logger.start("🧠 Updating training data via feedback loop…")

    # 1. Load today’s paper trades
    trades = load_data(settings.trades_table)
    if trades is None or trades.empty:
        logger.info("📬 No paper trades to update.")
        return

    # 2. Filter to today's trades
    today = get_simulation_date()
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], errors="coerce")
    today_trades = trades[trades["timestamp"].dt.date == pd.to_datetime(today).date()]
    if today_trades.empty:
        logger.info("📬 No trades executed today.")
        return

    # 3. Load pre-enriched features from SQL
    feats = load_data(settings.feature_table)
    if feats is None or feats.empty:
        logger.error("❌ No features available for feedback loop.")
        return

    # Normalize casing and dates
    today_trades["stock"] = today_trades["stock"].str.upper()
    feats["stock"] = feats["stock"].str.upper()
    today_trades["feature_date"] = (
        pd.to_datetime(today_trades["timestamp"]).dt.normalize() - pd.Timedelta(days=1)
    )
    feats["date"] = pd.to_datetime(feats["date"]).dt.normalize()

    # 4. As-of merge: latest feature ≤ trade date, per stock
    today_trades = today_trades.sort_values(["stock", "feature_date"])
    feats = feats.sort_values(["stock", "date"])
    merged = pd.merge_asof(
        today_trades,
        feats,
        left_on="feature_date",
        right_on="date",
        by="stock",
        direction="backward",
        suffixes=("", "_feat")
    )
    # drop trades without a matching feature row
    merged = merged.dropna(subset=["date"])
    if merged.empty:
        logger.warning("⚠️ No matching features for today’s trades.")
        return

    # 5. Build training set (target = did we exit?)
    merged["target"] = (merged["action"] == "sell").astype(int)
    training_df = merged[settings.training_columns]

    # 6. Persist into training_data table
    save_data(training_df, settings.training_data_table)
    logger.success(f"✅ Inserted {len(training_df)} new training rows.")
