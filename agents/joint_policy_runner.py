# agents/joint_policy_runner.py

import pandas as pd
from models.joint_policy import JointPolicyModel
from core.time_context.time_context import get_simulation_date
from core.data_provider.data_provider import load_data
from core.feature_provider import fetch_features
from core.config.config import settings
from db.conflict_utils import insert_with_conflict_handling
from core.logger.logger import logger
from datetime import datetime

def run_joint_policy_predictions():
    today = pd.to_datetime(get_simulation_date())
    today_str = today.strftime("%Y-%m-%d")

    # Step 1: Load today's recommendations
    df = load_data(settings.recommendations_table)
    if df is None or df.empty:
        logger.warning("⚠️ No recommendations found.")
        return

    df = df[df["trade_triggered"] == 1]
    if df.empty:
        logger.info("✅ No trade-triggered signals today.")
        return

    symbols = df["stock"].unique().tolist()

    # Step 2: Load features
    records = []
    model = JointPolicyModel()
    model.load()  # Load from default path

    for symbol in symbols:
        try:
            features = fetch_features(symbol, interval="day", refresh_if_missing=False)
            if features is None or features.empty:
                continue

            X = features.drop(columns=["date", "stock"], errors="ignore").tail(1)
            preds = model.predict(X).iloc[0]

            records.append({
                "date": today.date(),
                "stock": symbol,
                "enter_prob": float(preds["enter_prob"]),
                "position_size": float(preds["position_size"]),
                "exit_days": int(preds["exit_days"]),
                "strategy_config": {},  # optional
                "confidence": float(preds["enter_prob"]),
                "created_at": datetime.utcnow()
            })

        except Exception as e:
            logger.warning(f"⚠️ Failed to predict for {symbol}: {e}")

    if records:
        insert_with_conflict_handling(pd.DataFrame(records), "joint_policy_predictions")
        logger.success(f"✅ Joint policy predictions written for {len(records)} symbols.")
    else:
        logger.info("ℹ️ No joint policy predictions written.")

if __name__ == "__main__":
    run_joint_policy_predictions()
