import pandas as pd
import lightgbm as lgb
import json
import joblib
import numpy as np
from datetime import date
import matplotlib.pyplot as plt

from core.logger.logger import logger
from db.replay_buffer_sql import load_replay_episodes
from db.postgres_manager import run_query

def preprocess_training_data(df: pd.DataFrame) -> pd.DataFrame:
    # --- Unpack features and strategy_config safely ---
    if "features" in df.columns:
        try:
            df["features"] = df["features"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else (x or {})
            )
            features_df = pd.json_normalize(df["features"])
        except Exception as e:
            logger.warning(f"⚠️ Failed to unpack features: {e}")
            features_df = pd.DataFrame()
    else:
        features_df = pd.DataFrame()

    if "strategy_config" in df.columns:
        try:
            df["strategy_config"] = df["strategy_config"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else (x or {})
            )
            strat_df = pd.json_normalize(df["strategy_config"])
        except Exception as e:
            logger.warning(f"⚠️ Failed to unpack strategy_config: {e}")
            strat_df = pd.DataFrame()
    else:
        strat_df = pd.DataFrame()

    # --- Merge all features ---
    merged_df = pd.concat([features_df, strat_df], axis=1)

    # --- Encode interval if present ---
    if "interval" in df.columns:
        interval_map = {"day": 0, "15minute": 1, "60minute": 2, "minute": 3}
        merged_df["interval_encoded"] = df["interval"].map(interval_map).fillna(-1).astype(int)

    # --- Drop non-numeric or irrelevant fields ---
    drop_cols = ["source", "strategy", "interval"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # --- Keep only numeric and boolean columns ---
    merged_df = merged_df.select_dtypes(include=["number", "bool"]).fillna(0)
    return merged_df

def train_models(replay_buffer=None, up_to_date=None):
    logger.info("🧠 Starting joint policy model training...")

    df = load_replay_episodes()
    if df.empty:
        logger.warning("❌ No replay episodes found.")
        return

    df = df[df["reward"].notna()]
    if df.empty:
        logger.warning("❌ No valid rewards with labels.")
        return

    X = preprocess_training_data(df)
    y = df["reward"].astype(float)

    if X.empty or y.empty:
        logger.warning("⚠️ No valid training data after preprocessing.")
        return

    logger.info(f"📦 Training on {len(X)} episodes, {X.shape[1]} features")

    # 🔬 Log distributions for strategy parameters
    logger.info("🔬 Strategy parameter distributions:")
    for col in ["rsi", "rsi_thresh", "sma_short", "sma_long"]:
        if col in X.columns:
            desc = X[col].describe().to_dict()
            logger.info(f"   - {col}: {desc}")

    # Train LightGBM model
    model = lgb.LGBMRegressor(n_estimators=100, max_depth=6)
    model.fit(X, y)

    # 🔢 Log top feature importances
    importances = model.feature_importances_
    feature_names = X.columns
    sorted_idx = np.argsort(importances)[::-1]

    logger.info("📊 Top 10 Feature Importances:")
    for i in sorted_idx[:10]:
        logger.info(f"   - {feature_names[i]}: {importances[i]}")

    # 📈 Plot feature importance
    plt.figure(figsize=(10, 6))
    top_n = 15
    plt.barh(
        [feature_names[i] for i in sorted_idx[:top_n]][::-1],
        [importances[i] for i in sorted_idx[:top_n]][::-1],
    )
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("models/feature_importance.png")
    logger.info("📉 Feature importance chart saved to models/feature_importance.png")

    # 🗂️ Save to feature_importance_history table
    model_name = "joint_policy_model.lgb"
    today = date.today()
    for i in sorted_idx:
        run_query("""
            INSERT INTO feature_importance_history (date, feature, importance, model_name)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (date, feature, model_name) DO UPDATE SET importance = EXCLUDED.importance
        """, [today, feature_names[i], int(importances[i]), model_name], fetchall=False)
    logger.info("🗂️ Feature importances logged to feature_importance_history")

    # Save model
    joblib.dump(model, "models/joint_policy_model.lgb")
    logger.success("✅ Joint policy model trained and saved as joint_policy_model.lgb")

if __name__ == "__main__":
    train_models()
