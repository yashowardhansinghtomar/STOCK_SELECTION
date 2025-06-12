import pandas as pd
import lightgbm as lgb
from core.logger.logger import logger
from db.replay_buffer_sql import load_replay_episodes

def train_models(replay_buffer=None, up_to_date=None):
    logger.info("🧠 Starting joint policy model training...")

    # Step 1: Load replay episodes
    df = load_replay_episodes()
    if df.empty:
        logger.warning("❌ No replay episodes found.")
        return

    # Step 2: Drop null rewards
    df = df[df["reward"].notna()]
    if df.empty:
        logger.warning("❌ No valid rewards with labels.")
        return

    # Step 3: Extract features and target
    features_df = df["features"].apply(pd.Series)
    strategy_df = df["strategy_config"].apply(pd.Series)
    X = pd.concat([features_df, strategy_df], axis=1).fillna(0)

    y = df["reward"].astype(float)

    logger.info(f"📦 Training on {len(X)} episodes, {X.shape[1]} features")

    # Step 4: Train LightGBM regressor (reward predictor)
    model = lgb.LGBMRegressor(n_estimators=100, max_depth=6)
    model.fit(X, y)

    # Step 5: Save model to disk (or model_store table)
    import joblib
    joblib.dump(model, "models/joint_policy_model.lgb")
    logger.success("✅ Joint policy model trained and saved as joint_policy_model.lgb")
