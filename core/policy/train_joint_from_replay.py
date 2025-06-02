import pandas as pd
import json
from core.model_io import save_model
from core.logger.logger import logger
from core.time_context.time_context import get_simulation_date
from db.postgres_manager import run_query
from models.joint_policy import JointPolicyModel

def load_replay_data():
    date = get_simulation_date().date()
    query = f"""
        SELECT *
        FROM replay_buffer
        WHERE date <= '{date}'
    """
    result = run_query(query)
    if not result:
        logger.warning("No replay buffer data found.")
        return None

    df = pd.DataFrame(result)

    # Extract action labels (you store them directly as scalar columns)
    y = df[["position_size", "exit_days"]].copy()

    # Use all numeric columns excluding meta ones as features
    feature_cols = [col for col in df.columns if col not in [
        "id", "stock", "timestamp", "position_size", "exit_days", "reward",
        "done", "strategy_config", "next_state", "interval", "date", "reason"
    ] and pd.api.types.is_numeric_dtype(df[col])]
    X = df[feature_cols].copy()

    return X, y

def main():
    logger.start("🔄 Distilling PPO into JointPolicyModel...")

    result = load_replay_data()
    if result is None:
        return

    X, y = result
    model = JointPolicyModel()
    model.fit(X, y)
    model.save()

    logger.success("✅ Distilled joint model saved.")

if __name__ == "__main__":
    main()
