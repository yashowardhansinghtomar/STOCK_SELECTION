# core/predict/ppo_live_policy.py

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from core.feature_store.feature_store import get_or_compute
from core.logger.logger import logger

MODEL_PATH = "checkpoints/ppo_sb3_model"

class PPOLivePolicy:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model = PPO.load(model_path)
        logger.info("[PPO LIVE] PPO model loaded for inference.")

    def predict(self, stock: str, date: str) -> dict:
        try:
            df = get_or_compute(stock, interval="15minute", date=date)
            if df is None or df.empty:
                logger.warning(f"[PPO LIVE] No features found for {stock} @ {date}")
                return None

            features = df.drop(columns=["stock", "date", "interval"]).values[0]
            action, _ = self.model.predict(features, deterministic=True)

            position_size = float(np.clip(action[0], 0, 1))
            exit_days = int(np.clip(np.round(1 + action[1] * 4), 1, 5))

            return {
                "stock": stock,
                "date": date,
                "source": "ppo_sb3",
                "enter_prob": position_size,  # Since PPO doesn't return prob directly
                "position_size": position_size,
                "exit_days": exit_days
            }
        except Exception as e:
            logger.error(f"[PPO LIVE] Inference failed: {e}")
            return None
