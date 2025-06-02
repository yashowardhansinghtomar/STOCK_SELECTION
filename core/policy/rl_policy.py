# core/policy/rl_policy.py

import os
import torch
import pandas as pd
from core.logger.logger import logger

RL_MODEL_PATH = "models/rl_policy.pt"

class RLPolicyModel:
    def __init__(self):
        self.model = None
        self.loaded = False
        self.load()

    def load(self, path: str = RL_MODEL_PATH):
        if os.path.exists(path):
            try:
                self.model = torch.jit.load(path)
                self.model.eval()
                self.loaded = True
                logger.info(f"[RL POLICY] Loaded model from {path}")
            except Exception as e:
                logger.error(f"[RL POLICY] Failed to load model: {e}")
                self.loaded = False
        else:
            logger.warning(f"[RL POLICY] Model not found at {path}")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.loaded or self.model is None:
            raise ValueError("RL model is not loaded.")

        try:
            with torch.no_grad():
                inputs = torch.tensor(X.values, dtype=torch.float32)
                outputs = self.model(inputs).numpy()

            return pd.DataFrame({
                "enter_prob": outputs[:, 0],
                "position_size": outputs[:, 1],
                "exit_days": outputs[:, 2]
            }, index=X.index)
        except Exception as e:
            logger.warning(f"[RL POLICY] Prediction failed: {e}")
            return pd.DataFrame()
