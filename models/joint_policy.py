# models/joint_policy.py

import lightgbm as lgb
import pandas as pd
import joblib
import os

MODEL_PATH = "models/joint_policy_model.pkl"

class JointPolicyModel:
    def __init__(self):
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        data = lgb.Dataset(X, label=y["enter"])
        self.model = lgb.train(
            {"objective": "binary", "metric": "binary_logloss"},
            data,
            num_boost_round=100
        )

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model is not loaded.")
        enter_prob = self.model.predict(X)
        return pd.DataFrame({
            "enter_prob": enter_prob,
            "position_size": enter_prob,   # heuristic: scale same as prob
            "exit_days": (1 + 4 * enter_prob).round()  # 1–5 days
        }, index=X.index)

    def save(self, path: str = MODEL_PATH):
        if self.model:
            joblib.dump(self.model, path)

    def load(self, path: str = MODEL_PATH):
        if os.path.exists(path):
            self.model = joblib.load(path)
        else:
            raise FileNotFoundError(f"Model file not found at {path}")
