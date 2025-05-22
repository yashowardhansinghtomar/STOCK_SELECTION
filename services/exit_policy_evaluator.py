# services/exit_policy_evaluator.py

import pandas as pd
from pydantic import BaseModel
from typing import Literal, Optional
from core.time_context import get_simulation_date
from core.feature_enricher_multi import enrich_multi_interval_features
from core.model_io import load_model
import json
import numpy as np

class ExitRule(BaseModel):
    kind: Literal["fixed_pct", "sma_cross", "time_stop", "trailing_pct"]
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trail: Optional[float] = None
    sma_window: Optional[int] = None
    max_holding_days: Optional[int] = None

def should_exit_model_based(position: dict, threshold: float = 0.5) -> bool:
    try:
        stock = position["stock"]
        entry_date = pd.to_datetime(position["entry_date"], errors="coerce")
        if pd.isna(entry_date):
            return False

        today = pd.to_datetime(get_simulation_date())
        if today <= entry_date:
            return False

        feats = enrich_multi_interval_features(stock, today)
        if feats.empty:
            return False

        model_obj = load_model("exit_classifier")
        model, features = model_obj[0], model_obj[1]

        X = feats[features].fillna(0)
        proba = model.predict_proba(X)[0][1]  # Probability of exit = 1

        return proba > threshold

    except Exception as e:
        import traceback
        print(f"\u26a0\ufe0f ML-based exit check failed: {e}\n{traceback.format_exc()}")
        return False
