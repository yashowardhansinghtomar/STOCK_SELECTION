import pandas as pd
import numpy as np
import traceback
from typing import Literal, Optional
from pydantic import BaseModel
from core.time_context.time_context import get_simulation_date
from core.feature_engineering.feature_enricher_multi import enrich_multi_interval_features
from core.model_io import load_model
from core.logger.logger import logger

class ExitRule(BaseModel):
    kind: Literal["fixed_pct", "sma_cross", "time_stop", "trailing_pct"]
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trail: Optional[float] = None
    sma_window: Optional[int] = None
    max_holding_days: Optional[int] = None

def get_exit_probability(position: dict, default: float = 0.5) -> float:
    """
    Return probability that the position should be exited (1 = exit).
    """
    try:
        stock = position["stock"]
        entry_date = pd.to_datetime(position["entry_date"], errors="coerce")
        interval = position.get("interval", "day")
        if pd.isna(entry_date):
            logger.warnings(f"⚠️ Invalid entry_date for {stock}")
            return default

        today = pd.to_datetime(get_simulation_date())
        if today <= entry_date:
            return default

        feats = enrich_multi_interval_features(stock, today, intervals=[interval])
        if feats.empty:
            logger.warnings(f"⚠️ No features found for {stock} at {interval}")
            return default

        model_name = f"exit_classifier_{interval}"
        model_obj = load_model(model_name)
        model, features = model_obj["model"], model_obj["features"]

        X = feats[features].fillna(0).replace([np.inf, -np.inf], 0)
        proba = model.predict_proba(X)[0][1]
        return float(proba)

    except Exception as e:
        logger.warnings(f"⚠️ ML-based exit check failed for {position.get('stock')}: {e}\n{traceback.format_exc()}")
        return default

def should_exit_model_based(position: dict, threshold: float = 0.5) -> bool:
    return get_exit_probability(position) > threshold
