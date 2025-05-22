# core/predict_param_model.py

import pandas as pd
import numpy as np
from core.logger import logger
from core.model_io import load_model


def predict_param_config(enriched: pd.DataFrame) -> dict:
    try:
        model_obj = load_model("param_model")
        model = model_obj["model"]
        features = model_obj["features"]
        le = model_obj["label_encoder"]
    except Exception as e:
        logger.error(f"❌ Failed to load param_model: {e}")
        return {}

    if enriched is None or enriched.empty:
        logger.warning("⚠️ Empty features passed to param_model.")
        return {}

    X = enriched[features].copy()
    X = X.fillna(0).replace([np.inf, -np.inf], 0)

    pred = model.predict(X)[0]
    interval = le.inverse_transform([int(round(pred[0]))])[0]

    return {
        "interval": interval,
        "sma_short": int(round(pred[1])),
        "sma_long": int(round(pred[2])),
        "rsi_thresh": float(pred[3])
    }
