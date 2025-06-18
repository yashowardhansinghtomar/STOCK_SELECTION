import pandas as pd
import numpy as np
from core.logger.logger import logger
from core.model_io import load_model
from core.config.config import settings

def predict_param_config(enriched: pd.DataFrame) -> dict:
    if enriched is None or enriched.empty:
        logger.warning("⚠️ Empty features passed to param_model.")
        return {}

    # Determine interval
    interval = enriched.get("interval", "day")
    if isinstance(interval, pd.Series):
        interval = interval.iloc[0]
    interval = str(interval).lower()

    model_name = f"{settings.model_names['param']}_{interval}"

    try:
        model_obj = load_model(model_name)
        model = model_obj["model"]
        features = model_obj["features"]
        le = model_obj["label_encoder"]
    except Exception as e:
        logger.error(f"❌ Failed to load {model_name}: {e}")
        return {}

    missing_feats = [f for f in features if f not in enriched.columns]
    if missing_feats:
        logger.error(f"❌ Missing required features for {interval} param model: {missing_feats}")
        return {}

    try:
        X = enriched[features].copy().fillna(0).replace([np.inf, -np.inf], 0)
        pred = model.predict(X)[0]
        decoded_interval = le.inverse_transform([int(round(pred[0]))])[0]

        result = {
            "interval": decoded_interval,
            "sma_short": int(round(pred[1])),
            "sma_long": int(round(pred[2])),
            "rsi_thresh": float(pred[3]),
            "confidence_score": float(np.std(pred)),  # crude fallback proxy
            "explanation": f"Predicted using {model_name}"
        }
        return result
    except Exception as e:
        logger.error(f"❌ Param model prediction failed for {model_name}: {e}")
        return {}
