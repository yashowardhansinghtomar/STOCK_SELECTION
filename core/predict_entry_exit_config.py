import pandas as pd
import numpy as np
from core.logger import logger
from core.model_io import load_model


def predict_entry_exit_config(enriched: pd.DataFrame) -> dict:
    try:
        model_obj = load_model("entry_exit_model")
        clf = model_obj["clf"]
        reg = model_obj["reg"]
        features = model_obj["features"]
        label_enc = model_obj["exit_kind_encoder"]
    except Exception as e:
        logger.error(f"❌ Failed to load entry_exit_model: {e}")
        return {}

    if enriched is None or enriched.empty:
        logger.warning("⚠️ Empty features passed to entry_exit_model.")
        return {}

    X = enriched[features].fillna(0).replace([np.inf, -np.inf], 0)

    entry_pred = clf.predict(X)[0]
    exit_params = reg.predict(X)[0]

    exit_kind = label_enc.inverse_transform([int(round(exit_params[0]))])[0]

    return {
        "entry_signal": int(entry_pred),
        "exit_rule": {
            "kind": exit_kind,
            "stop_loss": float(exit_params[1]),
            "take_profit": float(exit_params[2]),
            "trail": float(exit_params[3]),
            "sma_window": int(round(exit_params[4])),
            "max_holding_days": int(round(exit_params[5])),
        },
    }
