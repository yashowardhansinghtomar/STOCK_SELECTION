import pandas as pd
from core.time_context.time_context import get_simulation_date
from models.joint_policy import JointPolicyModel
from core.logger.logger import logger

joint_model = None

try:
    joint_model = JointPolicyModel.load()
except Exception as e:
    logger.warning(f"[PREDICTOR] Failed to load joint model: {e}")
    joint_model = None

def predict_dual_model(stock: str, feature_df: pd.DataFrame = None) -> list:
    """
    Predicts if a trade should be triggered and its expected return.
    Returns a list of dicts with unified format:
    [{
       "stock": <symbol>,
       "trade_triggered": <0|1>,
       "predicted_return": <float>,
       "recommended_config": <dict>,
       "model_source": "joint"
    }]
    """
    if joint_model is None:
        logger.error("[PREDICTOR] Joint model is not available.")
        return []

    try:
        result = joint_model.predict(stock, feature_df)
        return [{
            "stock": stock,
            "trade_triggered": result.get("enter", 0),
            "predicted_return": result.get("predicted_return", 0.0),
            "recommended_config": result,
            "model_source": "joint"
        }]
    except Exception as e:
        logger.warning(f"[PREDICTOR] Joint model prediction failed for {stock}: {e}")
        return []

__all__ = ["predict_dual_model"]
