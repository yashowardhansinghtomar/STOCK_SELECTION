# core/predictor.py
import pandas as pd
from core.model_io import load_model
from models.ml_dual_model_prediction_sql import predict_dual_model
from core.config import settings
from db.conflict_utils import insert_with_conflict_handling
from core.time_context import get_simulation_date
from agents.signal_arbitration_agent import SignalArbitrationAgent


def predict_dual_model(stock: str, feature_df: pd.DataFrame = None) -> list:
    """
    Predicts if a trade should be triggered and its expected return.

    Returns a list of dicts:
      [{
         "stock": <symbol>,
         "trade_triggered": <0|1>,
         "predicted_return": <float>,
         "recommended_config": <dict of feature:value>
      }]
    """
    try:
        clf_model, clf_features = load_model(settings.dual_classifier_model_name)
        reg_model, reg_features = load_model(settings.dual_regressor_model_name)
    except Exception:
        return []

    X_clf = feature_df.reindex(columns=clf_features, fill_value=0)
    X_reg = feature_df.reindex(columns=reg_features, fill_value=0)

    trade_triggered = int(clf_model.predict(X_clf)[0])
    predicted_return = float(reg_model.predict(X_reg)[0])
    sim_date = get_simulation_date()

    row = {
        "date": sim_date,
        "stock": stock,
        "predicted_return": predicted_return,
        "trade_triggered": trade_triggered,
        **X_clf.to_dict(orient="records")[0]
    }

    insert_with_conflict_handling(pd.DataFrame([row]), settings.param_model_predictions_table)
    insert_with_conflict_handling(pd.DataFrame([row]), settings.filter_model_predictions_table)

    return [{
        "stock": stock,
        "trade_triggered": trade_triggered,
        "predicted_return": predicted_return,
        "recommended_config": X_clf.to_dict(orient="records")[0]
    }]

__all__ = ["predict_dual_model"]
