# models/ml_dual_model_prediction_sql.py
from core.data_provider import load_data
from core.logger import logger
from core.time_context import get_simulation_date
from core.model_io import load_model
from core.config import settings
import pandas as pd


def predict_dual_model(
    data_path: str = settings.fundamentals_table,
    feature_path: str = settings.feature_table,
    strategy_type: str = settings.strategy_type,
    top_n: int = settings.meta_top_n,
) -> list:
    """
    Dual‐model prediction pipeline:
    - Loads fundamentals from `data_path`
    - Loads feature table from `feature_path`
    - Runs filter and exit models, returns top_n signals
    """
    # load data
    df_fund = load_data(data_path)
    df_feat = load_data(feature_path)
    today = get_simulation_date()

    # load models
    filter_model, _ = load_model(settings.filter_model_name)
    exit_model, _ = load_model(settings.exit_model_name)

    results = []
    # iterate stocks
    for _, row in df_fund.iterrows():
        stock = row["stock"]
        feats = df_feat[df_feat["stock"] == stock]
        if feats.empty:
            continue

        # filter model probability
        prob = float(filter_model.predict_proba(feats)[:,1].mean())
        if prob < settings.filter_threshold:
            continue

        # exit model return
        ret = float(exit_model.predict(feats).mean())
        if ret <= 0:
            continue

        results.append({
            "stock": stock,
            "probability": prob,
            "predicted_return": ret,
            "trade_triggered": 1,
        })

    # sort by probability and take top_n
    results = sorted(results, key=lambda x: x["probability"], reverse=True)
    return results[:top_n]


if __name__ == '__main__':
    recs = predict_dual_model()
    for r in recs:
        logger.info(f"{r['stock']} | Prob: {r['probability']:.2f} | Return: {r['predicted_return']:.2f}")
