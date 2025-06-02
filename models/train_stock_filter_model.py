# models/train_stock_filter_model.py

import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from core.data_provider.data_provider import load_data, save_data
from db.postgres_manager import run_query
from core.logger.logger import logger

FEATURE_TABLE = "stock_features_day"
RECS_TABLE = "recommendations"
PRED_TABLE = "filter_model_predictions"
MODEL_PATH = "models/filter_model.lgb"

# ────────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "sma_short", "sma_long", "rsi_thresh", "macd", "vwap", "atr_14",
    "bb_width", "macd_histogram", "price_compression", "volatility_10",
    "volume_spike", "vwap_dev", "stock_encoded"
]

LABEL_COL = "label"

# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────
def train_filter_model():
    recs = load_data(RECS_TABLE)
    feats = load_data(FEATURE_TABLE)

    if recs.empty:
        logger.warning("No past recommendations yet — skipping filter training for now.")
        return
    if feats.empty:
        logger.warning("Feature data is missing. Aborting training.")
        return

    # Join and filter
    merged = recs.merge(feats, left_on=["stock", "date"], right_on=["stock", "date"])
    merged = merged.dropna(subset=FEATURE_COLS + [LABEL_COL])
    logger.info(f"Training on {len(merged)} rows.")

    if len(merged) < 100:
        logger.warning("Not enough labeled data to train a robust model. Aborting.")
        return

    X = merged[FEATURE_COLS]
    y = merged[LABEL_COL]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    logger.info("Model Performance on Validation Set:")
    logger.info("\n" + classification_report(y_val, y_pred))
    logger.info(f"ROC AUC: {roc_auc_score(y_val, y_proba):.3f}")

    # Save predictions for evaluation/debugging (optional)
    X_val = X_val.copy()
    X_val["score"] = y_proba
    X_val["rank"] = X_val["score"].rank(pct=True)
    X_val["confidence"] = X_val["score"]
    X_val["decision"] = (X_val["score"] > 0.5).map({True: "buy", False: "reject"})
    X_val["stock"] = merged.iloc[X_val.index]["stock"].values
    X_val["date"] = merged.iloc[X_val.index]["date"].values

    save_data(X_val[["date", "stock", "score", "rank", "confidence", "decision"]], PRED_TABLE, if_exists="replace")

    # Save the trained model to disk
    joblib.dump(model, MODEL_PATH)
    logger.success("Filter model trained and saved. Predictions written to DB.")

if __name__ == "__main__":
    train_filter_model()
