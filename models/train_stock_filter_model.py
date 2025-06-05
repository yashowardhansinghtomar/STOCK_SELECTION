# models/train_stock_filter_model.py

import pandas as pd
import lightgbm as lgb
import joblib
import random
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from core.config.config import settings
from core.data_provider.data_provider import load_data, save_data
from core.logger.logger import logger
from pathlib import Path

FEATURE_TABLE = "stock_features_day"
RECS_TABLE = "recommendations"
PRED_TABLE = "filter_model_predictions"

FEATURE_COLS = [
    "sma_short", "sma_long", "rsi_thresh", "macd", "vwap", "atr_14",
    "bb_width", "macd_histogram", "price_compression", "volatility_10",
    "volume_spike", "vwap_dev", "stock_encoded"
]
LABEL_COL = "label"


def save_model_and_predictions(model, X_val, y_proba, merged):
    preds = X_val.copy()
    preds["score"] = y_proba
    preds["rank"] = preds["score"].rank(pct=True)
    preds["confidence"] = preds["score"]
    preds["decision"] = preds["score"].gt(0.5).map({True: "buy", False: "reject"})
    preds["stock"] = merged.iloc[preds.index]["stock"].values
    preds["date"] = merged.iloc[preds.index]["date"].values

    save_data(preds[["date", "stock", "score", "rank", "confidence", "decision"]],
              PRED_TABLE, if_exists="replace")

    model_path = Path(settings.model_dir) / f"{settings.model_names['filter']}.lgb"
    joblib.dump(model, model_path)
    logger.success("✅ Filter model trained and saved. Predictions written to DB.")


def train_filter_model(force_dummy=False):
    recs = load_data(RECS_TABLE)
    feats = load_data(FEATURE_TABLE)

    if recs.empty or force_dummy:
        logger.warning("💡 No recommendations found — generating dummy data for bootstrap.")

        if feats.empty:
            logger.warning("❌ Feature data is missing. Cannot proceed.")
            return

        valid_dates = feats["date"].drop_duplicates().sort_values().head(5).tolist()
        valid_stocks = feats["stock"].drop_duplicates().tolist()
        dummy_stocks = [s for s in ["RELIANCE", "SBIN", "INFY", "LT", "ICICIBANK"] if s in valid_stocks]

        if not dummy_stocks or not valid_dates:
            logger.warning("⚠️ No valid dummy stocks or dates available.")
            return

        dummy_recs = pd.DataFrame([
            {"stock": stock, "date": date, "label": random.choice([0, 1])}
            for stock in dummy_stocks for date in valid_dates
        ])

        dummy_feats = pd.DataFrame([
            {
                "stock": stock,
                "date": date,
                "sma_short": random.randint(5, 15),
                "sma_long": random.randint(20, 50),
                "rsi_thresh": random.randint(30, 70),
                "macd": random.uniform(-1, 1),
                "vwap": random.uniform(100, 500),
                "atr_14": random.uniform(1, 5),
                "bb_width": random.uniform(0.5, 2),
                "macd_histogram": random.uniform(-0.5, 0.5),
                "price_compression": random.uniform(0, 1),
                "volatility_10": random.uniform(0.1, 2),
                "volume_spike": random.uniform(0.5, 3),
                "vwap_dev": random.uniform(-2, 2),
                "stock_encoded": hash(stock) % 1000,
            }
            for stock in dummy_stocks for date in valid_dates
        ])

        save_data(dummy_recs, RECS_TABLE)
        save_data(dummy_feats, FEATURE_TABLE)
        recs = dummy_recs
        feats = dummy_feats
        logger.info(f"✅ Inserted {len(dummy_recs)} dummy recs and {len(dummy_feats)} dummy features.")

    merged = recs.merge(feats, on=["stock", "date"])
    logger.info(f"🔍 Loaded {len(merged)} merged rows for training")

    missing_cols = [col for col in FEATURE_COLS + [LABEL_COL] if col not in merged.columns]
    if missing_cols:
        logger.error(f"❌ Missing required columns in merged data: {missing_cols}")
        return

    merged = merged.dropna(subset=FEATURE_COLS + [LABEL_COL])
    logger.info(f"📈 Training on {len(merged)} merged records after dropping NA.")

    if len(merged) < 100:
        logger.warning("⚠️ Not enough labeled data to train a robust model. Skipping model training.")
        return

    X = merged[FEATURE_COLS]
    y = merged[LABEL_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    logger.info("📊 Model Performance on Validation Set:")
    logger.info("\n" + classification_report(y_val, y_pred))
    logger.info(f"🔍 ROC AUC: {roc_auc_score(y_val, y_proba):.3f}")

    save_model_and_predictions(model, X_val, y_proba, merged)


if __name__ == "__main__":
    train_filter_model()
